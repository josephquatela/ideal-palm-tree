"""
features/engineer.py
Builds the training dataset from raw commits, branches, and outcomes.
Each row in the output is one Rehearsal scenario (branch) with:
  - rolling window features derived from commits in the lookback window
  - seasonality encodings derived from the branch timestamp
  - scenario delta features (how unusual is this branch vs. history)
  - target labels from outcomes (inventory_delta, cashflow_delta, order_velocity_delta)
Output: data/processed/training_data.csv
"""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COMMITS_PATH  = RAW_DIR / "commits.csv"
BRANCHES_PATH = RAW_DIR / "branches.csv"
OUTCOMES_PATH = RAW_DIR / "outcomes.csv"
OUTPUT_PATH   = PROCESSED_DIR / "training_data.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────

LOOKBACK_WINDOWS = [7, 30, 90]   # days of history to aggregate per window

# Seasonal peak months per category (matches ingest.py)
SEASONAL_PEAKS = {
    "retail":  [11, 12],
    "apparel": [3, 4, 9, 10],
    "b2b":     [3, 6, 9, 12],
}

ORDER_EVENTS      = {"ORDER_RECEIVED", "ORDER_FULFILLED"}
CANCELLATION_EVENTS = {"ORDER_CANCELLED"}
BLOCKED_EVENTS    = {"FULFILLMENT_BLOCKED"}
RESTOCK_EVENTS    = {"INVENTORY_RESTOCK", "SUPPLIER_PO_CREATED"}
PRICE_EVENTS      = {"PRICE_CHANGE"}
LEDGER_EVENTS     = {"LEDGER_ENTRY"}

# ── LOAD RAW DATA ─────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def parse_dt(s):
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(s)

# ── SEASONALITY FEATURES ──────────────────────────────────────────────────────

def sine_cosine_encode(value, period):
    """
    Encode a cyclic value (e.g. month 1-12) as sine + cosine pair.
    This keeps December and January numerically close instead of far apart.
    """
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)

def days_to_nearest_peak(date, category):
    """
    How many days until the next seasonal peak month for this category?
    Returns a value in [0, 182] — lower = closer to peak.
    """
    peaks = SEASONAL_PEAKS.get(category, [])
    if not peaks:
        return 180

    min_dist = 365
    for peak_month in peaks:
        # Distance in months (cyclic), convert to days
        diff = abs(date.month - peak_month)
        diff = min(diff, 12 - diff)  # cyclic wrap
        dist_days = diff * 30
        min_dist = min(min_dist, dist_days)
    return min_dist

def seasonality_features(dt, category):
    """Return dict of all seasonality-related features for a given datetime."""
    month_sin, month_cos         = sine_cosine_encode(dt.month, 12)
    dow_sin,   dow_cos           = sine_cosine_encode(dt.weekday(), 7)
    wom_sin,   wom_cos           = sine_cosine_encode((dt.day - 1) // 7, 4)
    peak_dist                    = days_to_nearest_peak(dt, category)
    peak_sin,  peak_cos          = sine_cosine_encode(peak_dist, 182)

    return {
        "month_sin":           round(month_sin, 6),
        "month_cos":           round(month_cos, 6),
        "day_of_week_sin":     round(dow_sin, 6),
        "day_of_week_cos":     round(dow_cos, 6),
        "week_of_month_sin":   round(wom_sin, 6),
        "week_of_month_cos":   round(wom_cos, 6),
        "days_to_peak":        peak_dist,
        "days_to_peak_sin":    round(peak_sin, 6),
        "days_to_peak_cos":    round(peak_cos, 6),
        "is_peak_month":       int(dt.month in SEASONAL_PEAKS.get(category, [])),
        "is_weekend":          int(dt.weekday() >= 5),
        "quarter":             (dt.month - 1) // 3 + 1,
    }

# ── ROLLING WINDOW FEATURES ───────────────────────────────────────────────────

def confidence_decay_weight(commit_dt, branch_dt, half_life_days=30):
    """
    Exponential decay weight — commits closer to branch_dt count more.
    At t=0 (branch date): weight=1.0
    At t=half_life_days:  weight=0.5
    """
    age_days = (branch_dt - commit_dt).total_seconds() / 86400
    return math.exp(-math.log(2) * age_days / half_life_days)

def rolling_features(commits_in_window, window_days, branch_dt, total_history_days):
    """
    Aggregate a list of commits into feature scalars for one lookback window.
    All monetary values are normalised to % change to equalise across revenue scales.
    """
    if not commits_in_window:
        return {
            f"w{window_days}_order_count":          0,
            f"w{window_days}_order_velocity":        0.0,
            f"w{window_days}_revenue_total":         0.0,
            f"w{window_days}_revenue_per_day":       0.0,
            f"w{window_days}_cancel_rate":           0.0,
            f"w{window_days}_blocked_rate":          0.0,
            f"w{window_days}_restock_count":         0,
            f"w{window_days}_price_change_count":    0,
            f"w{window_days}_avg_price_change_pct":  0.0,
            f"w{window_days}_latest_cash_balance":   0.0,
            f"w{window_days}_cash_trend":            0.0,
            f"w{window_days}_inventory_trend":       0.0,
            f"w{window_days}_weighted_order_count":  0.0,
        }

    orders, cancels, blocked, restocks = 0, 0, 0, 0
    revenue_total = 0.0
    price_changes = []
    cash_balances = []    # list of (datetime, balance)
    inventory_levels = [] # list of (datetime, units)
    weighted_order_sum = 0.0

    for c in commits_in_window:
        et = c["event_type"]
        try:
            payload = json.loads(c["payload"]) if c["payload"] else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}

        dt = parse_dt(c["timestamp"])
        w  = confidence_decay_weight(dt, branch_dt)

        if et == "ORDER_RECEIVED":
            orders += 1
            revenue_total += float(payload.get("revenue", 0))
            weighted_order_sum += w

        elif et == "ORDER_CANCELLED":
            cancels += 1

        elif et == "FULFILLMENT_BLOCKED":
            blocked += 1

        elif et in RESTOCK_EVENTS:
            restocks += 1

        elif et == "PRICE_CHANGE":
            pct = float(payload.get("pct_change", 0))
            price_changes.append(pct)

        elif et == "LEDGER_ENTRY":
            bal = float(payload.get("cash_balance", 0))
            inv = float(payload.get("inventory_units", 0))
            cash_balances.append((dt, bal))
            inventory_levels.append((dt, inv))

    # Derived scalars
    order_velocity    = orders / window_days
    cancel_rate       = cancels / max(orders, 1)
    blocked_rate      = blocked / max(orders + blocked, 1)
    revenue_per_day   = revenue_total / window_days
    avg_price_change  = sum(price_changes) / len(price_changes) if price_changes else 0.0

    # Cash trend: slope of cash balance over window (positive = growing)
    cash_trend = 0.0
    if len(cash_balances) >= 2:
        cash_balances.sort(key=lambda x: x[0])
        days_span = max((cash_balances[-1][0] - cash_balances[0][0]).days, 1)
        cash_trend = (cash_balances[-1][1] - cash_balances[0][1]) / days_span

    latest_cash = cash_balances[-1][1] if cash_balances else 0.0

    # Inventory trend
    inv_trend = 0.0
    if len(inventory_levels) >= 2:
        inventory_levels.sort(key=lambda x: x[0])
        days_span = max((inventory_levels[-1][0] - inventory_levels[0][0]).days, 1)
        inv_trend = (inventory_levels[-1][1] - inventory_levels[0][1]) / days_span

    return {
        f"w{window_days}_order_count":          orders,
        f"w{window_days}_order_velocity":        round(order_velocity, 4),
        f"w{window_days}_revenue_total":         round(revenue_total, 2),
        f"w{window_days}_revenue_per_day":       round(revenue_per_day, 4),
        f"w{window_days}_cancel_rate":           round(cancel_rate, 4),
        f"w{window_days}_blocked_rate":          round(blocked_rate, 4),
        f"w{window_days}_restock_count":         restocks,
        f"w{window_days}_price_change_count":    len(price_changes),
        f"w{window_days}_avg_price_change_pct":  round(avg_price_change, 4),
        f"w{window_days}_latest_cash_balance":   round(latest_cash, 2),
        f"w{window_days}_cash_trend":            round(cash_trend, 4),
        f"w{window_days}_inventory_trend":       round(inv_trend, 4),
        f"w{window_days}_weighted_order_count":  round(weighted_order_sum, 4),
    }

# ── SCENARIO DELTA FEATURES ───────────────────────────────────────────────────

def scenario_delta_features(branch, business_commits, branch_dt):
    """
    How far does this scenario deviate from the business's historical baseline?
    Compares branch_params to recent historical averages.
    """
    try:
        params = json.loads(branch["branch_params"]) if branch["branch_params"] else {}
    except (json.JSONDecodeError, TypeError):
        params = {}

    branch_type = branch["branch_type"]

    # Compute historical average price change from PRICE_CHANGE commits
    past_price_changes = []
    for c in business_commits:
        if c["event_type"] == "PRICE_CHANGE" and parse_dt(c["timestamp"]) < branch_dt:
            try:
                p = json.loads(c["payload"])
                past_price_changes.append(float(p.get("pct_change", 0)))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    hist_avg_price_change = (
        sum(past_price_changes) / len(past_price_changes)
        if past_price_changes else 0.0
    )

    # Scenario-specific deviation
    scenario_magnitude  = 0.0
    deviation_from_norm = 0.0

    if branch_type in ("price_increase", "price_decrease"):
        pct = float(params.get("pct_change", 0))
        scenario_magnitude  = abs(pct)
        deviation_from_norm = pct - hist_avg_price_change

    elif branch_type == "bulk_restock":
        qty = float(params.get("quantity", 0))
        scenario_magnitude = qty
        # Compare to average restock quantity in history
        past_restocks = []
        for c in business_commits:
            if c["event_type"] == "INVENTORY_RESTOCK" and parse_dt(c["timestamp"]) < branch_dt:
                try:
                    p = json.loads(c["payload"])
                    past_restocks.append(float(p.get("quantity_added", 0)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        avg_restock = sum(past_restocks) / len(past_restocks) if past_restocks else qty
        deviation_from_norm = (qty - avg_restock) / max(avg_restock, 1)

    elif branch_type == "promo_event":
        scenario_magnitude  = float(params.get("discount_pct", 0))
        deviation_from_norm = scenario_magnitude  # no baseline for promos

    elif branch_type == "markdown":
        scenario_magnitude  = float(params.get("markdown_pct", 0))
        deviation_from_norm = scenario_magnitude

    elif branch_type == "volume_discount":
        scenario_magnitude  = float(params.get("discount_pct", 0))
        deviation_from_norm = scenario_magnitude

    elif branch_type == "contract_change":
        scenario_magnitude  = float(params.get("contract_value", 0)) / 10_000  # normalise
        deviation_from_norm = scenario_magnitude

    elif branch_type == "new_collection":
        scenario_magnitude  = float(params.get("new_skus", 0))
        deviation_from_norm = scenario_magnitude

    elif branch_type == "supplier_change":
        scenario_magnitude  = abs(float(params.get("cost_change_pct", 0)))
        deviation_from_norm = float(params.get("cost_change_pct", 0))

    elif branch_type == "net_terms_change":
        new_terms = params.get("new_terms", "NET30")
        terms_map = {"NET15": -15, "NET30": 0, "NET45": 15, "NET60": 30}
        delta = terms_map.get(new_terms, 0) - terms_map.get("NET30", 0)
        scenario_magnitude  = abs(delta)
        deviation_from_norm = delta

    history_depth_days = max(
        [(branch_dt - parse_dt(c["timestamp"])).days for c in business_commits]
        if business_commits else [0]
    )

    return {
        "scenario_magnitude":      round(scenario_magnitude, 4),
        "deviation_from_norm":     round(deviation_from_norm, 4),
        "history_depth_days":      history_depth_days,
        "n_past_price_changes":    len(past_price_changes),
        "hist_avg_price_change":   round(hist_avg_price_change, 4),
    }

# ── CATEGORY ONE-HOT ENCODING ─────────────────────────────────────────────────

def category_features(category):
    return {
        "cat_retail":  int(category == "retail"),
        "cat_apparel": int(category == "apparel"),
        "cat_b2b":     int(category == "b2b"),
    }

# ── BRANCH TYPE ONE-HOT ENCODING ──────────────────────────────────────────────

ALL_BRANCH_TYPES = [
    "price_increase", "price_decrease", "bulk_restock", "promo_event",
    "new_collection", "markdown", "contract_change", "net_terms_change",
    "volume_discount", "supplier_change",
]

def branch_type_features(branch_type):
    return {f"bt_{bt}": int(branch_type == bt) for bt in ALL_BRANCH_TYPES}

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def build_training_data():
    print("Loading raw data...")
    commits  = load_csv(COMMITS_PATH)
    branches = load_csv(BRANCHES_PATH)
    outcomes = load_csv(OUTCOMES_PATH)

    print(f"  {len(commits):>7,} commits")
    print(f"  {len(branches):>7,} branches")
    print(f"  {len(outcomes):>7,} outcomes")

    # Index commits by business_id
    print("Indexing commits by business...")
    commits_by_business = defaultdict(list)
    for c in commits:
        commits_by_business[c["business_id"]].append(c)

    # Index outcomes by branch_id
    outcomes_by_branch = {o["branch_id"]: o for o in outcomes}

    # Only keep committed branches that have a matching outcome
    branches = [
        b for b in branches
        if b["branch_id"] in outcomes_by_branch and b["status"] == "committed"
    ]
    print(f"  {len(branches):>7,} committed branches with outcomes")

    rows = []
    skipped = 0

    for i, branch in enumerate(branches):
        if i % 200 == 0:
            print(f"  Processing branch {i}/{len(branches)}...")

        branch_id   = branch["branch_id"]
        business_id = branch["business_id"]
        category    = branch["category"]
        branch_type = branch["branch_type"]
        branch_dt   = parse_dt(branch["committed_at"])

        outcome = outcomes_by_branch[branch_id]
        business_commits = commits_by_business.get(business_id, [])

        # Filter to commits strictly before the branch date
        prior_commits = [
            c for c in business_commits
            if parse_dt(c["timestamp"]) < branch_dt
        ]

        if len(prior_commits) < 10:
            skipped += 1
            continue  # too sparse for meaningful features

        total_history = (branch_dt - parse_dt(prior_commits[0]["timestamp"])).days

        # ── Rolling window features for each lookback window
        window_feats = {}
        for window in LOOKBACK_WINDOWS:
            cutoff = branch_dt - timedelta(days=window)
            window_commits = [
                c for c in prior_commits
                if parse_dt(c["timestamp"]) >= cutoff
            ]
            window_feats.update(
                rolling_features(window_commits, window, branch_dt, total_history)
            )

        # ── Seasonality features
        season_feats = seasonality_features(branch_dt, category)

        # ── Scenario delta features
        delta_feats = scenario_delta_features(branch, prior_commits, branch_dt)

        # ── Category and branch type encodings
        cat_feats = category_features(category)
        bt_feats  = branch_type_features(branch_type)

        # ── Labels
        labels = {
            "label_inventory_delta":      float(outcome["inventory_delta"]),
            "label_cashflow_delta":       float(outcome["cashflow_delta"]),
            "label_order_velocity_delta": float(outcome["order_velocity_delta"]),
        }

        # ── Assemble row
        row = {
            "branch_id":   branch_id,
            "business_id": business_id,
            "category":    category,
            "branch_type": branch_type,
            "committed_at": branch["committed_at"],
        }
        row.update(window_feats)
        row.update(season_feats)
        row.update(delta_feats)
        row.update(cat_feats)
        row.update(bt_feats)
        row.update(labels)

        rows.append(row)

    print(f"\nFeature engineering complete.")
    print(f"  {len(rows):>7,} training rows built")
    print(f"  {skipped:>7,} branches skipped (insufficient history)")

    if not rows:
        print("ERROR: No training rows produced. Check data paths.")
        return

    # ── Write output
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"  {len(fieldnames)} features per row")
    print(f"\nFeature groups:")
    print(f"  Rolling windows (7/30/90d) : {len(LOOKBACK_WINDOWS) * 13} features")
    print(f"  Seasonality               : {len(seasonality_features(datetime.now(), 'retail'))} features")
    dummy_branch = {"branch_params": "{}", "branch_type": "price_increase"}
    print(f"  Scenario delta            : {len(scenario_delta_features(dummy_branch, [], datetime.now()))} features")
    print(f"  Category one-hot          : 3 features")
    print(f"  Branch type one-hot       : {len(ALL_BRANCH_TYPES)} features")
    print(f"  Labels (targets)          : 3 features")

    # ── Category breakdown
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    print(f"\nRows per category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<10} {count:>5}")

    return rows

if __name__ == "__main__":
    build_training_data()