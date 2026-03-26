"""
data/ingest.py
Generates synthetic Trunk commit log data across three business categories.
Outputs three CSVs: commits.csv, branches.csv, outcomes.csv — linked by branch_id.
"""

import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BUSINESSES    = 150      # 50 per category
DAYS_OF_HISTORY = 365      # 12 months of commits per business
N_BRANCHES      = 3000     # rehearsal scenarios to generate (1000 per category)
OUTCOME_HORIZON = 14       # days after commit to record outcome

START_DATE = datetime(2025, 9, 1)

CATEGORIES = ["retail", "apparel", "b2b"]

# ── BUSINESS PROFILES ────────────────────────────────────────────────────────
# Each category has distinct revenue scale, order frequency, and seasonality.

PROFILES = {
    "retail": {
        "revenue_range":     (200,   8_000),   # daily revenue baseline
        "order_freq":        (3, 15),           # orders per day
        "restock_freq":      14,                # days between restocks
        "price_change_freq": 45,                # days between price changes
        "seasonal_peaks":    [11, 12],          # holiday months
        "seasonal_boost":    1.6,
        "return_rate":       0.05,
    },
    "apparel": {
        "revenue_range":     (300,   12_000),
        "order_freq":        (2, 10),
        "restock_freq":      21,
        "price_change_freq": 30,
        "seasonal_peaks":    [3, 4, 9, 10],    # spring/fall drops
        "seasonal_boost":    1.8,
        "return_rate":       0.18,              # higher return rate
    },
    "b2b": {
        "revenue_range":     (1_000, 40_000),   # larger but less frequent
        "order_freq":        (0, 3),
        "restock_freq":      30,
        "price_change_freq": 90,
        "seasonal_peaks":    [3, 6, 9, 12],    # fiscal quarters
        "seasonal_boost":    1.3,
        "return_rate":       0.02,
    },
}

BRANCH_TYPES = {
    "retail":  ["price_increase", "price_decrease", "bulk_restock", "promo_event", "supplier_change"],
    "apparel": ["price_increase", "price_decrease", "bulk_restock", "promo_event", "new_collection", "markdown"],
    "b2b":     ["price_increase", "price_decrease", "bulk_restock", "contract_change", "net_terms_change", "volume_discount"],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def uid():
    return str(uuid.uuid4())

def jitter(value, pct=0.15):
    """Add ±pct% noise to a value."""
    return value * (1 + random.uniform(-pct, pct))

def seasonal_multiplier(date, profile):
    if date.month in profile["seasonal_peaks"]:
        return profile["seasonal_boost"]
    return 1.0

def day_of_week_multiplier(date):
    """Weekends are busier for retail/apparel, quieter for B2B."""
    return 1.2 if date.weekday() >= 5 else 1.0

def weeks_since_start(date):
    return (date - START_DATE).days // 7

# ── GENERATE BUSINESSES ───────────────────────────────────────────────────────

def generate_businesses():
    businesses = []
    for category in CATEGORIES:
        for _ in range(N_BUSINESSES // len(CATEGORIES)):
            profile = PROFILES[category]
            businesses.append({
                "business_id":       uid(),
                "category":          category,
                "revenue_baseline":  random.uniform(*profile["revenue_range"]),
                "inventory_units":   random.randint(50, 500),
                "price_per_unit":    round(random.uniform(15, 200), 2),
            })
    return businesses

# ── GENERATE COMMITS ─────────────────────────────────────────────────────────

def generate_commits(businesses):
    commits = []

    for biz in businesses:
        category = biz["category"]
        profile  = PROFILES[category]
        inventory = biz["inventory_units"]
        price     = biz["price_per_unit"]
        revenue_base = biz["revenue_baseline"]

        current_date = START_DATE
        restock_counter      = 0
        price_change_counter = 0

        for day in range(DAYS_OF_HISTORY):
            current_date = START_DATE + timedelta(days=day)
            s_mult = seasonal_multiplier(current_date, profile)
            d_mult = day_of_week_multiplier(current_date) if category != "b2b" else 1.0

            # ── Orders
            n_orders = int(random.randint(*profile["order_freq"]) * s_mult * d_mult)
            for _ in range(n_orders):
                qty = random.randint(1, 10 if category != "b2b" else 50)
                revenue = round(qty * price * jitter(1.0, 0.05), 2)
                if inventory >= qty:
                    inventory -= qty
                    commits.append({
                        "commit_id":         uid(),
                        "timestamp":         (current_date + timedelta(
                                               hours=random.randint(8, 20),
                                               minutes=random.randint(0, 59))).isoformat(),
                        "business_id":       biz["business_id"],
                        "business_category": category,
                        "event_type":        "ORDER_RECEIVED",
                        "branch_id":         None,
                        "payload":           json.dumps({
                            "quantity": qty,
                            "revenue":  revenue,
                            "price_per_unit": price,
                        }),
                    })
                    # Fulfillment commit
                    commits.append({
                        "commit_id":         uid(),
                        "timestamp":         (current_date + timedelta(
                                               hours=random.randint(8, 20),
                                               minutes=random.randint(0, 59))).isoformat(),
                        "business_id":       biz["business_id"],
                        "business_category": category,
                        "event_type":        "ORDER_FULFILLED",
                        "branch_id":         None,
                        "payload":           json.dumps({
                            "quantity":         qty,
                            "inventory_after":  inventory,
                        }),
                    })
                    # Occasional cancellation
                    if random.random() < profile["return_rate"]:
                        inventory += qty
                        commits.append({
                            "commit_id":         uid(),
                            "timestamp":         (current_date + timedelta(
                                                   hours=random.randint(8, 20),
                                                   minutes=random.randint(0, 59))).isoformat(),
                            "business_id":       biz["business_id"],
                            "business_category": category,
                            "event_type":        "ORDER_CANCELLED",
                            "branch_id":         None,
                            "payload":           json.dumps({
                                "quantity":         qty,
                                "revenue_reversed": revenue,
                                "inventory_after":  inventory,
                            }),
                        })
                else:
                    # Fulfillment blocked — out of stock
                    commits.append({
                        "commit_id":         uid(),
                        "timestamp":         current_date.isoformat(),
                        "business_id":       biz["business_id"],
                        "business_category": category,
                        "event_type":        "FULFILLMENT_BLOCKED",
                        "branch_id":         None,
                        "payload":           json.dumps({
                            "quantity_requested": qty,
                            "inventory_available": inventory,
                        }),
                    })

            # ── Restock (supplier PO + receipt)
            restock_counter += 1
            if restock_counter >= profile["restock_freq"]:
                restock_counter = 0
                restock_qty = random.randint(50, 300)
                inventory += restock_qty
                cost = round(restock_qty * price * random.uniform(0.4, 0.65), 2)
                commits.append({
                    "commit_id":         uid(),
                    "timestamp":         current_date.isoformat(),
                    "business_id":       biz["business_id"],
                    "business_category": category,
                    "event_type":        "SUPPLIER_PO_CREATED",
                    "branch_id":         None,
                    "payload":           json.dumps({
                        "quantity": restock_qty,
                        "cost":     cost,
                    }),
                })
                commits.append({
                    "commit_id":         uid(),
                    "timestamp":         (current_date + timedelta(days=random.randint(2, 7))).isoformat(),
                    "business_id":       biz["business_id"],
                    "business_category": category,
                    "event_type":        "INVENTORY_RESTOCK",
                    "branch_id":         None,
                    "payload":           json.dumps({
                        "quantity_added":  restock_qty,
                        "inventory_after": inventory,
                        "cost":            cost,
                    }),
                })

            # ── Price change
            price_change_counter += 1
            if price_change_counter >= profile["price_change_freq"]:
                price_change_counter = 0
                old_price = price
                price = round(price * random.uniform(0.85, 1.20), 2)
                commits.append({
                    "commit_id":         uid(),
                    "timestamp":         current_date.isoformat(),
                    "business_id":       biz["business_id"],
                    "business_category": category,
                    "event_type":        "PRICE_CHANGE",
                    "branch_id":         None,
                    "payload":           json.dumps({
                        "old_price":    old_price,
                        "new_price":    price,
                        "pct_change":   round((price - old_price) / old_price * 100, 2),
                    }),
                })

            # ── Ledger entry (cash position snapshot, weekly)
            if day % 7 == 0:
                cash_balance = round(revenue_base * s_mult * jitter(1.0, 0.2) * day, 2)
                commits.append({
                    "commit_id":         uid(),
                    "timestamp":         current_date.isoformat(),
                    "business_id":       biz["business_id"],
                    "business_category": category,
                    "event_type":        "LEDGER_ENTRY",
                    "branch_id":         None,
                    "payload":           json.dumps({
                        "cash_balance":   cash_balance,
                        "inventory_units": inventory,
                    }),
                })

        # Store final state back for use in branch generation
        biz["_final_inventory"] = inventory
        biz["_final_price"]     = price

    return commits

# ── GENERATE BRANCHES & OUTCOMES ─────────────────────────────────────────────

def generate_branches_and_outcomes(businesses, commits):
    """
    For each synthetic Rehearsal scenario (branch), pick a business and a
    scenario type, stamp a branch_id on a cluster of commits, then simulate
    a 14-day outcome and record it in outcomes.
    """

    # Index commits by business_id for fast lookup
    commit_index = {}
    for c in commits:
        commit_index.setdefault(c["business_id"], []).append(c)

    branches = []
    outcomes = []

    per_category = N_BRANCHES // len(CATEGORIES)

    for category in CATEGORIES:
        cat_businesses = [b for b in businesses if b["category"] == category]
        scenario_types = BRANCH_TYPES[category]
        profile = PROFILES[category]

        for _ in range(per_category):
            biz = random.choice(cat_businesses)
            branch_type = random.choice(scenario_types)
            branch_id   = uid()

            # Pick a commit date at least 14 days before end of history
            max_offset = DAYS_OF_HISTORY - OUTCOME_HORIZON - 1
            day_offset  = random.randint(90, max_offset)
            committed_at = START_DATE + timedelta(days=day_offset)

            # ── Build branch_params based on scenario type
            if branch_type == "price_increase":
                pct = round(random.uniform(5, 25), 1)
                params = {"pct_change": pct, "direction": "up"}
            elif branch_type == "price_decrease":
                pct = round(random.uniform(5, 20), 1)
                params = {"pct_change": -pct, "direction": "down"}
            elif branch_type == "bulk_restock":
                qty = random.randint(100, 500)
                params = {"quantity": qty, "cost_per_unit": round(random.uniform(10, 80), 2)}
            elif branch_type == "promo_event":
                disc = round(random.uniform(10, 30), 1)
                params = {"discount_pct": disc, "duration_days": random.randint(3, 14)}
            elif branch_type == "new_collection":
                skus = random.randint(5, 30)
                params = {"new_skus": skus, "avg_price": round(random.uniform(40, 200), 2)}
            elif branch_type == "markdown":
                pct = round(random.uniform(20, 50), 1)
                params = {"markdown_pct": pct, "skus_affected": random.randint(5, 50)}
            elif branch_type == "contract_change":
                params = {"contract_value": round(random.uniform(5_000, 100_000), 2),
                          "duration_months": random.randint(6, 24)}
            elif branch_type == "net_terms_change":
                params = {"old_terms": "NET30", "new_terms": random.choice(["NET15", "NET45", "NET60"])}
            elif branch_type == "volume_discount":
                pct = round(random.uniform(5, 15), 1)
                params = {"discount_pct": pct, "min_order_qty": random.randint(10, 100)}
            elif branch_type == "supplier_change":
                params = {"new_lead_time_days": random.randint(3, 21),
                          "cost_change_pct": round(random.uniform(-15, 10), 1)}
            else:
                params = {}

            branches.append({
                "branch_id":    branch_id,
                "business_id":  biz["business_id"],
                "category":     category,
                "branch_type":  branch_type,
                "branch_params": json.dumps(params),
                "committed_at": committed_at.isoformat(),
                "status":       random.choices(["committed", "discarded"], weights=[0.7, 0.3])[0],
            })

            # ── Simulate outcome
            # Use branch type to create realistic outcome deltas
            inv_delta  = 0.0
            cash_delta = 0.0
            vel_delta  = 0.0   # order velocity % change

            if branch_type == "price_increase":
                p = params["pct_change"]
                vel_delta  = round(-p * random.uniform(0.4, 0.9), 2)   # demand drops
                cash_delta = round(random.uniform(200, 2000) * (p / 10), 2)
                inv_delta  = round(vel_delta * 0.3, 2)                 # inventory depresses less

            elif branch_type == "price_decrease":
                p = abs(params["pct_change"])
                vel_delta  = round(p * random.uniform(0.5, 1.2), 2)    # demand rises
                cash_delta = round(-random.uniform(100, 1000) * (p / 10) + vel_delta * 30, 2)
                inv_delta  = round(-vel_delta * 0.5, 2)                # inventory depletes

            elif branch_type == "bulk_restock":
                inv_delta  = round(params["quantity"] * random.uniform(0.8, 1.0), 2)
                cash_delta = round(-(params["quantity"] * params["cost_per_unit"]), 2)
                vel_delta  = round(random.uniform(2, 8), 2)            # fewer stockouts

            elif branch_type == "promo_event":
                d = params["discount_pct"]
                vel_delta  = round(d * random.uniform(1.0, 2.0), 2)
                cash_delta = round(vel_delta * random.uniform(20, 80), 2)
                inv_delta  = round(-vel_delta * 0.8, 2)

            elif branch_type == "new_collection":
                vel_delta  = round(random.uniform(5, 20), 2)
                cash_delta = round(-(params["new_skus"] * params["avg_price"] * 0.4)
                                   + vel_delta * 50, 2)
                inv_delta  = round(params["new_skus"] * 10 * random.uniform(0.5, 1.0), 2)

            elif branch_type == "markdown":
                m = params["markdown_pct"]
                vel_delta  = round(m * random.uniform(0.8, 1.5), 2)
                cash_delta = round(-random.uniform(500, 3000) + vel_delta * 20, 2)
                inv_delta  = round(-vel_delta * 1.2, 2)

            elif branch_type == "contract_change":
                cash_delta = round(params["contract_value"] * random.uniform(0.08, 0.15), 2)
                vel_delta  = round(random.uniform(-2, 5), 2)
                inv_delta  = 0.0

            elif branch_type == "net_terms_change":
                new = params["new_terms"]
                cash_delta = round(random.uniform(-2000, 2000) *
                                   (1 if "15" in new else -1 if "60" in new else 0.5), 2)
                vel_delta  = round(random.uniform(-3, 3), 2)
                inv_delta  = 0.0

            elif branch_type == "volume_discount":
                vel_delta  = round(params["discount_pct"] * random.uniform(0.5, 1.0), 2)
                cash_delta = round(vel_delta * random.uniform(30, 100), 2)
                inv_delta  = round(-vel_delta * 0.5, 2)

            elif branch_type == "supplier_change":
                c = params["cost_change_pct"]
                cash_delta = round(-c * random.uniform(100, 1000), 2)
                vel_delta  = round(random.uniform(-5, 5), 2)
                inv_delta  = round(random.uniform(-10, 20), 2)

            # Add noise to all deltas
            inv_delta  = round(jitter(inv_delta  if inv_delta  != 0 else 1, 0.2), 2)
            cash_delta = round(jitter(cash_delta if cash_delta != 0 else 1, 0.2), 2)
            vel_delta  = round(jitter(vel_delta  if vel_delta  != 0 else 1, 0.2), 2)

            outcome_date = committed_at + timedelta(days=OUTCOME_HORIZON)
            outcomes.append({
                "outcome_id":            uid(),
                "branch_id":             branch_id,
                "business_id":           biz["business_id"],
                "category":              category,
                "recorded_at":           outcome_date.isoformat(),
                "horizon_days":          OUTCOME_HORIZON,
                "inventory_delta":       inv_delta,
                "cashflow_delta":        cash_delta,
                "order_velocity_delta":  vel_delta,
            })

    return branches, outcomes

# ── WRITE CSVs ────────────────────────────────────────────────────────────────

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6,} rows → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating businesses...")
    businesses = generate_businesses()

    print("Generating commits...")
    commits = generate_commits(businesses)

    print("Generating branches and outcomes...")
    branches, outcomes = generate_branches_and_outcomes(businesses, commits)

    print(f"\nWriting CSVs to {OUTPUT_DIR}/")
    write_csv(OUTPUT_DIR / "commits.csv",  commits,  [
        "commit_id", "timestamp", "business_id", "business_category",
        "event_type", "branch_id", "payload"
    ])
    write_csv(OUTPUT_DIR / "branches.csv", branches, [
        "branch_id", "business_id", "category", "branch_type",
        "branch_params", "committed_at", "status"
    ])
    write_csv(OUTPUT_DIR / "outcomes.csv", outcomes, [
        "outcome_id", "branch_id", "business_id", "category",
        "recorded_at", "horizon_days",
        "inventory_delta", "cashflow_delta", "order_velocity_delta"
    ])

    print(f"\nDone.")
    print(f"  {len(businesses)} businesses  ({N_BUSINESSES // len(CATEGORIES)} per category)")
    print(f"  {len(commits):,} commits")
    print(f"  {len(branches)} branches  ({N_BRANCHES // len(CATEGORIES)} per category)")
    print(f"  {len(outcomes)} outcomes")
    print(f"\nJoin key: branch_id links branches → commits → outcomes")