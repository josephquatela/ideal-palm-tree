"""
data/adapters/shopify_adapter.py
---------------------------------
Adapter for Shopify Benchmark Dataset (via ShopifyQL API or CSV export).

Takes raw Shopify merchant transaction data and outputs rows in the same
features.csv format produced by features/engineer.py — so the baseline
model and all downstream models can train on real cold-start data without
any schema changes.

Expected input schema (Shopify export or ShopifyQL result):
    - shop_id          : unique merchant identifier
    - created_at       : order timestamp (ISO 8601)
    - total_price      : order revenue (float)
    - cancelled_at     : null if fulfilled, timestamp if cancelled
    - product_type     : e.g. "Apparel", "Electronics", "Services"
    - inventory_qty    : units on hand at time of order (int)
    - cash_balance     : merchant account balance (float, if available)

Usage:
    python data/adapters/shopify_adapter.py \\
        --input  data/raw/shopify_orders.csv \\
        --output data/shopify_features.csv

    # Or pipe directly into baseline:
    python data/adapters/shopify_adapter.py --input data/raw/shopify_orders.csv \\
        | python models/baseline.py --features /dev/stdin
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Category mapping ──────────────────────────────────────────────────────────
# Maps Shopify product_type strings → our three internal categories.
# Extend this dict as you encounter new product types in the data.
CATEGORY_MAP = {
    "apparel":     "apparel",
    "clothing":    "apparel",
    "fashion":     "apparel",
    "accessories": "apparel",
    "electronics": "retail",
    "home":        "retail",
    "food":        "retail",
    "beauty":      "retail",
    "sports":      "retail",
    "services":    "b2b",
    "b2b":         "b2b",
    "wholesale":   "b2b",
    "consulting":  "b2b",
}

def map_category(product_type: str) -> str:
    key = str(product_type).lower().strip()
    for pattern, category in CATEGORY_MAP.items():
        if pattern in key:
            return category
    return "retail"  # safe default


# ── Rolling window features ───────────────────────────────────────────────────

def rolling_features(group: pd.DataFrame, window_days: int, as_of: pd.Timestamp) -> dict:
    """
    Compute rolling window features for one merchant over the last `window_days`
    days ending at `as_of`.

    This mirrors exactly what engineer.py does for synthetic data, so the
    feature names and semantics are identical.
    """
    prefix = f"w{window_days}"
    cutoff = as_of - pd.Timedelta(days=window_days)
    window = group[(group["created_at"] >= cutoff) & (group["created_at"] < as_of)]

    if len(window) == 0:
        # Return zeros — engineer.py uses the same convention for sparse history
        return {
            f"{prefix}_order_count":        0,
            f"{prefix}_order_velocity":     0.0,
            f"{prefix}_revenue_total":      0.0,
            f"{prefix}_revenue_per_day":    0.0,
            f"{prefix}_cancel_rate":        0.0,
            f"{prefix}_blocked_rate":       0.0,
            f"{prefix}_restock_count":      0,
            f"{prefix}_price_change_count": 0,
            f"{prefix}_avg_price_change_pct": 0.0,
            f"{prefix}_latest_cash_balance": 0.0,
            f"{prefix}_cash_trend":         0.0,
            f"{prefix}_inventory_trend":    0.0,
            f"{prefix}_weighted_order_count": 0.0,
        }

    order_count    = len(window)
    revenue_total  = window["total_price"].sum()
    cancel_rate    = window["cancelled_at"].notna().mean()
    # Shopify has no "blocked" concept — proxy with high-value cancellations
    blocked_rate   = ((window["cancelled_at"].notna()) & (window["total_price"] > window["total_price"].quantile(0.75))).mean()

    cash_balance   = window["cash_balance"].iloc[-1] if "cash_balance" in window.columns else 0.0
    cash_trend     = window["cash_balance"].diff().mean() if "cash_balance" in window.columns else 0.0
    inv_trend      = window["inventory_qty"].diff().mean() if "inventory_qty" in window.columns else 0.0

    # Recency-weighted order count (same formula as engineer.py)
    days_ago = (as_of - window["created_at"]).dt.days.clip(lower=1)
    weights  = 1.0 / days_ago
    weighted_order_count = float((weights).sum())

    return {
        f"{prefix}_order_count":          order_count,
        f"{prefix}_order_velocity":       order_count / window_days,
        f"{prefix}_revenue_total":        float(revenue_total),
        f"{prefix}_revenue_per_day":      float(revenue_total / window_days),
        f"{prefix}_cancel_rate":          float(cancel_rate),
        f"{prefix}_blocked_rate":         float(blocked_rate),
        f"{prefix}_restock_count":        0,        # not in Shopify export
        f"{prefix}_price_change_count":   0,        # not in Shopify export
        f"{prefix}_avg_price_change_pct": 0.0,
        f"{prefix}_latest_cash_balance":  float(cash_balance),
        f"{prefix}_cash_trend":           float(cash_trend) if not np.isnan(cash_trend) else 0.0,
        f"{prefix}_inventory_trend":      float(inv_trend) if not np.isnan(inv_trend) else 0.0,
        f"{prefix}_weighted_order_count": weighted_order_count,
    }


# ── Seasonality encoding ──────────────────────────────────────────────────────

PEAK_MONTH = 12   # December — most relevant for retail/apparel

def seasonality_features(ts: pd.Timestamp) -> dict:
    month        = ts.month
    dow          = ts.dayofweek
    week_of_month = (ts.day - 1) // 7 + 1
    days_to_peak  = abs((ts - pd.Timestamp(year=ts.year, month=PEAK_MONTH, day=15)).days)

    return {
        "month_sin":       np.sin(2 * np.pi * month / 12),
        "month_cos":       np.cos(2 * np.pi * month / 12),
        "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
        "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
        "week_of_month_sin": np.sin(2 * np.pi * week_of_month / 5),
        "week_of_month_cos": np.cos(2 * np.pi * week_of_month / 5),
        "days_to_peak":    days_to_peak,
        "days_to_peak_sin": np.sin(2 * np.pi * days_to_peak / 365),
        "days_to_peak_cos": np.cos(2 * np.pi * days_to_peak / 365),
        "is_peak_month":   int(month == PEAK_MONTH),
        "is_weekend":      int(dow >= 5),
        "quarter":         (month - 1) // 3 + 1,
    }


# ── Outcome labels ────────────────────────────────────────────────────────────

def compute_labels(group: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int = 14) -> dict:
    """
    Compute 14-day post-event outcome labels.
    For Shopify data these are approximations — real Trunk data has explicit
    outcome logging. These are good enough for cold-start bootstrapping.
    """
    future_end   = as_of + pd.Timedelta(days=horizon_days)
    pre_window   = group[group["created_at"] < as_of]
    post_window  = group[(group["created_at"] >= as_of) & (group["created_at"] < future_end)]

    if len(pre_window) == 0 or len(post_window) == 0:
        return {
            "label_inventory_delta":      np.nan,
            "label_cashflow_delta":       np.nan,
            "label_order_velocity_delta": np.nan,
        }

    pre_velocity  = len(pre_window) / max((as_of - pre_window["created_at"].min()).days, 1)
    post_velocity = len(post_window) / horizon_days

    pre_cash  = pre_window["cash_balance"].iloc[-1]  if "cash_balance"  in pre_window.columns  else 0.0
    post_cash = post_window["cash_balance"].iloc[-1] if "cash_balance"  in post_window.columns else 0.0

    pre_inv  = pre_window["inventory_qty"].iloc[-1]  if "inventory_qty" in pre_window.columns  else 0.0
    post_inv = post_window["inventory_qty"].iloc[-1] if "inventory_qty" in post_window.columns else 0.0

    return {
        "label_inventory_delta":      float(post_inv - pre_inv),
        "label_cashflow_delta":       float(post_cash - pre_cash),
        "label_order_velocity_delta": float(post_velocity - pre_velocity),
    }


# ── Main transform ────────────────────────────────────────────────────────────

def transform(df_raw: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """
    Convert raw Shopify order rows into one feature row per synthetic
    'branch event' — sampled at each merchant's order timestamps.

    Strategy: treat each order as a potential branch point and compute
    features + labels around it. This creates many rows per merchant,
    which is exactly what we want for a cold-start training corpus.
    """
    df_raw = df_raw.copy()
    df_raw["created_at"] = pd.to_datetime(df_raw["created_at"], utc=True).dt.tz_localize(None)
    if "cancelled_at" in df_raw.columns:
        df_raw["cancelled_at"] = pd.to_datetime(df_raw["cancelled_at"], errors="coerce")

    rows = []
    for shop_id, group in df_raw.groupby("shop_id"):
        group = group.sort_values("created_at").reset_index(drop=True)
        category = map_category(group["product_type"].mode()[0] if "product_type" in group.columns else "retail")

        # Sample branch points: every 7th order (avoids too-similar consecutive rows)
        sample_indices = range(30, len(group), 7)

        for idx in sample_indices:
            as_of = group.iloc[idx]["created_at"]

            row = {
                "branch_id":   f"shopify_{shop_id}_{idx}",
                "business_id": f"shopify_{shop_id}",
                "category":    category,
                "branch_type": "shopify_order",   # cold-start rows have no branch type
                "committed_at": as_of.isoformat(),
            }

            # Rolling windows
            for w in [7, 30, 90]:
                row.update(rolling_features(group, w, as_of))

            # Seasonality
            row.update(seasonality_features(as_of))

            # Scenario delta / history depth (no branch concept in Shopify — use 0 / actual depth)
            row["scenario_magnitude"]   = 0.0
            row["deviation_from_norm"]  = 0.0
            row["history_depth_days"]   = (as_of - group["created_at"].min()).days
            row["n_past_price_changes"] = 0
            row["hist_avg_price_change"] = 0.0

            # Category one-hots
            row["cat_retail"]  = int(category == "retail")
            row["cat_apparel"] = int(category == "apparel")
            row["cat_b2b"]     = int(category == "b2b")

            # Branch type one-hots — all zero for Shopify rows
            for bt in ["bt_price_increase","bt_price_decrease","bt_bulk_restock",
                       "bt_promo_event","bt_new_collection","bt_markdown",
                       "bt_contract_change","bt_net_terms_change",
                       "bt_volume_discount","bt_supplier_change"]:
                row[bt] = 0

            # Labels
            row.update(compute_labels(group, as_of, horizon_days))

            rows.append(row)

    result = pd.DataFrame(rows)
    # Drop rows where we couldn't compute labels (edge of dataset)
    result = result.dropna(subset=["label_inventory_delta", "label_cashflow_delta", "label_order_velocity_delta"])
    print(f"Shopify adapter: {len(result):,} training rows from {df_raw['shop_id'].nunique():,} merchants")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Shopify export → features.csv format")
    parser.add_argument("--input",  required=True,  help="Path to raw Shopify CSV")
    parser.add_argument("--output", default=None,   help="Output path (default: stdout)")
    parser.add_argument("--horizon", type=int, default=14, help="Label horizon in days")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    df_raw = pd.read_csv(args.input)
    df_out = transform(df_raw, horizon_days=args.horizon)

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"Written to {args.output}")
    else:
        df_out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()