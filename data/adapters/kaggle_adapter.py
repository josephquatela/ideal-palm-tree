"""
data/adapters/kaggle_adapter.py
--------------------------------
Adapter for Kaggle small-business datasets.

Handles two datasets queued for W2:
  1. Retail Transactions   — transaction-level sales data
  2. Fashion Retail Sales  — apparel-specific order history

Both get normalized to the same features.csv schema used by engineer.py,
so baseline.py and all downstream models work unchanged.

Expected input schemas:

  Retail Transactions CSV:
    transaction_id, customer_id, store_id, date, quantity, unit_price,
    product_category, total_amount, payment_method

  Fashion Retail Sales CSV:
    Order_ID, Customer_ID, Date, Product_Type, Quantity, Price,
    Total_Sales, Returns

Usage:
    # Retail
    python data/adapters/kaggle_adapter.py \\
        --dataset retail \\
        --input  data/raw/retail_transactions.csv \\
        --output data/kaggle_retail_features.csv

    # Fashion
    python data/adapters/kaggle_adapter.py \\
        --dataset fashion \\
        --input  data/raw/fashion_retail_sales.csv \\
        --output data/kaggle_fashion_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use shared helpers from shopify_adapter to avoid duplication
sys.path.insert(0, str(Path(__file__).parent))
from shopify_adapter import rolling_features, seasonality_features, compute_labels


# ── Schema normalizers ────────────────────────────────────────────────────────
# Each function takes a raw dataframe and returns a unified schema with columns:
#   shop_id, created_at, total_price, cancelled_at, product_type,
#   inventory_qty, cash_balance

def normalize_retail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retail Transactions dataset normalization.
    Groups by store_id as the 'merchant' unit.
    """
    out = pd.DataFrame()
    out["shop_id"]       = df["store_id"].astype(str)
    out["created_at"]    = pd.to_datetime(df["date"])
    out["total_price"]   = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)
    out["cancelled_at"]  = pd.NaT    # no cancellation field in this dataset
    out["product_type"]  = df["product_category"].fillna("retail")
    out["inventory_qty"] = pd.to_numeric(df.get("quantity", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    out["cash_balance"]  = out.groupby("shop_id")["total_price"].cumsum()  # proxy: running revenue
    return out.dropna(subset=["created_at"])


def normalize_fashion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fashion Retail Sales dataset normalization.
    Customer_ID is the 'merchant' unit (each customer is a small fashion store proxy).
    Returns are treated as cancellations.
    """
    out = pd.DataFrame()
    out["shop_id"]      = df["Customer_ID"].astype(str)
    out["created_at"]   = pd.to_datetime(df["Date"])
    out["total_price"]  = pd.to_numeric(df["Total_Sales"], errors="coerce").fillna(0)

    # If "Returns" column exists, treat non-zero returns as partial cancellation
    if "Returns" in df.columns:
        returns = pd.to_numeric(df["Returns"], errors="coerce").fillna(0)
        out["cancelled_at"] = pd.NaT
        out.loc[returns > 0, "cancelled_at"] = out.loc[returns > 0, "created_at"]
    else:
        out["cancelled_at"] = pd.NaT

    out["product_type"]  = df.get("Product_Type", pd.Series("apparel", index=df.index)).fillna("apparel")
    out["inventory_qty"] = pd.to_numeric(df.get("Quantity", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    out["cash_balance"]  = out.groupby("shop_id")["total_price"].cumsum()
    return out.dropna(subset=["created_at"])


NORMALIZERS = {
    "retail":  normalize_retail,
    "fashion": normalize_fashion,
}


# ── Category assignment ───────────────────────────────────────────────────────

def infer_category(product_type: str) -> str:
    pt = str(product_type).lower()
    if any(k in pt for k in ["apparel","fashion","clothing","wear","dress","shoe"]):
        return "apparel"
    if any(k in pt for k in ["service","consult","b2b","wholesale","software"]):
        return "b2b"
    return "retail"


# ── Main transform ────────────────────────────────────────────────────────────

def transform(df_normalized: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """
    Identical sampling strategy to shopify_adapter: treat each order as a
    potential branch point, compute rolling window features and labels.
    """
    rows = []

    for shop_id, group in df_normalized.groupby("shop_id"):
        group    = group.sort_values("created_at").reset_index(drop=True)
        category = infer_category(group["product_type"].mode()[0])

        # Need at least 30 rows of history + 14 days future to get a label
        if len(group) < 45:
            continue

        sample_indices = range(30, len(group), 7)

        for idx in sample_indices:
            as_of = group.iloc[idx]["created_at"]
            future_cutoff = as_of + pd.Timedelta(days=horizon_days)

            # Skip if we don't have enough future data for a label
            if future_cutoff > group["created_at"].max():
                continue

            row = {
                "branch_id":    f"kaggle_{shop_id}_{idx}",
                "business_id":  f"kaggle_{shop_id}",
                "category":     category,
                "branch_type":  "kaggle_transaction",
                "committed_at": as_of.isoformat(),
            }

            for w in [7, 30, 90]:
                row.update(rolling_features(group, w, as_of))

            row.update(seasonality_features(as_of))

            row["scenario_magnitude"]    = 0.0
            row["deviation_from_norm"]   = 0.0
            row["history_depth_days"]    = (as_of - group["created_at"].min()).days
            row["n_past_price_changes"]  = 0
            row["hist_avg_price_change"] = 0.0

            row["cat_retail"]  = int(category == "retail")
            row["cat_apparel"] = int(category == "apparel")
            row["cat_b2b"]     = int(category == "b2b")

            for bt in ["bt_price_increase","bt_price_decrease","bt_bulk_restock",
                       "bt_promo_event","bt_new_collection","bt_markdown",
                       "bt_contract_change","bt_net_terms_change",
                       "bt_volume_discount","bt_supplier_change"]:
                row[bt] = 0

            row.update(compute_labels(group, as_of, horizon_days))
            rows.append(row)

    result = pd.DataFrame(rows)
    result = result.dropna(subset=["label_inventory_delta","label_cashflow_delta","label_order_velocity_delta"])
    print(f"Kaggle adapter: {len(result):,} training rows from {df_normalized['shop_id'].nunique():,} merchants")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle dataset → features.csv format")
    parser.add_argument("--dataset", required=True, choices=list(NORMALIZERS.keys()),
                        help="Which Kaggle dataset: 'retail' or 'fashion'")
    parser.add_argument("--input",   required=True,  help="Path to raw Kaggle CSV")
    parser.add_argument("--output",  default=None,   help="Output path (default: stdout)")
    parser.add_argument("--horizon", type=int, default=14, help="Label horizon in days")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    df_raw        = pd.read_csv(args.input)
    normalize_fn  = NORMALIZERS[args.dataset]
    df_normalized = normalize_fn(df_raw)
    df_out        = transform(df_normalized, horizon_days=args.horizon)

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"Written to {args.output}")
    else:
        df_out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()