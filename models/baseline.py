"""
models/baseline.py
------------------
Linear regression baseline for the Business Rehearsal Outcome Predictor.

Trains three separate Ridge regression models — one per prediction target:
  - label_inventory_delta
  - label_cashflow_delta
  - label_order_velocity_delta

Uses a temporal train/test split (no random shuffling) to simulate real-world
use where the model only ever knows the past.

Outputs MAPE and calibration score per target. This is the performance floor
that XGBoost (W3) must beat.

Usage:
    python models/baseline.py --features data/features.csv
    python models/baseline.py --features data/features.csv --horizon 30
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Column definitions ────────────────────────────────────────────────────────

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]

# Columns to drop before training — identifiers and labels
DROP_COLS = [
    "branch_id",
    "business_id",
    "category",
    "branch_type",
    "committed_at",
] + TARGETS

TRAIN_RATIO = 0.80   # 80% train, 20% test (temporal)
CONFIDENCE_Z = 1.28  # z-score for ~80% prediction interval


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["committed_at"])
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Date range: {df['committed_at'].min().date()} → {df['committed_at'].max().date()}")
    print(f"  Businesses: {df['business_id'].nunique():,}")
    print(f"  Branches:   {len(df):,}")
    return df


# ── Train / test split ────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    Sort by commit time and split at a date-level boundary so that all training
    rows come strictly before all test rows with no same-date contamination.

    Using an index-level cutoff risks splitting branches that share the same
    committed_at date across train and test, violating the no-leakage contract.
    This version finds the date at the index cutoff and then assigns every row
    with that date (or later) to the test set.
    """
    df_sorted = df.sort_values("committed_at").reset_index(drop=True)
    cutoff_idx  = int(len(df_sorted) * train_ratio)
    cutoff_date = df_sorted.iloc[cutoff_idx]["committed_at"]

    # Date-level split: all rows on cutoff_date go into test
    train = df_sorted[df_sorted["committed_at"] <  cutoff_date].copy()
    test  = df_sorted[df_sorted["committed_at"] >= cutoff_date].copy()

    print(f"\nTemporal split at {cutoff_date.date()} ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print(f"  Train rows: {len(train):,}  |  Test rows: {len(test):,}")
    return train, test


# ── Feature matrix ────────────────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df[target].astype(float)
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, skipping rows where actual ≈ 0
    (dividing by zero produces meaningless percentages).
    """
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calibration_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals_std: float,
    z: float = CONFIDENCE_Z,
) -> float:
    """
    Checks whether the stated ~80% prediction interval actually contains
    ~80% of the true values.

    A well-calibrated model should score close to 0.80.
    Over 0.80 → intervals are too wide (overconfident uncertainty).
    Under 0.80 → intervals are too narrow (underconfident uncertainty).
    """
    lower = y_pred - z * residuals_std
    upper = y_pred + z * residuals_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return float(coverage)


# ── Training ──────────────────────────────────────────────────────────────────

def train_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Fit one Ridge regression pipeline per target.
    Returns a results dict with model, predictions, and metrics.
    """
    results = {}

    for target in TARGETS:
        X_train, y_train = get_X_y(train, target)
        X_test,  y_test  = get_X_y(test,  target)

        # Split columns into continuous vs binary one-hot groups.
        # StandardScaler should NOT rescale binary columns (bt_*, cat_*, is_*):
        # mean-centering a 0/1 column produces negative values and shifts the
        # Ridge L2 penalty in ways that distort coefficient magnitudes.
        binary_cols = [c for c in X_train.columns
                       if c.startswith(("bt_", "cat_", "is_"))]
        cont_cols   = [c for c in X_train.columns if c not in binary_cols]

        # ColumnTransformer: scale continuous features, pass binary through.
        # The output column order is cont_cols then binary_cols.
        ct = ColumnTransformer([
            ("scale", StandardScaler(), cont_cols),
            ("pass",  "passthrough",    binary_cols),
        ])
        pipe = Pipeline([
            ("ct",    ct),
            ("ridge", Ridge(alpha=1.0)),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Residual std from training set — used to build prediction intervals
        train_resid_std = float(np.std(y_train - pipe.predict(X_train)))

        results[target] = {
            "model":          pipe,
            "y_test":         y_test.values,
            "y_pred":         y_pred,
            "resid_std":      train_resid_std,
            "mape":           mape(y_test.values, y_pred),
            "calibration":    calibration_score(y_test.values, y_pred, train_resid_std),
            # ColumnTransformer outputs continuous cols first, then binary cols.
            "feature_cols":   cont_cols + binary_cols,
        }

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    labels = {
        "label_inventory_delta":      "Inventory Delta     ",
        "label_cashflow_delta":       "Cash Flow Delta     ",
        "label_order_velocity_delta": "Order Velocity Delta",
    }

    header = f"\n{'Target':<24}  {'MAPE':>8}  {'Calibration (80% CI)':>22}"
    print(header)
    print("─" * len(header))

    for target, r in results.items():
        mape_str  = f"{r['mape']:.1f}%" if not np.isnan(r['mape']) else "  n/a"
        cal_str   = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_str:>8}  {cal_str:>22}")

    print()
    print("Interpretation guide:")
    print("  MAPE < 15%   → strong baseline")
    print("  MAPE 15-30%  → acceptable, useful for a founder")
    print("  MAPE > 30%   → weak — XGBoost should beat this comfortably")
    print()
    print("  Calibration near 0.80 → prediction intervals are honest")
    print("  Calibration >> 0.80   → intervals too wide")
    print("  Calibration << 0.80   → intervals too narrow")


def top_features(results: dict, n: int = 10) -> None:
    """Print the top-N most influential features per target (by |coefficient|)."""
    print("\nTop features per target (Ridge |coefficient|):")
    for target, r in results.items():
        pipe   = r["model"]
        coefs  = pipe.named_steps["ridge"].coef_
        cols   = r["feature_cols"]
        ranked = sorted(zip(cols, coefs), key=lambda x: abs(x[1]), reverse=True)[:n]

        short = target.replace("label_", "").replace("_delta", "")
        print(f"\n  {short}:")
        for feat, coef in ranked:
            print(f"    {feat:<45}  {coef:+.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline linear regression models.")
    parser.add_argument("--features", type=str,
                        default="data/processed/training_data.csv",
                        help="Path to engineered features CSV (output of engineer.py)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                        help="Fraction of data to use for training (default: 0.80)")
    parser.add_argument("--horizon", type=int, default=14,
                        help="Expected outcome horizon in days (default: 14). "
                             "Informational — asserts that the features CSV was "
                             "generated with the matching horizon setting.")
    parser.add_argument("--top-features", action="store_true",
                        help="Print top 10 features per target by Ridge coefficient")
    args = parser.parse_args()

    if not Path(args.features).exists():
        raise FileNotFoundError(f"Features file not found: {args.features}\n"
                                f"Run features/engineer.py first.")

    df = load_features(args.features)
    train, test = temporal_split(df, train_ratio=args.train_ratio)
    results = train_baseline(train, test)
    print_results(results)

    if args.top_features:
        top_features(results)

    return results


if __name__ == "__main__":
    main()