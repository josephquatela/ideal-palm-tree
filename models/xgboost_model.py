"""
models/xgboost_model.py
-----------------------
XGBoost model for the Business Rehearsal Outcome Predictor.

Trains three XGBoost models per prediction target:
  - Point prediction:  reg:squarederror
  - Lower 80% CI:      reg:quantileerror at alpha=0.10
  - Upper 80% CI:      reg:quantileerror at alpha=0.90

Uses the same temporal train/test split as baseline.py so that MAPE and
calibration scores are directly comparable. No feature scaling is applied —
tree-based models are invariant to monotone feature transformations.

Usage:
    python models/xgboost_model.py
    python models/xgboost_model.py --features data/processed/training_data.csv
    python models/xgboost_model.py --n-estimators 500 --max-depth 6 --top-features
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Column definitions ────────────────────────────────────────────────────────

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]

DROP_COLS = [
    "branch_id",
    "business_id",
    "category",
    "branch_type",
    "committed_at",
] + TARGETS

TRAIN_RATIO = 0.80
CI_LOWER_Q  = 0.10   # lower quantile for ~80% prediction interval
CI_UPPER_Q  = 0.90   # upper quantile for ~80% prediction interval


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
    rows come strictly before all test rows — identical logic to baseline.py.
    """
    df_sorted = df.sort_values("committed_at").reset_index(drop=True)
    cutoff_idx  = int(len(df_sorted) * train_ratio)
    cutoff_date = df_sorted.iloc[cutoff_idx]["committed_at"]

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
    """Mean Absolute Percentage Error, skipping rows where actual ≈ 0."""
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calibration_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Fraction of test points that fall within [y_lower, y_upper].
    A well-calibrated 80% interval should score close to 0.80.
    Uses actual quantile bounds rather than a normal-distribution z-interval,
    so the measurement is model-agnostic and directly interpretable.
    """
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_estimators: int     = 300,
    max_depth: int        = 5,
    learning_rate: float  = 0.05,
    subsample: float      = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
) -> dict:
    """
    For each target: fit a point-prediction model and two quantile models
    (lower/upper) to produce calibrated 80% prediction intervals.
    """
    shared_params = dict(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight,
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist",
    )

    results = {}

    for target in TARGETS:
        X_train, y_train = get_X_y(train, target)
        X_test,  y_test  = get_X_y(test,  target)

        print(f"\nTraining: {target.replace('label_', '').replace('_delta', '')}")

        model_point = xgb.XGBRegressor(objective="reg:squarederror", **shared_params)
        model_point.fit(X_train, y_train, verbose=False)
        y_pred = model_point.predict(X_test)
        print(f"  point model done")

        model_lower = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=CI_LOWER_Q, **shared_params
        )
        model_lower.fit(X_train, y_train, verbose=False)
        y_lower = model_lower.predict(X_test)
        print(f"  lower quantile ({CI_LOWER_Q}) done")

        model_upper = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=CI_UPPER_Q, **shared_params
        )
        model_upper.fit(X_train, y_train, verbose=False)
        y_upper = model_upper.predict(X_test)
        print(f"  upper quantile ({CI_UPPER_Q}) done")

        results[target] = {
            "model_point":  model_point,
            "model_lower":  model_lower,
            "model_upper":  model_upper,
            "feature_cols": list(X_train.columns),
            "y_test":       y_test.values,
            "y_pred":       y_pred,
            "y_lower":      y_lower,
            "y_upper":      y_upper,
            "mape":         mape(y_test.values, y_pred),
            "calibration":  calibration_score(y_test.values, y_lower, y_upper),
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
        mape_str = f"{r['mape']:.1f}%" if not np.isnan(r['mape']) else "  n/a"
        cal_str  = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_str:>8}  {cal_str:>22}")

    print()
    print("Interpretation guide:")
    print("  MAPE < 15%   → strong; beats baseline comfortably")
    print("  MAPE 15-30%  → acceptable, on par with or better than baseline")
    print("  MAPE > 30%   → weak — revisit hyperparameters or feature set")
    print()
    print("  Calibration near 0.80 → quantile intervals are honest")
    print("  Calibration >> 0.80   → intervals too wide (conservative)")
    print("  Calibration << 0.80   → intervals too narrow (overconfident)")


def top_features(results: dict, n: int = 10) -> None:
    """Print top-N features per target by XGBoost gain importance."""
    print("\nTop features per target (XGBoost gain importance):")
    for target, r in results.items():
        scores = r["model_point"].get_booster().get_score(importance_type="gain")
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

        short = target.replace("label_", "").replace("_delta", "")
        print(f"\n  {short}:")
        for feat, gain in ranked:
            print(f"    {feat:<45}  gain={gain:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost models for business outcome prediction."
    )
    parser.add_argument("--features", type=str,
                        default="data/processed/training_data.csv",
                        help="Path to engineered features CSV (output of engineer.py)")
    parser.add_argument("--train-ratio",        type=float, default=TRAIN_RATIO)
    parser.add_argument("--n-estimators",       type=int,   default=300)
    parser.add_argument("--max-depth",          type=int,   default=5)
    parser.add_argument("--learning-rate",      type=float, default=0.05)
    parser.add_argument("--subsample",          type=float, default=0.8)
    parser.add_argument("--colsample-bytree",   type=float, default=0.8)
    parser.add_argument("--min-child-weight",   type=int,   default=5)
    parser.add_argument("--top-features",       action="store_true",
                        help="Print top-10 features per target by XGBoost gain")
    args = parser.parse_args()

    if not Path(args.features).exists():
        raise FileNotFoundError(
            f"Features file not found: {args.features}\n"
            f"Run features/engineer.py first."
        )

    df = load_features(args.features)
    train, test = temporal_split(df, train_ratio=args.train_ratio)
    results = train_xgboost(
        train, test,
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        learning_rate    = args.learning_rate,
        subsample        = args.subsample,
        colsample_bytree = args.colsample_bytree,
        min_child_weight = args.min_child_weight,
    )
    print_results(results)

    if args.top_features:
        top_features(results)

    return results


if __name__ == "__main__":
    main()
