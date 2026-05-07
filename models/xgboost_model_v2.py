"""
models/xgboost_model_v2.py
--------------------------
Regularised XGBoost with temporal early stopping (Experiment 2 — Week 6).

Changes vs xgboost_model.py
----------------------------
  min_child_weight  5  → 15   fewer splits on thin leaf populations
  max_depth         5  → 4    shallower trees generalise better
  reg_alpha         0  → 0.1  L1 sparsity on leaf weights
  reg_lambda        1  → 2.0  stronger L2 shrinkage
  n_estimators    300  → 500  ceiling; actual count decided by early stopping
  early_stopping        new   temporal val split (last VAL_RATIO of train);
                              point model stops when val loss plateaus;
                              quantile models reuse the same best_iteration.

The external interface (results dict format, CLI flags) is identical to
xgboost_model.py so notebooks can swap the import and re-run unchanged.

Usage:
    python models/xgboost_model_v2.py
    python models/xgboost_model_v2.py --min-child-weight 20 --max-depth 3
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]
DROP_COLS = [
    "branch_id", "business_id", "category", "branch_type", "committed_at",
] + TARGETS

TRAIN_RATIO          = 0.80
VAL_RATIO            = 0.15   # carved from the tail of the training split
CI_LOWER_Q           = 0.10
CI_UPPER_Q           = 0.90
EARLY_STOPPING_ROUNDS = 30


def load_features(path):
    df = pd.read_csv(path, parse_dates=["committed_at"])
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Date range: {df['committed_at'].min().date()} → {df['committed_at'].max().date()}")
    print(f"  Businesses: {df['business_id'].nunique():,}  |  Branches: {len(df):,}")
    return df


def temporal_split(df, train_ratio=TRAIN_RATIO):
    df_sorted   = df.sort_values("committed_at").reset_index(drop=True)
    cutoff_idx  = int(len(df_sorted) * train_ratio)
    cutoff_date = df_sorted.iloc[cutoff_idx]["committed_at"]
    train = df_sorted[df_sorted["committed_at"] <  cutoff_date].copy()
    test  = df_sorted[df_sorted["committed_at"] >= cutoff_date].copy()
    print(f"\nTemporal split at {cutoff_date.date()} ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print(f"  Train rows: {len(train):,}  |  Test rows: {len(test):,}")
    return train, test


def get_X_y(df, target):
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].fillna(0).astype(float)
    y = df[target].astype(float)
    return X, y


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calibration_score(y_true, y_lower, y_upper):
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


def train_xgboost(
    train, test,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_alpha=0.1,
    reg_lambda=2.0,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
):
    # ── Internal temporal val split for early stopping ────────────────────────
    # Carved from the tail of train so it stays strictly before the test set.
    train_sorted = train.sort_values("committed_at").reset_index(drop=True)
    val_cut = int(len(train_sorted) * (1 - VAL_RATIO))
    train_model = train_sorted.iloc[:val_cut]
    val_es      = train_sorted.iloc[val_cut:]

    print(f"\nEarly-stopping val split: {len(train_model):,} model rows | {len(val_es):,} val rows")

    shared_params = dict(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight,
        reg_alpha        = reg_alpha,
        reg_lambda       = reg_lambda,
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist",
    )

    results = {}

    for target in TARGETS:
        X_tr,  y_tr  = get_X_y(train_model, target)
        X_val, y_val = get_X_y(val_es,       target)
        X_te,  y_te  = get_X_y(test,         target)

        short = target.replace("label_", "").replace("_delta", "")
        print(f"\nTraining: {short}")

        # Point model — early stopping decides actual n_estimators
        model_point = xgb.XGBRegressor(
            objective            = "reg:squarederror",
            early_stopping_rounds = early_stopping_rounds,
            **shared_params,
        )
        model_point.fit(
            X_tr, y_tr,
            eval_set  = [(X_val, y_val)],
            verbose   = False,
        )
        best_n = model_point.best_iteration + 1
        print(f"  point model: early stopped at {best_n} trees  "
              f"(val RMSE={model_point.best_score:.4f})")
        y_pred = model_point.predict(X_te)

        # Quantile models use the same best_n (no early stopping needed)
        quantile_params = {**shared_params, "n_estimators": best_n}

        model_lower = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=CI_LOWER_Q, **quantile_params
        )
        model_lower.fit(X_tr, y_tr, verbose=False)
        y_lower = model_lower.predict(X_te)
        print(f"  lower quantile ({CI_LOWER_Q}) done")

        model_upper = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=CI_UPPER_Q, **quantile_params
        )
        model_upper.fit(X_tr, y_tr, verbose=False)
        y_upper = model_upper.predict(X_te)
        print(f"  upper quantile ({CI_UPPER_Q}) done")

        results[target] = {
            "model_point":   model_point,
            "model_lower":   model_lower,
            "model_upper":   model_upper,
            "feature_cols":  list(X_tr.columns),
            "best_n_trees":  best_n,
            "y_test":        y_te.values,
            "y_pred":        y_pred,
            "y_lower":       y_lower,
            "y_upper":       y_upper,
            "mape":          mape(y_te.values, y_pred),
            "calibration":   calibration_score(y_te.values, y_lower, y_upper),
        }

    return results


def print_results(results):
    labels = {
        "label_inventory_delta":      "Inventory Delta     ",
        "label_cashflow_delta":       "Cash Flow Delta     ",
        "label_order_velocity_delta": "Order Velocity Delta",
    }
    header = f"\n{'Target':<24}  {'MAPE':>8}  {'Calibration (80% CI)':>22}  {'Best N':>8}"
    print(header)
    print("─" * len(header))
    for target, r in results.items():
        mape_s = f"{r['mape']:.1f}%" if not np.isnan(r['mape']) else "n/a"
        cal_s  = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_s:>8}  {cal_s:>22}  {r['best_n_trees']:>8}")


def top_features(results, n=10):
    print("\nTop features per target (XGBoost gain importance):")
    for target, r in results.items():
        scores = r["model_point"].get_booster().get_score(importance_type="gain")
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        short  = target.replace("label_", "").replace("_delta", "")
        print(f"\n  {short}:")
        for feat, gain in ranked:
            print(f"    {feat:<45}  gain={gain:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",             type=str,   default="data/processed/training_data.csv")
    parser.add_argument("--train-ratio",          type=float, default=TRAIN_RATIO)
    parser.add_argument("--n-estimators",         type=int,   default=500)
    parser.add_argument("--max-depth",            type=int,   default=4)
    parser.add_argument("--learning-rate",        type=float, default=0.05)
    parser.add_argument("--subsample",            type=float, default=0.8)
    parser.add_argument("--colsample-bytree",     type=float, default=0.8)
    parser.add_argument("--min-child-weight",     type=int,   default=15)
    parser.add_argument("--reg-alpha",            type=float, default=0.1)
    parser.add_argument("--reg-lambda",           type=float, default=2.0)
    parser.add_argument("--early-stopping-rounds",type=int,   default=EARLY_STOPPING_ROUNDS)
    parser.add_argument("--top-features",         action="store_true")
    args = parser.parse_args()

    df = load_features(args.features)
    train, test = temporal_split(df, args.train_ratio)
    results = train_xgboost(
        train, test,
        n_estimators          = args.n_estimators,
        max_depth             = args.max_depth,
        learning_rate         = args.learning_rate,
        subsample             = args.subsample,
        colsample_bytree      = args.colsample_bytree,
        min_child_weight      = args.min_child_weight,
        reg_alpha             = args.reg_alpha,
        reg_lambda            = args.reg_lambda,
        early_stopping_rounds = args.early_stopping_rounds,
    )
    print_results(results)
    if args.top_features:
        top_features(results)


if __name__ == "__main__":
    main()
