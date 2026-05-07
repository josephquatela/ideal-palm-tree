"""
models/grouped_model.py
-----------------------
Branch-type-grouped XGBoost models (Experiment 4 — Week 6).

Motivation
----------
The baseline report shows extreme MAPE variance by branch type:
  bulk_restock    ~23%   ← easy, direct inventory signal
  supplier_change ~617%  ← hard, indirect and noisy
  net_terms_change ~340% ← hard, cash-timing mechanics differ entirely

A single global model must find one set of weights that fits all branch types.
Grouping lets each sub-model specialise on its own signal structure.

Groups
------
  price_promo  : price_increase, price_decrease, promo_event,
                 markdown, volume_discount, new_collection
  inventory    : bulk_restock, supplier_change
  b2b_contract : contract_change, net_terms_change

At inference, each sample is routed to its group's model based on branch_type.
The returned results dict matches the xgboost_model.py format exactly so it
plugs into evaluate_all_models() without changes.

Usage
-----
    python models/grouped_model.py
    python models/grouped_model.py --min-child-weight 15 --max-depth 4
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

TRAIN_RATIO = 0.80
CI_LOWER_Q  = 0.10
CI_UPPER_Q  = 0.90

BRANCH_GROUPS = {
    "price_promo": [
        "price_increase", "price_decrease", "promo_event",
        "markdown", "volume_discount", "new_collection",
    ],
    "inventory": [
        "bulk_restock", "supplier_change",
    ],
    "b2b_contract": [
        "contract_change", "net_terms_change",
    ],
}

# Reverse map: branch_type → group name
BRANCH_TO_GROUP = {bt: g for g, types in BRANCH_GROUPS.items() for bt in types}


def load_features(path):
    df = pd.read_csv(path, parse_dates=["committed_at"])
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Date range: {df['committed_at'].min().date()} → {df['committed_at'].max().date()}")
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


def _train_group_model(X_tr, y_tr, objective, alpha=None, **shared):
    """Train one XGBoost model for a given group and objective."""
    kwargs = dict(objective=objective, **shared)
    if alpha is not None:
        kwargs["quantile_alpha"] = alpha
    model = xgb.XGBRegressor(**kwargs)
    model.fit(X_tr, y_tr, verbose=False)
    return model


def train_grouped(
    train, test,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_alpha=0.1,
    reg_lambda=2.0,
):
    """
    Train one XGBoost (point + lower + upper) per branch-type group per target.
    Returns a results dict with the same structure as xgboost_model.train_xgboost().

    Predictions are assembled in the original test-row order so that downstream
    evaluate_all_models() receives correctly aligned arrays.
    """
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

    # Reset index so integer positions are contiguous
    test_reset  = test.reset_index(drop=True)
    train_reset = train.reset_index(drop=True)

    # Print group counts
    print("\nBranch-type group sizes (train / test):")
    for group, branch_types in BRANCH_GROUPS.items():
        n_tr = train_reset["branch_type"].isin(branch_types).sum()
        n_te = test_reset["branch_type"].isin(branch_types).sum()
        print(f"  {group:<14}  train={n_tr:>4}  test={n_te:>3}  "
              f"types={branch_types}")

    results = {}

    for target in TARGETS:
        short = target.replace("label_", "").replace("_delta", "")
        print(f"\n── Target: {short} ──────────────────────────────────────")

        y_pred_full  = np.full(len(test_reset), np.nan)
        y_lower_full = np.full(len(test_reset), np.nan)
        y_upper_full = np.full(len(test_reset), np.nan)
        y_test_full  = get_X_y(test_reset, target)[1].values

        group_models = {}

        for group, branch_types in BRANCH_GROUPS.items():
            tr_mask  = train_reset["branch_type"].isin(branch_types)
            te_mask  = test_reset["branch_type"].isin(branch_types)
            train_g  = train_reset[tr_mask]
            test_idx = test_reset[te_mask].index.tolist()

            if len(train_g) < 10:
                print(f"  {group}: SKIPPED (only {len(train_g)} train rows)")
                continue
            if len(test_idx) == 0:
                print(f"  {group}: no test rows — training only")

            X_tr, y_tr = get_X_y(train_g, target)
            X_te, _    = get_X_y(test_reset.loc[test_idx], target)

            m_point = _train_group_model(X_tr, y_tr, "reg:squarederror",        **shared_params)
            m_lower = _train_group_model(X_tr, y_tr, "reg:quantileerror", CI_LOWER_Q, **shared_params)
            m_upper = _train_group_model(X_tr, y_tr, "reg:quantileerror", CI_UPPER_Q, **shared_params)

            if test_idx:
                y_pred_full[test_idx]  = m_point.predict(X_te)
                y_lower_full[test_idx] = m_lower.predict(X_te)
                y_upper_full[test_idx] = m_upper.predict(X_te)

            group_models[group] = {
                "model_point": m_point, "model_lower": m_lower, "model_upper": m_upper,
            }
            g_mape = mape(y_test_full[test_idx], y_pred_full[test_idx]) if test_idx else float("nan")
            print(f"  {group:<14}  train={len(train_g):>4}  test={len(test_idx):>3}  "
                  f"mape={g_mape:.1f}%")

        # Keep only rows that were assigned to a group
        valid = ~np.isnan(y_pred_full)
        n_unrouted = (~valid).sum()
        if n_unrouted > 0:
            print(f"  WARNING: {n_unrouted} test rows unrouted "
                  f"(branch types not in any group)")

        results[target] = {
            "group_models":  group_models,
            "feature_cols":  [c for c in test_reset.columns if c not in DROP_COLS],
            "y_test":        y_test_full[valid],
            "y_pred":        y_pred_full[valid],
            "y_lower":       y_lower_full[valid],
            "y_upper":       y_upper_full[valid],
            "mape":          mape(y_test_full[valid], y_pred_full[valid]),
            "calibration":   calibration_score(
                y_test_full[valid], y_lower_full[valid], y_upper_full[valid]
            ),
        }

    return results


def print_results(results):
    labels = {
        "label_inventory_delta":      "Inventory Delta     ",
        "label_cashflow_delta":       "Cash Flow Delta     ",
        "label_order_velocity_delta": "Order Velocity Delta",
    }
    header = f"\n{'Target':<24}  {'MAPE':>8}  {'Calibration (80% CI)':>22}"
    print(header)
    print("─" * len(header))
    for target, r in results.items():
        mape_s = f"{r['mape']:.1f}%" if not np.isnan(r['mape']) else "n/a"
        cal_s  = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_s:>8}  {cal_s:>22}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",         type=str,   default="data/processed/training_data.csv")
    parser.add_argument("--train-ratio",      type=float, default=TRAIN_RATIO)
    parser.add_argument("--n-estimators",     type=int,   default=300)
    parser.add_argument("--max-depth",        type=int,   default=5)
    parser.add_argument("--learning-rate",    type=float, default=0.05)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=int,   default=15)
    parser.add_argument("--reg-alpha",        type=float, default=0.1)
    parser.add_argument("--reg-lambda",       type=float, default=2.0)
    args = parser.parse_args()

    df = load_features(args.features)
    train, test = temporal_split(df, args.train_ratio)
    results = train_grouped(
        train, test,
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        learning_rate    = args.learning_rate,
        subsample        = args.subsample,
        colsample_bytree = args.colsample_bytree,
        min_child_weight = args.min_child_weight,
        reg_alpha        = args.reg_alpha,
        reg_lambda       = args.reg_lambda,
    )
    print_results(results)


if __name__ == "__main__":
    main()
