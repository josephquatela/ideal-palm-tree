"""
evaluation/metrics.py
---------------------
Unified evaluation metrics for all three Business Rehearsal models.

Metrics
-------
MAPE   — Mean Absolute Percentage Error (accuracy, lower is better)
RMSE   — Root Mean Squared Error (accuracy, lower is better)
PICP   — Prediction Interval Coverage Probability (should be ~0.80 for 80% CI)
MPIW   — Mean Prediction Interval Width (smaller = tighter intervals)
NMPIW  — Normalised MPIW: MPIW / std(y_true) (scale-free, comparable across targets)

Usage
-----
    from evaluation.metrics import evaluate_all_models, plot_calibration, plot_interval_widths

    results = {'Baseline': bl, 'XGBoost': xg, 'Transformer': tr}
    df = evaluate_all_models(results)
    plot_calibration(results)
    plot_interval_widths(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_PICP = 0.80

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]

SHORT = {
    "label_inventory_delta":      "Inventory",
    "label_cashflow_delta":       "Cash Flow",
    "label_order_velocity_delta": "Order Velocity",
}


# ── Individual metrics ────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability — target 0.80 for an 80% CI."""
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))

# Alias used by evaluation/conformal.py and notebooks
calibration_score = picp


def mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean Prediction Interval Width — smaller means tighter (more useful) intervals."""
    return float(np.mean(y_upper - y_lower))


def nmpiw(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Normalised MPIW: MPIW divided by the standard deviation of y_true.
    Scale-free — allows fair comparison across targets with different units.
    """
    sigma = np.std(y_true)
    if sigma < 1e-9:
        return float("nan")
    return mpiw(y_lower, y_upper) / sigma


# ── Per-target summary ────────────────────────────────────────────────────────

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> dict:
    """Return all five metrics for one model / target pair."""
    return {
        "MAPE (%)": round(mape(y_true, y_pred), 1),
        "RMSE":     round(rmse(y_true, y_pred), 2),
        "PICP":     round(picp(y_true, y_lower, y_upper), 3),
        "MPIW":     round(mpiw(y_lower, y_upper), 2),
        "NMPIW":    round(nmpiw(y_true, y_lower, y_upper), 3),
    }


# ── Cross-model evaluation ────────────────────────────────────────────────────

def evaluate_all_models(results_by_model: dict) -> pd.DataFrame:
    """
    Compute all metrics for every model × target combination.

    Parameters
    ----------
    results_by_model : dict
        Keys are model names (e.g. 'Baseline', 'XGBoost', 'Transformer').
        Values are the results dicts returned by each model's train_* function.
        Each per-target sub-dict must contain y_test, y_pred, y_lower, y_upper.

    Returns
    -------
    pd.DataFrame with columns [Model, Target, MAPE (%), RMSE, PICP, MPIW, NMPIW].
    """
    rows = []
    for model_name, results in results_by_model.items():
        for target in TARGETS:
            r = results[target]
            metrics = compute_metrics(
                r["y_test"], r["y_pred"], r["y_lower"], r["y_upper"]
            )
            rows.append({"Model": model_name, "Target": SHORT[target], **metrics})
    return pd.DataFrame(rows)


def print_metrics_table(df: pd.DataFrame) -> None:
    """Pretty-print the evaluate_all_models DataFrame as a pivot per metric."""
    for metric in ["MAPE (%)", "RMSE", "PICP", "MPIW", "NMPIW"]:
        pivot = df.pivot(index="Target", columns="Model", values=metric)
        print(f"\n{metric}")
        if metric == "PICP":
            print(f"  (target: {TARGET_PICP})")
        print(pivot.to_string())
    print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_calibration(results_by_model: dict, figsize=(14, 4)) -> None:
    """
    Grouped bar chart of PICP per model and target.
    Red dashed line marks the 0.80 ideal.
    """
    df     = evaluate_all_models(results_by_model)
    models = list(results_by_model.keys())
    tgts   = [SHORT[t] for t in TARGETS]
    x      = np.arange(len(tgts))
    w      = 0.8 / len(models)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            df.loc[(df["Model"] == model) & (df["Target"] == t), "PICP"].values[0]
            for t in tgts
        ]
        ax.bar(x + i * w - (len(models) - 1) * w / 2, vals, w,
               label=model, color=color, alpha=0.85)

    ax.axhline(TARGET_PICP, color="red", linestyle="--", lw=1.5, label=f"target {TARGET_PICP}")
    ax.set_xticks(x)
    ax.set_xticklabels(tgts)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("Prediction Interval Coverage — closer to 0.80 is better")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_interval_widths(results_by_model: dict, figsize=(14, 4)) -> None:
    """
    Grouped bar chart of NMPIW per model and target.
    Smaller NMPIW = tighter intervals (more useful to founders).
    """
    df     = evaluate_all_models(results_by_model)
    models = list(results_by_model.keys())
    tgts   = [SHORT[t] for t in TARGETS]
    x      = np.arange(len(tgts))
    w      = 0.8 / len(models)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            df.loc[(df["Model"] == model) & (df["Target"] == t), "NMPIW"].values[0]
            for t in tgts
        ]
        ax.bar(x + i * w - (len(models) - 1) * w / 2, vals, w,
               label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tgts)
    ax.set_ylabel("NMPIW (normalised interval width)")
    ax.set_title("Interval Width — smaller is tighter (better for founders)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_mape(results_by_model: dict, figsize=(14, 4)) -> None:
    """Grouped bar chart of MAPE per model and target."""
    df     = evaluate_all_models(results_by_model)
    models = list(results_by_model.keys())
    tgts   = [SHORT[t] for t in TARGETS]
    x      = np.arange(len(tgts))
    w      = 0.8 / len(models)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            df.loc[(df["Model"] == model) & (df["Target"] == t), "MAPE (%)"].values[0]
            for t in tgts
        ]
        ax.bar(x + i * w - (len(models) - 1) * w / 2, vals, w,
               label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tgts)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by model and target — lower is better")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from models.baseline       import load_features, temporal_split, train_baseline
    from models.xgboost_model  import train_xgboost
    from models.transformer_model import train_transformer

    df    = load_features("data/processed/training_data.csv")
    train, test = temporal_split(df)

    print("Training models...")
    bl = train_baseline(df_train=train, df_test=test) if False else train_baseline(train, test)
    xg = train_xgboost(train, test)

    all_results = {"Baseline": bl, "XGBoost": xg}
    ev = evaluate_all_models(all_results)
    print_metrics_table(ev)
    plot_mape(all_results)
    plot_calibration(all_results)
    plot_interval_widths(all_results)
