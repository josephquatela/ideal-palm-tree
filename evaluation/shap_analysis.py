"""
evaluation/shap_analysis.py
---------------------------
SHAP-based feature importance for the XGBoost models.

Uses shap.TreeExplainer — exact, fast, and model-aware for gradient-boosted trees.
Ridge baseline uses Ridge |coefficients| (see models/baseline.py::top_features).
Transformer SHAP would require GradientExplainer and is out of scope here.

Usage
-----
    from evaluation.shap_analysis import shap_summary, shap_importance, shap_category_comparison

    # After training XGBoost:
    shap_summary(xg_results, X_train, target='label_inventory_delta')
    shap_importance(xg_results, X_train, target='label_inventory_delta', n=15)

    # After training per-category XGBoost models:
    shap_category_comparison(cat_results, X_by_cat, target='label_inventory_delta')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

shap.initjs()   # required for JS-based plots in notebooks

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]

SHORT = {
    "label_inventory_delta":      "Inventory Delta",
    "label_cashflow_delta":       "Cash Flow Delta",
    "label_order_velocity_delta": "Order Velocity Delta",
}

DROP_COLS = [
    "branch_id", "business_id", "category", "branch_type", "committed_at",
] + TARGETS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix (no meta or target columns)."""
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    return df[feat_cols].fillna(0).astype(float)


def _explainer_and_values(model, X: pd.DataFrame):
    """
    Build a TreeExplainer for the XGBoost point model and compute SHAP values.
    Returns (explainer, shap_values array, feature_names list).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values, list(X.columns)


# ── Beeswarm summary plot ─────────────────────────────────────────────────────

def shap_summary(
    xgb_results: dict,
    train_df: pd.DataFrame,
    target: str = "label_inventory_delta",
    max_display: int = 20,
) -> None:
    """
    Beeswarm plot of SHAP values for one target.

    Each dot is one training sample. X-axis = SHAP value (impact on prediction).
    Color = feature value (red = high, blue = low).
    Features sorted by mean |SHAP| — most impactful at top.

    Parameters
    ----------
    xgb_results : dict returned by models.xgboost_model.train_xgboost
    train_df    : the training split DataFrame (same used to train the model)
    target      : one of the three TARGETS strings
    max_display : number of features to show (default 20)
    """
    model = xgb_results[target]["model_point"]
    X     = _get_X(train_df)

    _, shap_values, feat_names = _explainer_and_values(model, X)

    plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names   = feat_names,
        max_display     = max_display,
        show            = False,
    )
    plt.title(f"SHAP summary — {SHORT[target]}", fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()


# ── Mean |SHAP| bar chart ─────────────────────────────────────────────────────

def shap_importance(
    xgb_results: dict,
    train_df: pd.DataFrame,
    target: str = "label_inventory_delta",
    n: int = 15,
) -> pd.DataFrame:
    """
    Horizontal bar chart of mean |SHAP| per feature (top-n).
    Returns a DataFrame of feature importances for further use.

    Mean |SHAP| is preferred over XGBoost's built-in gain because it measures
    the actual average impact on model output in the prediction unit, not a
    tree-structure proxy.
    """
    model = xgb_results[target]["model_point"]
    X     = _get_X(train_df)

    _, shap_values, feat_names = _explainer_and_values(model, X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.4)))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1],
            color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance (mean |SHAP|) — {SHORT[target]}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return importance


# ── Per-category side-by-side importance ─────────────────────────────────────

def shap_category_comparison(
    cat_results: dict,
    train_by_cat: dict,
    target: str = "label_inventory_delta",
    n: int = 10,
) -> None:
    """
    Side-by-side mean |SHAP| bar charts for retail, apparel, and B2B.
    Highlights which features drive predictions differently per category.

    Parameters
    ----------
    cat_results  : dict mapping category name → xgb_results dict
                   e.g. {'retail': xg_retail, 'apparel': xg_apparel, 'b2b': xg_b2b}
    train_by_cat : dict mapping category name → training DataFrame
    target       : target to compare
    n            : top-n features per category
    """
    categories = list(cat_results.keys())
    fig, axes  = plt.subplots(1, len(categories), figsize=(7 * len(categories), max(5, n * 0.5)))

    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        model = cat_results[cat][target]["model_point"]
        X     = _get_X(train_by_cat[cat])

        _, shap_values, feat_names = _explainer_and_values(model, X)

        mean_abs = np.abs(shap_values).mean(axis=0)
        top = (
            pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
        )

        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                color="steelblue", alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(cat.upper())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Per-category feature importance — {SHORT[target]}",
        fontweight="bold", fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.show()


# ── All-targets summary ───────────────────────────────────────────────────────

def shap_all_targets(
    xgb_results: dict,
    train_df: pd.DataFrame,
    n: int = 10,
) -> None:
    """
    Run shap_importance for all three targets in one call, producing three plots.
    """
    for target in TARGETS:
        shap_importance(xgb_results, train_df, target=target, n=n)
