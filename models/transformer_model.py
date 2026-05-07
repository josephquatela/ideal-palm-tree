"""
models/transformer_model.py
---------------------------
FT-Transformer (Feature Tokenizer + Transformer) for the Business Rehearsal
Outcome Predictor.

Architecture:
  1. FeatureTokenizer  — each input scalar is projected to a D-dim embedding
                         via a separate learned linear layer per feature.
  2. [CLS] token       — a learnable embedding prepended to the feature tokens.
  3. TransformerEncoder — N layers of multi-head self-attention + FFN (Pre-LN).
  4. Prediction head   — 3-output MLP on the [CLS] state:
                         (point, lower_10, upper_90).

Loss: MSE(point) + PinballLoss(lower, α=0.10) + PinballLoss(upper, α=0.90)
      trained jointly so all three outputs share the same feature representation.

One model is trained per target, matching the structure of baseline.py and
xgboost_model.py so the Week 4 comparison notebook can read the same
MAPE + calibration table across all three approaches.

Usage:
    python models/transformer_model.py
    python models/transformer_model.py --epochs 200 --embed-dim 128
    python models/transformer_model.py --embed-dim 64 --n-heads 4 --n-layers 3
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
VAL_RATIO   = 0.15   # carved from the training split, temporally
CI_LOWER_Q  = 0.10
CI_UPPER_Q  = 0.90


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["committed_at"])
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Date range: {df['committed_at'].min().date()} → {df['committed_at'].max().date()}")
    print(f"  Businesses: {df['business_id'].nunique():,}")
    print(f"  Branches:   {len(df):,}")
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    df_sorted = df.sort_values("committed_at").reset_index(drop=True)
    cutoff_idx  = int(len(df_sorted) * train_ratio)
    cutoff_date = df_sorted.iloc[cutoff_idx]["committed_at"]

    train = df_sorted[df_sorted["committed_at"] <  cutoff_date].copy()
    test  = df_sorted[df_sorted["committed_at"] >= cutoff_date].copy()

    print(f"\nTemporal split at {cutoff_date.date()} ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print(f"  Train rows: {len(train):,}  |  Test rows: {len(test):,}")
    return train, test


def get_X_y(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).astype(float)
    y = df[target].astype(float)
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calibration_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


# ── Loss ──────────────────────────────────────────────────────────────────────

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Quantile (pinball) loss for a single quantile level."""
    diff = target - pred
    return torch.where(diff >= 0, alpha * diff, (alpha - 1) * diff).mean()


def combined_loss(
    outputs: torch.Tensor,
    y: torch.Tensor,
    lower_alpha: float = CI_LOWER_Q,
    upper_alpha: float = CI_UPPER_Q,
) -> torch.Tensor:
    """MSE on point prediction + pinball loss on lower and upper quantiles."""
    point  = outputs[:, 0]
    lower  = outputs[:, 1]
    upper  = outputs[:, 2]
    return (
        F.mse_loss(point, y)
        + pinball_loss(lower, y, lower_alpha)
        + pinball_loss(upper, y, upper_alpha)
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Projects each input scalar to an embed_dim-dimensional token using a
    separate learned weight and bias per feature — the core idea of FT-Transformer.
    Input:  (batch, n_features)
    Output: (batch, n_features, embed_dim)
    """
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, embed_dim))
        self.bias   = nn.Parameter(torch.zeros(n_features, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, 1) * (F, D) → (B, F, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LayerNorm → Attention → residual, then LayerNorm → FFN → residual."""
    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    Full FT-Transformer.
    - Tokenizes each feature to an embedding.
    - Prepends a learnable [CLS] token.
    - Applies N Transformer encoder blocks.
    - Reads the [CLS] output and passes it through a 3-output head:
      index 0 = point prediction, 1 = lower quantile, 2 = upper quantile.
    """
    def __init__(
        self,
        n_features:  int,
        embed_dim:   int = 64,
        n_heads:     int = 4,
        n_layers:    int = 3,
        ffn_dim:     int = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)                                    # (B, F, D)
        cls    = self.cls_token.expand(x.size(0), -1, -1)            # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                      # (B, F+1, D)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.norm(tokens[:, 0])                             # (B, D)
        return self.head(cls_out)                                     # (B, 3)


# ── Training ──────────────────────────────────────────────────────────────────

def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_one_target(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test_raw:  pd.DataFrame,
    y_test_raw:  pd.Series,
    device:      torch.device,
    embed_dim:   int,
    n_heads:     int,
    n_layers:    int,
    ffn_dim:     int,
    dropout:     float,
    lr:          float,
    batch_size:  int,
    epochs:      int,
    patience:    int,
    target_name: str,
) -> dict:
    # ── Validation split (temporal: last VAL_RATIO of train rows)
    val_cut  = int(len(X_train_raw) * (1 - VAL_RATIO))
    X_tr, y_tr   = X_train_raw.iloc[:val_cut],  y_train_raw.iloc[:val_cut]
    X_val, y_val = X_train_raw.iloc[val_cut:], y_train_raw.iloc[val_cut:]

    # ── Feature scaling — neural nets need it, unlike XGBoost
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s  = scaler.transform(X_test_raw).astype(np.float32)

    # ── Tensors + DataLoader
    tr_dataset = TensorDataset(
        _to_tensor(X_tr_s, device),
        _to_tensor(y_tr.values, device),
    )
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = _to_tensor(X_val_s, device)
    y_val_t = _to_tensor(y_val.values, device)
    X_te_t  = _to_tensor(X_te_s, device)

    # ── Model
    n_features = X_tr_s.shape[1]
    model = FTTransformer(
        n_features=n_features,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop with early stopping on val MAPE
    best_val_mape  = float("inf")
    best_state     = None
    epochs_no_improve = 0

    short = target_name.replace("label_", "").replace("_delta", "")
    print(f"\nTraining: {short}  ({n_features} features, device={device})")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = combined_loss(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out   = model(X_val_t)
            val_point = val_out[:, 0].cpu().numpy()
        val_mape = mape(y_val.values, val_point)

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(tr_dataset)
            print(f"  epoch {epoch:>4}  loss={avg_loss:.4f}  val_mape={val_mape:.1f}%"
                  f"  best={best_val_mape:.1f}%")

        if epochs_no_improve >= patience:
            print(f"  early stop at epoch {epoch}  (no improvement for {patience} epochs)")
            break

    # ── Restore best weights and predict on test set
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_out = model(X_te_t).cpu().numpy()

    y_pred  = test_out[:, 0]
    y_lower = test_out[:, 1]
    y_upper = test_out[:, 2]

    return {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": list(X_train_raw.columns),
        "y_test":       y_test_raw.values,
        "y_pred":       y_pred,
        "y_lower":      y_lower,
        "y_upper":      y_upper,
        "mape":         mape(y_test_raw.values, y_pred),
        "calibration":  calibration_score(y_test_raw.values, y_lower, y_upper),
        "best_val_mape": best_val_mape,
    }


def train_transformer(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    embed_dim:  int   = 64,
    n_heads:    int   = 4,
    n_layers:   int   = 3,
    ffn_dim:    int   = 128,
    dropout:    float = 0.1,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    epochs:     int   = 150,
    patience:   int   = 20,
) -> dict:
    device = get_device()
    results = {}

    for target in TARGETS:
        X_train_raw, y_train_raw = get_X_y(train, target)
        X_test_raw,  y_test_raw  = get_X_y(test,  target)

        results[target] = train_one_target(
            X_train_raw, y_train_raw,
            X_test_raw,  y_test_raw,
            device      = device,
            embed_dim   = embed_dim,
            n_heads     = n_heads,
            n_layers    = n_layers,
            ffn_dim     = ffn_dim,
            dropout     = dropout,
            lr          = lr,
            batch_size  = batch_size,
            epochs      = epochs,
            patience    = patience,
            target_name = target,
        )

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    labels = {
        "label_inventory_delta":      "Inventory Delta     ",
        "label_cashflow_delta":       "Cash Flow Delta     ",
        "label_order_velocity_delta": "Order Velocity Delta",
    }

    header = f"\n{'Target':<24}  {'MAPE':>8}  {'Calibration (80% CI)':>22}  {'Val MAPE':>10}"
    print(header)
    print("─" * len(header))

    for target, r in results.items():
        mape_str     = f"{r['mape']:.1f}%"         if not np.isnan(r['mape'])         else "  n/a"
        val_mape_str = f"{r['best_val_mape']:.1f}%" if not np.isnan(r['best_val_mape']) else "  n/a"
        cal_str      = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_str:>8}  {cal_str:>22}  {val_mape_str:>10}")

    print()
    print("Interpretation guide:")
    print("  MAPE < 15%   → strong; beats baseline comfortably")
    print("  MAPE 15-30%  → acceptable, on par with or better than baseline")
    print("  MAPE > 30%   → weak — try more epochs, larger embed_dim, or lower dropout")
    print()
    print("  Calibration near 0.80 → quantile intervals are well-calibrated")
    print("  Calibration >> 0.80   → intervals too wide (conservative)")
    print("  Calibration << 0.80   → intervals too narrow (overconfident)")
    print()
    print("  Val MAPE = best validation MAPE seen during training (early-stop criterion)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an FT-Transformer for business outcome prediction."
    )
    parser.add_argument("--features",    type=str,   default="data/processed/training_data.csv")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    # Transformer architecture
    parser.add_argument("--embed-dim",   type=int,   default=64,
                        help="Feature embedding dimension D (default: 64)")
    parser.add_argument("--n-heads",     type=int,   default=4,
                        help="Number of attention heads (must divide embed-dim, default: 4)")
    parser.add_argument("--n-layers",    type=int,   default=3,
                        help="Number of Transformer encoder blocks (default: 3)")
    parser.add_argument("--ffn-dim",     type=int,   default=128,
                        help="FFN hidden dimension (default: 128 = 2x embed-dim)")
    parser.add_argument("--dropout",     type=float, default=0.1)
    # Training
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early-stop patience in epochs (default: 20)")
    args = parser.parse_args()

    if not Path(args.features).exists():
        raise FileNotFoundError(
            f"Features file not found: {args.features}\n"
            f"Run features/engineer.py first."
        )

    df = load_features(args.features)
    train, test = temporal_split(df, train_ratio=args.train_ratio)
    results = train_transformer(
        train, test,
        embed_dim  = args.embed_dim,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        ffn_dim    = args.ffn_dim,
        dropout    = args.dropout,
        lr         = args.lr,
        batch_size = args.batch_size,
        epochs     = args.epochs,
        patience   = args.patience,
    )
    print_results(results)
    return results


if __name__ == "__main__":
    main()
