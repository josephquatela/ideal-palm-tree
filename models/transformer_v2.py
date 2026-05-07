"""
models/transformer_v2.py
------------------------
FT-Transformer with per-target StandardScaler on y (Experiment 1 — Week 6).

Key change vs transformer_model.py:
  Each target's labels are normalised to zero mean / unit variance before
  training.  The combined MSE + pinball loss then operates in the same scale
  space for all three targets, preventing the large-magnitude cash flow target
  from drowning out the quantile head for the other two targets.

  Predictions are inverse-transformed back to original units before returning,
  so the external interface (y_pred, y_lower, y_upper, mape, calibration) is
  identical to transformer_model.py.

All other architecture / training decisions are unchanged so results are
directly comparable to the v1 baseline.

Usage:
    python models/transformer_v2.py
    python models/transformer_v2.py --epochs 200 --embed-dim 128
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

TARGETS = [
    "label_inventory_delta",
    "label_cashflow_delta",
    "label_order_velocity_delta",
]
DROP_COLS = [
    "branch_id", "business_id", "category", "branch_type", "committed_at",
] + TARGETS

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.15
CI_LOWER_Q  = 0.10
CI_UPPER_Q  = 0.90


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def pinball_loss(pred, target, alpha):
    diff = target - pred
    return torch.where(diff >= 0, alpha * diff, (alpha - 1) * diff).mean()


def combined_loss(outputs, y, lower_alpha=CI_LOWER_Q, upper_alpha=CI_UPPER_Q):
    return (
        F.mse_loss(outputs[:, 0], y)
        + pinball_loss(outputs[:, 1], y, lower_alpha)
        + pinball_loss(outputs[:, 2], y, upper_alpha)
    )


# ── Architecture (unchanged from v1) ─────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, embed_dim))
        self.bias   = nn.Parameter(torch.zeros(n_features, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features, embed_dim=64, n_heads=4, n_layers=3, ffn_dim=128, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0]))


# ── Training ──────────────────────────────────────────────────────────────────

def _tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_one_target(
    X_train_raw, y_train_raw, X_test_raw, y_test_raw,
    device, embed_dim, n_heads, n_layers, ffn_dim, dropout,
    lr, batch_size, epochs, patience, target_name,
):
    val_cut  = int(len(X_train_raw) * (1 - VAL_RATIO))
    X_tr, y_tr   = X_train_raw.iloc[:val_cut],  y_train_raw.iloc[:val_cut]
    X_val, y_val = X_train_raw.iloc[val_cut:],  y_train_raw.iloc[val_cut:]

    # ── Feature scaling (same as v1)
    feat_scaler = StandardScaler()
    X_tr_s  = feat_scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = feat_scaler.transform(X_val).astype(np.float32)
    X_te_s  = feat_scaler.transform(X_test_raw).astype(np.float32)

    # ── Target scaling (NEW in v2) ─────────────────────────────────────────────
    # Normalise y to zero mean / unit variance so MSE and pinball loss terms
    # operate at the same magnitude across all three targets.
    y_scaler     = StandardScaler()
    y_tr_scaled  = y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel().astype(np.float32)

    tr_dataset = TensorDataset(_tensor(X_tr_s, device), _tensor(y_tr_scaled, device))
    tr_loader  = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    X_val_t    = _tensor(X_val_s, device)
    y_val_t    = _tensor(y_val_scaled, device)
    X_te_t     = _tensor(X_te_s, device)

    model = FTTransformer(
        n_features=X_tr_s.shape[1], embed_dim=embed_dim, n_heads=n_heads,
        n_layers=n_layers, ffn_dim=ffn_dim, dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mape, best_state, epochs_no_improve = float("inf"), None, 0
    short = target_name.replace("label_", "").replace("_delta", "")
    print(f"\nTraining: {short}  ({X_tr_s.shape[1]} features, device={device})")

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            loss = combined_loss(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # Inverse-transform val predictions to get MAPE in original units
            val_out_scaled = model(X_val_t)[:, 0].cpu().numpy()
            val_point = y_scaler.inverse_transform(val_out_scaled.reshape(-1, 1)).ravel()
        val_m = mape(y_val.values, val_point)

        if val_m < best_val_mape:
            best_val_mape = val_m
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  val_mape={val_m:.1f}%  best={best_val_mape:.1f}%")

        if epochs_no_improve >= patience:
            print(f"  early stop at epoch {epoch}  (patience={patience})")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        out_scaled = model(X_te_t).cpu().numpy()

    # Inverse-transform all three output columns back to original units
    y_pred  = y_scaler.inverse_transform(out_scaled[:, 0:1]).ravel()
    y_lower = y_scaler.inverse_transform(out_scaled[:, 1:2]).ravel()
    y_upper = y_scaler.inverse_transform(out_scaled[:, 2:3]).ravel()

    return {
        "model":          model,
        "feat_scaler":    feat_scaler,
        "y_scaler":       y_scaler,
        "feature_cols":   list(X_train_raw.columns),
        "y_test":         y_test_raw.values,
        "y_pred":         y_pred,
        "y_lower":        y_lower,
        "y_upper":        y_upper,
        "mape":           mape(y_test_raw.values, y_pred),
        "calibration":    calibration_score(y_test_raw.values, y_lower, y_upper),
        "best_val_mape":  best_val_mape,
    }


def train_transformer(
    train, test,
    embed_dim=64, n_heads=4, n_layers=3, ffn_dim=128, dropout=0.1,
    lr=1e-3, batch_size=64, epochs=150, patience=20,
):
    device  = get_device()
    results = {}
    for target in TARGETS:
        X_train_raw, y_train_raw = get_X_y(train, target)
        X_test_raw,  y_test_raw  = get_X_y(test,  target)
        results[target] = train_one_target(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            device=device, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            ffn_dim=ffn_dim, dropout=dropout, lr=lr, batch_size=batch_size,
            epochs=epochs, patience=patience, target_name=target,
        )
    return results


def print_results(results):
    labels = {
        "label_inventory_delta":      "Inventory Delta     ",
        "label_cashflow_delta":       "Cash Flow Delta     ",
        "label_order_velocity_delta": "Order Velocity Delta",
    }
    header = f"\n{'Target':<24}  {'MAPE':>8}  {'Calibration (80% CI)':>22}  {'Val MAPE':>10}"
    print(header)
    print("─" * len(header))
    for target, r in results.items():
        mape_s   = f"{r['mape']:.1f}%"         if not np.isnan(r['mape'])         else "n/a"
        val_mape = f"{r['best_val_mape']:.1f}%" if not np.isnan(r['best_val_mape']) else "n/a"
        cal_s    = f"{r['calibration']:.2f}  (target: 0.80)"
        print(f"  {labels[target]}  {mape_s:>8}  {cal_s:>22}  {val_mape:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",    type=str,   default="data/processed/training_data.csv")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--embed-dim",   type=int,   default=64)
    parser.add_argument("--n-heads",     type=int,   default=4)
    parser.add_argument("--n-layers",    type=int,   default=3)
    parser.add_argument("--ffn-dim",     type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--patience",    type=int,   default=20)
    args = parser.parse_args()

    df = load_features(args.features)
    train, test = temporal_split(df, args.train_ratio)
    results = train_transformer(
        train, test, embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_dim=args.ffn_dim, dropout=args.dropout,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
    )
    print_results(results)


if __name__ == "__main__":
    main()
