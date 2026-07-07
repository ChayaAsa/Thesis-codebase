from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ── Feature mode  —  uncomment ONE block ─────────────────────────────────────

# MODE A: raw motor signals only  (9 features)
# INPUT_COLS = [
#     "tau_meas1_Nm",  "tau_meas2_Nm",  "tau_meas3_Nm",
#     "q1_rad",        "q2_rad",        "q3_rad",
#     "qdot1_rad_s",   "qdot2_rad_s",   "qdot3_rad_s",
# ]
# FEAT_MODE = "raw"

# # MODE B: raw + physics-model pre-compensation  (15 features)
# INPUT_COLS = [
#     "tau_meas1_Nm",  "tau_meas2_Nm",  "tau_meas3_Nm",
#     "q1_rad",        "q2_rad",        "q3_rad",
#     "qdot1_rad_s",   "qdot2_rad_s",   "qdot3_rad_s",
#     "G1_Nm",         "G2_Nm",         "G3_Nm",
#     "Cqdot1_Nm",     "Cqdot2_Nm",     "Cqdot3_Nm",
# ]
# FEAT_MODE = "phys"

# # MODE C: physics residual + pose  (9 features)
# tau_res_i = tau_meas_i − G_i − Cqdot_i  ≈ J^T @ F_contact  at slow motion
# INPUT_COLS = [
#     "tau_res1_Nm",   "tau_res2_Nm",   "tau_res3_Nm",
#     "q1_rad",        "q2_rad",        "q3_rad",
#     "qdot1_rad_s",   "qdot2_rad_s",   "qdot3_rad_s",
# ]
# FEAT_MODE = "res"

# MODE D: physics residual only  (3 features)  ← best for single fixed pose
# At INIT_Q pressed against wall: q ≈ const, qdot ≈ 0 → near-zero SNR after
# z-score. Removing them forces the LSTM to rely on tau_res, which is
# already ≈ J^T @ F_contact and nearly sufficient by itself.
INPUT_COLS = [
    "tau_res1_Nm",   "tau_res2_Nm",   "tau_res3_Nm",
]
FEAT_MODE = "res3"

# ── Target mode  —  uncomment ONE block ──────────────────────────────────────

# 1-D: push-axis scalar (−ATI_z), direct from CSV, no frame rotation needed
TARGET_COLS = ["f_push_N"]
TARGET_MODE = "1d"

# # 3-D: full force vector in ATI sensor frame
# # (proper use requires F_base logged in data; raw ATI frame is an approximation)
# TARGET_COLS = ["ati_fx_N", "ati_fy_N", "ati_fz_N"]
# TARGET_MODE = "3d"

# Paths

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(HERE, '..', '1_data'))
MODEL_DIR  = os.path.normpath(os.path.join(HERE, '..', '3_model'))
RESULT_DIR = os.path.normpath(os.path.join(HERE, '..', '2_train'))

MODE_TAG   = f"{FEAT_MODE}_{TARGET_MODE}"     # e.g. "raw_1d", "phys_3d"
SAVE_PATH  = os.path.join(MODEL_DIR,  f"lstm_{MODE_TAG}.pt")
CURVE_PATH = os.path.join(RESULT_DIR, f"train_curve_{MODE_TAG}.csv")
PLOT_PATH  = os.path.join(RESULT_DIR, f"train_result_{MODE_TAG}.png")

# Hyperparameters

LOOP_HZ    = 100
WINDOW_S   = 1.0          # context window length [s]
WINDOW_LEN = int(WINDOW_S * LOOP_HZ)   # 150 steps at 100 Hz
STRIDE     = 1            # sliding-window hop (1 = full overlap)
TRAIN_FRAC = 0.80         # per-file chronological split

LSTM_HIDDEN = 64
LSTM_LAYERS = 2
FC_HIDDEN   = 32
DROPOUT     = 0.2

LR         = 1e-3
WEIGHT_DEC = 1e-5
BATCH      = 256
MAX_EPOCHS = 500
PATIENCE   = 50           # early-stop patience on val loss
LR_PATIENCE= 20           # ReduceLROnPlateau patience
LR_FACTOR  = 0.5
SEED       = 42

# Optional: weight contact samples more heavily during training.
# Set > 1.0 to bias learning toward non-zero force; 1.0 = uniform.
CONTACT_WEIGHT = 3.0
USE_CONTACT_WEIGHT = False   # flip to True to enable

# Optional: filter rows by recording phase.
# None = use all phases;  'A' = free-space only;  'B' = contact only
PHASE_FILTER: str | None = None

# Derived constants

N_IN  = len(INPUT_COLS)
N_OUT = len(TARGET_COLS)    # 1 for 1-D mode, 3 for 3-D mode


# Data loading

def load_files() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print(f"Loading {len(files)} file(s) from {DATA_DIR}")
    result = []

    for fpath in files:
        with open(fpath, newline='') as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print(f"  SKIP (empty): {os.path.basename(fpath)}")
            continue

        # Optional phase filter
        if PHASE_FILTER is not None:
            rows = [r for r in rows if r.get('phase', '') == PHASE_FILTER]
            if not rows:
                print(f"  SKIP (no phase {PHASE_FILTER!r} rows): {os.path.basename(fpath)}")
                continue

        # Derive tau_res_i = tau_meas_i − G_i − Cqdot_i (computed, not in CSV)
        if any('tau_res' in c for c in INPUT_COLS):
            for row in rows:
                for j in (1, 2, 3):
                    row[f'tau_res{j}_Nm'] = str(
                        float(row[f'tau_meas{j}_Nm'])
                        - float(row[f'G{j}_Nm'])
                        - float(row[f'Cqdot{j}_Nm'])
                    )

        # Validate columns exist
        missing_feat = [c for c in INPUT_COLS  if c not in rows[0]]
        missing_tgt  = [c for c in TARGET_COLS if c not in rows[0]]
        if missing_feat or missing_tgt:
            print(f"  SKIP (missing cols {missing_feat + missing_tgt}): {os.path.basename(fpath)}")
            continue

        X = np.array([[float(r[c]) for c in INPUT_COLS]  for r in rows], dtype=np.float32)
        y = np.array([[float(r[c]) for c in TARGET_COLS] for r in rows], dtype=np.float32)

        # Per-sample weights
        if USE_CONTACT_WEIGHT and 'contact_flag' in rows[0]:
            flags = np.array([float(r['contact_flag']) for r in rows], dtype=np.float32)
            w = np.where(flags > 0, CONTACT_WEIGHT, 1.0).astype(np.float32)
        else:
            w = np.ones(len(rows), dtype=np.float32)

        print(f"  {os.path.basename(fpath):45s}  {len(rows):6d} rows")
        result.append((X, y, w))

    if not result:
        raise ValueError("No usable CSV files after filtering.")
    return result


# Sliding-window builder

def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = range(0, len(X) - window + 1, stride)
    X_win  = np.stack([X[s : s + window] for s in starts])          # (W, T, F)
    y_win  = np.array([y[s + window - 1] for s in starts])          # (W, n_out)
    w_win  = np.array([w[s + window - 1] for s in starts])          # (W,)
    return X_win, y_win, w_win


# Normalizer

class Normalizer:

    mean_X: np.ndarray   # (n_feat,)
    std_X:  np.ndarray   # (n_feat,)
    mean_y: np.ndarray   # (n_out,)
    std_y:  np.ndarray   # (n_out,)

    def fit(self, X_win: np.ndarray, y_win: np.ndarray) -> None:
        flat        = X_win.reshape(-1, X_win.shape[-1])
        self.mean_X = flat.mean(axis=0).astype(np.float32)
        self.std_X  = (flat.std(axis=0)  + 1e-8).astype(np.float32)
        self.mean_y = y_win.mean(axis=0).astype(np.float32)
        self.std_y  = (y_win.std(axis=0) + 1e-8).astype(np.float32)

    def nx(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_X) / self.std_X

    def ny(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean_y) / self.std_y

    def iy(self, y: np.ndarray) -> np.ndarray:
        return y * self.std_y + self.mean_y

    def as_dict(self) -> dict:
        return {
            "mean_X": self.mean_X.tolist(),
            "std_X":  self.std_X.tolist(),
            "mean_y": self.mean_y.tolist(),
            "std_y":  self.std_y.tolist(),
        }


# Model

class ForceLSTM(nn.Module):

    def __init__(self, n_in: int, n_out: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = n_in,
            hidden_size  = hidden,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, FC_HIDDEN),
            nn.Tanh(),
            nn.Linear(FC_HIDDEN, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])    # last timestep → head


# Training helpers

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
) -> float:
    model.train() if train else model.eval()
    total = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            xb, yb, wb = [t.to(device) for t in batch]
            pred = model(xb)                             # (B, n_out)

            if USE_CONTACT_WEIGHT:
                losses = ((pred - yb) ** 2).mean(dim=-1) # (B,)
                loss   = (losses * wb).mean()
            else:
                loss = criterion(pred, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * xb.size(0)

    return total / len(loader.dataset)


def make_loader(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
    norm: Normalizer, shuffle: bool,
) -> DataLoader:
    Xn = torch.from_numpy(norm.nx(X))
    yn = torch.from_numpy(norm.ny(y))
    wt = torch.from_numpy(w)
    return DataLoader(
        TensorDataset(Xn, yn, wt),
        batch_size=BATCH,
        shuffle=shuffle,
    )


# Main

def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  3-DOF LSTM training   mode={MODE_TAG}")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    print(f"  Features : {N_IN}  {INPUT_COLS}")
    print(f"  Targets  : {N_OUT}  {TARGET_COLS}")
    print(f"  Window   : {WINDOW_LEN} steps = {WINDOW_S:.1f}s @ {LOOP_HZ} Hz")
    print(f"  Stride   : {STRIDE}")
    print(f"  Contact weight : {'ON  x' + str(CONTACT_WEIGHT) if USE_CONTACT_WEIGHT else 'off'}")
    print(f"  Phase filter   : {PHASE_FILTER or 'all'}\n")

    # Load and window data
    files_data = load_files()
    tr_X, tr_y, tr_w = [], [], []
    va_X, va_y, va_w = [], [], []

    print(f"\nPer-file chronological split  "
          f"({TRAIN_FRAC:.0%} train / {1-TRAIN_FRAC:.0%} val):")

    for i, (X_raw, y_raw, w_raw) in enumerate(files_data):
        X_win, y_win, w_win = make_windows(X_raw, y_raw, w_raw, WINDOW_LEN, STRIDE)
        n_tr = int(len(X_win) * TRAIN_FRAC)

        tr_X.append(X_win[:n_tr]);   tr_y.append(y_win[:n_tr]);   tr_w.append(w_win[:n_tr])
        va_X.append(X_win[n_tr:]);   va_y.append(y_win[n_tr:]);   va_w.append(w_win[n_tr:])
        print(f"  file {i+1:2d}: {len(X_win):6d} windows → "
              f"train {n_tr:5d}  val {len(X_win)-n_tr:5d}")

    X_tr = np.concatenate(tr_X);  y_tr = np.concatenate(tr_y);  w_tr = np.concatenate(tr_w)
    X_va = np.concatenate(va_X);  y_va = np.concatenate(va_y);  w_va = np.concatenate(va_w)
    print(f"  merged   : train {len(X_tr):6d}  val {len(X_va):6d}\n")

    # Normalise (fit on train only)
    norm = Normalizer()
    norm.fit(X_tr, y_tr)
    tr_loader = make_loader(X_tr, y_tr, w_tr, norm, shuffle=True)
    va_loader = make_loader(X_va, y_va, w_va, norm, shuffle=False)

    # Model
    model = ForceLSTM(N_IN, N_OUT, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Parameters : {n_params:,}\n")

    # Training loop
    best_val, best_state, wait = float('inf'), None, 0
    tr_losses, va_losses = [], []
    curve_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, va_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]['lr']

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        curve_rows.append({'epoch': epoch, 'train_loss': tr_loss,
                           'val_loss': va_loss, 'lr': lr_now})

        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train {tr_loss:.6f}  val {va_loss:.6f}"
                  f"  lr {lr_now:.2e}  {'✓' if improved else f'wait {wait}'}")

        if wait >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  (best val {best_val:.6f})")
            break

    # Save training curve
    with open(CURVE_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr'])
        w.writeheader()
        w.writerows(curve_rows)
    print(f"\n  Curve  → {CURVE_PATH}")

    # Validation evaluation (denormalised)
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb, _ in va_loader:
            p = model(xb.to(device)).cpu().numpy()   # (B, n_out)
            preds.append(norm.iy(p))
            trues.append(norm.iy(yb.numpy()))

    y_pred = np.concatenate(preds)   # (N_val, n_out)
    y_true = np.concatenate(trues)

    print(f"\n  Validation metrics  (denormalised) [{MODE_TAG}]:")
    for i, col in enumerate(TARGET_COLS):
        err  = y_pred[:, i] - y_true[:, i]
        mae  = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        r2   = 1.0 - (err**2).sum() / ((y_true[:, i] - y_true[:, i].mean())**2 + 1e-12).sum()
        print(f"    {col:20s}  MAE={mae:.4f} N  RMSE={rmse:.4f} N  R²={r2:.4f}")

    # Plots
    _make_plots(tr_losses, va_losses, y_true, y_pred)

    # Prompt to save
    ans = input("\n  Press Enter to save model, q to cancel: ").strip().lower()
    if ans == 'q':
        print("  Save cancelled.")
        return

    _save_model(model, best_state, norm)


# Save model

def _save_model(model: ForceLSTM, best_state: dict, norm: Normalizer) -> None:
    payload = {
        "model_state": best_state,
        "n_in":        N_IN,
        "n_out":       N_OUT,
        "hidden":      LSTM_HIDDEN,
        "n_layers":    LSTM_LAYERS,
        "fc_hidden":   FC_HIDDEN,
        "dropout":     DROPOUT,
        # windowing (needed for ring buffer in run script)
        "window_len":  WINDOW_LEN,
        "window_s":    WINDOW_S,
        "loop_hz":     LOOP_HZ,
        # column names (needed so run script assembles features in correct order)
        "input_cols":  INPUT_COLS,
        "target_cols": TARGET_COLS,
        "feat_mode":   FEAT_MODE,
        "target_mode": TARGET_MODE,
        # normalisation stats (must travel with the weights)
        "norm":        norm.as_dict(),
    }
    torch.save(payload, SAVE_PATH)
    print(f"  Model  → {SAVE_PATH}")

    # Also write a human-readable JSON sidecar
    json_path = SAVE_PATH.replace('.pt', '_config.json')
    config = {k: v for k, v in payload.items() if k != 'model_state'}
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config → {json_path}")


# Plots

def _make_plots(
    tr_losses: list[float],
    va_losses: list[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    n_cols = N_OUT + 1          # loss curve + one scatter per output
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    fig.suptitle(f'LSTM training  [{MODE_TAG}]  window={WINDOW_LEN} steps', fontsize=11)

    # Loss curve
    ax = axes[0]
    ax.plot(tr_losses, label='train')
    ax.plot(va_losses, label='val')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (normalised)')
    ax.set_title('Loss curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter per output axis
    for i, col in enumerate(TARGET_COLS):
        ax  = axes[i + 1]
        yt  = y_true[:, i]
        yp  = y_pred[:, i]
        err = yp - yt
        mae  = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        r2   = 1.0 - (err**2).sum() / ((yt - yt.mean())**2 + 1e-12).sum()

        lim = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
        ax.scatter(yt, yp, s=3, alpha=0.3)
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_xlim(lim);  ax.set_ylim(lim)
        ax.set_xlabel(f'True {col} [N]')
        ax.set_ylabel(f'Predicted [N]')
        ax.set_title(f'{col}\nMAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"  Plot   → {PLOT_PATH}")


if __name__ == '__main__':
    main()
