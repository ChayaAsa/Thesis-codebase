from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Model selection — uncomment ONE block ────────────────────────────────────

MODEL_TAG = "phys_1d"
# MODEL_TAG = "raw_1d"
# MODEL_TAG = "phys_3d"
# MODEL_TAG = "raw_3d"
# MODEL_TAG = "res_1d"
# MODEL_TAG = "res3_1d"
# Evaluation CSV

# Set to a string path for a specific file, or None to use most-recent CSV.
EVAL_CSV: str | None = None
# Paths

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.normpath(os.path.join(HERE, '..', '3_model'))
DATA_DIR   = os.path.normpath(os.path.join(HERE, '..', '1_data'))
RESULT_DIR = HERE
BATCH      = 512
MAE_GATE   = 3.0      # N


# Model (must match train script)

class ForceLSTM(nn.Module):
    def __init__(self, n_in, n_out, hidden=64, n_layers=2, fc_hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# Helpers

def load_model(tag: str) -> tuple[ForceLSTM, dict, torch.device]:
    path   = os.path.join(MODEL_DIR, f"lstm_{tag}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(path, map_location=device, weights_only=False)

    model = ForceLSTM(
        n_in      = bundle["n_in"],
        n_out     = bundle["n_out"],
        hidden    = bundle["hidden"],
        n_layers  = bundle["n_layers"],
        fc_hidden = bundle["fc_hidden"],
        dropout   = bundle["dropout"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle, device


def pick_csv() -> str:
    if EVAL_CSV is not None:
        if not os.path.exists(EVAL_CSV):
            raise FileNotFoundError(f"EVAL_CSV not found: {EVAL_CSV}")
        return EVAL_CSV
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    chosen = csvs[-1]          # most recent
    print(f"  [auto] using most recent CSV: {os.path.basename(chosen)}")
    print("         Set EVAL_CSV = '<path>' to choose a different file.\n")
    return chosen


def load_csv(path: str, input_cols: list[str], target_cols: list[str]):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    # Derive tau_res_i = tau_meas_i − G_i − Cqdot_i if the model needs it
    if any('tau_res' in c for c in input_cols):
        for row in rows:
            for j in (1, 2, 3):
                row[f'tau_res{j}_Nm'] = str(
                    float(row[f'tau_meas{j}_Nm'])
                    - float(row[f'G{j}_Nm'])
                    - float(row[f'Cqdot{j}_Nm'])
                )

    missing = [c for c in input_cols + target_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    t    = np.array([float(r['time_s'])         for r in rows], dtype=np.float32)
    X    = np.array([[float(r[c]) for c in input_cols]  for r in rows], dtype=np.float32)
    y    = np.array([[float(r[c]) for c in target_cols] for r in rows], dtype=np.float32)

    # Optional: contact flag for shading
    cf = None
    if 'contact_flag' in rows[0]:
        cf = np.array([float(r['contact_flag']) for r in rows], dtype=np.float32)

    return t, X, y, cf


def run_inference(
    model:  ForceLSTM,
    bundle: dict,
    X_raw:  np.ndarray,      # (N, n_in)
    device: torch.device,
) -> np.ndarray:
    norm       = bundle["norm"]
    mean_X     = np.array(norm["mean_X"], dtype=np.float32)
    std_X      = np.array(norm["std_X"],  dtype=np.float32)
    mean_y     = np.array(norm["mean_y"], dtype=np.float32)
    std_y      = np.array(norm["std_y"],  dtype=np.float32)
    window_len = bundle["window_len"]
    n_out      = bundle["n_out"]

    X_norm = (X_raw - mean_X) / std_X
    N = len(X_norm)
    preds = np.full((N, n_out), np.nan, dtype=np.float32)

    # Build windows
    starts = np.arange(0, N - window_len + 1)
    if len(starts) == 0:
        print("  [warn] CSV shorter than window_len — no predictions.")
        return preds

    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            idx   = starts[i : i + BATCH]
            wins  = np.stack([X_norm[s : s + window_len] for s in idx])   # (B, T, F)
            xb    = torch.from_numpy(wins).to(device)
            out_n = model(xb).cpu().numpy()                                 # (B, n_out)
            out   = out_n * std_y + mean_y
            for j, s in enumerate(idx):
                preds[s + window_len - 1] = out[j]

    return preds


def cross_corr_lag(pred: np.ndarray, true: np.ndarray, loop_hz: float) -> float:
    p = pred - pred.mean()
    a = true - true.mean()
    corr = np.correlate(p, a, mode='full')
    lag  = np.argmax(corr) - (len(p) - 1)
    return lag * 1000.0 / loop_hz


def compute_metrics(
    y_true:   np.ndarray,   # (N, n_out)
    y_pred:   np.ndarray,   # (N, n_out)
    cols:     list[str],
    loop_hz:  float,
) -> dict:
    valid = ~np.isnan(y_pred[:, 0])
    yt = y_true[valid]
    yp = y_pred[valid]

    metrics = {}
    for i, col in enumerate(cols):
        err  = yp[:, i] - yt[:, i]
        mae  = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        r2   = float(1.0 - (err**2).sum() / ((yt[:,i]-yt[:,i].mean())**2+1e-12).sum())
        lag  = float(cross_corr_lag(yp[:, i], yt[:, i], loop_hz))
        metrics[col] = {"mae_N": mae, "rmse_N": rmse, "r2": r2, "lag_ms": lag,
                        "max_err_N": float(np.abs(err).max())}
    return metrics


# Plots

def _shade_contact(ax, t: np.ndarray, cf: np.ndarray | None) -> None:
    if cf is None:
        return
    in_contact = False
    t0 = 0.0
    for i, flag in enumerate(cf):
        if flag > 0 and not in_contact:
            t0 = t[i]; in_contact = True
        elif flag == 0 and in_contact:
            ax.axvspan(t0, t[i], alpha=0.08, color='steelblue', lw=0)
            in_contact = False
    if in_contact:
        ax.axvspan(t0, t[-1], alpha=0.08, color='steelblue', lw=0)


def plot_timeseries(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: list[str],
    cf: np.ndarray | None,
    tag: str,
    metrics: dict,
) -> None:
    n_out = len(cols)
    fig, axes = plt.subplots(n_out, 1, figsize=(12, 3 * n_out), sharex=True)
    if n_out == 1:
        axes = [axes]
    fig.suptitle(f'LSTM vs ATI  [{tag}]', fontsize=11)

    for i, (ax, col) in enumerate(zip(axes, cols)):
        _shade_contact(ax, t, cf)
        ax.plot(t, y_true[:, i], color='steelblue', lw=1.0, alpha=0.8, label='ATI (true)')
        ax.plot(t, y_pred[:, i], color='firebrick',  lw=1.0, alpha=0.9, label='LSTM (pred)')
        m = metrics[col]
        ax.set_ylabel(f'{col} [N]')
        ax.set_title(
            f'{col}   MAE={m["mae_N"]:.3f} N   RMSE={m["rmse_N"]:.3f} N'
            f'   R²={m["r2"]:.3f}   lag={m["lag_ms"]:.1f} ms',
            fontsize=9
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f'ts_{tag}.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ts         → {path}")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: list[str],
    tag: str,
    metrics: dict,
) -> None:
    valid = ~np.isnan(y_pred[:, 0])
    yt = y_true[valid]
    yp = y_pred[valid]

    n_out = len(cols)
    fig, axes = plt.subplots(1, n_out, figsize=(5 * n_out, 4))
    if n_out == 1:
        axes = [axes]
    fig.suptitle(f'Predicted vs True  [{tag}]', fontsize=11)

    for i, (ax, col) in enumerate(zip(axes, cols)):
        m    = metrics[col]
        lim  = [min(yt[:,i].min(), yp[:,i].min()) - 1,
                max(yt[:,i].max(), yp[:,i].max()) + 1]
        ax.scatter(yt[:, i], yp[:, i], s=3, alpha=0.3, color='steelblue')
        ax.plot(lim, lim, 'r--', lw=1.2, label='ideal')
        ax.set_xlim(lim);  ax.set_ylim(lim)
        ax.set_xlabel(f'True {col} [N]')
        ax.set_ylabel(f'Predicted [N]')
        ax.set_title(f'{col}\nMAE={m["mae_N"]:.3f}  R²={m["r2"]:.3f}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f'scatter_{tag}.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  scatter    → {path}")


def plot_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: list[str],
    tag: str,
    metrics: dict,
) -> None:
    valid = ~np.isnan(y_pred[:, 0])
    err   = y_pred[valid] - y_true[valid]

    n_out = len(cols)
    fig, axes = plt.subplots(1, n_out, figsize=(5 * n_out, 4))
    if n_out == 1:
        axes = [axes]
    fig.suptitle(f'Error distribution  [{tag}]', fontsize=11)

    for i, (ax, col) in enumerate(zip(axes, cols)):
        m = metrics[col]
        ax.hist(err[:, i], bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.axvline(0,  color='black', lw=1.2, linestyle='--')
        ax.axvline( m["mae_N"], color='firebrick', lw=1.0, linestyle=':', label=f'+MAE={m["mae_N"]:.3f}')
        ax.axvline(-m["mae_N"], color='firebrick', lw=1.0, linestyle=':')
        ax.set_xlabel(f'Error [N]  ({col})')
        ax.set_ylabel('Count')
        ax.set_title(f'{col}\nRMSE={m["rmse_N"]:.3f} N   max|e|={m["max_err_N"]:.2f} N', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f'hist_{tag}.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  histogram  → {path}")


# Gate check

def gate_check(metrics: dict) -> bool:
    passed = True
    for col, m in metrics.items():
        if m["mae_N"] > MAE_GATE:
            passed = False
            print(f"\n{'!'*60}")
            print(f"  GATE FAIL: {col}  MAE={m['mae_N']:.3f} N > {MAE_GATE} N")
            print(f"  Do NOT proceed to Step 5 (real-time controller).")
            print(f"  Suggested actions:")
            print(f"    1. Collect more Phase B data (longer DURATION or more sessions)")
            print(f"    2. Try physics-assisted features (FEAT_MODE='phys')")
            print(f"    3. Increase window_len (more context)")
            print(f"    4. Enable USE_CONTACT_WEIGHT = True in train script")
            print(f"{'!'*60}\n")
    if passed:
        print(f"\n  ✓ Gate passed: all MAE < {MAE_GATE} N  →  safe to proceed to Step 5.\n")
    return passed


# Main

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  LSTM eval   tag={MODEL_TAG}")
    print(f"{'='*60}\n")

    # Load model
    model, bundle, device = load_model(MODEL_TAG)
    input_cols  = bundle["input_cols"]
    target_cols = bundle["target_cols"]
    window_len  = bundle["window_len"]
    loop_hz     = bundle["loop_hz"]
    print(f"  Model     : {os.path.join(MODEL_DIR, f'lstm_{MODEL_TAG}.pt')}")
    print(f"  Device    : {device}")
    print(f"  Features  : {len(input_cols)}")
    print(f"  Window    : {window_len} steps = {window_len/loop_hz:.1f}s @ {loop_hz} Hz")
    print(f"  Targets   : {target_cols}\n")

    # Load CSV
    csv_path = pick_csv()
    t, X, y_true, cf = load_csv(csv_path, input_cols, target_cols)
    print(f"  CSV       : {os.path.basename(csv_path)}")
    print(f"  Rows      : {len(t)}   duration={t[-1]:.1f}s\n")

    if len(t) < window_len + 10:
        raise ValueError(f"CSV too short: {len(t)} rows < window_len ({window_len})")

    # Inference
    y_pred = run_inference(model, bundle, X, device)
    valid  = ~np.isnan(y_pred[:, 0])
    n_valid = valid.sum()
    print(f"  Predicted : {n_valid} windows  "
          f"(warm-up={window_len-1} rows, {(window_len-1)/loop_hz:.1f}s)\n")

    # Metrics
    metrics = compute_metrics(y_true, y_pred, target_cols, loop_hz)

    print(f"  {'Column':<20}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  {'lag ms':>8}  {'max|e|':>8}")
    print(f"  {'-'*68}")
    for col, m in metrics.items():
        print(f"  {col:<20}  {m['mae_N']:8.4f}  {m['rmse_N']:8.4f}  "
              f"{m['r2']:7.4f}  {m['lag_ms']:8.1f}  {m['max_err_N']:8.3f}")

    # Save metrics JSON
    metrics_path = os.path.join(RESULT_DIR, f'metrics_{MODEL_TAG}.json')
    with open(metrics_path, 'w') as f:
        json.dump({"model_tag": MODEL_TAG,
                   "eval_csv": os.path.basename(csv_path),
                   "n_rows": int(len(t)),
                   "n_valid": int(n_valid),
                   "window_len": window_len,
                   "metrics": metrics}, f, indent=2)
    print(f"\n  metrics    → {metrics_path}")

    # Gate
    gate_check(metrics)

    # Plots
    plot_timeseries(t, y_true, y_pred, target_cols, cf, MODEL_TAG, metrics)
    plot_scatter(y_true, y_pred, target_cols, MODEL_TAG, metrics)
    plot_histogram(y_true, y_pred, target_cols, MODEL_TAG, metrics)

    print(f"\n  Done.  All figures saved to {RESULT_DIR}")


if __name__ == '__main__':
    main()
