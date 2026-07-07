import csv, os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# paths
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', '1_data')
SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', '3_model', '3-2_LSTM_model.pt')
PLOT_DIR  = os.path.join(os.path.dirname(__file__), 'pic')

# columns
INPUT_COLS = ["tau_meas_Nm", "pos_rad", "vel_rad_s"]
TARGET_COL = "f_raw_N"

# window / sampling
WINDOW_S    = 3    # [s]
LOOP_HZ     = 100    # [Hz]
STRIDE      = 1
WINDOW_SIZE = max(1, int(WINDOW_S * LOOP_HZ))

# split
TRAIN_FRAC = 0.80    # val = 1 - TRAIN_FRAC (per file, chronological)

# model hyper-params
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
FC_HIDDEN   = 32
DROPOUT     = 0.1

LR         = 1e-3
BATCH      = 128
MAX_EPOCHS = 300
PATIENCE   = 50
SEED       = 42


# ─── data loading — one (X, y) pair per file ─────────────────────────────────
def load_files() -> list[tuple[np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    assert files, f"No CSV files in {DATA_DIR}"
    print(f"Loading {len(files)} file(s):")
    result = []
    for fpath in files:
        rows = []
        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        X = np.array([[float(r[c]) for c in INPUT_COLS] for r in rows], dtype=np.float32)
        y = np.array([float(r[TARGET_COL]) for r in rows], dtype=np.float32)
        print(f"  {os.path.basename(fpath):40s}  {len(rows)} rows")
        result.append((X, y))
    return result


# sliding window
def make_windows(X, y, window, stride):
    starts = range(0, len(X) - window + 1, stride)
    X_win  = np.stack([X[s:s + window] for s in starts])
    y_win  = np.array([y[s + window - 1] for s in starts])
    return X_win, y_win


# normaliser
class Normalizer:
    def fit(self, X_win, y_win):
        flat        = X_win.reshape(-1, X_win.shape[-1])
        self.mean_X = flat.mean(axis=0).astype(np.float32)
        self.std_X  = (flat.std(axis=0) + 1e-8).astype(np.float32)
        self.mean_y = float(y_win.mean())
        self.std_y  = float(y_win.std() + 1e-8)

    def nx(self, X): return (X - self.mean_X) / self.std_X
    def ny(self, y): return (y - self.mean_y) / self.std_y
    def iy(self, y): return y * self.std_y + self.mean_y


# model
class TauNetLSTM(nn.Module):
    def __init__(self, n_in, lstm_hidden, lstm_layers, fc_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden), nn.Tanh(), nn.Linear(fc_hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# epoch helper
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total = 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1);  loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad();  loss.backward();  optimizer.step()
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


# main
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Window  : {WINDOW_SIZE} steps  ({WINDOW_S*1000:.0f} ms at {LOOP_HZ} Hz)")
    print(f"Stride  : {STRIDE}")
    print(f"Inputs  : {INPUT_COLS}\n")

    # per-file windowing + chronological split
    files_data = load_files()
    tr_X, tr_y, va_X, va_y = [], [], [], []

    print(f"\nPer-file chronological split  ({TRAIN_FRAC:.0%} train / {1-TRAIN_FRAC:.0%} val):")
    for i, (X_raw, y_raw) in enumerate(files_data):
        X_win, y_win = make_windows(X_raw, y_raw, WINDOW_SIZE, STRIDE)
        n_tr = int(len(X_win) * TRAIN_FRAC)
        tr_X.append(X_win[:n_tr]);   tr_y.append(y_win[:n_tr])
        va_X.append(X_win[n_tr:]);   va_y.append(y_win[n_tr:])
        print(f"  file {i+1}: {len(X_win)} windows  →  train {n_tr}  val {len(X_win)-n_tr}")

    X_tr = np.concatenate(tr_X);  y_tr = np.concatenate(tr_y)
    X_va = np.concatenate(va_X);  y_va = np.concatenate(va_y)
    print(f"  merged : train {len(X_tr)}  val {len(X_va)}\n")

    # normalise (fit on train only)
    norm = Normalizer()
    norm.fit(X_tr, y_tr)

    def to_loader(X, y, shuffle):
        Xn = torch.from_numpy(norm.nx(X))
        yn = torch.from_numpy(norm.ny(y).astype(np.float32))
        return DataLoader(TensorDataset(Xn, yn), batch_size=BATCH, shuffle=shuffle)

    tr_loader = to_loader(X_tr, y_tr, shuffle=True)
    va_loader = to_loader(X_va, y_va, shuffle=False)

    # model
    model = TauNetLSTM(len(INPUT_COLS), LSTM_HIDDEN, LSTM_LAYERS,
                       FC_HIDDEN, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)
    print(model)
    print(f"Parameters : {sum(p.numel() for p in model.parameters())}\n")

    # training loop
    best_val, best_state, wait = float("inf"), None, 0
    tr_losses, va_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, va_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_loss)
        tr_losses.append(tr_loss);  va_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {tr_loss:.6f} | val {va_loss:.6f}"
                  f"  {'*' if wait == 0 else ''}")

        if wait >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}  (best val MSE {best_val:.6f})")
            break

    model.load_state_dict(best_state)

    # val evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            preds.append(norm.iy(model(xb.to(device)).squeeze(-1).cpu().numpy()))
            trues.append(norm.iy(yb.numpy()))
    y_pred = np.concatenate(preds);  y_true = np.concatenate(trues)
    mae  = np.abs(y_pred - y_true).mean()
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    print(f"\nVal  |  MAE {mae:.5f} Nm  |  RMSE {rmse:.5f} Nm")


    # plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'LSTM training  (window={WINDOW_SIZE} steps)', fontsize=11)

    ax1.plot(tr_losses, label="train");  ax1.plot(va_losses, label="val")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("MSE (normalised)")
    ax1.set_title("Loss curves");  ax1.legend();  ax1.set_yscale("log")

    lim = [min(y_true.min(), y_pred.min()) - 0.02,
           max(y_true.max(), y_pred.max()) + 0.02]
    ax2.scatter(y_true, y_pred, s=4, alpha=0.4)
    ax2.plot(lim, lim, "r--", lw=1)
    ax2.set_xlim(lim);  ax2.set_ylim(lim)
    ax2.set_xlabel("True force_raw [N]");  ax2.set_ylabel("Predicted force [N]")
    ax2.set_title(f"Val scatter  (MAE={mae:.4f}, RMSE={rmse:.4f})")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "2-2_LSTM_results.png")
    plt.savefig(plot_path, dpi=150);  plt.show()
    print(f"Plot saved  -> {plot_path}")

    # save
    t = input("Press Enter to save the model.../q to cancel: ")
    if t.lower() == 'q':
        print("Save cancelled.")
        exit()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state": best_state,
        "model_type":  "lstm",
        "n_in":        len(INPUT_COLS),
        "input_cols":  INPUT_COLS,
        "target_col":  TARGET_COL,
        "window_size": WINDOW_SIZE,
        "window_s":    WINDOW_S,
        "loop_hz":     LOOP_HZ,
        "lstm_hidden": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "fc_hidden":   FC_HIDDEN,
        "dropout":     DROPOUT,
        "norm": {"mean_X": norm.mean_X, "std_X": norm.std_X,
                 "mean_y": norm.mean_y, "std_y": norm.std_y},
    }, SAVE_PATH)
    print(f"Model saved -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
