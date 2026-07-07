import csv, glob, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, '..', '3_model', '3-2_LSTM_model.pt')
DATA_DIR   = os.path.join(HERE, '..', '1_data')


class TauNetLSTM(nn.Module):
    def __init__(self, n_in, lstm_hidden, lstm_layers, fc_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# load model
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = TauNetLSTM(
    n_in        = ckpt['n_in'],
    lstm_hidden = ckpt['lstm_hidden'],
    lstm_layers = ckpt['lstm_layers'],
    fc_hidden   = ckpt['fc_hidden'],
    dropout     = ckpt['dropout'],
)
model.load_state_dict(ckpt['model_state'])
model.eval()

norm        = {k: np.float32(v) if np.ndim(v) == 0 else np.array(v, dtype=np.float32)
               for k, v in ckpt['norm'].items()}
WINDOW_SIZE = ckpt['window_size']
WINDOW_S    = ckpt['window_s']
LOOP_HZ     = ckpt['loop_hz']
input_cols  = ckpt['input_cols']
n_in        = ckpt['n_in']

print(f"Model loaded.  Inputs: {input_cols}")
print(f"Window: {WINDOW_SIZE} steps ({WINDOW_S*1000:.0f} ms at {LOOP_HZ} Hz)\n")

# load real data
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
assert files, f"No CSV files in {DATA_DIR}"

all_X, all_y = [], []
for fpath in files:
    with open(fpath, newline='') as f:
        rows = list(csv.DictReader(f))
    all_X.append(np.array([[float(r[c]) for c in input_cols] for r in rows], dtype=np.float32))
    all_y.append(np.array([float(r['f_raw_N']) for r in rows], dtype=np.float32))
    print(f"  {os.path.basename(fpath)}  {len(rows)} rows")

X_raw = np.concatenate(all_X)   # (N, n_in)
y_raw = np.concatenate(all_y)   # (N,)
print(f"  total: {len(X_raw)} samples\n")

# build sliding windows
starts  = range(0, len(X_raw) - WINDOW_SIZE + 1)
X_win   = np.stack([X_raw[s:s + WINDOW_SIZE] for s in starts])        # (W, T, F)
y_true  = np.array([y_raw[s + WINDOW_SIZE - 1] for s in starts])      # (W,)

# normalise
X_win_n = (X_win - norm['mean_X']) / norm['std_X']

# predict on all windows (batched)
BATCH = 512
y_pred_list = []
with torch.no_grad():
    for i in range(0, len(X_win_n), BATCH):
        xb = torch.from_numpy(X_win_n[i:i + BATCH])
        yb = model(xb).squeeze(-1).numpy()
        y_pred_list.append(yb)

y_pred = np.concatenate(y_pred_list) * float(norm['std_y']) + float(norm['mean_y'])

mae  = np.abs(y_pred - y_true).mean()
rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
print(f"Real data  |  MAE {mae:.5f} N  |  RMSE {rmse:.5f} N")

# ── sweep curve (tau_meas varies, pos=0, vel=0) ───────────────────────────────
tau_sweep_in  = np.linspace(-0.5, 0.5, 300, dtype=np.float32)
f_raw_sweep   = np.empty_like(tau_sweep_in)
with torch.no_grad():
    for i, tau in enumerate(tau_sweep_in):
        window        = np.zeros((1, WINDOW_SIZE, n_in), dtype=np.float32)
        window[:, :, 0] = tau          # tau_meas_Nm channel; pos=vel=0
        w_norm        = (window - norm['mean_X']) / norm['std_X']
        y_n           = model(torch.from_numpy(w_norm))
        f_raw_sweep[i] = float(y_n.numpy().flat[0]) * float(norm['std_y']) + float(norm['mean_y'])

# ── 3D surface grid (tau × pos, vel=0, constant window) ──────────────────────
tau_g = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 30, dtype=np.float32)
pos_g = np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), 30, dtype=np.float32)
T_g, P_g = np.meshgrid(tau_g, pos_g)
# fill every timestep in each window with constant (tau, pos, vel=0)
surf_wins = np.zeros((T_g.size, WINDOW_SIZE, n_in), dtype=np.float32)
surf_wins[:, :, 0] = T_g.ravel()[:, None]
surf_wins[:, :, 1] = P_g.ravel()[:, None]
surf_wins_n = (surf_wins - norm['mean_X']) / norm['std_X']
with torch.no_grad():
    F_surf_flat = model(torch.from_numpy(surf_wins_n)).squeeze(-1).numpy()
F_surf = (F_surf_flat * float(norm['std_y']) + float(norm['mean_y'])).reshape(T_g.shape)

# plots
residual = y_pred - y_true
# use last timestep of each window for scatter x-axes
tau_last = X_win[:, -1, 0]   # tau_meas_Nm at last step
pos_last = X_win[:, -1, 1]   # pos_rad at last step
vel_last = X_win[:, -1, 2]   # vel_rad_s at last step

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f'LSTM virtual force sensor  (window={WINDOW_SIZE} steps, {WINDOW_S*1000:.0f} ms)', fontsize=13)

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

# (1) scatter colored by pos_rad + 1D model curve (pos=vel=0 slice)
sc = ax1.scatter(tau_last, y_true, s=4, alpha=0.3,
                 c=pos_last, cmap='coolwarm')
plt.colorbar(sc, ax=ax1, label='pos_rad [rad]')
ax1.plot(tau_sweep_in, f_raw_sweep, color='tomato', lw=2, label='model (pos=vel=0)')
ax1.set_xlabel('tau_meas_Nm [N·m]  (last step)')
ax1.set_ylabel('f_raw [N]')
ax1.set_title('Scatter colored by pos_rad + model curve')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# (2) pred vs true
lim = [min(y_true.min(), y_pred.min()) - 0.1,
       max(y_true.max(), y_pred.max()) + 0.1]
ax2.scatter(y_true, y_pred, s=4, alpha=0.3, color='steelblue')
ax2.plot(lim, lim, 'r--', lw=1)
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel('True f_raw [N]')
ax2.set_ylabel('Predicted f_raw [N]')
ax2.set_title(f'Pred vs True  (MAE={mae:.4f} N, RMSE={rmse:.4f} N)')
ax2.grid(True, alpha=0.3)

# (3) 3D model surface with real data projected on top
ax3.plot_surface(T_g, P_g, F_surf, alpha=0.55, cmap='viridis', linewidth=0)
ax3.scatter(tau_last, pos_last, y_true,
            s=1, alpha=0.15, color='tomato', depthshade=True)
ax3.set_xlabel('tau_meas [N·m]', fontsize=8)
ax3.set_ylabel('pos_rad [rad]', fontsize=8)
ax3.set_zlabel('f_raw [N]', fontsize=8)
ax3.set_title('Model surface (vel=0)\n+ real data', fontsize=9)

# (4) residual vs tau_meas
ax4.scatter(tau_last, residual, s=4, alpha=0.3, color='mediumpurple')
ax4.axhline(0, color='r', lw=1, linestyle='--')
ax4.set_xlabel('tau_meas_Nm [N·m]  (last step)')
ax4.set_ylabel('Residual (pred - true) [N]')
ax4.set_title('Residual vs tau_meas')
ax4.grid(True, alpha=0.3)

# (5) residual vs pos_rad
ax5.scatter(pos_last, residual, s=4, alpha=0.3, color='darkorange')
ax5.axhline(0, color='r', lw=1, linestyle='--')
ax5.set_xlabel('pos_rad [rad]  (last step)')
ax5.set_ylabel('Residual (pred - true) [N]')
ax5.set_title('Residual vs pos_rad')
ax5.grid(True, alpha=0.3)

# (6) residual vs vel_rad_s
ax6.scatter(vel_last, residual, s=4, alpha=0.3, color='teal')
ax6.axhline(0, color='r', lw=1, linestyle='--')
ax6.set_xlabel('vel_rad_s [rad/s]  (last step)')
ax6.set_ylabel('Residual (pred - true) [N]')
ax6.set_title('Residual vs vel_rad_s')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(HERE, 'pic', '4-2_LSTM_predict.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150)
print(f"Plot saved -> {plot_path}")
plt.show()
