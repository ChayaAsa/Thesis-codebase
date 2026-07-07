import csv, glob, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, '..', '3_model', '3-1_NN_model.pt')
DATA_DIR   = os.path.join(HERE, '..', '1_data')


class TauNet(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# load model
ckpt  = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = TauNet(n_in=ckpt['n_in'], hidden=ckpt['hidden'])
model.load_state_dict(ckpt['model_state'])
model.eval()
norm        = {k: v.astype(np.float32) for k, v in ckpt['norm'].items()}
input_cols  = ckpt['input_cols']
print(f"Model loaded.  Inputs: {input_cols}")

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

X_real = np.concatenate(all_X)   # (N, n_in)
y_real = np.concatenate(all_y)   # (N,)
print(f"  total: {len(X_real)} samples\n")

# predict on real data
X_norm = (X_real - norm['mean_X']) / norm['std_X']
with torch.no_grad():
    y_norm = model(torch.from_numpy(X_norm))
y_pred = y_norm.numpy().flatten() * norm['std_y'].flat[0] + norm['mean_y'].flat[0]

mae  = np.abs(y_pred - y_real).mean()
rmse = np.sqrt(((y_pred - y_real) ** 2).mean())
print(f"Real data  |  MAE {mae:.5f} N  |  RMSE {rmse:.5f} N")

# ── sweep curve (tau_meas varies, pos=0, vel=0) ───────────────────────────────
n_in         = ckpt['n_in']
tau_sweep_in = np.linspace(-0.5, 0.5, 300, dtype=np.float32)
x_sweep      = np.zeros((300, n_in), dtype=np.float32)
x_sweep[:, 0] = tau_sweep_in              # tau_meas_Nm channel
x_sweep_n    = (x_sweep - norm['mean_X']) / norm['std_X']
with torch.no_grad():
    f_sweep_out = model(torch.from_numpy(x_sweep_n))
f_raw_sweep = f_sweep_out.numpy().flatten() * norm['std_y'].flat[0] + norm['mean_y'].flat[0]

# ── 3D surface grid (tau × pos, vel=0) ───────────────────────────────────────
tau_g = np.linspace(X_real[:, 0].min(), X_real[:, 0].max(), 40, dtype=np.float32)
pos_g = np.linspace(X_real[:, 1].min(), X_real[:, 1].max(), 40, dtype=np.float32)
T_g, P_g = np.meshgrid(tau_g, pos_g)
grid_in = np.zeros((T_g.size, n_in), dtype=np.float32)
grid_in[:, 1] = P_g.ravel()
grid_n = (grid_in - norm['mean_X']) / norm['std_X']
with torch.no_grad():
    F_surf_flat = model(torch.from_numpy(grid_n)).numpy().flatten()
F_surf = (F_surf_flat * norm['std_y'].flat[0] + norm['mean_y'].flat[0]).reshape(T_g.shape)

# plots
residual = y_pred - y_real

fig = plt.figure(figsize=(18, 10))
fig.suptitle('MLP virtual force sensor', fontsize=13)

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

# (1) scatter colored by pos_rad + 1D model curve (pos=vel=0 slice)
sc = ax1.scatter(X_real[:, 0], y_real, s=4, alpha=0.3,
                 c=X_real[:, 1], cmap='coolwarm')
plt.colorbar(sc, ax=ax1, label='pos_rad [rad]')
ax1.plot(tau_sweep_in, f_raw_sweep, color='tomato', lw=2, label='model (pos=vel=0)')
ax1.set_xlabel('tau_meas_Nm [N·m]')
ax1.set_ylabel('f_raw [N]')
ax1.set_title('Scatter colored by pos_rad + model curve')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# (2) pred vs true
lim = [min(y_real.min(), y_pred.min()) - 0.1,
       max(y_real.max(), y_pred.max()) + 0.1]
ax2.scatter(y_real, y_pred, s=4, alpha=0.3, color='steelblue')
ax2.plot(lim, lim, 'r--', lw=1)
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel('True f_raw [N]')
ax2.set_ylabel('Predicted f_raw [N]')
ax2.set_title(f'Pred vs True  (MAE={mae:.4f} N, RMSE={rmse:.4f} N)')
ax2.grid(True, alpha=0.3)

# (3) 3D model surface with real data projected on top
ax3.plot_surface(T_g, P_g, F_surf, alpha=0.55, cmap='viridis', linewidth=0)
ax3.scatter(X_real[:, 0], X_real[:, 1], y_real,
            s=1, alpha=0.15, color='tomato', depthshade=True)
ax3.set_xlabel('tau_meas [N·m]', fontsize=8)
ax3.set_ylabel('pos_rad [rad]', fontsize=8)
ax3.set_zlabel('f_raw [N]', fontsize=8)
ax3.set_title('Model surface (vel=0)\n+ real data', fontsize=9)

# (4) residual vs tau_meas
ax4.scatter(X_real[:, 0], residual, s=4, alpha=0.3, color='mediumpurple')
ax4.axhline(0, color='r', lw=1, linestyle='--')
ax4.set_xlabel('tau_meas_Nm [N·m]')
ax4.set_ylabel('Residual (pred - true) [N]')
ax4.set_title('Residual vs tau_meas')
ax4.grid(True, alpha=0.3)

# (5) residual vs pos_rad
ax5.scatter(X_real[:, 1], residual, s=4, alpha=0.3, color='darkorange')
ax5.axhline(0, color='r', lw=1, linestyle='--')
ax5.set_xlabel('pos_rad [rad]')
ax5.set_ylabel('Residual (pred - true) [N]')
ax5.set_title('Residual vs pos_rad')
ax5.grid(True, alpha=0.3)

# (6) residual vs vel_rad_s
ax6.scatter(X_real[:, 2], residual, s=4, alpha=0.3, color='teal')
ax6.axhline(0, color='r', lw=1, linestyle='--')
ax6.set_xlabel('vel_rad_s [rad/s]')
ax6.set_ylabel('Residual (pred - true) [N]')
ax6.set_title('Residual vs vel_rad_s')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(HERE, 'pic', '4-1_NN_predict.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150)
print(f"Plot saved -> {plot_path}")
plt.show()
