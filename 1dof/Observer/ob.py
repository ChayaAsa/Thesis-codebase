import numpy as np
import matplotlib.pyplot as plt

# True plant parameters --------------------------------------------------
J   = 0.01     # rotor inertia [kg*m^2]
B   = 0.05     # viscous friction [N*m*s/rad]
r   = 0.20     # lever arm [m]

# ---- Nominal model (inside observers) ---------------------------------------
# Try J_n = 0.012 or B_n = 0.03 to explore robustness to model mismatch.
J_n = 0.01
B_n = 0.05

# Observer Q-filter bandwidths -------------------------------------------
WQ_DOB  = 50.0   # [rad/s] DOB cutoff ~ 8 Hz
WQ_RFOB = 30.0   # [rad/s] RFOB cutoff ~ 5 Hz

# PD controller ----------------------------------------------------------
KP      = 5.0    # [N*m/rad]
KD      = 0.5    # [N*m*s/rad]
THETA_D = 0.0    # desired position [rad]

# Simulation setup -------------------------------------------------------
DT    = 0.001   # [s] 1 kHz
T_END = 8.0     # [s]

t = np.arange(0.0, T_END, DT)
N = len(t)

# External contact force profile
F_true = np.zeros(N)
F_true[(t >= 2.0) & (t < 5.0)] = 5.0   # 5 N step

# Logging arrays ---------------------------------------------------------
theta   = np.zeros(N)
omega   = np.zeros(N)
tau_log = np.zeros(N)
d_true  = np.zeros(N)   # true disturbance torque [N*m]
d_dob   = np.zeros(N)   # DOB estimate [N*m]
f_rfob  = np.zeros(N)   # RFOB force estimate [N]

# Observer states --------------------------------------------------------
p_dob  = 0.0
p_rfob = 0.0

# Simulation loop --------------------------------------------------------
for k in range(N - 1):
    th = theta[k]
    w  = omega[k]
    Fk = F_true[k]

    # PD controller
    tau = KP * (THETA_D - th) + KD * (0.0 - w)

    # True disturbance torque on the joint
    tau_dis = -B * w - Fk * r

    # DOB update
    p_dob += (-WQ_DOB * p_dob + J_n * WQ_DOB**2 * w + WQ_DOB * tau) * DT
    d_hat  = J_n * WQ_DOB * w - p_dob

    # RFOB update
    p_rfob += (-WQ_RFOB * p_rfob
               + J_n * WQ_RFOB**2 * w
               + WQ_RFOB * (tau - B_n * w)) * DT
    d_r_hat   = J_n * WQ_RFOB * w - p_rfob
    f_ext_hat = -d_r_hat / r

    # Plant (Euler integration)
    alpha      = (tau + tau_dis) / J
    omega[k+1] = w  + alpha * DT
    theta[k+1] = th + w     * DT

    # Log
    tau_log[k] = tau
    d_true[k]  = tau_dis
    d_dob[k]   = d_hat
    f_rfob[k]  = f_ext_hat

# Fill last sample
for arr in (tau_log, d_true, d_dob, f_rfob):
    arr[-1] = arr[-2]

# Console summary --------------------------------------------------------
idx_on  = (t >= 3.5) & (t < 4.5)   # mid-disturbance window
idx_off = (t >= 6.5)                # post-disturbance

print("== Steady-state summary ==")
print(f"  During contact (t in [3.5, 4.5] s):")
print(f"    d_true  mean = {d_true[idx_on].mean():.4f} N*m"
      f"  (expected = {-5.0*r:.4f} N*m)")
print(f"    d_hat   mean = {d_dob[idx_on].mean():.4f} N*m  (DOB)")
print(f"    f_rfob  mean = {f_rfob[idx_on].mean():.4f} N"
      f"   (expected = 5.0000 N)")
print(f"  After contact (t > 6.5 s):")
print(f"    d_hat   mean = {d_dob[idx_off].mean():.4f} N*m  (expected ~= 0)")
print(f"    f_rfob  mean = {f_rfob[idx_off].mean():.4f} N    (expected ~= 0)")
print("=" * 42)

# Plots ------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
fig.suptitle(
    'DOB + RFOB  --  1-DOF Simulation\n'
    'J=0.01 kg*m^2,  B=0.05,  r=0.20 m  |  '
    'F_ext = 5 N  @  t in [2, 5] s  |  '
    'wq_DOB=50, wq_RFOB=30 rad/s',
    fontsize=9,
)

axes[0].plot(t, np.degrees(theta), 'b', lw=1.5, label='theta measured')
axes[0].axhline(np.degrees(THETA_D), color='k', ls='--', lw=1, label='theta desired')
axes[0].set_ylabel('Position [deg]')

axes[1].plot(t, tau_log, 'g', lw=1.2, label='tau_cmd [N*m]')
axes[1].set_ylabel('Torque [N*m]')

axes[2].plot(t, d_true, 'k',   lw=1.5, label='d_true (total disturbance)')
axes[2].plot(t, d_dob,  'r--', lw=1.5, label='d_hat  (DOB estimate)')
axes[2].set_ylabel('Disturbance [N*m]')

axes[3].plot(t, F_true, 'k',   lw=1.5, label='F_ext true [N]')
axes[3].plot(t, f_rfob, 'r--', lw=1.5, label='F_hat RFOB [N]')
axes[3].set_ylabel('Contact force [N]')
axes[3].set_xlabel('Time [s]')

for ax in axes:
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvspan(2.0, 5.0, alpha=0.07, color='orange')

plt.tight_layout()
plt.show()
