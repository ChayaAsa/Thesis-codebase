import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize

warnings.filterwarnings("ignore")

# CONSTANTS
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
FS          = 100.0          # nominal sampling frequency [Hz]
DT          = 1.0 / FS      # nominal sample interval [s]
JACOBIAN    = 0.2            # lever arm length [m] (theory: K_static ~ 1/J = 5 N/Nm)


# SECTION 1 - LOAD DATA
# Why 3 trials?
#   ? Using multiple trials averages out run-to-run randomness.
#   ? We train on the first two and validate on the third -- a strict
#     "out-of-experiment" test that cannot overfit.
print("=" * 60)
print("  PLANT SYSTEM IDENTIFICATION -- tau_cmd -> f_raw")
print("=" * 60)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "random_step_*.csv")))
assert len(csv_files) >= 1, "No CSV files found in the script directory!"

trials = []
for f in csv_files:
    df = pd.read_csv(f)
    trials.append(df)
    print(f"  Loaded: {os.path.basename(f)}  ({len(df)} samples, "
          f"{df['time_s'].iloc[-1]:.1f} s)")


# SECTION 2 - PREPROCESS
# Step (a) -- Skip the warmup window
#   The first ~3 s have tau_cmd = 0.  During warmup the system is not being
#   excited, so including it would just add zeros to the regression.
#   We find the first sample where the absolute torque exceeds 10 mNm.
#
# Step (b) -- Resample to a uniform time grid
#   The hardware clock runs at "~100 Hz" but individual sample intervals vary
#   by ?1-2 ms due to OS scheduling jitter.
#   FFT-based spectral methods (Welch, coherence) REQUIRE equal time spacing.
#   We create a perfect grid t_uniform = [t0, t0+DT, t0+2DT, ...] and
#   interpolate both signals onto it.
#
# Step (c) -- Remove the mean (detrend by constant)
#   System ID identifies HOW CHANGES in input cause CHANGES in output.
#   Any non-zero mean is just a constant offset; it carries no dynamic info.
#   Removing the mean ensures the identified model's input is ?? and
#   output is ?F (deviations from equilibrium).

def preprocess(df, tau_thresh=0.01):
    t = df["time_s"].values
    u = df["tau_cmd_Nm"].values
    y = df["f_raw_N"].values

    # (a) find excitation start
    active = np.where(np.abs(u) > tau_thresh)[0]
    if len(active) == 0:
        raise ValueError("No excitation found above threshold!")
    i_start = active[0]

    t = t[i_start:]
    u = u[i_start:]
    y = y[i_start:]

    # (b) resample to uniform grid
    t_u = np.arange(t[0], t[-1], DT)
    u_u = np.interp(t_u, t, u)
    y_u = np.interp(t_u, t, y)

    # (c) remove mean
    u_d = u_u - np.mean(u_u)
    y_d = y_u - np.mean(y_u)

    return t_u, u_d, y_d


# Process all trials
processed = [preprocess(df) for df in trials]
t0, u0, y0 = processed[0]            # trial 0 -- main estimation trial
N = len(t0)

# Train / validation split (70% / 30%) within trial 0
N_tr = int(0.7 * N)
t_tr, u_tr, y_tr = t0[:N_tr],       u0[:N_tr],       y0[:N_tr]
t_va, u_va, y_va = t0[N_tr:],       u0[N_tr:],       y0[N_tr:]

print(f"\nTrial 0 after preprocessing: {N} samples  ({N_tr} train / {N-N_tr} val)")

# Overview figure
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle("Preprocessed Data -- Trial 0  (train|val split shown)")
ax[0].plot(t0, u0, "b", lw=0.7); ax[0].axvline(t0[N_tr], color="r", ls="--", lw=1)
ax[0].set_ylabel("tau_cmd [Nm]"); ax[0].grid(True)
ax[1].plot(t0, y0, "r", lw=0.7); ax[1].axvline(t0[N_tr], color="r", ls="--", lw=1)
ax[1].set_ylabel("f_raw [N]"); ax[1].set_xlabel("Time [s]"); ax[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sysid_01_overview.png"), dpi=130); plt.show()


# SECTION 3 - STATIC GAIN ANALYSIS
# Idea:
#   During each constant-torque plateau, wait for transients to die out,
#   then measure the average force.  Plotting (tau_cmd_level, mean_force)
#   and fitting a line gives the static (DC) gain:
#       f_ss ~ K_static ? tau_cmd
#
# What K_static tells you:
#   ? Nominal expected value: K = 1 / JACOBIAN = 1 / 0.2 = 5 N/Nm
#   ? Any deviation means friction, gravity, or sensor offset is present.
#   ? The identified dynamic model's DC gain should match K_static.
#     If they differ, the identification window was not long enough.
#   ? Dead-zone: if points near zero don't follow the line -> stiction.
#
# How we find plateaus:
#   Detect edges (large du/dt) to split the signal into constant segments.
#   For each segment, skip the first 40 % (transients), average the rest.

def static_gain_analysis(t, u, y, hold_frac=0.6, min_seg_len=30):
    # Find edges: samples where |?u| is large
    du = np.diff(u)
    edges = np.concatenate([[0], np.where(np.abs(du) > 0.008)[0] + 1, [len(u)]])

    levels, means, stds = [], [], []
    for i in range(len(edges) - 1):
        i0, i1 = edges[i], edges[i + 1]
        if i1 - i0 < min_seg_len:
            continue
        seg_u = u[i0:i1]
        seg_y = y[i0:i1]
        if np.std(seg_u) > 0.01:          # not truly constant -> skip
            continue
        tau_lvl = np.mean(seg_u)
        n_skip  = int((1 - hold_frac) * len(seg_y))
        steady  = seg_y[n_skip:]
        levels.append(tau_lvl); means.append(np.mean(steady)); stds.append(np.std(steady))

    levels = np.array(levels); means = np.array(means); stds = np.array(stds)

    K_static, offset = (5.0, 0.0)
    if len(levels) >= 3:
        nz = np.abs(levels) > 0.02        # exclude zero-input points for line fit
        if np.sum(nz) >= 2:
            p = np.polyfit(levels[nz], means[nz], 1)
            K_static, offset = p[0], p[1]

    return levels, means, stds, K_static, offset


tau_lvl, f_mean, f_std, K_static, K_offset = static_gain_analysis(t0, u0, y0)

print(f"\nStatic gain:  K_static = {K_static:.4f} N/Nm  (theoretical = {1/JACOBIAN:.1f} N/Nm)")
print(f"              offset   = {K_offset:.4f} N")
print(f"              plateaus found: {len(tau_lvl)}")

if len(tau_lvl) >= 3:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(tau_lvl, f_mean, yerr=f_std, fmt="ob", ms=5, capsize=3, label="Measured SS force")
    tau_fit = np.linspace(tau_lvl.min(), tau_lvl.max(), 100)
    ax.plot(tau_fit, K_static * tau_fit + K_offset, "r--", lw=2,
            label=f"Linear fit  K = {K_static:.3f} N/Nm")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("tau_cmd [Nm]"); ax.set_ylabel("Steady-state f_raw [N]")
    ax.set_title("Static Gain Curve")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "sysid_02_static_gain.png"), dpi=130); plt.show()


# SECTION 4 - NON-PARAMETRIC FREQUENCY RESPONSE (ETFE)
# Theory (H(jw) without assuming a model structure):
#
#   For a linear system with additive output noise n(t):
#       Y(jw) = H(jw)*U(jw) + N(jw)
#
#   The minimum-variance estimate of H at each frequency is:
#
#       H_hat(jw) = S_uy(w) / S_uu(w)
#
#   where:
#       S_uu(w) = E[|U(jw)|2]            -- input  Power Spectral Density
#       S_uy(w) = E[U*(jw) * Y(jw)]      -- Cross-Power Spectral Density
#
#   We compute these using Welch's method:
#   ? Divide the signal into overlapping windows of length NPERSEG
#   ? Apply a Hann window (reduces spectral leakage)
#   ? Average the periodograms across windows -> smoother estimate
#   -> More windows = smoother PSD; fewer windows = finer freq resolution.
#   Rule of thumb: N / NPERSEG ? 8 overlapping segments for a useful estimate.
#
# Coherence ?2(w):
#       ?2(w) = |S_uy(w)|2 / (S_uu(w) * S_yy(w))
#
#   Range [0, 1].
#   ? ?2 ~ 1 : at this frequency, almost ALL of the output power is caused
#               by the input -> the ETFE is trustworthy here.
#   ? ?2 < 0.8 : noise, nonlinearity, or weak excitation dominates -> the
#                 ETFE estimate is unreliable here.  Do NOT fit parameters
#                 using frequencies with low coherence!

# Window size: at N_tr ~ 12 000 samples and NPERSEG=512 we get ~23 averages
NPERSEG = min(512, N_tr // 8)

freqs, S_uu = signal.welch(u_tr, fs=FS, nperseg=NPERSEG, window="hann")
_,    S_yy = signal.welch(y_tr, fs=FS, nperseg=NPERSEG, window="hann")
_,    S_uy = signal.csd(u_tr, y_tr, fs=FS, nperseg=NPERSEG, window="hann")

# ETFE: H_hat(jw) = S_uy / S_uu
# Guard against division by near-zero (frequencies where input has no power)
H_etfe   = np.where(S_uu > 1e-12, S_uy / S_uu, 0j)
mag_etfe = 20 * np.log10(np.abs(H_etfe) + 1e-12)   # dB
phs_etfe = np.rad2deg(np.unwrap(np.angle(H_etfe)))  # degrees, unwrapped
coh      = np.abs(S_uy) ** 2 / (S_uu * S_yy + 1e-20)

# Frequency mask: frequencies that are excited AND coherent enough to trust
inp_mask  = S_uu > 0.01 * np.max(S_uu)   # input power > 1 % of peak
coh_mask  = coh  > 0.60                   # coherence above 60 %
rel_mask  = inp_mask & coh_mask           # "reliable" band

f_min_rel = freqs[rel_mask][0]  if rel_mask.any() else 0.0
f_max_rel = freqs[rel_mask][-1] if rel_mask.any() else FS / 2
print(f"\nETFE reliable band: {f_min_rel:.3f} - {f_max_rel:.3f} Hz  "
      f"({rel_mask.sum()} frequency bins)")

# Non-parametric plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("Non-Parametric Frequency Response (ETFE)")

axes[0].semilogy(freqs, S_uu, "b", lw=1.2, label="S_uu  (input PSD)")
axes[0].semilogy(freqs, S_yy, "r", lw=1.2, label="S_yy  (output PSD)")
axes[0].set_ylabel("PSD"); axes[0].legend(); axes[0].grid(True)
axes[0].set_title("Input & Output Power Spectral Density")

axes[1].semilogx(freqs[1:], mag_etfe[1:], "g", lw=0.8, alpha=0.4, label="ETFE (all)")
axes[1].semilogx(freqs[rel_mask], mag_etfe[rel_mask], "g", lw=2, label="ETFE (reliable)")
axes[1].set_ylabel("Magnitude [dB]"); axes[1].grid(True); axes[1].legend()
axes[1].set_title("Bode -- Magnitude"); axes[1].set_xlim([0.1, FS / 2])

axes[2].plot(freqs, coh, "purple", lw=1.5, label="Coherence ?2")
axes[2].axhline(0.8, color="r", ls="--", lw=1, label="threshold 0.8")
axes[2].fill_between(freqs, coh, 0.8, where=(coh > 0.8), alpha=0.15, color="green")
axes[2].set_xlabel("Frequency [Hz]"); axes[2].set_ylabel("Coherence ?2")
axes[2].set_ylim([0, 1.1]); axes[2].legend(); axes[2].grid(True)
axes[2].set_title("Coherence  (> 0.8 = trustworthy)")

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sysid_03_etfe.png"), dpi=130); plt.show()


# SECTION 5 - ARX MODEL  (discrete-time, pure least-squares)
# ARX = Auto-Regressive eXogenous input.
# Discrete-time model for order (na, nb, nk):
#
#   y[k] = ?a1y[k?1] ? a2y[k?2] ? ? ? a??y[k?na]
#          + b0u[k?nk] + b1u[k?nk?1] + ? + b(nb?1)u[k?nk?nb+1]
#
# Key insight: if we define:
#   ?[k] = [?y[k?1], ?, ?y[k?na], u[k?nk], ?, u[k?nk?nb+1]]   (row vector)
#   ?    = [a1, ?, a??, b0, ?, b(nb?1)]                        (parameter vector)
# then:
#   y[k] = ?[k] * ?
#
# Stacking all time steps:   Y = ? * ?
# Least-squares solution:    ?? = (???)?1 ?? Y   (no iterations!)
#
# WHY ARX is great for beginners:
#   + Closed-form solution -- fast, always finds the global optimum
#   + No tuning, no learning rate
#   + Easy to understand: it's just linear regression
#   - Assumes noise enters AT the input (through A-polynomial)
#     -> slightly biased when true noise is at the output
#   -> For output noise: use OE or PEM models (iterative, more complex)
#
# nk = 1 (one-sample delay) is physically correct here because the
# hardware control loop runs at 100 Hz and sends the torque command 1 step
# ahead of when the force sensor reads.

def fit_arx(u, y, na, nb, nk=1):
    N       = len(u)
    n_start = max(na, nb + nk - 1)   # first valid time step

    n_rows  = N - n_start
    Phi     = np.zeros((n_rows, na + nb))
    Y_vec   = np.zeros(n_rows)

    for i in range(n_rows):
        k = i + n_start
        # Past outputs (negative sign already baked in -- matches ARX convention)
        for j in range(na):
            Phi[i, j] = -y[k - 1 - j]
        # Past inputs with delay nk
        for j in range(nb):
            idx = k - nk - j
            Phi[i, na + j] = u[idx] if idx >= 0 else 0.0
        Y_vec[i] = y[k]

    theta, _, _, _ = np.linalg.lstsq(Phi, Y_vec, rcond=None)
    a = theta[:na]
    b = theta[na:]
    y_hat = Phi @ theta    # one-step-ahead on training data

    return a, b, y_hat, n_start


def simulate_arx(u, y_init, a, b, nk, n_start):
    na = len(a); nb = len(b)
    N  = len(u)
    y_sim = np.zeros(N)
    y_sim[:n_start] = y_init[:n_start]   # warm-start with true output

    for k in range(n_start, N):
        pred = 0.0
        for j in range(na):
            pred -= a[j] * y_sim[k - 1 - j]
        for j in range(nb):
            idx = k - nk - j
            pred += b[j] * (u[idx] if idx >= 0 else 0.0)
        y_sim[k] = np.clip(pred, -50, 50)   # safety clip against divergence

    return y_sim


def arx_bode(a, b, nk, freqs_hz):
    omega = 2 * np.pi * np.asarray(freqs_hz) * DT   # w*T (dimensionless)
    z_inv = np.exp(-1j * omega)

    A = np.ones(len(freqs_hz), dtype=complex)
    for k, ak in enumerate(a):
        A += ak * z_inv ** (k + 1)

    B = np.zeros(len(freqs_hz), dtype=complex)
    for k, bk in enumerate(b):
        B += bk * z_inv ** (nk + k)

    return B / A


def fit_percent(y_true, y_pred):
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.std(y_true)
    return max(0.0, 100.0 * (1.0 - num / (den + 1e-12)))


# Compare several ARX model orders
orders = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1)]
arx_models = {}

print(f"\n{'Model':<16}  {'Train fit%':>10}  {'Val  fit%':>10}")
print("-" * 42)

for (na, nb, nk) in orders:
    a, b, y_hat_tr, n_st = fit_arx(u_tr, y_tr, na, nb, nk)

    # One-step fit on training set
    fp_tr = fit_percent(y_tr[n_st:], y_hat_tr)

    # Free-run simulation on VALIDATION set (the honest test!)
    y_sim_va = simulate_arx(u_va, y_va, a, b, nk, n_st)
    fp_va    = fit_percent(y_va[n_st:], y_sim_va[n_st:])

    label = f"ARX({na},{nb},{nk})"
    arx_models[label] = dict(a=a, b=b, nk=nk, na=na, nb=nb,
                              n_st=n_st, y_sim_va=y_sim_va, fp_va=fp_va)
    print(f"  {label:<14}  {fp_tr:>10.1f}  {fp_va:>10.1f}")

# Choose best ARX (highest validation fit%)
best_arx_key = max(arx_models, key=lambda k: arx_models[k]["fp_va"])
best_arx     = arx_models[best_arx_key]
print(f"\n  Best ARX: {best_arx_key}  (val fit = {best_arx['fp_va']:.1f}%)")

# ARX Bode curve (evaluated on unit circle)
H_arx_best = arx_bode(best_arx["a"], best_arx["b"], best_arx["nk"], freqs[1:])


# SECTION 6 - 2ND-ORDER CONTINUOUS TF FITTING
# Standard 2nd-order system:
#
#              K * w?2
# G(s)
#           s2 + 2*z*w?*s + w?2
#
# Parameters and their physical meaning:
#   K   [N/Nm]   DC gain -- steady-state force per unit torque command
#                G(0) = K*w?2/w?2 = K
#   w?  [rad/s]  Natural frequency -- how fast the system oscillates or responds
# relates to resonance peak in Bode magnitude
#   z   [-]      Damping ratio
#                  z < 1 -> underdamped: oscillatory step response, Bode peak
#                  z = 1 -> critically damped: fastest non-overshoot response
#                  z > 1 -> overdamped: sluggish, two real poles (two 1st-order)
#
# Evaluate at s = jw (sinusoidal steady state):
#   G(jw) = K*w?2 / (w?2 ? w2 + 2j*z*w?*w)
#
# Fitting strategy -- FREQUENCY DOMAIN OPTIMISATION:
#   1. Use the ETFE H_hat(jw) as our "measurement" of the real system.
#   2. Define a cost function = weighted error between model and ETFE.
#   3. Minimise with scipy.optimize.minimize (L-BFGS-B with bounds).
#
# Why fit in frequency domain?
#   ? We can weight frequencies by coherence -> trust reliable data more.
#   ? Physical meaning of w? and z is directly visible in the Bode shape.
#   ? Time-domain fitting is more sensitive to initial conditions / DC drift.
#   ? Faster to evaluate: no ODE integration needed.
#
# We try multiple initial guesses to avoid local minima.

def tf2_freq(K, wn, zeta, f_hz):
    w  = 2 * np.pi * np.asarray(f_hz)
    jw = 1j * w
    return K * wn**2 / (wn**2 + 2 * zeta * wn * jw + jw**2)


def cost_tf2(params, f_fit, H_meas, weights):
    K, wn, zeta = params
    if K <= 0 or wn <= 0 or zeta <= 0 or zeta > 8:
        return 1e10

    H_m    = tf2_freq(K, wn, zeta, f_fit)
    mag_e  = (20 * np.log10(np.abs(H_m) + 1e-12) -
              20 * np.log10(np.abs(H_meas) + 1e-12))
    phs_e  = np.rad2deg(np.angle(H_m)) - np.rad2deg(np.angle(H_meas))
    phs_e  = (phs_e + 180) % 360 - 180    # wrap to (?180, 180)

    # Magnitude weighted 10? heavier than phase (typical choice)
    return np.sum(weights * mag_e**2) + 0.1 * np.sum(weights * phs_e**2)


# Frequencies used for fitting: reliable band, exclude DC and above Nyquist/2
fit_mask = rel_mask & (freqs > 0.1) & (freqs < 20.0)
f_fit    = freqs[fit_mask]
H_fit    = H_etfe[fit_mask]
w_fit    = coh[fit_mask]

# Multiple starting points (L-BFGS-B with bounds)
K0    = max(abs(K_static), 0.5)
bnds  = [(0.1, 50), (2*np.pi*0.1, 2*np.pi*40), (0.05, 5.0)]

starts = [
    [K0,         2*np.pi*5.0,  0.7],
    [K0,         2*np.pi*2.0,  0.3],
    [K0,         2*np.pi*10.0, 1.0],
    [K0 * 0.5,   2*np.pi*5.0,  0.5],
]

best_result = None
for p0 in starts:
    res = optimize.minimize(cost_tf2, p0, args=(f_fit, H_fit, w_fit),
                            method="L-BFGS-B", bounds=bnds,
                            options={"maxiter": 3000, "ftol": 1e-14})
    if best_result is None or res.fun < best_result.fun:
        best_result = res

K_fit, wn_fit, zeta_fit = best_result.x
fn_fit = wn_fit / (2 * np.pi)

print(f"\n2nd-order TF fit (freq-domain optimisation):")
print(f"  K    = {K_fit:.4f}  N/Nm      (DC gain, compare to K_static = {K_static:.4f})")
print(f"  w?   = {wn_fit:.4f} rad/s   ({fn_fit:.4f} Hz)")
print(f"  z    = {zeta_fit:.4f}          ({'underdamped' if zeta_fit < 1 else 'overdamped / critically damped'})")
print(f"  cost = {best_result.fun:.5f}   (converged = {best_result.success})")

# Build scipy LTI for simulation
# G(s) = K*w?2 / (s2 + 2*z*w?*s + w?2)
# scipy.signal.lti(num_coeffs, den_coeffs)
#   num = [K*w?2]
#   den = [1,   2*z*w?,   w?2]
# s2       s1       s0
num_tf2 = [K_fit * wn_fit**2]
den_tf2 = [1.0, 2*zeta_fit*wn_fit, wn_fit**2]
sys_lti = signal.lti(num_tf2, den_tf2)

print(f"\n  G(s) = {K_fit*wn_fit**2:.4f}")
print(f"         -----------------------------------------")
print(f"         s2 + {2*zeta_fit*wn_fit:.4f}*s + {wn_fit**2:.4f}")

# Frequency response of the fitted TF (for Bode)
H_tf2_bode = tf2_freq(K_fit, wn_fit, zeta_fit, freqs[1:])


# SECTION 7 - BODE PLOT COMPARISON
# A Bode plot shows:
#   ? Magnitude [dB] vs frequency -- how much the system amplifies each freq
#   ? Phase [deg]    vs frequency -- how much the system delays each freq
#
# The goal is for our parametric models to overlap with the ETFE in the
# reliable frequency range (shaded green via coherence).
#
# If there is a big discrepancy:
#   ? Low freq (< 1 Hz): may need higher model order for more complex dynamics
#   ? High freq (> 10 Hz): might be sensor noise or anti-alias filtering
#   ? At the resonance peak: w? and z need adjustment

f_plot = freqs[1:]   # skip DC bin

fig, axes = plt.subplots(3, 1, figsize=(12, 11))
fig.suptitle("Bode Plot: ETFE vs ARX vs 2nd-Order TF")

# Magnitude
ax = axes[0]
ax.semilogx(f_plot, 20*np.log10(np.abs(H_etfe[1:])+1e-12), "g", lw=0.7, alpha=0.3)
ax.semilogx(f_plot[rel_mask[1:]], 20*np.log10(np.abs(H_etfe[1:][rel_mask[1:]])+1e-12),
            "g", lw=2, label="ETFE (reliable)")
ax.semilogx(f_plot, 20*np.log10(np.abs(H_arx_best)+1e-12), "b--", lw=1.5,
            label=f"{best_arx_key}")
ax.semilogx(f_plot, 20*np.log10(np.abs(H_tf2_bode)+1e-12), "r-", lw=2,
            label=f"2nd-order TF  (w?={fn_fit:.2f} Hz, z={zeta_fit:.3f})")
ax.set_ylabel("Magnitude [dB]"); ax.grid(True, which="both"); ax.legend()
ax.set_xlim([0.1, FS/2]); ax.set_title("Magnitude")

# Phase
ax = axes[1]
ax.semilogx(f_plot, np.rad2deg(np.unwrap(np.angle(H_etfe[1:]))), "g", lw=0.7, alpha=0.3)
ax.semilogx(f_plot[rel_mask[1:]], np.rad2deg(np.unwrap(np.angle(H_etfe[1:][rel_mask[1:]]))),
            "g", lw=2)
ax.semilogx(f_plot, np.rad2deg(np.unwrap(np.angle(H_arx_best))), "b--", lw=1.5)
ax.semilogx(f_plot, np.rad2deg(np.unwrap(np.angle(H_tf2_bode))), "r-",  lw=2)
ax.set_ylabel("Phase [deg]"); ax.grid(True, which="both")
ax.set_xlim([0.1, FS/2]); ax.set_title("Phase")

# Coherence
ax = axes[2]
ax.semilogx(freqs, coh, "purple", lw=1.5)
ax.axhline(0.8, color="r", ls="--", lw=1, label="threshold 0.8")
ax.fill_between(freqs, coh, 0.8, where=(coh > 0.8), alpha=0.12, color="green")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Coherence ?2")
ax.set_xlim([0.1, FS/2]); ax.set_ylim([0, 1.1])
ax.grid(True, which="both"); ax.legend(); ax.set_title("Coherence")

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sysid_04_bode.png"), dpi=130); plt.show()


# SECTION 8 - TIME-DOMAIN VALIDATION
# We simulate BOTH models on the VALIDATION data (the 30% they never saw).
#
# "Simulation" vs "one-step-ahead prediction":
#   ? One-step-ahead: at each step, uses the TRUE past output y[k-1].
#     Result always looks good because real data corrects errors.
#   ? Simulation (free-run): uses only the model's own past predictions.
#     Errors accumulate. This is the HONEST test.
#
# fit% metric (MATLAB convention):
#   fit% = 100 ? (1 ? ?y_meas ? y_sim?2 / ?y_meas ? ??2)
#
#   ? 100% : perfect match
#   ?   0% : model just predicts the mean (no better than constant)
#   ?  <0% : model is actively wrong
#
# Typical acceptable fit% for a mechatronic system:
#   ? > 75% : very good -- ready for controller design
#   ? 50-75%: acceptable -- may need higher order or nonlinear terms
#   ? < 50% : poor -- reconsider experiment design or model structure

# Simulate continuous TF (2nd-order) with scipy.signal.lsim
t_va_rel = t_va - t_va[0]              # relative time (scipy.signal.lsim needs t ? 0)
_, y_tf2_sim, _ = signal.lsim(sys_lti, u_va, t_va_rel, X0=None)

# ARX simulation (already computed in Section 5)
y_arx_sim = best_arx["y_sim_va"]
n_st_arx  = best_arx["n_st"]

fp_tf2 = fit_percent(y_va, y_tf2_sim)
fp_arx = fit_percent(y_va[n_st_arx:], y_arx_sim[n_st_arx:])

print(f"\nTime-domain validation on 30 % held-out data:")
print(f"  2nd-order TF:  {fp_tf2:.1f}%")
print(f"  {best_arx_key}: {fp_arx:.1f}%")

t_rel = t_va - t_va[0]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Time-Domain Validation (30 % hold-out)", fontsize=13)

axes[0].plot(t_rel, u_va, "k", lw=0.7)
axes[0].set_ylabel("tau_cmd [Nm]"); axes[0].grid(True)
axes[0].set_title("Input (torque command)")

axes[1].plot(t_rel, y_va,       "k",  lw=1.2, label="Measured f_raw")
axes[1].plot(t_rel, y_tf2_sim,  "r--", lw=1.5, label=f"2nd-order TF ({fp_tf2:.0f}%)")
axes[1].set_ylabel("Force [N]"); axes[1].legend(); axes[1].grid(True)

axes[2].plot(t_rel,           y_va,      "k",  lw=1.2, label="Measured f_raw")
axes[2].plot(t_rel,           y_arx_sim, "b--", lw=1.5, label=f"{best_arx_key} ({fp_arx:.0f}%)")
axes[2].set_xlabel("Time [s]"); axes[2].set_ylabel("Force [N]")
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sysid_05_validation.png"), dpi=130); plt.show()


# SECTION 9 - STEP RESPONSE OF IDENTIFIED MODEL
# Step response = response to a sudden step input of amplitude 1 Nm.
# Because G(s) is in units N/Nm, the steady-state output is K [N/Nm].
#
# Key step-response metrics:
#   DC gain K   : steady-state output  (must equal K_static from Section 3!)
#   Rise time   : time to travel from 10% to 90% of steady state
#   Settling time: time for output to stay within ?2% of steady state
#   Overshoot % : how much the peak exceeds the steady state
#                 (only occurs when z < 1, i.e. underdamped)
#
# Analytical formula for overshoot (2nd-order underdamped):
#   OS% = 100 ? exp(??*z / ?(1?z2))
#
# If the step response overshoots too much (z < 0.5):
#   -> this plant will be hard to control; the controller must compensate.
#   -> consider adding damping hardware, or designing a controller with
#     derivative action to add virtual damping.

t_step = np.linspace(0, 5.0, 5000)
_, y_step = signal.step(sys_lti, T=t_step)

# Normalise for metric computation
y_ss   = y_step[-1]
y_norm = y_step / (y_ss + 1e-12)

OS_pct = (100 * np.exp(-np.pi * zeta_fit / np.sqrt(max(1 - zeta_fit**2, 1e-9)))
          if zeta_fit < 1.0 else 0.0)

i10 = np.searchsorted(y_norm, 0.10)
i90 = np.searchsorted(y_norm, 0.90)
t_rise = (t_step[i90] - t_step[i10]) if i90 < len(t_step) else float("nan")

settled_idx = np.where(np.abs(y_norm - 1.0) > 0.02)[0]
t_settle    = t_step[settled_idx[-1]] if len(settled_idx) > 0 else 0.0

print(f"\nStep Response Characteristics (2nd-order TF):")
print(f"  DC gain (K)     = {y_ss:.4f}  N/Nm")
print(f"  Rise time        = {t_rise*1000:.1f} ms")
print(f"  Settling time    = {t_settle*1000:.1f} ms")
print(f"  Overshoot        = {OS_pct:.1f} %")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_step * 1000, y_step, "r", lw=2, label="Step response G(s)")
ax.axhline(y_ss, color="k", ls="--", lw=1, label=f"Steady-state = {y_ss:.3f} N/Nm")
ax.axhline(y_ss * 1.02, color="gray", ls=":", lw=0.8)
ax.axhline(y_ss * 0.98, color="gray", ls=":", lw=0.8)
if not np.isnan(t_rise):
    ax.annotate("", xy=(t_step[i90]*1000, 0.9*y_ss), xytext=(t_step[i10]*1000, 0.1*y_ss),
                arrowprops=dict(arrowstyle="<->", color="blue"))
    ax.text((t_step[i10]+t_step[i90])*500, 0.5*y_ss,
            f"rise {t_rise*1000:.1f} ms", color="blue", fontsize=9, ha="center")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Force per unit torque [N/Nm]")
ax.set_title(f"Unit Step Response  (z={zeta_fit:.3f},  f?={fn_fit:.2f} Hz,  OS={OS_pct:.1f}%)")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "sysid_06_step.png"), dpi=130); plt.show()


# FINAL SUMMARY
print("\n" + "=" * 62)
print("   SYSTEM IDENTIFICATION -- FINAL SUMMARY")
print("=" * 62)
print(f"\n  Plant:  tau_cmd_Nm  -->  f_raw_N")
print(f"\n  Identified model: 2nd-order continuous transfer function")
print(f"\n            K * w?2")
print(f"  G(s) = -------------------------")
print(f"          s2 + 2*z*w?*s + w?2")
print(f"\n  Parameters:")
print(f"    K    = {K_fit:.4f}  N/Nm       (static gain, expected ~ {1/JACOBIAN:.1f})")
print(f"    w?   = {wn_fit:.4f} rad/s    = {fn_fit:.3f} Hz")
print(f"    z    = {zeta_fit:.4f}            ({'underdamped' if zeta_fit < 1 else 'overdamped'})")
print(f"\n  Numerator  : {K_fit*wn_fit**2:.4f}")
print(f"  Denominator: s2 + {2*zeta_fit*wn_fit:.4f}*s + {wn_fit**2:.4f}")
print(f"\n  Validation fit %:")
print(f"    2nd-order TF  : {fp_tf2:.1f} %")
print(f"    {best_arx_key:<14}: {fp_arx:.1f} %")
print(f"\n  DC gain check  : fitted K = {K_fit:.4f}  vs  static K = {K_static:.4f}")
print(f"  Step OS        : {OS_pct:.1f} %   rise = {t_rise*1000:.1f} ms   settle = {t_settle*1000:.1f} ms")
print(f"\n  Saved plots:")
for fname in ["sysid_01_overview.png", "sysid_02_static_gain.png",
              "sysid_03_etfe.png",     "sysid_04_bode.png",
              "sysid_05_validation.png","sysid_06_step.png"]:
    print(f"    {fname}")
print("=" * 62)
