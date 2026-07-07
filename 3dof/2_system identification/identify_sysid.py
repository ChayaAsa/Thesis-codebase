from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
import sympy as sp
from scipy.signal import butter, filtfilt

import sysid_config as cfg


# ═════════════════════════════════════════════════════════════════════════════
#  1.  Symbolic regressor   τ = Y(q, q̇, q̈) · θ
# ═════════════════════════════════════════════════════════════════════════════
def _rot(axis: str, a):
    c, s = sp.cos(a), sp.sin(a)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])           # z


def build_regressor():
    q1, q2, q3   = sp.symbols('q1 q2 q3', real=True)
    dq1, dq2, dq3 = sp.symbols('dq1 dq2 dq3', real=True)
    a1s, a2s, a3s = sp.symbols('ddq1 ddq2 ddq3', real=True)
    q   = [q1, q2, q3]
    dq  = sp.Matrix([dq1, dq2, dq3])
    ddq = sp.Matrix([a1s, a2s, a3s])

    L1, L2, L3, Gsym = sp.symbols('L1 L2 L3 G', positive=True)

    # ── forward kinematics, identical convention to rrr_robot.m ───────────────
    z = sp.Matrix([0, 0, 1])
    R01 = _rot('z', q1)
    p1  = sp.Matrix([0, 0, 0])                       # link-1 frame origin (on yaw axis)

    R_pre2 = R01 * _rot('x', -sp.pi / 2)             # fixed Rx(-90°) after +Z offset L1
    p2     = sp.Matrix([0, 0, L1])                   # shoulder (link-2 origin)
    R02    = R_pre2 * _rot('z', q2)

    R_pre3 = R02                                     # J2→J3 fixed xform is pure translation
    p3     = p2 + R02 * sp.Matrix([L2, 0, 0])        # elbow (link-3 origin)
    R03    = R02 * _rot('z', q3)

    # joint axes in world (z of the frame BEFORE each joint angle)
    a_axis = [z, R_pre2 * z, R_pre3 * z]
    R0     = [R01, R02, R03]
    p_o    = [p1, p2, p3]

    # angular velocity of each link (world), cumulative
    omega = [a_axis[0] * dq1,
             a_axis[0] * dq1 + a_axis[1] * dq2,
             a_axis[0] * dq1 + a_axis[1] * dq2 + a_axis[2] * dq3]

    # per-link inertial parameter symbols
    inertial_params, param_names = [], []
    KE, PE = sp.Integer(0), sp.Integer(0)
    grav = sp.Matrix([0, 0, -Gsym])

    for i in range(3):
        m  = sp.Symbol(f'm{i+1}', real=True)
        hx, hy, hz = sp.symbols(f'hx{i+1} hy{i+1} hz{i+1}', real=True)
        Lxx, Lyy, Lzz = sp.symbols(f'Lxx{i+1} Lyy{i+1} Lzz{i+1}', real=True)
        Lxy, Lxz, Lyz = sp.symbols(f'Lxy{i+1} Lxz{i+1} Lyz{i+1}', real=True)
        inertial_params += [m, hx, hy, hz, Lxx, Lyy, Lzz, Lxy, Lxz, Lyz]
        param_names += [f'm{i+1}', f'h x{i+1}', f'h y{i+1}', f'h z{i+1}',
                        f'Lxx{i+1}', f'Lyy{i+1}', f'Lzz{i+1}',
                        f'Lxy{i+1}', f'Lxz{i+1}', f'Lyz{i+1}']

        h = sp.Matrix([hx, hy, hz])
        I = sp.Matrix([[Lxx, Lxy, Lxz], [Lxy, Lyy, Lyz], [Lxz, Lyz, Lzz]])

        # velocities expressed in the LINK frame keep KE linear in the parameters
        v_o = p_o[i].jacobian(sp.Matrix(q)) * dq        # world origin velocity
        vL  = R0[i].T * v_o
        wL  = R0[i].T * omega[i]

        KE += (sp.Rational(1, 2) * m * (vL.T * vL)[0]
               + (vL.T * wL.cross(h))[0]
               + sp.Rational(1, 2) * (wL.T * (I * wL))[0])
        # PE = -grav·(m p_o + R0 h)   (linear in m, h)
        PE += -(grav.T * (m * p_o[i] + R0[i] * h))[0]

    # ── M, C, g  →  τ_rbd = M q̈ + C q̇ + g ─────────────────────────────────────
    M = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            M[i, j] = sp.diff(KE, dq[i], dq[j])

    C = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            cij = sp.Integer(0)
            for k in range(3):
                cij += sp.Rational(1, 2) * (sp.diff(M[i, j], q[k])
                                            + sp.diff(M[i, k], q[j])
                                            - sp.diff(M[j, k], q[i])) * dq[k]
            C[i, j] = cij

    g_vec = sp.Matrix([sp.diff(PE, q[i]) for i in range(3)])
    tau_rbd = M * ddq + C * dq + g_vec                  # 3×1, linear in inertial_params

    # regressor columns for the inertial params
    Y_in = tau_rbd.jacobian(sp.Matrix(inertial_params))    # 3×30

    # friction columns appended manually: Fv_i*q̇_i + Fc_i*sign(q̇_i) on joint i only
    Y_fr = sp.zeros(3, 6)
    for i in range(3):
        Y_fr[i, 2 * i]     = dq[i]
        Y_fr[i, 2 * i + 1] = sp.sign(dq[i])
        param_names += [f'Fv{i+1}', f'Fc{i+1}']

    Y = Y_in.row_join(Y_fr)                             # 3×36
    Y = Y.subs({L1: cfg.L1, L2: cfg.L2, L3: cfg.L3, Gsym: cfg.G})

    Yfunc = sp.lambdify((q1, q2, q3, dq1, dq2, dq3, a1s, a2s, a3s),
                        Y, modules='numpy', cse=True)
    return Yfunc, param_names


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Data processing
# ═════════════════════════════════════════════════════════════════════════════
def process_csv(path: str, cutoff_hz: float, trim_s: float):
    df = pd.read_csv(path)
    t   = df['time_s'].to_numpy(float)
    q   = df[['q1', 'q2', 'q3']].to_numpy(float)
    tau = df[['tau1', 'tau2', 'tau3']].to_numpy(float)
    return _process_arrays(t, q, tau, cutoff_hz, trim_s)


def _process_arrays(t, q, tau, cutoff_hz, trim_s):
    # resample onto an exactly-uniform grid (filtfilt / gradient assume uniform dt)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], dt)
    qu   = np.column_stack([np.interp(tu, t, q[:, j])   for j in range(3)])
    tauu = np.column_stack([np.interp(tu, t, tau[:, j]) for j in range(3)])

    # zero-phase Butterworth low-pass (no phase lag → derivatives stay aligned)
    nyq = 0.5 * fs
    wc  = min(cutoff_hz, 0.9 * nyq) / nyq
    b, a = butter(4, wc)
    qf   = np.column_stack([filtfilt(b, a, qu[:, j])   for j in range(3)])
    tauf = np.column_stack([filtfilt(b, a, tauu[:, j]) for j in range(3)])

    # differentiate filtered position for velocity and acceleration
    qd  = np.column_stack([np.gradient(qf[:, j], dt) for j in range(3)])
    qdd = np.column_stack([np.gradient(qd[:, j], dt) for j in range(3)])

    # trim filter/ramp transients off both ends
    k = int(round(trim_s / dt))
    sl = slice(k, len(tu) - k) if k > 0 else slice(None)
    return tu[sl], qf[sl], qd[sl], qdd[sl], tauf[sl]


def assemble(Yfunc, q, qd, qdd, tau):
    N = len(q)
    P = np.asarray(Yfunc(0, 0, 0, 0, 0, 0, 0, 0, 0)).shape[1]
    W = np.empty((3 * N, P))
    y = np.empty(3 * N)
    for n in range(N):
        Yn = np.asarray(Yfunc(q[n, 0], q[n, 1], q[n, 2],
                              qd[n, 0], qd[n, 1], qd[n, 2],
                              qdd[n, 0], qdd[n, 1], qdd[n, 2]), float)
        W[3 * n:3 * n + 3, :] = Yn
        y[3 * n:3 * n + 3] = tau[n, :]
    return W, y


# ═════════════════════════════════════════════════════════════════════════════
#  3.  Base-parameter identification
# ═════════════════════════════════════════════════════════════════════════════
def identify(W, y, base_tol=1e-3):
    Q, R, piv = sla.qr(W, pivoting=True, mode='economic')
    absdiag = np.abs(np.diag(R))
    rank = int(np.sum(absdiag > base_tol * absdiag[0]))
    base_cols = np.sort(piv[:rank])

    Wb = W[:, base_cols]
    theta_b, *_ = np.linalg.lstsq(Wb, y, rcond=None)

    theta = np.zeros(W.shape[1])
    theta[base_cols] = theta_b
    cond = float(np.linalg.cond(Wb))
    return theta, base_cols, cond


def per_joint_r2(W, y, theta):
    pred = W @ theta
    yj, pj = y.reshape(-1, 3), pred.reshape(-1, 3)
    r2, rms = [], []
    for j in range(3):
        resid = yj[:, j] - pj[:, j]
        ss_res = float(resid @ resid)
        ss_tot = float(((yj[:, j] - yj[:, j].mean()) ** 2).sum())
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))
        rms.append(float(np.sqrt(np.mean(resid ** 2))))
    return r2, rms


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Report / plot / save
# ═════════════════════════════════════════════════════════════════════════════
def report(theta, names, base_cols, cond, r2, rms, val=None):
    base = set(base_cols.tolist())
    print("\n" + "=" * 60)
    print("IDENTIFIED PARAMETERS  (* = identifiable base parameter)")
    print("=" * 60)
    for i, (nm, v) in enumerate(zip(names, theta)):
        mark = '*' if i in base else ' '
        if i in base or abs(v) > 1e-9:
            print(f"  {mark} {nm:8s} = {v:+.5f}")
    print("-" * 60)
    print(f"base parameters : {len(base_cols)} / {len(names)}")
    print(f"regressor cond. : {cond:.1f}   (lower = better excited; <100 is good)")
    print("train torque fit:")
    for j in range(3):
        print(f"    joint {j+1}:  R² = {r2[j]:.4f}   RMS resid = {rms[j]:.4f} N·m")
    if val is not None:
        vr2, vrms = val
        print("validation torque fit:")
        for j in range(3):
            print(f"    joint {j+1}:  R² = {vr2[j]:.4f}   RMS resid = {vrms[j]:.4f} N·m")
    print("=" * 60)


def save_json(path, theta, names, base_cols, cond, r2, rms):
    out = {
        'parameters': {nm: float(v) for nm, v in zip(names, theta)},
        'base_parameter_indices': [int(i) for i in base_cols],
        'base_parameter_names': [names[i] for i in base_cols],
        'regressor_condition_number': cond,
        'train_R2_per_joint': r2,
        'train_RMS_per_joint': rms,
    }
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"Saved parameters → {path}")


def plot_fit(t, W, y, theta, png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return
    pred = (W @ theta).reshape(-1, 3)
    meas = y.reshape(-1, 3)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for j in range(3):
        ax[j].plot(t, meas[:, j], 'k', lw=1.0, label='measured τ')
        ax[j].plot(t, pred[:, j], 'r--', lw=1.0, label='predicted τ')
        ax[j].set_ylabel(f'τ{j+1} [N·m]'); ax[j].grid(True)
    ax[0].legend(loc='upper right'); ax[-1].set_xlabel('time [s]')
    fig.suptitle('Measured vs. identified-model joint torque')
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"Saved plot → {png}")


# ═════════════════════════════════════════════════════════════════════════════
#  5.  Self-test (no hardware): fabricate data from known θ, check the recovery
# ═════════════════════════════════════════════════════════════════════════════
def selftest(Yfunc, names):
    print("[selftest] fabricating a trajectory from known parameters…")
    fs, T = 200.0, 12.0
    t = np.arange(0, T, 1 / fs)
    # smooth analytic q, q̇, q̈ (a few sines per joint)
    q   = np.column_stack([0.5 * np.sin(2 * np.pi * 0.3 * t),
                           0.4 * np.sin(2 * np.pi * 0.25 * t + 1.0),
                           0.6 * np.sin(2 * np.pi * 0.35 * t + 2.0)])
    qd  = np.column_stack([0.5 * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t),
                           0.4 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * t + 1.0),
                           0.6 * 2 * np.pi * 0.35 * np.cos(2 * np.pi * 0.35 * t + 2.0)])
    qdd = np.column_stack([-0.5 * (2 * np.pi * 0.3) ** 2 * np.sin(2 * np.pi * 0.3 * t),
                           -0.4 * (2 * np.pi * 0.25) ** 2 * np.sin(2 * np.pi * 0.25 * t + 1.0),
                           -0.6 * (2 * np.pi * 0.35) ** 2 * np.sin(2 * np.pi * 0.35 * t + 2.0)])

    theta_true = np.zeros(len(names))
    setp = lambda nm, v: theta_true.__setitem__(names.index(nm), v)
    setp('m1', 2.5); setp('m2', 1.8); setp('m3', 1.2)
    setp('h x2', 1.8 * 0.15); setp('h x3', 1.2 * 0.125)
    setp('h z1', 2.5 * 0.15)
    setp('Lzz1', 0.03); setp('Lyy2', 0.05); setp('Lyy3', 0.03)
    for j in (1, 2, 3):
        setp(f'Fv{j}', 0.30); setp(f'Fc{j}', 0.20)

    W, _ = assemble(Yfunc, q, qd, qdd, np.zeros_like(q))
    tau = (W @ theta_true).reshape(-1, 3)
    tau += 0.01 * np.random.default_rng(0).standard_normal(tau.shape)   # 10 mN·m noise

    tp, qp, qdp, qddp, taup = _process_arrays(t, q, tau, cutoff_hz=15.0, trim_s=1.0)
    Wp, yp = assemble(Yfunc, qp, qdp, qddp, taup)
    theta, base_cols, cond = identify(Wp, yp)
    r2, rms = per_joint_r2(Wp, yp, theta)
    report(theta, names, base_cols, cond, r2, rms)
    ok = all(r >= 0.99 for r in r2)
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'} - "
          f"all-joint torque R2 {'>=' if ok else '<'} 0.99")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='?', help='training CSV (default: newest in ./data)')
    ap.add_argument('--val', help='separate validation CSV')
    ap.add_argument('--cutoff', type=float, default=8.0, help='low-pass cutoff [Hz]')
    ap.add_argument('--trim', type=float, default=2.5, help='seconds trimmed each end')
    ap.add_argument('--base-tol', type=float, default=1e-3, help='base-param QR tolerance')
    ap.add_argument('--plot', action='store_true', help='save measured-vs-fit PNG')
    ap.add_argument('--selftest', action='store_true', help='verify the math, no hardware')
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("Deriving symbolic regressor…")
    Yfunc, names = build_regressor()
    print(f"  done in {time.perf_counter() - t0:.1f} s  ({len(names)} parameters)")

    if args.selftest:
        selftest(Yfunc, names)
        return

    csv = args.csv
    if csv is None:
        data = sorted((Path(__file__).parent / 'data').glob('sysid_*.csv'))
        if not data:
            ap.error("no CSV given and none found in ./data")
        csv = str(data[-1])
    print(f"Training data: {csv}")

    t, q, qd, qdd, tau = process_csv(csv, args.cutoff, args.trim)
    W, y = assemble(Yfunc, q, qd, qdd, tau)
    theta, base_cols, cond = identify(W, y, base_tol=args.base_tol)
    r2, rms = per_joint_r2(W, y, theta)

    val = None
    if args.val:
        tv, qv, qdv, qddv, tauv = process_csv(args.val, args.cutoff, args.trim)
        Wv, yv = assemble(Yfunc, qv, qdv, qddv, tauv)
        val = per_joint_r2(Wv, yv, theta)

    report(theta, names, base_cols, cond, r2, rms, val)
    save_json(Path(csv).with_suffix('.params.json'), theta, names, base_cols, cond, r2, rms)
    if args.plot:
        plot_fit(t, W, y, theta, Path(csv).with_suffix('.fit.png'))


if __name__ == '__main__':
    main()
