from __future__ import annotations

import collections
import os
import sys
import threading
import time

import numpy as np

from ATI_FTsensor.ftsensor import ftsensor
from tmotorcan import RealtimeLoop, update_all
from tmotorcan.protocol import MotorFaultError, MotorTimeoutError

HERE = os.path.dirname(os.path.abspath(__file__))
from easy_path import WS_ROOT
_CTRL_ROOT = os.path.join(WS_ROOT, '3dof', '3_control')
if _CTRL_ROOT not in sys.path:
    sys.path.insert(0, _CTRL_ROOT)

from control_config import (MOTOR_IDS, SIGN, LOCK_KP, LOCK_KD,
                             build_motors, make_bus, Dynamic, DYN_CACHE)
from helpers import KeyboardLine
from data_logger import DataLogger

dyn = Dynamic.get_or_build(DYN_CACHE)

RFOB_EXTRA_COLUMNS = [
    'f_des_N', 'f_rfob_N', 'f_sensor_N', 'f_err_N',
    'dob1_Nm', 'dob2_Nm', 'dob3_Nm',
    'rfob1_Nm', 'rfob2_Nm', 'rfob3_Nm',
    'tau_cmd1_Nm', 'tau_cmd2_Nm', 'tau_cmd3_Nm',
    'g_dob', 'g_rfob', 'dob_comp', 'force_fb',
]

# Hardware
BIAS_DURATION = 5.0
KD_FREE       = 1.0

# Init pose
INIT_Q = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad

# ATI sensor
FORCE_AXIS = 2              # 0=Fx 1=Fy 2=Fz
FORCE_SIGN = -1.0

# Push direction
PUSH_DIR = np.array([1.0, 0.0, 0.0])
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# Force setpoint
F_DES_N = 0.0

# PI gains
KP_F = 0.0
KI_F = 0.0

# Velocity damping
KD_VEL = 0.1

# ── LPF on ATI sensor (only used for display / fused mode) ───────────────────
LPF_ALPHA = 0.3

# ── DOB + RFOB bandwidths [rad/s] ─────────────────────────────────────────────
# g_RFOB > g_DOB improves stability (Sariyildiz & Ohnishi, eq. 8)
G_DOB  = 30.0
G_RFOB = 60.0

# ── Nominal joint inertia [kg·m²] — computed from dynamic model at home pose ───
# Diagonal of M(INIT_Q) gives effective link inertia per joint.
# Motor rotor inertia (not in the model) adds to the true value, so this
# naturally satisfies the RFOB stability rule: J_hat ≤ J_actual.
_M_init, _, _ = dyn.evaluate_MCG(INIT_Q, np.zeros(3))
J_N = np.diag(_M_init)
print(f"[RFOB] Nominal inertia from M(INIT_Q): {np.round(J_N, 5)} kg·m²")

# DOB compensation
# True  → add tau_dis_hat to command (disturbance rejection, increases loop gain)
# False → DOB only used for estimation (safer starting point)
USE_DOB_COMP = False

# Force feedback source
# 'rfob'   → sensorless, uses RFOB Cartesian estimate
# 'sensor' → ATI F/T sensor (same as pure_force.py)
# 'fuse'   → FUSE_WEIGHT * f_rfob + (1-FUSE_WEIGHT) * f_sensor
FORCE_FB    = 'rfob'
FUSE_WEIGHT = 0.5    # weight on RFOB when FORCE_FB='fuse'

# Safety
TORQUE_LIMIT = 3.0

# Loop rate
LOOP_HZ = 100

# Shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6

# GUI-tunable params (floats; force_fb / dob_comp encoded as 0/1/2 int floats)
ctrl_params = {
    'f_des':    float(F_DES_N),
    'kp':       float(KP_F),
    'ki':       float(KI_F),
    'kd_vel':   float(KD_VEL),
    'lpf_alpha':float(LPF_ALPHA),
    'g_dob':    float(G_DOB),
    'g_rfob':   float(G_RFOB),
    'dob_comp': float(int(USE_DOB_COMP)),   # 0 or 1
    'force_fb': float({'rfob': 0, 'sensor': 1, 'fuse': 2}[FORCE_FB]),
    'fuse_w':   float(FUSE_WEIGHT),
    'reset':    False,
}


# F/T reader thread

def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            msg = str(e)
            if 'not yet been acquired' in msg or 'samples requested' in msg:
                time.sleep(0.02)
                continue
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


# Live plot

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._frame     = 0
        self._t         = collections.deque(maxlen=self.MAX_PTS)
        self._f_des     = collections.deque(maxlen=self.MAX_PTS)
        self._f_rfob    = collections.deque(maxlen=self.MAX_PTS)
        self._f_sensor  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_l_hat = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]
        self._q         = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]

    def push(self, t: float, f_des: float, f_rfob: float, f_sensor: float,
             tau_l_hat: np.ndarray, q: np.ndarray) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_rfob.append(f_rfob)
            self._f_sensor.append(f_sensor)
            for i in range(3):
                self._tau_l_hat[i].append(float(tau_l_hat[i]))
                self._q[i].append(float(q[i]))

    def _run(self) -> None:
        import tkinter as tk
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title('3-DOF RFOB + DOB Force Control')
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=0, minsize=320)
        root.rowconfigure(0, weight=1)

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky='nsew')

        fig, (ax_f, ax_tau, ax_q) = plt.subplots(3, 1, figsize=(9, 8))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.06, hspace=0.38)
        fig.suptitle('3-DOF RFOB + DOB Force Control', fontsize=11)

        ax_f.set_ylabel('Force [N]');      ax_f.grid(True, alpha=0.3)
        ax_tau.set_ylabel('τ_contact [N·m]'); ax_tau.grid(True, alpha=0.3)
        ax_q.set_ylabel('q [rad]');         ax_q.set_xlabel('Time [s]')
        ax_q.grid(True, alpha=0.3)

        line_fdes,    = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_frfob,   = ax_f.plot([], [], 'r-',  lw=1.5, label='RFOB est.')
        line_fsen,    = ax_f.plot([], [], color='# aaaaaa', lw=1.0, alpha=0.7, label='ATI sensor')
        ax_f.legend(loc='upper left', fontsize=8)

        tau_colors = ['green', 'orange', 'purple']
        tau_lines  = [ax_tau.plot([], [], color=tau_colors[i], lw=1.5,
                                  label=f'τ_l̂{i+1}')[0] for i in range(3)]
        ax_tau.legend(loc='upper left', fontsize=8)

        q_colors = ['steelblue', 'firebrick', 'goldenrod']
        q_lines  = [ax_q.plot([], [], color=q_colors[i], lw=1.2,
                               label=f'q{i+1}')[0] for i in range(3)]
        ax_q.legend(loc='upper left', fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Control panel
        ctrl = tk.Frame(root, relief='groove', bd=2, padx=8, pady=8)
        ctrl.grid(row=0, column=1, sticky='nsew', padx=(0, 6), pady=6)
        ctrl.columnconfigure(1, weight=1)

        tk.Label(ctrl, text='Parameters',
                 font=('TkDefaultFont', 11, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 4))

        # (ctrl_key, label, clamp_lo, clamp_hi)
        _PARAMS = [
            ('f_des',    'F des  [N]',         0.0,  50.0),
            ('kp',       'Kp_f',               0.0, 100.0),
            ('ki',       'Ki_f',               0.0, 100.0),
            ('kd_vel',   'Kd_vel',             0.0,  10.0),
            ('lpf_alpha','ATI LPF α',          0.01,  1.0),
            ('g_dob',    'g_DOB [rad/s]',       1.0, 300.0),
            ('g_rfob',   'g_RFOB [rad/s]',      1.0, 600.0),
            ('dob_comp', 'DOB comp (0=off 1=on)',0.0,   1.0),
            ('force_fb', 'FB: 0=RFOB 1=ATI 2=fuse', 0.0, 2.0),
            ('fuse_w',   'Fuse w RFOB [0-1]',  0.0,   1.0),
        ]

        entries = {}
        for r, (key, lbl, lo, hi) in enumerate(_PARAMS, start=1):
            tk.Label(ctrl, text=lbl + ':', anchor='w',
                     font=('TkDefaultFont', 9)).grid(
                row=r, column=0, sticky='w', pady=2, padx=(0, 4))
            var = tk.StringVar(value=str(ctrl_params.get(key, 0.0)))
            ent = tk.Entry(ctrl, textvariable=var, width=9,
                           font=('Courier', 12), justify='right',
                           relief='solid', bd=1)
            ent.grid(row=r, column=1, sticky='ew', pady=2)
            entries[key] = (var, ent, lo, hi)

            def _apply(*_, k=key, sv=var, e=ent, clo=lo, chi=hi, rt=root):
                try:
                    v = float(sv.get())
                    v = max(clo, min(chi, v))
                    ctrl_params[k] = v
                    if k in ('f_des', 'ki'):
                        ctrl_params['reset'] = True
                    sv.set(f'{v:.4g}')
                    e.config(bg='# d4edda')
                    rt.after(600, lambda: e.config(bg='white'))
                except ValueError:
                    e.config(bg='# f8d7da')
                    rt.after(600, lambda: e.config(bg='white'))

            ent.bind('<Return>', _apply)
            ent.bind('<FocusOut>', _apply)

        next_row = len(_PARAMS) + 1

        def _apply_all():
            for k, (_sv, _e, _lo, _hi) in entries.items():
                try:
                    v = float(_sv.get())
                    v = max(_lo, min(_hi, v))
                    ctrl_params[k] = v
                    if k in ('f_des', 'ki'):
                        ctrl_params['reset'] = True
                    _sv.set(f'{v:.4g}')
                    _e.config(bg='# d4edda')
                    root.after(600, lambda e=_e: e.config(bg='white'))
                except ValueError:
                    _e.config(bg='# f8d7da')
                    root.after(600, lambda e=_e: e.config(bg='white'))

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=next_row, column=0, columnspan=2, sticky='ew', pady=6)

        tk.Button(ctrl, text='Apply All  [Enter]',
                  font=('TkDefaultFont', 10), bg='# d0e8ff',
                  relief='raised', pady=3,
                  command=_apply_all).grid(row=next_row+1, column=0, columnspan=2,
                                           sticky='ew', pady=2)

        tk.Button(ctrl, text='Reset Integrator + Observers',
                  font=('TkDefaultFont', 9), bg='# fffacd',
                  relief='raised', pady=3,
                  command=lambda: ctrl_params.__setitem__('reset', True)
                  ).grid(row=next_row+2, column=0, columnspan=2,
                         sticky='ew', pady=2)

        tk.Button(ctrl, text='E - S T O P',
                  font=('TkDefaultFont', 12, 'bold'),
                  bg='# e05050', fg='white', relief='raised', pady=5,
                  command=lambda: (Terminate.set(), root.quit())
                  ).grid(row=next_row+3, column=0, columnspan=2,
                         sticky='ew', pady=(6, 2))

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=next_row+4, column=0, columnspan=2, sticky='ew', pady=4)
        tk.Label(ctrl, text='Live status', font=('TkDefaultFont', 9, 'bold')
                 ).grid(row=next_row+5, column=0, columnspan=2, sticky='w')
        lbl_force = tk.Label(ctrl, text='--', anchor='w',
                             font=('Courier', 9), justify='left')
        lbl_force.grid(row=next_row+6, column=0, columnspan=2, sticky='w', pady=1)
        lbl_q = tk.Label(ctrl, text='--', anchor='w',
                         font=('Courier', 9), justify='left')
        lbl_q.grid(row=next_row+7, column=0, columnspan=2, sticky='w', pady=1)

        # Periodic update
        def _update():
            with self._lock:
                if len(self._t) < 2:
                    root.after(self.UPDATE_MS, _update)
                    return
                t        = list(self._t)
                f_des    = list(self._f_des)
                f_rfob   = list(self._f_rfob)
                f_sensor = list(self._f_sensor)
                tau_lhats = [list(d) for d in self._tau_l_hat]
                qs        = [list(d) for d in self._q]

            t_now = t[-1];  t_lo = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_frfob.set_data(t, f_rfob)
            line_fsen.set_data(t, f_sensor)
            for i in range(3):
                tau_lines[i].set_data(t, tau_lhats[i])
                q_lines[i].set_data(t, qs[i])
            for ax in (ax_f, ax_tau, ax_q):
                ax.set_xlim(t_lo, t_now + 0.1)

            self._frame += 1
            if self._frame % 5 == 0:
                def _ylim(ax, *series):
                    vals = [v for s in series for v in s]
                    if not vals:
                        return
                    lo, hi = min(vals), max(vals)
                    pad = max(0.3, (hi - lo) * 0.15)
                    ax.set_ylim(lo - pad, hi + pad)
                _ylim(ax_f, f_des, f_rfob, f_sensor)
                _ylim(ax_tau, *tau_lhats)
                _ylim(ax_q, *qs)

            canvas.draw_idle()

            lbl_force.config(
                text=f'RFOB: {f_rfob[-1]:+.3f} N\n'
                     f'ATI:  {f_sensor[-1]:+.3f} N\n'
                     f'Des:  {f_des[-1]:+.3f} N')
            lbl_q.config(
                text='\n'.join(f'q{i+1}: {qs[i][-1]:+.3f} rad' for i in range(3)))

            if Terminate.is_set():
                root.quit()
                return
            root.after(self.UPDATE_MS, _update)

        root.after(self.UPDATE_MS, _update)
        root.protocol('WM_DELETE_WINDOW', lambda: (Terminate.set(), root.quit()))
        root.mainloop()


# Startup sequence

def setup(motors: list, sensor: ftsensor) -> bool:
    try:
        dt_boot = 1.0 / LOOP_HZ
        print("\n--- STARTUP ---")
        print(f"Step 1: Move arm to INIT_Q = {np.round(INIT_Q, 3)} rad, then press Enter.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for m in motors:
                m.cmd.kp = 0.0;  m.cmd.kd = KD_FREE
                m.cmd.position = m.cmd.velocity = m.cmd.torque = 0.0
            update_all(motors, timeout=5.0)
            time.sleep(dt_boot)

        for m in motors:
            m.zero()
        print(f"  Motors zeroed.")

        print("Step 2: Holding init pose. Press Enter to bias ATI sensor.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for i, mid in enumerate(MOTOR_IDS):
                motors[i].cmd.kp = LOCK_KP[mid];  motors[i].cmd.kd = LOCK_KD[mid]
                motors[i].cmd.position = motors[i].cmd.velocity = motors[i].cmd.torque = 0.0
            update_all(motors, timeout=5.0)
            time.sleep(dt_boot)
        print(f"  Biasing ATI for {BIAS_DURATION:.1f}s ...")
        sensor.reBias(duration=BIAS_DURATION)
        print("  Bias complete.")

        print("Step 3: Press Enter to START RFOB force control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for i, mid in enumerate(MOTOR_IDS):
                motors[i].cmd.kp = LOCK_KP[mid];  motors[i].cmd.kd = LOCK_KD[mid]
                motors[i].cmd.position = motors[i].cmd.velocity = motors[i].cmd.torque = 0.0
            update_all(motors, timeout=5.0)
            time.sleep(dt_boot)
        print(
            f"\nConfig: F_des={F_DES_N}N  Kp={KP_F}  Ki={KI_F}\n"
            f"  g_DOB={G_DOB} rad/s  g_RFOB={G_RFOB} rad/s\n"
            f"  J_N={J_N} kg·m²  DOB_comp={USE_DOB_COMP}  FB={FORCE_FB}\n"
            f"  Torque limit ±{TORQUE_LIMIT} N·m\n"
        )
        return True
    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")
        return False


# Control loop

def loop(motors: list, plotter: PlotThread, logger: DataLogger | None) -> None:
    try:
        for m in motors:
            m.cmd.kp = m.cmd.kd = m.cmd.position = m.cmd.velocity = m.cmd.torque = 0.0

        # Observer states
        dob_state    = np.zeros(3)   # tau_dis_hat (total disturbance per joint)
        rfob_state   = np.zeros(3)   # tau_l_hat (contact torque per joint)
        tau_cmd_prev = np.zeros(3)   # torque sent last step (kinematic frame)

        # Controller states
        integral_F    = 0.0
        f_sensor_filt = FORCE_SIGN * ft_latest[FORCE_AXIS]

        dt_nom  = 1.0 / LOOP_HZ
        prev_t  = None
        rl      = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        try:
            for t in rl:
                if Terminate.is_set():
                    rl.stop()
                    break

                # Live params
                f_des      = ctrl_params['f_des']
                kp         = ctrl_params['kp']
                ki         = ctrl_params['ki']
                kd_vel     = ctrl_params['kd_vel']
                lpf_alpha  = ctrl_params['lpf_alpha']
                g_dob      = ctrl_params['g_dob']
                g_rfob     = ctrl_params['g_rfob']
                dob_comp   = ctrl_params['dob_comp'] > 0.5
                force_fb   = int(round(ctrl_params['force_fb']))   # 0/1/2
                fuse_w     = ctrl_params['fuse_w']

                if ctrl_params['reset']:
                    integral_F    = 0.0
                    dob_state[:]  = 0.0
                    rfob_state[:] = 0.0
                    tau_cmd_prev[:] = 0.0
                    f_sensor_filt = FORCE_SIGN * ft_latest[FORCE_AXIS]
                    ctrl_params['reset'] = False

                dt     = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Joint state
                q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                                  for i, mid in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[mid] * motors[i].state.velocity
                                  for i, mid in enumerate(MOTOR_IDS)])

                # ── Model: gravity + Jacobian ─────────────────────────────
                _, _, G_torq = dyn.evaluate_MCG(q, np.zeros(3))
                Jv = dyn.evaluate_jacobian(q)[:3, :]   # 3×3 linear-velocity rows

                # DOB update (residual disturbance per joint)
                # Subtract gravity before filtering so dob_state captures only
                # friction + contact + model error — NOT gravity.  This prevents
                # double-adding gravity when DOB compensation is enabled
                # (tau_des already contains G_torq explicitly).
                alpha_dob = np.exp(-g_dob * dt)
                u_dob     = (tau_cmd_prev - G_torq) - J_N * g_dob * qdot
                dob_state = alpha_dob * dob_state + (1.0 - alpha_dob) * u_dob

                # RFOB update (contact torque estimate per joint)
                # Same as DOB but subtracts known gravity first, leaving contact only
                alpha_rfob = np.exp(-g_rfob * dt)
                u_rfob     = tau_cmd_prev - G_torq - J_N * g_rfob * qdot
                rfob_state = alpha_rfob * rfob_state + (1.0 - alpha_rfob) * u_rfob

                # Cartesian force from RFOB
                # F_hat = J^{-T} @ tau_l_hat  (inverse transpose mapping)
                cond = np.linalg.cond(Jv)
                if cond < 1e6:
                    F_cart_hat = np.linalg.solve(Jv.T, rfob_state)
                else:
                    F_cart_hat = np.linalg.pinv(Jv.T) @ rfob_state
                f_hat = float(PUSH_DIR @ F_cart_hat)

                # ATI sensor (filtered)
                f_raw         = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_sensor_filt = lpf_alpha * f_raw + (1.0 - lpf_alpha) * f_sensor_filt

                # Force feedback selection
                if force_fb == 0:
                    f_measured = f_hat
                elif force_fb == 1:
                    f_measured = f_sensor_filt
                else:
                    f_measured = fuse_w * f_hat + (1.0 - fuse_w) * f_sensor_filt

                # PI force controller
                f_err      = f_des - f_measured
                integral_F += f_err * dt
                if ki > 1e-12:
                    windup     = TORQUE_LIMIT / ki
                    integral_F = float(np.clip(integral_F, -windup, windup))

                F_ctrl_scalar = kp * f_err + ki * integral_F
                tau_force     = Jv.T @ (PUSH_DIR * F_ctrl_scalar)

                # Assemble total torque command
                tau_des = tau_force + G_torq          # force ctrl + gravity comp
                if dob_comp:
                    tau_des = tau_des + dob_state      # DOB disturbance rejection

                tau_cmd = tau_des - kd_vel * qdot
                tau_cmd = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

                # Motor command
                for i, mid in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp     = 0.0
                    motors[i].cmd.kd     = 0.0
                    motors[i].cmd.torque = float(SIGN[mid] * tau_cmd[i]) * rl.fade

                update_all(motors, timeout=2.0)

                # Store actual sent torque for DOB next step (kinematic frame)
                tau_cmd_prev = tau_cmd * rl.fade

                # Telemetry
                plotter.push(t, f_des, f_hat, f_sensor_filt, rfob_state, q)

                tau_meas = np.array([SIGN[mid] * motors[i].state.torque
                                      for i, mid in enumerate(MOTOR_IDS)])
                if logger is not None:
                    logger.log(
                        time_s=t,
                        q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                        qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                        tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1], tau_meas3_Nm=tau_meas[2],
                        f_des_N=f_des, f_rfob_N=f_hat, f_sensor_N=f_sensor_filt, f_err_N=f_err,
                        dob1_Nm=dob_state[0], dob2_Nm=dob_state[1], dob3_Nm=dob_state[2],
                        rfob1_Nm=rfob_state[0], rfob2_Nm=rfob_state[1], rfob3_Nm=rfob_state[2],
                        tau_cmd1_Nm=tau_cmd[0], tau_cmd2_Nm=tau_cmd[1], tau_cmd3_Nm=tau_cmd[2],
                        g_dob=g_dob, g_rfob=g_rfob,
                        dob_comp=float(dob_comp), force_fb=float(force_fb),
                    )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"des={f_des:+.2f}N  RFOB={f_hat:+.3f}N  ATI={f_sensor_filt:+.3f}N  "
                        f"err={f_err:+.3f}  "
                        f"τ=[{tau_cmd[0]:+.2f},{tau_cmd[1]:+.2f},{tau_cmd[2]:+.2f}]N·m  "
                        f"T={temps}"
                    )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


# Entry point

def main() -> None:
    print("Initialising ATI F/T sensor ...")
    sensor = ftsensor()
    sensor.start_task()
    time.sleep(0.5)

    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("F/T background reader started.")

    bus    = make_bus()
    motors = build_motors(bus)

    bus.drain()
    time.sleep(0.5)

    for m in motors:
        m.enable()
    update_all(motors, timeout=2.0)

    plotter = PlotThread()
    logger  = DataLogger('rfob_force', RFOB_EXTRA_COLUMNS, directory=HERE)

    complete = setup(motors, sensor)
    if not complete:
        Terminate.set()
        logger.close()
        for m in motors:
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        print("Startup failed.")
        return

    ctrl_thread = threading.Thread(
        target=loop, args=(motors, plotter, logger),
        daemon=True, name='Control',
    )
    ctrl_thread.start()

    try:
        plotter._run()
    except KeyboardInterrupt:
        Terminate.set()
    finally:
        Terminate.set()
        ctrl_thread.join(timeout=3.0)
        logger.close()
        for m in motors:
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
