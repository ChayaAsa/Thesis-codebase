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

# Columns specific to the cascade torque controller.
PURE_TORQUE_EXTRA_COLUMNS = [
    'f_des_N', 'f_raw_N', 'f_filt_N', 'f_err_N',
    'tau1_des_Nm', 'tau2_des_Nm', 'tau3_des_Nm',
    'tau1_meas_filt_Nm', 'tau2_meas_filt_Nm', 'tau3_meas_filt_Nm',
    'tau1_cmd_Nm', 'tau2_cmd_Nm', 'tau3_cmd_Nm',
    'kp_f', 'ki_f',
    'kp_t1', 'kp_t2', 'kp_t3',
    'ki_t1', 'ki_t2', 'ki_t3',
    'lpf_alpha', 'lpf_alpha_t',
]

# Hardware
BIAS_DURATION = 5.0
KD_FREE       = 1.0   # velocity damping while user positions arm [N·m·s/rad]

# ── Init pose (DH angles when arm is at the zeroed reference position) ────────
# All q values during control are measured relative to this pose.
INIT_Q = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad

# ATI sensor axis
FORCE_AXIS  = 2
FORCE_SIGN  = -1.0

# Push direction
PUSH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# Force setpoint
F_DES_N = 0.0

# ── Outer loop — force PI gains ───────────────────────────────────────────────
KP_F        = 0.0    # force proportional gain [N·m / N]
KI_F        = 0.0    # force integral gain [N·m / (N·s)]
LPF_ALPHA   = 0.3    # IIR for ATI force measurement (1.0=raw, 0.1=heavy)

# ── Inner loop — per-joint torque PI gains ────────────────────────────────────
KP_T        = [0.0, 0.0, 0.0]   # proportional gain [dimensionless] [j1, j2, j3]
KI_T        = [0.0, 0.0, 0.0]   # integral gain [1/s] [j1, j2, j3]
LPF_ALPHA_T = 0.5                # IIR for motor torque measurement

# Gravity compensation
GRAV_COMP   = True   # set True to add gravity feedforward from Dynamic model

# Safety
TORQUE_LIMIT = 3.0   # per-joint [N·m]

# Timing
LOOP_HZ = 100

# Shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6

ctrl_params = {
    'f_des':       float(F_DES_N),
    'kp_f':        float(KP_F),
    'ki_f':        float(KI_F),
    'kp_t1':       float(KP_T[0]),
    'kp_t2':       float(KP_T[1]),
    'kp_t3':       float(KP_T[2]),
    'ki_t1':       float(KI_T[0]),
    'ki_t2':       float(KI_T[1]),
    'ki_t3':       float(KI_T[2]),
    'lpf_alpha':   float(LPF_ALPHA),
    'lpf_alpha_t': float(LPF_ALPHA_T),
    'reset':       False,
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


# Live plot (4 panels + 3-row TextBox control panel)

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._frame   = 0
        self._t       = collections.deque(maxlen=self.MAX_PTS)
        self._f_des   = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw   = collections.deque(maxlen=self.MAX_PTS)
        self._f_filt  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_des = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]
        self._tau_cmd = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]
        self._q       = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]

    def push(self, t: float, f_des: float, f_raw: float, f_filt: float,
             tau_des: np.ndarray, tau_cmd: np.ndarray, q: np.ndarray) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._f_filt.append(f_filt)
            for i in range(3):
                self._tau_des[i].append(float(tau_des[i]))
                self._tau_cmd[i].append(float(tau_cmd[i]))
                self._q[i].append(float(q[i]))

    def _run(self) -> None:
        import tkinter as tk
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title('3-DOF Pure Torque Control')
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=0, minsize=320)
        root.rowconfigure(0, weight=1)

        # ── Left: matplotlib figure ───────────────────────────────────────
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky='nsew')

        fig, axes = plt.subplots(4, 1, figsize=(9, 10))
        ax_f, ax_td, ax_tc, ax_q = axes
        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.06, hspace=0.35)
        fig.suptitle('3-DOF Pure Torque Control (cascade)', fontsize=11)

        for ax, ylabel in zip(axes, ['Force [N]', 'tau_des [N·m]',
                                      'tau_cmd [N·m]', 'q [rad]']):
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        ax_q.set_xlabel('Time [s]')

        line_fdes,  = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_fraw,  = ax_f.plot([], [], color='# fd9999', lw=0.8, alpha=0.6, label='F raw')
        line_ffilt, = ax_f.plot([], [], 'r-',  lw=1.5, label='F filt')
        ax_f.legend(loc='upper left', fontsize=8)

        colors = ['green', 'orange', 'purple']
        td_lines = [ax_td.plot([], [], color=colors[i], lw=1.2, ls='--',
                               label=f'des{i+1}')[0] for i in range(3)]
        tc_lines = [ax_tc.plot([], [], color=colors[i], lw=1.5,
                               label=f'cmd{i+1}')[0] for i in range(3)]
        q_lines  = [ax_q.plot([], [], color=['steelblue', 'firebrick', 'goldenrod'][i],
                              lw=1.2, label=f'q{i+1}')[0] for i in range(3)]
        for ax in (ax_td, ax_tc, ax_q):
            ax.legend(loc='upper left', fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # ── Right: control panel ──────────────────────────────────────────
        ctrl = tk.Frame(root, relief='groove', bd=2, padx=8, pady=8)
        ctrl.grid(row=0, column=1, sticky='nsew', padx=(0, 6), pady=6)
        ctrl.columnconfigure(1, weight=1)

        tk.Label(ctrl, text='Parameters',
                 font=('TkDefaultFont', 11, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 4))

        _KI_RESET_KEYS = {'f_des', 'ki_f', 'ki_t1', 'ki_t2', 'ki_t3'}

        _OUTER_PARAMS = [
            ('f_des',       'F des [N]',   0.0,  50.0),
            ('kp_f',        'Kp_f',        0.0, 100.0),
            ('ki_f',        'Ki_f',        0.0, 100.0),
            ('lpf_alpha',   'LPF force',   0.01,  1.0),
            ('lpf_alpha_t', 'LPF torque',  0.01,  1.0),
        ]
        _INNER_PARAMS = [
            ('kp_t1', 'Kp_t J1', 0.0, 20.0),
            ('kp_t2', 'Kp_t J2', 0.0, 20.0),
            ('kp_t3', 'Kp_t J3', 0.0, 20.0),
            ('ki_t1', 'Ki_t J1', 0.0, 20.0),
            ('ki_t2', 'Ki_t J2', 0.0, 20.0),
            ('ki_t3', 'Ki_t J3', 0.0, 20.0),
        ]

        entries = {}
        r = 1

        tk.Label(ctrl, text='── Outer Loop ──',
                 font=('TkDefaultFont', 9, 'italic'), fg='gray').grid(
            row=r, column=0, columnspan=2, pady=(4, 2))
        r += 1

        for key, lbl, lo, hi in _OUTER_PARAMS:
            tk.Label(ctrl, text=lbl + ':', anchor='w',
                     font=('TkDefaultFont', 10)).grid(
                row=r, column=0, sticky='w', pady=3, padx=(0, 6))
            var = tk.StringVar(value=str(ctrl_params.get(key, 0.0)))
            ent = tk.Entry(ctrl, textvariable=var, width=10,
                           font=('Courier', 13), justify='right',
                           relief='solid', bd=1)
            ent.grid(row=r, column=1, sticky='ew', pady=3)
            entries[key] = (var, ent, lo, hi)
            r += 1

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=r, column=0, columnspan=2, sticky='ew', pady=4)
        r += 1
        tk.Label(ctrl, text='── Inner Loop ──',
                 font=('TkDefaultFont', 9, 'italic'), fg='gray').grid(
            row=r, column=0, columnspan=2, pady=(2, 2))
        r += 1

        for key, lbl, lo, hi in _INNER_PARAMS:
            tk.Label(ctrl, text=lbl + ':', anchor='w',
                     font=('TkDefaultFont', 10)).grid(
                row=r, column=0, sticky='w', pady=3, padx=(0, 6))
            var = tk.StringVar(value=str(ctrl_params.get(key, 0.0)))
            ent = tk.Entry(ctrl, textvariable=var, width=10,
                           font=('Courier', 13), justify='right',
                           relief='solid', bd=1)
            ent.grid(row=r, column=1, sticky='ew', pady=3)
            entries[key] = (var, ent, lo, hi)
            r += 1

        def _apply_entry(key, var, ent, lo, hi):
            try:
                v = float(var.get())
                v = max(lo, min(hi, v))
                ctrl_params[key] = v
                if key in _KI_RESET_KEYS:
                    ctrl_params['reset'] = True
                var.set(f'{v:.4g}')
                ent.config(bg='# d4edda')
                root.after(600, lambda: ent.config(bg='white'))
            except ValueError:
                ent.config(bg='# f8d7da')
                root.after(600, lambda: ent.config(bg='white'))

        for key, (var, ent, lo, hi) in entries.items():
            ent.bind('<Return>',
                     lambda _, k=key, v=var, e=ent, lo=lo, hi=hi:
                         _apply_entry(k, v, e, lo, hi))
            ent.bind('<FocusOut>',
                     lambda _, k=key, v=var, e=ent, lo=lo, hi=hi:
                         _apply_entry(k, v, e, lo, hi))

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=r, column=0, columnspan=2, sticky='ew', pady=6)
        r += 1

        def _apply_all():
            for k, (sv, e, lo, hi) in entries.items():
                try:
                    v = float(sv.get())
                    v = max(lo, min(hi, v))
                    ctrl_params[k] = v
                    if k in _KI_RESET_KEYS:
                        ctrl_params['reset'] = True
                    sv.set(f'{v:.4g}')
                    e.config(bg='# d4edda')
                    root.after(600, lambda e=e: e.config(bg='white'))
                except ValueError:
                    e.config(bg='# f8d7da')
                    root.after(600, lambda e=e: e.config(bg='white'))

        tk.Button(ctrl, text='Apply All  [Enter]',
                  font=('TkDefaultFont', 10), bg='# d0e8ff',
                  relief='raised', pady=4,
                  command=_apply_all).grid(row=r, column=0, columnspan=2,
                                           sticky='ew', pady=2)
        r += 1

        tk.Button(ctrl, text='Reset Integrator',
                  font=('TkDefaultFont', 10), bg='# fffacd',
                  relief='raised', pady=4,
                  command=lambda: ctrl_params.__setitem__('reset', True)
                  ).grid(row=r, column=0, columnspan=2, sticky='ew', pady=2)
        r += 1

        tk.Button(ctrl, text='E - S T O P',
                  font=('TkDefaultFont', 12, 'bold'),
                  bg='# e05050', fg='white', relief='raised', pady=6,
                  command=lambda: (Terminate.set(), root.quit())
                  ).grid(row=r, column=0, columnspan=2,
                         sticky='ew', pady=(6, 2))
        r += 1

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=r, column=0, columnspan=2, sticky='ew', pady=6)
        r += 1
        tk.Label(ctrl, text='Status', font=('TkDefaultFont', 10, 'bold')
                 ).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1
        lbl_force = tk.Label(ctrl, text='F: --', anchor='w',
                             font=('Courier', 10), justify='left')
        lbl_force.grid(row=r, column=0, columnspan=2, sticky='w', pady=1)
        r += 1
        lbl_tau = tk.Label(ctrl, text='tau: --', anchor='w',
                           font=('Courier', 10), justify='left')
        lbl_tau.grid(row=r, column=0, columnspan=2, sticky='w', pady=1)
        r += 1
        lbl_q = tk.Label(ctrl, text='q: --', anchor='w',
                         font=('Courier', 10), justify='left')
        lbl_q.grid(row=r, column=0, columnspan=2, sticky='w', pady=1)

        # Periodic plot update
        def _update():
            with self._lock:
                if len(self._t) < 2:
                    root.after(self.UPDATE_MS, _update)
                    return
                t      = list(self._t)
                f_des  = list(self._f_des)
                f_raw  = list(self._f_raw)
                f_filt = list(self._f_filt)
                t_des  = [list(d) for d in self._tau_des]
                t_cmd  = [list(d) for d in self._tau_cmd]
                qs     = [list(d) for d in self._q]

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_ffilt.set_data(t, f_filt)
            for i in range(3):
                td_lines[i].set_data(t, t_des[i])
                tc_lines[i].set_data(t, t_cmd[i])
                q_lines[i].set_data(t, qs[i])
            for ax in axes:
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
                _ylim(ax_f,  f_des, f_raw, f_filt)
                _ylim(ax_td, *t_des)
                _ylim(ax_tc, *t_cmd)
                _ylim(ax_q,  *qs)

            canvas.draw_idle()

            lbl_force.config(
                text=f'F filt:  {f_filt[-1]:+.3f} N\n'
                     f'F des:   {f_des[-1]:+.3f} N')
            lbl_tau.config(
                text='tau_cmd:\n' + '\n'.join(
                    f'  J{i+1}: {t_cmd[i][-1]:+.2f} N·m' for i in range(3)))
            lbl_q.config(
                text='\n'.join(
                    f'q{i+1}: {qs[i][-1]:+.3f} rad' for i in range(3)))

            if Terminate.is_set():
                root.quit()
                return
            root.after(self.UPDATE_MS, _update)

        root.after(self.UPDATE_MS, _update)
        root.protocol('WM_DELETE_WINDOW', lambda: (Terminate.set(), root.quit()))
        root.mainloop()


# Control worker

def _control_worker(bus, motors: list,
                    plotter: PlotThread, logger: DataLogger,
                    sensor: ftsensor) -> None:
    try:
        bus.drain()
        time.sleep(0.5)

        dt_boot = 1.0 / LOOP_HZ

        print("\n--- STARTUP ---")
        print(f"Step 1: Move arm to INIT_Q = {np.round(INIT_Q, 3)} rad, then press Enter to zero motors.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for m in motors:
                m.cmd.kp       = 0.0
                m.cmd.kd       = KD_FREE
                m.cmd.position = 0.0
                m.cmd.velocity = 0.0
                m.cmd.torque   = 0.0
            update_all(motors)
            time.sleep(dt_boot)

        for m in motors:
            m.zero()
        print(f"  Motors zeroed at INIT_Q = {np.round(INIT_Q, 4)} rad.")

        print("Step 2: Arm holding init pose. Press Enter to bias ATI sensor.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for i, mid in enumerate(MOTOR_IDS):
                motors[i].cmd.kp       = LOCK_KP[mid]
                motors[i].cmd.kd       = LOCK_KD[mid]
                motors[i].cmd.position = 0.0
                motors[i].cmd.velocity = 0.0;  motors[i].cmd.torque = 0.0
            update_all(motors)
            time.sleep(dt_boot)
        print(f"  Biasing ATI for {BIAS_DURATION:.1f}s (keep sensor unloaded) ...")
        sensor.reBias(duration=BIAS_DURATION)
        print("  Bias complete.")

        print("Step 3: Press Enter to START force control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for i, mid in enumerate(MOTOR_IDS):
                motors[i].cmd.kp       = LOCK_KP[mid]
                motors[i].cmd.kd       = LOCK_KD[mid]
                motors[i].cmd.position = 0.0
                motors[i].cmd.velocity = 0.0;  motors[i].cmd.torque = 0.0
            update_all(motors)
            time.sleep(dt_boot)
        print("  Force control starting!\n")

        print(
            f"Config: Outer Kp_f={KP_F}  Ki_f={KI_F}  LPF={LPF_ALPHA}\n"
            f"  Inner: Kp_t={KP_T}  Ki_t={KI_T}  LPF_t={LPF_ALPHA_T}\n"
            f"  Torque limit: +/-{TORQUE_LIMIT} N·m  "
            f"  Gravity comp: {'ON' if GRAV_COMP else 'OFF'}\n"
            f"  Tip: set Kp_t=Ki_t=[0,0,0] to verify outer loop == pure_force\n"
            f"Press Ctrl+C or close plot to stop.\n"
        )

        for m in motors:
            m.cmd.kp       = 0.0
            m.cmd.kd       = 0.0
            m.cmd.position = 0.0
            m.cmd.velocity = 0.0
            m.cmd.torque   = 0.0

        # Integrator states
        integral_F    = 0.0
        integral_t    = np.zeros(3)
        tau_meas_filt = np.zeros(3)
        f_filt        = FORCE_SIGN * ft_latest[FORCE_AXIS]

        dt_nom = 1.0 / LOOP_HZ
        prev_t: float | None = None

        rl          = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        try:
            for t in rl:
                if Terminate.is_set():
                    rl.stop()
                    break

                f_des   = ctrl_params['f_des']
                kp_f    = ctrl_params['kp_f']
                ki_f    = ctrl_params['ki_f']
                kp_t    = np.array([ctrl_params['kp_t1'],
                                    ctrl_params['kp_t2'],
                                    ctrl_params['kp_t3']])
                ki_t    = np.array([ctrl_params['ki_t1'],
                                    ctrl_params['ki_t2'],
                                    ctrl_params['ki_t3']])
                alpha   = ctrl_params['lpf_alpha']
                alpha_t = ctrl_params['lpf_alpha_t']

                if ctrl_params['reset']:
                    integral_F    = 0.0
                    integral_t[:] = 0.0
                    f_filt        = FORCE_SIGN * ft_latest[FORCE_AXIS]
                    ctrl_params['reset'] = False

                dt = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Force measurement
                f_raw  = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_filt = alpha * f_raw + (1.0 - alpha) * f_filt
                f_err  = f_des - f_filt

                # ── Outer loop: force PI -> desired joint torques ─────────────
                integral_F += f_err * dt
                if ki_f > 1e-12:
                    windup     = TORQUE_LIMIT / ki_f
                    integral_F = float(np.clip(integral_F, -windup, windup))
                F_ctrl  = f_des + kp_f * f_err + ki_f * integral_F
                F_vec   = PUSH_DIR * F_ctrl

                q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                                  for i, mid in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[mid] * motors[i].state.velocity
                                  for i, mid in enumerate(MOTOR_IDS)])

                Jv      = dyn.evaluate_jacobian(q)[:3, :]   # 3×3 linear rows
                tau_des = Jv.T @ F_vec

                if GRAV_COMP:
                    _, _, G = dyn.evaluate_MCG(q, np.zeros(3))
                    tau_des = tau_des + G

                # ── Inner loop: per-joint torque PI ───────────────────────────
                for i, mid in enumerate(MOTOR_IDS):
                    raw_tau = SIGN[mid] * motors[i].state.torque
                    tau_meas_filt[i] = (alpha_t * raw_tau
                                        + (1.0 - alpha_t) * tau_meas_filt[i])

                tau_err    = tau_des - tau_meas_filt
                integral_t += tau_err * dt
                for i in range(3):
                    if ki_t[i] > 1e-12:
                        integral_t[i] = float(np.clip(integral_t[i],
                                                       -TORQUE_LIMIT / ki_t[i],
                                                        TORQUE_LIMIT / ki_t[i]))

                tau_cmd = tau_des + kp_t * tau_err + ki_t * integral_t
                tau_cmd = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

                # Motor command
                for i, mid in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp     = 0.0
                    motors[i].cmd.kd     = 0.0
                    motors[i].cmd.torque = float(SIGN[mid] * tau_cmd[i]) * rl.fade

                update_all(motors)

                plotter.push(t, f_des, f_raw, f_filt, tau_des, tau_cmd, q)
                # logger.log(
                #     time_s=t,
                #     q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                #     qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                #     tau_meas1_Nm=tau_meas_filt[0], tau_meas2_Nm=tau_meas_filt[1], tau_meas3_Nm=tau_meas_filt[2],
                #     f_des_N=f_des, f_raw_N=f_raw, f_filt_N=f_filt, f_err_N=f_err,
                #     tau1_des_Nm=tau_des[0], tau2_des_Nm=tau_des[1], tau3_des_Nm=tau_des[2],
                #     tau1_meas_filt_Nm=tau_meas_filt[0], tau2_meas_filt_Nm=tau_meas_filt[1], tau3_meas_filt_Nm=tau_meas_filt[2],
                #     tau1_cmd_Nm=tau_cmd[0], tau2_cmd_Nm=tau_cmd[1], tau3_cmd_Nm=tau_cmd[2],
                #     kp_f=kp_f, ki_f=ki_f,
                #     kp_t1=kp_t[0], kp_t2=kp_t[1], kp_t3=kp_t[2],
                #     ki_t1=ki_t[0], ki_t2=ki_t[1], ki_t3=ki_t[2],
                #     lpf_alpha=alpha, lpf_alpha_t=alpha_t,
                # )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"F={f_filt:+.3f}N(des={f_des:+.2f})  "
                        f"tau_des=[{tau_des[0]:+.2f},{tau_des[1]:+.2f},{tau_des[2]:+.2f}]  "
                        f"tau_cmd=[{tau_cmd[0]:+.2f},{tau_cmd[1]:+.2f},{tau_cmd[2]:+.2f}]N·m  "
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
    for m in motors:
        m.enable()

    plotter = PlotThread()
    # logger  = DataLogger('pure_torque', PURE_TORQUE_EXTRA_COLUMNS, directory=HERE)
    logger  = None

    ctrl_thread = threading.Thread(
        target=_control_worker, args=(bus, motors, plotter, logger, sensor),
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
        # logger.close()
        for m in motors:
            try:
                m.coast()
                m.disable()
            except Exception:
                pass
        try:
            sensor.stop_task()
        except Exception:
            pass
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
