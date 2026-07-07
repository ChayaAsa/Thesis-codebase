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

# Columns specific to the pure-force PI controller.
# Add to this list + pass the kwarg in logger.log() — nothing else needed.
PURE_FORCE_EXTRA_COLUMNS = [
    'f_des_N', 'f_raw_N', 'f_filt_N', 'f_err_N',
    'tau1_des_Nm', 'tau2_des_Nm', 'tau3_des_Nm',
    'kp_f', 'ki_f', 'lpf_alpha',
]

# Hardware
BIAS_DURATION = 5.0           # seconds for ATI bias (sensor unloaded)
KD_FREE       = 1.0           # velocity damping while user positions arm [N·m·s/rad]

# ── Init pose (DH angles when arm is at the zeroed reference position) ────────
# All q values during control are measured relative to this pose.
INIT_Q = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad

# ATI sensor axis selection
FORCE_AXIS   = 2              # 0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz
FORCE_SIGN   = -1.0           # negate so that push reads positive

# Push direction (unit vector in world frame)
PUSH_DIR = np.array([1, 0, 0], dtype=float)
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# Force setpoint
F_DES_N      = 0.0            # desired contact force [N] (live-tunable)

# Force PI gains
KP_F         = 0.0            # proportional gain [N·m / N]
KI_F         = 0.0            # integral gain [N·m / (N·s)]

# Stability terms
KD_VEL       = 0.1            # joint velocity damping [N·m·s/rad] applied to all joints

# Gravity compensation
GRAV_COMP    = True           # set True to add gravity feedforward

# Signal processing
LPF_ALPHA    = 0.3            # IIR: y = alpha*x + (1-alpha)*y_prev

# Safety
TORQUE_LIMIT = 3.0            # per-joint hard cap [N·m]

# Timing
LOOP_HZ      = 100

# Shared state
Terminate  = threading.Event()
ft_latest  = [0.0] * 6       # [Fx, Fy, Fz, Tx, Ty, Tz]

ctrl_params = {
    'f_des':     float(F_DES_N),
    'kp':        float(KP_F),
    'ki':        float(KI_F),
    'kd_vel':    float(KD_VEL),
    'lpf_alpha': float(LPF_ALPHA),
    'reset':     False,
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


# Live plot (tkinter + matplotlib, 3 panels + control panel)

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._frame  = 0
        self._t      = collections.deque(maxlen=self.MAX_PTS)
        self._f_des  = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw  = collections.deque(maxlen=self.MAX_PTS)
        self._f_filt = collections.deque(maxlen=self.MAX_PTS)
        self._tau    = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]
        self._q      = [collections.deque(maxlen=self.MAX_PTS) for _ in range(3)]

    def push(self, t: float, f_des: float, f_raw: float, f_filt: float,
             tau: np.ndarray, q: np.ndarray) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._f_filt.append(f_filt)
            for i in range(3):
                self._tau[i].append(float(tau[i]))
                self._q[i].append(float(q[i]))

    def _run(self) -> None:
        import tkinter as tk
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title('3-DOF Pure Force Control')
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=0, minsize=300)
        root.rowconfigure(0, weight=1)

        # ── Left: matplotlib figure ───────────────────────────────────────
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky='nsew')

        fig, (ax_f, ax_t, ax_q) = plt.subplots(3, 1, figsize=(9, 8))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.06, hspace=0.35)
        fig.suptitle('3-DOF Pure Force Control', fontsize=11)

        ax_f.set_ylabel('Force [N]');     ax_f.grid(True, alpha=0.3)
        ax_t.set_ylabel('Torque [N·m]'); ax_t.grid(True, alpha=0.3)
        ax_q.set_ylabel('q [rad]');       ax_q.set_xlabel('Time [s]')
        ax_q.grid(True, alpha=0.3)

        line_fdes,  = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_fraw,  = ax_f.plot([], [], color='# fd9999', lw=0.8, alpha=0.5, label='F raw')
        line_ffilt, = ax_f.plot([], [], 'r-',  lw=1.5,  label='F filt')
        ax_f.legend(loc='upper left', fontsize=8)

        tau_colors = ['green', 'orange', 'purple']
        tau_lines  = [ax_t.plot([], [], color=tau_colors[i], lw=1.5,
                                label=f'tau{i+1}')[0] for i in range(3)]
        ax_t.legend(loc='upper left', fontsize=8)

        q_colors = ['steelblue', 'firebrick', 'goldenrod']
        q_lines  = [ax_q.plot([], [], color=q_colors[i], lw=1.2,
                               label=f'q{i+1}')[0] for i in range(3)]
        ax_q.legend(loc='upper left', fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # ── Right: control panel ──────────────────────────────────────────
        ctrl = tk.Frame(root, relief='groove', bd=2, padx=8, pady=8)
        ctrl.grid(row=0, column=1, sticky='nsew', padx=(0, 6), pady=6)
        ctrl.columnconfigure(1, weight=1)

        tk.Label(ctrl, text='Parameters',
                 font=('TkDefaultFont', 11, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 6))

        # (ctrl_key, label, clamp_lo, clamp_hi)
        _PARAMS = [
            ('f_des',    'F des  [N]',   0.0,  50.0),
            ('kp',       'Kp_f',         0.0, 100.0),
            ('ki',       'Ki_f',         0.0, 100.0),
            ('kd_vel',   'Kd_vel',       0.0,  10.0),
            ('lpf_alpha','LPF  alpha',  0.01,   1.0),
        ]

        entries = {}
        for r, (key, lbl, lo, hi) in enumerate(_PARAMS, start=1):
            tk.Label(ctrl, text=lbl + ':', anchor='w',
                     font=('TkDefaultFont', 10)).grid(
                row=r, column=0, sticky='w', pady=4, padx=(0, 6))

            var = tk.StringVar(value=str(ctrl_params.get(key, 0.0)))
            ent = tk.Entry(ctrl, textvariable=var, width=10,
                           font=('Courier', 13), justify='right',
                           relief='solid', bd=1)
            ent.grid(row=r, column=1, sticky='ew', pady=4)
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
            row=next_row, column=0, columnspan=2, sticky='ew', pady=8)

        tk.Button(ctrl, text='Apply All  [Enter]',
                  font=('TkDefaultFont', 10), bg='# d0e8ff',
                  relief='raised', pady=4,
                  command=_apply_all).grid(row=next_row+1, column=0, columnspan=2,
                                           sticky='ew', pady=2)

        tk.Button(ctrl, text='Reset Integrator',
                  font=('TkDefaultFont', 10), bg='# fffacd',
                  relief='raised', pady=4,
                  command=lambda: ctrl_params.__setitem__('reset', True)
                  ).grid(row=next_row+2, column=0, columnspan=2,
                         sticky='ew', pady=2)

        tk.Button(ctrl, text='E - S T O P',
                  font=('TkDefaultFont', 12, 'bold'),
                  bg='# e05050', fg='white', relief='raised', pady=6,
                  command=lambda: (Terminate.set(), root.quit())
                  ).grid(row=next_row+3, column=0, columnspan=2,
                         sticky='ew', pady=(6, 2))

        tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
            row=next_row+4, column=0, columnspan=2, sticky='ew', pady=6)
        tk.Label(ctrl, text='Status', font=('TkDefaultFont', 10, 'bold')
                 ).grid(row=next_row+5, column=0, columnspan=2, sticky='w')
        lbl_force = tk.Label(ctrl, text='F: --', anchor='w',
                             font=('Courier', 10), justify='left')
        lbl_force.grid(row=next_row+6, column=0, columnspan=2, sticky='w', pady=1)
        lbl_q = tk.Label(ctrl, text='q: --', anchor='w',
                         font=('Courier', 10), justify='left')
        lbl_q.grid(row=next_row+7, column=0, columnspan=2, sticky='w', pady=1)

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
                taus   = [list(d) for d in self._tau]
                qs     = [list(d) for d in self._q]

            t_now = t[-1];  t_lo = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_ffilt.set_data(t, f_filt)
            for i in range(3):
                tau_lines[i].set_data(t, taus[i])
                q_lines[i].set_data(t, qs[i])
            for ax in (ax_f, ax_t, ax_q):
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
                _ylim(ax_f, f_des, f_raw, f_filt)
                _ylim(ax_t, *taus)
                _ylim(ax_q, *qs)

            canvas.draw_idle()

            lbl_force.config(
                text=f'F filt:  {f_filt[-1]:+.3f} N\n'
                     f'F des:   {f_des[-1]:+.3f} N')
            lbl_q.config(
                text='\n'.join(
                    f'q{i+1}: {qs[i][-1]:+.3f} rad'
                    for i in range(3)))

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
        print(f"Step 1: Move arm to INIT_Q = {np.round(INIT_Q, 3)} rad, then press Enter to zero motors.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            for m in motors:
                m.cmd.kp       = 0.0
                m.cmd.kd       = KD_FREE
                m.cmd.position = 0.0
                m.cmd.velocity = 0.0
                m.cmd.torque   = 0.0
            update_all(motors, timeout=5.0)
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
            update_all(motors, timeout=5.0)
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
            update_all(motors, timeout=5.0)
            time.sleep(dt_boot)
        print("  Force control starting!\n")

        print(
            f"Config: F_des={F_DES_N}N  Kp={KP_F}  Ki={KI_F}  Kd_vel={KD_VEL}\n"
            f"  PUSH_DIR={PUSH_DIR}  LPF_alpha={LPF_ALPHA}\n"
            f"  Torque limit: +/-{TORQUE_LIMIT} N·m  "
            f"  Gravity comp: {'ON' if GRAV_COMP else 'OFF'}\n"
            f"Press Ctrl+C or close plot to stop.\n"
        )
        return True
    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")
        return False


# Control loop

def loop(motors: list, plotter: PlotThread, logger: DataLogger) -> None:
    try:
        # zero torque before entering loop
        for m in motors:
            m.cmd.kp       = 0.0
            m.cmd.kd       = 0.0
            m.cmd.position = 0.0
            m.cmd.velocity = 0.0
            m.cmd.torque   = 0.0

        integral_F = 0.0
        f_filt     = FORCE_SIGN * ft_latest[FORCE_AXIS]
        dt_nom     = 1.0 / LOOP_HZ
        prev_t: float | None = None

        rl         = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        try:
            for t in rl:
                if Terminate.is_set():
                    rl.stop()
                    break

                f_des  = ctrl_params['f_des']
                kp     = ctrl_params['kp']
                ki     = ctrl_params['ki']
                kd_vel = ctrl_params['kd_vel']
                alpha  = ctrl_params['lpf_alpha']

                if ctrl_params['reset']:
                    integral_F = 0.0
                    f_filt     = FORCE_SIGN * ft_latest[FORCE_AXIS]
                    ctrl_params['reset'] = False

                dt = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Force measurement
                f_raw  = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_filt = alpha * f_raw + (1.0 - alpha) * f_filt
                f_err  = f_des - f_filt

                # PI force controller
                integral_F += f_err * dt
                if ki > 1e-12:
                    windup     = TORQUE_LIMIT / ki
                    integral_F = float(np.clip(integral_F, -windup, windup))
                F_ctrl = f_des + kp * f_err + ki * integral_F

                # Read joint state
                q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                                  for i, mid in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[mid] * motors[i].state.velocity
                                  for i, mid in enumerate(MOTOR_IDS)])

                # Jacobian transpose mapping
                Jv      = dyn.evaluate_jacobian(q)[:3, :]   # 3×3 linear-velocity rows
                F_vec   = PUSH_DIR * F_ctrl
                tau_des = Jv.T @ F_vec

                # Optional gravity compensation
                if GRAV_COMP:
                    _, _, G = dyn.evaluate_MCG(q, np.zeros(3))
                    tau_des = tau_des + G

                # Velocity damping and torque limit
                tau_cmd = tau_des - kd_vel * qdot
                tau_cmd = np.clip(tau_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

                # ── Motor command (pure torque, kp=kd=0) ─────────────────────
                for i, mid in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp     = 0.0
                    motors[i].cmd.kd     = 0.0
                    motors[i].cmd.torque = float(SIGN[mid] * tau_cmd[i]) * rl.fade

                update_all(motors, timeout=2.0)

                tau_meas = np.array([SIGN[mid] * motors[i].state.torque
                                      for i, mid in enumerate(MOTOR_IDS)])

                plotter.push(t, f_des, f_raw, f_filt, tau_cmd, q)
                # logger.log(
                #     time_s=t,
                #     q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                #     qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                #     tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1], tau_meas3_Nm=tau_meas[2],
                #     f_des_N=f_des, f_raw_N=f_raw, f_filt_N=f_filt, f_err_N=f_err,
                #     tau1_des_Nm=tau_des[0], tau2_des_Nm=tau_des[1], tau3_des_Nm=tau_des[2],
                #     kp_f=kp, ki_f=ki, lpf_alpha=alpha,
                # )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  F={f_filt:+.4f}N  err={f_err:+.4f}N  "
                        f"intg={integral_F:+.4f}  "
                        f"tau=[{tau_cmd[0]:+.2f},{tau_cmd[1]:+.2f},{tau_cmd[2]:+.2f}]N·m  "
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
    # logger  = DataLogger('pure_force', PURE_FORCE_EXTRA_COLUMNS, directory=HERE)
    logger = None
    complete = setup(motors, sensor)
    if not complete:
        Terminate.set()
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
        print("\nStartup failed. Shutdown complete.")
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
