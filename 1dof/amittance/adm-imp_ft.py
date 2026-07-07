from __future__ import annotations

import collections
import csv
import os
import queue
import threading
import time

import can
import numpy as np
from ATI_FTsensor.ftsensor import ftsensor
from tmotorcan import MotorBus, MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorFaultError, MotorTimeoutError

# Hardware
COM_PORT    = 'COM18'
MOTOR_ID    = 1
MOTOR_MODEL = 'AK45-10'

# Physical
JACOBIAN   = 0.2     # lever arm [m]: τ_joint = JACOBIAN · F_ext [N·m]
FORCE_AXIS = 2       # ATI channel: 0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz
FORCE_SIGN = -1.0    # negate Fz → positive f_ati = push / compression

# Force setpoint
F_DES_N    = 0.0     # desired contact force [N] (live-tunable via panel)

# Admittance model (outer)
B_ADM      = 5.0     # virtual damping [N·m·s/rad]
K_SPRING   = 1.0     # restoring spring [N·m/rad]

# ── Inner impedance gains — co-tune KP and KD (see stability note above) ──────
KP_MOTOR   = 30.0    # position stiffness [N·m/rad]
KD_MOTOR   = 2.0     # velocity damping [N·m·s/rad]

# Signal processing
LPF_ALPHA  = 0.2     # IIR on F_est: y = α·x + (1−α)·y_prev (1.0=raw, 0.1=heavy)

# Safety
VEL_LIMIT    = 1.0   # hard cap on |q̇_des| [rad/s]
TORQUE_LIMIT = 3.0   # hardware torque cap [N·m]

# Timing
LOOP_HZ       = 100
BIAS_DURATION = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))

Terminate = threading.Event()
ft_latest = [0.0] * 6   # [Fx, Fy, Fz, Tx, Ty, Tz]

ctrl_params = {
    'f_des':     float(F_DES_N),
    'b_adm':     float(B_ADM),
    'k_spring':  float(K_SPRING),
    'kp_motor':  float(KP_MOTOR),
    'kd_motor':  float(KD_MOTOR),
    'lpf_alpha': float(LPF_ALPHA),
    'vel_limit': float(VEL_LIMIT),
}


# ── F/T reader thread (ground truth — not used for control) ───────────────────

def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


# Real-time plot

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1500: only keep the visible window

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_est    = collections.deque(maxlen=self.MAX_PTS)
        self._f_ati    = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_des = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_m   = collections.deque(maxlen=self.MAX_PTS)
        self._q        = collections.deque(maxlen=self.MAX_PTS)
        self._tau      = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float, f_des: float, f_est: float, f_ati: float,
             qdot_des: float, qdot_meas: float, q: float, tau: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_est.append(f_est)
            self._f_ati.append(f_ati)
            self._qdot_des.append(qdot_des)
            self._qdot_m.append(qdot_meas)
            self._q.append(q)
            self._tau.append(tau)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_f, ax_v, ax_q, ax_tau) = plt.subplots(4, 1, figsize=(10, 11))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.30)
        fig.suptitle('Admittance (outer) + Impedance (inner) — τ/J force estimate',
                     fontsize=11)

        ax_f.set_ylabel('Force [N]');   ax_f.grid(True, alpha=0.3)
        ax_v.set_ylabel('q̇ [rad/s]');  ax_v.grid(True, alpha=0.3)
        ax_q.set_ylabel('q [rad]');     ax_q.grid(True, alpha=0.3)
        ax_tau.set_ylabel('τ [N·m]');   ax_tau.set_xlabel('Time [s]')
        ax_tau.grid(True, alpha=0.3)

        line_fdes, = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_fest, = ax_f.plot([], [], 'r-',  lw=1.5, label='F est τ/J (control)')
        line_fati, = ax_f.plot([], [], color='# 2ca02c', lw=1.0,
                               alpha=0.8, ls='--', label='F ati (ground truth)')
        ax_f.legend(loc='upper left', fontsize=8)

        line_vdes,  = ax_v.plot([], [], 'b--', lw=1.5, label='q̇ des')
        line_vmeas, = ax_v.plot([], [], 'g-',  lw=1.5, label='q̇ meas')
        ax_v.legend(loc='upper left', fontsize=8)

        line_q, = ax_q.plot([], [], 'm-', lw=1.5, label='q')
        ax_q.axhline(0, color='k', lw=0.8, ls='--')
        ax_q.legend(loc='upper left', fontsize=8)

        line_tau, = ax_tau.plot([], [], color='darkorange', lw=1.5, label='τ meas [N·m]')
        ax_tau.axhline(0, color='k', lw=0.8, ls='--')
        ax_tau.legend(loc='upper left', fontsize=8)

        row1_y, row2_y = 0.15, 0.05
        w, h = 0.11, 0.07
        axes_tb = {
            'f_des':     fig.add_axes([0.05, row1_y, w, h]),
            'b_adm':     fig.add_axes([0.19, row1_y, w, h]),
            'k_spring':  fig.add_axes([0.33, row1_y, w, h]),
            'kp_motor':  fig.add_axes([0.47, row1_y, w, h]),
            'kd_motor':  fig.add_axes([0.61, row1_y, w, h]),
            'lpf_alpha': fig.add_axes([0.75, row1_y, w, h]),
            'vel_limit': fig.add_axes([0.05, row2_y, w, h]),
        }
        labels = {
            'f_des':     'F_des [N]',
            'b_adm':     'B_adm',
            'k_spring':  'K_spring',
            'kp_motor':  'Kp_motor',
            'kd_motor':  'Kd_motor',
            'lpf_alpha': 'LPF α',
            'vel_limit': 'VelLim r/s',
        }
        textboxes = {}
        for key, ax in axes_tb.items():
            tb = TextBox(ax, labels[key], initial=str(ctrl_params[key]),
                         color='lightyellow', hovercolor='yellow')
            def _make_handler(k):
                def _handler(val):
                    try:
                        v = float(val)
                        if k == 'lpf_alpha':
                            v = float(np.clip(v, 0.0, 1.0))
                        ctrl_params[k] = v
                        print(f"[panel] {k} → {v:.6f}")
                    except ValueError:
                        pass
                return _handler
            tb.on_submit(_make_handler(key))
            textboxes[key] = tb

        def _update(_):
            with self._lock:
                t      = list(self._t)
                f_des  = list(self._f_des)
                f_est  = list(self._f_est)
                f_ati  = list(self._f_ati)
                vd     = list(self._qdot_des)
                vm     = list(self._qdot_m)
                q      = list(self._q)
                tau    = list(self._tau)

            if len(t) < 2:
                return (line_fdes, line_fest, line_fati,
                        line_vdes, line_vmeas, line_q, line_tau)

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fest.set_data(t, f_est)
            line_fati.set_data(t, f_ati)
            line_vdes.set_data(t, vd)
            line_vmeas.set_data(t, vm)
            line_q.set_data(t, q)
            line_tau.set_data(t, tau)

            for ax in (ax_f, ax_v, ax_q, ax_tau):
                ax.set_xlim(t_lo, t_now + 0.1)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s]
                if not vals:
                    return
                lo, hi = min(vals), max(vals)
                pad = max(0.3, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            self._frame += 1
            if self._frame % 10 == 0:   # autoscale ~1 Hz at 10 fps
                _ylim(ax_f, f_des, f_est, f_ati)
                _ylim(ax_v, vd, vm)
                _ylim(ax_q, q)
                _ylim(ax_tau, tau)

            return (line_fdes, line_fest, line_fati,
                    line_vdes, line_vmeas, line_q, line_tau)

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


# CSV logger

class DataLogger:
    COLUMNS = [
        'time_s',
        'f_des_N', 'f_est_N', 'f_ati_N', 'f_err_N',
        'tau_meas_Nm',
        'qdot_des_rad_s', 'qdot_meas_rad_s',
        'q_des_rad', 'q_rad',
        'b_adm', 'k_spring', 'kp_motor', 'kd_motor', 'lpf_alpha', 'vel_limit',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'adm_imp_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, f_est: float, f_ati: float, f_err: float,
            tau_meas: float,
            qdot_des: float, qdot_meas: float,
            q_des: float, q: float,
            b_adm: float, k_spring: float, kp_motor: float, kd_motor: float,
            lpf_alpha: float, vel_limit: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_est:.6f}', f'{f_ati:.6f}', f'{f_err:.6f}',
            f'{tau_meas:.6f}',
            f'{qdot_des:.6f}', f'{qdot_meas:.6f}',
            f'{q_des:.6f}', f'{q:.6f}',
            f'{b_adm:.6f}', f'{k_spring:.6f}', f'{kp_motor:.6f}', f'{kd_motor:.6f}',
            f'{lpf_alpha:.4f}', f'{vel_limit:.4f}',
        ))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        print(f"[log] saved → {self.path}")

    def _writer(self) -> None:
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)


# Control worker

def _control_worker(bus: 'MotorBus', motor: 'MITMotor',
                    plotter: PlotThread, logger: DataLogger) -> None:
    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        kp = ctrl_params['kp_motor']
        kd = ctrl_params['kd_motor']
        kd_crit = 2.0 * np.sqrt(kp * 0.01)   # estimate with J≈0.01 kg·m²
        if kd < kd_crit * 0.7:
            print(
                f"[WARN] KP={kp}, KD={kd}: underdamped! "
                f"KD_crit ≈ {kd_crit:.2f} — recommend KD ≥ {kd_crit:.1f}"
            )

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Inner impedance : KP={kp}  KD={kd}  (KD_crit≈{kd_crit:.2f})\n"
            f"  Force estimate  : F_est = LPF(τ_meas / J)  J={JACOBIAN} m\n"
            f"  Outer admittance: q̇_des = (J·(F_des−F_est) − K·q) / B\n"
            f"  ATI sensor : ground-truth comparison only (not used for control)\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = kp
        motor.cmd.kd       = kd
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0
        motor.cmd.torque   = 0.0

        f_est_filt = 0.0   # IIR state: filtered force estimate from τ/J
        q_des      = 0.0   # integrated virtual position
        dt_nom     = 1.0 / LOOP_HZ
        prev_t: float | None = None

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0   # throttle console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                dt = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Live tuning params
                f_des     = ctrl_params['f_des']
                b_adm     = ctrl_params['b_adm']
                k_spring  = ctrl_params['k_spring']
                kp_motor  = ctrl_params['kp_motor']
                kd_motor  = ctrl_params['kd_motor']
                alpha     = ctrl_params['lpf_alpha']
                vel_limit = ctrl_params['vel_limit']

                # Motor state
                q       = motor.state.position
                omega   = motor.state.velocity
                tau_meas = motor.state.torque    # output-shaft torque from phase current

                # ── Force estimation: F_est = LPF(τ_meas / J) ────────────────
                # Accurate at contact (ω≈0). Biased during motion (inertia/friction).
                f_est_raw  = tau_meas / JACOBIAN
                f_est_filt = alpha * f_est_raw + (1.0 - alpha) * f_est_filt

                # ── ATI ground truth (comparison — not used for control) ───────
                f_ati = FORCE_SIGN * float(ft_latest[FORCE_AXIS])

                # ── Admittance outer: q̇_des = (J·(F_des−F_est) − K·q) / B ───
                f_err    = f_des - f_est_filt
                f_joint  = JACOBIAN * f_err
                qdot_des = (f_joint - k_spring * q) / b_adm
                qdot_des = float(np.clip(qdot_des, -vel_limit, vel_limit))

                # ── Integrate q̇_des → q_des ──────────────────────────────────
                q_des += qdot_des * dt * loop.fade

                # Inner impedance command
                motor.cmd.kp       = kp_motor
                motor.cmd.kd       = kd_motor
                motor.cmd.position = q_des
                motor.cmd.velocity = 0.0    # inner loop holds q_des
                motor.cmd.torque   = 0.0

                motor.update(t)

                plotter.push(t, f_des, f_est_filt, f_ati,
                             qdot_des, omega, q, tau_meas)

                logger.log(
                    t,
                    f_des, f_est_filt, f_ati, f_err,
                    tau_meas,
                    qdot_des, omega,
                    q_des, q,
                    b_adm, k_spring, kp_motor, kd_motor, alpha, vel_limit,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  "
                        f"F_est={f_est_filt:+.3f}N  "
                        f"F_ati={f_ati:+.3f}N  "
                        f"err={f_err:+.3f}N  "
                        f"q̇_des={qdot_des:+.4f}r/s  "
                        f"q̇={omega:+.4f}r/s  "
                        f"τ={tau_meas:+.3f}N·m  "
                        f"T={motor.state.temp}°C"
                    )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


# Entry point

def main() -> None:
    print("Initialising ATI F/T sensor (ground-truth comparison) …")
    sensor = ftsensor()
    sensor.start_task()
    time.sleep(0.5)
    print(f"Biasing F/T sensor for {BIAS_DURATION:.1f} s — keep sensor unloaded …")
    sensor.reBias(duration=BIAS_DURATION)
    print("Bias complete.")
    time.sleep(1)

    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("ATI F/T background reader started.")

    raw_bus = can.interface.Bus(
        interface="slcan",
        channel=COM_PORT,
        bitrate=1_000_000,
    )
    bus   = MotorBus(raw_bus)
    motor = MITMotor(bus, motor_id=MOTOR_ID, model=MOTOR_MODEL)
    motor.enable()

    plotter = PlotThread()
    logger  = DataLogger()

    ctrl_thread = threading.Thread(
        target=_control_worker, args=(bus, motor, plotter, logger),
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
        try:
            motor.coast()
            motor.disable()
        except Exception:
            pass
        sensor.stop_task()
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
