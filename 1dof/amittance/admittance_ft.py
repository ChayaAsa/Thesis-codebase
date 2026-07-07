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
COM_PORT     = 'COM18'
MOTOR_ID     = 1
MOTOR_MODEL  = 'AK45-10'

# Physical
JACOBIAN     = 0.2       # lever arm [m] : F_ext [N] → τ_joint = J·F [N·m]
FORCE_AXIS   = 2         # ATI channel index: 0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz
FORCE_SIGN   = -1.0      # negate Fz so positive f_filt = push/compression

# Force setpoint
F_DES_N      = 0.0       # desired contact force [N] (live-tunable via panel)

# ── Admittance model  B·q̇_des = J·(F_des − F_ext) − K_spring·q ───────────────
B_ADM        = 5.0       # virtual damping [N·m·s/rad]
K_SPRING     = 1.0       # restoring spring [N·m/rad]

# Motor position + velocity tracking gains
KP_MOTOR     = 5.0       # motor kp [N·m/rad]
                         # increase to raise max achievable force
KD_MOTOR     = 0.3       # motor kd [N·m·s/rad]

# Signal processing
LPF_ALPHA    = 0.3       # IIR: y = α·x + (1−α)·y_prev (1.0=raw, 0.1=heavy)

# Safety
VEL_LIMIT    = 2.0       # hard cap on |q̇_des| [rad/s]
TORQUE_LIMIT = 3.0       # hardware torque cap [N·m]

# Timing
LOOP_HZ       = 100
BIAS_DURATION = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))

Terminate = threading.Event()
ft_latest = [0.0] * 6   # [Fx, Fy, Fz, Tx, Ty, Tz]

# Live-tunable params — GIL-atomic dict writes, safe across threads
ctrl_params = {
    'f_des':     float(F_DES_N),
    'b_adm':     float(B_ADM),
    'k_spring':  float(K_SPRING),
    'kp_motor':  float(KP_MOTOR),
    'kd_motor':  float(KD_MOTOR),
    'lpf_alpha': float(LPF_ALPHA),
    'vel_limit': float(VEL_LIMIT),
}


# F/T reader thread

def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


# Real-time plot thread

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1500: only keep the visible window

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw    = collections.deque(maxlen=self.MAX_PTS)
        self._f_filt   = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_des = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_m   = collections.deque(maxlen=self.MAX_PTS)
        self._q        = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float, f_des: float, f_raw: float, f_filt: float,
             qdot_des: float, qdot_meas: float, q: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._f_filt.append(f_filt)
            self._qdot_des.append(qdot_des)
            self._qdot_m.append(qdot_meas)
            self._q.append(q)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_f, ax_v, ax_q) = plt.subplots(3, 1, figsize=(10, 9))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.30)
        fig.suptitle('1-DOF Admittance Control — real-time', fontsize=11)

        ax_f.set_ylabel('Force [N]');  ax_f.grid(True, alpha=0.3)
        ax_v.set_ylabel('q̇ [rad/s]'); ax_v.grid(True, alpha=0.3)
        ax_q.set_ylabel('q [rad]');    ax_q.set_xlabel('Time [s]')
        ax_q.grid(True, alpha=0.3)

        line_fdes,  = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_fraw,  = ax_f.plot([], [], color='# fd9999', lw=0.8, alpha=0.7, label='F raw')
        line_ffilt, = ax_f.plot([], [], 'r-',  lw=1.5, label='F filtered')
        ax_f.legend(loc='upper left', fontsize=8)

        line_vdes,  = ax_v.plot([], [], 'b--', lw=1.5, label='q̇ des')
        line_vmeas, = ax_v.plot([], [], 'g-',  lw=1.5, label='q̇ meas')
        ax_v.legend(loc='upper left', fontsize=8)

        line_q,     = ax_q.plot([], [], 'm-', lw=1.5, label='q')
        ax_q.axhline(0, color='k', lw=0.8, ls='--')
        ax_q.legend(loc='upper left', fontsize=8)

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
            textboxes[key] = tb  # keep reference

        def _update(_):
            with self._lock:
                t      = list(self._t)
                f_des  = list(self._f_des)
                f_raw  = list(self._f_raw)
                f_filt = list(self._f_filt)
                vd     = list(self._qdot_des)
                vm     = list(self._qdot_m)
                q      = list(self._q)

            if len(t) < 2:
                return line_fdes, line_fraw, line_ffilt, line_vdes, line_vmeas, line_q

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_ffilt.set_data(t, f_filt)
            line_vdes.set_data(t, vd)
            line_vmeas.set_data(t, vm)
            line_q.set_data(t, q)

            for ax in (ax_f, ax_v, ax_q):
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
                _ylim(ax_f, f_des, f_raw, f_filt)
                _ylim(ax_v, vd, vm)
                _ylim(ax_q, q)

            return line_fdes, line_fraw, line_ffilt, line_vdes, line_vmeas, line_q

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


# CSV logger

class DataLogger:

    COLUMNS = [
        'time_s',
        'f_des_N', 'f_raw_N', 'f_filt_N', 'f_err_N', 'f_joint_Nm',
        'qdot_des_rad_s', 'qdot_meas_rad_s',
        'q_des_rad', 'q_rad',
        'tau_meas_Nm',
        'b_adm', 'k_spring', 'kp_motor', 'kd_motor', 'lpf_alpha', 'vel_limit',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'admittance_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, f_raw: float, f_filt: float, f_err: float, f_joint: float,
            qdot_des: float, qdot_meas: float,
            q_des: float, q: float, tau_meas: float,
            b_adm: float, k_spring: float, kp_motor: float, kd_motor: float,
            lpf_alpha: float, vel_limit: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_raw:.6f}', f'{f_filt:.6f}',
            f'{f_err:.6f}', f'{f_joint:.6f}',
            f'{qdot_des:.6f}', f'{qdot_meas:.6f}',
            f'{q_des:.6f}', f'{q:.6f}',
            f'{tau_meas:.6f}',
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
        motor.zero()   # q=0 at current (wall-contact) position

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Admittance: q̇_des = (J·(F_des−F_ext) − K·q) / B\n"
            f"              q_des += q̇_des · dt   (integrated)\n"
            f"  F_des={F_DES_N}N  B_adm={B_ADM}  K_spring={K_SPRING}\n"
            f"  Kp_motor={KP_MOTOR}  Kd_motor={KD_MOTOR}  VelLimit={VEL_LIMIT} rad/s\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = KP_MOTOR
        motor.cmd.kd       = KD_MOTOR
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0
        motor.cmd.torque   = 0.0

        f_filt = FORCE_SIGN * ft_latest[FORCE_AXIS]  # init IIR state
        q_des  = 0.0                                  # integrated virtual position
        dt_nom = 1.0 / LOOP_HZ
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

                # Read live tuning params
                f_des     = ctrl_params['f_des']
                b_adm     = ctrl_params['b_adm']
                k_spring  = ctrl_params['k_spring']
                kp_motor  = ctrl_params['kp_motor']
                kd_motor  = ctrl_params['kd_motor']
                alpha     = ctrl_params['lpf_alpha']
                vel_limit = ctrl_params['vel_limit']

                # F/T measurement
                f_raw   = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_filt  = alpha * f_raw + (1.0 - alpha) * f_filt
                f_err   = f_des - f_filt                # [N] positive → need more force
                f_joint = JACOBIAN * f_err              # [N·m] joint-space force error

                # ── Admittance: q̇_des = (J·(F_des−F_ext) − K·q) / B ──────────
                q        = motor.state.position
                qdot_des = (f_joint - k_spring * q) / b_adm
                qdot_des = float(np.clip(qdot_des, -vel_limit, vel_limit))

                # ── Integrate q̇_des → q_des ──────────────────────────────────
                # → KP_MOTOR·(q_des−q) increases → force builds to F_des
                # Zero steady-state error (vs pure velocity mode: F_ss ≈ 9% F_des)
                q_des += qdot_des * dt * loop.fade

                # Motor command (position + velocity feedforward)
                motor.cmd.kp       = kp_motor
                motor.cmd.kd       = kd_motor
                motor.cmd.position = q_des
                motor.cmd.velocity = qdot_des * loop.fade
                motor.cmd.torque   = 0.0

                motor.update(t)

                plotter.push(t, f_des, f_raw, f_filt, qdot_des,
                             motor.state.velocity, q)

                logger.log(
                    t,
                    f_des, f_raw, f_filt, f_err, f_joint,
                    qdot_des, motor.state.velocity,
                    q_des, q, motor.state.torque,
                    b_adm, k_spring, kp_motor, kd_motor, alpha, vel_limit,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  "
                        f"F={f_filt:+.3f}N  "
                        f"err={f_err:+.3f}N  "
                        f"q̇_des={qdot_des:+.4f}r/s  "
                        f"q̇={motor.state.velocity:+.4f}r/s  "
                        f"q_des={q_des:+.4f}rad  "
                        f"q={q:+.4f}rad  "
                        f"τ={motor.state.torque:+.3f}N·m  "
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
    # ATI F/T sensor init
    print("Initialising ATI F/T sensor …")
    sensor = ftsensor()
    sensor.start_task()
    time.sleep(0.5)
    print(f"Biasing F/T sensor for {BIAS_DURATION:.1f} s — keep the sensor unloaded …")
    sensor.reBias(duration=BIAS_DURATION)
    print("Bias complete.")
    time.sleep(1)

    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("F/T background reader started.")

    # CAN bus + motor init
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
        plotter._run()          # blocks in plt.show()
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
