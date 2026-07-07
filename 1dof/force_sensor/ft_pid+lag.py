from __future__ import annotations

import collections
import csv
import os
import queue
import threading
import time

import can
import numpy as np
from nidaqmx import Task
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.constants import AcquisitionType

from tmotorcan import MotorBus, MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorTimeoutError, MotorFaultError

from ATI_FTsensor.ftsensor import ftsensor


COM_PORT      = 'COM18'          # ← change to your USB-CAN port

MOTOR_ID      = 1                # ← CAN ID of your motor
MOTOR_MODEL   = 'AK45-10'        # ← motor model name or path to custom .yaml

HERE       = os.path.dirname(os.path.abspath(__file__))
F_DESIRED_N   = 0                # target force [N]

JACOBIAN      = 0.2              # effective lever arm / coupling ratio [m]
                                 #   τ_cmd = JACOBIAN * (kp_f * err + ki_f * integral)
                                 #   set to 1.0 if motor directly commands force

KP_F          = 0.0              # force proportional gain [N·m / N]
KI_F          = 0.0             # force integral gain [N·m / (N·s)]
KD_F          = 0.0              # force derivative gain [N·m·s / N]

LPF_ALPHA     = 0.3              # F_meas low-pass filter: 1.0=no filter, 0.1=heavy
                                 #   α=0.3 → ~5 Hz cutoff at 100 Hz sample rate

TORQUE_LIMIT   = 3.0             # hard torque cap [N·m]
BIAS_DURATION  = 5.0             # seconds to average for F/T bias (sensor unloaded)


LOOP_HZ       = 100              # control frequency [Hz]
# SENSOR_     = 120              # F/T sensor read frequency [Hz] — set equal to or higher than LOOP_HZ


Terminate  = threading.Event()
ft_latest  = [0.0] * 6          # most recent [Fx, Fy, Fz, Tx, Ty, Tz]

# Tunable controller params — updated live by the plot-thread TextBoxes,
ctrl_params = {
    'f_des':      float(F_DESIRED_N),
    'kp':         float(KP_F),
    'ki':         float(KI_F),
    'kd_f':       float(KD_F),
    'lpf_alpha':  float(LPF_ALPHA),
    'reset':      False,              # set True by panel to flush the integrator
}


def ft_reader_thread(sensor: ftsensor) -> None:

    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()   # in-place update
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break

    print("[ATI] reader stopped")


# Real-time plot thread

class PlotThread:

    WINDOW_S  = 10.0                           # seconds of history to display
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1500: only keep the visible window

    def __init__(self):
        self._lock   = threading.Lock()
        self._frame  = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw    = collections.deque(maxlen=self.MAX_PTS)
        self._f_meas   = collections.deque(maxlen=self.MAX_PTS)
        self._tau      = collections.deque(maxlen=self.MAX_PTS)
        self._tau_meas = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float, f_des: float, f_raw: float, f_meas: float,
             tau: float, tau_meas: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._f_meas.append(f_meas)
            self._tau.append(tau)
            self._tau_meas.append(tau_meas)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(10, 7))
        # leave a strip at the bottom for the input panel
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.23)
        fig.suptitle('1-DOF Force Control — real-time', fontsize=11)

        ax_f.set_ylabel('Force [N]')
        ax_f.set_xlabel('Time [s]')
        ax_f.grid(True, alpha=0.3)
        line_fdes,  = ax_f.plot([], [], 'b--', lw=1.5,  label='F desired')
        line_fraw,  = ax_f.plot([], [], color="# fd9999", lw=0.8, alpha=0.6, label='F raw')
        line_fmeas, = ax_f.plot([], [], 'r-',  lw=1.5,  label='F filtered')
        ax_f.legend(loc='upper left', fontsize=8)

        ax_t.set_ylabel('Torque [N·m]')
        ax_t.set_xlabel('Time [s]')
        ax_t.grid(True, alpha=0.3)
        line_tau,      = ax_t.plot([], [], 'g-',  lw=2.0, label='τ cmd')
        line_tau_meas, = ax_t.plot([], [], 'm-', lw=1.0, label='τ measured')
        ax_t.legend(loc='upper left', fontsize=8)

        # Input panel
        # Four TextBoxes across the bottom strip.  Press Enter to apply.
        # [left, bottom, width, height] in figure-fraction coordinates.
        ax_tb_fdes = fig.add_axes([0.05, 0.07, 0.14, 0.08])
        ax_tb_kp   = fig.add_axes([0.23, 0.07, 0.14, 0.08])
        ax_tb_ki   = fig.add_axes([0.41, 0.07, 0.14, 0.08])
        ax_tb_kdf  = fig.add_axes([0.59, 0.07, 0.14, 0.08])
        ax_tb_lpf  = fig.add_axes([0.77, 0.07, 0.14, 0.08])

        tb_fdes = TextBox(ax_tb_fdes, 'F_des [N]', initial=str(ctrl_params['f_des']),
                          color='lightyellow', hovercolor='yellow')
        tb_kp   = TextBox(ax_tb_kp,   'Kp_f     ', initial=str(ctrl_params['kp']),
                          color='lightyellow', hovercolor='yellow')
        tb_ki   = TextBox(ax_tb_ki,   'Ki_f     ', initial=str(ctrl_params['ki']),
                          color='lightyellow', hovercolor='yellow')
        tb_kdf  = TextBox(ax_tb_kdf,  'Kd_f     ', initial=str(ctrl_params['kd_f']),
                          color='lightyellow', hovercolor='yellow')
        tb_lpf  = TextBox(ax_tb_lpf,  'LPF α    ', initial=str(ctrl_params['lpf_alpha']),
                          color='lightcyan',   hovercolor='cyan')

        def _on_fdes(val):
            try:
                ctrl_params['f_des'] = float(val)
                ctrl_params['reset'] = True    # flush integrator on setpoint change
                print(f"[panel] F_des → {ctrl_params['f_des']:.4f} N  (integral reset)")
            except ValueError:
                pass

        def _on_kp(val):
            try:
                ctrl_params['kp'] = float(val)
                print(f"[panel] Kp → {ctrl_params['kp']:.6f}")
            except ValueError:
                pass

        def _on_ki(val):
            try:
                ctrl_params['ki'] = float(val)
                ctrl_params['reset'] = True    # reset integral when Ki changes
                print(f"[panel] Ki → {ctrl_params['ki']:.6f}  (integral reset)")
            except ValueError:
                pass

        def _on_kdf(val):
            try:
                ctrl_params['kd_f'] = float(val)
                print(f"[panel] Kd_f → {ctrl_params['kd_f']:.6f}")
            except ValueError:
                pass

        def _on_lpf(val):
            try:
                a = float(val)
                ctrl_params['lpf_alpha'] = float(np.clip(a, 0.0, 1.0))
                print(f"[panel] LPF α → {ctrl_params['lpf_alpha']:.4f}  (1.0=raw, 0.1=heavy)")
            except ValueError:
                pass

        tb_fdes.on_submit(_on_fdes)
        tb_kp.on_submit(_on_kp)
        tb_ki.on_submit(_on_ki)
        tb_kdf.on_submit(_on_kdf)
        tb_lpf.on_submit(_on_lpf)

        # Animation
        def _update(_):
            with self._lock:
                t        = list(self._t)
                f_des    = list(self._f_des)
                f_raw    = list(self._f_raw)
                f_meas   = list(self._f_meas)
                tau      = list(self._tau)
                tau_meas = list(self._tau_meas)

            if len(t) < 2:
                return line_fdes, line_fraw, line_fmeas, line_tau, line_tau_meas

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_fmeas.set_data(t, f_meas)
            line_tau.set_data(t, tau)
            line_tau_meas.set_data(t, tau_meas)

            ax_f.set_xlim(t_lo, t_now + 0.1)
            ax_t.set_xlim(t_lo, t_now + 0.1)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s]
                if not vals:
                    return
                lo, hi = min(vals), max(vals)
                pad = max(0.5, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            self._frame += 1
            if self._frame % 10 == 0:   # autoscale ~1 Hz at 10 fps
                _ylim(ax_f, f_des, f_raw, f_meas)
                _ylim(ax_t, tau, tau_meas)

            return line_fdes, line_fraw, line_fmeas, line_tau, line_tau_meas

        # keep reference on self — FuncAnimation is GC'd if only held in a local
        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


class DataLogger:

    COLUMNS = [
        'time_s',
        'f_des_N', 'f_raw_N', 'f_filt_N',
        'tau_cmd_Nm', 'tau_meas_Nm',
        'pos_rad', 'vel_rad_s', 'accel_rad_s2',
        'kp', 'ki', 'kd_f', 'lpf_alpha',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'sysid_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, f_raw: float, f_filt: float,
            tau_cmd: float, tau_meas: float,
            pos: float, vel: float, accel: float,
            kp: float, ki: float, kd_f: float, lpf_alpha: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_raw:.6f}', f'{f_filt:.6f}',
            f'{tau_cmd:.6f}', f'{tau_meas:.6f}',
            f'{pos:.6f}', f'{vel:.6f}', f'{accel:.6f}',
            f'{kp:.6f}', f'{ki:.6f}', f'{kd_f:.6f}', f'{lpf_alpha:.4f}',
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


def _control_worker(bus: 'MotorBus', motor: 'MITMotor',
                    plotter: PlotThread, logger: DataLogger) -> None:
    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Target force : {F_DESIRED_N:+.2f} N\n"
            f"  Kp={KP_F}  Ki={KI_F}  Kd_f={KD_F}\n"
            f"  Motor kp=0  kd=0  (pure torque mode)\n"
            f"  Torque limit : ±{TORQUE_LIMIT} N·m\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = 0.0
        motor.cmd.kd       = 0.1
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0

        integral   = 0.0
        prev_f_err = 0.0
        f_filt     = float(-ft_latest[2])   # IIR filter state
        dt_nom     = 1.0 / LOOP_HZ
        prev_t: float | None = None

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0   # throttle console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                f_des = ctrl_params['f_des']
                kp    = ctrl_params['kp']
                ki    = ctrl_params['ki']
                kd_f  = ctrl_params['kd_f']
                alpha = ctrl_params['lpf_alpha']

                if ctrl_params['reset']:
                    integral   = 0.0
                    prev_f_err = 0.0
                    f_filt     = float(-ft_latest[2])
                    ctrl_params['reset'] = False

                f_raw  = float(-ft_latest[2])
                f_filt = alpha * f_raw + (1.0 - alpha) * f_filt
                f_meas = f_filt

                dt_actual = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                f_err      = f_des - f_meas
                integral  += f_err * dt_actual
                windup_lim = TORQUE_LIMIT / (JACOBIAN * ki) if ki > 1e-12 else 1e6
                integral   = float(np.clip(integral, -windup_lim, windup_lim))
                # d_err      = (f_err - prev_f_err) / dt_actual
                prev_f_err = f_err

                # tau_raw = JACOBIAN * (f_des + kp * f_err + ki * integral + kd_f * d_err)
                tau_raw = JACOBIAN * (f_des + kp * f_err + ki * integral)
                tau_cmd = float(np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT))
                motor.cmd.torque = tau_cmd * loop.fade

                plotter.push(t, f_des, f_raw, f_meas, tau_cmd, motor.state.torque)
                motor.update(t)

                logger.log(
                    t,
                    f_des, f_raw, f_meas,
                    tau_cmd, motor.state.torque,
                    motor.state.position, motor.state.velocity, motor.state.acceleration,
                    kp, ki, kd_f, alpha,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  "
                        f"F_meas={f_meas:+.4f}N  "
                        f"err={f_err:+.4f}N  "
                        f"intg={integral:+.4f}  "
                        f"τ={tau_cmd:+.3f}N·m  "
                        f"kp={kp:.4f} ki={ki:.4f} kd_f={kd_f:.4f}  "
                        f"pos={motor.state.position:+.4f}rad  "
                        f"vel={motor.state.velocity:+.4f}rad/s  "
                        f"T={motor.state.temp}°C"
                    )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


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
    bus = MotorBus(raw_bus)
    motor = MITMotor(bus, motor_id=MOTOR_ID, model=MOTOR_MODEL)
    motor.enable()

    plotter = PlotThread()
    logger  = DataLogger()

    # control loop runs in a worker thread; plot runs on the main thread below
    ctrl_thread = threading.Thread(
        target=_control_worker, args=(bus, motor, plotter, logger),
        daemon=True, name='Control'
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
