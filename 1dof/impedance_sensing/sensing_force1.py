from __future__ import annotations

import collections
import csv
import os
import queue
import threading
import time

import can

from tmotorcan import MotorBus, MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorTimeoutError, MotorFaultError

from ATI_FTsensor.ftsensor import ftsensor


COM_PORT      = 'COM18'
MOTOR_ID      = 1
MOTOR_MODEL   = 'AK45-10'

HERE          = os.path.dirname(os.path.abspath(__file__))

FORCE_AXIS    = 2      # 0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz
BIAS_DURATION = 5.0

JACOBIAN      = 0.2    # F_ext = τ_meas / J [m]
TORQUE_LIMIT  = 3.0    # hard cap on feedforward torque [N·m]
LOOP_HZ       = 100

ctrl_params = {
    'f_des':    0.0,    # desired force → feedforward torque = f_des × J
    'motor_kp': 5.0,    # MIT motor position gain [N·m/rad]
    'motor_kd': 0.1,    # MIT motor velocity gain [N·m·s/rad]
}

Terminate = threading.Event()
ft_latest = [0.0] * 6


def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1500: only keep the visible window

    def __init__(self):
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw    = collections.deque(maxlen=self.MAX_PTS)
        self._f_ati    = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmd  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_meas = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t, f_des, f_raw, f_ati, tau_cmd, tau_meas):
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._f_ati.append(f_ati)
            self._tau_cmd.append(tau_cmd)
            self._tau_meas.append(tau_meas)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(11, 7))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.22)
        fig.suptitle('Force Sensing — motor current feedback', fontsize=11)

        ax_f.set_ylabel('Force [N]')
        ax_f.set_xlabel('Time [s]')
        ax_f.grid(True, alpha=0.3)
        line_fdes, = ax_f.plot([], [], 'b--', lw=1.5, label='F desired (ref)')
        line_fraw, = ax_f.plot([], [], 'r-',  lw=1.5, label='F est (τ/J)')
        line_fati, = ax_f.plot([], [], 'k-',  lw=1.5, label='F ATI (sensor)')
        ax_f.legend(loc='upper left', fontsize=8)

        ax_t.set_ylabel('Torque [N·m]')
        ax_t.set_xlabel('Time [s]')
        ax_t.grid(True, alpha=0.3)
        line_tau_cmd,  = ax_t.plot([], [], 'g--', lw=1.5, label='τ cmd (feedforward)')
        line_tau_meas, = ax_t.plot([], [], 'm-',  lw=1.5, label='τ meas (motor current)')
        ax_t.legend(loc='upper left', fontsize=8)

        # TextBox row
        W, H = 0.20, 0.08
        ax_mkp  = fig.add_axes([0.05,              0.07, W, H])
        ax_mkd  = fig.add_axes([0.05 + W + 0.03,   0.07, W, H])
        ax_fdes = fig.add_axes([0.05 + 2*(W+0.03), 0.07, W, H])

        tb_mkp  = TextBox(ax_mkp,  'motor Kp',  initial=str(ctrl_params['motor_kp']),
                          color='lightgreen', hovercolor='limegreen')
        tb_mkd  = TextBox(ax_mkd,  'motor Kd',  initial=str(ctrl_params['motor_kd']),
                          color='lightgreen', hovercolor='limegreen')
        tb_fdes = TextBox(ax_fdes, 'F_des [N]', initial=str(ctrl_params['f_des']),
                          color='lightyellow', hovercolor='yellow')

        def _mk_cb(key, lo=None):
            def cb(val):
                try:
                    v = float(val)
                    if lo is not None:
                        v = max(lo, v)
                    ctrl_params[key] = v
                    print(f"[panel] {key} → {v}")
                except ValueError:
                    pass
            return cb

        tb_mkp.on_submit(_mk_cb('motor_kp', lo=0.0))
        tb_mkd.on_submit(_mk_cb('motor_kd', lo=0.0))
        tb_fdes.on_submit(_mk_cb('f_des'))

        def _update(_):
            with self._lock:
                t        = list(self._t)
                f_des    = list(self._f_des)
                f_raw    = list(self._f_raw)
                f_ati    = list(self._f_ati)
                tau_cmd  = list(self._tau_cmd)
                tau_meas = list(self._tau_meas)

            if len(t) < 2:
                return line_fdes, line_fraw, line_fati, line_tau_cmd, line_tau_meas

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_fati.set_data(t, f_ati)
            line_tau_cmd.set_data(t, tau_cmd)
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
                _ylim(ax_f, f_des, f_raw, f_ati)
                _ylim(ax_t, tau_cmd, tau_meas)

            return line_fdes, line_fraw, line_fati, line_tau_cmd, line_tau_meas

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


class DataLogger:
    COLUMNS = [
        'time_s',
        'f_des_N', 'f_est_N', 'f_ati_N',
        'tau_cmd_Nm', 'tau_meas_Nm',
        'pos_rad', 'vel_rad_s',
        'motor_kp', 'motor_kd',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'sensing_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s, f_des, f_raw, f_ati,
            tau_cmd, tau_meas, pos, vel, motor_kp, motor_kd):
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_raw:.6f}', f'{f_ati:.6f}',
            f'{tau_cmd:.6f}', f'{tau_meas:.6f}',
            f'{pos:.6f}', f'{vel:.6f}',
            f'{motor_kp:.4f}', f'{motor_kd:.4f}',
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


def _control_worker(bus: MotorBus, motor: MITMotor,
                    plotter: PlotThread, logger: DataLogger) -> None:
    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        print(
            f"\nMotor {MOTOR_ID} ready — force sensing via motor current\n"
            f"  Jacobian : {JACOBIAN} m  →  F_est = τ_meas / J\n"
            f"  motor Kp={ctrl_params['motor_kp']}  Kd={ctrl_params['motor_kd']}\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = ctrl_params['motor_kp']
        motor.cmd.kd       = ctrl_params['motor_kd']
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0
        motor.cmd.torque   = 0.0

        loop = RealtimeLoop(dt=1.0 / LOOP_HZ, report=True, fade=0.5)
        _last_print = 0.0   # throttle console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                motor.update(t)

                f_des    = ctrl_params['f_des']
                motor_kp = ctrl_params['motor_kp']
                motor_kd = ctrl_params['motor_kd']

                motor.cmd.kp = motor_kp
                motor.cmd.kd = motor_kd

                # Feedforward: f_des → torque, clamped for safety
                tau_cmd = max(-TORQUE_LIMIT, min(TORQUE_LIMIT, f_des * JACOBIAN))
                motor.cmd.torque = tau_cmd * loop.fade

                tau_meas = motor.state.torque
                f_raw    = tau_meas / JACOBIAN
                f_ati    = float(-ft_latest[FORCE_AXIS])

                plotter.push(t, f_des, f_raw, f_ati, tau_cmd, tau_meas)

                logger.log(
                    t,
                    f_des, f_raw, f_ati,
                    tau_cmd, tau_meas,
                    motor.state.position, motor.state.velocity,
                    motor_kp, motor_kd,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_est={f_raw:+.4f}N  F_des={f_des:+.2f}N  "
                        f"τ_cmd={tau_cmd:+.3f}N·m  τ_meas={tau_meas:+.3f}N·m  "
                        f"mKp={motor_kp:.1f}  mKd={motor_kd:.2f}  "
                        f"pos={motor.state.position:+.4f}rad  "
                        f"T={motor.state.temp}°C"
                    )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


def main() -> None:
    print("Initialising ATI F/T sensor …")
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
