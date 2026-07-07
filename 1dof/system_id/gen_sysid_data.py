from __future__ import annotations

import collections
import csv
import os
import queue
import threading
import time

import can
import numpy as np
from tmotorcan import MotorBus, MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorTimeoutError, MotorFaultError
from ATI_FTsensor.ftsensor import ftsensor

# hardware
COM_PORT    = 'COM18'
MOTOR_ID    = 1
MOTOR_MODEL = 'AK45-10'

HERE     = os.path.dirname(os.path.abspath(__file__))

# experiment config
FORCE_MIN    = -5.0     # [N] lower bound for random force step
FORCE_MAX    =  5.0     # [N] upper bound
HOLD_MIN     =  1.0     # [s] minimum hold time per step
HOLD_MAX     =  5.0     # [s] maximum hold time per step

JACOBIAN     = 0.2      # [m] effective lever arm; tau = f_des * JACOBIAN
TORQUE_LIMIT = 3.0      # [N·m] hard motor torque cap
KD_MOT       = 0.1      # [N·m·s/rad] small velocity damping to prevent runaway

RNG_SEED     = 42       # fixed seed for reproducible step sequence; None = random
RUN_DURATION = 0.0      # [s] 0 = run until Ctrl+C / window closed
LOOP_HZ      = 100      # [Hz] control frequency
BIAS_DURATION= 5.0      # [s] ATI sensor bias window (keep unloaded)

# shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6     # [Fx, Fy, Fz, Tx, Ty, Tz]

# F/T reader
def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


# live plot
class PlotThread:
    WINDOW_S  = 12.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1800: only keep the visible window

    def __init__(self):
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw    = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmd  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_meas = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float, f_des: float, f_raw: float,
             tau_cmd: float, tau_meas: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_raw.append(f_raw)
            self._tau_cmd.append(tau_cmd)
            self._tau_meas.append(tau_meas)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(10, 7))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08)
        fig.suptitle('Open-loop sysID — random force steps (no controller)', fontsize=11)

        ax_f.set_ylabel('Force [N]')
        ax_f.set_xlabel('Time [s]')
        ax_f.grid(True, alpha=0.3)
        line_fdes, = ax_f.plot([], [], 'b--', lw=1.5, label='f_des (step cmd)')
        line_fraw, = ax_f.plot([], [], 'r-',  lw=1.0, label='f_raw (sensor)')
        ax_f.legend(loc='upper left', fontsize=8)

        ax_t.set_ylabel('Torque [N·m]')
        ax_t.set_xlabel('Time [s]')
        ax_t.grid(True, alpha=0.3)
        line_tcmd,  = ax_t.plot([], [], 'g-',  lw=2.0, label='tau_cmd')
        line_tmeas, = ax_t.plot([], [], 'm-',  lw=1.0, label='tau_meas')
        ax_t.legend(loc='upper left', fontsize=8)

        def _update(_):
            with self._lock:
                t        = list(self._t)
                f_des    = list(self._f_des)
                f_raw    = list(self._f_raw)
                tau_cmd  = list(self._tau_cmd)
                tau_meas = list(self._tau_meas)

            if len(t) < 2:
                return line_fdes, line_fraw, line_tcmd, line_tmeas

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_fdes.set_data(t, f_des)
            line_fraw.set_data(t, f_raw)
            line_tcmd.set_data(t, tau_cmd)
            line_tmeas.set_data(t, tau_meas)

            ax_f.set_xlim(t_lo, t_now + 0.1)
            ax_t.set_xlim(t_lo, t_now + 0.1)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s]
                if not vals:
                    return
                lo, hi = min(vals), max(vals)
                pad = max(0.3, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            self._frame += 1
            if self._frame % 10 == 0:   # autoscale ~1 Hz at 10 fps
                _ylim(ax_f, f_des, f_raw)
                _ylim(ax_t, tau_cmd, tau_meas)
            return line_fdes, line_fraw, line_tcmd, line_tmeas

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


# data logger
class DataLogger:
    COLUMNS = [
        'time_s',
        'f_des_N', 'f_raw_N',
        'tau_cmd_Nm', 'tau_meas_Nm',
        'pos_rad', 'vel_rad_s',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(HERE)
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'sysid_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] saving to {self.path}")

    def log(self, time_s: float,
            f_des: float, f_raw: float,
            tau_cmd: float, tau_meas: float,
            pos: float, vel: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}',   f'{f_raw:.6f}',
            f'{tau_cmd:.6f}', f'{tau_meas:.6f}',
            f'{pos:.6f}',     f'{vel:.6f}',
        ))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        print(f"[log] saved -> {self.path}")

    def _writer(self) -> None:
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)


# control worker
def _control_worker(bus: MotorBus, motor: MITMotor,
                    plotter: PlotThread, logger: DataLogger) -> None:
    rng = np.random.default_rng(RNG_SEED)

    FORCE_STEPS = np.arange(FORCE_MIN, FORCE_MAX + 1e-9, 0.1)   # [-2.0, -1.9, ..., 2.0]

    def next_step() -> tuple[float, float]:
        f   = float(rng.choice(FORCE_STEPS))
        dur = rng.uniform(HOLD_MIN, HOLD_MAX)
        return round(f, 1), float(dur)

    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Force range  : [{FORCE_MIN:+.1f}, {FORCE_MAX:+.1f}] N\n"
            f"  Hold range   : [{HOLD_MIN:.1f}, {HOLD_MAX:.1f}] s\n"
            f"  JACOBIAN     : {JACOBIAN} m\n"
            f"  Torque limit : +/-{TORQUE_LIMIT} N·m\n"
            f"  NO control loop — direct open-loop torque commands\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = 0.0
        motor.cmd.kd       = KD_MOT
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0

        WARMUP_S  = 3.0       # hold 0 N before random steps begin

        dt_nom    = 1.0 / LOOP_HZ
        t_start   = None
        t_step    = 0.0       # time when current step started
        f_des     = 0.0
        hold_dur  = 0.0
        step_num  = 0
        in_warmup = True

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0   # throttle per-tick console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                if t_start is None:
                    t_start  = t
                    t_step   = t
                    f_des    = 0.0
                    hold_dur = WARMUP_S
                    print(f"[warmup]         f_des= 0.000 N  hold={WARMUP_S:.2f} s")

                # transition from warmup to random steps
                if in_warmup and (t - t_step) >= WARMUP_S:
                    in_warmup = False
                    t_step    = t
                    f_des, hold_dur = next_step()
                    step_num  = 1
                    print(f"[step {step_num:3d}]  f_des={f_des:+.3f} N  hold={hold_dur:.2f} s")

                # advance to next random step when hold time expires
                elif not in_warmup and (t - t_step) >= hold_dur:
                    t_step = t
                    f_des, hold_dur = next_step()
                    step_num += 1
                    print(f"[step {step_num:3d}]  f_des={f_des:+.3f} N  hold={hold_dur:.2f} s")

                # stop after RUN_DURATION if configured
                if RUN_DURATION > 0 and (t - t_start) >= RUN_DURATION:
                    print(f"\nRun duration {RUN_DURATION:.1f} s reached — stopping.")
                    loop.stop()
                    break

                # pure open-loop: tau_cmd = f_des * JACOBIAN, no feedback
                tau_raw = f_des * JACOBIAN
                tau_cmd = float(np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT))
                motor.cmd.torque = tau_cmd * loop.fade

                f_raw = float(-ft_latest[2])    # Fz channel, sign convention from test1.py

                plotter.push(t, f_des, f_raw, tau_cmd, motor.state.torque)
                motor.update(t)

                logger.log(
                    t,
                    f_des, f_raw,
                    tau_cmd, motor.state.torque,
                    motor.state.position, motor.state.velocity,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"f_des={f_des:+.3f}N  "
                        f"f_raw={f_raw:+.4f}N  "
                        f"tau_cmd={tau_cmd:+.4f}Nm  "
                        f"tau_meas={motor.state.torque:+.4f}Nm  "
                        f"pos={motor.state.position:+.4f}rad  "
                        f"vel={motor.state.velocity:+.4f}rad/s"
                    )

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


# main
def main() -> None:

    print("Initialising ATI F/T sensor ...")
    sensor = ftsensor()
    sensor.start_task()
    time.sleep(0.5)
    print(f"Biasing F/T sensor for {BIAS_DURATION:.1f} s — keep sensor unloaded ...")
    sensor.reBias(duration=BIAS_DURATION)
    print("Bias complete.")
    time.sleep(1)

    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("F/T background reader started.")

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
        plotter._run()          # blocks on main thread (Tk requirement)
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
