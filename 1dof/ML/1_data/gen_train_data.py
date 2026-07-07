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

HERE = os.path.dirname(os.path.abspath(__file__))

# signal mode
# Choose ONE: 'RANDOM_STEP' | 'SINE' | 'TRIANGLE' | 'SQUARE' | 'RAMP' | 'COMBINED'
SIGNAL_MODE = 'COMBINED'

# RANDOM_STEP
STEP_FORCE_MIN =  0.0   # [N] lower bound (snapped to 0.1 N grid)
STEP_FORCE_MAX =  5.0   # [N] upper bound
STEP_HOLD_MIN  =  1.0   # [s] minimum hold time per step
STEP_HOLD_MAX  =  5.0   # [s] maximum hold time per step
RNG_SEED       = 99     # fixed seed for reproducibility; None = random
#seed 42, 69, 123 combine:99
# SINE / TRIANGLE / SQUARE
WAVE_AMP    = 3.0       # [N] amplitude (peak)
WAVE_FREQ   = 0.2       # [Hz] wave frequency → period = 1/WAVE_FREQ
WAVE_OFFSET = 3.0       # [N] DC offset

# RAMP
RAMP_MIN  = 0.0        # [N] lower turnaround
RAMP_MAX  =  5.0        # [N] upper turnaround
RAMP_RATE =  1.0        # [N/s] ramp speed (always positive; direction reverses at bounds)

# ── COMBINED — list of (mode, duration_s), cycles indefinitely ────────────────
COMBINED_SEQ = [
    ('SINE',         30.0/3),
    ('RANDOM_STEP',  60.0/3),
    # ('RAMP',         30.0/3),
    ('TRIANGLE',     30.0/3),
    ('RANDOM_STEP',  60.0/3),
    ('SQUARE',       30.0/3),
    ('RANDOM_STEP',  60.0/3),
]

# motor / experiment params
JACOBIAN      = 0.2     # [m] lever arm; tau_cmd = f_des * JACOBIAN
TORQUE_LIMIT  = 4.0     # [N·m] hard motor torque cap
KD_MOT        = 0.1     # [N·m·s/rad] velocity damping
WARMUP_S      = 3.0     # [s] hold 0 N before signal starts
RUN_DURATION  = 120.0     # [s] total run time; 0 = run until Ctrl+C / window close
LOOP_HZ       = 100     # [Hz] control loop rate
BIAS_DURATION = 5.0     # [s] ATI sensor bias window (keep sensor unloaded)

# shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6   # [Fx, Fy, Fz, Tx, Ty, Tz]


# F/T reader
def ft_reader_thread(sensor: ftsensor) -> None:
    while not Terminate.is_set():
        try:
            ft_latest[:] = sensor.read_ft()
        except Exception as e:
            print(f"[ATI] read error: {e}")
            break
    print("[ATI] reader stopped")


# signal generator
class SignalGenerator:

    def __init__(self, mode: str, rng: np.random.Generator):
        self._mode = mode.upper()
        self._rng  = rng

        if self._mode == 'RANDOM_STEP':
            self._steps     = np.arange(STEP_FORCE_MIN, STEP_FORCE_MAX + 1e-9, 0.1)
            self._f_des     = 0.0
            self._t_next    = 0.0   # t_rel when to pick next value
            self._step_num  = 0

        elif self._mode == 'RAMP':
            span         = RAMP_MAX - RAMP_MIN
            self._period = 2.0 * span / RAMP_RATE   # full up+down cycle [s]

        elif self._mode == 'COMBINED':
            self._seg_idx = 0
            self._seg_t0  = 0.0
            mode0 = COMBINED_SEQ[0][0]
            self._cur_gen = SignalGenerator(mode0, rng)
            print(f"[combined] start -> {mode0}")

        elif self._mode not in ('SINE', 'TRIANGLE', 'SQUARE'):
            raise ValueError(
                f"Unknown SIGNAL_MODE: {self._mode!r}. "
                "Choose from RANDOM_STEP, SINE, TRIANGLE, SQUARE, RAMP, COMBINED."
            )

    # public call
    def __call__(self, t_rel: float) -> float:
        dispatch = {
            'RANDOM_STEP': self._random_step,
            'SINE':        self._sine,
            'TRIANGLE':    self._triangle,
            'SQUARE':      self._square,
            'RAMP':        self._ramp,
            'COMBINED':    self._combined,
        }
        return dispatch[self._mode](t_rel)

    # per-mode implementations
    def _random_step(self, t_rel: float) -> float:
        if t_rel >= self._t_next:
            self._f_des    = round(float(self._rng.choice(self._steps)), 1)
            hold           = float(self._rng.uniform(STEP_HOLD_MIN, STEP_HOLD_MAX))
            self._t_next   = t_rel + hold
            self._step_num += 1
            print(f"[step {self._step_num:3d}]  f_des={self._f_des:+.3f} N  hold={hold:.2f} s")
        return self._f_des

    def _sine(self, t_rel: float) -> float:
        return float(WAVE_OFFSET + WAVE_AMP * np.sin(2.0 * np.pi * WAVE_FREQ * t_rel))

    def _triangle(self, t_rel: float) -> float:
        period = 1.0 / WAVE_FREQ
        phase  = (t_rel % period) / period          # 0 → 1
        # rises 0→1 in first half, falls 1→0 in second half  →  scale to ±AMP
        tri    = 4.0 * abs(phase - 0.5) - 1.0       # -1 → +1 → -1
        return float(WAVE_OFFSET + WAVE_AMP * tri)

    def _square(self, t_rel: float) -> float:
        period = 1.0 / WAVE_FREQ
        phase  = (t_rel % period) / period
        sign   = 1.0 if phase < 0.5 else -1.0
        return float(WAVE_OFFSET + WAVE_AMP * sign)

    def _ramp(self, t_rel: float) -> float:
        span  = RAMP_MAX - RAMP_MIN
        phase = (t_rel % self._period) / self._period   # 0 → 1
        dur   = 0.9
        # phase 0→0.5 : RAMP_MIN → RAMP_MAX
        # phase 0.5→1 : RAMP_MAX → RAMP_MIN
        # if phase < 0.5:
        #     val = RAMP_MIN + 2.0 * phase * span
        # else:
        #     val = RAMP_MAX - 2.0 * (phase - 0.5) * span
        if phase < dur:
            val = RAMP_MIN + 1.0/dur *phase * span
        else:
            val = RAMP_MIN
        return float(val)

    def _combined(self, t_rel: float) -> float:
        # advance segment when current one has elapsed
        while True:
            _, dur = COMBINED_SEQ[self._seg_idx]
            if (t_rel - self._seg_t0) < dur:
                break
            self._seg_t0 += dur
            self._seg_idx = (self._seg_idx + 1) % len(COMBINED_SEQ)
            next_mode     = COMBINED_SEQ[self._seg_idx][0]
            self._cur_gen = SignalGenerator(next_mode, self._rng)
            print(f"[combined] -> {next_mode}")
        return self._cur_gen(t_rel - self._seg_t0)


# live plot
class PlotThread:
    WINDOW_S  = 12.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1800: only keep the visible window

    def __init__(self, mode: str):
        self._mode     = mode
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._f_des    = collections.deque(maxlen=self.MAX_PTS)
        self._f_raw    = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmd  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_meas = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t, f_des, f_raw, tau_cmd, tau_meas):
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
        fig.suptitle(f'Open-loop data gen — mode: {self._mode}  (no controller)', fontsize=11)

        ax_f.set_ylabel('Force [N]');  ax_f.set_xlabel('Time [s]');  ax_f.grid(True, alpha=0.3)
        line_fdes, = ax_f.plot([], [], 'b--', lw=1.5, label='f_des (cmd)')
        line_fraw, = ax_f.plot([], [], 'r-',  lw=1.0, label='f_raw (sensor)')
        ax_f.legend(loc='upper left', fontsize=8)

        ax_t.set_ylabel('Torque [N·m]'); ax_t.set_xlabel('Time [s]'); ax_t.grid(True, alpha=0.3)
        line_tcmd,  = ax_t.plot([], [], 'g-', lw=2.0, label='tau_cmd')
        line_tmeas, = ax_t.plot([], [], 'm-', lw=1.0, label='tau_meas')
        ax_t.legend(loc='upper left', fontsize=8)

        def _update(_):
            with self._lock:
                t = list(self._t); f_des = list(self._f_des); f_raw = list(self._f_raw)
                tau_cmd = list(self._tau_cmd); tau_meas = list(self._tau_meas)
            if len(t) < 2:
                return line_fdes, line_fraw, line_tcmd, line_tmeas
            t_now = t[-1]; t_lo = t_now - self.WINDOW_S
            line_fdes.set_data(t, f_des);   line_fraw.set_data(t, f_raw)
            line_tcmd.set_data(t, tau_cmd); line_tmeas.set_data(t, tau_meas)
            ax_f.set_xlim(t_lo, t_now + 0.1); ax_t.set_xlim(t_lo, t_now + 0.1)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s]
                if not vals: return
                lo, hi = min(vals), max(vals)
                pad = max(0.3, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            self._frame += 1
            if self._frame % 10 == 0:   # autoscale ~1 Hz at 10 fps
                _ylim(ax_f, f_des, f_raw); _ylim(ax_t, tau_cmd, tau_meas)
            return line_fdes, line_fraw, line_tcmd, line_tmeas

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS, blit=False, cache_frame_data=False)
        plt.show()


# data logger
class DataLogger:
    COLUMNS = ['time_s', 'f_des_N', 'f_raw_N', 'tau_cmd_Nm', 'tau_meas_Nm', 'pos_rad', 'vel_rad_s']

    def __init__(self, mode: str, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(HERE, exist_ok=True)
            path = os.path.join(HERE, f'{mode.lower()}_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] saving to {self.path}")

    def log(self, time_s, f_des, f_raw, tau_cmd, tau_meas, pos, vel) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}', f'{f_des:.6f}', f'{f_raw:.6f}',
            f'{tau_cmd:.6f}', f'{tau_meas:.6f}', f'{pos:.6f}', f'{vel:.6f}',
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
    rng     = np.random.default_rng(RNG_SEED)
    sig_gen = SignalGenerator(SIGNAL_MODE, rng)

    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Signal mode  : {SIGNAL_MODE}\n"
            f"  JACOBIAN     : {JACOBIAN} m\n"
            f"  Torque limit : +/-{TORQUE_LIMIT} N·m\n"
            f"  Warmup       : {WARMUP_S:.1f} s at 0 N\n"
            f"  NO control loop — direct open-loop torque commands\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp = 0.0;  motor.cmd.kd = KD_MOT
        motor.cmd.position = 0.0;  motor.cmd.velocity = 0.0

        dt_nom    = 1.0 / LOOP_HZ
        t_start   = None
        in_warmup = True
        f_des     = 0.0

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0   # throttle per-tick console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                if t_start is None:
                    t_start = t
                    print(f"[warmup]  0.000 N for {WARMUP_S:.1f} s ...")

                t_elapsed = t - t_start

                # warmup: hold 0 N
                if in_warmup:
                    if t_elapsed >= WARMUP_S:
                        in_warmup = False
                        print(f"[signal]  {SIGNAL_MODE} started")
                    f_des = 0.0
                else:
                    f_des = sig_gen(t_elapsed - WARMUP_S)

                # stop after RUN_DURATION if set
                if RUN_DURATION > 0 and t_elapsed >= RUN_DURATION:
                    print(f"\nRun duration {RUN_DURATION:.1f} s reached — stopping.")
                    loop.stop()
                    break

                tau_raw = f_des * JACOBIAN
                tau_cmd = float(np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT))
                motor.cmd.torque = tau_cmd * loop.fade

                f_raw = float(-ft_latest[2])

                plotter.push(t, f_des, f_raw, tau_cmd, motor.state.torque)
                motor.update(t)
                logger.log(t, f_des, f_raw, tau_cmd, motor.state.torque,
                           motor.state.position, motor.state.velocity)

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  f_des={f_des:+.3f}N  f_raw={f_raw:+.4f}N  "
                        f"tau_cmd={tau_cmd:+.4f}Nm  tau_meas={motor.state.torque:+.4f}Nm  "
                        f"pos={motor.state.position:+.4f}rad  vel={motor.state.velocity:+.4f}rad/s"
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

    raw_bus = can.interface.Bus(interface="slcan", channel=COM_PORT, bitrate=1_000_000)
    bus     = MotorBus(raw_bus)
    motor   = MITMotor(bus, motor_id=MOTOR_ID, model=MOTOR_MODEL)
    motor.enable()

    plotter = PlotThread(SIGNAL_MODE)
    logger  = DataLogger(SIGNAL_MODE)

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
