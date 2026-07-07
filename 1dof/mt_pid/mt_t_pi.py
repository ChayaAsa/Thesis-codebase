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


COM_PORT     = 'COM18'       # ← change to your USB-CAN port
MOTOR_ID     = 1             # ← CAN ID of your motor
MOTOR_MODEL  = 'AK45-10'    # ← motor model name or path to custom .yaml

HERE         = os.path.dirname(os.path.abspath(__file__))

F_DESIRED_N  = 0.0           # target force [N] → τ_des = F_desired × JACOBIAN

JACOBIAN     = 0.2           # effective lever arm [m]

KP_T         = 1.0           # proportional torque gain [dimensionless]
KI_T         = 0.0           # integral torque gain [1/s]

LPF_ALPHA    = 0.3           # torque feedback IIR: 1.0=raw, ~0.1=heavy (~5 Hz @ 100 Hz)

TORQUE_LIMIT = 3.0           # hard torque cap [N·m]

LOOP_HZ      = 100           # control frequency [Hz]


# Tunable controller params — updated live by the plot-thread TextBoxes,
ctrl_params = {
    'f_des':     float(F_DESIRED_N),
    'kp_t':      float(KP_T),
    'ki_t':      float(KI_T),
    'lpf_alpha': float(LPF_ALPHA),
    'reset':     False,
}

Terminate = threading.Event()


# Real-time plot thread

class PlotThread:

    WINDOW_S  = 10.0
    UPDATE_MS = 100                            # 10 Hz redraw (was 50 → 20 Hz)
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)  # ~1500: only keep the visible window

    def __init__(self):
        self._lock     = threading.Lock()
        self._frame    = 0   # redraw counter
        self._t        = collections.deque(maxlen=self.MAX_PTS)
        self._tau_des  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_raw  = collections.deque(maxlen=self.MAX_PTS)
        self._tau_filt = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmd  = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float, tau_des: float, tau_raw: float,
             tau_filt: float, tau_cmd: float) -> None:
        with self._lock:
            self._t.append(t)
            self._tau_des.append(tau_des)
            self._tau_raw.append(tau_raw)
            self._tau_filt.append(tau_filt)
            self._tau_cmd.append(tau_cmd)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_tau, ax_cmd) = plt.subplots(2, 1, figsize=(10, 7))
        fig.subplots_adjust(left=0.10, right=0.88, top=0.93, bottom=0.23)
        fig.suptitle('1-DOF Torque PI — semi-open-loop force', fontsize=11)

        # ── Top panel: torque tracking ────────────────────────────────────────
        ax_tau.set_ylabel('Torque [N·m]')
        ax_tau.set_xlabel('Time [s]')
        ax_tau.grid(True, alpha=0.3)
        line_tdes,  = ax_tau.plot([], [], 'b--', lw=1.5,  label='τ desired')
        line_traw,  = ax_tau.plot([], [], color="# fd9999", lw=0.8, alpha=0.6, label='τ meas (raw)')
        line_tfilt, = ax_tau.plot([], [], 'r-',  lw=1.5,  label='τ meas (filtered)')
        ax_tau.legend(loc='upper left', fontsize=8)

        # secondary axis — force equivalent
        ax_f2 = ax_tau.twinx()
        ax_f2.set_ylabel(f'Force equiv. [N]  (J={JACOBIAN} m)', fontsize=8)

        # ── Bottom panel: control output ──────────────────────────────────────
        ax_cmd.set_ylabel('τ cmd [N·m]')
        ax_cmd.set_xlabel('Time [s]')
        ax_cmd.grid(True, alpha=0.3)
        line_cmd, = ax_cmd.plot([], [], 'g-', lw=2.0, label='τ cmd')
        ax_cmd.legend(loc='upper left', fontsize=8)

        # Input panel
        ax_tb_fdes = fig.add_axes([0.05, 0.07, 0.17, 0.08])
        ax_tb_kpt  = fig.add_axes([0.27, 0.07, 0.17, 0.08])
        ax_tb_kit  = fig.add_axes([0.49, 0.07, 0.17, 0.08])
        ax_tb_lpf  = fig.add_axes([0.71, 0.07, 0.17, 0.08])

        tb_fdes = TextBox(ax_tb_fdes, 'F_des [N]', initial=str(ctrl_params['f_des']),
                          color='lightyellow', hovercolor='yellow')
        tb_kpt  = TextBox(ax_tb_kpt,  'Kp_t     ', initial=str(ctrl_params['kp_t']),
                          color='lightyellow', hovercolor='yellow')
        tb_kit  = TextBox(ax_tb_kit,  'Ki_t [/s]', initial=str(ctrl_params['ki_t']),
                          color='lightyellow', hovercolor='yellow')
        tb_lpf  = TextBox(ax_tb_lpf,  'LPF α    ', initial=str(ctrl_params['lpf_alpha']),
                          color='lightcyan',   hovercolor='cyan')

        def _on_fdes(val):
            try:
                ctrl_params['f_des'] = float(val)
                ctrl_params['reset'] = True
                print(f"[panel] F_des → {ctrl_params['f_des']:.4f} N  "
                      f"(τ_des = {ctrl_params['f_des']*JACOBIAN:.4f} N·m, integral reset)")
            except ValueError:
                pass

        def _on_kpt(val):
            try:
                ctrl_params['kp_t'] = float(val)
                print(f"[panel] Kp_t → {ctrl_params['kp_t']:.6f}")
            except ValueError:
                pass

        def _on_kit(val):
            try:
                ctrl_params['ki_t'] = float(val)
                ctrl_params['reset'] = True
                print(f"[panel] Ki_t → {ctrl_params['ki_t']:.6f}  (integral reset)")
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
        tb_kpt.on_submit(_on_kpt)
        tb_kit.on_submit(_on_kit)
        tb_lpf.on_submit(_on_lpf)

        # Animation
        def _update(_):
            with self._lock:
                t        = list(self._t)
                tau_des  = list(self._tau_des)
                tau_raw  = list(self._tau_raw)
                tau_filt = list(self._tau_filt)
                tau_cmd  = list(self._tau_cmd)

            if len(t) < 2:
                return line_tdes, line_traw, line_tfilt, line_cmd

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            line_tdes.set_data(t, tau_des)
            line_traw.set_data(t, tau_raw)
            line_tfilt.set_data(t, tau_filt)
            line_cmd.set_data(t, tau_cmd)

            ax_tau.set_xlim(t_lo, t_now + 0.1)
            ax_cmd.set_xlim(t_lo, t_now + 0.1)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s]
                if not vals:
                    return
                lo, hi = min(vals), max(vals)
                pad = max(0.1, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            self._frame += 1
            if self._frame % 10 == 0:   # autoscale ~1 Hz at 10 fps
                _ylim(ax_tau, tau_des, tau_raw, tau_filt)
                _ylim(ax_cmd, tau_cmd)

                # keep secondary force axis in sync
                lo_t, hi_t = ax_tau.get_ylim()
                ax_f2.set_ylim(lo_t / JACOBIAN, hi_t / JACOBIAN)

            return line_tdes, line_traw, line_tfilt, line_cmd

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


class DataLogger:

    COLUMNS = [
        'time_s',
        'f_des_N', 'tau_des_Nm',
        'tau_raw_Nm', 'tau_filt_Nm', 'tau_cmd_Nm',
        'tau_err_Nm', 'integral_Nm_s',
        'pos_rad', 'vel_rad_s', 'accel_rad_s2',
        'kp_t', 'ki_t', 'lpf_alpha',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'mt_t_pi_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, tau_des: float,
            tau_raw: float, tau_filt: float, tau_cmd: float,
            tau_err: float, integral: float,
            pos: float, vel: float, accel: float,
            kp_t: float, ki_t: float, lpf_alpha: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{tau_des:.6f}',
            f'{tau_raw:.6f}', f'{tau_filt:.6f}', f'{tau_cmd:.6f}',
            f'{tau_err:.6f}', f'{integral:.6f}',
            f'{pos:.6f}', f'{vel:.6f}', f'{accel:.6f}',
            f'{kp_t:.6f}', f'{ki_t:.6f}', f'{lpf_alpha:.4f}',
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

        tau_des_init = F_DESIRED_N * JACOBIAN
        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  Target force  : {F_DESIRED_N:+.2f} N\n"
            f"  Target torque : {tau_des_init:+.4f} N·m  (= F × J)\n"
            f"  Kp_t={KP_T}  Ki_t={KI_T}\n"
            f"  Jacobian      : {JACOBIAN} m\n"
            f"  Motor kp=0  kd=0.1  (pure torque mode)\n"
            f"  Torque limit  : ±{TORQUE_LIMIT} N·m\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = 0.0
        motor.cmd.kd       = 0.1
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0
        motor.cmd.torque   = 0.0

        # Prime: one update to populate motor.state.torque
        motor.update(time.monotonic())
        tau_init = motor.state.torque if motor.state.torque is not None else 0.0

        integral  = 0.0
        tau_filt  = tau_init          # IIR filter state
        dt_nom    = 1.0 / LOOP_HZ
        prev_t: float | None = None

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0   # throttle console output to ~5 Hz

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                # ── Send previous command, receive fresh motor state ──────────
                motor.update(t)

                # Read controller params (live-tunable from UI)
                f_des = ctrl_params['f_des']
                kp_t  = ctrl_params['kp_t']
                ki_t  = ctrl_params['ki_t']
                alpha = ctrl_params['lpf_alpha']

                # ── Desired torque — open-loop F→τ conversion ─────────────────
                tau_des = f_des * JACOBIAN

                if ctrl_params['reset']:
                    integral = 0.0
                    tau_filt = motor.state.torque
                    ctrl_params['reset'] = False

                # Torque feedback (motor current sense) + LPF
                tau_raw  = motor.state.torque
                tau_filt = alpha * tau_raw + (1.0 - alpha) * tau_filt

                # PI on torque error
                dt_actual = (t - prev_t) if prev_t is not None else dt_nom
                prev_t    = t

                tau_err   = tau_des - tau_filt
                integral += tau_err * dt_actual
                windup_lim = TORQUE_LIMIT / ki_t if ki_t > 1e-12 else 1e6
                integral   = float(np.clip(integral, -windup_lim, windup_lim))

                tau_raw_cmd = kp_t * tau_err + ki_t * integral
                tau_cmd     = float(np.clip(tau_raw_cmd, -TORQUE_LIMIT, TORQUE_LIMIT))

                motor.cmd.torque = tau_cmd * loop.fade

                # ── Log & display ─────────────────────────────────────────────
                plotter.push(t, tau_des, tau_raw, tau_filt, tau_cmd)

                logger.log(
                    t,
                    f_des, tau_des,
                    tau_raw, tau_filt, tau_cmd,
                    tau_err, integral,
                    motor.state.position, motor.state.velocity, motor.state.acceleration,
                    kp_t, ki_t, alpha,
                )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  "
                        f"τ_des={tau_des:+.4f}N·m  "
                        f"τ_meas={tau_filt:+.4f}N·m  "
                        f"τ_err={tau_err:+.4f}N·m  "
                        f"intg={integral:+.5f}  "
                        f"τ_cmd={tau_cmd:+.3f}N·m  "
                        f"kp_t={kp_t:.4f} ki_t={ki_t:.4f}  "
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
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
