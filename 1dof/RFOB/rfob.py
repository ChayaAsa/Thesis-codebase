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
JACOBIAN   = 0.2     # lever arm [m]: tau_joint = JACOBIAN · F_ext [N·m]
FORCE_AXIS = 2       # ATI channel: 0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz
FORCE_SIGN = -1.0    # negate Fz so positive f_ati = compression

# ── Plant model  ⚠️  PLACEHOLDERS — run sysid_python.py first ─────────────────
# J_N ≤ J_actual required for RFOB stability (paper eq. 7 / Fig. 8b).
# Wrong J_N → force estimate offset during acceleration.
# Wrong B_N → sustained error proportional to omega (DOB reduces sensitivity).
J_N       = 0.01     # nominal rotor inertia [kg·m²] ← PLACEHOLDER
B_N       = 0.05     # nominal viscous friction [N·m·s/rad] ← PLACEHOLDER
RFOB_SIGN = 1.0      # flip to −1.0 if f_rfob sign is opposite to f_ati

# ── Observer bandwidths — wq_rfob MUST be > wq_dob (paper stability rule) ─────
# At 100 Hz (dt=0.01 s), Euler stability requires wq < 200 rad/s.
# Keep wq_dob ≤ 30, wq_rfob ≤ 90 for safe margins.
WQ_DOB    = 20.0     # DOB cutoff [rad/s] (~3 Hz)
WQ_RFOB   = 60.0     # RFOB cutoff [rad/s] (~10 Hz, 3× DOB per paper ratio)

# ── Velocity-P gain (converts admittance q̇_des → nominal torque) ─────────────
# Rule of thumb: KV ≈ J_N / tau_vel where tau_vel is desired velocity time
# constant.  J_N=0.01, tau_vel=0.02 s → KV=0.5.  Raise for stiffer tracking.
KV        = 0.5      # [N·m·s/rad]

# Force setpoint
F_DES_N   = 0.0      # desired contact force [N] (live-tunable via panel)

# Admittance outer loop
B_ADM     = 5.0      # virtual damping [N·m·s/rad]
K_SPRING  = 1.0      # restoring spring [N·m/rad]

# Signal processing
LPF_ALPHA  = 0.3     # IIR on RFOB output: y = α·x + (1−α)·y_prev

# Safety
VEL_LIMIT    = 2.0   # admittance velocity cap [rad/s]; guard fires at 1.5×
TORQUE_LIMIT = 3.0   # hardware torque cap [N·m]
TAU_CMP_MAX  = 2.0   # DOB compensation clip [N·m]

# Timing
LOOP_HZ       = 100
BIAS_DURATION = 5.0

HERE = os.path.dirname(os.path.abspath(__file__))

Terminate = threading.Event()
ft_latest = [0.0] * 6

ctrl_params = {
    'f_des':    float(F_DES_N),
    'b_adm':    float(B_ADM),
    'k_spring': float(K_SPRING),
    'kv':       float(KV),
    'lpf_alpha':float(LPF_ALPHA),
    'vel_limit':float(VEL_LIMIT),
    'wq_dob':   float(WQ_DOB),
    'wq_rfob':  float(WQ_RFOB),
}


# ── F/T reader thread  (ground truth — not used by controller) ───────────────

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
    UPDATE_MS = 100
    MAX_PTS   = int(WINDOW_S * LOOP_HZ * 1.5)

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._frame     = 0
        self._t         = collections.deque(maxlen=self.MAX_PTS)
        self._f_des     = collections.deque(maxlen=self.MAX_PTS)
        self._f_rfob    = collections.deque(maxlen=self.MAX_PTS)
        self._f_ati     = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_des  = collections.deque(maxlen=self.MAX_PTS)
        self._qdot_m    = collections.deque(maxlen=self.MAX_PTS)
        self._q         = collections.deque(maxlen=self.MAX_PTS)
        self._tau_des   = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmp   = collections.deque(maxlen=self.MAX_PTS)
        self._tau_cmd   = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float,
             f_des: float, f_rfob: float, f_ati: float,
             qdot_des: float, qdot_meas: float, q: float,
             tau_des: float, tau_cmp: float, tau_cmd: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des);   self._f_rfob.append(f_rfob)
            self._f_ati.append(f_ati)
            self._qdot_des.append(qdot_des); self._qdot_m.append(qdot_meas)
            self._q.append(q)
            self._tau_des.append(tau_des); self._tau_cmp.append(tau_cmp)
            self._tau_cmd.append(tau_cmd)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import TextBox

        fig, (ax_f, ax_v, ax_q, ax_tau) = plt.subplots(4, 1, figsize=(10, 11))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.30)
        fig.suptitle('RFOB Force Control — DOB inner + RFOB + Admittance (torque mode)',
                     fontsize=10)

        ax_f.set_ylabel('Force [N]');    ax_f.grid(True, alpha=0.3)
        ax_v.set_ylabel('q̇ [rad/s]');   ax_v.grid(True, alpha=0.3)
        ax_q.set_ylabel('q [rad]');      ax_q.grid(True, alpha=0.3)
        ax_tau.set_ylabel('τ [N·m]');    ax_tau.set_xlabel('Time [s]')
        ax_tau.grid(True, alpha=0.3)

        line_fdes,  = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_frfob, = ax_f.plot([], [], 'r-',  lw=1.5, label='F rfob (control)')
        line_fati,  = ax_f.plot([], [], color='# 2ca02c', lw=1.0,
                                alpha=0.8, ls='--', label='F ati (ground truth)')
        ax_f.legend(loc='upper left', fontsize=8)

        line_vdes,  = ax_v.plot([], [], 'b--', lw=1.5, label='q̇ des')
        line_vmeas, = ax_v.plot([], [], 'g-',  lw=1.5, label='q̇ meas')
        ax_v.legend(loc='upper left', fontsize=8)

        line_q, = ax_q.plot([], [], 'm-', lw=1.5, label='q')
        ax_q.axhline(0, color='k', lw=0.8, ls='--')
        ax_q.legend(loc='upper left', fontsize=8)

        line_tdes, = ax_tau.plot([], [], 'b--', lw=1.2, label='τ des')
        line_tcmp, = ax_tau.plot([], [], color='orange', lw=1.2, label='τ cmp (DOB)')
        line_tcmd, = ax_tau.plot([], [], 'r-',  lw=1.5, label='τ cmd (total)')
        ax_tau.axhline(0, color='k', lw=0.8, ls='--')
        ax_tau.legend(loc='upper left', fontsize=8)

        # TextBox: row1 = 6 params, row2 = 2 bandwidths
        row1_y, row2_y = 0.15, 0.05
        w, h = 0.11, 0.07
        axes_tb = {
            'f_des':     fig.add_axes([0.05, row1_y, w, h]),
            'b_adm':     fig.add_axes([0.19, row1_y, w, h]),
            'k_spring':  fig.add_axes([0.33, row1_y, w, h]),
            'kv':        fig.add_axes([0.47, row1_y, w, h]),
            'lpf_alpha': fig.add_axes([0.61, row1_y, w, h]),
            'vel_limit': fig.add_axes([0.75, row1_y, w, h]),
            'wq_dob':    fig.add_axes([0.05, row2_y, w, h]),
            'wq_rfob':   fig.add_axes([0.19, row2_y, w, h]),
        }
        labels = {
            'f_des':    'F_des [N]',
            'b_adm':    'B_adm',
            'k_spring': 'K_spring',
            'kv':       'KV [Nm·s/r]',
            'lpf_alpha':'LPF α',
            'vel_limit':'VelLim r/s',
            'wq_dob':   'wq_DOB r/s',
            'wq_rfob':  'wq_RFOB r/s',
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
                        elif k in ('wq_dob', 'wq_rfob'):
                            v = max(1.0, v)
                        elif k == 'kv':
                            v = max(0.0, v)
                        ctrl_params[k] = v
                        print(f"[panel] {k} → {v:.4f}")
                    except ValueError:
                        pass
                return _handler
            tb.on_submit(_make_handler(key))
            textboxes[key] = tb

        def _update(_):
            with self._lock:
                t       = list(self._t)
                f_des   = list(self._f_des)
                f_rfob  = list(self._f_rfob)
                f_ati   = list(self._f_ati)
                vd      = list(self._qdot_des)
                vm      = list(self._qdot_m)
                q       = list(self._q)
                tdes    = list(self._tau_des)
                tcmp    = list(self._tau_cmp)
                tcmd    = list(self._tau_cmd)

            if len(t) < 2:
                return (line_fdes, line_frfob, line_fati,
                        line_vdes, line_vmeas, line_q,
                        line_tdes, line_tcmp, line_tcmd)

            t_now, t_lo = t[-1], t[-1] - self.WINDOW_S
            line_fdes.set_data(t, f_des);   line_frfob.set_data(t, f_rfob)
            line_fati.set_data(t, f_ati)
            line_vdes.set_data(t, vd);      line_vmeas.set_data(t, vm)
            line_q.set_data(t, q)
            line_tdes.set_data(t, tdes);    line_tcmp.set_data(t, tcmp)
            line_tcmd.set_data(t, tcmd)

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
            if self._frame % 10 == 0:
                _ylim(ax_f, f_des, f_rfob, f_ati)
                _ylim(ax_v, vd, vm)
                _ylim(ax_q, q)
                _ylim(ax_tau, tdes, tcmp, tcmd)

            return (line_fdes, line_frfob, line_fati,
                    line_vdes, line_vmeas, line_q,
                    line_tdes, line_tcmp, line_tcmd)

        self._ani = animation.FuncAnimation(
            fig, _update, interval=self.UPDATE_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()


# CSV logger

class DataLogger:
    COLUMNS = [
        'time_s',
        'f_des_N', 'f_rfob_N', 'f_ati_N', 'f_err_N',
        'd_hat_dob_Nm', 'tau_des_Nm', 'tau_cmp_Nm', 'tau_cmd_Nm',
        'qdot_des_rad_s', 'qdot_meas_rad_s', 'q_rad',
        'b_adm', 'k_spring', 'kv', 'lpf_alpha', 'vel_limit',
        'wq_dob', 'wq_rfob',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'rfob_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, f_rfob: float, f_ati: float, f_err: float,
            d_hat_dob: float, tau_des: float, tau_cmp: float, tau_cmd: float,
            qdot_des: float, qdot_meas: float, q: float,
            b_adm: float, k_spring: float, kv: float,
            lpf_alpha: float, vel_limit: float,
            wq_dob: float, wq_rfob: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_rfob:.6f}', f'{f_ati:.6f}', f'{f_err:.6f}',
            f'{d_hat_dob:.6f}', f'{tau_des:.6f}',
            f'{tau_cmp:.6f}', f'{tau_cmd:.6f}',
            f'{qdot_des:.6f}', f'{qdot_meas:.6f}', f'{q:.6f}',
            f'{b_adm:.6f}', f'{k_spring:.6f}', f'{kv:.6f}',
            f'{lpf_alpha:.4f}', f'{vel_limit:.4f}',
            f'{wq_dob:.4f}', f'{wq_rfob:.4f}',
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

        print(
            f"Motor {MOTOR_ID} ready — PURE TORQUE MODE (kp=0, kd=0)\n"
            f"  DOB:   wq={WQ_DOB} rad/s  → cancels friction + inertia variation\n"
            f"  RFOB:  wq={WQ_RFOB} rad/s  → extracts F_ext  (wq_rfob > wq_dob ✓)\n"
            f"  ⚠️  J_N={J_N}, B_N={B_N} are PLACEHOLDERS — run sysid first!\n"
            f"  Admittance: q̇_des = (J·(F_des−f_rfob) − K·q) / B\n"
            f"  Velocity-P: tau_des = KV·(q̇_des − omega)  KV={KV}\n"
            f"  TORQUE_LIMIT={TORQUE_LIMIT} N·m  TAU_CMP_MAX={TAU_CMP_MAX} N·m\n"
            f"  VEL guard fires at |omega| > {VEL_LIMIT * 1.5:.1f} rad/s\n"
            f"Press Ctrl+C or close the plot to stop.\n"
        )

        # Pure torque mode — zero position/velocity stiffness
        motor.cmd.kp       = 0.0
        motor.cmd.kd       = 0.0
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0
        motor.cmd.torque   = 0.0

        # Observer states
        p_dob       = 0.0   # DOB Q-filter integrator
        p_rfob      = 0.0   # RFOB Q-filter integrator
        f_rfob_filt = 0.0   # IIR-smoothed RFOB force estimate [N]
        tau_cmd     = 0.0   # torque applied last tick (DOB input this tick)

        dt_nom = 1.0 / LOOP_HZ
        prev_t: float | None = None
        _last_print = 0.0

        loop = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)

        try:
            for t in loop:
                if Terminate.is_set():
                    loop.stop()
                    break

                dt = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Live params
                f_des     = ctrl_params['f_des']
                b_adm     = ctrl_params['b_adm']
                k_spring  = ctrl_params['k_spring']
                kv        = ctrl_params['kv']
                alpha     = ctrl_params['lpf_alpha']
                vel_limit = ctrl_params['vel_limit']
                wq_dob    = ctrl_params['wq_dob']
                wq_rfob   = ctrl_params['wq_rfob']

                # Motor state
                q     = motor.state.position
                omega = motor.state.velocity

                # Velocity guard (pure-torque safety)
                if abs(omega) > vel_limit * 1.5:
                    motor.cmd.torque = 0.0
                    motor.update(t)
                    tau_cmd = 0.0
                    print(f"[GUARD] |omega|={abs(omega):.2f} > {vel_limit*1.5:.1f} r/s — torque zeroed")
                    continue

                # DOB inner loop
                # Uses tau_cmd from the PREVIOUS tick (what was actually applied).
                # Estimates total disturbance: d_hat ≈ −B·omega − F_ext·r
                # tau_cmp cancels it → plant behaves as pure inertia J_N·alpha
                p_dob += (-wq_dob * p_dob
                          + J_N * wq_dob**2 * omega
                          + wq_dob * tau_cmd) * dt
                d_hat_dob = J_N * wq_dob * omega - p_dob
                tau_cmp   = float(np.clip(-d_hat_dob, -TAU_CMP_MAX, TAU_CMP_MAX))

                # Admittance outer loop
                f_err    = f_des - f_rfob_filt
                qdot_des = (JACOBIAN * f_err - k_spring * q) / b_adm
                qdot_des = float(np.clip(qdot_des, -vel_limit, vel_limit))

                # ── Velocity-P controller → nominal torque ────────────────────
                # Converts admittance velocity command to a torque demand.
                # DOB compensation makes this effective despite friction/inertia.
                tau_des = kv * (qdot_des - omega)

                # Total torque command
                tau_cmd_new = float(np.clip(
                    tau_des + tau_cmp, -TORQUE_LIMIT, TORQUE_LIMIT
                ))

                # RFOB
                # Uses tau_cmd_new (the torque we're about to send — same as
                # I^m = I^des + I^cmp in the paper's Fig. 1b).
                # wq_rfob > wq_dob per paper stability requirement (eq. 8).
                p_rfob += (-wq_rfob * p_rfob
                           + J_N * wq_rfob**2 * omega
                           + wq_rfob * (tau_cmd_new - B_N * omega)) * dt
                d_r_hat    = J_N * wq_rfob * omega - p_rfob
                f_rfob_raw = RFOB_SIGN * (-d_r_hat / JACOBIAN)
                f_rfob_filt = alpha * f_rfob_raw + (1.0 - alpha) * f_rfob_filt

                # ATI ground truth
                f_ati = FORCE_SIGN * float(ft_latest[FORCE_AXIS])

                # ── Motor command — pure torque ───────────────────────────────
                motor.cmd.torque = tau_cmd_new * loop.fade
                motor.update(t)

                # Store for next tick's DOB (what was actually applied this tick)
                tau_cmd = tau_cmd_new * loop.fade

                plotter.push(t, f_des, f_rfob_filt, f_ati,
                             qdot_des, motor.state.velocity, q,
                             tau_des, tau_cmp, tau_cmd)

                logger.log(
                    t,
                    f_des, f_rfob_filt, f_ati, f_err,
                    d_hat_dob, tau_des, tau_cmp, tau_cmd,
                    qdot_des, motor.state.velocity, q,
                    b_adm, k_spring, kv, alpha, vel_limit,
                    wq_dob, wq_rfob,
                )

                if t - _last_print >= 0.2:
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}N  "
                        f"F_rfob={f_rfob_filt:+.3f}N  "
                        f"F_ati={f_ati:+.3f}N  "
                        f"τ_des={tau_des:+.3f}  "
                        f"τ_cmp={tau_cmp:+.3f}  "
                        f"τ_cmd={tau_cmd:+.3f}N·m  "
                        f"ω={motor.state.velocity:+.3f}r/s  "
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
