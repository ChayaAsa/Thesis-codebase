from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np

from ATI_FTsensor.ftsensor import ftsensor
from tmotorcan import MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorFaultError, MotorTimeoutError

HERE = os.path.dirname(os.path.abspath(__file__))
from easy_path import WS_ROOT
_CTRL_ROOT = os.path.join(WS_ROOT, '3dof', '3_control')
if _CTRL_ROOT not in sys.path:
    sys.path.insert(0, _CTRL_ROOT)

from control_config import (PORT, MOTOR_IDS, JOINT_LIMITS, SIGN,
                             DEFAULT_KP, DEFAULT_KD,
                             Dynamic, DYN_CACHE,
                             cmd_free, cmd_lock,
                             build_motors, make_bus)
from helpers import KeyboardLine
from data_logger import DataLogger
from plot_thread import PlotThread, Line

dyn = Dynamic.get_or_build(DYN_CACHE)

# Columns specific to the admittance-impedance controller.
ADM_IMP_EXTRA_COLUMNS = [
    'f_des_N', 'f_raw_N', 'f_filt_N', 'f_err_N',
    'xdot_des_x', 'xdot_des_y', 'xdot_des_z',
    'qdot_des1_rad_s', 'qdot_des2_rad_s', 'qdot_des3_rad_s',
    'q_des1_rad', 'q_des2_rad', 'q_des3_rad',
    'kp1', 'kp2', 'kp3', 'kd1', 'kd2', 'kd3', 'm_adm', 'xdot_alpha', 'b_adm', 'k_adm', 'lpf_alpha', 'vel_limit',
]

# Hardware
COM_PORT      = PORT
BIAS_DURATION = 5

INIT_Q = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad

# ATI sensor axis
FORCE_AXIS  = 2
FORCE_SIGN  = -1

# Push direction
PUSH_DIR = np.array([1, 0, 0], dtype=float)
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# Force setpoint
F_DES_N     = 0.0


# 'cos' | 'random_step'

SIGNAL_MODE = 'random_step'       

# Cosine: f = A* sin(2π·w·t+pi/2)  — starts at 0
COS_AMPLITUDE = 8.0         # N
COS_FREQ_HZ   =  0.2        # Hz


STEP_LO      =  0.0         
STEP_HI      = 8.0         
STEP_DUR_LO  =  1.0         # s
STEP_DUR_HI  =  5.0         # s
RANDOM_SEED  =   42         


RECORD_ZERO_S     = 5.0     # s
RECORD_DURATION_S = 30.0    # s (0 = unlimited)
# ═══════════════════════════════════════════════════════════════════════════════

# ── B_adm * xdot_des = F_err_vec − K_spring * x_err ────────
B_ADM          = 20.0
M_ADM          = 0.0
K_ADM          = 2.0
XDOT_LPF_ALPHA = 1.0

# Motor MIT impedance gains (per-motor)
KP_MOTOR    = [DEFAULT_KP[id] for id in MOTOR_IDS]
KD_MOTOR    = [DEFAULT_KD[id] for id in MOTOR_IDS]

# Signal processing
LPF_ALPHA   = 0.1

# Safety
VEL_LIMIT   = 0.8
TORQUE_LIMIT = 3.0

# Timing
LOOP_HZ     = 100

# Joint safety limits from sysid_config (custom frame)
_JLO = np.array([JOINT_LIMITS[id][0] for id in MOTOR_IDS])
_JHI = np.array([JOINT_LIMITS[id][1] for id in MOTOR_IDS])

# ── Control-panel parameter definitions (key, label, lo, hi) ─────────────────
_PARAMS = [
    ('signal_active', f'Signal ON ({SIGNAL_MODE})', False),
    ('f_des',    'F const [N]', -50.0,  50.0),
    ('m_adm',      'M adm [kg]',  0.0,   5.0),
    ('xdot_alpha', 'xdot LPF α', 0.01,  1.0),
    ('b_adm',      'B adm',       0.1, 500.0),
    ('k_adm',    'K adm',       0.0, 200.0),
    ('kp1',      'Kp motor 1',  0.0,  50.0),
    ('kp2',      'Kp motor 2',  0.0,  50.0),
    ('kp3',      'Kp motor 3',  0.0,  50.0),
    ('kd1',      'Kd motor 1',  0.0,  10.0),
    ('kd2',      'Kd motor 2',  0.0,  10.0),
    ('kd3',      'Kd motor 3',  0.0,  10.0),
    ('lpf_alpha','LPF  alpha',  0.01,  1.0),
    ('vel_limit','Vel  [r/s]',  0.05, 10.0),
]

# Signal generator (pre-computed, seeded — always same sequence for same seed)

class SignalGenerator:

    def __init__(self) -> None:
        self._steps: list[tuple[float, float]] = []
        self.reset()

    def reset(self) -> None:
        if SIGNAL_MODE == 'random_step':
            rng = np.random.default_rng(RANDOM_SEED)
            self._steps = []
            t = 0.0
            while t < 600.0:
                level = float(rng.uniform(STEP_LO, STEP_HI))
                dur   = float(rng.uniform(STEP_DUR_LO, STEP_DUR_HI))
                t    += dur
                self._steps.append((level, t))

    def get(self, t_sig: float) -> float:
        if SIGNAL_MODE == 'cos':
            return (COS_AMPLITUDE / 2.0) * (1.0 - math.cos(2.0 * math.pi * COS_FREQ_HZ * t_sig))
        else:
            for level, end_t in self._steps:
                if t_sig < end_t:
                    return level
            return self._steps[-1][0] if self._steps else 0.0


# Shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6


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

# Control worker
_TIMEOUT_ABORT = 5


def setup(motors: list[MITMotor], sensor: ftsensor) -> bool:
    try:
        dt_boot = 1.0 / LOOP_HZ

        print("\n--- STARTUP ---")
        print(f"Step 1: Move arm to INIT_Q = {np.round(INIT_Q, 3)} rad, then press Enter to zero motors.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_free(motors)
            time.sleep(dt_boot)

        for m in motors:
            for attempt in range(3):
                if m.zero():
                    break
                print(f"  [warn] motor {m.id} zero no-reply (attempt {attempt+1}/3), retrying")
                time.sleep(0.05)
            else:
                print(f"  [ERROR] motor {m.id} zero command never ACK'd — q may not start at 0")
        print(f"  Motors zeroed at INIT_Q = {np.round(INIT_Q, 4)} rad.")

        print("Step 2: Arm holding init pose. Press Enter to bias ATI sensor.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print(f"  Biasing ATI for {BIAS_DURATION:.1f}s (keep sensor unloaded) ...")
        sensor.reBias(duration=BIAS_DURATION)
        print("  Bias complete.")

        print("Step 3: Press Enter to START admittance control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print("  Admittance control starting!\n")

        print(
            f"Config: F_des={F_DES_N}N  B_adm={B_ADM}  K_spring={K_ADM}\n"
            f"  Kp_motor={KP_MOTOR}  Kd_motor={KD_MOTOR}  VelLim={VEL_LIMIT} rad/s\n"
            f"  Arm FEELS force through motor current (MIT impedance mode).\n"
            f"Press Ctrl+C or close plot to stop.\n"
        )
        return True
    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s  (>{_TIMEOUT_ABORT} consecutive)")
        return False

def loop(motors: list[MITMotor],
         plotter: PlotThread, logger_box: list,
         signal_gen: SignalGenerator,
         ln_fdes: Line, ln_fraw: Line, ln_ffilt: Line,
         ln_qdes: list[Line], ln_qmeas: list[Line], ln_qdot: list[Line],
         ln_ddq_filt: list[Line]) -> None:
    try:
        q_des     = INIT_Q.copy()
        xdot_int  = np.zeros(3)
        xdot_filt = np.zeros(3)
        qdot_prev = np.zeros(3)
        ddq_filt  = np.zeros(3)
        f_filt    = FORCE_SIGN * ft_latest[FORCE_AXIS]
        dt_nom   = 1.0 / LOOP_HZ
        prev_t: float | None = None

        rt_loop     = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        # Signal / recording state
        prev_signal_active = False
        signal_start_t     = 0.0
        recording          = False
        record_start_t     = 0.0

        try:
            for t in rt_loop:
                if Terminate.is_set():
                    rt_loop.stop()
                    break

                # Record trigger (one-shot from Record button)
                if plotter.params.pop('record_trigger', False):
                    if logger_box[0] is not None:
                        logger_box[0].close()
                    logger_box[0] = DataLogger(
                        'adm_imp', ADM_IMP_EXTRA_COLUMNS, directory=HERE)
                    signal_gen.reset()
                    record_start_t = t
                    signal_start_t = t
                    recording      = True
                    plotter.params['signal_active'] = True
                    prev_signal_active = True

                # ── Signal active (checkbox) — detect rising edge ─────────
                signal_active = bool(plotter.params.get('signal_active', False))
                if signal_active and not prev_signal_active:
                    signal_start_t = t
                    signal_gen.reset()
                prev_signal_active = signal_active

                # ── F des: const / signal / recording ────────────────────
                if recording and not signal_active:   # user unchecked Signal ON → exit recording
                    recording = False
                if recording and RECORD_DURATION_S > 0 and (t - record_start_t) >= RECORD_DURATION_S:
                    recording = False
                if recording:
                    t_rec = t - record_start_t
                    if t_rec < RECORD_ZERO_S:
                        f_des = 0.0
                    else:
                        f_des = signal_gen.get(t_rec - RECORD_ZERO_S)
                elif signal_active:
                    f_des = signal_gen.get(t - signal_start_t)
                else:
                    f_des = plotter.params['f_des']

                # reset integrator if GUI button was pressed
                if plotter.params.pop('reset', False):
                    q_des         = INIT_Q.copy()
                    xdot_int[:]   = 0.0
                    xdot_filt[:]  = 0.0
                    qdot_prev[:]  = 0.0
                    ddq_filt[:]   = 0.0
                    f_filt        = FORCE_SIGN * ft_latest[FORCE_AXIS]

                m_adm      = float(plotter.params.get('m_adm', 0.0))
                xdot_alpha = float(plotter.params.get('xdot_alpha', 1.0))
                b_adm      = plotter.params['b_adm']
                k_adm      = plotter.params['k_adm']
                kp_motors = [plotter.params['kp1'], plotter.params['kp2'], plotter.params['kp3']]
                kd_motors = [plotter.params['kd1'], plotter.params['kd2'], plotter.params['kd3']]
                alpha     = plotter.params['lpf_alpha']
                vel_limit = plotter.params['vel_limit']

                dt = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Force measurement
                f_raw  = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_filt = alpha * f_raw + (1.0 - alpha) * f_filt
                f_err  = f_des - f_filt

                # ── Read joint state (absolute DH angles = INIT_Q + delta) ───
                q    = np.array([INIT_Q[i] + SIGN[id] * motors[i].state.position
                                  for i, id in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[id] * motors[i].state.velocity
                                  for i, id in enumerate(MOTOR_IDS)])

                ddq_raw      = (qdot - qdot_prev) / dt
                ddq_filt[:]  = 0.1 * ddq_raw + 0.9 * ddq_filt
                qdot_prev[:] = qdot

                # ── Admittance: task-space desired velocity ───────────────────
                if m_adm > 1e-6 or k_adm > 1e-12:
                    p_ee   = dyn.evaluate_fk(q)['ee_position']
                    p_home = dyn.evaluate_fk(INIT_Q)['ee_position']
                    x_err  = p_ee - p_home
                else:
                    x_err  = np.zeros(3)

                if m_adm > 1e-6:
                    xdot_filt  = xdot_alpha * xdot_int + (1.0 - xdot_alpha) * xdot_filt
                    xddot      = (f_err * PUSH_DIR - b_adm * xdot_filt - k_adm * x_err) / m_adm
                    xdot_int  += xddot * dt
                    xdot_des   = xdot_filt.copy()
                else:
                    xdot_des     = PUSH_DIR * f_err / b_adm - (k_adm / b_adm) * x_err
                    xdot_int[:]  = xdot_des
                    xdot_filt[:] = xdot_des

                # ── Joint velocity (q2, q3 only — q1 locked) ────────────────
                Jv               = dyn.evaluate_jacobian(q)[:3, :]
                Jv_23            = Jv[:, 1:]
                qdot_23, _, _, _ = np.linalg.lstsq(Jv_23, xdot_des, rcond=1e-3)
                qdot_des         = np.zeros(3)
                qdot_des[1:]     = qdot_23
                qdot_des = np.clip(qdot_des, -vel_limit, vel_limit)

                # Integrate q_des
                q_des += qdot_des * dt * rt_loop.fade
                q_des[0] = INIT_Q[0]
                q_des  = np.clip(q_des, INIT_Q + _JLO, INIT_Q + _JHI)

                # Motor command (MIT impedance mode)
                for i, id in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp       = kp_motors[i]
                    motors[i].cmd.kd       = kd_motors[i]
                    motors[i].cmd.position = float(SIGN[id] * (q_des[i] - INIT_Q[i]))
                    motors[i].cmd.velocity = float(SIGN[id] * qdot_des[i]) * rt_loop.fade
                    motors[i].cmd.torque   = 0.0
                    motors[i].update()

                tau_meas = np.array([SIGN[id] * motors[i].state.torque
                                      for i, id in enumerate(MOTOR_IDS)])

                # Plot
                plotter.ts = t
                ln_fdes.push(f_des);   ln_fraw.push(f_raw);  ln_ffilt.push(f_filt)
                for i in range(3):
                    ln_qdes[i].push(q_des[i])
                    ln_qmeas[i].push(q[i])
                    ln_qdot[i].push(qdot_des[i])
                    ln_ddq_filt[i].push(ddq_filt[i])
                if recording:
                    t_rec = t - record_start_t
                    _rec_str = (f'[⏺ REC {t_rec:.1f}s]  '
                                + ('ZERO' if t_rec < RECORD_ZERO_S
                                   else f'sig {SIGNAL_MODE}'))
                else:
                    _rec_str = '(not recording)'
                plotter.info = [
                    _rec_str,
                    f'Signal: {"ON" if signal_active else "OFF"}  ({SIGNAL_MODE})',
                    f'F filt:  {f_filt:+.3f} N',
                    f'F des:   {f_des:+.3f} N',
                    *[f'q{i+1}: meas {q[i]:+.3f}  des {q_des[i]:+.3f}' for i in range(3)],
                ]
                plotter.update()

                if logger_box[0] is not None:
                    logger_box[0].log(
                        time_s=t,
                        q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                        qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                        tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1], tau_meas3_Nm=tau_meas[2],
                        f_des_N=f_des, f_raw_N=f_raw, f_filt_N=f_filt, f_err_N=f_err,
                        xdot_des_x=xdot_des[0], xdot_des_y=xdot_des[1], xdot_des_z=xdot_des[2],
                        qdot_des1_rad_s=qdot_des[0], qdot_des2_rad_s=qdot_des[1], qdot_des3_rad_s=qdot_des[2],
                        q_des1_rad=q_des[0], q_des2_rad=q_des[1], q_des3_rad=q_des[2],
                        kp1=kp_motors[0], kp2=kp_motors[1], kp3=kp_motors[2],
                        kd1=kd_motors[0], kd2=kd_motors[1], kd3=kd_motors[2],
                        m_adm=m_adm, xdot_alpha=xdot_alpha, b_adm=b_adm,
                        k_adm=k_adm, lpf_alpha=alpha, vel_limit=vel_limit,
                    )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"F={f_filt:+.3f}N(des={f_des:+.2f})  "
                        f"qdot_des=[{qdot_des[0]:+.3f},{qdot_des[1]:+.3f},{qdot_des[2]:+.3f}]r/s  "
                        f"q_des=[{q_des[0]:+.3f},{q_des[1]:+.3f},{q_des[2]:+.3f}]  "
                        f"q=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}]  "
                        f"tau=[{tau_meas[0]:+.2f},{tau_meas[1]:+.2f},{tau_meas[2]:+.2f}]N·m  "
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
        m.update(timeout=2.0)

    signal_gen: SignalGenerator = SignalGenerator()
    logger_box: list = [None]

    # Line objects
    colors = ['steelblue', 'firebrick', 'goldenrod']
    ln_fdes  = Line(color='blue',    label='F des',  lw=1.5, ls='--')
    ln_fraw  = Line(color='# fd9999', label='F raw', lw=0.8, alpha=0.5)
    ln_ffilt = Line(color='red',     label='F filt', lw=1.5)
    ln_qdes     = [Line(color=colors[i], ls='--', lw=1.5, label=f'q{i+1} des')     for i in range(3)]
    ln_qmeas    = [Line(color=colors[i], lw=1.0,  alpha=0.7, label=f'q{i+1} meas') for i in range(3)]
    ln_qdot     = [Line(color=colors[i], lw=1.2,  label=f'qd{i+1}')               for i in range(3)]
    ln_ddq_filt = [Line(color=colors[i], lw=1.5, ls='--',   label=f'q̈{i+1} filt') for i in range(3)]

    def _on_reset() -> None:
        plotter.params['reset'] = True

    def _on_record() -> None:
        plotter.params['record_trigger'] = True

    plotter = PlotThread(
        title='3-DOF Admittance-Impedance Control',
        on_close=Terminate.set,
        on_reset=_on_reset,
        on_stop=Terminate.set,
        on_record=_on_record,
    )

    plotter.plot(1).set([ln_ffilt, ln_fdes, ln_fraw])
    plotter.plot(1).ylabel = 'Force [N]'
    plotter.plot(1).title  = 'Force'

    plotter.plot(2).set(ln_qdes + ln_qmeas)
    plotter.plot(2).ylabel = 'q [rad]'
    plotter.plot(2).title  = 'Joint positions'

    plotter.plot(3).set(ln_qdot)
    plotter.plot(3).ylabel = 'qdot_des [r/s]'
    plotter.plot(3).title  = 'Joint velocities'

    plotter.plot(4).set(ln_ddq_filt)
    plotter.plot(4).ylabel = 'q̈ [rad/s²]'
    plotter.plot(4).title  = 'Joint acceleration (filtered)'

    plotter.command = _PARAMS
    plotter.params.update({
        'signal_active': False,
        'f_des':     float(F_DES_N),
        'm_adm':      float(M_ADM),
        'xdot_alpha': float(XDOT_LPF_ALPHA),
        'b_adm':      float(B_ADM),
        'k_adm':     float(K_ADM),
        'kp1':       float(KP_MOTOR[0]),
        'kp2':       float(KP_MOTOR[1]),
        'kp3':       float(KP_MOTOR[2]),
        'kd1':       float(KD_MOTOR[0]),
        'kd2':       float(KD_MOTOR[1]),
        'kd3':       float(KD_MOTOR[2]),
        'lpf_alpha': float(LPF_ALPHA),
        'vel_limit': float(VEL_LIMIT),
    })

    complete = setup(motors, sensor)
    if not complete:
        Terminate.set()
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
        return

    ctrl_thread = threading.Thread(
        target=loop,
        args=(motors, plotter, logger_box, signal_gen,
              ln_fdes, ln_fraw, ln_ffilt, ln_qdes, ln_qmeas, ln_qdot,
              ln_ddq_filt),
        daemon=True, name='Control',
    )
    ctrl_thread.start()

    try:
        plotter.run()
    except KeyboardInterrupt:
        Terminate.set()
    finally:
        Terminate.set()
        ctrl_thread.join(timeout=3.0)
        if logger_box[0] is not None:
            logger_box[0].close()
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
