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

from control_config import (MOTOR_IDS, JOINT_LIMITS, SIGN,
                             DEFAULT_KP, DEFAULT_KD,
                             build_motors, make_bus, Dynamic, DYN_CACHE,
                             cmd_free, cmd_lock)
from helpers import KeyboardLine
from data_logger import DataLogger
from plot_thread import PlotThread, Line

dyn = Dynamic.get_or_build(DYN_CACHE)

RFOB_ADM_EXTRA_COLUMNS = [
    'f_des_N', 'f_rfob_N', 'f_sensor_N', 'f_err_N',
    'dob1_Nm', 'dob2_Nm', 'dob3_Nm',
    'rfob1_Nm', 'rfob2_Nm', 'rfob3_Nm',
    'tau_eff1_Nm', 'tau_eff2_Nm', 'tau_eff3_Nm',
    'q_des1_rad', 'q_des2_rad', 'q_des3_rad',
    'qdot_des1_rad_s', 'qdot_des2_rad_s', 'qdot_des3_rad_s',
    'g_dob', 'g_rfob', 'dob_comp', 'force_fb', 'm_adm', 'xdot_alpha', 'b_adm', 'k_adm',
]

# Hardware
BIAS_DURATION = 5.0

# Init pose
INIT_Q = np.array([0.0, 0.78, -0.78])

# ATI sensor
FORCE_AXIS = 2
FORCE_SIGN = -1.0

# Push direction
PUSH_DIR = np.array([1.0, 0.0, 0.0])
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# Force setpoint
F_DES_N = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Signal input — choose ONE: 'cos' | 'random_step'
# ═══════════════════════════════════════════════════════════════════════════════
SIGNAL_MODE = 'random_step'          # <── change here

COS_AMPLITUDE = 8.0         # N
COS_FREQ_HZ   =  0.2        # Hz

STEP_LO      =  0.0         # N
STEP_HI      = 8.0         # N
STEP_DUR_LO  =  1.0         # s
STEP_DUR_HI  =  5.0         # s
RANDOM_SEED  =   42

RECORD_ZERO_S     = 5.0     # s
RECORD_DURATION_S = 30.0    # s total recording length (0 = unlimited)
# ═══════════════════════════════════════════════════════════════════════════════

# Admittance model
B_ADM         = 20.0
M_ADM         = 0.0
K_ADM         = 2.0
XDOT_LPF_ALPHA = 1.0

# MIT impedance inner loop
KP_MOTOR = [DEFAULT_KP[mid] for mid in MOTOR_IDS]
KD_MOTOR = [DEFAULT_KD[mid] for mid in MOTOR_IDS]

# Signal processing
LPF_ALPHA = 0.3

# Safety
VEL_LIMIT    = 1.0
TORQUE_LIMIT = 3.0

# ── DOB + RFOB bandwidths [rad/s] ─────────────────────────────────────────────
G_DOB  = 20.0
G_RFOB = 50.0

# ── Nominal joint inertia [kg·m²] — diagonal of M(INIT_Q) ────────────────────
_M_init, _, _ = dyn.evaluate_MCG(INIT_Q, np.zeros(3))
J_N = np.diag(_M_init)
print(f"[RFOB_ADM] Nominal inertia from M(INIT_Q): {np.round(J_N, 5)} kg·m²")

# DOB compensation
USE_DOB_COMP = False

# Force feedback source
FORCE_FB    = 'sensor'   # 'rfob' | 'sensor' | 'fuse'
FUSE_WEIGHT = 0.5

# Timing
LOOP_HZ = 100

_JLO = np.array([JOINT_LIMITS[mid][0] for mid in MOTOR_IDS])
_JHI = np.array([JOINT_LIMITS[mid][1] for mid in MOTOR_IDS])

# Control-panel parameter definitions
_PARAMS = [
    ('signal_active', f'Signal ON ({SIGNAL_MODE})',    False),
    ('f_des',    'F const [N]',              -50.0,  50.0),
    ('m_adm',      'M adm [kg]',               0.0,   5.0),
    ('xdot_alpha', 'xdot LPF α',              0.01,  1.0),
    ('b_adm',      'B adm [N·s/m]',             0.1, 500.0),
    ('k_adm',    'K adm [N/m]',               0.0, 200.0),
    ('g_dob',    'g_DOB [rad/s]',             1.0, 300.0),
    ('g_rfob',   'g_RFOB [rad/s]',            1.0, 600.0),
    ('dob_comp', 'DOB comp (0=off 1=on)',     0.0,   1.0),
    ('force_fb', 'FB: 0=RFOB 1=ATI 2=fuse',  0.0,   2.0),
    ('fuse_w',   'Fuse w RFOB [0-1]',         0.0,   1.0),
    ('kp1',      'Kp motor 1',                0.0,  50.0),
    ('kp2',      'Kp motor 2',                0.0,  50.0),
    ('kp3',      'Kp motor 3',                0.0,  50.0),
    ('kd1',      'Kd motor 1',                0.0,  10.0),
    ('kd2',      'Kd motor 2',                0.0,  10.0),
    ('kd3',      'Kd motor 3',                0.0,  10.0),
    ('lpf_alpha','ATI LPF α',               0.01,   1.0),
    ('vel_limit','Vel  [r/s]',              0.05,  10.0),
]

# Signal generator

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


# Startup sequence

_TIMEOUT_ABORT = 5


def setup(motors: list[MITMotor], sensor: ftsensor) -> bool:
    try:
        dt_boot = 1.0 / LOOP_HZ

        print("\n--- STARTUP ---")
        print(f"Step 1: Move arm to INIT_Q = {np.round(INIT_Q, 3)} rad, then press Enter.")
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

        print("Step 2: Holding init pose. Press Enter to bias ATI sensor.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print(f"  Biasing ATI for {BIAS_DURATION:.1f}s (keep sensor unloaded) ...")
        sensor.reBias(duration=int(BIAS_DURATION))
        print("  Bias complete.")

        print("Step 3: Press Enter to START RFOB+DOB admittance control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print(
            f"\nConfig: F_des={F_DES_N}N  B_adm={B_ADM}  K_spring={K_ADM}\n"
            f"  g_DOB={G_DOB} rad/s  g_RFOB={G_RFOB} rad/s\n"
            f"  J_N={np.round(J_N, 4)} kg·m²  DOB_comp={USE_DOB_COMP}  FB={FORCE_FB}\n"
            f"  Kp_motor={KP_MOTOR}  Kd_motor={KD_MOTOR}\n"
            f"Press Ctrl+C or close plot to stop.\n"
        )
        return True
    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s  (>{_TIMEOUT_ABORT} consecutive)")
        return False


# Control loop

def loop(motors: list[MITMotor],
         plotter: PlotThread, logger_box: list,
         signal_gen: SignalGenerator,
         ln_fdes: Line, ln_frfob: Line, ln_fsen: Line,
         ln_rfob: list[Line], ln_qdes: list[Line], ln_qmeas: list[Line],
         ln_ddq_filt: list[Line]) -> None:
    try:
        q_des    = INIT_Q.copy()
        qdot_des = np.zeros(3)
        xdot_int  = np.zeros(3)
        xdot_filt = np.zeros(3)
        qdot_prev = np.zeros(3)
        ddq_filt  = np.zeros(3)

        # Observer states
        dob_state    = np.zeros(3)
        rfob_state   = np.zeros(3)
        tau_eff_prev = np.zeros(3)

        f_sen_filt = FORCE_SIGN * ft_latest[FORCE_AXIS]

        dt_nom  = 1.0 / LOOP_HZ
        prev_t: float | None = None
        lp      = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        # Signal / recording state
        prev_signal_active = False
        signal_start_t     = 0.0
        recording          = False
        record_start_t     = 0.0

        try:
            for t in lp:
                if Terminate.is_set():
                    lp.stop()
                    break

                # Record trigger
                if plotter.params.pop('record_trigger', False):
                    if logger_box[0] is not None:
                        logger_box[0].close()
                    logger_box[0] = DataLogger(
                        'rfob_adm', RFOB_ADM_EXTRA_COLUMNS, directory=HERE)
                    signal_gen.reset()
                    record_start_t = t
                    signal_start_t = t
                    recording      = True
                    plotter.params['signal_active'] = True
                    prev_signal_active = True

                # ── Signal active — detect rising edge ────────────────────
                signal_active = bool(plotter.params.get('signal_active', False))
                if signal_active and not prev_signal_active:
                    signal_start_t = t
                    signal_gen.reset()
                prev_signal_active = signal_active

                # F des
                if recording and not signal_active:   # user unchecked Signal ON → exit recording
                    recording = False
                if recording and RECORD_DURATION_S > 0 and (t - record_start_t) >= RECORD_DURATION_S:
                    recording = False
                if recording:
                    t_rec = t - record_start_t
                    f_des = 0.0 if t_rec < RECORD_ZERO_S else signal_gen.get(t_rec - RECORD_ZERO_S)
                elif signal_active:
                    f_des = signal_gen.get(t - signal_start_t)
                else:
                    f_des = plotter.params['f_des']

                # Live params
                m_adm      = float(plotter.params.get('m_adm', 0.0))
                xdot_alpha = float(plotter.params.get('xdot_alpha', 1.0))
                b_adm      = plotter.params['b_adm']
                k_adm      = plotter.params['k_adm']
                g_dob     = plotter.params['g_dob']
                g_rfob    = plotter.params['g_rfob']
                dob_comp  = plotter.params['dob_comp'] > 0.5
                force_fb  = int(round(plotter.params['force_fb']))
                fuse_w    = plotter.params['fuse_w']
                kp_motors = [plotter.params['kp1'], plotter.params['kp2'], plotter.params['kp3']]
                kd_motors = [plotter.params['kd1'], plotter.params['kd2'], plotter.params['kd3']]
                ati_alpha = plotter.params['lpf_alpha']
                vel_limit = plotter.params['vel_limit']

                if plotter.params.pop('reset', False):
                    dob_state[:]     = 0.0
                    rfob_state[:]    = 0.0
                    tau_eff_prev[:]  = 0.0
                    q_des[:]         = INIT_Q.copy()
                    qdot_des[:]      = 0.0
                    xdot_int[:]      = 0.0
                    xdot_filt[:]     = 0.0
                    qdot_prev[:]     = 0.0
                    ddq_filt[:]      = 0.0
                    f_sen_filt       = FORCE_SIGN * ft_latest[FORCE_AXIS]

                dt     = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Joint state (DH frame)
                q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                                  for i, mid in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[mid] * motors[i].state.velocity
                                  for i, mid in enumerate(MOTOR_IDS)])

                ddq_raw      = (qdot - qdot_prev) / dt
                ddq_filt[:]  = 0.1 * ddq_raw + 0.9 * ddq_filt
                qdot_prev[:] = qdot

                # ── Model: gravity + Jacobian ─────────────────────────────
                _, _, G_torq = dyn.evaluate_MCG(q, np.zeros(3))
                Jv = dyn.evaluate_jacobian(q)[:3, :]

                # DOB update
                alpha_dob = np.exp(-g_dob * dt)
                u_dob     = (tau_eff_prev - G_torq) - J_N * g_dob * qdot
                dob_state = alpha_dob * dob_state + (1.0 - alpha_dob) * u_dob

                # RFOB update
                alpha_rfob = np.exp(-g_rfob * dt)
                u_rfob     = tau_eff_prev - G_torq - J_N * g_rfob * qdot
                rfob_state = alpha_rfob * rfob_state + (1.0 - alpha_rfob) * u_rfob

                # Cartesian force from RFOB
                cond = np.linalg.cond(Jv)
                if cond < 1e6:
                    F_cart_hat = np.linalg.solve(Jv.T, rfob_state)
                else:
                    F_cart_hat = np.linalg.pinv(Jv.T) @ rfob_state
                f_rfob = float(PUSH_DIR @ F_cart_hat)

                # ── ATI sensor (filtered, reference) ─────────────────────
                f_raw      = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_sen_filt = ati_alpha * f_raw + (1.0 - ati_alpha) * f_sen_filt

                # Force feedback selection
                if force_fb == 0:
                    f_measured = f_rfob
                elif force_fb == 1:
                    f_measured = f_sen_filt
                else:
                    f_measured = fuse_w * f_rfob + (1.0 - fuse_w) * f_sen_filt

                # Admittance
                f_err    = f_des - f_measured
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

                # ── Joint velocity (q2, q3 only — q1 locked) ─────────────
                Jv_23            = Jv[:, 1:]
                qdot_23, _, _, _ = np.linalg.lstsq(Jv_23, xdot_des, rcond=1e-3)
                qdot_des         = np.zeros(3)
                qdot_des[1:]     = qdot_23
                qdot_des         = np.clip(qdot_des, -vel_limit, vel_limit)

                # Integrate q_des
                q_des   += qdot_des * dt * lp.fade
                q_des[0] = INIT_Q[0]
                q_des    = np.clip(q_des, INIT_Q + _JLO, INIT_Q + _JHI)

                # Optional DOB compensation via q_des shift
                if dob_comp:
                    for i in range(3):
                        if kp_motors[i] > 1e-6:
                            q_des[i] += dob_state[i] / kp_motors[i]
                    q_des[0] = INIT_Q[0]
                    q_des    = np.clip(q_des, INIT_Q + _JLO, INIT_Q + _JHI)

                # Motor command (MIT impedance mode)
                for i, mid in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp       = kp_motors[i]
                    motors[i].cmd.kd       = kd_motors[i]
                    motors[i].cmd.position = float(SIGN[mid] * (q_des[i] - INIT_Q[i]))
                    motors[i].cmd.velocity = float(SIGN[mid] * qdot_des[i]) * lp.fade
                    motors[i].cmd.torque   = 0.0
                    motors[i].update()

                tau_eff_prev = np.clip(
                    np.array([
                        kp_motors[i] * (q_des[i] - q[i]) + kd_motors[i] * (qdot_des[i] - qdot[i])
                        for i in range(3)
                    ]),
                    -TORQUE_LIMIT, TORQUE_LIMIT,
                )

                tau_meas = np.array([SIGN[mid] * motors[i].state.torque
                                      for i, mid in enumerate(MOTOR_IDS)])

                # Plot
                plotter.ts = t
                ln_fdes.push(f_des)
                ln_frfob.push(f_rfob)
                ln_fsen.push(f_sen_filt)
                for i in range(3):
                    ln_rfob[i].push(float(rfob_state[i]))
                    ln_qdes[i].push(q_des[i])
                    ln_qmeas[i].push(q[i])
                    ln_ddq_filt[i].push(ddq_filt[i])
                _src = 'ATI' if force_fb == 1 else ('Fuse' if force_fb == 2 else 'RFOB')
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
                    f'[{_src}]  active: {f_measured:+.3f} N',
                    f'RFOB: {f_rfob:+.3f} N',
                    f'ATI:  {f_sen_filt:+.3f} N',
                    f'Des:  {f_des:+.3f} N',
                    *[f'q{i+1}: meas {q[i]:+.3f}  des {q_des[i]:+.3f}' for i in range(3)],
                ]
                plotter.update()

                if logger_box[0] is not None:
                    logger_box[0].log(
                        time_s=t,
                        q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                        qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                        tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1], tau_meas3_Nm=tau_meas[2],
                        f_des_N=f_des, f_rfob_N=f_rfob, f_sensor_N=f_sen_filt, f_err_N=f_err,
                        dob1_Nm=dob_state[0], dob2_Nm=dob_state[1], dob3_Nm=dob_state[2],
                        rfob1_Nm=rfob_state[0], rfob2_Nm=rfob_state[1], rfob3_Nm=rfob_state[2],
                        tau_eff1_Nm=tau_eff_prev[0], tau_eff2_Nm=tau_eff_prev[1], tau_eff3_Nm=tau_eff_prev[2],
                        q_des1_rad=q_des[0], q_des2_rad=q_des[1], q_des3_rad=q_des[2],
                        qdot_des1_rad_s=qdot_des[0], qdot_des2_rad_s=qdot_des[1], qdot_des3_rad_s=qdot_des[2],
                        g_dob=g_dob, g_rfob=g_rfob,
                        dob_comp=float(dob_comp), force_fb=float(force_fb),
                        m_adm=m_adm, xdot_alpha=xdot_alpha, b_adm=b_adm, k_adm=k_adm,
                    )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"RFOB={f_rfob:+.3f}N  ATI={f_sen_filt:+.3f}N  des={f_des:+.2f}N  "
                        f"err={f_err:+.3f}  "
                        f"q_des=[{q_des[0]:+.3f},{q_des[1]:+.3f},{q_des[2]:+.3f}]  "
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

    colors   = ['steelblue', 'firebrick', 'goldenrod']
    ln_fdes  = Line(color='blue',    label='F des',      lw=1.5, ls='--')
    ln_frfob = Line(color='red',     label='RFOB est.',  lw=1.5)
    ln_fsen  = Line(color='# aaaaaa', label='ATI ref.', lw=1.0, alpha=0.7)
    ln_rfob     = [Line(color=colors[i], lw=1.5, label=f'τ_l̂{i+1}')               for i in range(3)]
    ln_qdes     = [Line(color=colors[i], lw=1.5, ls='--', label=f'q{i+1} des')    for i in range(3)]
    ln_qmeas    = [Line(color=colors[i], lw=1.0, alpha=0.65, label=f'q{i+1} meas') for i in range(3)]
    ln_ddq_filt = [Line(color=colors[i], lw=1.5, ls='--',   label=f'q̈{i+1} filt') for i in range(3)]

    def _on_reset() -> None:
        plotter.params['reset'] = True

    def _on_record() -> None:
        plotter.params['record_trigger'] = True

    plotter = PlotThread(
        title='3-DOF RFOB+DOB Admittance Control',
        on_close=Terminate.set,
        on_reset=_on_reset,
        on_stop=Terminate.set,
        on_record=_on_record,
    )

    plotter.plot(1).set([ln_fdes, ln_frfob, ln_fsen])
    plotter.plot(1).ylabel = 'Force [N]'
    plotter.plot(1).title  = 'Force'

    plotter.plot(2).set(ln_rfob)
    plotter.plot(2).ylabel = 'τ_contact [N·m]'
    plotter.plot(2).title  = 'RFOB joint torques'

    plotter.plot(3).set(ln_qdes + ln_qmeas)
    plotter.plot(3).ylabel = 'q [rad]'
    plotter.plot(3).title  = 'Joint positions'

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
        'g_dob':     float(G_DOB),
        'g_rfob':    float(G_RFOB),
        'dob_comp':  float(int(USE_DOB_COMP)),
        'force_fb':  float({'rfob': 0, 'sensor': 1, 'fuse': 2}[FORCE_FB]),
        'fuse_w':    float(FUSE_WEIGHT),
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
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        print("Startup failed.")
        return

    ctrl_thread = threading.Thread(
        target=loop,
        args=(motors, plotter, logger_box, signal_gen,
              ln_fdes, ln_frfob, ln_fsen, ln_rfob, ln_qdes, ln_qmeas,
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
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
