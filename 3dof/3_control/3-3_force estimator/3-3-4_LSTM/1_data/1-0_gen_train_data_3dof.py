from __future__ import annotations

import os
import random
import sys
import threading
import time

import numpy as np

# Workspace paths
HERE = os.path.dirname(os.path.abspath(__file__))

from easy_path import WS_ROOT
_CTRL_ROOT = os.path.join(WS_ROOT, '3dof', '3_control')
if _CTRL_ROOT not in sys.path:
    sys.path.insert(0, _CTRL_ROOT)

from control_config import (MOTOR_IDS, JOINT_LIMITS, SIGN,
                             DEFAULT_KP, DEFAULT_KD,
                             build_motors, make_bus, Dynamic, DYN_CACHE,
                             cmd_free, cmd_lock)
from helpers     import KeyboardLine
from data_logger import DataLogger
from plot_thread import PlotThread, Line

from ATI_FTsensor.ftsensor import ftsensor
from tmotorcan import MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorFaultError, MotorTimeoutError

# ── CONFIG — change between sessions ─────────────────────────────────────────

PHASE   = 'B'    # 'A' = free-space sweep | 'B' = contact push
DRY_RUN = False

# Arm pose

INIT_Q    = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad
PUSH_DIR  = np.array([1., 0., 0.])          # push direction in robot base frame

# ATI axis that aligns with PUSH_DIR (same convention as gmo_cur_adm.py)
FORCE_AXIS = 2       # ATI Fz
FORCE_SIGN = -1.0    # robot +X ≡ −ATI_z

BIAS_DURATION = 5.0  # ATI bias settle time [s]
ATI_LPF_ALPHA = 1.0  # display-only filter on f_push; raw values are always logged

DURATION = 60.0
# ── Signal defaults  (all tunable live from the panel) ───────────────────────

# Phase A — independent sine per joint (different frequencies, 120° offset)
# amp unit: rad/s  (panel label changes to match)
FREQS_A   = np.array([0.10, 0.25, 0.10])    # Hz, one per joint
OFFSETS_A = np.array([0.0,  2.094, 4.189])  # rad (120° phase shifts)
AMP_A_DEF = 1.0                              # Phase A sweep fraction 0..1 (1 = full open↔near)

# Phase A sweep endpoints (joint-space, rad). You init at the wall pose (INIT_Q),
# then Phase A sweeps q2/q3 back and forth between these two poses so the model
# sees the WHOLE range from open air to just-short-of-contact. q1 stays locked.
#   POSE_A_OPEN — fully retracted, arm hangs in open air (ATI ≈ 0)
#   POSE_A_NEAR — closest to the wall WITHOUT pressing it (watch f_push ≈ 0)
# Tune both per your rig. Keep POSE_A_NEAR just short of contact.
POSE_A_OPEN     = np.array([0.0, 1.57, -1.57])  # rad
POSE_A_NEAR     = np.array([0.0, 0.78, -0.78])  # rad
RETRACT_A_TIME  = 3.0                            # s to ramp from INIT_Q to POSE_A_OPEN before sweeping

# Phase B — sine or random-step force command along PUSH_DIR
# f_des oscillates between F_LO_B and F_HI_B [N]
# q_des is solved each tick via Jacobian transpose so kp*(q_des-q) ≈ f_des at EEF
FREQ_B_DEF = 0.20    # Hz
F_LO_B     = -4.0     # N minimum desired contact force (panel: f_lo_N)
F_HI_B     = 12.0    # N maximum desired contact force (panel: f_hi_N)

# Phase B signal mode: 'sine' = fixed/swept-frequency sine (below);
#                      'step' = random force step, random hold duration
SIGNAL_B = 'step'

# Sine mode: linear frequency sweep low -> high over the full session DURATION.
# Set both equal to FREQ_B_DEF for a fixed-frequency sine (old behaviour).
SWEEP_B = True        # False -> use fixed FREQ_B_DEF instead of sweeping
FREQ_B_LO = 0.10      # Hz at t=0
FREQ_B_HI = 0.50      # Hz at t=DURATION

# Step mode: force level and hold duration are both re-randomized each step
STEP_F_LO   = -4.0   # N min random step level
STEP_F_HI   = 12.0   # N max random step level
STEP_DUR_LO = 1.0      # s min hold duration
STEP_DUR_HI = 4.0      # s max hold duration

RANDOM_SEED = 69
random.seed(RANDOM_SEED)


def sweep_phase_rad(t: float) -> tuple[float, float]:
    t_clamped = min(max(t, 0.0), DURATION)
    slope = (FREQ_B_HI - FREQ_B_LO) / DURATION
    freq_now = FREQ_B_LO + slope * t_clamped
    phase = 2.0 * np.pi * (FREQ_B_LO * t_clamped + 0.5 * slope * t_clamped * t_clamped)
    return phase, freq_now


# Motor / safety

KP_MOTOR      = [DEFAULT_KP[mid] for mid in MOTOR_IDS]
KD_MOTOR      = [DEFAULT_KD[mid] for mid in MOTOR_IDS]
TAU_LPF_ALPHA = 0.3
VEL_LIMIT     = 0.5   # rad/s per joint
LOOP_HZ       = 100

_Q_LO = np.array([JOINT_LIMITS[mid][0] for mid in MOTOR_IDS])
_Q_HI = np.array([JOINT_LIMITS[mid][1] for mid in MOTOR_IDS])

# ── Dynamic model (loaded once, lambdified — cheap per-tick) ─────────────────

dyn = Dynamic.get_or_build(DYN_CACHE)

# Control-panel parameters

if PHASE == 'A':
    _signal_params = [('amp', 'Sweep frac', 0.0, 1.0)]
    _signal_defs   = {'amp': AMP_A_DEF}
    _freq_def      = FREQS_A[1]
else:
    # Phase B: panel exposes force range directly; f_des sine oscillates between them
    _signal_params = [
        ('f_lo_N', 'F_lo [N]', 0.0, 50.0),
        ('f_hi_N', 'F_hi [N]', 0.0, 50.0),
    ]
    _signal_defs   = {'f_lo_N': F_LO_B, 'f_hi_N': F_HI_B}
    _freq_def      = FREQ_B_DEF

_PARAMS = _signal_params + [
    ('freq_hz',       'Freq [Hz]',    0.05,    2.0),
    ('tau_lpf_alpha', 'τ LPF α',      0.01,    1.0),
    ('kp1',           'Kp motor 1',   0.0,    50.0),
    ('kp2',           'Kp motor 2',   0.0,    50.0),
    ('kp3',           'Kp motor 3',   0.0,    50.0),
    ('kd1',           'Kd motor 1',   0.0,    10.0),
    ('kd2',           'Kd motor 2',   0.0,    10.0),
    ('kd3',           'Kd motor 3',   0.0,    10.0),
    ('vel_limit',     'Vel  [r/s]',   0.05,    2.0),
]

# ── CSV extra columns  (LOG_COMMON_COLUMNS already has q, qdot, tau_meas) ────

EXTRA_COLUMNS = [
    # 6-axis ATI in sensor frame (labels used by 3-D training path)
    'ati_fx_N',  'ati_fy_N',  'ati_fz_N',
    'ati_mx_Nm', 'ati_my_Nm', 'ati_mz_Nm',
    # scalar 1-D label (no frame rotation — directly −ATI_z)
    'f_push_N',
    # dynamics for physics-assisted (Mode B) training
    'G1_Nm', 'G2_Nm', 'G3_Nm',
    'Cqdot1_Nm', 'Cqdot2_Nm', 'Cqdot3_Nm',
    # inner-loop setpoints
    'q_des1_rad',      'q_des2_rad',      'q_des3_rad',
    'qdot_des1_rad_s', 'qdot_des2_rad_s', 'qdot_des3_rad_s',
    # estimated commanded torque (diagnostic)
    'tau_cmd1_Nm', 'tau_cmd2_Nm', 'tau_cmd3_Nm',
    # metadata
    'phase',
]

# Shared state

Terminate = threading.Event()
ft_latest = [0.0] * 6    # [Fx, Fy, Fz, Mx, My, Mz] ATI sensor frame


# ATI reader thread

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

def setup(motors: list[MITMotor], sensor: ftsensor) -> bool:
    try:
        dt_boot = 1.0 / LOOP_HZ

        print(f"\n--- STARTUP  (Phase {PHASE}) ---")
        print(f"INIT_Q = {np.round(INIT_Q, 3)} rad")
        print("Step 1: Move arm to INIT_Q in free mode, then press Enter.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_free(motors)
            time.sleep(dt_boot)

        for m in motors:
            for attempt in range(3):
                if m.zero():
                    break
                print(f"  [warn] motor {m.id} zero no-reply (attempt {attempt+1}/3)")
                time.sleep(0.05)
            else:
                print(f"  [ERROR] motor {m.id} zero never ACK'd")
        print("  Encoders zeroed at INIT_Q.")

        print("Step 2: Holding pose. Press Enter to bias ATI.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print(f"  Biasing ATI for {BIAS_DURATION:.1f}s — keep sensor unloaded ...")
        sensor.reBias(duration=int(BIAS_DURATION))
        print("  Bias complete.")

        if PHASE == 'B':
            print("Step 3: Manually bring arm into LIGHT contact with wall,")
            print("        then press Enter to start recording.")
        else:
            print("Step 3: Press Enter to start recording.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print("Recording started.  Close plot or Ctrl+C to stop.\n")
        return True

    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")
        return False


# Control + data-collection loop

def loop(motors: list[MITMotor],
         plotter:  PlotThread,
         logger:   DataLogger | None,
         ln_fpush: Line,
         ln_fx: Line, ln_fy: Line, ln_fz: Line,
         ln_qdes:  list[Line], ln_qmeas: list[Line],
         ln_tau:   list[Line],
         ln_jit:   Line) -> None:

    q_des    = INIT_Q.copy()
    qdot_des = np.zeros(3)

    # Warm-start torque LPF at gravity so it doesn't spike on tick 1
    _, _, G_init  = dyn.evaluate_MCG(INIT_Q, np.zeros(3))
    tau_meas_filt = G_init.copy()

    f_push_disp = 0.0   # filtered for display only

    # Step-mode state (Phase B, SIGNAL_B == 'step')
    step_level    = random.uniform(STEP_F_LO, STEP_F_HI)
    step_next_t   = random.uniform(STEP_DUR_LO, STEP_DUR_HI)

    dt_nom     = 1.0 / LOOP_HZ
    prev_t: float | None = None
    _last_print = 0.0
    lp = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)

    try:
        for t in lp:
            if Terminate.is_set() or t >= DURATION:
                lp.stop()
                break

            # Live params
            freq          = float(plotter.params['freq_hz'])
            tau_lpf_alpha = float(plotter.params['tau_lpf_alpha'])
            kp_motors     = [plotter.params['kp1'],
                              plotter.params['kp2'],
                              plotter.params['kp3']]
            kd_motors     = [plotter.params['kd1'],
                              plotter.params['kd2'],
                              plotter.params['kd3']]
            vel_limit     = float(plotter.params['vel_limit'])

            dt     = (t - prev_t) if prev_t is not None else dt_nom
            prev_t = t

            # Joint state (DH frame)
            q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                              for i, mid in enumerate(MOTOR_IDS)])
            qdot = np.array([SIGN[mid] * motors[i].state.velocity
                              for i, mid in enumerate(MOTOR_IDS)])

            # ── Measured torque (DH frame), LPF ──────────────────────────────
            tau_raw       = np.array([SIGN[mid] * motors[i].state.torque
                                       for i, mid in enumerate(MOTOR_IDS)])
            tau_meas_filt = (tau_lpf_alpha * tau_raw
                             + (1.0 - tau_lpf_alpha) * tau_meas_filt)

            # Dynamics
            _, C, G = dyn.evaluate_MCG(q, qdot)
            Cqdot   = C @ qdot

            # ATI reading
            ft = list(ft_latest)                            # snapshot 6-axis
            f_push  = FORCE_SIGN * ft[FORCE_AXIS]           # 1-D label
            f_push_disp = ATI_LPF_ALPHA * f_push + (1.0 - ATI_LPF_ALPHA) * f_push_disp

            # ── Sine signal → q_des ──────────────────────────────────────────
            if PHASE == 'A':
                # Free-space: first ramp OFF the wall (INIT_Q → POSE_A_OPEN over
                # RETRACT_A_TIME), then sweep q2/q3 back and forth between
                # POSE_A_OPEN (open air) and POSE_A_NEAR (just short of the wall).
                # s in [0,1]: 0 = open air, 1 = near wall.  q1 stays locked.
                # 'amp' (panel) scales sweep size 0..1 so you can shrink it live
                # if f_push starts rising (i.e. it began touching the wall).
                amp      = float(plotter.params['amp'])
                sweep_f  = FREQS_A[1]                       # single sweep frequency [Hz]
                s        = 0.5 * (1.0 - np.cos(2.0 * np.pi * sweep_f * t))  # 0→1→0
                s        = amp * s                          # amp acts as sweep fraction
                if RETRACT_A_TIME > 0 and t < RETRACT_A_TIME:
                    # smooth approach to the open-air pose before sweeping
                    ramp  = t / RETRACT_A_TIME
                    q_des = INIT_Q + ramp * (POSE_A_OPEN - INIT_Q)
                else:
                    q_des = POSE_A_OPEN + s * (POSE_A_NEAR - POSE_A_OPEN)
                q_des[0] = INIT_Q[0]   # lock q1
                q_des    = np.clip(q_des, _Q_LO, _Q_HI)
                qdot_des = np.zeros(3)
                f_des    = 0.0   # free-space; for display only
                freq_now = 0.0   # unused in Phase A; kept defined for the info line

            else:  # PHASE == 'B'
                # Force-based: q_des is solved each tick via J^T so
                # kp*(q_des−q) ≈ f_des at EEF.
                freq_now = 0.0   # only meaningful for SIGNAL_B == 'sine'; shown in info line
                if SIGNAL_B == 'step':
                    # Random force level, held for a random duration, then
                    # re-rolled — both level and hold picked fresh each step.
                    if t >= step_next_t:
                        step_level  = random.uniform(STEP_F_LO, STEP_F_HI)
                        step_next_t = t + random.uniform(STEP_DUR_LO, STEP_DUR_HI)
                    f_des = step_level
                else:
                    # Sine oscillates between f_lo_N and f_hi_N. Frequency is
                    # either fixed (freq_hz panel) or swept low->high (SWEEP_B).
                    f_lo  = float(plotter.params['f_lo_N'])
                    f_hi  = float(plotter.params['f_hi_N'])
                    phase, freq_now = sweep_phase_rad(t) if SWEEP_B else (2.0 * np.pi * freq * t, freq)
                    f_des = (0.5 * (f_hi + f_lo)
                             + 0.5 * (f_hi - f_lo) * np.sin(phase))
                Jv         = dyn.evaluate_jacobian(q)[:3, :]
                # Motor must cancel gravity PLUS provide the push torque.
                # tau_motor = G(q) + J^T @ (f_des * PUSH_DIR)
                # kp*(q_des - q) = tau_motor  →  q_des = q + tau_motor/kp
                tau_need   = G + Jv.T @ (f_des * PUSH_DIR)
                kp_arr     = np.maximum(np.array(kp_motors, dtype=float), 0.1)
                q_des      = q + tau_need / kp_arr
                q_des[0]   = INIT_Q[0]   # lock q1
                q_des      = np.clip(q_des, _Q_LO, _Q_HI)
                qdot_des   = np.zeros(3)  # pure position spring; no velocity ff

            # Estimated commanded torque (diagnostic log)
            tau_cmd = np.array([
                kp_motors[i] * (q_des[i] - q[i]) + kd_motors[i] * (qdot_des[i] - qdot[i])
                for i in range(3)
            ])

            # Motor command (MIT impedance)
            for i, mid in enumerate(MOTOR_IDS):
                motors[i].cmd.kp       = kp_motors[i]
                motors[i].cmd.kd       = kd_motors[i]
                motors[i].cmd.position = float(SIGN[mid] * (q_des[i] - INIT_Q[i]))
                motors[i].cmd.velocity = float(SIGN[mid] * qdot_des[i]) * lp.fade
                motors[i].cmd.torque   = 0.0
                motors[i].update()

            # Plot
            plotter.ts = t
            ln_fpush.push(f_push_disp)
            ln_fx.push(ft[0]);  ln_fy.push(ft[1]);  ln_fz.push(ft[2])
            for i in range(3):
                ln_qdes[i].push(q_des[i])
                ln_qmeas[i].push(q[i])
                ln_tau[i].push(tau_raw[i])
            ln_jit.push((dt - dt_nom) * 1e3)

            if PHASE == 'B':
                _b_sig_line = f'f_des   = {f_des:+.2f} N   ({SIGNAL_B}, freq={freq_now:.3f} Hz)' \
                    if SIGNAL_B == 'sine' else f'f_des   = {f_des:+.2f} N   ({SIGNAL_B})'
            else:
                _b_sig_line = f'amp = {plotter.params.get("amp", 0):.3f} rad/s'
            plotter.info = [
                f'Phase {PHASE}   t={t:.1f}s / {DURATION:.0f}s',
                _b_sig_line,
                f'f_push  = {f_push_disp:+.3f} N',
                f'F_ati   = [{ft[0]:+.2f}, {ft[1]:+.2f}, {ft[2]:+.2f}] N',
                f'M_ati   = [{ft[3]:+.2f}, {ft[4]:+.2f}, {ft[5]:+.2f}] N·m',
                f'q_meas  = [{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}]',
                f'q_des   = [{q_des[0]:+.3f}, {q_des[1]:+.3f}, {q_des[2]:+.3f}]',
                f'τ_raw   = [{tau_raw[0]:+.2f}, {tau_raw[1]:+.2f}, {tau_raw[2]:+.2f}] N·m',
            ]
            plotter.update()

            # Log
            if logger is not None:
                logger.log(
                    time_s          = t,
                    q1_rad          = q[0],       q2_rad          = q[1],       q3_rad          = q[2],
                    qdot1_rad_s     = qdot[0],    qdot2_rad_s     = qdot[1],    qdot3_rad_s     = qdot[2],
                    tau_meas1_Nm    = tau_raw[0], tau_meas2_Nm    = tau_raw[1], tau_meas3_Nm    = tau_raw[2],
                    ati_fx_N        = ft[0],      ati_fy_N        = ft[1],      ati_fz_N        = ft[2],
                    ati_mx_Nm       = ft[3],      ati_my_Nm       = ft[4],      ati_mz_Nm       = ft[5],
                    f_push_N        = f_push,
                    G1_Nm           = G[0],       G2_Nm           = G[1],       G3_Nm           = G[2],
                    Cqdot1_Nm       = Cqdot[0],   Cqdot2_Nm       = Cqdot[1],   Cqdot3_Nm       = Cqdot[2],
                    q_des1_rad      = q_des[0],   q_des2_rad      = q_des[1],   q_des3_rad      = q_des[2],
                    qdot_des1_rad_s = qdot_des[0],qdot_des2_rad_s = qdot_des[1],qdot_des3_rad_s = qdot_des[2],
                    tau_cmd1_Nm     = tau_cmd[0], tau_cmd2_Nm     = tau_cmd[1], tau_cmd3_Nm     = tau_cmd[2],
                    phase           = PHASE,
                )

            # Console heartbeat
            if t - _last_print >= 0.5:
                _last_print = t
                _sig = f"f_des={f_des:+.1f}N" if PHASE == 'B' else f"amp={plotter.params.get('amp',0):.3f}"
                print(
                    f"t={t:6.1f}/{DURATION:.0f}s  {_sig}  "
                    f"f_push={f_push_disp:+.2f}N  "
                    f"F=[{ft[0]:+.1f},{ft[1]:+.1f},{ft[2]:+.1f}]N  "
                    f"q=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}]  "
                    f"T={[m.state.temp for m in motors]}"
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
    print("F/T reader started.")

    bus    = make_bus()
    motors = build_motors(bus)
    bus.drain()
    time.sleep(0.5)

    for m in motors:
        m.enable()
        m.update(timeout=2.0)

    # Lines
    _col = ['steelblue', 'firebrick', 'goldenrod']

    ln_fpush = Line(color='black',  label='f_push (−ATI_z)', lw=1.5)
    ln_fx    = Line(color=_col[0],  label='ATI Fx', lw=1.0)
    ln_fy    = Line(color=_col[1],  label='ATI Fy', lw=1.0)
    ln_fz    = Line(color=_col[2],  label='ATI Fz', lw=1.0)
    ln_qdes  = [Line(color=_col[i], ls='--', lw=1.5, label=f'q{i+1} des')      for i in range(3)]
    ln_qmeas = [Line(color=_col[i], lw=1.0, alpha=0.7, label=f'q{i+1} meas')   for i in range(3)]
    ln_tau   = [Line(color=_col[i], lw=1.0, label=f'τ_meas{i+1}')              for i in range(3)]
    ln_jit   = Line(color='mediumpurple', label='jitter [ms]', lw=1.0)

    # Plotter
    plotter = PlotThread(
        title=f'LSTM Data Collector — Phase {PHASE}',
        on_close=Terminate.set,
        on_stop=Terminate.set,
    )

    plotter.plot(1).set([ln_fpush])
    plotter.plot(1).ylabel = 'f_push [N]'
    plotter.plot(1).title  = '1-D label: push-axis force  (−ATI Fz)'

    plotter.plot(2).set([ln_fx, ln_fy, ln_fz])
    plotter.plot(2).ylabel = 'Force [N]'
    plotter.plot(2).title  = 'ATI 3-D force (sensor frame)'

    plotter.plot(3).set(ln_qdes + ln_qmeas)
    plotter.plot(3).ylabel = 'q [rad]'
    plotter.plot(3).title  = 'Joint positions'

    plotter.plot(4).set(ln_tau)
    plotter.plot(4).ylabel = 'τ [N·m]'
    plotter.plot(4).title  = 'Measured joint torque (from current)'

    plotter.plot(5).set([ln_jit])
    plotter.plot(5).ylabel = 'ms'
    plotter.plot(5).title  = 'Loop jitter'

    plotter.command = _PARAMS
    plotter.params.update({
        **_signal_defs,
        'freq_hz':       _freq_def,
        'tau_lpf_alpha': float(TAU_LPF_ALPHA),
        'kp1': float(KP_MOTOR[0]), 'kp2': float(KP_MOTOR[1]), 'kp3': float(KP_MOTOR[2]),
        'kd1': float(KD_MOTOR[0]), 'kd2': float(KD_MOTOR[1]), 'kd3': float(KD_MOTOR[2]),
        'vel_limit': float(VEL_LIMIT),
    })

    logger = DataLogger(
        prefix        = f'train_phase{PHASE}',
        extra_columns = EXTRA_COLUMNS,
        directory     = HERE,
    )

    complete = setup(motors, sensor)
    if not complete:
        Terminate.set()
        logger.close()
        for m in motors:
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        return

    ctrl_thread = threading.Thread(
        target=loop,
        args=(motors, plotter, logger,
              ln_fpush, ln_fx, ln_fy, ln_fz,
              ln_qdes, ln_qmeas, ln_tau, ln_jit),
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
        logger.close()
        for m in motors:
            try: m.coast(); m.disable()
            except Exception: pass
        try: sensor.stop_task()
        except Exception: pass
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    if DRY_RUN:
        all_cols = [
            'time_s', 'q1_rad', 'q2_rad', 'q3_rad',
            'qdot1_rad_s', 'qdot2_rad_s', 'qdot3_rad_s',
            'tau_meas1_Nm', 'tau_meas2_Nm', 'tau_meas3_Nm',
        ] + EXTRA_COLUMNS
        print(f"[DRY RUN]  PHASE={PHASE}  INIT_Q={INIT_Q}")
        print(f"  CSV columns ({len(all_cols)}): {all_cols}")
        print("  Syntax OK.")
        sys.exit(0)

    main()
