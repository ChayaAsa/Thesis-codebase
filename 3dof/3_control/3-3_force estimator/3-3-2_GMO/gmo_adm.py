from __future__ import annotations

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

GMO_ADM_EXTRA_COLUMNS = [
    'f_des_N', 'f_gmo_N', 'f_sensor_N',
    'r1_Nm', 'r2_Nm', 'r3_Nm',
    'tau_cmd1_Nm', 'tau_cmd2_Nm', 'tau_cmd3_Nm',
    'q_des1_rad', 'q_des2_rad', 'q_des3_rad',
    'qdot_des1_rad_s', 'qdot_des2_rad_s', 'qdot_des3_rad_s',
    'k_obs', 'b_adm', 'k_spring',
    'kp1', 'kp2', 'kp3', 'kd1', 'kd2', 'kd3',
    'use_ati',
]

# Hardware
BIAS_DURATION = 5.0

# Init pose
INIT_Q = np.array([0.0, 0.78, -0.78])   # [q1, q2, q3] rad

# ATI sensor (comparison / validation only)
FORCE_AXIS = 2        # 0=Fx 1=Fy 2=Fz
FORCE_SIGN = -1.0
LPF_ALPHA  = 0.3

# Push direction (Cartesian)
PUSH_DIR = np.array([1, 0, 0])
PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)

# ── Force setpoint (desired contact force along PUSH_DIR) ─────────────────────
F_DES_N = 0.0

# ── GMO observer gain [rad/s] ─────────────────────────────────────────────────
K_OBS = 30.0

# ── Admittance model  B_adm * xdot_des = f_ext - f_des - K_spring * x_err ───
B_ADM    = 20.0
K_SPRING = 2.0

# MIT impedance inner loop
KP_MOTOR = [DEFAULT_KP[id] for id in MOTOR_IDS]   # position stiffness per motor [N·m/rad]
KD_MOTOR = [DEFAULT_KD[id] for id in MOTOR_IDS]   # velocity damping per motor [N·m·s/rad]

# Safety
VEL_LIMIT    = 1.0    # rad/s per joint
TORQUE_LIMIT = 3.0    # display only

# Timing
LOOP_HZ = 100

# Joint safety limits
_JLO = np.array([JOINT_LIMITS[mid][0] for mid in MOTOR_IDS])
_JHI = np.array([JOINT_LIMITS[mid][1] for mid in MOTOR_IDS])

# ── Control-panel parameter definitions (key, label, lo, hi) ─────────────────
_PARAMS = [
    ('use_ati',   'Use ATI sensor',    True),
    ('f_des',     'F des  [N]',       0.0,  50.0),
    ('k_obs',     'K_obs [rad/s]',     1.0, 300.0),
    ('b_adm',     'B adm [N·s/m]',     0.1, 500.0),
    ('k_spring',  'K spring [N/m]',    0.0, 200.0),
    ('kp1',       'Kp motor 1',         0.0,  50.0),
    ('kp2',       'Kp motor 2',         0.0,  50.0),
    ('kp3',       'Kp motor 3',         0.0,  50.0),
    ('kd1',       'Kd motor 1',         0.0,  10.0),
    ('kd2',       'Kd motor 2',         0.0,  10.0),
    ('kd3',       'Kd motor 3',         0.0,  10.0),
    ('lpf_alpha', 'ATI LPF α',        0.01,  1.0),
    ('vel_limit', 'Vel  [r/s]',       0.05, 10.0),
]

# Shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6


# F/T reader thread (ATI, reference only)

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

        print("Step 3: Press Enter to START GMO admittance control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print(
            f"\nConfig: K_obs={K_OBS} rad/s  B_adm={B_ADM}  K_spring={K_SPRING}\n"
            f"  Kp_motor={KP_MOTOR}  Kd_motor={KD_MOTOR}  VelLim={VEL_LIMIT:.1f} rad/s\n"
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
         plotter: PlotThread, logger: DataLogger | None,
         ln_fdes: Line, ln_fgmo: Line, ln_fsen: Line,
         ln_r: list[Line], ln_qdes: list[Line], ln_qmeas: list[Line]) -> None:
    try:
        q_des    = INIT_Q.copy()
        qdot_des = np.zeros(3)

        # GMO state
        M0, C0, G_init = dyn.evaluate_MCG(INIT_Q, np.zeros(3))
        p0           = M0 @ np.zeros(3)
        gmo_integral = np.zeros(3)
        r_gmo        = np.zeros(3)
        # Warm-start: tau_cmd + beta = (C·qdot + G) + (C^T·qdot - G) → 0 at rest
        tau_cmd_prev = C0 @ np.zeros(3) + G_init

        f_sen_filt = FORCE_SIGN * ft_latest[FORCE_AXIS]

        dt_nom     = 1.0 / LOOP_HZ
        prev_t: float | None = None
        lp         = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        try:
            for t in lp:
                if Terminate.is_set():
                    lp.stop()
                    break

                # Live params
                f_des     = plotter.params['f_des']
                k_obs     = plotter.params['k_obs']
                b_adm     = plotter.params['b_adm']
                k_spring  = plotter.params['k_spring']
                kp_motors = [plotter.params['kp1'], plotter.params['kp2'], plotter.params['kp3']]
                kd_motors = [plotter.params['kd1'], plotter.params['kd2'], plotter.params['kd3']]
                ati_alpha = plotter.params['lpf_alpha']
                vel_limit = plotter.params['vel_limit']
                use_ati   = plotter.params.get('use_ati', False)

                if plotter.params.pop('reset', False):
                    gmo_integral[:] = 0.0
                    r_gmo[:]        = 0.0
                    q_des[:]        = INIT_Q.copy()
                    qdot_des[:]     = 0.0
                    M0r, C0r, G0r   = dyn.evaluate_MCG(INIT_Q, np.zeros(3))
                    p0              = M0r @ np.zeros(3)
                    tau_cmd_prev[:] = C0r @ np.zeros(3) + G0r
                    f_sen_filt      = FORCE_SIGN * ft_latest[FORCE_AXIS]

                dt     = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Joint state (DH frame)
                q    = np.array([INIT_Q[i] + SIGN[mid] * motors[i].state.position
                                  for i, mid in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[mid] * motors[i].state.velocity
                                  for i, mid in enumerate(MOTOR_IDS)])

                # Dynamics
                M, C, G = dyn.evaluate_MCG(q, qdot)

                # GMO update
                beta = C.T @ qdot - G
                p    = M @ qdot

                gmo_integral += dt * (tau_cmd_prev + beta + r_gmo)
                r_gmo = k_obs * (p - p0 - gmo_integral)

                # Cartesian force estimate
                # r_gmo converges to the JOINT REACTION torque (= -tau_ext), so the
                # externally-applied force is F_ext = J^{-T} * (-r_gmo).  Negating here
                # (not in the observer integral) keeps the observer stable while making
                # the force estimate match the ATI sign convention.
                tau_ext = -r_gmo
                Jv   = dyn.evaluate_jacobian(q)[:3, :]
                cond = np.linalg.cond(Jv)
                if cond < 1e6:
                    F_cart = np.linalg.solve(Jv.T, tau_ext)
                else:
                    F_cart = np.linalg.pinv(Jv.T) @ tau_ext
                f_gmo_scalar = float(PUSH_DIR @ F_cart)

                # ATI sensor (reference)
                f_raw      = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                f_sen_filt = ati_alpha * f_raw + (1.0 - ati_alpha) * f_sen_filt

                # ── Admittance: task-space desired velocity ───────────────
                # f_err = f_des - f_active: positive = contact below setpoint → advance
                f_active = f_sen_filt if use_ati else f_gmo_scalar
                f_err    = f_des - f_active
                xdot_des = PUSH_DIR * f_err / b_adm

                if k_spring > 1e-12:
                    p_ee   = dyn.evaluate_fk(q)['ee_position']
                    p_home = dyn.evaluate_fk(INIT_Q)['ee_position']
                    xdot_des -= (k_spring / b_adm) * (p_ee - p_home)

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

                # Motor command (MIT impedance mode)
                for i, mid in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp       = kp_motors[i]
                    motors[i].cmd.kd       = kd_motors[i]
                    motors[i].cmd.position = float(SIGN[mid] * (q_des[i] - INIT_Q[i]))
                    motors[i].cmd.velocity = float(SIGN[mid] * qdot_des[i]) * lp.fade
                    motors[i].cmd.torque   = 0.0
                    motors[i].update()

                # ── tau_cmd for GMO next step: impedance law in DH frame ──
                # Clamp to physical motor limit so joint-limit fighting can't corrupt the integral
                tau_cmd_prev = np.clip(
                    [kp_motors[i] * (q_des[i] - q[i]) + kd_motors[i] * (qdot_des[i] - qdot[i])
                     for i in range(3)],
                    -TORQUE_LIMIT, TORQUE_LIMIT,
                )

                tau_meas = np.array([SIGN[mid] * motors[i].state.torque
                                      for i, mid in enumerate(MOTOR_IDS)])

                # Plot
                plotter.ts = t
                ln_fdes.push(f_des)
                ln_fgmo.push(f_gmo_scalar)
                ln_fsen.push(f_sen_filt)
                for i in range(3):
                    ln_r[i].push(float(r_gmo[i]))
                    ln_qdes[i].push(q_des[i])
                    ln_qmeas[i].push(q[i])
                _src = 'ATI' if use_ati else 'GMO'
                plotter.info = [
                    f'[{_src}]  active: {f_active:+.3f} N',
                    f'GMO:  {f_gmo_scalar:+.3f} N',
                    f'ATI:  {f_sen_filt:+.3f} N',
                    f'Des:  {f_des:+.3f} N',
                    *[f'q{i+1}: meas {q[i]:+.3f}  des {q_des[i]:+.3f}' for i in range(3)],
                ]
                plotter.update()

                if logger is not None:
                    logger.log(
                        time_s=t,
                        q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                        qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                        tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1],
                        tau_meas3_Nm=tau_meas[2],
                        f_des_N=f_des, f_gmo_N=f_gmo_scalar, f_sensor_N=f_sen_filt,
                        r1_Nm=r_gmo[0], r2_Nm=r_gmo[1], r3_Nm=r_gmo[2],
                        tau_cmd1_Nm=tau_cmd_prev[0], tau_cmd2_Nm=tau_cmd_prev[1],
                        tau_cmd3_Nm=tau_cmd_prev[2],
                        q_des1_rad=q_des[0], q_des2_rad=q_des[1], q_des3_rad=q_des[2],
                        qdot_des1_rad_s=qdot_des[0], qdot_des2_rad_s=qdot_des[1],
                        qdot_des3_rad_s=qdot_des[2],
                        k_obs=k_obs, b_adm=b_adm, k_spring=k_spring,
                        kp1=kp_motors[0], kp2=kp_motors[1], kp3=kp_motors[2],
                        kd1=kd_motors[0], kd2=kd_motors[1], kd3=kd_motors[2],
                        use_ati=int(use_ati),
                    )

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  "
                        f"GMO={f_gmo_scalar:+.3f}N  ATI={f_sen_filt:+.3f}N  des={f_des:+.2f}N  "
                        f"r=[{r_gmo[0]:+.2f},{r_gmo[1]:+.2f},{r_gmo[2]:+.2f}]N·m  "
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

    # Line objects
    colors  = ['steelblue', 'firebrick', 'goldenrod']
    ln_fdes = Line(color='blue',    label='F des',      lw=1.5, ls='--')
    ln_fgmo = Line(color='red',     label='GMO est.',   lw=1.5)
    ln_fsen = Line(color='# aaaaaa', label='ATI ref.', lw=1.0, alpha=0.7)
    ln_r    = [Line(color=colors[i], lw=1.5, label=f'r{i+1}')              for i in range(3)]
    ln_qdes = [Line(color=colors[i], lw=1.5, ls='--', label=f'q{i+1} des') for i in range(3)]
    ln_qmeas= [Line(color=colors[i], lw=1.0, alpha=0.65, label=f'q{i+1} meas') for i in range(3)]

    def _on_reset() -> None:
        plotter.params['reset'] = True

    plotter = PlotThread(
        title='3-DOF GMO Admittance Control',
        on_close=Terminate.set,
        on_reset=_on_reset,
        on_stop=Terminate.set,
    )

    plotter.plot(1).set([ln_fdes, ln_fgmo, ln_fsen])
    plotter.plot(1).ylabel = 'Force [N]'
    plotter.plot(1).title  = 'Force'

    plotter.plot(2).set(ln_r)
    plotter.plot(2).ylabel = 'tau_ext [N·m]'
    plotter.plot(2).title  = 'GMO joint torques'

    plotter.plot(3).set(ln_qdes + ln_qmeas)
    plotter.plot(3).ylabel = 'q [rad]'
    plotter.plot(3).title  = 'Joint positions'

    plotter.command = _PARAMS
    plotter.params.update({
        'use_ati':   True,
        'f_des':     float(F_DES_N),
        'k_obs':     float(K_OBS),
        'b_adm':     float(B_ADM),
        'k_spring':  float(K_SPRING),
        'kp1':       float(KP_MOTOR[0]),
        'kp2':       float(KP_MOTOR[1]),
        'kp3':       float(KP_MOTOR[2]),
        'kd1':       float(KD_MOTOR[0]),
        'kd2':       float(KD_MOTOR[1]),
        'kd3':       float(KD_MOTOR[2]),
        'lpf_alpha': float(LPF_ALPHA),
        'vel_limit': float(VEL_LIMIT),
    })

    logger = DataLogger('gmo_adm', GMO_ADM_EXTRA_COLUMNS, directory=HERE)

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
        print("Startup failed.")
        return

    ctrl_thread = threading.Thread(
        target=loop,
        args=(motors, plotter, logger,
              ln_fdes, ln_fgmo, ln_fsen, ln_r, ln_qdes, ln_qmeas),
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
    main()
