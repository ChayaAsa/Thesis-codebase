from __future__ import annotations

import collections
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn

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
from tmotorcan import MITMotor, RealtimeLoop
from tmotorcan.protocol import MotorFaultError, MotorTimeoutError
from plot_thread import PlotThread, Line

dyn = Dynamic.get_or_build(DYN_CACHE)

# Model
MODEL_TAG  = "phys_1d"
MODEL_DIR  = os.path.normpath(os.path.join(HERE, '..', '3_model'))
MODEL_PATH = os.path.join(MODEL_DIR, f'lstm_{MODEL_TAG}.pt')

# Hardware
COM_PORT      = PORT
BIAS_DURATION = 5
INIT_Q        = np.array([0.0, 0.78, -0.78])

# Push axis
PUSH_DIR    = np.array([1.0, 0.0, 0.0])
FORCE_AXIS  = 2      # ATI Fz (only used when USE_ATI=True)
FORCE_SIGN  = -1.0

# Force source
# Toggled live via the "Use LSTM" checkbox in the GUI panel.
# ATI always starts (needed for both modes and ground-truth logging).
USE_ATI = True

# ── Admittance (second-order: M*xddot + B*xdot + K*x = F_err) ───────────────
# M adds virtual inertia → smooth velocity ramp-up, no torque spikes
# B is damping → steady-state compliance (higher = softer / slower)
# K is spring → restoring force toward INIT_Q when unloaded
F_DES_N   = 0.0
M_ADM     = 2.0    # virtual mass [kg]. Higher = slower, smoother response
B_ADM     = 20.0
K_SPRING  = 2.0

# Contact detection
# If arm moves this far from INIT_Q (q2, q3), it cannot be touching the wall.
# Force f_lstm = 0 so the controller drives arm back toward wall instead of
# running away backward from LSTM extrapolation errors.
CONTACT_RANGE_RAD = 0.30

# MIT impedance gains
KP_MOTOR  = [DEFAULT_KP[id] for id in MOTOR_IDS]
KD_MOTOR  = [DEFAULT_KD[id] for id in MOTOR_IDS]

# Signal processing
LPF_ALPHA      = 0.1   # IIR on force signal (1.0 = raw, lower = smoother)
QDOT_LPF_ALPHA = 0.1   # IIR on qdot before physics residual computation
DEAD_BAND_N    = 0.5   # ±N dead-band on f_err

# Safety
VEL_LIMIT  = 1.0
LOOP_HZ    = 100

_JLO = np.array([JOINT_LIMITS[id][0] for id in MOTOR_IDS])
_JHI = np.array([JOINT_LIMITS[id][1] for id in MOTOR_IDS])

# Control-panel sliders
_PARAMS = [
    ('f_des',    'F des  [N]',  0.0,  50.0),
    ('m_adm',    'M adm [kg]',  0.01,  5.0),
    ('b_adm',    'B adm',       0.1, 500.0),
    ('k_spring', 'K spring',    0.0, 200.0),
    ('kp1',      'Kp motor 1',  0.0,  50.0),
    ('kp2',      'Kp motor 2',  0.0,  50.0),
    ('kp3',      'Kp motor 3',  0.0,  50.0),
    ('kd1',      'Kd motor 1',  0.0,  10.0),
    ('kd2',      'Kd motor 2',  0.0,  10.0),
    ('kd3',      'Kd motor 3',  0.0,  10.0),
    ('lpf_alpha','LPF  alpha',  0.01,  1.0),
    ('qdot_alpha','qdot LPF α', 0.01,  1.0),
    ('f_dead',   'F dead [N]',  0.0,  5.0),
    ('vel_limit','Vel  [r/s]',  0.05, 10.0),
]

# Shared state
Terminate  = threading.Event()
ft_latest  = [0.0] * 6   # only populated when USE_ATI=True


class ForceLSTM(nn.Module):
    def __init__(self, n_in, n_out, hidden=64, n_layers=2, fc_hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _load_lstm(path: str):
    device = torch.device('cpu')
    bundle = torch.load(path, map_location=device, weights_only=False)
    model  = ForceLSTM(
        n_in      = bundle['n_in'],
        n_out     = bundle['n_out'],
        hidden    = bundle['hidden'],
        n_layers  = bundle['n_layers'],
        fc_hidden = bundle['fc_hidden'],
        dropout   = bundle['dropout'],
    )
    model.load_state_dict(bundle['model_state'])
    model.eval()

    norm = bundle['norm']
    mean_X = np.array(norm['mean_X'], dtype=np.float32)
    std_X  = np.array(norm['std_X'],  dtype=np.float32)
    mean_y = float(np.array(norm['mean_y']).ravel()[0])
    std_y  = float(np.array(norm['std_y']).ravel()[0])

    window_len  = int(bundle['window_len'])
    n_in        = int(bundle['n_in'])
    input_cols  = bundle['input_cols']
    return model, mean_X, std_X, mean_y, std_y, window_len, n_in, input_cols


# Sliding window buffer

class WindowBuffer:

    def __init__(self, window_len: int, n_features: int):
        self._wl  = window_len
        self._buf = collections.deque(maxlen=window_len)
        self._n   = n_features

    def push(self, feat: np.ndarray) -> None:
        self._buf.append(feat.astype(np.float32))

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._wl

    def to_tensor(self) -> torch.Tensor:
        arr = np.stack(self._buf, axis=0)   # (window_len, n_features)
        return torch.from_numpy(arr).unsqueeze(0)   # (1, window_len, n_features)


# ATI reader (optional)

def ft_reader_thread(sensor) -> None:
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


# Startup

_TIMEOUT_ABORT = 5


def setup(motors: list[MITMotor], sensor=None) -> bool:
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
                print(f"  [ERROR] motor {m.id} zero command never ACK'd")
        print(f"  Motors zeroed.")

        print("Step 2: Arm holding init pose. Press Enter to bias ATI sensor (or skip if USE_ATI=False).")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        if sensor is not None:
            print(f"  Biasing ATI for {BIAS_DURATION:.1f}s ...")
            sensor.reBias(duration=BIAS_DURATION)
            print("  Bias complete.")
        else:
            print("  (ATI disabled — skipped bias)")

        print("Step 3: Press Enter to START LSTM admittance control.")
        kb = KeyboardLine()
        while kb.poll() is None and not Terminate.is_set():
            cmd_lock(motors)
            time.sleep(dt_boot)
        print("  LSTM admittance control starting!\n")
        return True

    except MotorFaultError as e:
        print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        return False
    except MotorTimeoutError as e:
        print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")
        return False


# Control loop

def loop(motors: list[MITMotor],
         plotter: PlotThread, logger: DataLogger | None,
         model, mean_X, std_X, mean_y, std_y, window_len: int,
         input_cols: list[str],
         ln_fdes: Line, ln_fctrl: Line, ln_fati: Line,
         ln_qdes: list[Line], ln_qmeas: list[Line],
         ln_tres: list[Line]) -> None:
    try:
        q_des   = INIT_Q.copy()
        f_filt  = 0.0
        dt_nom  = 1.0 / LOOP_HZ
        prev_t: float | None = None

        buf       = WindowBuffer(window_len, mean_X.shape[0])
        xdot_int  = np.zeros(3)   # integrated task-space velocity for 2nd-order admittance
        qdot_filt = np.zeros(3)
        rt_loop   = RealtimeLoop(dt=dt_nom, report=True, fade=0.5)
        _last_print = 0.0

        try:
            for t in rt_loop:
                if Terminate.is_set():
                    rt_loop.stop()
                    break

                if plotter.params.pop('reset', False):
                    q_des        = INIT_Q.copy()
                    xdot_int[:]  = 0.0
                    qdot_filt[:] = 0.0
                    f_filt       = 0.0

                f_des      = plotter.params['f_des']
                m_adm      = max(1e-3, float(plotter.params['m_adm']))
                b_adm      = plotter.params['b_adm']
                k_spring   = plotter.params['k_spring']
                kp_motors  = [plotter.params['kp1'], plotter.params['kp2'], plotter.params['kp3']]
                kd_motors  = [plotter.params['kd1'], plotter.params['kd2'], plotter.params['kd3']]
                alpha      = plotter.params['lpf_alpha']
                alpha_qdot = float(plotter.params.get('qdot_alpha', 0.1))
                f_dead     = float(plotter.params.get('f_dead', 0.0))
                vel_limit  = plotter.params['vel_limit']

                dt     = (t - prev_t) if prev_t is not None else dt_nom
                prev_t = t

                # Joint state
                q    = np.array([INIT_Q[i] + SIGN[id] * motors[i].state.position
                                  for i, id in enumerate(MOTOR_IDS)])
                qdot = np.array([SIGN[id] * motors[i].state.velocity
                                  for i, id in enumerate(MOTOR_IDS)])
                tau_meas = np.array([SIGN[id] * motors[i].state.torque
                                      for i, id in enumerate(MOTOR_IDS)])

                # LP-filter qdot to reduce Cqdot noise in tau_res
                qdot_filt = alpha_qdot * qdot + (1.0 - alpha_qdot) * qdot_filt

                # Physics residual
                _, C, G = dyn.evaluate_MCG(q, qdot_filt)
                Cqdot   = C @ qdot_filt
                tau_res = tau_meas - G - Cqdot   # same formula as training data

                # ── Build feature vector from model's input_cols ──────────────
                _col_map = {
                    'tau_meas1_Nm': tau_meas[0], 'tau_meas2_Nm': tau_meas[1], 'tau_meas3_Nm': tau_meas[2],
                    'q1_rad': q[0],   'q2_rad': q[1],   'q3_rad': q[2],
                    'qdot1_rad_s': qdot[0], 'qdot2_rad_s': qdot[1], 'qdot3_rad_s': qdot[2],
                    'G1_Nm': G[0],    'G2_Nm': G[1],    'G3_Nm': G[2],
                    'Cqdot1_Nm': Cqdot[0], 'Cqdot2_Nm': Cqdot[1], 'Cqdot3_Nm': Cqdot[2],
                    'tau_res1_Nm': tau_res[0], 'tau_res2_Nm': tau_res[1], 'tau_res3_Nm': tau_res[2],
                }
                feat = np.array([_col_map[c] for c in input_cols], dtype=np.float32)

                # Normalise and push to window
                feat_norm = (feat - mean_X) / std_X
                buf.push(feat_norm)

                # Force estimation (live-toggled via GUI checkbox)
                use_lstm = bool(plotter.params.get('use_lstm', False))
                if not use_lstm:
                    # ATI mode: sensor drives control directly
                    f_raw  = FORCE_SIGN * float(ft_latest[FORCE_AXIS])
                    warmup = False
                else:
                    # LSTM mode
                    warmup = not buf.ready
                    if not warmup:
                        with torch.no_grad():
                            y_norm = model(buf.to_tensor())      # (1, 1)
                        f_raw = max(0.0, float(y_norm.item()) * std_y + mean_y)
                        # Contact detection: arm too far from INIT_Q → LSTM extrapolates
                        # wrongly in free space; clamp to 0 so arm drives back to wall
                        if np.max(np.abs(q[1:] - INIT_Q[1:])) > CONTACT_RANGE_RAD:
                            f_raw = 0.0
                    else:
                        # Warmup: seed LPF from ATI so filter starts at real force level
                        f_raw = FORCE_SIGN * float(ft_latest[FORCE_AXIS])

                # Low-pass filter
                f_filt = alpha * f_raw + (1.0 - alpha) * f_filt
                f_err  = f_des - f_filt
                # Dead-band: suppress noise-driven motion near desired force
                f_err  = np.sign(f_err) * max(0.0, abs(f_err) - f_dead)

                # ── Second-order admittance: M*xdot_dot + B*xdot + K*x = F_err ─
                # Integrate xdot_int (task-space velocity state).
                # M adds virtual inertia → smooth torque, no spikes from force noise.
                x_err = np.zeros(3)
                if k_spring > 1e-12:
                    p_ee   = dyn.evaluate_fk(q)['ee_position']
                    p_home = dyn.evaluate_fk(INIT_Q)['ee_position']
                    x_err  = p_ee - p_home
                xdot_int += (dt / m_adm) * (
                    f_err * PUSH_DIR
                    - b_adm   * xdot_int
                    - k_spring * x_err
                )
                xdot_des = xdot_int.copy()

                # ── Joint velocity (q2, q3 only) ──────────────────────────────
                Jv               = dyn.evaluate_jacobian(q)[:3, :]
                Jv_23            = Jv[:, 1:]
                qdot_23, _, _, _ = np.linalg.lstsq(Jv_23, xdot_des, rcond=1e-3)
                qdot_des         = np.zeros(3)
                qdot_des[1:]     = qdot_23
                qdot_des         = np.clip(qdot_des, -vel_limit, vel_limit)

                # Freeze arm during LSTM warmup — admittance runs but no motion
                if warmup:
                    qdot_des[:] = 0.0

                # Integrate q_des
                q_des    += qdot_des * dt * rt_loop.fade
                q_des[0]  = INIT_Q[0]
                q_des     = np.clip(q_des, INIT_Q + _JLO, INIT_Q + _JHI)

                # Motor command
                for i, id in enumerate(MOTOR_IDS):
                    motors[i].cmd.kp       = kp_motors[i]
                    motors[i].cmd.kd       = kd_motors[i]
                    motors[i].cmd.position = float(SIGN[id] * (q_des[i] - INIT_Q[i]))
                    motors[i].cmd.velocity = float(SIGN[id] * qdot_des[i]) * rt_loop.fade
                    motors[i].cmd.torque   = 0.0
                    motors[i].update()

                # ── ATI ground truth (always logged; also drives control in ATI mode) ─
                f_ati = FORCE_SIGN * float(ft_latest[FORCE_AXIS])

                # Plot
                plotter.ts = t
                ln_fdes.push(f_des)
                ln_fctrl.push(f_filt)
                ln_fati.push(f_ati)
                for i in range(3):
                    ln_qdes[i].push(q_des[i])
                    ln_qmeas[i].push(q[i])
                    ln_tres[i].push(tau_res[i])
                src = 'ATI' if not use_lstm else ('LSTM' if not warmup else f'warmup {len(buf._buf)}/{window_len}')
                plotter.info = [
                    f'f_ctrl [{src}]: {f_filt:+.3f} N  f_ati: {f_ati:+.3f} N',
                    f'f_des:   {f_des:+.3f} N',
                    *[f'q{i+1}: meas {q[i]:+.3f}  des {q_des[i]:+.3f}' for i in range(3)],
                ]
                plotter.update()

                if logger is not None:
                    log_kwargs = dict(
                        time_s=t,
                        q1_rad=q[0], q2_rad=q[1], q3_rad=q[2],
                        qdot1_rad_s=qdot[0], qdot2_rad_s=qdot[1], qdot3_rad_s=qdot[2],
                        tau_meas1_Nm=tau_meas[0], tau_meas2_Nm=tau_meas[1], tau_meas3_Nm=tau_meas[2],
                        G1_Nm=G[0], G2_Nm=G[1], G3_Nm=G[2],
                        Cqdot1_Nm=Cqdot[0], Cqdot2_Nm=Cqdot[1], Cqdot3_Nm=Cqdot[2],
                        tau_res1_Nm=tau_res[0], tau_res2_Nm=tau_res[1], tau_res3_Nm=tau_res[2],
                        f_ctrl_N=f_filt,
                        f_des_N=f_des, f_err_N=f_err,
                        xdot_des_x=xdot_des[0], xdot_des_y=xdot_des[1], xdot_des_z=xdot_des[2],
                        qdot_des1_rad_s=qdot_des[0], qdot_des2_rad_s=qdot_des[1], qdot_des3_rad_s=qdot_des[2],
                        q_des1_rad=q_des[0], q_des2_rad=q_des[1], q_des3_rad=q_des[2],
                        kp1=kp_motors[0], kp2=kp_motors[1], kp3=kp_motors[2],
                        kd1=kd_motors[0], kd2=kd_motors[1], kd3=kd_motors[2],
                        b_adm=b_adm, k_spring=k_spring, lpf_alpha=alpha, vel_limit=vel_limit,
                        f_ati_N=f_ati,
                    )
                    logger.log(**log_kwargs)

                if t - _last_print >= 0.2:
                    _last_print = t
                    temps = [m.state.temp for m in motors]
                    print(
                        f"t={t:7.3f}s  [{src}]  "
                        f"f_ctrl={f_filt:+.3f}N(des={f_des:+.2f})  "
                        f"tau_res=[{tau_res[0]:+.3f},{tau_res[1]:+.3f},{tau_res[2]:+.3f}]Nm  "
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
    print(f"Loading LSTM model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model, mean_X, std_X, mean_y, std_y, window_len, n_in, input_cols = _load_lstm(MODEL_PATH)
    print(f"  Model: {MODEL_TAG}  n_in={n_in}  window_len={window_len}  "
          f"mean_y={mean_y:.2f}N  std_y={std_y:.2f}N")
    print(f"  Warmup: {window_len/LOOP_HZ:.1f}s ({window_len} steps at {LOOP_HZ}Hz)")
    print("  Toggle ATI ↔ LSTM live via the 'Use LSTM' checkbox in the GUI.")

    from ATI_FTsensor.ftsensor import ftsensor
    print("Initialising ATI F/T sensor ...")
    sensor = ftsensor()
    sensor.start_task()
    time.sleep(0.5)
    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("  ATI reader started.")

    bus    = make_bus()
    motors = build_motors(bus)

    bus.drain()
    time.sleep(0.5)
    for m in motors:
        m.enable()
        m.update(timeout=2.0)

    # Lines
    colors = ['steelblue', 'firebrick', 'goldenrod']
    ln_fdes  = Line(color='blue',   label='F des',  lw=1.5, ls='--')
    ln_fctrl = Line(color='green',  label='F ctrl', lw=1.5)
    ln_fati  = Line(color='orange', label='F ati',  lw=1.0, alpha=0.7)
    ln_qdes  = [Line(color=colors[i], ls='--', lw=1.5, label=f'q{i+1} des')    for i in range(3)]
    ln_qmeas = [Line(color=colors[i], lw=1.0,  alpha=0.7, label=f'q{i+1} meas') for i in range(3)]
    ln_tres  = [Line(color=colors[i], lw=1.2,  label=f'τ_res{i+1}')             for i in range(3)]

    def _on_reset() -> None:
        plotter.params['reset'] = True

    plotter = PlotThread(
        title='3-DOF Force-Sensorless Admittance Control',
        on_close=Terminate.set,
        on_reset=_on_reset,
        on_stop=Terminate.set,
    )

    plotter.plot(1).set([ln_fctrl, ln_fati, ln_fdes])
    plotter.plot(1).ylabel = 'Force [N]'
    plotter.plot(1).title  = 'Force  (green=ctrl src, orange=ATI, blue--=des)'

    plotter.plot(2).set(ln_qdes + ln_qmeas)
    plotter.plot(2).ylabel = 'q [rad]'
    plotter.plot(2).title  = 'Joint positions'

    plotter.plot(3).set(ln_tres)
    plotter.plot(3).ylabel = 'τ_res [N·m]'
    plotter.plot(3).title  = 'Torque residuals (LSTM input)'

    plotter.command = _PARAMS + [('use_lstm', 'Use LSTM', False)]
    plotter.params.update({
        'f_des':     float(F_DES_N),
        'm_adm':     float(M_ADM),
        'b_adm':     float(B_ADM),
        'k_spring':  float(K_SPRING),
        'kp1':       float(KP_MOTOR[0]),
        'kp2':       float(KP_MOTOR[1]),
        'kp3':       float(KP_MOTOR[2]),
        'kd1':       float(KD_MOTOR[0]),
        'kd2':       float(KD_MOTOR[1]),
        'kd3':       float(KD_MOTOR[2]),
        'lpf_alpha':  float(LPF_ALPHA),
        'qdot_alpha': float(QDOT_LPF_ALPHA),
        'f_dead':     float(DEAD_BAND_N),
        'vel_limit':  float(VEL_LIMIT),
        'use_lstm':   False,   # start in ATI mode; tick to switch to LSTM
    })

    _EXTRA_COLS = [
        'G1_Nm', 'G2_Nm', 'G3_Nm',
        'Cqdot1_Nm', 'Cqdot2_Nm', 'Cqdot3_Nm',
        'tau_res1_Nm', 'tau_res2_Nm', 'tau_res3_Nm',
        'f_ctrl_N', 'f_des_N', 'f_err_N',
        'xdot_des_x', 'xdot_des_y', 'xdot_des_z',
        'qdot_des1_rad_s', 'qdot_des2_rad_s', 'qdot_des3_rad_s',
        'q_des1_rad', 'q_des2_rad', 'q_des3_rad',
        'kp1', 'kp2', 'kp3', 'kd1', 'kd2', 'kd3',
        'b_adm', 'k_spring', 'lpf_alpha', 'vel_limit',
        'f_ati_N',
    ]
    logger = DataLogger('lstm_adm', _EXTRA_COLS, directory=HERE)

    complete = setup(motors, sensor)
    if not complete:
        Terminate.set()
        for m in motors:
            try:
                m.coast(); m.disable()
            except Exception:
                pass
        if sensor is not None:
            try:
                sensor.stop_task()
            except Exception:
                pass
        bus.close()
        print("\nShutdown complete.")
        return

    ctrl_thread = threading.Thread(
        target=loop,
        args=(motors, plotter, logger,
              model, mean_X, std_X, mean_y, std_y, window_len, input_cols,
              ln_fdes, ln_fctrl, ln_fati, ln_qdes, ln_qmeas, ln_tres),
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
            try:
                m.coast(); m.disable()
            except Exception:
                pass
        if sensor is not None:
            try:
                sensor.stop_task()
            except Exception:
                pass
        bus.close()
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
