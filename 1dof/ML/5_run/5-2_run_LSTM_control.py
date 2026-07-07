from __future__ import annotations

import collections
import csv
import os
import queue
import threading
import time

import can
import numpy as np
import torch
import torch.nn as nn

from tmotorcan import MotorBus, MITMotor
from tmotorcan.protocol import MotorTimeoutError, MotorFaultError
from ATI_FTsensor.ftsensor import ftsensor


# hardware config
COM_PORT    = 'COM18'
MOTOR_ID    = 1
MOTOR_MODEL = 'AK45-10'

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE,'..', '3_model', '3-2_LSTM_model.pt')

# control defaults
F_DESIRED_N   = 0.0
KP_F          = 0.0
KI_F          = 0.0
LPF_ALPHA     = 1.0     # IIR alpha for both ATI and NN: 1.0=raw, 0.1=heavy
JACOBIAN      = 0.2     # effective lever arm [m]
TORQUE_LIMIT  = 3.0     # hard torque cap [N·m]
BIAS_DURATION = 5.0     # F/T bias averaging time [s]
LOOP_HZ       = 100     # control frequency [Hz]
# SENSOR_HZ     = 100     # F/T sensor read frequency [Hz]  (must be ≥ control Hz)

# shared state
Terminate = threading.Event()
ft_latest = [0.0] * 6   # [Fx, Fy, Fz, Tx, Ty, Tz]

ctrl_params = {
    'f_des':     float(F_DESIRED_N),
    'kp':        float(KP_F),
    'ki':        float(KI_F),
    'lpf_alpha': float(LPF_ALPHA),
    'use_nn':    False,   # False=ATI, True=NN
    'reset':     False,
}


# NN virtual force sensor
class TauNet(nn.Module):
    def __init__(self, n_in: int, hidden: list[int]):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model(path: str, device: torch.device):
    if not os.path.exists(path):
        print(f"[NN] model not found at {path}  — NN toggle will be disabled")
        return None, None
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = TauNet(n_in=ckpt['n_in'], hidden=ckpt['hidden']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    norm = {k: v.astype(np.float32) for k, v in ckpt['norm'].items()}
    print(f"[NN] loaded   : {path}")
    print(f"[NN] inputs   : {ckpt['input_cols']}")
    print(f"[NN] output   : {ckpt['target_col']}\n")
    return model, norm


def nn_infer(model, norm, tau_meas: float, pos: float, vel: float,
             device: torch.device) -> float:
    x  = np.array([[tau_meas, pos, vel]], dtype=np.float32)
    xn = (x - norm['mean_X']) / norm['std_X']
    with torch.no_grad():
        yn = model(torch.from_numpy(xn).to(device))
    return float(yn.cpu().numpy().flat[0] * norm['std_y'].flat[0] + norm['mean_y'].flat[0])


# LSTM timeseries predictor (TSLM)
class TauNetLSTM(nn.Module):
    def __init__(self, n_in, lstm_hidden, lstm_layers, fc_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.Tanh(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TSLMPredictor:
    def __init__(self, ckpt: dict, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        # normalization and metadata from checkpoint
        self.norm = {k: np.float32(v) if np.ndim(v) == 0 else np.array(v, dtype=np.float32)
                     for k, v in ckpt['norm'].items()}
        self.window_size = int(ckpt['window_size'])
        self.n_in = int(ckpt['n_in'])
        # buffer holds most recent timesteps (window_size, n_in)
        self._buf = np.zeros((self.window_size, self.n_in), dtype=np.float32)

    def reset(self, val: float = 0.0) -> None:
        self._buf.fill(val)

    def predict(self, tau_meas: float, pos: float, vel: float) -> float:
        # append new sample into buffer (shift left)
        if self.n_in >= 3:
            sample = np.array([tau_meas, pos, vel], dtype=np.float32)
        else:
            sample = np.array([tau_meas], dtype=np.float32)
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = sample

        xb_n = (self._buf - self.norm['mean_X']) / self.norm['std_X']
        with torch.no_grad():
            xb = torch.from_numpy(xb_n[None, ...]).to(self.device)
            y = self.model(xb).squeeze(-1).cpu().numpy()
        return float(y.flat[0] * float(self.norm['std_y']) + float(self.norm['mean_y']))


def load_ts_model(path: str, device: torch.device):
    if not os.path.exists(path):
        print(f"[TSLM] model not found at {path}  — TSLM will be disabled")
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = TauNetLSTM(
        n_in        = ckpt['n_in'],
        lstm_hidden = ckpt['lstm_hidden'],
        lstm_layers = ckpt['lstm_layers'],
        fc_hidden   = ckpt['fc_hidden'],
        dropout     = ckpt['dropout'],
    )
    model.load_state_dict(ckpt['model_state'])
    pred = TSLMPredictor(ckpt, model, device)
    print(f"[TSLM] loaded   : {path}")
    print(f"[TSLM] inputs   : {ckpt.get('input_cols', '<unknown>')}")
    print(f"[TSLM] window   : {pred.window_size} steps ({ckpt.get('window_s', 0.0)*1000:.0f} ms)\n")
    return pred


# F/T sensor reader
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
    WINDOW_S  = 10.0
    UPDATE_MS = 100     # 10 Hz visual refresh
    MAX_PTS   = 1500    # ~15 s @ 100 Hz

    def __init__(self, nn_available: bool):
        self._lock        = threading.Lock()
        self._nn_avail    = nn_available
        self._t           = collections.deque(maxlen=self.MAX_PTS)
        # subplot 1
        self._f_des       = collections.deque(maxlen=self.MAX_PTS)
        self._f_ati       = collections.deque(maxlen=self.MAX_PTS)
        self._f_nn        = collections.deque(maxlen=self.MAX_PTS)
        # subplot 2 — pre-selected source (raw + filtered)
        self._f_sel_raw   = collections.deque(maxlen=self.MAX_PTS)
        self._f_sel_filt  = collections.deque(maxlen=self.MAX_PTS)
        # subplot 3
        self._tau         = collections.deque(maxlen=self.MAX_PTS)
        self._tau_meas    = collections.deque(maxlen=self.MAX_PTS)

    def push(self, t: float,
             f_des: float, f_ati: float, f_nn: float,
             f_sel_raw: float, f_sel_filt: float,
             tau_cmd: float, tau_meas: float) -> None:
        with self._lock:
            self._t.append(t)
            self._f_des.append(f_des)
            self._f_ati.append(f_ati)
            self._f_nn.append(f_nn)
            self._f_sel_raw.append(f_sel_raw)
            self._f_sel_filt.append(f_sel_filt)
            self._tau.append(tau_cmd)
            self._tau_meas.append(tau_meas)

    def _run(self) -> None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox, Button

        fig, (ax_f, ax_ctrl, ax_t) = plt.subplots(
            3, 1, figsize=(10, 9),
            gridspec_kw={'height_ratios': [2, 2, 1.5]},
        )
        fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.13)
        fig.suptitle('1-DOF Force Control  (PI + virtual force sensor)', fontsize=11)

        # ── subplot 1: force comparison ───────────────────────────────────────
        ax_f.set_ylabel('Force [N]')
        ax_f.set_xlabel('Time [s]')
        ax_f.grid(True, alpha=0.3)
        ax_f.set_title('Force comparison', fontsize=9)
        line_fdes, = ax_f.plot([], [], 'b--', lw=1.5, label='F des')
        line_fati, = ax_f.plot([], [], 'r-',  lw=1.5, label='F ATI (filt)')
        line_fnn,  = ax_f.plot([], [], color='darkorange', lw=1.2, label='F NN (filt)')
        ax_f.legend(loc='upper left', fontsize=8)

        # ── subplot 2: control tuner ──────────────────────────────────────────
        ax_ctrl.set_ylabel('Force [N]')
        ax_ctrl.set_xlabel('Time [s]')
        ax_ctrl.grid(True, alpha=0.3)
        line_ctrl_des,  = ax_ctrl.plot([], [], 'b--', lw=1.5, label='F des')
        line_ctrl_raw,  = ax_ctrl.plot([], [], lw=0.8, alpha=0.5, label='raw')
        line_ctrl_filt, = ax_ctrl.plot([], [], lw=1.8, label='filtered')
        ax_ctrl.legend(loc='upper left', fontsize=8)

        # ── subplot 3: torque ─────────────────────────────────────────────────
        ax_t.set_ylabel('Torque [N·m]')
        ax_t.set_xlabel('Time [s]')
        ax_t.grid(True, alpha=0.3)
        ax_t.set_title('Torque', fontsize=9)
        line_tau,      = ax_t.plot([], [], 'g-', lw=2.0, label='tau_cmd')
        line_tau_meas, = ax_t.plot([], [], 'm-', lw=1.0, label='tau_meas')
        ax_t.legend(loc='upper left', fontsize=8)

        # ── input panel — single row ──────────────────────────────────────────
        ax_fdes = fig.add_axes([0.05, 0.02, 0.12, 0.07])
        ax_kp   = fig.add_axes([0.21, 0.02, 0.08, 0.07])
        ax_ki   = fig.add_axes([0.33, 0.02, 0.08, 0.07])
        ax_lpf  = fig.add_axes([0.45, 0.02, 0.08, 0.07])
        ax_tog  = fig.add_axes([0.57, 0.02, 0.18, 0.07])
        ax_rst  = fig.add_axes([0.78, 0.02, 0.11, 0.07])

        tb_fdes = TextBox(ax_fdes, 'F_des [N]', initial=str(ctrl_params['f_des']),
                          color='lightyellow', hovercolor='yellow')
        tb_kp   = TextBox(ax_kp,   'Kp       ', initial=str(ctrl_params['kp']),
                          color='lightyellow', hovercolor='yellow')
        tb_ki   = TextBox(ax_ki,   'Ki       ', initial=str(ctrl_params['ki']),
                          color='lightyellow', hovercolor='yellow')
        tb_lpf  = TextBox(ax_lpf,  'LPF α    ', initial=str(ctrl_params['lpf_alpha']),
                          color='lightcyan',   hovercolor='cyan')

        _tog_lbl = 'Force src: ATI' if not ctrl_params['use_nn'] else 'Force src: NN'
        _tog_col = '# c8f5c8' if not ctrl_params['use_nn'] else '#ffd8a0'
        btn_tog  = Button(ax_tog, _tog_lbl, color=_tog_col,    hovercolor='# e8e8e8')
        btn_rst  = Button(ax_rst, 'Reset  ∫', color='# f5c8c8', hovercolor='#ffaaaa')

        if not self._nn_avail:
            btn_tog.label.set_text('Force src: ATI  (NN N/A)')
            btn_tog.ax.set_facecolor('# d0d0d0')

        # callbacks
        def _on_fdes(val):
            try:
                ctrl_params['f_des'] = float(val)
                ctrl_params['reset'] = True
                print(f"[panel] F_des → {ctrl_params['f_des']:.4f} N  (integral reset)")
            except ValueError:
                pass

        def _on_kp(val):
            try:
                ctrl_params['kp'] = float(val)
                print(f"[panel] Kp → {ctrl_params['kp']:.6f}")
            except ValueError:
                pass

        def _on_ki(val):
            try:
                ctrl_params['ki'] = float(val)
                ctrl_params['reset'] = True
                print(f"[panel] Ki → {ctrl_params['ki']:.6f}  (integral reset)")
            except ValueError:
                pass

        def _on_lpf(val):
            try:
                ctrl_params['lpf_alpha'] = float(np.clip(float(val), 0.0, 1.0))
                print(f"[panel] LPF α → {ctrl_params['lpf_alpha']:.4f}")
            except ValueError:
                pass

        def _on_toggle(_):
            if not self._nn_avail:
                print("[panel] NN model not loaded — toggle disabled")
                return
            ctrl_params['use_nn'] = not ctrl_params['use_nn']
            ctrl_params['reset']  = True
            if ctrl_params['use_nn']:
                btn_tog.label.set_text('Force src: NN')
                btn_tog.ax.set_facecolor('# ffd8a0')
                print("[panel] Force source → NN  (integral reset)")
            else:
                btn_tog.label.set_text('Force src: ATI')
                btn_tog.ax.set_facecolor('# c8f5c8')
                print("[panel] Force source → ATI  (integral reset)")
            fig.canvas.draw_idle()

        def _on_reset(_):
            ctrl_params['reset'] = True
            print("[panel] Integrator reset")

        tb_fdes.on_submit(_on_fdes)
        tb_kp.on_submit(_on_kp)
        tb_ki.on_submit(_on_ki)
        tb_lpf.on_submit(_on_lpf)
        btn_tog.on_clicked(_on_toggle)
        btn_rst.on_clicked(_on_reset)

        # timer-based redraw
        def _update():
            with self._lock:
                t          = list(self._t)
                f_des      = list(self._f_des)
                f_ati      = list(self._f_ati)
                f_nn       = list(self._f_nn)
                f_sel_raw  = list(self._f_sel_raw)
                f_sel_filt = list(self._f_sel_filt)
                tau        = list(self._tau)
                tau_meas   = list(self._tau_meas)

            if len(t) < 2:
                return

            t_now = t[-1]
            t_lo  = t_now - self.WINDOW_S

            # update lines
            line_fdes.set_data(t, f_des)
            line_fati.set_data(t, f_ati)
            line_fnn.set_data(t, f_nn)

            line_ctrl_des.set_data(t, f_des)
            line_ctrl_raw.set_data(t, f_sel_raw)
            line_ctrl_filt.set_data(t, f_sel_filt)

            line_tau.set_data(t, tau)
            line_tau_meas.set_data(t, tau_meas)

            # update subplot 2 color + title based on active source
            use_nn = ctrl_params['use_nn']
            if use_nn:
                line_ctrl_raw.set_color('# ffd8a0')
                line_ctrl_filt.set_color('darkorange')
                ax_ctrl.set_title('Tuner — NN  (raw + filtered)', fontsize=9)
            else:
                line_ctrl_raw.set_color('# ffaaaa')
                line_ctrl_filt.set_color('red')
                ax_ctrl.set_title('Tuner — ATI  (raw + filtered)', fontsize=9)

            for ax in (ax_f, ax_ctrl, ax_t):
                ax.set_xlim(t_lo, t_now + 0.1)

            vis = slice(max(0, len(t) - int(self.WINDOW_S * LOOP_HZ)), None)

            def _ylim(ax, *series):
                vals = [v for s in series for v in s[vis]]
                if not vals:
                    return
                lo, hi = min(vals), max(vals)
                pad = max(0.5, (hi - lo) * 0.15)
                ax.set_ylim(lo - pad, hi + pad)

            _ylim(ax_f,    f_des, f_ati, f_nn)
            _ylim(ax_ctrl, f_des, f_sel_raw, f_sel_filt)
            _ylim(ax_t,    tau, tau_meas)

            fig.canvas.draw_idle()

        self._timer = fig.canvas.new_timer(interval=self.UPDATE_MS)
        self._timer.add_callback(_update)
        self._timer.start()
        plt.show()
        self._timer.stop()


# data logger
class DataLogger:
    COLUMNS = [
        'time_s',
        'f_des_N', 'f_ati_N', 'f_nn_N', 'f_cal_N',
        'use_nn_sensor',
        'f_sel_raw_N', 'f_sel_filt_N',
        'tau_cmd_Nm', 'tau_meas_Nm',
        'pos_rad', 'vel_rad_s',
        'kp', 'ki', 'lpf_alpha',
    ]

    def __init__(self, path: str | None = None) -> None:
        from datetime import datetime
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(HERE, f'nn_ctrl_{ts}.csv')
        self.path    = path
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True, name='DataLogger')
        self._thread.start()
        print(f"[log] {self.path}")

    def log(self, time_s: float,
            f_des: float, f_ati: float, f_nn: float, f_cal: float,
            use_nn: bool,
            f_sel_raw: float, f_sel_filt: float,
            tau_cmd: float, tau_meas: float,
            pos: float, vel: float,
            kp: float, ki: float, lpf_alpha: float) -> None:
        self._queue.put_nowait((
            f'{time_s:.6f}',
            f'{f_des:.6f}', f'{f_ati:.6f}', f'{f_nn:.6f}', f'{f_cal:.6f}',
            '1' if use_nn else '0',
            f'{f_sel_raw:.6f}', f'{f_sel_filt:.6f}',
            f'{tau_cmd:.6f}', f'{tau_meas:.6f}',
            f'{pos:.6f}', f'{vel:.6f}',
            f'{kp:.6f}', f'{ki:.6f}', f'{lpf_alpha:.4f}',
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


# control loop
def _control_worker(bus: MotorBus, motor: MITMotor,
                    predictor, 
                    plotter: PlotThread, logger: 'DataLogger | None' = None) -> None:
    try:
        bus.drain()
        time.sleep(0.5)
        motor.zero()

        print(
            f"Motor {MOTOR_ID} ready.\n"
            f"  F_des={F_DESIRED_N:+.2f} N  Kp={KP_F}  Ki={KI_F}\n"
            f"  LPF α={LPF_ALPHA}  JACOBIAN={JACOBIAN}\n"
            f"  Torque limit : ±{TORQUE_LIMIT} N·m\n"
            f"  Control law  : τ = J*(Kp*err + Ki*∫)  [pure PI, no feedforward]\n"
            f"  TSLM available : {'yes' if predictor is not None else 'no (model not found)'}\n"
            f"Press Ctrl+C or close the plot window to stop.\n"
        )

        motor.cmd.kp       = 0.0
        motor.cmd.kd       = 0.1
        motor.cmd.position = 0.0
        motor.cmd.velocity = 0.0

        integral   = 0.0
        f_ati_filt = float(-ft_latest[2])
        f_nn_filt  = f_ati_filt
        dt_nom     = 1.0 / LOOP_HZ
        t0         = time.perf_counter()
        prev_t     = t0
        _last_print = 0.0   # throttle console output to ~5 Hz

        try:
            while not Terminate.is_set():
                now = time.perf_counter()
                t   = now - t0

                f_des  = ctrl_params['f_des']
                kp     = ctrl_params['kp']
                ki     = ctrl_params['ki']
                alpha  = ctrl_params['lpf_alpha']
                use_nn = ctrl_params['use_nn']

                if ctrl_params['reset']:
                    integral   = 0.0
                    f_ati_filt = float(-ft_latest[2])
                    f_nn_filt  = f_ati_filt
                    ctrl_params['reset'] = False

                # sense
                tau_meas = motor.state.torque
                pos      = motor.state.position
                vel      = motor.state.velocity

                f_ati_raw  = float(-ft_latest[2])
                f_ati_filt = alpha * f_ati_raw + (1.0 - alpha) * f_ati_filt

                f_nn_raw  = predictor.predict(tau_meas, pos, vel) if predictor is not None else f_ati_raw
                f_nn_filt = alpha * f_nn_raw + (1.0 - alpha) * f_nn_filt

                f_cal  = tau_meas / JACOBIAN if abs(JACOBIAN) > 1e-9 else 0.0
                f_ctrl = f_nn_filt if use_nn else f_ati_filt

                # pre-select raw/filt for subplot 2
                f_sel_raw  = f_nn_raw  if use_nn else f_ati_raw
                f_sel_filt = f_nn_filt if use_nn else f_ati_filt

                # ── pure PI (no feedforward: Kp=Ki=0 → τ=0) ──────────────────
                dt_actual = now - prev_t
                prev_t = now

                f_err     = f_des - f_ctrl
                integral += f_err * dt_actual
                if ki > 1e-12:
                    windup_lim = TORQUE_LIMIT / (JACOBIAN * ki)
                    integral   = float(np.clip(integral, -windup_lim, windup_lim))

                tau_raw = JACOBIAN * (f_des + kp * f_err + ki * integral)
                tau_cmd = float(np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT))
                motor.cmd.torque = tau_cmd

                plotter.push(t, f_des, f_ati_raw, f_nn_raw,
                             f_sel_raw, f_sel_filt, tau_cmd, tau_meas)
                motor.update(t)


                # logger.log(
                #     t,
                #     f_des, f_ati_filt, f_nn_filt, f_cal,
                #     use_nn,
                #     f_sel_raw, f_sel_filt,
                #     tau_cmd, tau_meas,
                #     pos, vel,
                #     kp, ki, alpha,
                # )

                if t - _last_print >= 0.2:   # ~5 Hz console output
                    _last_print = t
                    print(
                        f"t={t:7.3f}s  "
                        f"F_des={f_des:+.2f}  "
                        f"F_ctrl={f_ctrl:+.4f}({'NN' if use_nn else 'ATI'})  "
                        f"F_nn={f_nn_filt:+.4f}  F_cal={f_cal:+.4f}  "
                        f"err={f_err:+.4f}  ∫={integral:+.6f}  "
                        f"τ={tau_cmd:+.3f}N·m  "
                        f"pos={pos:+.4f}rad  vel={vel:+.4f}rad/s  "
                        f"T={motor.state.temp}°C"
                    )
                sleep_time = dt_nom - (time.perf_counter() - now)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        except MotorFaultError as e:
            print(f"\n[FAULT]   motor_id={e.motor_id}  code={e.code}  {e}")
        except MotorTimeoutError as e:
            print(f"\n[TIMEOUT] motor_id={e.motor_id}  timeout={e.timeout}s")

    finally:
        Terminate.set()


# entry point
def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TSLM] device: {device}")
    predictor = load_ts_model(MODEL_PATH, device)

    print("Initialising ATI F/T sensor ...")
    sensor = ftsensor()
    # sensor.set_read_freq(SENSOR_HZ)
    sensor.start_task()
    time.sleep(0.1)
    print(f"Biasing F/T sensor for {BIAS_DURATION:.1f} s — keep the sensor unloaded ...")
    sensor.reBias(duration=BIAS_DURATION)
    print("Bias complete.")
    time.sleep(0.5)

    threading.Thread(target=ft_reader_thread, args=(sensor,),
                     daemon=True, name='FTReader').start()
    print("F/T background reader started.")

    raw_bus = can.interface.Bus(interface='slcan', channel=COM_PORT, bitrate=1_000_000)
    bus     = MotorBus(raw_bus)
    motor   = MITMotor(bus, motor_id=MOTOR_ID, model=MOTOR_MODEL)
    motor.enable()

    plotter = PlotThread(nn_available=(predictor is not None))
    # logger  = DataLogger()          # uncomment to enable CSV logging
    logger = None                     # remove this line when logger is enabled

    ctrl_thread = threading.Thread(
        target=_control_worker,
        args=(bus, motor, predictor, plotter, logger),
        daemon=True, name='Control',
    )
    ctrl_thread.start()

    try:
        plotter._run()          # blocks in plt.show()
    except KeyboardInterrupt:
        Terminate.set()
    finally:
        Terminate.set()
        ctrl_thread.join(timeout=3.0)
        # logger.close()
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
