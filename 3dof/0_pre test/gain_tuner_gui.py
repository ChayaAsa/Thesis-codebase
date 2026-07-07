"""
GUI Kp/Kd tuner for the 3-DOF arm (AK45-10 motors, IDs 1, 2, 3) in MIT CAN mode.

What it does
------------
A Tkinter window with one panel per motor. For each motor you can type:
    kp, kd, p (position rad), v (velocity rad/s), t (feed-forward torque N·m)
and press Apply (or Enter in any field) to send them live. Each panel also has a
HOLD toggle:
    HOLD  -> latches the motor's CURRENT position and holds it stiffly with
             kp=100, kd=1, v=0, t=0  (no sudden motion — it holds where it is).
    FREE  -> restores the kp/kd/p/v/t you have in the entry fields.
The entry fields are never touched by HOLD, so FREE always returns to exactly the
"previous" values you had.

A background thread runs the control loop at 50 Hz and calls update_all() every
tick, so the motors never time out while you sit editing numbers.

Usage
-----
    1. Set the COM port (default COM18) and current limit, click Connect.
       On connect, all motors are ENABLED, zeroed at their current pose, and left
       FREE (kp=kd=0) so you can pose the arm by hand.
    2. Type gains for a motor, press Apply / Enter. Tune kp, kd, p, v, t live.
    3. Press a motor's HOLD button to lock it in place while you work on another.
    4. "Zero all" frees every motor and re-defines the current pose as 0 rad.
    5. Disconnect (or close the window) fades the gains out, then coasts.

SAFETY: start with small gains. kp ramps stiffness fast — a heavy arm with a
position error and high kp can snap hard. The current limit caps phase current.
"""

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

# ── Make tmotorcan importable whether it's pip-installed or run from the repo ──
try:
    import can
    from tmotorcan import (
        MITMotor, MotorBus, update_all,
        MotorFaultError, MotorTimeoutError,
    )
except ImportError:
    _src = Path(__file__).resolve().parents[2] / 'T-Motor-API' / 'src'
    sys.path.insert(0, str(_src))
    import can
    from tmotorcan import (
        MITMotor, MotorBus, update_all,
        MotorFaultError, MotorTimeoutError,
    )

# ── Configuration ─────────────────────────────────────────────────────────────
MOTOR_IDS         = [1, 2, 3]
MODEL             = 'AK45-10'
DEFAULT_PORT      = 'COM18'
BITRATE           = 1_000_000
MAX_TEMP_C        = 80
LOOP_HZ           = 50.0
DT                = 1.0 / LOOP_HZ
DEFAULT_CURRENT_A = 6.0          # per-motor phase-current cap [A]

HOLD_KP = 100.0                  # stiffness while holding [N·m/rad]
HOLD_KD = 1.0                    # damping while holding   [N·m·s/rad]

FIELDS = ['kp', 'kd', 'p', 'v', 't']
LABELS = {'kp': 'kp  [N·m/rad]', 'kd': 'kd  [N·m·s/rad]',
          'p':  'p   [rad]',     'v':  'v   [rad/s]', 't': 't   [N·m]'}
# Clamp ranges = AK45-10 MIT-mode encoding ranges (from ak45-10.yaml).
LIMITS = {'kp': (0.0, 500.0), 'kd': (0.0, 5.0),
          'p': (-12.5, 12.5), 'v': (-20.0, 20.0), 't': (-7.0, 7.0)}


def _fmt(x: float) -> str:
    return f"{x:g}"


# ═════════════════════════════════════════════════════════════════════════════
#  Shared state between the UI thread and the control thread
# ═════════════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe hand-off. The UI writes cmd/hold; the worker writes snap/status."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cmd  = {mid: {f: 0.0 for f in FIELDS} for mid in MOTOR_IDS}
        self.hold = {mid: False for mid in MOTOR_IDS}
        self.snap = {mid: {'pos': 0.0, 'vel': 0.0, 'tor': 0.0, 'temp': 0.0, 'err': 0}
                     for mid in MOTOR_IDS}
        self.requests = queue.Queue()      # 'zero' etc. — Queue is already thread-safe
        self.connected = False
        self._status = ("Disconnected.", None)   # (message, ok: True/False/None)

    def set_status(self, msg, ok=None):
        with self.lock:
            self._status = (msg, ok)

    def get_status(self):
        with self.lock:
            return self._status


# ═════════════════════════════════════════════════════════════════════════════
#  Control thread — owns the bus + motors, runs the 50 Hz loop
# ═════════════════════════════════════════════════════════════════════════════
class ControlWorker(threading.Thread):
    def __init__(self, port, current_limit, shared: SharedState):
        super().__init__(daemon=True)
        self.port          = port
        self.current_limit = current_limit
        self.shared        = shared
        self.stop_event    = threading.Event()

    def run(self):
        shared  = self.shared
        bus     = None
        motors  = []
        eff     = {mid: (0.0, 0.0) for mid in MOTOR_IDS}   # last effective (kp, kd)
        try:
            shared.set_status(f"Opening {self.port} ...", None)
            raw = can.interface.Bus(interface='slcan', channel=self.port,
                                    bitrate=BITRATE, frame_type='STD')
            bus    = MotorBus(raw)
            motors = [MITMotor(bus, motor_id=mid, model=MODEL, max_temp=MAX_TEMP_C)
                      for mid in MOTOR_IDS]

            bus.drain()
            time.sleep(1.5)                       # let the hardware finish booting

            for m in motors:
                m.enable()
                m.coast()
                if self.current_limit and self.current_limit > 0:
                    m.set_current_limit(self.current_limit)
                m.zero()                          # current pose -> 0 rad

            shared.connected = True
            shared.set_status("Connected — motors FREE (zeroed at current pose).", True)

            prev_hold = {mid: False for mid in MOTOR_IDS}
            hold_pos  = {mid: 0.0 for mid in MOTOR_IDS}
            t0        = time.perf_counter()
            next_t    = time.perf_counter()

            while not self.stop_event.is_set():
                self._drain_requests(motors, prev_hold, hold_pos)

                t = time.perf_counter() - t0
                with shared.lock:
                    holds = dict(shared.hold)
                    cmds  = {mid: dict(shared.cmd[mid]) for mid in MOTOR_IDS}

                for m in motors:
                    mid = m.id
                    if holds[mid]:
                        if not prev_hold[mid]:
                            hold_pos[mid] = m.state.position   # latch current pose
                        m.cmd.position = hold_pos[mid]
                        m.cmd.velocity = 0.0
                        m.cmd.torque   = 0.0
                        m.cmd.kp       = HOLD_KP
                        m.cmd.kd       = HOLD_KD
                    else:
                        c = cmds[mid]
                        m.cmd.position = c['p']
                        m.cmd.velocity = c['v']
                        m.cmd.torque   = c['t']
                        m.cmd.kp       = c['kp']
                        m.cmd.kd       = c['kd']
                    prev_hold[mid] = holds[mid]
                    eff[mid] = (m.cmd.kp, m.cmd.kd)

                update_all(motors, t)

                with shared.lock:
                    for m in motors:
                        s = shared.snap[m.id]
                        s['pos']  = m.state.position
                        s['vel']  = m.state.velocity
                        s['tor']  = m.state.torque
                        s['temp'] = m.state.temp
                        s['err']  = m.state.error

                next_t += DT
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.perf_counter()        # fell behind — resync

            self._fade_out(motors, eff)                 # soft release on clean stop

        except MotorFaultError as e:
            shared.set_status(f"FAULT: {e} — motors coasted.", False)
        except MotorTimeoutError as e:
            shared.set_status(f"TIMEOUT: {e} — check power/port. Motors coasted.", False)
        except Exception as e:                          # bus open failure, etc.
            shared.set_status(f"ERROR: {e}", False)
        finally:
            for m in motors:
                try:
                    m.close()
                except Exception:
                    pass
            if bus is not None:
                try:
                    bus.close()
                except Exception:
                    pass
            shared.connected = False
            msg, ok = shared.get_status()
            if ok is not False:                         # don't clobber a fault message
                shared.set_status("Disconnected.", None)

    def _drain_requests(self, motors, prev_hold, hold_pos):
        try:
            while True:
                req = self.shared.requests.get_nowait()
                if req == 'zero':
                    for m in motors:
                        m.coast()
                        m.zero()
                    with self.shared.lock:
                        for mid in MOTOR_IDS:
                            self.shared.hold[mid] = False
                    for mid in MOTOR_IDS:
                        prev_hold[mid] = False
                        hold_pos[mid]  = 0.0
                    self.shared.set_status("Zeroed all motors at current pose.", True)
        except queue.Empty:
            pass

    def _fade_out(self, motors, eff, dur=0.4):
        """Ramp each motor's gains to zero while holding its current pose, then coast."""
        steps = max(1, int(dur / DT))
        for i in range(steps - 1, -1, -1):
            f = i / steps
            for m in motors:
                kp0, kd0 = eff[m.id]
                m.cmd.position = m.state.position
                m.cmd.velocity = 0.0
                m.cmd.torque   = 0.0
                m.cmd.kp       = kp0 * f
                m.cmd.kd       = kd0 * f
            update_all(motors)
            time.sleep(DT)
        for m in motors:
            m.coast()


# ═════════════════════════════════════════════════════════════════════════════
#  Tkinter UI (runs on the main thread; never touches motor objects directly)
# ═════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.shared = SharedState()
        self.worker = None

        self.entries   = {mid: {f: tk.StringVar(value='0') for f in FIELDS}
                          for mid in MOTOR_IDS}
        self.hold_vars = {mid: tk.BooleanVar(value=False) for mid in MOTOR_IDS}
        self.readout   = {mid: {} for mid in MOTOR_IDS}

        root.title("3-DOF Motor Gain Tuner — kp / kd / p / v / t")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_topbar()
        self._build_panels()
        self._build_footer()
        self._poll()

    # ── Top bar: connection + global actions ──────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self.root, padx=8, pady=6)
        bar.pack(fill='x')

        tk.Label(bar, text="COM port:").pack(side='left')
        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        tk.Entry(bar, textvariable=self.port_var, width=8).pack(side='left', padx=(2, 10))

        tk.Label(bar, text="Current limit [A]:").pack(side='left')
        self.current_var = tk.StringVar(value=_fmt(DEFAULT_CURRENT_A))
        tk.Entry(bar, textvariable=self.current_var, width=6).pack(side='left', padx=(2, 10))

        self.connect_btn = tk.Button(bar, text="Connect", width=12,
                                     command=self.on_toggle_connection)
        self.connect_btn.pack(side='left', padx=4)

        tk.Button(bar, text="Apply all", command=self.apply_all).pack(side='left', padx=4)
        tk.Button(bar, text="Zero all",  command=self.on_zero_all).pack(side='left', padx=4)

    # ── Per-motor panels ──────────────────────────────────────────────────────
    def _build_panels(self):
        panels = tk.Frame(self.root, padx=8, pady=4)
        panels.pack(fill='both', expand=True)

        for col, mid in enumerate(MOTOR_IDS):
            lf = tk.LabelFrame(panels, text=f"  Motor {mid}  ", padx=8, pady=8,
                               font=('Segoe UI', 10, 'bold'))
            lf.grid(row=0, column=col, padx=6, pady=4, sticky='nsew')
            panels.columnconfigure(col, weight=1)

            for r, f in enumerate(FIELDS):
                tk.Label(lf, text=LABELS[f], anchor='w', width=15)\
                    .grid(row=r, column=0, sticky='w', pady=1)
                e = tk.Entry(lf, textvariable=self.entries[mid][f], width=10,
                             justify='right')
                e.grid(row=r, column=1, sticky='e', pady=1)
                e.bind('<Return>', lambda _e, m=mid: self.apply_one(m))

            tk.Button(lf, text="Apply", command=lambda m=mid: self.apply_one(m))\
                .grid(row=len(FIELDS), column=0, columnspan=2, sticky='ew', pady=(6, 2))

            hold = tk.Checkbutton(
                lf, text="HOLD", variable=self.hold_vars[mid],
                indicatoron=False, width=12, pady=6,
                font=('Segoe UI', 10, 'bold'),
                selectcolor='#e74c3c', activebackground='#f1948a',
                command=lambda m=mid: self.on_hold(m))
            hold.grid(row=len(FIELDS) + 1, column=0, columnspan=2, sticky='ew', pady=2)

            # ── live readout ──
            ro = tk.Frame(lf)
            ro.grid(row=len(FIELDS) + 2, column=0, columnspan=2, sticky='ew', pady=(8, 0))
            for rr, (key, cap) in enumerate(
                    [('pos', 'pos'), ('vel', 'vel'), ('tor', 'torque'), ('temp', 'temp')]):
                tk.Label(ro, text=cap, anchor='w', width=7,
                         fg='#555').grid(row=rr, column=0, sticky='w')
                lbl = tk.Label(ro, text='—', anchor='e', width=12,
                               font=('Consolas', 9))
                lbl.grid(row=rr, column=1, sticky='e')
                self.readout[mid][key] = lbl

    def _build_footer(self):
        self.status_lbl = tk.Label(self.root, text="Disconnected.", anchor='w',
                                   padx=8, pady=4, font=('Segoe UI', 9))
        self.status_lbl.pack(fill='x')
        tk.Label(self.root, anchor='w', padx=8, fg='#777',
                 font=('Segoe UI', 8),
                 text="HOLD latches current position with kp=100, kd=1. "
                      "FREE returns to the entry values. Enter applies a field.")\
            .pack(fill='x', pady=(0, 6))

    # ── Actions ───────────────────────────────────────────────────────────────
    def on_toggle_connection(self):
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop_event.set()
            self.shared.set_status("Disconnecting (soft fade-out) ...", None)
        else:
            port = self.port_var.get().strip()
            if not port:
                self.shared.set_status("Enter a COM port first.", False)
                return
            try:
                cur = float(self.current_var.get())
            except ValueError:
                cur = 0.0
            self.apply_all()                      # push current entry values first
            self.worker = ControlWorker(port, cur, self.shared)
            self.worker.start()

    def apply_one(self, mid):
        vals = {}
        for f in FIELDS:
            raw = self.entries[mid][f].get()
            try:
                x = float(raw)
            except ValueError:
                self.shared.set_status(f"Motor {mid}: '{f}' = '{raw}' is not a number.", False)
                return
            lo, hi = LIMITS[f]
            x = max(lo, min(hi, x))
            self.entries[mid][f].set(_fmt(x))     # write back the clamped value
            vals[f] = x
        with self.shared.lock:
            self.shared.cmd[mid] = vals
        self.shared.set_status(
            f"Motor {mid}: applied  " +
            "  ".join(f"{f}={_fmt(vals[f])}" for f in FIELDS), True)

    def apply_all(self):
        for mid in MOTOR_IDS:
            self.apply_one(mid)

    def on_hold(self, mid):
        with self.shared.lock:
            self.shared.hold[mid] = self.hold_vars[mid].get()

    def on_zero_all(self):
        for mid in MOTOR_IDS:
            for f in FIELDS:
                self.entries[mid][f].set('0')
            self.hold_vars[mid].set(False)
        self.apply_all()
        with self.shared.lock:
            for mid in MOTOR_IDS:
                self.shared.hold[mid] = False
        if self.shared.connected:
            self.shared.requests.put('zero')
        else:
            self.shared.set_status("Entries reset to 0 (connect to zero the motors).", None)

    # ── Periodic refresh ──────────────────────────────────────────────────────
    def _poll(self):
        with self.shared.lock:
            snaps = {mid: dict(self.shared.snap[mid]) for mid in MOTOR_IDS}
        msg, ok = self.shared.get_status()
        alive = self.worker is not None and self.worker.is_alive()

        self.connect_btn.config(text="Disconnect" if alive else "Connect")
        self.status_lbl.config(
            text=msg, fg={True: '#1e8449', False: '#c0392b', None: '#000'}[ok])

        for mid in MOTOR_IDS:
            s = snaps[mid]
            self.readout[mid]['pos'].config(text=f"{s['pos']:+.3f} rad")
            self.readout[mid]['vel'].config(text=f"{s['vel']:+.3f} r/s")
            self.readout[mid]['tor'].config(text=f"{s['tor']:+.3f} Nm")
            temp = s['temp']
            self.readout[mid]['temp'].config(
                text=f"{temp:.0f} °C" + ("  ERR" if s['err'] else ""),
                fg='#c0392b' if (temp >= MAX_TEMP_C - 10 or s['err']) else '#000')

        self.root.after(60, self._poll)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def on_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop_event.set()
            self._wait_then_destroy()
        else:
            self.root.destroy()

    def _wait_then_destroy(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self._wait_then_destroy)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
