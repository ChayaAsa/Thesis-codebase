from __future__ import annotations

import collections
import threading
from typing import Any, Callable, Sequence


_WINDOW_S  = 10.0   # rolling x-axis window [s]
_UPDATE_MS = 100    # GUI refresh period [ms]
_LOOP_HZ   = 100    # assumed control-loop rate for deque sizing


# Line  — one plotted time series

class Line:

    def __init__(self, *,
                 color: str | None = None,
                 label: str = '',
                 lw: float = 1.5,
                 ls: str = '-',
                 alpha: float = 1.0,
                 window_s: float = _WINDOW_S,
                 loop_hz: float = _LOOP_HZ) -> None:
        self.color = color
        self.label = label
        self.lw    = lw
        self.ls    = ls
        self.alpha = alpha

        _maxlen = int(window_s * loop_hz * 1.5)
        self._x: collections.deque[float] = collections.deque(maxlen=_maxlen)
        self._y: collections.deque[float] = collections.deque(maxlen=_maxlen)
        self._ts_box: list[float] | None = None   # injected by _PlotSlot.set()

    def push(self, x_or_y: float, y: float | None = None) -> None:
        if y is None:
            if self._ts_box is None:
                raise RuntimeError(
                    "push(y) requires the Line to be registered with a PlotThread "
                    "via slot.set([...]) before calling p.start().  "
                    "Use push(x, y) instead, or call slot.set() first."
                )
            self._x.append(self._ts_box[0])
            self._y.append(float(x_or_y))
        else:
            self._x.append(float(x_or_y))
            self._y.append(float(y))

    def snapshot(self) -> tuple[list[float], list[float]]:
        return list(self._x), list(self._y)


# _PlotSlot  — one subplot row

class _PlotSlot:

    def __init__(self, ts_box: list[float]) -> None:
        self.title  = ''
        self.ylabel = ''
        self.xlabel = ''
        self.grid   = True
        self._lines:  list[Line]   = []
        self._ts_box: list[float]  = ts_box

    def set(self, lines: Sequence[Line]) -> None:
        self._lines = list(lines)
        for line in self._lines:
            line._ts_box = self._ts_box


# PlotThread

class PlotThread:

    def __init__(self, *,
                 title: str = 'Live Plot',
                 window_s: float = _WINDOW_S,
                 update_ms: int = _UPDATE_MS,
                 on_close:  Callable[[], None] | None = None,
                 on_reset:  Callable[[], None] | None = None,
                 on_stop:   Callable[[], None] | None = None,
                 on_record: Callable[[], None] | None = None) -> None:
        self._title     = title
        self._window_s  = window_s
        self._update_ms = update_ms
        self._on_close  = on_close
        self._on_reset  = on_reset
        self._on_stop   = on_stop
        self._on_record = on_record

        self._ts_box: list[float] = [0.0]
        self._slots = [_PlotSlot(self._ts_box) for _ in range(5)]

        self._lock        = threading.Lock()
        self._info_lines: list[str]   = []
        self._cmd:        list[tuple] = []

        self.params: dict[str, Any] = {}
        """Values edited in the GUI control panel.  Read from the control loop."""

    @property
    def ts(self) -> float:
        return self._ts_box[0]

    @ts.setter
    def ts(self, value: float) -> None:
        self._ts_box[0] = float(value)

    # Slot access

    def plot(self, idx: int) -> _PlotSlot:
        if not 1 <= idx <= 5:
            raise ValueError(f"plot index must be 1–5, got {idx}")
        return self._slots[idx - 1]

    # Control-panel config

    @property
    def command(self) -> list:
        with self._lock:
            return list(self._cmd)

    @command.setter
    def command(self, params: list[tuple]) -> None:
        with self._lock:
            self._cmd = list(params)
        for item in params:
            key = item[0]
            if key not in self.params:
                # 3-tuple (key, label, default) seeds from the provided default value
                self.params[key] = item[2] if len(item) == 3 else 0.0

    # Live info text

    @property
    def info(self) -> list[str]:
        with self._lock:
            return list(self._info_lines)

    @info.setter
    def info(self, lines: list[str]) -> None:
        with self._lock:
            self._info_lines = list(lines)

    # Lifecycle

    def update(self) -> None:
        pass

    def run(self) -> None:
        self._run()

    # Internal GUI (must run on the main thread)

    def _run(self) -> None:
        import tkinter as tk
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title(self._title)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=0, minsize=320)
        root.rowconfigure(0, weight=1)

        # ── Left: 5-panel matplotlib figure ──────────────────────────────
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky='nsew')

        height_ratios = [3 if s._lines else 1 for s in self._slots]
        fig, axes = plt.subplots(
            5, 1, figsize=(9, 10),
            gridspec_kw={'height_ratios': height_ratios},
        )
        fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.05, hspace=0.50)
        fig.suptitle(self._title, fontsize=11)

        _colors = ['steelblue', 'firebrick', 'goldenrod', 'mediumseagreen',
                   'mediumpurple', 'darkorange', 'teal', 'deeppink']

        mpl_lines: list[list] = [[] for _ in range(5)]

        for si, slot in enumerate(self._slots):
            ax = axes[si]
            ax.set_ylabel(slot.ylabel or (f'Slot {si+1}' if slot._lines else ''),
                          fontsize=8)
            if slot.title:
                ax.set_title(slot.title, fontsize=9, pad=2)
            if slot.xlabel or si == 4:
                ax.set_xlabel(slot.xlabel or 'Time [s]', fontsize=8)
            ax.grid(slot.grid, alpha=0.3)
            ax.tick_params(labelsize=7)

            for li, line in enumerate(slot._lines):
                c = line.color or _colors[li % len(_colors)]
                obj, = ax.plot([], [], color=c, lw=line.lw,
                               ls=line.ls, alpha=line.alpha,
                               label=line.label or f'line{li+1}')
                mpl_lines[si].append(obj)

            if slot._lines:
                ax.legend(loc='upper left', fontsize=7,
                          ncol=min(len(slot._lines), 3))

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # ── Right: control panel ──────────────────────────────────────────
        with self._lock:
            cmd_snap = list(self._cmd)

        ctrl = tk.Frame(root, relief='groove', bd=2, padx=8, pady=8)
        ctrl.grid(row=0, column=1, sticky='nsew', padx=(0, 6), pady=6)
        ctrl.columnconfigure(1, weight=1)

        _row = [0]
        def _r(inc: int = 1) -> int:
            r = _row[0]; _row[0] += inc; return r

        def _sep():
            tk.Frame(ctrl, height=2, bg='# aaaaaa').grid(
                row=_r(), column=0, columnspan=2, sticky='ew', pady=6)

        tk.Label(ctrl, text='Parameters',
                 font=('TkDefaultFont', 11, 'bold')).grid(
            row=_r(), column=0, columnspan=2, pady=(0, 6))

        # entries[key] = (var_or_boolvar, widget_or_None, lo_or_None, hi_or_None, kind)
        # kind: 'float' | 'int' | 'str' | 'bool'
        entries: dict[str, tuple] = {}

        def _kind_of(item: tuple) -> str:
            if len(item) > 4:
                return str(item[4])          # explicit 5th element: 'int' or 'float'
            if len(item) >= 4:
                return 'float'
            if len(item) < 3:
                return 'float'
            # 3-tuple (key, label, default): infer from default value
            default = item[2]
            if isinstance(default, bool):    # bool before int (bool is subclass of int)
                return 'bool'
            if isinstance(default, int):
                return 'int'
            if isinstance(default, str):
                return 'str'
            return 'float'

        for item in cmd_snap:
            key, lbl = item[0], item[1]
            kind = _kind_of(item)
            r = _r()

            tk.Label(ctrl, text=lbl + ':', anchor='w',
                     font=('TkDefaultFont', 10)).grid(
                row=r, column=0, sticky='w', pady=3, padx=(0, 6))

            if kind == 'bool':
                bvar = tk.BooleanVar(value=bool(self.params.get(key, item[2] if len(item) == 3 else False)))
                def _on_toggle(k=key, bv=bvar):
                    self.params[k] = bv.get()
                chk = tk.Checkbutton(ctrl, variable=bvar, command=_on_toggle)
                chk.grid(row=r, column=1, sticky='w', pady=3)
                entries[key] = (bvar, chk, None, None, 'bool')

            elif kind == 'str':
                current = str(self.params.get(key, item[2] if len(item) == 3 else ''))
                var = tk.StringVar(value=current)
                ent = tk.Entry(ctrl, textvariable=var, width=10,
                               font=('Courier', 11), justify='right',
                               relief='solid', bd=1)
                ent.grid(row=r, column=1, sticky='ew', pady=3)
                def _apply_str(*_, k=key, sv=var, e=ent):
                    self.params[k] = sv.get()
                    e.config(bg='# d4edda')
                    root.after(600, lambda: e.config(bg='white'))
                ent.bind('<Return>',   _apply_str)
                ent.bind('<FocusOut>', _apply_str)
                entries[key] = (var, ent, None, None, 'str')

            else:  # 'float' or 'int'
                lo, hi = float(item[2]), float(item[3])
                cur = self.params.get(key, 0)
                fmt = str(int(round(float(cur)))) if kind == 'int' else f'{float(cur):.4g}'
                var = tk.StringVar(value=fmt)
                ent = tk.Entry(ctrl, textvariable=var, width=10,
                               font=('Courier', 13), justify='right',
                               relief='solid', bd=1)
                ent.grid(row=r, column=1, sticky='ew', pady=3)
                def _apply_num(*_, k=key, sv=var, e=ent, clo=lo, chi=hi, kd=kind):
                    try:
                        raw = float(sv.get())
                        v: Any = int(round(raw)) if kd == 'int' else raw
                        v = max(type(v)(clo), min(type(v)(chi), v))
                        self.params[k] = v
                        sv.set(str(v) if kd == 'int' else f'{v:.4g}')
                        e.config(bg='# d4edda')
                        root.after(600, lambda: e.config(bg='white'))
                    except ValueError:
                        e.config(bg='# f8d7da')
                        root.after(600, lambda: e.config(bg='white'))
                ent.bind('<Return>',   _apply_num)
                ent.bind('<FocusOut>', _apply_num)
                entries[key] = (var, ent, lo, hi, kind)

        _sep()

        def _apply_all():
            for k, (sv, e, lo, hi, kind) in entries.items():
                if kind == 'bool':
                    continue   # live-updated on toggle
                elif kind == 'str':
                    self.params[k] = sv.get()
                    e.config(bg='# d4edda')
                    root.after(600, lambda _e=e: _e.config(bg='white'))
                else:  # float or int
                    try:
                        raw = float(sv.get())
                        v: Any = int(round(raw)) if kind == 'int' else raw
                        v = max(type(v)(lo), min(type(v)(hi), v))
                        self.params[k] = v
                        sv.set(str(v) if kind == 'int' else f'{v:.4g}')
                        e.config(bg='# d4edda')
                        root.after(600, lambda _e=e: _e.config(bg='white'))
                    except ValueError:
                        e.config(bg='# f8d7da')
                        root.after(600, lambda _e=e: _e.config(bg='white'))

        tk.Button(ctrl, text='Apply All  [Enter]',
                  font=('TkDefaultFont', 10), bg='# d0e8ff',
                  relief='raised', pady=4,
                  command=_apply_all).grid(
            row=_r(), column=0, columnspan=2, sticky='ew', pady=2)

        def _do_reset():
            if self._on_reset is not None:
                self._on_reset()

        def _do_stop():
            if self._on_stop is not None:
                self._on_stop()
            _on_close()

        tk.Button(ctrl, text='Reset',
                  font=('TkDefaultFont', 10), bg='# fffacd',
                  relief='raised', pady=4,
                  command=_do_reset).grid(
            row=_r(), column=0, columnspan=2, sticky='ew', pady=2)

        if self._on_record is not None:
            _rec_btn_state = [False]   # [is_recording]
            _rec_btn = tk.Button(ctrl, text='⏺  Start Record',
                                 font=('TkDefaultFont', 10, 'bold'),
                                 bg='# ffcccc', relief='raised', pady=4)

            def _do_record(_btn=_rec_btn, _state=_rec_btn_state):
                if not _state[0]:
                    _state[0] = True
                    _btn.config(text='⏺  Recording…', bg='# ff4444', fg='white')
                else:
                    _state[0] = False
                    _btn.config(text='⏺  Start Record', bg='# ffcccc', fg='black')
                if self._on_record is not None:
                    self._on_record()

            _rec_btn.config(command=_do_record)
            _rec_btn.grid(row=_r(), column=0, columnspan=2, sticky='ew', pady=2)

        tk.Button(ctrl, text='E - S T O P',
                  font=('TkDefaultFont', 12, 'bold'),
                  bg='# e05050', fg='white',
                  relief='raised', pady=6,
                  command=_do_stop).grid(
            row=_r(), column=0, columnspan=2, sticky='ew', pady=(4, 2))

        _sep()

        # Info / status area
        tk.Label(ctrl, text='Info',
                 font=('TkDefaultFont', 10, 'bold')).grid(
            row=_r(), column=0, columnspan=2, sticky='w', pady=(0, 2))

        _INFO_SLOTS = 8
        info_labels: list[tk.Label] = []
        for _ in range(_INFO_SLOTS):
            lbl = tk.Label(ctrl, text='', anchor='w',
                           font=('Courier', 9), justify='left', wraplength=290)
            lbl.grid(row=_r(), column=0, columnspan=2, sticky='w', pady=0)
            info_labels.append(lbl)

        # Periodic refresh
        _frame = [0]
        _running = [True]

        def _tick() -> None:
            if not _running[0]:
                return
            for si, slot in enumerate(self._slots):
                ax = axes[si]
                any_data = False
                xs: list[float] = []
                for li, line in enumerate(slot._lines):
                    xs, ys = line.snapshot()
                    if len(xs) < 2:
                        continue
                    mpl_lines[si][li].set_data(xs, ys)
                    any_data = True
                if any_data:
                    t_now = xs[-1]
                    ax.set_xlim(t_now - self._window_s, t_now + 0.1)

            _frame[0] += 1
            if _frame[0] % 5 == 0:
                for si, slot in enumerate(self._slots):
                    if not slot._lines:
                        continue
                    vals: list[float] = []
                    for line in slot._lines:
                        _, ys = line.snapshot()
                        vals.extend(ys)
                    if vals:
                        lo, hi = min(vals), max(vals)
                        pad = max(0.3, (hi - lo) * 0.15)
                        axes[si].set_ylim(lo - pad, hi + pad)

            canvas.draw_idle()

            with self._lock:
                info_snap = list(self._info_lines)
            for i, lbl in enumerate(info_labels):
                lbl.config(text=info_snap[i] if i < len(info_snap) else '')

            if _running[0]:
                root.after(self._update_ms, _tick)

        root.after(self._update_ms, _tick)

        def _on_close() -> None:
            if not _running[0]:
                return
            _running[0] = False
            if self._on_close is not None:
                self._on_close()
            root.quit()

        root.protocol('WM_DELETE_WINDOW', _on_close)

        # Ctrl+C in the terminal would otherwise be swallowed by Tk's callback
        # exception handler (it catches everything but SystemExit), leaving the
        # mainloop running and spewing "Exception in Tkinter callback".  Install
        # a SIGINT handler so Ctrl+C closes the window cleanly instead.
        import signal
        try:
            signal.signal(signal.SIGINT, lambda *_: _on_close())
        except (ValueError, OSError):
            pass   # not on the main thread / platform unsupported

        try:
            root.mainloop()
        finally:
            _running[0] = False
            try:
                root.destroy()   # cancel pending after()/draw callbacks
            except Exception:
                pass


# Visual test

if __name__ == '__main__':
    import math
    import time

    stop = threading.Event()

    # Line descriptors
    f_filt = Line(color='red',         label='F filt',  lw=1.5)
    f_des  = Line(color='blue',        label='F des',   lw=1.5, ls='--')
    f_raw  = Line(color='# fd9999', label='F raw', lw=0.8, alpha=0.5)

    q_des  = [Line(label=f'q{i+1} des',  ls='--', lw=1.5) for i in range(3)]
    q_meas = [Line(label=f'q{i+1} meas', lw=1.0,  alpha=0.7) for i in range(3)]

    qdot   = [Line(label=f'qd{i+1}', lw=1.2) for i in range(3)]

    # x_ee   = Line(color='teal',       label='EE x',  lw=1.5)
    # z_ee   = Line(color='darkorange', label='EE z',  lw=1.5, ls='--')

    loop_jitter = Line(color='mediumpurple', label='jitter [ms]', lw=1.0)

    # PlotThread setup
    p = PlotThread(title='plot_thread.py — visual test', on_close=stop.set)

    p.plot(1).set([f_filt, f_des, f_raw])
    p.plot(1).ylabel = 'Force [N]'
    p.plot(1).title  = 'Force'

    p.plot(2).set(q_des + q_meas)
    p.plot(2).ylabel = 'q [rad]'
    p.plot(2).title  = 'Joint positions'

    p.plot(3).set(qdot)
    p.plot(3).ylabel = 'qdot [rad/s]'
    p.plot(3).title  = 'Joint velocities'

    # p.plot(4).set([x_ee, z_ee])
    # p.plot(4).ylabel = 'pos [m]'
    # p.plot(4).title  = 'EE position'

    p.plot(5).set([loop_jitter])
    p.plot(5).ylabel = 'ms'
    p.plot(5).title  = 'Loop jitter'

    p.command = [
        ('f_des',    'F des  [N]',   0.0,  50.0),
        ('b_adm',    'B adm',        0.1, 500.0),
        ('k_spring', 'K spring',     0.0, 200.0),
        ('kp1',      'Kp motor 1',   0.0,  50.0),
        ('kp2',      'Kp motor 2',   0.0,  50.0),
        ('kp3',      'Kp motor 3',   0.0,  50.0),
        ('kd1',      'Kd motor 1',   0.0,  10.0),
        ('kd2',      'Kd motor 2',   0.0,  10.0),
        ('kd3',      'Kd motor 3',   0.0,  10.0),
        ('lpf_alpha','LPF  alpha',   0.01,  1.0),
        ('vel_limit','Vel  [r/s]',   0.05, 10.0),
    ]
    p.params.update({
        'f_des': 5.0, 'b_adm': 10.0, 'k_spring': 1.0,
        'kp1': 5.0, 'kp2': 5.0, 'kp3': 5.0,
        'kd1': 0.5, 'kd2': 0.5, 'kd3': 0.5,
        'lpf_alpha': 0.3, 'vel_limit': 1.0,
    })

    # Simulated control loop (background thread)
    dt = 0.01   # 100 Hz

    def _sim() -> None:
        t0     = time.monotonic()
        prev   = time.monotonic()
        phases = [0.0, 0.5, 1.0]
        filt   = 0.0

        while not stop.is_set():
            tick_start = time.monotonic()
            t = tick_start - t0
            p.ts = t

            amp   = p.params.get('f_des',     5.0)
            alpha = p.params.get('lpf_alpha',  0.3)

            # Slot 1 — force
            noise   = 0.4 * math.sin(17.3 * t) + 0.2 * math.sin(31.1 * t)
            raw_val = amp * math.sin(2 * math.pi * 0.3 * t) + noise
            filt    = alpha * raw_val + (1 - alpha) * filt
            f_raw.push(raw_val)
            f_filt.push(filt)
            f_des.push(amp * 0.8)

            # Slot 2 — joint positions
            for i in range(3):
                qd = 0.5 * math.sin(2 * math.pi * 0.2 * t + phases[i])
                qm = qd + 0.05 * math.sin(2 * math.pi * 1.1 * t + phases[i])
                q_des[i].push(qd)
                q_meas[i].push(qm)

            # Slot 3 — joint velocities
            for i in range(3):
                qdot[i].push(0.3 * math.cos(2 * math.pi * 0.2 * t + phases[i]))

            # Slot 4 — EE position
            # x_ee.push(0.35 + 0.05 * math.sin(2 * math.pi * 0.15 * t))
            # z_ee.push(0.20 + 0.03 * math.cos(2 * math.pi * 0.15 * t))

            # Slot 5 — loop jitter
            jitter_ms = (time.monotonic() - prev - dt) * 1000.0
            loop_jitter.push(jitter_ms)
            prev = tick_start

            p.info = [
                f't      = {t:7.3f} s',
                f'F filt = {filt:+7.3f} N',
                f'F des  = {amp:+7.3f} N',
                f'q_des  = [{q_des[0]._y[-1]:+.3f}, {q_des[1]._y[-1]:+.3f}, {q_des[2]._y[-1]:+.3f}]',
                f'jitter = {jitter_ms:+.2f} ms',
                '',
                'Close window to stop.',
            ]
            p.update()

            elapsed = time.monotonic() - tick_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    threading.Thread(target=_sim, daemon=True, name='SimLoop').start()
    p.run()   # blocks on main thread
