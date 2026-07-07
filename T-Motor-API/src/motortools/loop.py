import os
import signal
import sys
import threading
import time
from math import sqrt

_WINDOWS     = sys.platform == 'win32'
_BUSY_WAIT_S = 0.0002   # busy-spin duration at end of each sleep interval (200 µs)


class RealtimeLoop:

    def __init__(self, dt: float = 0.001, report: bool = False,
                 fade: float = 0.0, fifo_priority: int = 0) -> None:
        self.dt = dt
        self.report = report
        self._fade_duration = fade

        self._stopped = False
        self._fading  = False
        self._fade_t0 = None

        # Welford online stats (numerically stable, no overflow risk)
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0

        self._t0    = None
        self._t1    = None
        self._ttarg = None

        # Linux FIFO real-time scheduling
        if fifo_priority > 0 and not _WINDOWS:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO,
                                      os.sched_param(fifo_priority))
            except PermissionError:
                print("[RealtimeLoop] Warning: FIFO scheduling requires root — skipping.")
            except AttributeError:
                pass   # os.sched_setscheduler not available on this platform

        # Signal handlers can only be installed from the main thread. When the
        # loop runs in a worker thread, skip them and rely on loop.stop() being
        # called from outside (Ctrl+C handling stays in the main thread).
        self._prev_sigint  = None
        self._prev_sigterm = None
        self._prev_sighup  = None
        if threading.current_thread() is threading.main_thread():
            self._prev_sigint  = signal.signal(signal.SIGINT,  self._on_signal)
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)
            if not _WINDOWS:
                self._prev_sighup = signal.signal(signal.SIGHUP, self._on_signal)

    # Public API

    @property
    def fade(self) -> float:
        if not self._fading:
            return 1.0
        elapsed = time.perf_counter() - self._fade_t0
        return max(0.0, 1.0 - elapsed / self._fade_duration)

    def stop(self) -> None:
        self._stopped = True

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    def close(self) -> None:
        self._restore_signals()

    # Iterator protocol

    def __iter__(self) -> 'RealtimeLoop':
        self._t0 = self._t1 = time.perf_counter()
        self._ttarg   = None
        self._stopped = False
        self._fading  = False
        self._fade_t0 = None
        self._n = 0; self._mean = 0.0; self._M2 = 0.0
        return self

    def __next__(self) -> float:
        if self._stopped:
            self._finalize()
            raise StopIteration

        if self._fading and self.fade == 0.0:
            self._stopped = True
            self._finalize()
            raise StopIteration

        deadline = self._t1 + self.dt

        # Sleep most of the interval, then busy-spin the last _BUSY_WAIT_S seconds
        remaining = deadline - _BUSY_WAIT_S - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        while time.perf_counter() < deadline:
            if self._stopped:
                break

        now = time.perf_counter()
        self._t1 = deadline   # drift-free advance

        if self._ttarg is None:
            self._ttarg = now + self.dt
        else:
            # Welford's algorithm: running mean and variance
            err = now - self._ttarg
            self._n += 1
            delta = err - self._mean
            self._mean += delta / self._n
            self._M2   += delta * (err - self._mean)
            self._ttarg += self.dt

        return now - self._t0

    # Internal

    def _on_signal(self, signum, frame) -> None:
        if self._fade_duration > 0.0 and not self._fading:
            self._fading  = True
            self._fade_t0 = time.perf_counter()
        else:
            self._stopped = True   # second signal or no fade → hard stop

    def _finalize(self) -> None:
        self._print_report()
        self._restore_signals()

    def _restore_signals(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
            self._prev_sigint = None
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
            self._prev_sigterm = None
        if not _WINDOWS and self._prev_sighup is not None:
            signal.signal(signal.SIGHUP, self._prev_sighup)
            self._prev_sighup = None

    def _print_report(self) -> None:
        if not self.report or self._n < 2:
            return
        stddev = sqrt(self._M2 / (self._n - 1)) if self._n > 1 else 0.0
        print(
            f"\n[RealtimeLoop] {self._n} cycles @ {1/self.dt:.1f} Hz\n"
            f"  mean timing error : {self._mean * 1e3:+.3f} ms\n"
            f"  std dev           : {stddev * 1e3:.3f} ms"
        )
