from __future__ import annotations

import math
import sys as _sys
import time

import numpy as np

try:
    import msvcrt                       # Windows-only; used for non-blocking Enter
except ImportError:
    msvcrt = None

# Make stdout UTF-8 so the unicode in status prints (·, ², ⚠, arrows) never raises
# UnicodeEncodeError on a legacy cp1252 Windows console. Imported by every script.
try:
    _sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Hardware
PORT       = 'COM18'           # slcan channel (matches your position-hold test)
BITRATE    = 1_000_000         # 1 Mbit/s
MOTOR_IDS  = [1, 2, 3]         # joint 1, 2, 3 (CAN arbitration IDs)
MODEL      = 'AK45-10'         # CubeMars model YAML used by tmotorcan
MAX_TEMP_C = 80                # hard over-temperature fault threshold

# Motor spin direction vs our kinematic ("custom frame", rrr_params.m) convention.
# Some joints' encoders count the OPPOSITE way: commanding +1 rad drives them to
# and identify_sysid.py / the regressor need no change.
INVERT = {1: True, 2: True, 3: False}
SIGN   = {mid: (-1.0 if INVERT[mid] else 1.0) for mid in MOTOR_IDS}

# Loop timing
SAMPLE_HZ = 100.0              # control + logging rate. 100 Hz is a good balance
DT        = 1.0 / SAMPLE_HZ    # for slcan-over-USB with 3 motors. Drop to 50 Hz
#                                  (SAMPLE_HZ = 50) if the loop's timing report
#                                  shows large overruns.

#  ⚠  PER-JOINT SAFETY LIMITS  —  EDIT THESE FOR YOUR ARM  ⚠
#  Travel allowed AROUND THE ZEROED START POSE, in radians. The arm is zeroed at
#  whatever pose you hand-hold it in at the start, so (0,0,0) = that pose. These
#  bounds are enforced two ways:
#     • the excitation trajectory is amplitude-scaled to stay inside them, and
#     • a hard clamp / abort triggers if a joint ever leaves the band.
#  Set them CONSERVATIVELY the first time, watch one run, then widen.
#  Values below are PLACEHOLDERS — verify against your real workspace / hard stops
#  / self-collisions before trusting them.
JOINT_LIMITS = {
    1: (-0.524, 0.524),   # base yaw [rad] ~30
    2: (-0.175, 1.396),   # shoulder [rad] ~-10-80
    3: (-1.571, 1.571),   # elbow [rad] ~90
}

#  Excitation signal  (band-limited sum-of-sines, a.k.a. finite Fourier series)
DURATION_S   = 60.0            # length of the excitation phase (excl. ramps)
RAMP_S       = 2.0             # raised-cosine ramp in/out (starts & ends at rest)
BASE_FREQ_HZ = 0.10            # fundamental frequency f0 (period = 10 s)
N_HARMONICS  = 6               # harmonics k*f0 for k = 1..N -> 0.1 .. 0.6 Hz
SEED         = 12345           # RNG seed for the random phases (reproducible runs)

# The impedance run oscillates around the CENTRE of each joint's safe band — not
# necessarily the zeroed pose — so an off-centre zero (e.g. a shoulder whose band
# is -10..80 deg) does NOT shrink the motion. POS_CENTER is that midpoint (in the
# half-span. The trajectory is windowed, so it still starts/ends at the zero pose.
POS_CENTER = {
    j: 0.5 * (lo + hi)
    for j, (lo, hi) in JOINT_LIMITS.items()
}
POS_AMP = {
    j: 0.80 * 0.5 * (hi - lo)
    for j, (lo, hi) in JOINT_LIMITS.items()
}

# Tracking impedance for the position-reference run. Stiff enough to follow the
# reference, soft enough to stay gentle. The motor still REPORTS the true applied
# torque, so identification is unaffected by how stiff this is.
TRACK_KP = {1: 7.0, 2: 35.0, 3: 8.0}   # N·m/rad
TRACK_KD = {1: 1.0,   2: 1.5,   3: 2.0}    # N·m·s/rad

# Peak torque per joint for the OPEN-LOOP torque run [N·m]. Keep modest: the arm
# moves freely under these. AK45-10 peak is ~7 N·m; start small.
TORQUE_AMP = {1: 1.0, 2: 2.0, 3: 1.0}

# Open-loop containment: velocity damping (kp stays 0 after ramp-in) tames runaway
# without injecting position information. Set to 0.0 for a "pure" open-loop torque
CONTAIN_KD     = {1: 1.0, 2: 2.0, 3: 1.5}  # N·m·s/rad (raised for stability)
CURRENT_LIMIT_A = 6.0                       # per-motor phase-current cap [A]

# End-of-run recovery for the OPEN-LOOP torque script: after excitation the arm
# and then ramp the target back to 0 in small steps (see recover_to_zero). Torque
# is still capped by CURRENT_LIMIT_A, so a high kp saturates rather than snaps.
# Tune for your arm; start a touch above TRACK_KP/KD.
HOLD_KP = {1: 10.0, 2: 40.0, 3: 12.0}   # N·m/rad
HOLD_KD = {1: 1.5,  2: 2.5,  3: 2.5}    # N·m·s/rad
RECOVER_SETTLE_S = 1.0    # s to hold the latched pose before ramping down
RECOVER_STEP     = 0.10   # rad the target moves toward 0 each step
RECOVER_STEP_S   = 0.50   # s held per step

#  Arm geometry (mirror of rrr_params.m) — used ONLY by identify_sysid.py
L1 = 0.120      # base column height / shoulder offset [m]
L2 = 0.150     # upper-arm length [m]
L3 = 0.120     # forearm length [m]
G  = 9.81      # gravity magnitude [m/s^2]

# CSV column layout written by both generators (identify_sysid.py reads these).
LOG_COLUMNS = [
    'time_s',
    'q1', 'qd1', 'tau1',
    'q2', 'qd2', 'tau2',
    'q3', 'qd3', 'tau3',
    'ref1', 'ref2', 'ref3',   # position ref [rad] OR commanded torque [N·m]
    'temp1', 'temp2', 'temp3',
]


# ═════════════════════════════════════════════════════════════════════════════
# Excitation generator
# ═════════════════════════════════════════════════════════════════════════════
class MultiSine:

    def __init__(self, amps: list[float], base_freq_hz: float, n_harmonics: int,
                 duration_s: float, ramp_s: float, seed: int = 0,
                 center: list[float] | None = None) -> None:
        self.amps      = list(amps)
        self.n_ch      = len(self.amps)
        self.f0        = base_freq_hz
        self.N         = n_harmonics
        self.T         = duration_s
        self.ramp      = ramp_s
        self.w0        = 2.0 * math.pi * base_freq_hz       # fundamental ω [rad/s]
        self._k        = np.arange(1, n_harmonics + 1)      # harmonic indices
        # Per-channel DC offset the oscillation rides on (e.g. the CENTRE of a
        # joint's safe band when the zeroed pose is off-centre). It is WINDOWED
        # too, so the signal still starts and ends at zero -> no jolt, and the arm
        # eases from the held zero pose out to the band centre and back.
        self._center   = (np.zeros(self.n_ch) if center is None
                          else np.asarray(center, dtype=float))

        rng = np.random.default_rng(seed)
        # phases[ch, k]
        self._phases = rng.uniform(0.0, 2.0 * math.pi, size=(self.n_ch, self.N))
        # raw harmonic amplitudes ∝ 1/k (more weight to low frequencies -> bounded ω)
        self._a_k = 1.0 / self._k

        # Auto-scale each channel so its windowed peak == amps[ch].
        self._scale = np.ones(self.n_ch)
        t_grid = np.linspace(0.0, self.T, int(self.T * 1000) + 1)   # 1 kHz grid
        w_grid = np.array([self._window(t) for t in t_grid])
        for ch in range(self.n_ch):
            raw = self._ac(ch, t_grid)            # un-windowed, un-scaled
            peak = np.max(np.abs(w_grid * raw))
            self._scale[ch] = (self.amps[ch] / peak) if peak > 1e-12 else 0.0

    # windows
    def _window(self, t: float) -> float:
        if t <= 0.0 or t >= self.T:
            return 0.0
        if t < self.ramp:
            return 0.5 * (1.0 - math.cos(math.pi * t / self.ramp))
        if t > self.T - self.ramp:
            return 0.5 * (1.0 - math.cos(math.pi * (self.T - t) / self.ramp))
        return 1.0

    def _dwindow(self, t: float) -> float:
        if t <= 0.0 or t >= self.T:
            return 0.0
        if t < self.ramp:
            return 0.5 * math.pi / self.ramp * math.sin(math.pi * t / self.ramp)
        if t > self.T - self.ramp:
            return -0.5 * math.pi / self.ramp * math.sin(math.pi * (self.T - t) / self.ramp)
        return 0.0

    # ── unscaled AC part and its derivative (vectorised over t) ───────────────
    def _ac(self, ch: int, t):
        t = np.asarray(t, dtype=float)
        phase = self.w0 * np.outer(self._k, t) + self._phases[ch][:, None]  # (N, len(t))
        return (self._a_k[:, None] * np.sin(phase)).sum(axis=0)

    def _dac(self, ch: int, t):
        t = np.asarray(t, dtype=float)
        phase = self.w0 * np.outer(self._k, t) + self._phases[ch][:, None]
        coeff = (self._a_k * self._k * self.w0)[:, None]    # a_k * (k ω0)
        return (coeff * np.cos(phase)).sum(axis=0)

    # ── public, scalar-time API used inside the control loop ──────────────────
    @staticmethod
    def _scalar(arr) -> float:
        return float(np.ravel(arr)[0])

    def value(self, t: float) -> list[float]:
        w = self._window(t)
        return [w * (self._center[ch] + self._scale[ch] * self._scalar(self._ac(ch, t)))
                for ch in range(self.n_ch)]

    def deriv(self, t: float) -> list[float]:
        w, dw = self._window(t), self._dwindow(t)
        out = []
        for ch in range(self.n_ch):
            ac  = self._scalar(self._ac(ch, t))
            dac = self._scalar(self._dac(ch, t))
            # d/dt [ w*(centre + scale*AC) ] = dw*(centre + scale*AC) + w*scale*AC'
            out.append(dw * self._center[ch]
                       + self._scale[ch] * (dw * ac + w * dac))
        return out

    @property
    def total_time(self) -> float:
        return self.T


# ═════════════════════════════════════════════════════════════════════════════
# Hardware helpers
# ═════════════════════════════════════════════════════════════════════════════
def make_bus():
    import can
    from tmotorcan import MotorBus
    raw = can.interface.Bus(
        interface='slcan',
        channel=PORT,
        bitrate=BITRATE,
        frame_type='STD',
    )
    return MotorBus(raw)


def build_motors(bus):
    from tmotorcan import MITMotor
    return [MITMotor(bus, motor_id=mid, model=MODEL, max_temp=MAX_TEMP_C)
            for mid in MOTOR_IDS]


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def motor_limits(mid: int, lo: float, hi: float) -> tuple[float, float]:
    a, b = SIGN[mid] * lo, SIGN[mid] * hi
    return (a, b) if a <= b else (b, a)


class KeyboardLine:

    def __init__(self):
        self._buf = []

    def poll(self):
        if msvcrt is None:
            return None
        line = None
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):        # arrow / function key: swallow 2nd code
                msvcrt.getwch()
                continue
            if ch in ('\r', '\n'):            # Enter -> line complete
                line = ''.join(self._buf)
                self._buf.clear()
                print()
            elif ch == '\x08':                # Backspace
                if self._buf:
                    self._buf.pop()
            elif ch == '\x03':                # Ctrl+C
                raise KeyboardInterrupt
            else:
                self._buf.append(ch)
        return line


def recover_to_zero(motors, kp, kd, dt,
                    settle_s=RECOVER_SETTLE_S, step=RECOVER_STEP, step_s=RECOVER_STEP_S):
    from tmotorcan import update_all
    target = {m.id: float(m.state.position) for m in motors}   # latch current pose

    def push():
        for m in motors:
            m.cmd.position = target[m.id]
            m.cmd.velocity = 0.0
            m.cmd.torque   = 0.0
            m.cmd.kp       = kp[m.id]
            m.cmd.kd       = kd[m.id]
        update_all(motors)
        time.sleep(dt)

    def hold_for(seconds):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            push()

    hold_for(settle_s)                                       # 1) firm hold, settle
    while any(abs(v) > 1e-9 for v in target.values()):       # 2) stepped ramp to 0
        for mid in target:
            v = target[mid]
            target[mid] = 0.0 if abs(v) <= step else v - math.copysign(step, v)
        hold_for(step_s)


def wait_for_enter_holding(motors, kp, kd, dt):
    from tmotorcan import update_all
    kb = KeyboardLine()
    while kb.poll() is None:
        for m in motors:
            m.cmd.position = 0.0
            m.cmd.velocity = 0.0
            m.cmd.torque   = 0.0
            m.cmd.kp       = kp[m.id]
            m.cmd.kd       = kd[m.id]
        update_all(motors)
        time.sleep(dt)
