from math import pi

_RAD_S_TO_RPM = 60.0 / (2 * pi)   # ≈ 9.5493
_RPM_TO_RAD_S = (2 * pi) / 60.0
_RAD_S_TO_DPS = 180.0 / pi        # ≈ 57.2958
_DPS_TO_RAD_S = pi / 180.0


# ── Torque ↔ current ─────────────────────────────────────────────────────────

def torque_to_current(torque_Nm: float,
                      kt: float | None = None,
                      gear_ratio: float = 1.0,
                      *,
                      kt_out: float | None = None) -> float:
    if kt_out is not None:
        return torque_Nm / kt_out
    if kt is None:
        raise ValueError("Provide kt_out= or kt=")
    return torque_Nm / (kt * gear_ratio)


def current_to_torque(current_A: float,
                      kt: float | None = None,
                      gear_ratio: float = 1.0,
                      *,
                      kt_out: float | None = None) -> float:
    if kt_out is not None:
        return current_A * kt_out
    if kt is None:
        raise ValueError("Provide kt_out= or kt=")
    return current_A * kt * gear_ratio


# ── rad/s ↔ MRPM ─────────────────────────────────────────────────────────────

def rads_to_rpm(rads: float) -> float:
    return rads * _RAD_S_TO_RPM


def rpm_to_rads(rpm: float) -> float:
    return rpm * _RPM_TO_RAD_S


# ── rad/s ↔ DPS ──────────────────────────────────────────────────────────────

def rads_to_dps(rads: float) -> float:
    return rads * _RAD_S_TO_DPS


def dps_to_rads(dps: float) -> float:
    return dps * _DPS_TO_RAD_S


# ── MRPM ↔ DPS ───────────────────────────────────────────────────────────────

def rpm_to_dps(rpm: float) -> float:
    return rpm * 6.0   # 360° / 60 s


def dps_to_rpm(dps: float) -> float:
    return dps / 6.0


# ERPM conversions (require pole_pairs)

def rads_to_erpm(rads: float, pole_pairs: int) -> float:
    return rads * _RAD_S_TO_RPM * pole_pairs


def erpm_to_rads(erpm: float, pole_pairs: int) -> float:
    return erpm / pole_pairs * _RPM_TO_RAD_S


def rpm_to_erpm(rpm: float, pole_pairs: int) -> float:
    return rpm * pole_pairs


def erpm_to_rpm(erpm: float, pole_pairs: int) -> float:
    return erpm / pole_pairs
