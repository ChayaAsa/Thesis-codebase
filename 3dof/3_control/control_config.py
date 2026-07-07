from __future__ import annotations

import os
import sys

from easy_path import WS_ROOT
_TOOLS = os.path.normpath(os.path.join(WS_ROOT, 'tools'))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import numpy as np

# Model selection
# Set USE_SYSID = True to switch every control script to the identified model.
USE_SYSID = False

DYNAMIC_DIR = os.path.join(WS_ROOT, '3dof', '1_dynamic', 'python')
SYSID_DIR   = os.path.join(WS_ROOT, '3dof', '2_system identification')

if USE_SYSID:
    if SYSID_DIR not in sys.path:
        sys.path.insert(0, SYSID_DIR)
    from sysid_dynamic import SysIDDynamic as Dynamic
    DYN_CACHE = os.path.join(SYSID_DIR, 'sid_cache.pkl')
else:
    if DYNAMIC_DIR not in sys.path:
        sys.path.insert(0, DYNAMIC_DIR)
    from dynamic import Dynamic
    DYN_CACHE = os.path.join(DYNAMIC_DIR, 'dynamic_cache.pkl')

# Hardware
PORT       = 'COM18'           # slcan channel (matches your position-hold test)
BITRATE    = 1_000_000         # 1 Mbit/s
MOTOR_IDS  = [1, 2, 3]         # joint 1, 2, 3 (CAN arbitration IDs)
MODEL      = 'AK45-10'         # CubeMars model YAML used by tmotorcan
MAX_TEMP_C = 80                # hard over-temperature fault threshold
INVERT = {1: True, 2: True, 3: False}
SIGN   = {id: (-1.0 if INVERT[id] else 1.0) for id in MOTOR_IDS}

# Loop timing
SAMPLE_HZ = 100.0              # control + logging rate. 100 Hz is a good balance
DT        = 1.0 / SAMPLE_HZ    # for slcan-over-USB with 3 motors. Drop to 50 Hz


JOINT_LIMITS = {
    1: (-0.524, 0.524),   # base yaw [rad] ~30
    2: (-0.175, 1.396),   # shoulder [rad] ~-10-80
    3: (-1.571, 1.571),   # elbow [rad] ~90
}

CURRENT_LIMIT_A = 6.0                       # per-motor phase-current cap [A]

DEFAULT_KP = {1: 10.0, 2: 25.0, 3: 15.0}   # N·m/rad
DEFAULT_KD = {1: 1.0,  2: 1.8,  3: 1.5}   # N·m·s/rad

LOCK_KP = {1: 10.0, 2: 35.0, 3: 15.0}   # N·m/rad
LOCK_KD = {1: 1.0,  2: 1.8,  3: 1.5}   # N·m·s/rad

FREE_KD = {1: 0.5,  2: 0.8,  3: 0.5}   # N·m·s/rad

EEF_ATI_FRAME = np.array([[0., 0., 1., 0.],
                          [1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]], dtype=float)

from helpers import (make_bus     as _make_bus,
                     build_motors as _build_motors,
                     cmd_free     as _cmd_free,
                     cmd_lock     as _cmd_lock,
                     motor_limits as _motor_limits)


def make_bus():
    return _make_bus(PORT, BITRATE)

def build_motors(bus):
    return _build_motors(bus, MOTOR_IDS, MODEL, MAX_TEMP_C)

def cmd_free(motors: list) -> None:
    return _cmd_free(motors, FREE_KD)

def cmd_lock(motors: list) -> None:
    return _cmd_lock(motors, LOCK_KP, LOCK_KD)

def motor_limits(id: int, lo: float, hi: float) -> tuple[float, float]:
    return _motor_limits(SIGN, id, lo, hi)
