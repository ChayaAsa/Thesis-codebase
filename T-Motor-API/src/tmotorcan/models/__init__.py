from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MotorParams:
    p_min: float;  p_max: float
    v_min: float;  v_max: float
    t_min: float;  t_max: float
    kp_min: float; kp_max: float
    kd_min: float; kd_max: float
    gear_ratio:  float
    kt:          float
    pole_pairs:  int
    max_temp:    int


def load_params(model: str) -> MotorParams:
    if model.lower().endswith(('.yaml', '.yml')):
        text = Path(model).read_text(encoding='utf-8')
    else:
        filename = model.lower().replace(' ', '-') + '.yaml'
        resource = files('tmotorcan.models').joinpath(filename)
        text = resource.read_text(encoding='utf-8')

    data = yaml.safe_load(text)
    try:
        params = MotorParams(**data)
    except TypeError as e:
        raise KeyError(f"Motor model '{model}' YAML is missing a required field: {e}") from e

    _validate(params, model)
    return params


def _validate(p: MotorParams, model: str) -> None:
    def _check(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"Motor model '{model}': {msg}")

    for lo, hi, name in (
        (p.p_min,  p.p_max,  'position'),
        (p.v_min,  p.v_max,  'velocity'),
        (p.t_min,  p.t_max,  'torque'),
        (p.kp_min, p.kp_max, 'kp'),
        (p.kd_min, p.kd_max, 'kd'),
    ):
        _check(lo < hi, f"{name}: min ({lo}) must be < max ({hi})")

    _check(p.gear_ratio > 0, f"gear_ratio must be > 0, got {p.gear_ratio}")
    _check(p.kt         > 0, f"kt must be > 0, got {p.kt}")
    _check(p.pole_pairs > 0, f"pole_pairs must be > 0, got {p.pole_pairs}")
    _check(p.max_temp   > 0, f"max_temp must be > 0, got {p.max_temp}")
