from __future__ import annotations

import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS: tuple[str, ...] = (
    'p_min', 'p_max',
    'v_min', 'v_max',
    't_min', 't_max',
    'kp_min', 'kp_max',
    'kd_min', 'kd_max',
    'gear_ratio',
    'kt',
    'pole_pairs',
    'max_temp',
)

_MIN_MAX_PAIRS: tuple[tuple[str, str, str], ...] = (
    ('p_min',  'p_max',  'position'),
    ('v_min',  'v_max',  'velocity'),
    ('t_min',  't_max',  'torque'),
    ('kp_min', 'kp_max', 'kp'),
    ('kd_min', 'kd_max', 'kd'),
)

_POSITIVE_FIELDS: tuple[str, ...] = ('gear_ratio', 'kt', 'pole_pairs', 'max_temp')


def _models_dir() -> Path:
    resource = files('tmotorcan.models')
    # importlib.resources returns a Traversable; for a regular package on disk
    # this is a Path. ``str()`` round-trip handles the editable-install case.
    return Path(str(resource))


def list_models() -> list[str]:
    return sorted(p.stem for p in _models_dir().glob('*.yaml'))


def model_path(name: str) -> Path:
    safe = name.lower().replace(' ', '-')
    if not safe:
        raise ValueError("model name cannot be empty")
    return _models_dir() / f'{safe}.yaml'


def read_model_text(name: str) -> str:
    return model_path(name).read_text(encoding='utf-8')


def write_model_text(name: str, text: str, *, backup: bool = True) -> Path:
    path = model_path(name)
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix('.yaml.bak'))
    path.write_text(text, encoding='utf-8')
    return path


def delete_model_file(name: str, *, backup: bool = True) -> Path:
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(path)
    if backup:
        shutil.copy2(path, path.with_suffix('.yaml.bak'))
    path.unlink()
    return path


def parse_model_values(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML must be a mapping, got {type(data).__name__}")
    return data


# Match: optional leading whitespace, key, colon, whitespace, value (anything
# until end-of-line or the start of an inline `#` comment), then preserve the
# rest of the line verbatim. The value capture is non-greedy on the trailing
# whitespace so we don't eat the run of spaces before an inline comment.
_VALUE_LINE_RE_TEMPLATE = (
    r'(?P<lead>^[ \t]*){key}'
    r'(?P<colon>[ \t]*:[ \t]*)'
    r'(?P<value>[^# \n]*?)'
    r'(?P<tail>[ \t]*(?:# .*)?)$'
)


def patch_yaml_value(text: str, key: str, value: Any) -> str:
    formatted = _format_scalar(value)
    pattern = re.compile(
        _VALUE_LINE_RE_TEMPLATE.format(key=re.escape(key)),
        flags=re.MULTILINE,
    )
    new_text, n = pattern.subn(
        lambda m: f"{m.group('lead')}{key}{m.group('colon')}{formatted}{m.group('tail')}",
        text,
        count=1,
    )
    if n == 0:
        sep = '' if text.endswith('\n') or text == '' else '\n'
        new_text = f"{text}{sep}{key}: {formatted}\n"
    return new_text


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # %g drops trailing zeros; force a decimal point so the value still
        # round-trips as a float (e.g. "1" → "1.0") to match the existing
        # YAML files' style.
        s = f'{value:.6g}'
        if '.' not in s and 'e' not in s and 'E' not in s and 'inf' not in s and 'nan' not in s:
            s += '.0'
        return s
    if value is None:
        return 'null'
    s = str(value)
    if any(c in s for c in '# :\n') or s.strip() != s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def validate_model_text(text: str) -> list[str]:
    issues: list[str] = []
    try:
        data = parse_model_values(text)
    except (yaml.YAMLError, ValueError) as e:
        return [f"YAML parse error: {e}"]

    missing = [k for k in REQUIRED_FIELDS if k not in data]
    for k in missing:
        issues.append(f"missing required field: {k}")

    for k in REQUIRED_FIELDS:
        if k not in data:
            continue
        v = data[k]
        if k in ('pole_pairs', 'max_temp'):
            if not isinstance(v, int) or isinstance(v, bool):
                issues.append(f"{k} must be an integer, got {type(v).__name__} ({v!r})")
        else:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                issues.append(f"{k} must be a number, got {type(v).__name__} ({v!r})")

    for lo_key, hi_key, label in _MIN_MAX_PAIRS:
        lo, hi = data.get(lo_key), data.get(hi_key)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if lo == hi:
                issues.append(
                    f"{label}: {lo_key} == {hi_key} ({lo}) — must be strictly less; "
                    f"equal would divide by zero in protocol._encode"
                )
            elif lo > hi:
                issues.append(f"{label}: {lo_key} ({lo}) must be < {hi_key} ({hi})")

    for k in _POSITIVE_FIELDS:
        v = data.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not v > 0:
            issues.append(f"{k} must be > 0, got {v}")

    extras = sorted(k for k in data if k not in REQUIRED_FIELDS)
    for k in extras:
        issues.append(f"unknown field (will be ignored by load_params): {k}")

    return issues


class EditableModel:

    __slots__ = ('_name', '_text')

    _INT_FIELDS = frozenset({'pole_pairs', 'max_temp'})

    def __init__(self, name: str, text: str) -> None:
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_text', text)

    @classmethod
    def load(cls, name: str) -> 'EditableModel':
        return cls(name, read_model_text(name))

    @classmethod
    def new(cls, name: str) -> 'EditableModel':
        return cls(name, template_text(name))

    @property
    def name(self) -> str:
        return self._name

    @property
    def text(self) -> str:
        return self._text

    def replace_text(self, value: str) -> None:
        object.__setattr__(self, '_text', value)

    def values(self) -> dict[str, Any]:
        try:
            return parse_model_values(self._text)
        except (yaml.YAMLError, ValueError):
            return {}

    def validate(self) -> list[str]:
        return validate_model_text(self._text)

    def save(self, *, backup: bool = True, force: bool = False) -> Path:
        issues = self.validate()
        if issues and not force:
            raise ValueError(
                f"validation failed for {self._name!r}: " + '; '.join(issues)
            )
        return write_model_text(self._name, self._text, backup=backup)

    def __getattr__(self, key: str) -> Any:
        # Only fires when normal lookup fails — won't shadow _name / _text / methods.
        if key in REQUIRED_FIELDS:
            return self.values().get(key)
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in REQUIRED_FIELDS:
            self._set_field(key, value)
            return
        if key.startswith('_') or key in {'text', 'name'}:
            # ``text`` is a read-only property; force users through replace_text().
            if key == 'text':
                raise AttributeError("use .replace_text(...) to set text")
            object.__setattr__(self, key, value)
            return
        raise AttributeError(f"unknown field: {key!r}")

    def _set_field(self, key: str, value: Any) -> None:
        if key in self._INT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{key} must be an int, got {type(value).__name__}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be a number, got {type(value).__name__}")
        object.__setattr__(self, '_text', patch_yaml_value(self._text, key, value))

    def __repr__(self) -> str:
        d = self.values()
        present = ', '.join(f'{k}={d[k]!r}' for k in REQUIRED_FIELDS if k in d)
        return f"EditableModel({self._name!r}, {present})"


def template_text(name: str) -> str:
    display_name = name.upper() if name else '<NEW MODEL>'
    return f"""\
# {display_name} — MIT CAN mode parameter ranges
# Source: <add datasheet + firmware manual reference here>
#
# These ranges must match the firmware's encoding window. Values outside the
# wire range are clamped by protocol._encode() before transmission.

# CAN frame encoding ranges (must match firmware)
p_min: -12.5    # rad (16-bit resolution)
p_max:  12.5
v_min: -20.0    # rad/s (12-bit resolution)
v_max:  20.0
t_min:  -8.0    # N·m (12-bit resolution)
t_max:   8.0
kp_min:  0.0    # N·m/rad (12-bit resolution)
kp_max: 500.0
kd_min:  0.0    # N·m·s/rad (12-bit resolution)
kd_max:  5.0

# Motor physical constants
gear_ratio:  10.0  # output:rotor reduction (e.g. 10.0 for a 10:1 gearbox)
kt:           0.10 # N·m/A motor-side torque constant (datasheet)
                   # output-shaft effective Kt = kt × gear_ratio
pole_pairs:  14    # electrical pole pairs (poles / 2)

# Safety defaults
max_temp: 80       # °C
"""
