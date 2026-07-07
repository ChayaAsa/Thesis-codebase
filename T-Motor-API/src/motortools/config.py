from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .loop import RealtimeLoop


@dataclass
class MotorConfig:

    interface: str        = 'seeedstudio'
    channel:   str        = 'COM4'
    bitrate:   int        = 1_000_000
    motors:    list[dict] = field(default_factory=list)
    dt:        float      = 0.05
    fade:      float      = 0.5
    report:    bool       = True

    # Persistence

    @classmethod
    def load(cls, path: str | Path) -> 'MotorConfig':
        raw   = yaml.safe_load(Path(path).read_text())
        bus_d = raw.get('bus',  {})
        lp_d  = raw.get('loop', {})
        return cls(
            interface = bus_d.get('interface', 'seeedstudio'),
            channel   = bus_d.get('channel',   'COM4'),
            bitrate   = int(bus_d.get('bitrate', 1_000_000)),
            motors    = raw.get('motors', []),
            dt        = float(lp_d.get('dt',     0.05)),
            fade      = float(lp_d.get('fade',   0.5)),
            report    = bool(lp_d.get('report',  True)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'bus': {
                'interface': self.interface,
                'channel':   self.channel,
                'bitrate':   self.bitrate,
            },
            'motors': self.motors,
            'loop': {
                'dt':     self.dt,
                'fade':   self.fade,
                'report': self.report,
            },
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        print(f"[MotorConfig] saved → {path}")

    # Object construction

    def open_bus(self):
        import can
        return can.interface.Bus(
            interface=self.interface,
            channel=self.channel,
            bitrate=self.bitrate,
        )

    def open_motors(self, bus) -> list:
        from tmotorcan.mit_mode import MITMotor
        result = []
        for entry in self.motors:
            mt = entry.get('max_temp')
            motor = MITMotor(
                bus,
                motor_id = int(entry['id']),
                model    = entry.get('model', 'AK45-10'),
                # Pass None when not specified so MITMotor falls back to the
                # model YAML's max_temp instead of a hardcoded constant.
                max_temp = int(mt) if mt is not None else None,
            )
            motor.cmd.kp = float(entry.get('kp', 0.0))
            motor.cmd.kd = float(entry.get('kd', 0.0))
            for fname, lo_hi in entry.get('limits', {}).items():
                motor.set_limits(fname, tuple(lo_hi))
            result.append(motor)
        return result

    def make_loop(self) -> RealtimeLoop:
        return RealtimeLoop(dt=self.dt, report=self.report, fade=self.fade)
