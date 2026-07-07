# tmotorcan — API Reference

Reference for everything exported from the top-level `tmotorcan` package, plus
the standalone `motortools` utilities.

### `tmotorcan` core — CAN protocol & motor driver

| File | Contents |
|---|---|
| [tmotorcan/motor-bus.md](tmotorcan/motor-bus.md) | `MotorBus` — CAN bus wrapper with per-motor frame routing |
| [tmotorcan/mit-motor.md](tmotorcan/mit-motor.md) | `MITMotor` — single-motor MIT-mode driver |
| [tmotorcan/dataclasses.md](tmotorcan/dataclasses.md) | `MotorState`, `MotorSetpoint`, `MotorParams` (field tables) |
| [tmotorcan/models.md](tmotorcan/models.md) | `load_params` — load motor model YAML files |
| [tmotorcan/exceptions.md](tmotorcan/exceptions.md) | `MotorFaultError`, `MotorTimeoutError` |

### `motortools` — utilities (re-exported via `tmotorcan` when available)

| File | Contents |
|---|---|
| [motortools/loop.md](motortools/loop.md) | `RealtimeLoop` — fixed-rate loop with Ctrl+C fade |
| [motortools/filters.md](motortools/filters.md) | `IIRFilter`, `FiniteDiff`, `PositionUnwrapper` |
| [motortools/limits.md](motortools/limits.md) | `SoftLimiter`, `ThermalDerate` |
| [motortools/logger.md](motortools/logger.md) | `DataLogger` — CSV telemetry logger |
| [motortools/group.md](motortools/group.md) | `MotorGroup`, `update_all` — multi-motor coordination |
| [motortools/config.md](motortools/config.md) | `MotorConfig` — YAML rig config |
| [motortools/trajectory.md](motortools/trajectory.md) | `TrapezoidalProfile` — constant-accel motion profile |
| [motortools/units.md](motortools/units.md) | torque/current, rad↔rpm/dps/erpm conversions |
| [motortools/can_tools.md](motortools/can_tools.md) | `inspect`, `sniff`, `peek` — CAN frame debugging |
| [motortools/models_io.md](motortools/models_io.md) | `list_models`, `EditableModel`, model YAML helpers |
| [motortools/virtual.md](motortools/virtual.md) | `VirtualMotor`, `serve_in_thread`, `SimHandle` — sim with no hardware |

### Install tiers

```bash
pip install tmotorcan[mini]    # core only — MotorBus, MITMotor, protocol
pip install tmotorcan[full]    # + all motortools utilities
```

### Running examples without hardware

All examples in this reference work on the in-process virtual CAN bus, so they
run without an adapter. Replace the `interface='virtual'` line with your real
adapter (e.g. `interface='seeedstudio', channel='COM4'`) on the rig.

```python
from motortools import serve_in_thread
sim = serve_in_thread(motor_id=1, channel='virt', model='AK45-10')

import can
from tmotorcan import MotorBus, MITMotor

raw = can.interface.Bus(interface='virtual', channel='virt', bitrate=1_000_000)
with MotorBus(raw) as bus:
    with MITMotor(bus, motor_id=1, model='AK45-10') as m:
        m.update()
        print(m.state.position)

sim.set()
```
