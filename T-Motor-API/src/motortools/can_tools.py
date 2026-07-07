import time


# ── MIT CAN bit-decode helpers (self-contained, no tmotorcan import) ─────────

def _mit_decode(raw: int, val_min: float, val_max: float, bits: int) -> float:
    return val_min + (val_max - val_min) * raw / ((1 << bits) - 1)


def _mit_decode_response(data: bytes, params) -> dict:
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    t_int = ((data[4] & 0xF) << 8) | data[5]
    return {
        'motor_id': data[0],
        'position': _mit_decode(p_int, params.p_min, params.p_max, 16),
        'velocity': _mit_decode(v_int, params.v_min, params.v_max, 12),
        'torque':   _mit_decode(t_int, params.t_min, params.t_max, 12),
        'temp_C':   int(data[6]) if len(data) > 6 else None,
        'error':    int(data[7]) if len(data) > 7 else None,
    }


def _mit_decode_command(data: bytes, params) -> dict:
    p_int  = (data[0] << 8) | data[1]
    v_int  = (data[2] << 4) | (data[3] >> 4)
    kp_int = ((data[3] & 0xF) << 8) | data[4]
    kd_int = (data[5] << 4) | (data[6] >> 4)
    t_int  = ((data[6] & 0xF) << 8) | data[7]
    return {
        'position': _mit_decode(p_int,  params.p_min,  params.p_max,  16),
        'velocity': _mit_decode(v_int,  params.v_min,  params.v_max,  12),
        'kp':       _mit_decode(kp_int, params.kp_min, params.kp_max, 12),
        'kd':       _mit_decode(kd_int, params.kd_min, params.kd_max, 12),
        'torque':   _mit_decode(t_int,  params.t_min,  params.t_max,  12),
    }


def _is_mit_special(data: bytes) -> str | None:
    if len(data) < 8:
        return None
    if data[:7] == bytes([0xFF] * 7):
        return {0xFC: 'ENABLE', 0xFD: 'DISABLE', 0xFE: 'ZERO'}.get(data[7])
    return None


# Public API

def inspect(msg, params=None, direction: str = 'auto') -> str:
    if isinstance(msg, (bytes, bytearray)):
        arb_id, data, dlc = 0, bytes(msg), len(msg)
    else:
        arb_id = msg.arbitration_id
        data   = bytes(msg.data)
        dlc    = msg.dlc

    hex_str = ' '.join(f'{b:02X}' for b in data)
    bin_str = ' '.join(f'{b:08b}' for b in data)
    lines   = [
        f"ID: 0x{arb_id:03X}  DLC: {dlc}",
        f"  Hex: [{hex_str}]",
        f"  Bin: [{bin_str}]",
    ]

    if params is not None and len(data) == 8:
        special = _is_mit_special(data)
        if special:
            lines.append(f"  MIT: {special}")
        else:
            show_rx = direction in ('rx', 'auto')
            show_tx = direction in ('tx', 'auto')
            rx_label = 'MIT response       ' if direction == 'rx' else 'MIT (if RX response)'
            tx_label = 'MIT command        ' if direction == 'tx' else 'MIT (if TX command) '
            if show_rx:
                r = _mit_decode_response(data, params)
                lines.append(
                    f"  {rx_label} — id:{r['motor_id']}  "
                    f"pos:{r['position']:+.4f} rad  "
                    f"vel:{r['velocity']:+.4f} rad/s  "
                    f"tor:{r['torque']:+.4f} N·m  "
                    f"temp:{r['temp_C']}°C  err:{r['error']}"
                )
            if show_tx:
                c = _mit_decode_command(data, params)
                lines.append(
                    f"  {tx_label} — "
                    f"pos:{c['position']:+.4f} rad  "
                    f"vel:{c['velocity']:+.4f} rad/s  "
                    f"tor:{c['torque']:+.4f} N·m  "
                    f"kp:{c['kp']:.3f}  kd:{c['kd']:.4f}"
                )

    return '\n'.join(lines)


# Per-motor frame capture

_capture_state: dict = {}   # id(bus) -> {motor_id: {'tx': bytes, 'rx': bytes}}


def _ensure_hooked(bus) -> dict:
    key = id(bus)
    if key in _capture_state:
        return _capture_state[key]

    caps: dict = {}
    _capture_state[key] = caps
    orig_send, orig_recv = bus.send, bus.recv

    def _send(motor_id, data):
        caps.setdefault(motor_id, {'tx': b'', 'rx': b''})['tx'] = bytes(data)
        return orig_send(motor_id, data)

    def _recv(motor_id, timeout: float = 1.0):
        msg = orig_recv(motor_id, timeout=timeout)
        if msg is not None:
            caps.setdefault(motor_id, {'tx': b'', 'rx': b''})['rx'] = bytes(msg.data)
        return msg

    bus.send, bus.recv = _send, _recv
    return caps


def peek(motor, direction: str = 'tx', hex_only: bool = False) -> str:
    if direction not in ('tx', 'rx'):
        raise ValueError(f"direction must be 'tx' or 'rx', got {direction!r}")

    caps = _ensure_hooked(motor._bus)
    data = caps.setdefault(motor.id, {'tx': b'', 'rx': b''})[direction]

    if not data:
        return '(no frame yet)'
    if hex_only:
        return ' '.join(f'{b:02X}' for b in data)
    return inspect(data, motor.params, direction=direction)


# Live sniffer

def sniff(bus, duration: float | None = None, count: int | None = None,
          params=None, direction: str = 'rx') -> None:
    t0      = time.monotonic()
    seen    = 0
    try:
        for msg in bus:
            if msg is None:
                continue
            print(inspect(msg, params, direction=direction))
            print()
            seen += 1
            if count is not None and seen >= count:
                break
            if duration is not None and time.monotonic() - t0 >= duration:
                break
    except KeyboardInterrupt:
        pass
