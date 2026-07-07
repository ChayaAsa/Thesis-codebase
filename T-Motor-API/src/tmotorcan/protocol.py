# Exceptions

class MotorFaultError(RuntimeError):

    def __init__(self, motor_id: int, code: int, detail: str = '') -> None:
        self.motor_id = motor_id
        self.code     = code
        msg = f"Motor {motor_id} fault {code}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class MotorTimeoutError(RuntimeError):

    def __init__(self, motor_id: int, timeout: float) -> None:
        self.motor_id = motor_id
        self.timeout  = timeout
        super().__init__(f"Motor {motor_id}: no reply within {timeout}s")


# Fault-code dictionary

FAULT_MESSAGES: dict[int, str] = {
    0: 'No error',
    1: 'Over temperature',
    2: 'Over current',
    3: 'Over voltage',
    4: 'Under voltage',
    5: 'Encoder fault',
    6: 'Phase current unbalance',
}


# Frame field codec

def _encode(x: float, xmin: float, xmax: float, bits: int) -> int:
    x = max(xmin, min(xmax, x))
    return int((x - xmin) / (xmax - xmin) * ((1 << bits) - 1))


def _decode(x: int, xmin: float, xmax: float, bits: int) -> float:
    return x * (xmax - xmin) / ((1 << bits) - 1) + xmin
