import queue
import threading

import can


class MotorBus:

    def __init__(self, bus: can.BusABC) -> None:
        self._bus = bus
        self._tx_lock = threading.Lock()
        self._queues: dict[int, queue.Queue] = {}
        self._active = True

        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name='MotorBus-RX')
        self._rx_thread.start()
        print(f"[MotorBus] Connected on {self._bus.channel_info}")

    # Public

    def register(self, motor_id: int) -> None:
        if motor_id in self._queues:
            raise ValueError(
                f"motor_id {motor_id} is already registered on this bus — "
                f"duplicate MITMotor? Use a unique id, or close the existing "
                f"motor first.")
        # maxsize=1 + drop-oldest in _rx_loop: recv() always returns the newest
        # frame. Keeps memory bounded if update() falls behind.
        self._queues[motor_id] = queue.Queue(maxsize=1)

    def unregister(self, motor_id: int) -> None:
        self._queues.pop(motor_id, None)

    def send(self, motor_id: int, data: bytes | bytearray) -> None:
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        with self._tx_lock:
            self._bus.send(msg)

    def recv(self, motor_id: int, timeout: float = 1.0) -> can.Message | None:
        if not self._rx_thread.is_alive():
            raise RuntimeError(
                "MotorBus RX thread has died — CAN bus lost or USB disconnected. "
                "Check your USB-CAN adapter and restart.")
        try:
            return self._queues[motor_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        # Drain the per-motor queues rather than the raw bus — the RX thread
        # owns self._bus.recv() and calling it from here would race.
        for q in self._queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        # _rx_loop polls bus.recv() with timeout=0.1, so it should observe
        # _active=False and exit within ~100 ms. Allow generous margin.
        self._rx_thread.join(timeout=2.0)
        if self._rx_thread.is_alive():
            raise RuntimeError(
                "MotorBus RX thread did not stop within 2s; refusing to shut "
                "down the underlying bus to avoid a use-after-close race.")
        self._bus.shutdown()

    # alias so both names work
    shutdown = close

    # Context manager

    def __enter__(self) -> 'MotorBus':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # Internal

    def _rx_loop(self) -> None:
        while self._active:
            try:
                msg = self._bus.recv(timeout=0.1)
            except Exception as e:
                if self._active:
                    print(f"[MotorBus] RX thread error: {e!r} — CAN bus lost or USB disconnected")
                break
            if msg is None or len(msg.data) < 6:
                continue
            mid = int(msg.data[0])
            q = self._queues.get(mid)
            if q is None:
                continue
            # drop-oldest policy: keep the newest reply, discard anything stale.
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(msg)
        if self._active:
            print("[MotorBus] RX thread stopped unexpectedly")
