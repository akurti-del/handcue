"""
HandController - the single source of truth for controlling the robotic hand.

Design rules:
- HandController is an abstract contract. Real hardware and simulator both
  implement the same interface, so upstream code (voice, intent) doesn't know
  or care which is running.
- Every command goes through safety validation BEFORE it reaches the driver.
- State is tracked in the controller, not the driver.
- No LLM-generated code is allowed to import this module directly. Safety-
  critical code must be human-written and reviewed.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, List, Optional

log = logging.getLogger("handcue.hand")


class Finger(Enum):
    """Fingers indexed consistently across all code."""
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


# Global safety limits - apply no matter what the caller asks for.
MAX_FORCE_PCT = 100.0
SAFE_FORCE_PCT = 60.0
MIN_FINGER_ANGLE = 0.0
MAX_FINGER_ANGLE = 180.0
MAX_ANGULAR_VELOCITY = 180.0
DEFAULT_VELOCITY = 90.0
COMMAND_TTL_SECONDS = 0.5


@dataclass
class FingerCommand:
    finger: Finger
    target_angle: float
    max_force_pct: float = SAFE_FORCE_PCT
    velocity_deg_per_sec: float = DEFAULT_VELOCITY
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class HandState:
    finger_angles: List[float] = field(default_factory=lambda: [0.0] * 5)
    finger_forces: List[float] = field(default_factory=lambda: [0.0] * 5)
    moving: bool = False
    emergency_stopped: bool = False
    last_update: float = field(default_factory=time.monotonic)
    driver_connected: bool = False


class HandDriver(ABC):
    """Low-level hand I/O. Serial, BLE, or simulator - all implement this."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def send_finger_command(self, cmd: FingerCommand) -> bool: ...

    @abstractmethod
    def read_state(self) -> Optional[dict]: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...


class SimulatorDriver(HandDriver):
    """Fake driver that just logs what would happen. For dev without hardware."""

    def __init__(self):
        self._connected = False
        self._angles = [0.0] * 5
        self._forces = [0.0] * 5
        self._stopped = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        self._connected = True
        log.info("[SIM] Driver connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        log.info("[SIM] Driver disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def send_finger_command(self, cmd: FingerCommand) -> bool:
        if not self._connected or self._stopped:
            return False
        with self._lock:
            self._angles[cmd.finger.value] = cmd.target_angle
            self._forces[cmd.finger.value] = min(cmd.max_force_pct, cmd.target_angle / 1.8)
        log.info(
            "[SIM] Move %s -> %.1f deg @ %.0f%% force",
            cmd.finger.name, cmd.target_angle, cmd.max_force_pct
        )
        return True

    def read_state(self) -> Optional[dict]:
        if not self._connected:
            return None
        with self._lock:
            return {
                "angles": list(self._angles),
                "forces": list(self._forces),
                "stopped": self._stopped,
            }

    def emergency_stop(self) -> None:
        self._stopped = True
        log.warning("[SIM] EMERGENCY STOP")

    def clear_emergency_stop(self) -> None:
        self._stopped = False


class HandController:
    """Top-level hand control. Everything upstream talks to this, not the driver."""

    def __init__(self, driver: HandDriver, *, watchdog_timeout_sec: float = 2.0):
        self.driver = driver
        self.state = HandState()
        self._state_lock = threading.Lock()
        self._watchdog_timeout = watchdog_timeout_sec
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog = threading.Event()
        self._on_estop_callbacks: List[Callable[[], None]] = []

    def start(self) -> bool:
        if not self.driver.connect():
            return False
        with self._state_lock:
            self.state.driver_connected = True
            self.state.last_update = time.monotonic()
        self._start_watchdog()
        return True

    def stop(self) -> None:
        self._stop_watchdog.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.0)
        self.driver.disconnect()

    def emergency_stop(self, reason: str = "manual") -> None:
        log.warning("E-STOP: %s", reason)
        try:
            self.driver.emergency_stop()
        except Exception as e:
            log.error("E-stop raised: %s", e)
        with self._state_lock:
            self.state.emergency_stopped = True
            self.state.moving = False
        for cb in list(self._on_estop_callbacks):
            try:
                cb()
            except Exception as e:
                log.error("E-stop callback raised: %s", e)

    def clear_emergency_stop(self) -> None:
        with self._state_lock:
            self.state.emergency_stopped = False
        if hasattr(self.driver, "clear_emergency_stop"):
            self.driver.clear_emergency_stop()

    def on_emergency_stop(self, callback: Callable[[], None]) -> None:
        self._on_estop_callbacks.append(callback)

    def move_finger(
        self,
        finger: Finger,
        target_angle: float,
        *,
        max_force_pct: float = SAFE_FORCE_PCT,
        velocity_deg_per_sec: float = DEFAULT_VELOCITY,
    ) -> bool:
        if self.state.emergency_stopped:
            return False
        if not isinstance(finger, Finger):
            return False
        clamped_angle = _clamp(target_angle, MIN_FINGER_ANGLE, MAX_FINGER_ANGLE)
        clamped_force = _clamp(max_force_pct, 0.0, MAX_FORCE_PCT)
        clamped_vel = _clamp(velocity_deg_per_sec, 0.0, MAX_ANGULAR_VELOCITY)

        cmd = FingerCommand(
            finger=finger,
            target_angle=clamped_angle,
            max_force_pct=clamped_force,
            velocity_deg_per_sec=clamped_vel,
        )

        if time.monotonic() - cmd.created_at > COMMAND_TTL_SECONDS:
            return False

        ok = self.driver.send_finger_command(cmd)
        if ok:
            with self._state_lock:
                self.state.finger_angles[finger.value] = clamped_angle
                self.state.moving = True
                self.state.last_update = time.monotonic()
        return ok

    def move_all_fingers(self, angles: List[float], **kwargs) -> bool:
        if len(angles) != 5:
            return False
        return all(
            self.move_finger(Finger(i), angle, **kwargs)
            for i, angle in enumerate(angles)
        )

    def grip(self, force_pct: float = SAFE_FORCE_PCT, speed: str = "normal") -> bool:
        vel = {"slow": 45.0, "normal": 90.0, "fast": 150.0}.get(speed, 90.0)
        return self.move_all_fingers(
            [140.0, 160.0, 160.0, 160.0, 160.0],
            max_force_pct=force_pct,
            velocity_deg_per_sec=vel,
        )

    def release(self, speed: str = "normal") -> bool:
        vel = {"slow": 45.0, "normal": 90.0, "fast": 150.0}.get(speed, 90.0)
        return self.move_all_fingers([0.0] * 5, velocity_deg_per_sec=vel)

    def get_state(self) -> HandState:
        with self._state_lock:
            return HandState(
                finger_angles=list(self.state.finger_angles),
                finger_forces=list(self.state.finger_forces),
                moving=self.state.moving,
                emergency_stopped=self.state.emergency_stopped,
                last_update=self.state.last_update,
                driver_connected=self.state.driver_connected,
            )

    def _start_watchdog(self) -> None:
        self._stop_watchdog.clear()
        t = threading.Thread(target=self._watchdog_loop, daemon=True, name="HandWatchdog")
        t.start()
        self._watchdog_thread = t

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.is_set():
            time.sleep(0.1)
            st = self.driver.read_state()
            now = time.monotonic()
            if st is None:
                with self._state_lock:
                    gap = now - self.state.last_update
                if gap > self._watchdog_timeout and not self.state.emergency_stopped:
                    self.emergency_stop(reason=f"watchdog: no response for {gap:.1f}s")
                continue
            with self._state_lock:
                self.state.finger_forces = st.get("forces", self.state.finger_forces)
                self.state.last_update = now


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
