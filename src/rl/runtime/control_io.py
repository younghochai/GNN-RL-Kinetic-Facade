from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ControlBus(ABC):
    @abstractmethod
    def send(self, theta_deg: np.ndarray) -> bool:  # noqa: D401
        """각도 명령을 송신한다."""
        raise NotImplementedError


class DummyBus(ControlBus):
    def __init__(self) -> None:
        pass

    def send(self, theta_deg: np.ndarray) -> bool:
        # 개발 환경: 실제 전송 대신 성공 True 반환
        print(f"  [BUS] DummyBus: {len(theta_deg)}개 모듈 각도 전송 (시뮬레이션)")
        print(f"  [BUS]   - 각도 범위: [{theta_deg.min():.2f}, {theta_deg.max():.2f}]°")
        print(f"  [BUS]   - 평균 각도: {theta_deg.mean():.2f}°")
        return True


class ModbusBus(ControlBus):
    def __init__(self, port: str, baudrate: int = 115200, retries: int = 2) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.retries = int(retries)
        print(f"  [BUS] ModbusBus 초기화: port={port}, baudrate={baudrate}, retries={retries}")
        # 실제 구현 시 pymodbus/serial 설정 추가

    def send(self, theta_deg: np.ndarray) -> bool:
        # 장비 없는 환경에서는 더미 동작
        n_modules = len(theta_deg)
        print(f"  [BUS] ModbusBus: {n_modules}개 모듈 각도 전송 시도")
        print(f"  [BUS]   - Port: {self.port}, Baudrate: {self.baudrate}")
        print(f"  [BUS]   - 각도 범위: [{theta_deg.min():.2f}, {theta_deg.max():.2f}]°")
        print(f"  [BUS]   - 평균 각도: {theta_deg.mean():.2f}°")
        # TODO: 실제 RS-485/Modbus 프레임 규약 구현 연결
        print("  [BUS] ⚠️  실제 Modbus 미구현, 더미 성공 반환")
        return True


def build_bus(cfg: dict) -> ControlBus:
    bus_type = str(cfg.get("control", {}).get("bus", "rs485")).lower()
    if bus_type in ("dummy", "none"):
        return DummyBus()
    if bus_type in ("rs485", "modbus"):
        c = cfg.get("control", {})
        return ModbusBus(port=c.get("port", "/dev/ttyUSB0"), baudrate=int(c.get("baudrate", 115200)), retries=int(c.get("retries", 2)))
    return DummyBus()
