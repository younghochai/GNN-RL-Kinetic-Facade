from __future__ import annotations

import os

from src.rl.runtime.controller import RealTimeController


def test_run_once_cli_like():
    # 환경 변수 키가 없어도 더미 경로/더미 버스로 실행 가능해야 함
    os.environ.setdefault("KMA_API_KEY", "DUMMY")
    ctrl = RealTimeController("configs/runtime.yaml")
    ctrl.run_once()
