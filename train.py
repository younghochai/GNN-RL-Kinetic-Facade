#!/usr/bin/env python3
"""
RL 파이프라인 학습 진입점(wrapper).
요구사항에 맞춰 루트에서 `python train.py --config configs/default.yaml` 실행을 지원한다.
실제 구현은 `src.rl.src.train`의 `main`을 호출한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args, unknown = parser.parse_known_args()

    # src.rl.src.train 으로 위임
    sys.path.append(str(Path(__file__).resolve().parent))
    from src.rl.src.train import main as train_main  # type: ignore

    train_main(config_path=args.config, run_name=args.run_name, device=args.device)


if __name__ == "__main__":
    main()
