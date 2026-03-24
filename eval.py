#!/usr/bin/env python3
"""
RL 파이프라인 평가 진입점(wrapper).
실제 구현은 `src.rl.src.eval`의 `main`을 호출한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--device", type=str, default="auto")
    args, unknown = parser.parse_known_args()

    sys.path.append(str(Path(__file__).resolve().parent))
    from src.rl.src.eval import main as eval_main  # type: ignore

    eval_main(config_path=args.config, checkpoint=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()


