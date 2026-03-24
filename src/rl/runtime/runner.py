from __future__ import annotations

import argparse
import time

from .controller import RealTimeController


def main() -> None:
    print("="*80)
    print(" " * 20 + "실시간 파사드 제어 시스템 시작")
    print("="*80)

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    args = p.parse_args()

    print(f"\n[RUNNER] 설정 파일: {args.config}")
    mode_text = '단일 실행 (--once)' if args.once else '무한 루프 (--loop)'
    print(f"[RUNNER] 실행 모드: {mode_text}")
    print()

    print("[RUNNER] RealTimeController 초기화 중...")
    ctrl = RealTimeController(args.config)
    print("[RUNNER] ✓ 초기화 완료\n")

    if args.once:
        print("[RUNNER] 단일 제어 사이클 실행 시작\n")
        ctrl.run_once()
        print("\n[RUNNER] ✓ 단일 제어 사이클 완료")
    else:
        print("[RUNNER] 무한 루프 시작 (간격: 1.0초)\n")
        cycle_count = 0
        while True:
            cycle_count += 1
            print(f"\n{'*'*80}")
            print(f"[RUNNER] 제어 사이클 #{cycle_count}")
            print(f"{'*'*80}")

            try:
                ctrl.run_once()
            except KeyboardInterrupt:
                print("\n\n[RUNNER] ⚠️  사용자 중단 (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n[RUNNER] ❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()

            print("\n[RUNNER] 다음 사이클까지 1초 대기...")
            time.sleep(1.0)

        print("\n[RUNNER] 프로그램 종료")


if __name__ == "__main__":
    main()
