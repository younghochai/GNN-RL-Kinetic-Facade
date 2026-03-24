"""
Solcast API 통합 테스트 스크립트

사용 방법:
1. API 키 설정:
   export SOLCAST_API_KEY="your_api_key_here"

2. 실행:
   python test_solcast.py

또는 runtime.yaml에 API 키 설정 후:
   python test_solcast.py --use-config
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rl.runtime.weather import fetch_weather_data, fetch_solcast_irradiance
from src.rl.runtime.sun import compute_sun_position


def test_fetch_weather_data(cfg: dict) -> None:
    """기상 데이터 조회 테스트."""
    print("\n" + "=" * 80)
    print("TEST 1: fetch_weather_data() - Solcast API 호출")
    print("=" * 80)

    ts_kst = pd.Timestamp.now(tz="Asia/Seoul")
    print(f"조회 시각: {ts_kst}")

    # 태양 위치 계산 (dummy 모드용)
    sun = compute_sun_position(
        ts_utc=ts_kst.tz_convert("UTC"),
        latitude=cfg["site"]["lat"],
        longitude=cfg["site"]["lon"],
    )
    print(f"태양 위치: 고도={sun.sun_alt_deg:.2f}°, 방위각={sun.sun_azi_deg:.2f}°")

    try:
        weather = fetch_weather_data(ts_kst, cfg, sun)

        print("\n✅ 성공!")
        print(f"  - GHI: {weather.ghi_wh_per_m2:.2f} W/m²")
        print(f"  - 기온: {weather.T1H_degC:.1f}°C")
        print(f"  - 습도: {weather.REH_pct:.1f}%")
        print(f"  - 풍속: {weather.WSD_ms:.1f} m/s")
        print(f"  - 강수형태: {weather.PTY_code}")

        # 검증
        assert weather.ghi_wh_per_m2 >= 0, "GHI는 0 이상이어야 합니다"
        assert -50 <= weather.T1H_degC <= 60, "기온이 비정상적입니다"
        assert 0 <= weather.REH_pct <= 100, "습도가 비정상적입니다"
        assert weather.WSD_ms >= 0, "풍속은 0 이상이어야 합니다"
        assert weather.PTY_code in [0, 1, 2, 3, 4], "강수형태 코드가 잘못되었습니다"

        print("\n✅ 데이터 검증 통과")

    except Exception as e:
        print(f"\n❌ 실패: {e}")
        raise


def test_fetch_solcast_irradiance(cfg: dict) -> None:
    """DNI/DHI 조회 테스트."""
    print("\n" + "=" * 80)
    print("TEST 2: fetch_solcast_irradiance() - DNI/DHI 직접 조회")
    print("=" * 80)

    ts_kst = pd.Timestamp.now(tz="Asia/Seoul")
    print(f"조회 시각: {ts_kst}")

    try:
        irradiance = fetch_solcast_irradiance(ts_kst, cfg)

        print("\n✅ 성공!")
        print(f"  - DNI: {irradiance.dni_wh_per_m2:.2f} W/m²")
        print(f"  - DHI: {irradiance.dhi_wh_per_m2:.2f} W/m²")

        # 검증
        assert irradiance.dni_wh_per_m2 >= 0, "DNI는 0 이상이어야 합니다"
        assert irradiance.dhi_wh_per_m2 >= 0, "DHI는 0 이상이어야 합니다"
        assert irradiance.dni_wh_per_m2 <= 1500, "DNI가 비정상적으로 큽니다"
        assert irradiance.dhi_wh_per_m2 <= 1000, "DHI가 비정상적으로 큽니다"

        print("\n✅ 데이터 검증 통과")

    except Exception as e:
        print(f"\n❌ 실패: {e}")
        raise


def test_caching(cfg: dict) -> None:
    """캐싱 테스트."""
    print("\n" + "=" * 80)
    print("TEST 3: 캐싱 동작 확인")
    print("=" * 80)

    ts_kst = pd.Timestamp.now(tz="Asia/Seoul")

    # 첫 번째 호출 (API 호출)
    print("\n[1차 호출] API 호출 예상...")
    weather1 = fetch_weather_data(ts_kst, cfg)
    print(f"GHI: {weather1.ghi_wh_per_m2:.2f} W/m²")

    # 두 번째 호출 (캐시 hit)
    print("\n[2차 호출] 캐시 사용 예상...")
    weather2 = fetch_weather_data(ts_kst, cfg)
    print(f"GHI: {weather2.ghi_wh_per_m2:.2f} W/m²")

    # 검증
    assert weather1.ghi_wh_per_m2 == weather2.ghi_wh_per_m2, "캐시된 데이터가 다릅니다"
    print("\n✅ 캐싱 동작 확인")


def main():
    parser = argparse.ArgumentParser(description="Solcast API 통합 테스트")
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="runtime.yaml의 API 키 사용 (기본: 환경변수)",
    )
    parser.add_argument(
        "--skip-irradiance",
        action="store_true",
        help="DNI/DHI 테스트 건너뛰기 (API 호출 절약)",
    )
    args = parser.parse_args()

    # 설정 로드
    config_path = project_root / "src" / "rl" / "configs" / "runtime.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Provider 설정
    cfg["weather_provider"] = "solcast"

    # API 키 확인
    api_key = os.environ.get("SOLCAST_API_KEY", "")
    if not api_key and not args.use_config:
        print("❌ 오류: SOLCAST_API_KEY 환경변수가 설정되지 않았습니다.")
        print("\n해결 방법:")
        print("  1. export SOLCAST_API_KEY='your_api_key'")
        print("  2. python test_solcast.py --use-config (runtime.yaml 사용)")
        sys.exit(1)

    if api_key:
        print(f"✓ API 키 확인: {api_key[:10]}...")
    else:
        print("✓ runtime.yaml의 API 키 사용")

    # 테스트 실행
    try:
        test_fetch_weather_data(cfg)
        
        if not args.skip_irradiance:
            test_fetch_solcast_irradiance(cfg)
        else:
            print("\n⏭️  DNI/DHI 테스트 건너뛰기")
        
        test_caching(cfg)

        print("\n" + "=" * 80)
        print("🎉 모든 테스트 통과!")
        print("=" * 80)
        print("\n다음 단계:")
        print("  1. runtime.yaml에서 weather_provider를 'solcast'로 설정")
        print("  2. policy.ipynb 또는 run.ipynb에서 실행")
        print("  3. 캐시 TTL 조정 (필요 시)")
        print("\n⚠️  주의: 무료 플랜 50 calls/일 제한")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 테스트 실패: {e}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

