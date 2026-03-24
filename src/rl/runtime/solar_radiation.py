from __future__ import annotations

"""
태양 복사 (Solar Radiation) 계산 모듈

태양 위치, 구름량, 강수량을 기반으로 GHI (Global Horizontal Irradiance)를 계산합니다.
"""

import numpy as np


def calculate_ghi_from_weather(
    sun_altitude_deg: float,
    cloud_coverage_pct: float = 0.0,
    precipitation_mm: float = 0.0,
    method: str = "simplified_bird"
) -> float:
    """날씨 조건을 고려한 GHI 계산.

    Args:
        sun_altitude_deg: 태양 고도각 [도] (0~90, 0=수평선, 90=천정)
        cloud_coverage_pct: 구름량 [%] (0~100)
        precipitation_mm: 1시간 강수량 [mm]
        method: 계산 방법 ("simplified_bird", "simple")

    Returns:
        GHI [Wh/m²]

    Notes:
        1. 청명 일사량 계산 (Clear Sky Model)
        2. 구름 효과 적용 (Kasten & Czeplak)
        3. 강수 효과 적용
    """
    print("  [SOLAR_RAD] GHI 계산 시작")
    print(f"  [SOLAR_RAD]   - 태양 고도: {sun_altitude_deg:.2f}°")
    print(f"  [SOLAR_RAD]   - 구름량: {cloud_coverage_pct:.1f}%")
    print(f"  [SOLAR_RAD]   - 강수량: {precipitation_mm:.1f} mm/h")

    # 1. 야간 체크
    if sun_altitude_deg <= 0:
        print("  [SOLAR_RAD] → 야간 (태양 고도 ≤ 0°), GHI=0")
        return 0.0

    # 2. 청명 일사량 계산
    if method == "simplified_bird":
        clear_sky_ghi = _calculate_clear_sky_bird(sun_altitude_deg)
    else:
        clear_sky_ghi = _calculate_clear_sky_simple(sun_altitude_deg)

    print(f"  [SOLAR_RAD]   - 청명 일사량: {clear_sky_ghi:.2f} Wh/m²")

    # 3. 구름 효과 적용
    ghi_with_clouds = _apply_cloud_effect(clear_sky_ghi, cloud_coverage_pct)
    print(f"  [SOLAR_RAD]   - 구름 효과 후: {ghi_with_clouds:.2f} Wh/m²")

    # 4. 강수 효과 적용
    final_ghi = _apply_precipitation_effect(ghi_with_clouds, precipitation_mm)
    print(f"  [SOLAR_RAD]   - 최종 GHI: {final_ghi:.2f} Wh/m²")

    return final_ghi


def _calculate_clear_sky_bird(sun_altitude_deg: float) -> float:
    """Simplified Bird Clear Sky Model.

    Args:
        sun_altitude_deg: 태양 고도각 [도]

    Returns:
        청명 조건에서의 GHI [Wh/m²]

    Notes:
        - Bird & Hulstrom (1981) 간소화 버전
        - 대기 투과율, 에어로졸, 수증기 등을 단순화
        - 정확도: ±5-10% (청명 조건)
    """
    if sun_altitude_deg <= 0:
        return 0.0

    # 태양 상수 (대기권 밖 일사량)
    SOLAR_CONSTANT = 1367.0  # W/m²

    # 천정각 계산
    zenith_angle_deg = 90.0 - sun_altitude_deg
    zenith_angle_rad = np.deg2rad(zenith_angle_deg)
    cos_zenith = np.cos(zenith_angle_rad)

    if cos_zenith <= 0:
        return 0.0

    # Air Mass 계산 (Kasten & Young 1989)
    # AM = 1 / [cos(θz) + 0.50572 * (96.07995 - θz)^-1.6364]
    air_mass = 1.0 / (
        cos_zenith + 0.50572 * ((96.07995 - zenith_angle_deg) ** -1.6364)
    )

    # Air Mass가 너무 크면 (낮은 고도) 제한
    air_mass = min(air_mass, 10.0)

    # 대기 투과율 (Simplified)
    # Rayleigh scattering, Ozone absorption, Water vapor, Aerosol 등을 통합
    # 경험식: T = a^(AM^b)
    a = 0.75  # 청명도 계수 (0.7~0.8: 맑음, 0.5~0.7: 보통)
    b = 0.678  # 대기 감쇠 지수

    transmittance = a ** (air_mass ** b)

    # Direct Normal Irradiance (DNI)
    dni = SOLAR_CONSTANT * transmittance

    # Direct Horizontal Irradiance
    dhi_direct = dni * cos_zenith

    # Diffuse Horizontal Irradiance (산란광)
    # 간소화: DNI의 10-15%
    dhi_diffuse = dni * 0.15 * (1.0 - transmittance)

    # Global Horizontal Irradiance
    ghi = dhi_direct + dhi_diffuse

    return float(max(ghi, 0.0))


def _calculate_clear_sky_simple(sun_altitude_deg: float) -> float:
    """간단한 청명 일사량 계산 (단순 모델).

    Args:
        sun_altitude_deg: 태양 고도각 [도]

    Returns:
        청명 조건에서의 GHI [Wh/m²]

    Notes:
        - 매우 단순한 모델 (교육용)
        - sin 함수 기반
        - 정확도: ±20-30%
    """
    if sun_altitude_deg <= 0:
        return 0.0

    # 최대 일사량 (정오, 청명 조건)
    MAX_GHI = 1000.0  # W/m²

    # sin 함수로 일사량 추정
    # GHI = MAX * sin(altitude)
    altitude_rad = np.deg2rad(sun_altitude_deg)
    ghi = MAX_GHI * np.sin(altitude_rad)

    return float(max(ghi, 0.0))


def _apply_cloud_effect(clear_sky_ghi: float, cloud_coverage_pct: float) -> float:
    """구름량에 따른 일사량 감소 적용.

    Args:
        clear_sky_ghi: 청명 조건 GHI [Wh/m²]
        cloud_coverage_pct: 구름량 [0-100%]

    Returns:
        구름 효과 적용된 GHI [Wh/m²]

    Notes:
        - Kasten & Czeplak (1980) 모델 기반
        - 구름량 0%: 100% 일사량
        - 구름량 50%: ~60% 일사량
        - 구름량 100%: ~20-25% 일사량 (산란광만)

    Reference:
        Kasten, F., & Czeplak, G. (1980). Solar and terrestrial radiation
        dependent on the amount and type of cloud. Solar Energy, 24(2), 177-189.
    """
    # 구름 커버 비율 (0~1)
    cloud_fraction = np.clip(cloud_coverage_pct / 100.0, 0.0, 1.0)

    # Kasten & Czeplak 경험식 (단순화 버전)
    # GHI_cloudy = GHI_clear * (1 - c1 * cloud^c2)
    # c1: 구름 효과 강도 (0.75~0.85)
    # c2: 비선형 계수 (1.0~1.3)

    c1 = 0.78  # 구름 효과 강도
    c2 = 1.0   # 선형 근사

    # 일사량 감소 계수
    # 완전 흐림일 때도 약 20-25%의 산란광 존재
    reduction_factor = 1.0 - c1 * (cloud_fraction ** c2)

    # 최소값 보장 (완전 흐림일 때 최소 20%)
    reduction_factor = max(reduction_factor, 0.20)

    ghi_with_clouds = clear_sky_ghi * reduction_factor

    return float(max(ghi_with_clouds, 0.0))


def _apply_precipitation_effect(ghi: float, precipitation_mm: float) -> float:
    """강수량에 따른 추가 일사량 감소.

    Args:
        ghi: 구름 효과 적용된 GHI [Wh/m²]
        precipitation_mm: 1시간 강수량 [mm]

    Returns:
        최종 GHI [Wh/m²]

    Notes:
        - 비나 눈이 오면 추가 감소
        - 약한 비 (1mm/h): ~30% 추가 감소
        - 보통 비 (5mm/h): ~50% 추가 감소
        - 강한 비 (10mm/h): ~70% 추가 감소
    """
    if precipitation_mm <= 0:
        return ghi

    # 강수량에 따른 감소 계수
    # 경험식: factor = exp(-k * rain)
    k = 0.15  # 감소 계수

    rain_factor = np.exp(-k * precipitation_mm)

    # 최소값 보장 (강한 비에도 최소 10% 산란광)
    rain_factor = max(rain_factor, 0.10)

    final_ghi = ghi * rain_factor

    return float(max(final_ghi, 0.0))


def estimate_cloud_coverage_from_description(weather_description: str) -> float:
    """날씨 설명으로부터 구름량 추정.

    Args:
        weather_description: OpenWeatherMap 날씨 설명
            예: "clear sky", "few clouds", "overcast clouds"

    Returns:
        추정 구름량 [%]

    Notes:
        OpenWeatherMap의 날씨 설명을 구름량으로 변환
    """
    description_lower = weather_description.lower()

    # 날씨 설명 → 구름량 매핑
    if "clear" in description_lower:
        return 0.0
    elif "few clouds" in description_lower:
        return 20.0
    elif "scattered clouds" in description_lower:
        return 40.0
    elif "broken clouds" in description_lower:
        return 70.0
    elif "overcast" in description_lower:
        return 95.0
    elif "clouds" in description_lower:
        return 50.0  # 기본값
    else:
        return 30.0  # 알 수 없는 경우


def validate_ghi(ghi: float, sun_altitude_deg: float) -> float:
    """GHI 값 검증 및 보정.

    Args:
        ghi: 계산된 GHI [Wh/m²]
        sun_altitude_deg: 태양 고도각 [도]

    Returns:
        검증된 GHI [Wh/m²]

    Notes:
        - 물리적으로 불가능한 값 제거
        - 태양 상수 초과 방지
        - 음수 방지
    """
    # 음수 방지
    if ghi < 0:
        return 0.0

    # 야간
    if sun_altitude_deg <= 0:
        return 0.0

    # 최대값 제한 (태양 상수 + 여유)
    MAX_POSSIBLE_GHI = 1500.0  # W/m² (안전 마진 포함)

    if ghi > MAX_POSSIBLE_GHI:
        print(f"  [SOLAR_RAD] ⚠️ GHI 과다 ({ghi:.0f} > {MAX_POSSIBLE_GHI}), 클리핑")
        return MAX_POSSIBLE_GHI

    return float(ghi)


# 테스트 함수
if __name__ == "__main__":
    """일사량 계산 테스트."""
    print("\n" + "="*60)
    print("일사량 계산 테스트")
    print("="*60)

    # 테스트 케이스
    test_cases = [
        # (태양 고도, 구름량, 강수량, 설명)
        (60, 0, 0, "정오, 맑음"),
        (60, 50, 0, "정오, 반 흐림"),
        (60, 100, 0, "정오, 완전 흐림"),
        (60, 50, 5, "정오, 반 흐림 + 비"),
        (30, 0, 0, "아침/저녁, 맑음"),
        (10, 0, 0, "일출/일몰"),
        (0, 0, 0, "지평선"),
        (-10, 0, 0, "야간"),
    ]

    for sun_alt, clouds, rain, desc in test_cases:
        ghi = calculate_ghi_from_weather(
            sun_altitude_deg=sun_alt,
            cloud_coverage_pct=clouds,
            precipitation_mm=rain,
            method="simplified_bird"
        )
        print(f"\n{desc}:")
        print(f"  GHI = {ghi:.0f} Wh/m²")

    print("\n" + "="*60)
