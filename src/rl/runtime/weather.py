from __future__ import annotations

from typing import Dict, Optional
import os
import time

import numpy as np
import pandas as pd
import requests

from .types import WeatherData, SunPosition, IrradianceData
from .solar_radiation import calculate_ghi_from_weather


# 캐시: 최근 성공한 데이터 저장 (신선도 체크용)
_CACHE: Dict[str, any] = {"ts": None, "data": None}
_IRRADIANCE_CACHE: Dict[str, any] = {"ts": None, "data": None}


def fetch_weather_data(
    ts_kst: pd.Timestamp,
    cfg: Dict,
    sun: Optional[SunPosition] = None
) -> WeatherData:
    """기상 데이터 조회 (Provider 선택).

    Args:
        ts_kst: 조회 시각 (KST, timezone-aware)
        cfg: 설정 (weather_provider, solcast, kma, site 등)
        sun: 태양 위치 (일부 provider에서 사용)

    Returns:
        WeatherData: GHI, 기온, 습도, 풍속, 강수형태

    Notes:
        - Provider: "solcast" (Solcast API), "kma" (기상청 API), "dummy" (시뮬레이션)
        - Solcast: GHI, DNI, DHI 직접 제공 (정확)
        - KMA: ICSR → GHI 변환, 초단기실황 조회
        - Dummy: 태양 위치 기반 시뮬레이션
        - 캐싱: TTL 내에서 캐시 재사용 (API 호출 절약)
    """
    print("\n" + "="*60)
    print(f"[WEATHER] 기상 데이터 조회 시작: {ts_kst}")
    print("="*60)

    provider = cfg.get("weather_provider", "dummy")
    print(f"[WEATHER] Provider: {provider}")

    # 캐싱 체크 (Solcast, OpenWeatherMap Solar의 경우 API 호출 제한 절약)
    if provider in ["solcast", "openweathermap_solar"]:
        provider_cfg_key = provider
        cache_ttl = cfg.get(provider_cfg_key, {}).get("cache_ttl_s", 300)
        if _CACHE.get("ts") is not None and _CACHE.get("data") is not None:
            elapsed = (ts_kst - _CACHE["ts"]).total_seconds()
            if elapsed < cache_ttl:
                print(f"[WEATHER] ✓ 캐시 사용 (경과: {elapsed:.0f}s < TTL: {cache_ttl}s)")
                print("="*60 + "\n")
                return _CACHE["data"]

    # Provider별 데이터 조회
    if provider == "openweathermap_solar":
        print("[WEATHER] OpenWeatherMap Solar Irradiance API 호출...")
        data = _fetch_openweathermap_solar(ts_kst, cfg, sun)
    
    elif provider == "solcast":
        print("[WEATHER] Solcast API 호출...")
        data = _fetch_solcast(ts_kst, cfg, sun)
    
    elif provider == "kma":
        print("[WEATHER] 기상청 API 호출...")
        api_cfg = cfg.get("kma", {})
        print(f"[WEATHER] API 설정: station_id={api_cfg.get('station_id')}, timeout={api_cfg.get('timeout_s')}s")
        
        # 1. ICSR (일사량) 조회
        print("[WEATHER] STEP 1/2: ICSR (일사량) 조회 중...")
        ghi = _fetch_icsr(ts_kst, api_cfg, sun)
        print(f"[WEATHER] ✓ ICSR → GHI 변환 완료: {ghi:.2f} Wh/m²")

        # 2. 기상 데이터 조회
        print("[WEATHER] STEP 2/2: 초단기실황 기상 데이터 조회 중...")
        weather_fields = _fetch_ultrashort(ts_kst, api_cfg)
        print(f"[WEATHER] ✓ 초단기실황 조회 완료: {weather_fields}")

        # WeatherData 생성
        data = WeatherData(
            ts=ts_kst,
            ghi_wh_per_m2=ghi,
            T1H_degC=weather_fields.get("T1H", 0.0),
            REH_pct=weather_fields.get("REH", 0.0),
            WSD_ms=weather_fields.get("WSD", 0.0),
            PTY_code=int(weather_fields.get("PTY", 0)),
        )
    
    elif provider == "dummy":
        print("[WEATHER] 더미 모드: 시뮬레이션 데이터 생성...")
        # 더미 모드 (현재 _fetch_icsr, _fetch_ultrashort의 더미 로직 사용)
        ghi = _fetch_icsr(ts_kst, {"use_dummy": True}, sun)
        weather_fields = _fetch_ultrashort(ts_kst, {"use_dummy": True})
        data = WeatherData(
            ts=ts_kst,
            ghi_wh_per_m2=ghi,
            T1H_degC=weather_fields.get("T1H", 0.0),
            REH_pct=weather_fields.get("REH", 0.0),
            WSD_ms=weather_fields.get("WSD", 0.0),
            PTY_code=int(weather_fields.get("PTY", 0)),
        )
    
    else:
        raise ValueError(f"Unknown weather_provider: {provider}. Use 'solcast', 'kma', or 'dummy'.")

    print("\n[WEATHER] 최종 WeatherData 생성:")
    print(f"  - GHI: {data.ghi_wh_per_m2:.2f} Wh/m²")
    print(f"  - 기온 (T1H): {data.T1H_degC:.1f}°C")
    print(f"  - 습도 (REH): {data.REH_pct:.1f}%")
    print(f"  - 풍속 (WSD): {data.WSD_ms:.1f} m/s")
    print(f"  - 강수형태 (PTY): {data.PTY_code}")
    print("="*60 + "\n")

    # 캐시 업데이트
    _CACHE["ts"] = ts_kst
    _CACHE["data"] = data

    return data


def _fetch_icsr(
    ts_kst: pd.Timestamp,
    api_cfg: Dict,
    sun: Optional[SunPosition] = None
) -> float:
    """ASOS ICSR (일사량) 조회 및 GHI 변환.

    Args:
        ts_kst: 조회 시각
        api_cfg: API 설정
        sun: 태양 위치 (더미 모드 일사량 계산용)

    Returns:
        GHI [Wh/m²]
    """
    # 더미 모드 체크
    if api_cfg.get("use_dummy", False):
        print("  [ICSR] 더미 모드: 태양 위치 기반 일사량 계산")

        if sun is None:
            # Sun 정보가 없으면 간단한 시뮬레이션
            print("  [ICSR] ⚠️ 태양 위치 정보 없음, 간단한 시뮬레이션 사용")
            hour = ts_kst.hour
            if 6 <= hour <= 18:
                solar_factor = np.sin(np.pi * (hour - 6) / 12)
                ghi = float(800 * solar_factor)
                print(f"  [ICSR] 더미 데이터: 시각={hour}시, GHI={ghi:.2f} Wh/m²")
                return ghi
            else:
                print(f"  [ICSR] 더미 데이터: 야간 (시각={hour}시), GHI=0.0 Wh/m²")
                return 0.0

        # 태양 위치 기반 실제 계산
        # 더미 모드에서는 맑은 날씨 가정 (구름량 20%, 강수 없음)
        cloud_coverage = 20.0  # 약간의 구름
        precipitation = 0.0     # 비 없음

        ghi = calculate_ghi_from_weather(
            sun_altitude_deg=sun.sun_alt_deg,
            cloud_coverage_pct=cloud_coverage,
            precipitation_mm=precipitation,
            method="simplified_bird"
        )

        print(f"  [ICSR] 계산된 GHI: {ghi:.2f} Wh/m² (고도={sun.sun_alt_deg:.1f}°)")
        return ghi

    timeout = float(api_cfg.get("timeout_s", 5))
    retry = int(api_cfg.get("retry", {}).get("max", 3))
    backoff = float(api_cfg.get("retry", {}).get("backoff", 0.6))

    url_tmpl = api_cfg["endpoints"]["asos_icsr"]
    api_key = os.environ.get("KMA_API_KEY", "")
    if api_key == "":
        api_key = api_cfg.get("api_key", "")

    params = {
        "serviceKey": api_key,
        "stationId": api_cfg.get("station_id"),
        "ts": ts_kst.strftime("%Y%m%d%H%M"),
    }

    print("  [ICSR] API 요청 준비:")
    print(f"    - URL: {url_tmpl}")
    print(f"    - Params: {params}")
    print(f"    - Retry 설정: max={retry}, backoff={backoff}s")

    icsr_mj: Optional[float] = None
    for attempt in range(retry):
        try:
            print(f"  [ICSR] API 호출 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url_tmpl, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            print(f"  [ICSR] ✓ API 응답 수신: {data}")
            # API 스키마에 맞춰 파싱 (예시)
            icsr_mj = float(data.get("icsr", 0.0))
            print(f"  [ICSR] ✓ ICSR 파싱 완료: {icsr_mj} MJ/m²·h")
            break
        except Exception as e:
            print(f"  [ICSR] ✗ API 호출 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [ICSR] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)

    # 변환: ICSR [MJ/m²·h] → GHI [Wh/m²]
    if icsr_mj is None or not np.isfinite(icsr_mj) or icsr_mj < 0:
        print(f"  [ICSR] ⚠️  ICSR 데이터 유효하지 않음 (icsr_mj={icsr_mj})")
        # 캐시 fallback
        if _CACHE.get("data") is not None:
            cached_ghi = _CACHE["data"].ghi_wh_per_m2
            print(f"  [ICSR] → 캐시 사용: {cached_ghi:.2f} Wh/m²")
            return cached_ghi
        print("  [ICSR] → 기본값 사용: 0.0 Wh/m²")
        return 0.0

    ghi = float(icsr_mj * 277.78)
    print(f"  [ICSR] 변환 완료: {icsr_mj:.4f} MJ/m²·h → {ghi:.2f} Wh/m²")
    return ghi


def _fetch_ultrashort(ts_kst: pd.Timestamp, api_cfg: Dict) -> Dict[str, float]:
    """초단기실황 기상 데이터 조회.

    Returns:
        {"T1H": 기온, "REH": 습도, "WSD": 풍속, "PTY": 강수형태}
    """
    # 더미 모드 체크
    if api_cfg.get("use_dummy", False):
        print("  [초단기] 더미 모드: 시뮬레이션 데이터 사용")
        # 시간대와 계절에 따라 합리적인 기상 데이터 생성
        hour = ts_kst.hour
        month = ts_kst.month

        # 기온: 계절 및 시간대 반영
        base_temp = 15.0  # 봄/가을 평균
        if month in [6, 7, 8]:  # 여름
            base_temp = 25.0
        elif month in [12, 1, 2]:  # 겨울
            base_temp = 0.0

        # 일교차 반영 (낮 +10도, 밤 -5도)
        temp_variation = 7.5 * np.sin(np.pi * (hour - 6) / 12) - 2.5
        T1H = base_temp + temp_variation

        # 습도: 60~80% 범위
        REH = 70.0 + 10.0 * np.sin(np.pi * hour / 12)

        # 풍속: 0~5 m/s 범위
        WSD = 2.0 + 1.5 * np.random.random()

        # 강수형태: 대부분 맑음
        PTY = 0

        fields = {
            "T1H": float(T1H),
            "REH": float(REH),
            "WSD": float(WSD),
            "PTY": int(PTY),
        }
        print(f"  [초단기] 더미 데이터: T1H={T1H:.1f}°C, REH={REH:.1f}%, WSD={WSD:.1f}m/s, PTY={PTY}")
        return fields

    timeout = float(api_cfg.get("timeout_s", 5))
    retry = int(api_cfg.get("retry", {}).get("max", 3))
    backoff = float(api_cfg.get("retry", {}).get("backoff", 0.6))

    url_tmpl = api_cfg["endpoints"]["ultrashort"]
    api_key = os.environ.get("KMA_API_KEY", "")

    params = {
        "serviceKey": api_key,
        "stationId": api_cfg.get("station_id"),
        "ts": ts_kst.strftime("%Y%m%d%H%M"),
    }

    print("  [초단기] API 요청 준비:")
    print(f"    - URL: {url_tmpl}")
    print(f"    - Params: {params}")
    print(f"    - Retry 설정: max={retry}, backoff={backoff}s")

    fields = {"T1H": 0.0, "REH": 0.0, "WSD": 0.0, "PTY": 0}
    for attempt in range(retry):
        try:
            print(f"  [초단기] API 호출 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url_tmpl, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            print(f"  [초단기] ✓ API 응답 수신: {data}")
            # API 스키마에 맞춰 파싱
            fields["T1H"] = float(data.get("T1H", 0.0))
            fields["REH"] = float(data.get("REH", 0.0))
            fields["WSD"] = float(data.get("WSD", 0.0))
            fields["PTY"] = int(data.get("PTY", 0))
            print(f"  [초단기] ✓ 파싱 완료: T1H={fields['T1H']}°C, REH={fields['REH']}%, WSD={fields['WSD']}m/s, PTY={fields['PTY']}")
            break
        except Exception as e:
            print(f"  [초단기] ✗ API 호출 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [초단기] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)

    # 캐시 fallback
    if not _is_valid(fields):
        print(f"  [초단기] ⚠️  데이터 유효하지 않음: {fields}")
        if _CACHE.get("data") is not None:
            cached = _CACHE["data"]
            cached_fields = {
                "T1H": cached.T1H_degC,
                "REH": cached.REH_pct,
                "WSD": cached.WSD_ms,
                "PTY": cached.PTY_code,
            }
            print(f"  [초단기] → 캐시 사용: {cached_fields}")
            return cached_fields
        print(f"  [초단기] → 기본값 사용: {fields}")

    return fields


def _is_valid(fields: Dict) -> bool:
    """기상 데이터 유효성 검사."""
    for k, v in fields.items():
        if k == "PTY":
            continue
        if not np.isfinite(v) or v < 0:
            return False
    return True


def _precipitation_to_pty(precip_rate_mmph: float) -> int:
    """강수율을 PTY 코드로 변환.
    
    Args:
        precip_rate_mmph: 강수율 [mm/h]
    
    Returns:
        PTY 코드 (0: 없음, 1: 비, 4: 소나기)
    """
    if precip_rate_mmph <= 0.1:
        return 0  # 없음
    elif precip_rate_mmph < 5.0:
        return 1  # 비
    else:
        return 4  # 소나기


def _fetch_solcast(
    ts_kst: pd.Timestamp,
    cfg: Dict,
    sun: Optional[SunPosition] = None
) -> WeatherData:
    """Solcast API에서 일사량 및 기상 데이터 조회.
    
    Args:
        ts_kst: 조회 시각 (KST)
        cfg: 설정
        sun: 태양 위치 (사용 안 함, Solcast가 직접 제공)
    
    Returns:
        WeatherData: GHI, DNI, DHI, 기온, 습도, 풍속, 강수
    """
    print("  [SOLCAST] API 호출 시작...")
    
    solcast_cfg = cfg.get("solcast", {})
    site_cfg = cfg.get("site", {})
    
    # API 키 (환경변수 우선)
    api_key = os.environ.get("SOLCAST_API_KEY", "")
    if not api_key:
        api_key = solcast_cfg.get("api_key", "")
    
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("SOLCAST_API_KEY not found in env or config")
    
    # 엔드포인트
    url = solcast_cfg["endpoints"]["live"]
    
    # 파라미터
    params = {
        "latitude": site_cfg["lat"],
        "longitude": site_cfg["lon"],
        "output_parameters": ",".join(solcast_cfg["output_parameters"]),
        "format": "json"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    timeout = float(solcast_cfg.get("timeout_s", 10))
    retry = int(solcast_cfg.get("retry", {}).get("max", 3))
    backoff = float(solcast_cfg.get("retry", {}).get("backoff", 1.0))
    
    print(f"  [SOLCAST] URL: {url}")
    print(f"  [SOLCAST] Params: lat={params['latitude']}, lon={params['longitude']}")
    
    data = None
    for attempt in range(retry):
        try:
            print(f"  [SOLCAST] API 호출 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            print("  [SOLCAST] ✓ API 응답 수신")
            
            # 최신 데이터 추출
            if "estimated_actuals" in result and len(result["estimated_actuals"]) > 0:
                data = result["estimated_actuals"][0]
                print(f"  [SOLCAST] ✓ 데이터 파싱 완료: {data}")
                break
            else:
                raise ValueError("No estimated_actuals in response")
                
        except Exception as e:
            print(f"  [SOLCAST] ✗ API 호출 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [SOLCAST] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)
    
    # 데이터 추출 실패 시 캐시 fallback
    if data is None:
        print("  [SOLCAST] ⚠️  API 호출 실패, 캐시 확인...")
        if _CACHE.get("data") is not None:
            print("  [SOLCAST] → 캐시 사용")
            return _CACHE["data"]
        else:
            raise RuntimeError("Solcast API failed and no cache available")
    
    # WeatherData 생성
    weather_data = WeatherData(
        ts=ts_kst,
        ghi_wh_per_m2=float(data.get("ghi", 0.0)),
        T1H_degC=float(data.get("air_temp", 20.0)),
        REH_pct=float(data.get("relative_humidity", 50.0)),
        WSD_ms=float(data.get("wind_speed_10m", 0.0)),
        PTY_code=_precipitation_to_pty(float(data.get("precipitation_rate", 0.0)))
    )
    
    print("  [SOLCAST] WeatherData 생성 완료:")
    print(f"    - GHI: {weather_data.ghi_wh_per_m2:.2f} W/m²")
    print(f"    - 기온: {weather_data.T1H_degC:.1f}°C")
    print(f"    - 습도: {weather_data.REH_pct:.1f}%")
    print(f"    - 풍속: {weather_data.WSD_ms:.1f} m/s")
    print(f"    - 강수형태: {weather_data.PTY_code}")
    
    return weather_data


def _fetch_openweathermap_solar(
    ts_kst: pd.Timestamp,
    cfg: Dict,
    sun: Optional[SunPosition] = None
) -> WeatherData:
    """OpenWeatherMap Solar Irradiance API에서 일사량 데이터 조회.
    
    Args:
        ts_kst: 조회 시각 (KST)
        cfg: 설정
        sun: 태양 위치 (사용 안 함, Solar API가 직접 제공)
    
    Returns:
        WeatherData: GHI, 기온, 습도, 풍속
    """
    print("  [OWM_SOLAR] Solar Irradiance API 호출 시작...")
    
    owm_cfg = cfg.get("openweathermap_solar", {})
    site_cfg = cfg.get("site", {})
    
    # API 키 (환경변수 우선)
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
    if not api_key:
        api_key = owm_cfg.get("api_key", "")
    
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("OPENWEATHERMAP_API_KEY not found in env or config")
    
    # 1. Solar Irradiance API 호출
    url = owm_cfg["endpoints"]["solar_irradiance"]
    date_str = ts_kst.strftime("%Y-%m-%d")
    
    params = {
        "lat": site_cfg["lat"],
        "lon": site_cfg["lon"],
        "date": date_str,
        "interval": owm_cfg.get("interval", "1h"),
        "appid": api_key,
    }
    
    timeout = float(owm_cfg.get("timeout_s", 10))
    retry = int(owm_cfg.get("retry", {}).get("max", 3))
    backoff = float(owm_cfg.get("retry", {}).get("backoff", 1.0))
    
    print(f"  [OWM_SOLAR] URL: {url}")
    print(f"  [OWM_SOLAR] Params: lat={params['lat']}, lon={params['lon']}, date={date_str}")
    
    solar_data = None
    for attempt in range(retry):
        try:
            print(f"  [OWM_SOLAR] API 호출 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            solar_data = resp.json()
            print("  [OWM_SOLAR] ✓ API 응답 수신")
            break
        except Exception as e:
            print(f"  [OWM_SOLAR] ✗ 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [OWM_SOLAR] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)
    
    if solar_data is None:
        # 캐시 fallback
        print("  [OWM_SOLAR] ⚠️  API 호출 실패, 캐시 확인...")
        if _CACHE.get("data") is not None:
            print("  [OWM_SOLAR] → 캐시 사용")
            return _CACHE["data"]
        else:
            raise RuntimeError("OpenWeatherMap Solar API failed and no cache available")
    
    # 2. 현재 시간대 interval 찾기
    current_hour = ts_kst.hour
    current_interval = None
    for interval in solar_data.get("intervals", []):
        start_hour = int(interval["start"].split(":")[0])
        if start_hour == current_hour:
            current_interval = interval
            break
    
    if current_interval is None:
        # 가장 가까운 interval 사용
        print(f"  [OWM_SOLAR] ⚠️  정확한 시간대를 찾을 수 없음, 첫 번째 interval 사용")
        current_interval = solar_data["intervals"][0] if solar_data.get("intervals") else {}
    
    # 3. Cloudy Sky 또는 Clear Sky 모델 선택
    use_cloudy = owm_cfg.get("use_cloudy_sky", True)
    sky_model = "cloudy_sky" if use_cloudy else "clear_sky"
    
    irradiation = current_interval.get("irradiation", {}).get(sky_model, {})
    ghi = float(irradiation.get("ghi", 0.0))
    
    print(f"  [OWM_SOLAR] 모델: {sky_model}")
    print(f"  [OWM_SOLAR] GHI: {ghi:.2f} Wh/m²")
    
    # 4. 기상 데이터 조회 (옵션)
    weather_source = owm_cfg.get("weather_source", "dummy")
    
    if weather_source == "current_weather":
        # Current Weather API 호출
        print("  [OWM_SOLAR] Current Weather API 호출...")
        weather_url = owm_cfg["endpoints"]["current_weather"]
        weather_params = {
            "lat": site_cfg["lat"],
            "lon": site_cfg["lon"],
            "appid": api_key,
            "units": "metric"
        }
        
        try:
            resp = requests.get(weather_url, params=weather_params, timeout=timeout)
            resp.raise_for_status()
            weather_data = resp.json()
            
            temp = float(weather_data["main"]["temp"])
            humidity = float(weather_data["main"]["humidity"])
            wind_speed = float(weather_data["wind"]["speed"])
            
            # 강수형태
            weather_main = weather_data["weather"][0]["main"]
            pty_code = 0
            if weather_main == "Rain":
                pty_code = 1
            elif weather_main == "Snow":
                pty_code = 3
            elif weather_main == "Drizzle":
                pty_code = 4
            
            print(f"  [OWM_SOLAR] ✓ 기상: temp={temp}°C, humidity={humidity}%, wind={wind_speed}m/s")
            
        except Exception as e:
            print(f"  [OWM_SOLAR] ⚠️  Current Weather 실패: {e}, 더미 사용")
            temp, humidity, wind_speed, pty_code = 20.0, 50.0, 2.0, 0
    
    else:
        # 더미 데이터
        print("  [OWM_SOLAR] 더미 기상 데이터 사용")
        temp, humidity, wind_speed, pty_code = 20.0, 50.0, 2.0, 0
    
    # WeatherData 생성
    result = WeatherData(
        ts=ts_kst,
        ghi_wh_per_m2=ghi,
        T1H_degC=temp,
        REH_pct=humidity,
        WSD_ms=wind_speed,
        PTY_code=pty_code,
    )
    
    print("  [OWM_SOLAR] WeatherData 생성 완료:")
    print(f"    - GHI: {result.ghi_wh_per_m2:.2f} W/m²")
    print(f"    - 기온: {result.T1H_degC:.1f}°C")
    print(f"    - 습도: {result.REH_pct:.1f}%")
    print(f"    - 풍속: {result.WSD_ms:.1f} m/s")
    print(f"    - 강수형태: {result.PTY_code}")
    
    return result


def fetch_solcast_irradiance(
    ts_kst: pd.Timestamp,
    cfg: Dict
) -> IrradianceData:
    """Solcast API에서 DNI/DHI 직접 가져오기.
    
    Args:
        ts_kst: 조회 시각 (KST)
        cfg: 설정
    
    Returns:
        IrradianceData: DNI, DHI [Wh/m²]
    
    Notes:
        - Solcast는 DNI/DHI를 직접 제공하므로 decompose.split_ghi() 불필요
        - GHI 분해 오차 없음 (±10-15% 오차 제거)
        - 캐싱 적용 (API 호출 절약)
    """
    print("  [SOLCAST_IRR] DNI/DHI 조회 시작...")
    
    # 캐싱 체크
    cache_ttl = cfg.get("solcast", {}).get("cache_ttl_s", 300)
    if _IRRADIANCE_CACHE.get("ts") is not None and _IRRADIANCE_CACHE.get("data") is not None:
        elapsed = (ts_kst - _IRRADIANCE_CACHE["ts"]).total_seconds()
        if elapsed < cache_ttl:
            print(f"  [SOLCAST_IRR] ✓ 캐시 사용 (경과: {elapsed:.0f}s < TTL: {cache_ttl}s)")
            return _IRRADIANCE_CACHE["data"]
    
    solcast_cfg = cfg.get("solcast", {})
    site_cfg = cfg.get("site", {})
    
    # API 키 (환경변수 우선)
    api_key = os.environ.get("SOLCAST_API_KEY", "")
    if not api_key:
        api_key = solcast_cfg.get("api_key", "")
    
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("SOLCAST_API_KEY not found in env or config")
    
    url = solcast_cfg["endpoints"]["live"]
    
    params = {
        "latitude": site_cfg["lat"],
        "longitude": site_cfg["lon"],
        "output_parameters": "dni,dhi",
        "format": "json"
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = float(solcast_cfg.get("timeout_s", 10))
    retry = int(solcast_cfg.get("retry", {}).get("max", 3))
    backoff = float(solcast_cfg.get("retry", {}).get("backoff", 1.0))
    
    print(f"  [SOLCAST_IRR] API 호출: lat={params['latitude']}, lon={params['longitude']}")
    
    data = None
    for attempt in range(retry):
        try:
            print(f"  [SOLCAST_IRR] 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            
            if "estimated_actuals" in result and len(result["estimated_actuals"]) > 0:
                data = result["estimated_actuals"][0]
                print(f"  [SOLCAST_IRR] ✓ 데이터 수신: DNI={data.get('dni')}, DHI={data.get('dhi')}")
                break
            else:
                raise ValueError("No estimated_actuals in response")
                
        except Exception as e:
            print(f"  [SOLCAST_IRR] ✗ 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [SOLCAST_IRR] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)
    
    # 실패 시 캐시 fallback
    if data is None:
        print("  [SOLCAST_IRR] ⚠️  API 호출 실패, 캐시 확인...")
        if _IRRADIANCE_CACHE.get("data") is not None:
            print("  [SOLCAST_IRR] → 캐시 사용")
            return _IRRADIANCE_CACHE["data"]
        else:
            raise RuntimeError("Solcast API failed and no irradiance cache available")
    
    irradiance_data = IrradianceData(
        dni_wh_per_m2=float(data.get("dni", 0.0)),
        dhi_wh_per_m2=float(data.get("dhi", 0.0))
    )
    
    print(f"  [SOLCAST_IRR] ✓ IrradianceData: DNI={irradiance_data.dni_wh_per_m2:.2f}, DHI={irradiance_data.dhi_wh_per_m2:.2f} Wh/m²")
    
    # 캐시 업데이트
    _IRRADIANCE_CACHE["ts"] = ts_kst
    _IRRADIANCE_CACHE["data"] = irradiance_data
    
    return irradiance_data


def fetch_openweathermap_solar_irradiance(
    ts_kst: pd.Timestamp,
    cfg: Dict
) -> IrradianceData:
    """OpenWeatherMap Solar API에서 DNI/DHI 직접 조회.
    
    Args:
        ts_kst: 조회 시각 (KST)
        cfg: 설정
    
    Returns:
        IrradianceData: DNI, DHI [Wh/m²]
    
    Notes:
        - OpenWeatherMap Solar API는 DNI/DHI를 직접 제공
        - GHI 분해 불필요 (Solcast와 동일)
        - 캐싱 적용
    """
    print("  [OWM_SOLAR_IRR] DNI/DHI 조회 시작...")
    
    # 캐싱 체크
    cache_ttl = cfg.get("openweathermap_solar", {}).get("cache_ttl_s", 300)
    if _IRRADIANCE_CACHE.get("ts") is not None and _IRRADIANCE_CACHE.get("data") is not None:
        elapsed = (ts_kst - _IRRADIANCE_CACHE["ts"]).total_seconds()
        if elapsed < cache_ttl:
            print(f"  [OWM_SOLAR_IRR] ✓ 캐시 사용 (경과: {elapsed:.0f}s < TTL: {cache_ttl}s)")
            return _IRRADIANCE_CACHE["data"]
    
    owm_cfg = cfg.get("openweathermap_solar", {})
    site_cfg = cfg.get("site", {})
    
    # API 키
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
    if not api_key:
        api_key = owm_cfg.get("api_key", "")
    
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("OPENWEATHERMAP_API_KEY not found in env or config")
    
    url = owm_cfg["endpoints"]["solar_irradiance"]
    date_str = ts_kst.strftime("%Y-%m-%d")
    
    params = {
        "lat": site_cfg["lat"],
        "lon": site_cfg["lon"],
        "date": date_str,
        "interval": owm_cfg.get("interval", "1h"),
        "appid": api_key,
    }
    
    timeout = float(owm_cfg.get("timeout_s", 10))
    retry = int(owm_cfg.get("retry", {}).get("max", 3))
    backoff = float(owm_cfg.get("retry", {}).get("backoff", 1.0))
    
    print(f"  [OWM_SOLAR_IRR] API 호출: lat={params['lat']}, lon={params['lon']}, date={date_str}")
    
    solar_data = None
    for attempt in range(retry):
        try:
            print(f"  [OWM_SOLAR_IRR] 시도 {attempt + 1}/{retry}...")
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            solar_data = resp.json()
            print("  [OWM_SOLAR_IRR] ✓ 데이터 수신")
            break
        except Exception as e:
            print(f"  [OWM_SOLAR_IRR] ✗ 실패 ({attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                sleep_time = backoff * (attempt + 1)
                print(f"  [OWM_SOLAR_IRR] {sleep_time}초 후 재시도...")
                time.sleep(sleep_time)
    
    if solar_data is None:
        print("  [OWM_SOLAR_IRR] ⚠️  API 호출 실패, 캐시 확인...")
        if _IRRADIANCE_CACHE.get("data") is not None:
            print("  [OWM_SOLAR_IRR] → 캐시 사용")
            return _IRRADIANCE_CACHE["data"]
        else:
            raise RuntimeError("OpenWeatherMap Solar API failed and no irradiance cache available")
    
    # 현재 시간대 interval 찾기
    current_hour = ts_kst.hour
    current_interval = None
    for interval in solar_data.get("intervals", []):
        start_hour = int(interval["start"].split(":")[0])
        if start_hour == current_hour:
            current_interval = interval
            break
    
    if current_interval is None:
        print(f"  [OWM_SOLAR_IRR] ⚠️  정확한 시간대를 찾을 수 없음, 첫 번째 interval 사용")
        current_interval = solar_data["intervals"][0] if solar_data.get("intervals") else {}
    
    # Cloudy Sky 또는 Clear Sky 모델 선택
    use_cloudy = owm_cfg.get("use_cloudy_sky", True)
    sky_model = "cloudy_sky" if use_cloudy else "clear_sky"
    
    irradiation = current_interval.get("irradiation", {}).get(sky_model, {})
    dni = float(irradiation.get("dni", 0.0))
    dhi = float(irradiation.get("dhi", 0.0))
    
    irradiance_data = IrradianceData(
        dni_wh_per_m2=dni,
        dhi_wh_per_m2=dhi
    )
    
    print(f"  [OWM_SOLAR_IRR] ✓ IrradianceData: DNI={dni:.2f}, DHI={dhi:.2f} Wh/m²")
    
    # 캐시 업데이트
    _IRRADIANCE_CACHE["ts"] = ts_kst
    _IRRADIANCE_CACHE["data"] = irradiance_data
    
    return irradiance_data
