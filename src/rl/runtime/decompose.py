from __future__ import annotations

import numpy as np
from pvlib.irradiance import erbs

from .types import IrradianceData


def split_ghi(ghi_wh: float, theta_z_deg: float, method: str = "erbs") -> IrradianceData:
    """GHI를 DNI/DHI로 분해.

    Args:
        ghi_wh: 전천일사 [Wh/m²]
        theta_z_deg: 태양 천정각 [도]
        method: 분해 모델 ("erbs" 또는 "dirint")

    Returns:
        IrradianceData: DNI, DHI

    Notes:
        - Erbs 모델: 통계적 경험 모델 (정확도 중간)
        - 야간/저일사 시 0 반환
    """
    print("  [DECOMPOSE] GHI 분해 시작")
    print(f"  [DECOMPOSE] 입력: GHI={ghi_wh:.2f} Wh/m², θ_z={theta_z_deg:.2f}°, method={method}")

    # 야간 또는 저일사 처리
    cos_theta = float(np.maximum(np.cos(np.deg2rad(theta_z_deg)), 0.0))
    print(f"  [DECOMPOSE] cos(θ_z) = {cos_theta:.4f}")

    if not np.isfinite(ghi_wh) or ghi_wh <= 1e-3 or cos_theta <= 1e-6:
        print("  [DECOMPOSE] ⚠️  야간 또는 저일사 조건 → DNI=0, DHI=0 반환")
        return IrradianceData(dni_wh_per_m2=0.0, dhi_wh_per_m2=0.0)

    ghi = float(np.clip(ghi_wh, 0.0, 2000.0))
    zenith = float(np.clip(theta_z_deg, 0.0, 90.0))
    print(f"  [DECOMPOSE] 클리핑 후: GHI={ghi:.2f}, zenith={zenith:.2f}°")

    if method.lower() == "erbs":
        # pvlib.erbs 사용
        print("  [DECOMPOSE] Erbs 모델 적용 중...")
        out = erbs(ghi=ghi, zenith=zenith, datetime_or_doy=1)
        print(f"  [DECOMPOSE] Erbs 출력: DHI={out['dhi']:.2f}, DNI={out['dni']:.2f}")
        dhi = float(np.clip(out["dhi"], 0.0, ghi))
        dni = float(np.clip((ghi - dhi) / max(cos_theta, 1e-6), 0.0, 1200.0))
        print(f"  [DECOMPOSE] 1차 클리핑: DNI={dni:.2f}, DHI={dhi:.2f}")
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'erbs'")

    # 물리적 클리핑
    dni = float(np.clip(dni, 0.0, 1500.0))
    dhi = float(np.clip(dhi, 0.0, ghi))
    print("  [DECOMPOSE] 최종 클리핑: DNI=[0, 1500], DHI=[0, GHI]")
    print(f"  [DECOMPOSE] ✓ 분해 완료: DNI={dni:.2f} Wh/m², DHI={dhi:.2f} Wh/m²")

    return IrradianceData(dni_wh_per_m2=dni, dhi_wh_per_m2=dhi)
