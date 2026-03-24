from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.runtime.weather import to_ghi
from src.rl.runtime.solar import compute_sun
from src.rl.runtime.decompose import split_ghi


def test_to_ghi_conversion():
    # 1 MJ/m^2·h = 277.78 Wh/m^2
    assert abs(to_ghi(1.0) - 277.78) < 0.5


def test_solar_position_basic_range():
    ts = pd.Timestamp("2025-06-21 12:00:00", tz="Asia/Seoul")
    sun = compute_sun(ts, lat=37.5, lon=127.0, alt_m=50)
    assert 0.0 <= sun.theta_z_deg <= 90.0
    assert -180.0 <= sun.sun_azi_deg <= 360.0


def test_decomposition_energy_relation():
    # Midday scenario
    ghi = 800.0
    theta_z = 30.0
    res = split_ghi(ghi, theta_z, method="erbs")
    cosz = max(np.cos(np.deg2rad(theta_z)), 0.0)
    lhs = res.dhi_wh_per_m2 + res.dni_wh_per_m2 * cosz
    assert abs(lhs - ghi) / max(ghi, 1e-6) < 0.5  # loose bound for model approx
