from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from .types import RuntimeArtifacts, SafetyConfig
from .weather import fetch_weather_data
from .solar import compute_sun_position
from .decompose import split_ghi
from .state import build_pyg_data
from .inference import (
    load_checkpoint,
    load_surrogate,
    load_edge_structure,
    policy_inference,
    surrogate_inference,
)
from .safety import rate_limit, angle_clip, hard_override
from .control_io import build_bus
from .logging_utils import JsonlLogger

# GNN Policy import 추가 경로
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..src.policy import GNNPolicy  # noqa: E402
from ..src.utils import get_device, resolve_path  # noqa: E402


@dataclass
class RuntimeCache:
    """Runtime 실행 중 캐시 데이터."""
    # 모듈 좌표
    modules_xyz: Optional[np.ndarray] = None  # [N, 3]
    # 현재 각도
    theta: Optional[np.ndarray] = None  # [N]
    # 그래프 구조
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    # 모델
    policy: Optional[torch.nn.Module] = None
    surrogate: Optional[torch.nn.Module] = None
    # 매핑 행렬
    B: Optional[any] = None
    M: Optional[any] = None
    # 액션 bins
    bins: Optional[np.ndarray] = None


class RealTimeController:
    """실시간 제어 오케스트레이터."""

    def __init__(self, cfg_path: str) -> None:
        """초기화.

        Args:
            cfg_path: runtime.yaml 경로
        """
        # 설정 로드
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        device_pref = self.cfg.get("device", "auto")
        self.device = get_device(device_pref)
        print(f"[CONTROLLER] 사용 디바이스: {self.device}")

        self.tz = self.cfg["site"]["timezone"]
        self.logger = JsonlLogger(Path("src/rl/outputs/runtime_logs.jsonl"))
        self.cache = RuntimeCache()
        self.bus = build_bus(self.cfg)

        # 아티팩트 로드
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """아티팩트 로드 (모델, 행렬, 그래프 구조 등)."""
        arts: RuntimeArtifacts = self.cfg["artifacts"]

        # 1. 모듈 좌표 + 초기 각도 (CSV)
        modules_df = pd.read_csv(arts["modules_csv_path"])
        # 예상 형식: id,x,y,z,theta_init
        self.cache.modules_xyz = modules_df[["x", "y", "z"]].values.astype(np.float32)
        self.cache.theta = modules_df["theta_init"].values.astype(np.float32)

        N = self.cache.modules_xyz.shape[0]
        print(f"✓ 모듈 데이터 로드: {N}개 모듈")

        # 2. 그래프 구조
        self.cache.edge_index, self.cache.edge_attr = load_edge_structure(
            arts["edge_index_path"],
            arts["edge_attr_path"]
        )
        print(f"✓ 그래프 구조 로드: {self.cache.edge_index.size(1)}개 엣지")

        # 3. 체크포인트 로드 (policy + B + M + bins)
        try:
            print("체크포인트 로드 중...")
            checkpoint = load_checkpoint(arts["policy_checkpoint_path"])

            # B, M 행렬
            self.cache.B = checkpoint["B"]
            self.cache.M = checkpoint["M"]
            S = self.cache.M.shape[0]  # 섹터 개수
            print(f"✓ 매핑 행렬 로드: B{self.cache.B.shape}, M{self.cache.M.shape}")

            # Bins
            self.cache.bins = np.array(checkpoint["bins"], dtype=np.float32)
            B_bins = len(self.cache.bins)
            print(f"✓ Bins: {self.cache.bins}")

            # Policy 복원
            print("GNNPolicy 복원 중...")
            pretrained_model_path = (
                checkpoint.get("policy_backbone_path")
                or arts.get("policy_backbone_path")
            )

            if not pretrained_model_path:
                raise RuntimeError("정책 백본 경로를 찾을 수 없습니다. runtime.yaml에 artifacts.policy_backbone_path를 지정하세요.")

            backbone_path = resolve_path(pretrained_model_path)
            if not backbone_path.exists():
                raise FileNotFoundError(f"정책 백본 파일이 존재하지 않습니다: {backbone_path}")

            # GNNPolicy 초기화
            self.cache.policy = GNNPolicy(
                S=S,
                B=B_bins,
                pretrained_model_path=str(backbone_path),
                freeze_backbone=True,
                device=self.device,
            )

            # state_dict 로드
            self.cache.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.cache.policy.to(self.device)
            self.cache.policy.eval()
            print(f"✓ 정책 모델 로드 완료: S={S}, B={B_bins}")
            print(f"✓ 정책 모델 백본: {backbone_path}")
            print(f"✓ 정책 모델 체크포인트: {arts["policy_checkpoint_path"]}")

        except Exception as e:
            print(f"⚠️  체크포인트 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            print("    Fallback: 항등 행렬 및 기본 bins 사용")
            self.cache.policy = None
            from scipy.sparse import eye as _eye
            self.cache.B = _eye(N, N, format="csr")
            self.cache.M = _eye(N, N, format="csr")
            self.cache.bins = np.array(
                self.cfg.get("bins", [-5, -3, -1, 0, 1, 3, 5]),
                dtype=np.float32
            )

        # 4. 대리모델 (optional)
        try:
            self.cache.surrogate = load_surrogate(arts.get("surrogate_path"), self.device)
            if self.cache.surrogate is not None:
                print(f"✓ 대리모델 로드: {arts['surrogate_path']}")
        except Exception:
            self.cache.surrogate = None

    def run_once(self) -> None:
        """1회 제어 실행."""
        print("\n" + "#"*70)
        print("# 실시간 제어 사이클 시작")
        print("#"*70)

        cfg = self.cfg
        site = cfg["site"]
        ts_kst = pd.Timestamp.now(tz=site["timezone"])
        print(f"[CONTROLLER] 현재 시각 (KST): {ts_kst}")
        alt_m = site['altitude_m']
        print(f"[CONTROLLER] 사이트 정보: lat={site['lat']}, lon={site['lon']}, alt={alt_m}m")

        # 1. 태양 위치 계산 (일사량 계산에 필요)
        print("\n[CONTROLLER] STEP 1/11: 태양 위치 계산")
        sun = compute_sun_position(
            ts_kst,
            lat=site["lat"],
            lon=site["lon"],
            alt_m=site["altitude_m"]
        )
        print("[CONTROLLER] ✓ 태양 위치 계산 완료")

        # 2. 기상 데이터 조회 (태양 위치 정보 전달)
        print("\n[CONTROLLER] STEP 2/11: 기상 데이터 조회")
        weather = fetch_weather_data(ts_kst, cfg, sun)
        print("[CONTROLLER] ✓ 기상 데이터 조회 완료")

        # 3. GHI → DNI/DHI 분해
        print("\n[CONTROLLER] STEP 3/11: GHI → DNI/DHI 분해")
        print(f"[CONTROLLER] 입력: GHI={weather.ghi_wh_per_m2:.2f} Wh/m², θ_z={sun.theta_z_deg:.2f}°")
        irradiance = split_ghi(
            weather.ghi_wh_per_m2,
            sun.theta_z_deg,
            method=cfg.get("decomposition", {}).get("method", "erbs"),
        )
        dni = irradiance.dni_wh_per_m2
        dhi = irradiance.dhi_wh_per_m2
        print(f"[CONTROLLER] ✓ 분해 완료: DNI={dni:.2f}, DHI={dhi:.2f} Wh/m²")

        # 4. PyG Data 생성
        print("\n[CONTROLLER] STEP 4/11: PyG Data 생성")
        n_modules = self.cache.modules_xyz.shape[0]
        n_edges = self.cache.edge_index.size(1)
        print(f"[CONTROLLER] 모듈 개수: {n_modules}, 엣지 개수: {n_edges}")
        data = build_pyg_data(
            weather=weather,
            sun=sun,
            irradiance=irradiance,
            modules_xyz=self.cache.modules_xyz,
            theta=self.cache.theta,
            edge_index=self.cache.edge_index,
            edge_attr=self.cache.edge_attr,
        )
        print("[CONTROLLER] ✓ PyG Data 생성 완료")

        # 5. 정책 추론
        print("\n[CONTROLLER] STEP 5/11: 정책 추론")
        if self.cache.policy is None:
            # Fallback: 0 액션
            S = self.cache.M.shape[0]
            z = np.zeros((S,), dtype=np.float32)
            print(f"[CONTROLLER] ⚠️  정책이 로드되지 않음 → 0 액션 사용 (섹터 개수: {S})")
        else:
            z = policy_inference(data, self.cache.policy, self.cache.bins, self.device)
            z_mean = np.mean(z)
            z_std = np.std(z)
            print(f"[CONTROLLER] ✓ 정책 추론 완료: z shape={z.shape}, mean={z_mean:.2f}, std={z_std:.2f}")

        # 6. 매핑: z → z_sm → a
        print("\n[CONTROLLER] STEP 6/11: 액션 매핑 (z → z_sm → a)")
        print(f"[CONTROLLER] z (섹터 액션): shape={z.shape}, range=[{z.min():.2f}, {z.max():.2f}]")
        z_sm = (self.cache.M @ z.reshape(-1, 1)).ravel().astype(np.float32)
        print(f"[CONTROLLER] z_sm (M @ z): shape={z_sm.shape}, range=[{z_sm.min():.2f}, {z_sm.max():.2f}]")
        a = (self.cache.B @ z_sm.reshape(-1, 1)).ravel().astype(np.float32)
        print(f"[CONTROLLER] a (모듈 액션): shape={a.shape}, range=[{a.min():.2f}, {a.max():.2f}]")
        b_shape = self.cache.B.shape
        m_shape = self.cache.M.shape
        print(f"[CONTROLLER] ✓ 매핑 완료: B{b_shape} @ M{m_shape}")

        # 7. 안전 제약
        print("\n[CONTROLLER] STEP 7/11: 안전 제약 적용")
        safety_cfg: SafetyConfig = cfg.get("safety", {})
        max_rate = float(safety_cfg.get("max_rate_deg_per_step", 5.0))
        angle_bounds = safety_cfg.get("angle_bounds", [0.0, 90.0])
        theta_mean = np.mean(self.cache.theta)
        theta_min = self.cache.theta.min()
        theta_max = self.cache.theta.max()
        print(f"[CONTROLLER] 안전 설정: max_rate={max_rate}°/step, bounds=[{angle_bounds[0]}, {angle_bounds[1]}]°")
        print(f"[CONTROLLER] 현재 각도: mean={theta_mean:.2f}°, range=[{theta_min:.2f}, {theta_max:.2f}]°")

        theta_next = rate_limit(
            self.cache.theta,
            a,
            max_rate=max_rate
        )
        tn_mean = np.mean(theta_next)
        tn_min = theta_next.min()
        tn_max = theta_next.max()
        print(f"[CONTROLLER] Rate limit 후: mean={tn_mean:.2f}°, range=[{tn_min:.2f}, {tn_max:.2f}]°")

        theta_next = angle_clip(
            theta_next,
            float(angle_bounds[0]),
            float(angle_bounds[1]),
        )
        tn2_mean = np.mean(theta_next)
        tn2_min = theta_next.min()
        tn2_max = theta_next.max()
        print(f"[CONTROLLER] Angle clip 후: mean={tn2_mean:.2f}°, range=[{tn2_min:.2f}, {tn2_max:.2f}]°")
        print("[CONTROLLER] ✓ 안전 제약 완료")

        # 8. 하드 오버라이드 (기상 조건)
        print("\n[CONTROLLER] STEP 8/11: 하드 오버라이드 (기상 조건)")
        print(f"[CONTROLLER] 기상 조건: PTY={weather.PTY_code}, WSD={weather.WSD_ms:.1f}m/s")
        theta_final = hard_override(theta_next, weather, safety_cfg)
        if np.array_equal(theta_next, theta_final):
            print("[CONTROLLER] → 오버라이드 없음 (정상 제어)")
        else:
            n_changed = np.sum(theta_next != theta_final)
            tf_mean = np.mean(theta_final)
            print(f"[CONTROLLER] ⚠️  오버라이드 적용됨: {n_changed}개 모듈 변경")
            print(f"[CONTROLLER] → 최종 각도: mean={tf_mean:.2f}°")

        # 9. 제어 신호 송신
        print("\n[CONTROLLER] STEP 9/11: 제어 신호 송신")
        tf_min = theta_final.min()
        tf_max = theta_final.max()
        print(f"[CONTROLLER] 송신할 각도: shape={theta_final.shape}, range=[{tf_min:.2f}, {tf_max:.2f}]°")
        ok = self.bus.send(theta_final)
        status_icon = '✓' if ok else '✗'
        status_text = '성공' if ok else '실패'
        print(f"[CONTROLLER] {status_icon} 송신 {status_text}")

        # 10. 상태 갱신
        print("\n[CONTROLLER] STEP 10/11: 상태 갱신")
        old_theta_mean = np.mean(self.cache.theta)
        self.cache.theta = theta_final
        new_theta_mean = np.mean(theta_final)
        print(f"[CONTROLLER] ✓ 상태 갱신 완료: {old_theta_mean:.2f}° → {new_theta_mean:.2f}°")

        # 11. 로깅
        print("\n[CONTROLLER] STEP 11/11: 로깅")
        metrics = surrogate_inference(data, self.cache.surrogate, self.device)
        log_data = {
            "ts": str(ts_kst),
            "ghi": float(weather.ghi_wh_per_m2),
            "dni": float(irradiance.dni_wh_per_m2),
            "dhi": float(irradiance.dhi_wh_per_m2),
            "sun_alt": float(sun.sun_alt_deg),
            "sun_azi": float(sun.sun_azi_deg),
            "T1H": float(weather.T1H_degC),
            "REH": float(weather.REH_pct),
            "z_mean": float(np.mean(z)),
            "a_mean": float(np.mean(a)),
            "theta_mean": float(np.mean(theta_final)),
            "send_ok": bool(ok),
            **{f"metric_{k}": v for k, v in metrics.items()},
        }
        self.logger.log(log_data)
        print(f"[CONTROLLER] ✓ 로그 기록 완료: {len(log_data)}개 필드")

        print(f"\n{'='*70}")
        print(f"[요약] [{ts_kst.strftime('%Y-%m-%d %H:%M')}] "
              f"DNI={irradiance.dni_wh_per_m2:.0f} Wh/m², "
              f"θ_mean={np.mean(theta_final):.1f}°, "
              f"송신={'OK' if ok else 'FAIL'}")
        print(f"{'='*70}\n")

    def run_forever(self) -> None:
        """무한 루프 실행."""
        import time as _time

        interval = float(self.cfg.get("control", {}).get("send_interval_s", 10.0))
        print(f"🚀 Runtime 시작 (interval={interval}s)")

        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                # Fallback: 현재 각도 유지

            _time.sleep(interval)
