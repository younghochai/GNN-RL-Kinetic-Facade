from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import csr_matrix, load_npz


def load_checkpoint(path: str) -> dict:
    """체크포인트 로드 (policy + B + M + config).

    Args:
        path: checkpoint.pt 경로

    Returns:
        체크포인트 dict:
            - policy: GNNPolicy 모델
            - B: 매핑 행렬
            - M: 매핑 행렬
            - bins: 액션 bins
            - config: 기타 설정
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # 체크포인트 형식 확인
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        # 새 체크포인트 형식
        return {
            "policy_state_dict": checkpoint["policy_state_dict"],
            "policy_config": checkpoint.get("policy_config", {}),
            "B": checkpoint.get("B"),
            "M": checkpoint.get("M"),
            "bins": checkpoint.get("bins"),
            "max_rate": checkpoint.get("max_rate"),
            "angle_bounds": checkpoint.get("angle_bounds"),
        }
    else:
        # 이전 형식 (policy만 저장된 경우)
        raise ValueError(
            "체크포인트 형식이 올바르지 않습니다. "
            "checkpoint_final.pt 파일을 사용하세요."
        )


def load_surrogate(path: Optional[str], device: torch.device | str = "cpu") -> Optional[torch.nn.Module]:
    """대리모델 로드.

    Args:
        path: surrogate.pt 경로 (optional)
        device: 로드할 디바이스

    Returns:
        대리모델 (eval 모드) 또는 None
    """
    if device is None:
        device = "cpu"
    device = torch.device(device)

    if not path:
        return None
    try:
        model = torch.load(path, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(device)
            return model
    except Exception:
        pass
    return None


def load_sparse_matrix(path: str) -> csr_matrix:
    """희소 행렬 로드 (B, M).

    Args:
        path: .npz 파일 경로

    Returns:
        csr_matrix
    """
    return load_npz(path)


def load_edge_structure(edge_index_path: str, edge_attr_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """그래프 엣지 구조 로드.

    Args:
        edge_index_path: edge_index.pt 경로
        edge_attr_path: edge_attr.pt 경로

    Returns:
        (edge_index, edge_attr)
    """
    edge_index = torch.load(edge_index_path, weights_only=True)
    edge_attr = torch.load(edge_attr_path, weights_only=True)
    return edge_index, edge_attr


@torch.no_grad()
def policy_inference(
    data: Data,
    policy: torch.nn.Module,
    bins: np.ndarray,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """GNN Policy 추론.

    Args:
        data: PyG Data 객체
        policy: GNNPolicy 모델
        bins: 액션 bins (예: [-5, -3, -1, 0, 1, 3, 5])

    Returns:
        z: 섹터 액션 [S] (각 섹터의 각도 변화량)
    """
    print("  [INFERENCE] 정책 추론 시작")
    print(f"  [INFERENCE] 입력 데이터: x={data.x.shape}, global_x={data.global_x.shape}")
    print(f"  [INFERENCE] Bins: {bins}")

    if device is None:
        device = "cpu"
    device = torch.device(device)

    policy.eval()

    # GNN Policy 추론
    print("  [INFERENCE] Policy 순전파 중...")
    # 원본을 보존하기 위해 복사본 사용
    data_device = data.clone()
    if not hasattr(data_device, "batch") or data_device.batch is None:
        data_device.batch = torch.zeros(data_device.x.size(0), dtype=torch.long)
    data_device = data_device.to(device)
    logits, _ = policy(data_device)  # (1, S, B)
    print(f"  [INFERENCE] ✓ Logits: shape={logits.shape}")

    # Softmax → 확률 분포
    probs = torch.softmax(logits, dim=-1)[0]  # (S, B)
    print(f"  [INFERENCE] ✓ Probs (softmax): shape={probs.shape}")
    print("  [INFERENCE]   - 첫 3개 섹터 확률 분포:")
    for i in range(min(3, probs.shape[0])):
        print(f"  [INFERENCE]     Sector {i}: {probs[i].detach().cpu().numpy()}")

    # 기댓값 계산 (확률 * bins)
    bins_tensor = torch.from_numpy(bins).float().to(device)
    z = (probs * bins_tensor[None, :]).sum(dim=1)  # (S,)
    print(f"  [INFERENCE] ✓ 기댓값 계산 완료: z shape={z.shape}")
    z_mean = z.mean()
    z_std = z.std()
    z_min = z.min()
    z_max = z.max()
    print(f"  [INFERENCE]   - z 통계: mean={z_mean:.4f}, std={z_std:.4f}, range=[{z_min:.4f}, {z_max:.4f}]")

    result = z.detach().cpu().numpy().astype(np.float32)
    print("  [INFERENCE] ✓ 추론 완료: 섹터 액션 반환")

    return result


@torch.no_grad()
def surrogate_inference(
    data: Data,
    surrogate: Optional[torch.nn.Module],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """대리모델 추론 (로깅용).

    Args:
        data: PyG Data 객체
        surrogate: 대리모델 (optional)

    Returns:
        {"field_rad": ..., "crowd_rad": ...} 또는 빈 dict
    """
    if device is None:
        device = "cpu"
    device = torch.device(device)

    if surrogate is None:
        return {}

    try:
        surrogate.eval()
        data_device = data.clone()
        if not hasattr(data_device, "batch") or data_device.batch is None:
            data_device.batch = torch.zeros(data_device.x.size(0), dtype=torch.long)
        data_device = data_device.to(device)
        out = surrogate(data_device)

        # 출력 형식 처리
        if isinstance(out, torch.Tensor):
            # (1, 2) → field, crowd
            return {
                "field_rad": float(out[0, 0].item()),
                "crowd_rad": float(out[0, 1].item()) if out.size(1) > 1 else 0.0,
            }
        elif isinstance(out, dict):
            return {k: float(v.item()) for k, v in out.items() if hasattr(v, "item")}
    except Exception:
        pass

    return {}
