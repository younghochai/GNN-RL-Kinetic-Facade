from __future__ import annotations

"""
유틸리티 모음: 시드 고정, 경로 해석, 로깅, 체크포인트 입출력.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_device(device_pref: str = "auto") -> torch.device:
    if device_pref != "auto":
        return torch.device(device_pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def project_root() -> Path:
    """리포지토리 루트(이 파일에서 상위 상위 상위)를 반환."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path_str: str | os.PathLike[str]) -> Path:
    """상대 경로는 리포 루트 기준으로 해석한다."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class Checkpoint:
    policy_state: Dict[str, Any]
    value_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scaler_state: Optional[Dict[str, Any]] = None


def save_checkpoint(path: Path, ckpt: Checkpoint) -> None:
    path = resolve_path(path)
    ensure_dir(path.parent)
    torch.save({
        "policy": ckpt.policy_state,
        "value": ckpt.value_state,
        "optimizer": ckpt.optimizer_state,
        "scaler": ckpt.scaler_state,
    }, path)


def load_checkpoint(path: Path) -> Checkpoint:
    blob = torch.load(resolve_path(path), map_location="cpu")
    return Checkpoint(
        policy_state=blob.get("policy", {}),
        value_state=blob.get("value", {}),
        optimizer_state=blob.get("optimizer", {}),
        scaler_state=blob.get("scaler"),
    )


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path = resolve_path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
