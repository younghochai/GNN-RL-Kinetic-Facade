from __future__ import annotations

"""
평가 스크립트: 학습된 정책으로 1 에피소드 시뮬레이션 및 간단 리포트 저장.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

from .clustering import cluster_modules
from .datasets import InitialStateDataset
from .env import SectorEnv
from .mapping import build_B, build_M, sector_adjacency_from_labels
from .policy import MLPPolicy, build_features, MultiCategorical
from .surrogate import SurrogateModel
from .utils import ensure_dir, get_device, resolve_path


def load_cfg(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str | Path, checkpoint: str | None, device: str = "auto") -> None:
    cfg = load_cfg(config_path)
    dev = get_device(device)
    N = int(cfg.get("N_modules", 2064))
    S = int(cfg.get("S_sectors", 128))
    bins: List[float] = list(map(float, cfg.get("bins", [-5, -3, -1, 0, 1, 3, 5])))

    coords = np.load(resolve_path(cfg.get("coords_path", "src/rl/data/coords.npy")))
    labels = cluster_modules(coords, cfg)
    B = build_B(coords, labels, cfg)
    A_sec = sector_adjacency_from_labels(labels, coords)
    M = build_M(A_sec, alpha=float(cfg.get("alpha_sector_smooth", 0.2)))

    dataset = InitialStateDataset(cfg.get("s0_path", "src/rl/data/s0.parquet"), N)
    obs = dataset.get_item(0)
    surrogate = SurrogateModel(cfg.get("surrogate_path", None))
    env = SectorEnv(B=B, M=M, bins=bins, surrogate=surrogate, init_obs=obs, max_rate=float(cfg.get("max_rate", 5.0)), angle_bounds=tuple(map(float, cfg.get("angle_bounds", [0.0, 90.0]))), continuous=False)

    policy = MLPPolicy(input_dim=8, S=S, B=len(bins))
    if checkpoint:
        sd = torch.load(resolve_path(checkpoint), map_location=dev)
        policy.load_state_dict(sd)
    policy.to(dev)
    policy.eval()

    # 시뮬레이션 100 스텝
    obs = env.reset(obs)
    rewards = []
    for t in range(100):
        feat = torch.from_numpy(build_features(obs)).float().unsqueeze(0).to(dev)
        logits, _ = policy(feat)
        dist = MultiCategorical(logits=logits)
        action = torch.argmax(logits, dim=-1)[0].cpu().numpy()  # greedy
        obs, r, d, tr, info = env.step(action)
        rewards.append(r)

    out_dir = resolve_path("src/rl/outputs/eval")
    ensure_dir(out_dir)
    np.savetxt(out_dir / "rewards.txt", np.array(rewards))
    print(f"mean_reward={np.mean(rewards):.4f}")


