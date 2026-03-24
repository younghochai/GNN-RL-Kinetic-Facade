# flake8: noqa
from __future__ import annotations

"""
학습 스크립트: 시드 고정 → 좌표/클러스터링 → B,M → 데이터/대리모델 → Env → Rollout → PPO 업데이트.
"""
from pathlib import Path
import time
from dataclasses import dataclass
import argparse
from typing import List, Tuple

import numpy as np
import torch
import yaml

from .buffer import RolloutBuffer
from .clustering import cluster_modules
from .datasets import PyGDataset
from .env import SectorEnv
from .mapping import (
    build_B,
    build_M,
    sector_adjacency_from_labels,
    validate_constant_preservation,
    boundary_step_response,
)
from .policy import MLPPolicy, GNNPolicy, build_features, MultiCategorical
from .ppo import PPOAgent
from .surrogate import SurrogateModel
from .utils import ensure_dir, get_device, project_root, resolve_path, set_seed, write_json


@dataclass
class Cfg:
    seed: int
    S_sectors: int
    bins: List[float]
    tau: float
    k_nn: int  # 경계 판단용 k-NN
    alpha_sector_smooth: float
    max_rate: float
    angle_bounds: Tuple[float, float]
    weights: Tuple[float, float, float]
    gamma: float
    gae_lambda: float
    normalize_rewards: bool
    clip_range: float
    entropy_coef: float
    value_coef: float
    target_kl: float
    lr: float
    num_iterations: int
    rollout_steps: int
    minibatch_size: int
    epochs: int
    dataset_path: str
    pretrained_gnn_model_path_field: str
    pretrained_gnn_model_path_crowd: str
    max_traj_steps: int = 10
    use_interpolation: bool = False
    num_interpolations: int = 8


def load_cfg(path: str | Path) -> Cfg:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Cfg(
        seed=int(raw.get("seed", 7)),
        S_sectors=int(raw.get("S_sectors", 128)),
        bins=list(map(float, raw.get("bins", [-5, -3, -1, 0, 1, 3, 5]))),
        tau=float(raw.get("tau", 1.5)),
        k_nn=int(raw.get("k_nn", 8)),
        alpha_sector_smooth=float(raw.get("alpha_sector_smooth", 0.2)),
        max_rate=float(raw.get("max_rate", 5.0)),
        angle_bounds=tuple(map(float, raw.get("angle_bounds", [0.0, 90.0]))),
        weights=tuple(map(float, raw.get("weights", [1.0, 1.0, 0.01]))),
        gamma=float(raw.get("gamma", 0.99)),
        gae_lambda=float(raw.get("gae_lambda", 0.95)),
        normalize_rewards=bool(raw.get("normalize_rewards", True)),
        clip_range=float(raw.get("clip_range", 0.2)),
        entropy_coef=float(raw.get("entropy_coef", 0.01)),
        value_coef=float(raw.get("value_coef", 0.5)),
        target_kl=float(raw.get("target_kl", 0.02)),
        lr=float(raw.get("lr", 3e-4)),
        num_iterations=int(raw.get("num_iterations", 100)),
        rollout_steps=int(raw.get("rollout_steps", 4096)),
        minibatch_size=int(raw.get("minibatch_size", 512)),
        epochs=int(raw.get("epochs", 10)),
        dataset_path=str(raw.get("dataset_path", "src/gnn/datapyg_dataset_rl.pt")),
        pretrained_gnn_model_path_field=str(raw.get("pretrained_gnn_model_path_field", "src/rl/outputs/best_model_gin_with_class.pt")),
        pretrained_gnn_model_path_crowd=str(raw.get("pretrained_gnn_model_path_crowd", "src/rl/outputs/best_model_gin_with_class.pt")),
        max_traj_steps=int(raw.get("max_traj_steps", 10)),
        use_interpolation=bool(raw.get("use_interpolation", False)),
        num_interpolations=int(raw.get("num_interpolations", 8)),
    )


def prepare_dataset(cfg: Cfg) -> PyGDataset:
    """PyG dataset 파일 로드."""
    dataset = PyGDataset(cfg.dataset_path, use_interpolation=cfg.use_interpolation)
    return dataset


def collect_rollout(
    env: SectorEnv,
    policy: GNNPolicy,
    init_obs: object,
    dataset: PyGDataset,
    rollout_steps: int,
    num_interpolations: int = 0,
    verbose: bool = False,
) -> Tuple[RolloutBuffer, dict, object]:
    """롤아웃 수집 (done 처리 포함).
    
    Args:
        env: 환경
        policy: 정책 네트워크
        init_obs: 초기 관측
        dataset: 데이터셋 (새 에피소드 샘플링용)
        rollout_steps: 수집할 총 스텝 수
        verbose: 상세 로그 출력
    
    Returns:
        (buffer, stats, last_obs): 롤아웃 버퍼, 통계, 마지막 관측
    """
    buf = RolloutBuffer(capacity=rollout_steps)
    obs = env.reset(init_obs)
    policy.eval()
    policy_device = next(policy.parameters()).device
    
    # 통계
    episode_rewards = []
    episode_values = []
    episode_entropies = []
    num_episodes = 0
    
    with torch.no_grad():
        for t in range(rollout_steps):
            print(f"  Step {t:4d}:", end="\r")
            obs_device = obs.clone().to(policy_device)
            logits, value = policy(obs_device)
            
            multi = MultiCategorical(logits=logits)
            action_tensor = multi.sample()
            logp_tensor = multi.log_prob(action_tensor)
            entropy_tensor = multi.entropy()
            action = action_tensor[0].detach().cpu().numpy()
            logp = logp_tensor[0].item()
            entropy = entropy_tensor[0].item()
            value_s = float(value.item())
            
            obs_next, reward, done, truncated, info = env.step(action)
            
            buf.add(obs=obs, act=action, rew=reward, done=done, logp=logp, value=value_s)
            
            episode_rewards.append(reward)
            episode_values.append(value_s)
            episode_entropies.append(entropy)
            
            # 상세 로그 (첫 3개, 마지막 3개)
            if verbose and (t < 3 or t >= rollout_steps - 3):
                print(f"  Step {t:4d}:")
                print(f"    Action: {action[:5]}... | Reward: {reward:.6f} | "
                      f"Value: {value_s:.6f} | Entropy: {entropy:.4f}")
                print(f"    Done: {done}, Truncated: {truncated}")
            elif verbose and t == 3:
                print("  ...")
            
            # Done 처리: 새 에피소드 시작
            if done or truncated:
                num_episodes += 1
                # 새 에피소드 샘플링
                if dataset.use_interpolation:
                    episode = dataset.sample_episode(num_interpolations=num_interpolations)
                    obs = env.reset_episode(episode)
                else:
                    new_obs = dataset.get_item(np.random.randint(len(dataset)))
                    obs = env.reset(new_obs)
            else:
                obs = obs_next
    
    stats = {
        "total_reward": sum(episode_rewards),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_value": np.mean(episode_values),
        "mean_entropy": np.mean(episode_entropies),
        "num_episodes": num_episodes,
    }
    
    return buf, stats, obs  # 마지막 obs 반환


def main(config_path: str | Path, run_name: str | None, device: str = "auto") -> None:
    cfg = load_cfg(config_path)
    set_seed(cfg.seed)
    dev = get_device(device)
    print(f"  사용 장치: {dev}")
    torch.set_default_device(dev)

    # PyG dataset 로드
    print("\n" + "="*80)
    print("1. 데이터셋 로딩")
    print("="*80)
    dataset = prepare_dataset(cfg)
    init_obs = dataset.get_item(0)  # Data 객체
    N_modules = dataset.num_modules  # 데이터셋에서 자동 추출
    
    print(f"  초기 관측 (init_obs):")
    print(f"    - x.shape: {init_obs.x.shape} (노드 특징)")
    print(f"    - edge_index.shape: {init_obs.edge_index.shape}")
    print(f"    - global_x.shape: {init_obs.global_x.shape}")
    print(f"    - global_x 값: {init_obs.global_x.squeeze()}")
    print(f"    - y (target): {init_obs.y.item() if init_obs.y.numel() == 1 else init_obs.y.squeeze()}")

    # 좌표 추출 및 클러스터/매핑
    print("\n" + "="*80)
    print("2. 모듈-섹터 매핑 생성")
    print("="*80)
    coords = dataset.extract_coords()
    print(f"  좌표 추출: {coords.shape}")
    print(f"  좌표 범위: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
          f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
    labels = cluster_modules(coords, cfg)
    print(f"  클러스터링 파라미터: k_nn={cfg.k_nn}, tau={cfg.tau}")
    
    sector_counts = np.bincount(labels, minlength=cfg.S_sectors)
    non_empty = (sector_counts > 0).sum()
    empty_sectors = (sector_counts == 0).sum()
    
    print(f"  클러스터링 결과: {cfg.S_sectors}개 섹터")
    print(f"    - 비어있지 않은 섹터: {non_empty}/{cfg.S_sectors} ({non_empty/cfg.S_sectors*100:.1f}%)")
    print(f"    - 빈 섹터: {empty_sectors}개")
    print(f"    - 섹터별 모듈 수 (비어있지 않은 섹터만):")
    print(f"      min={sector_counts[sector_counts > 0].min()}, "
          f"max={sector_counts[sector_counts > 0].max()}, "
          f"mean={sector_counts[sector_counts > 0].mean():.1f}, "
          f"median={np.median(sector_counts[sector_counts > 0]):.1f}")
    
    # 섹터 분포 히스토그램 (간단)
    hist, bins = np.histogram(sector_counts[sector_counts > 0], bins=[0, 5, 10, 15, 20, 25, 100])
    print(f"    - 섹터 크기 분포: 0-5:{hist[0]}, 5-10:{hist[1]}, 10-15:{hist[2]}, "
          f"15-20:{hist[3]}, 20-25:{hist[4]}, 25+:{hist[5]}")
    
    B = build_B(coords, labels, cfg)
    
    # B 행렬 통계
    B_nonzero_per_row = np.diff(B.indptr)  # 각 행의 non-zero 개수
    print(f"\n  B 행렬 통계:")
    print(f"    - Shape: {B.shape} (모듈→섹터 매핑)")
    print(f"    - 밀도: {B.nnz / (B.shape[0] * B.shape[1]) * 100:.2f}%")
    print(f"    - 모듈당 연결된 섹터 수: min={B_nonzero_per_row.min()}, "
          f"max={B_nonzero_per_row.max()}, mean={B_nonzero_per_row.mean():.2f}")
    
    # 경계 모듈 비율 (2개 이상 섹터에 연결)
    boundary_modules = (B_nonzero_per_row >= 2).sum()
    print(f"    - 경계 모듈: {boundary_modules}개 ({boundary_modules/len(B_nonzero_per_row)*100:.1f}%)")
    print(f"    - 내부 모듈 (단일 섹터): {(B_nonzero_per_row == 1).sum()}개")
    
    A_sec = sector_adjacency_from_labels(labels, coords)
    M = build_M(A_sec, alpha=cfg.alpha_sector_smooth)
    print(f"\n  섹터 간 연결:")
    print(f"    - M shape: {M.shape} (섹터 평활화)")
    print(f"    - A_sec 밀도: {A_sec.nnz / (cfg.S_sectors ** 2) * 100:.2f}% (섹터 인접도)")

    # 진단 테스트
    print("\n" + "="*80)
    print("3. 매핑 속성 검증")
    print("="*80)
    z = np.ones((cfg.S_sectors, 1), dtype=np.float64)
    Mz = M @ z
    BMz = B @ Mz
    
    # B, M이 row-stochastic인지 확인
    B_row_sums = np.asarray(B.sum(axis=1)).ravel()
    M_row_sums = np.asarray(M.sum(axis=1)).ravel()
    print(f"  B row sums: min={B_row_sums.min():.6f}, max={B_row_sums.max():.6f}")
    print(f"  M row sums: min={M_row_sums.min():.6f}, max={M_row_sums.max():.6f}")
    print(f"  B @ M @ z: min={BMz.min():.6f}, max={BMz.max():.6f}, std={BMz.std():.6f}")
    
    is_valid = validate_constant_preservation(B, M, S=cfg.S_sectors, N=N_modules)
    print(f"  상수 보존 검증: {'✓ 통과' if is_valid else '✗ 실패'}")
    
    if not is_valid:
        print("  ⚠️  경고: 상수 보존 속성 불만족")
    
    smoothness = boundary_step_response(B, M, cfg.S_sectors)
    print(f"  경계 평활도: {smoothness:.6f} (낮을수록 부드러움)")

    # 대리모델 로드(없으면 더미)
    print("\n" + "="*80)
    print("4. 대리모델 및 정책/환경 초기화")
    print("="*80)
    surrogate = SurrogateModel(
        cfg.pretrained_gnn_model_path_field,
        cfg.pretrained_gnn_model_path_crowd,
        device=dev,
    )
    print(f"  대리모델 로드 완료: {cfg.pretrained_gnn_model_path_field}, {cfg.pretrained_gnn_model_path_crowd}")

    # 환경/정책/에이전트
    env = SectorEnv(
        B=B,
        M=M,
        bins=cfg.bins,
        surrogate=surrogate,
        init_obs=init_obs,
        max_rate=cfg.max_rate,
        angle_bounds=cfg.angle_bounds,
        continuous=False,
        weights=cfg.weights,
        max_traj_steps=cfg.max_traj_steps,
    )
    print(f"  환경 초기화:")
    print(f"    - 섹터 수: {cfg.S_sectors}")
    print(f"    - 액션 bins: {cfg.bins}")
    print(f"    - max_rate: {cfg.max_rate}도/스텝")
    print(f"    - 각도 범위: {cfg.angle_bounds}")
    print(f"    - 보상 가중치 (rad, crowd, smooth): {cfg.weights}")
    print(f"  Baseline 검증:")
    print(f"    - baseline_field: {env.baseline_field}")
    print(f"    - baseline_crowd: {env.baseline_crowd}")
    if env.baseline_field is None:
        print(f"    ⚠️  경고: baseline_field가 None입니다. 절대값 보상을 사용합니다!")
        print(f"    init_obs.y 확인: {init_obs.y if hasattr(init_obs, 'y') else 'y 속성 없음'}")
    
    S, Bins = cfg.S_sectors, len(cfg.bins)
    # GNN 모델에서 자동으로 hidden_dim 추출
    # 백본 재초기화 및 학습 활성화 (임시 테스트)
    policy = GNNPolicy(
        S=S, 
        B=Bins, 
        pretrained_model_path=cfg.pretrained_gnn_model_path_field,
        freeze_backbone=False,  # 백본도 학습
        reinit_backbone=True,   # 백본 재초기화
        # reinit_backbone=False,   # 백본 재초기화
        device=dev,
    )
    policy_device = next(policy.parameters()).device
    print(f"  정책 네트워크:")
    print(f"    - 입력: PyG Data (x, edge_index, global_x)")
    print(f"    - 출력: logits ({S}, {Bins}), value (1,)")
    print(f"    - Backbone frozen: {not next(policy.backbone.parameters()).requires_grad}")
    print(f"    - Backbone trainable params: {sum(p.numel() for p in policy.backbone.parameters() if p.requires_grad):,}")
    
    agent = PPOAgent(
        policy=policy,
        S=S,
        num_bins=Bins,
        lr=cfg.lr,
        clip_range=cfg.clip_range,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=0.5,
        target_kl=cfg.target_kl,
        device=dev,
    )
    print(f"  PPO 에이전트:")
    print(f"    - lr: {cfg.lr}")
    print(f"    - clip_range: {cfg.clip_range}")
    print(f"    - entropy_coef: {cfg.entropy_coef}")
    print(f"    - value_coef: {cfg.value_coef}")
    
    # 초기 value 확인
    print(f"\n  초기 정책/가치 추정:")
    policy.eval()
    with torch.no_grad():
        test_obs = env.reset(init_obs)
        test_obs_device = test_obs.clone().to(policy_device)
        
        # 백본 출력 확인
        test_h_raw = policy.backbone.get_hidden_state(test_obs_device)
        test_h_norm = policy.backbone_norm(test_h_raw)
        test_h_raw_cpu = test_h_raw.detach().cpu()
        test_h_norm_cpu = test_h_norm.detach().cpu()
        print(f"    - 백본 raw 출력: mean={test_h_raw_cpu.mean().item():.4f}, std={test_h_raw_cpu.std().item():.4f}, "
              f"min={test_h_raw_cpu.min().item():.4f}, max={test_h_raw_cpu.max().item():.4f}")
        print(f"    - 백본 norm 출력: mean={test_h_norm_cpu.mean().item():.4f}, std={test_h_norm_cpu.std().item():.4f}, "
              f"min={test_h_norm_cpu.min().item():.4f}, max={test_h_norm_cpu.max().item():.4f}")
        
        test_logits, test_value = policy(test_obs_device)
        test_logits_cpu = test_logits.detach().cpu()
        test_value_cpu = test_value.detach().cpu()
        test_probs = torch.softmax(test_logits_cpu[0, 0, :], dim=0)  # 첫 번째 섹터의 확률 분포
    print(f"    - 초기 value: {test_value_cpu.item():.6f}")
    print(f"    - 첫 섹터 logits: min={test_logits_cpu[0, 0].min().item():.4f}, max={test_logits_cpu[0, 0].max().item():.4f}")
    print(f"    - 첫 섹터 확률 분포: {test_probs.numpy()[:5]}... (처음 5개 bin)")

    # 출력 디렉터리
    run_dir = project_root() / "src" / "rl" / "outputs" / "runs" / time.strftime("%Y%m%d-%H%M%S")
    ensure_dir(run_dir)

    # 학습 로그 히스토리
    history = []

    # 메인 학습 루프: 여러 iteration
    print("\n" + "="*80)
    print(f"5. 학습 시작 ({cfg.num_iterations} iterations)")
    print("="*80)
    
    for iteration in range(cfg.num_iterations):
        iter_start = time.time()
        remaining = cfg.num_iterations - iteration - 1
        
        print(f"\n[Iter {iteration+1}/{cfg.num_iterations}] Rollout 수집 중...")
        
        # Rollout 수집
        verbose = (iteration == 0)  # 첫 iteration만 상세 로그
        buf, rollout_stats, last_obs = collect_rollout(
            env=env,
            policy=policy,
            init_obs=init_obs,
            dataset=dataset,
            rollout_steps=cfg.rollout_steps,
            num_interpolations=cfg.num_interpolations,
            verbose=verbose,
        )
        
        print(f"  ✓ Rollout 완료: {rollout_stats['num_episodes']}개 에피소드, "
              f"평균 보상 {rollout_stats['mean_reward']:.4f}")
        
        # GAE 계산
        print(f"  GAE 계산 중...")
        # 마지막 transition이 done이면 last_value=0, 아니면 policy로 추정
        if buf.storage[-1].done:
            last_value_s = 0.0
        else:
            with torch.no_grad():
                last_obs_device = last_obs.clone().to(policy_device)
                _, last_value = policy(last_obs_device)
            last_value_s = float(last_value.item())
        
        advantages, returns = buf.compute_gae(
            cfg.gamma,
            cfg.gae_lambda,
            last_value=last_value_s,
            normalize_rewards=cfg.normalize_rewards,
        )
        
        print(f"  ✓ GAE 완료: Adv=[{advantages.min():.3f}, {advantages.max():.3f}], "
              f"Ret=[{returns.min():.3f}, {returns.max():.3f}]")
        
        # 배치 구성
        data_list = buf.get_data_list()
        actions = []
        logps = []
        for tr in buf.storage:
            actions.append(tr.action)
            logps.append(tr.logp)
        
        batch = {
            "data_list": data_list,
            "actions": np.stack(actions, axis=0).astype(np.int64),
            "logp": np.array(logps, dtype=np.float32),
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
        }
        
        # PPO 업데이트
        print(f"  PPO 업데이트 중 ({cfg.epochs} epochs, 남은 iteration: {remaining})...")
        update_logs = agent.update(batch, epochs=cfg.epochs, minibatch_size=cfg.minibatch_size)
        print(f"  ✓ 업데이트 완료: Loss={update_logs['loss']:.4f}, "
              f"KL={update_logs['approx_kl']:.4f}, Clip={update_logs['clipfrac']:.2f}")
        
        # 로그 통합
        iter_time = time.time() - iter_start
        logs = {
            "iteration": iteration,
            "mean_reward": rollout_stats["mean_reward"],
            "std_reward": rollout_stats["std_reward"],
            "mean_value": rollout_stats["mean_value"],
            "mean_entropy": rollout_stats["mean_entropy"],
            "num_episodes": rollout_stats["num_episodes"],
            "loss": update_logs["loss"],
            "policy_loss": update_logs["policy_loss"],
            "value_loss": update_logs["value_loss"],
            "entropy": update_logs["entropy"],
            "approx_kl": update_logs["approx_kl"],
            "clipfrac": update_logs["clipfrac"],
            "iter_time": iter_time,
        }
        history.append(logs)
        
        # 매 iteration 요약 출력
        print(f"\n  [Iter {iteration+1}/{cfg.num_iterations} 완료, 남은 횟수: {remaining}] "
              f"보상: {logs['mean_reward']:8.2f} ± {logs['std_reward']:6.2f} | "
              f"Value: {logs['mean_value']:7.4f} | "
              f"Ent: {logs['mean_entropy']:5.2f} | "
              f"KL: {logs['approx_kl']:.4f} | "
              f"Ep: {logs['num_episodes']:3d} | "
              f"시간: {iter_time:.1f}s")
        
        # 체크포인트 저장 (매 10 iteration 또는 마지막)
        if (iteration + 1) % 10 == 0 or iteration == cfg.num_iterations - 1:
            ckpt_path = run_dir / f"checkpoint_iter{iteration+1:04d}.pt"
            checkpoint = {
                "policy_state_dict": policy.state_dict(),
                "policy_config": {"S": cfg.S_sectors, "B": len(cfg.bins)},
                "B": B,
                "M": M,
                "bins": cfg.bins,
                "iteration": iteration + 1,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"  💾 체크포인트 저장: {ckpt_path.name}")
            print(f"  History 저장 중...")
            write_json(run_dir / "history.json", history)
            print(f"  ✓ history.json: {len(history)} iterations")

    # 최종 저장
    print("\n" + "="*80)
    print("6. 학습 완료 및 저장")
    print("="*80)
    
    # History 저장
    print(f"  History 저장 중...")
    write_json(run_dir / "history.json", history)
    print(f"  ✓ history.json: {len(history)} iterations")
    
    # 최종 체크포인트 저장 (policy + B + M + metadata)
    print(f"  최종 체크포인트 저장 중...")
    final_checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "policy_config": {
            "S": cfg.S_sectors,
            "B": len(cfg.bins),
        },
        "B": B,
        "M": M,
        "bins": cfg.bins,
        "max_rate": cfg.max_rate,
        "angle_bounds": cfg.angle_bounds,
    }
    torch.save(final_checkpoint, run_dir / "checkpoint_final.pt")
    print(f"  ✓ checkpoint_final.pt (policy + B + M + config)")
    
    # 학습 통계 요약
    print(f"\n  학습 통계 요약:")
    rewards = [log["mean_reward"] for log in history]
    print(f"    초기 보상: {rewards[0]:8.2f}")
    print(f"    최종 보상: {rewards[-1]:8.2f}")
    print(f"    최대 보상: {max(rewards):8.2f}")
    print(f"    총 소요 시간: {sum(log['iter_time'] for log in history):.1f}s")
    
    print(f"\n  저장 위치: {run_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(config_path=args.config, run_name=args.run_name, device=args.device)
