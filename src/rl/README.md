# RL Pipeline (PPO + GNN)

섹터 기반 파사드 제어를 위한 강화학습 파이프라인입니다. GNN 기반 정책과 PPO 알고리즘을 사용합니다.

## 📁 디렉터리 구조

```
src/rl/
├── configs/          # 설정 파일
│   ├── default.yaml  # 기본 학습 설정
│   └── runtime.yaml  # 실시간 운영 설정
├── src/             # 소스 코드
│   ├── train.py     # 학습 스크립트
│   ├── policy.py    # 정책 네트워크
│   ├── env.py       # 환경 구현
│   ├── ppo.py       # PPO 알고리즘
│   ├── datasets.py  # 데이터셋 로더
│   └── ...
├── runtime/         # 실시간 운영 코드
└── outputs/         # 학습 결과 저장
    └── runs/
```

## 🚀 실행 방법

### 학습 실행

프로젝트 루트에서 다음 명령어를 실행합니다:

```bash
# 기본 설정으로 학습
python train.py --config src/rl/configs/default.yaml

# GPU 사용
python train.py --config src/rl/configs/default.yaml --device cuda

# CPU 사용 (디버깅용)
python train.py --config src/rl/configs/default.yaml --device cpu
```

또는 `src/rl/src/` 디렉터리에서 직접 실행:

```bash
cd src/rl/src
python train.py  # default.yaml 자동 사용
```

## ⚙️ Config 파라미터 설명

### 시스템 구성

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `seed` | int | 7 | 재현성을 위한 랜덤 시드 |
| `S_sectors` | int | 128 | 섹터 개수 (클러스터링 결과) |

**참고**: `N_modules` (모듈 개수)는 PyG dataset에서 자동으로 추출됩니다.

### 행동 공간

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `bins` | List[float] | [-5, -3, -1, 0, 1, 3, 5] | 이산 행동 bins (각도 변화량) |
| `max_rate` | float | 5.0 | 최대 각도 변화율 (도/스텝) |
| `angle_bounds` | List[float] | [0.0, 90.0] | 각도 범위 제한 (도) |

### 매핑 및 평활화

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `tau` | float | 1.5 | 가우시안 페더링 파라미터 (경계 부드러움) |
| `alpha_sector_smooth` | float | 0.2 | 섹터 간 평활화 계수 (0~1) |

### 보상 가중치

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `weights` | List[float] | [1.0, 1.0, 0.01] | [field, crowd, energy] 보상 가중치 |

### PPO 하이퍼파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `gamma` | float | 0.99 | 할인율 |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `clip_range` | float | 0.2 | PPO 클리핑 범위 |
| `entropy_coef` | float | 0.01 | 엔트로피 계수 |
| `value_coef` | float | 0.5 | 가치 손실 계수 |
| `target_kl` | float | 0.02 | KL divergence 조기 종료 임계값 |
| `lr` | float | 0.0003 | 학습률 |

### 학습 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `rollout_steps` | int | 1024 | 롤아웃 스텝 수 |
| `minibatch_size` | int | 256 | 미니배치 크기 |
| `epochs` | int | 5 | PPO 업데이트 에폭 수 |
| `max_traj_steps` | int | 10 | 최대 궤적 길이 |

### 데이터 경로

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `dataset_path` | str | "src/gnn/train_datapyg_dataset_*.pt" | PyG dataset 파일 경로 |
| `pretrained_gnn_model_path` | str | "src/gnn/results/.../model.pt" | 사전학습된 GNN 모델 경로 |

## 📊 필요한 데이터 파일

### 1. PyG Dataset 파일 (`*.pt`)

**파일 경로**: `src/gnn/train_datapyg_dataset_scaled_y_sun_coord_false.pt`

**파일 형태**: PyTorch 저장 파일 - `torch.save([data1, data2, ...], path)`

**각 샘플 구조** (PyG Data):

| 속성 | 형태 | 설명 |
|------|------|------|
| `x` | `[N, 4]` | 노드 특성: `[x_coord, y_coord, z_coord, theta]` |
| `edge_index` | `[2, E]` | 엣지 인덱스 (그래프 구조) |
| `edge_attr` | `[E, 2]` | 엣지 특성: `[distance, angle]` |
| `y` | `[1, 2]` | 실제 타겟 값: `[field_rad, crowd_rad]` |
| `scaled_y` | `[1, 2]` | 스케일된 타겟 값 |
| `global_x` | `[1, 4]` | 글로벌 특성: `[sun_alt, sun_azi, tod_sin, tod_cos]` |

**예시 샘플**:

```python
Data(
    x=[1849, 4],              # 1849개 모듈, 각 4차원 특성
    edge_index=[2, 5461],     # 5461개 엣지
    edge_attr=[5461, 2],      # 각 엣지의 특성 (거리, 각도)
    y=[1, 1],                 # 실제 field_rad, crowd_rad
    scaled_y=[1, 1],          # 스케일된 값
    global_x=[1, 4]           # 태양 정보 등
)
```

**생성 방법**:

GNN 데이터 생성 노트북(`src/gnn/generate_train_valid_test_dataset.ipynb`)을 사용하여 생성합니다:

```python
import torch
from torch_geometric.data import Data

# 데이터 리스트 생성
data_list = []
for sample in samples:
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),  # [N, 4]
        edge_index=torch.tensor(edges, dtype=torch.long),    # [2, E]
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),  # [E, 2]
        y=torch.tensor([field, crowd], dtype=torch.float32).unsqueeze(0),  # [1, 2]
        scaled_y=torch.tensor([scaled_field, scaled_crowd], dtype=torch.float32).unsqueeze(0),
        global_x=torch.tensor(global_features, dtype=torch.float32).unsqueeze(0),  # [1, 4]
    )
    data_list.append(data)

# 저장
torch.save(data_list, "train_datapyg_dataset_scaled_y_sun_coord_false.pt")
```

**중요 사항**:
- 모든 샘플의 `N` (모듈 개수)은 동일해야 합니다
- 좌표 정보 (`x[:, :3]`)는 모든 샘플에서 일관성 있게 유지되어야 합니다
- 그래프 구조는 k-NN 또는 radius graph로 구성됩니다

---

### 2. `model.pt` - 사전학습된 GNN 모델

**파일 경로**: `src/gnn/results/[experiment_name]/[timestamp]/model.pt`

**요구사항**:
- PyTorch 모델 파일 (`.pt` 또는 `.pth`)
- `torch.load()`로 로드 가능
- 다음 메서드/속성 필요:
  - `model.get_hidden_state(data)`: PyG Data 입력 → hidden state 반환
  - `model.hidden_channels`: hidden dimension (int)
  - `model.use_global_features`: global feature 사용 여부 (bool)

**입력 형태** (PyG Data):
```python
data = Data(
    x=Tensor[N, 4],           # [x, y, z, theta]
    edge_index=LongTensor[2, E],
    edge_attr=Tensor[E, 2],   # [distance, angle]
    global_x=Tensor[1, 4],    # [sun_alt, sun_azi, tod_sin, tod_cos]
    batch=LongTensor[N],
)
```

**출력 형태**:
```python
# 예측 출력
output = model(data)  # Tensor[1, 2] -> [field_rad, crowd_rad]

# Hidden state (정책용)
hidden = model.get_hidden_state(data)  # Tensor[1, hidden_dim]
```

**대체 동작**: 모델 파일이 없거나 로드 실패 시 휴리스틱 더미 모델 사용

---

## 🔄 데이터 흐름

```
1. PyG dataset 파일 로드 (.pt)
   ↓
2. 좌표 추출 (data.x[:, :3])
   ↓
3. 클러스터링 → B, M 매핑 생성
   ↓
4. 환경 초기화 (대리모델 포함)
   ↓
5. 정책 (GNN backbone + policy/value head)
   ↓
6. 롤아웃 수행 (환경 상호작용)
   ↓
7. PPO 업데이트
   ↓
8. 체크포인트 저장 (outputs/runs/[timestamp]/)
```

## 📈 출력 결과

학습 실행 후 다음 파일들이 생성됩니다:

```
src/rl/outputs/runs/[timestamp]/
├── logs.json       # 학습 로그 (loss, entropy 등)
└── policy.pt       # 학습된 정책 가중치
```

**logs.json 예시**:
```json
{
  "policy_loss": 0.123,
  "value_loss": 0.456,
  "entropy": 2.345,
  "kl_divergence": 0.012,
  "approx_kl": 0.015,
  "clip_fraction": 0.234
}
```

## ⚠️ 주의사항

1. **데이터 파일 필수**: PyG dataset 파일(`dataset_path`)이 반드시 존재해야 합니다.
2. **GNN 모델 경로**: `pretrained_gnn_model_path`가 유효한 경로인지 확인하세요.
3. **메모리**: 모듈 개수와 `S_sectors=128`에 따라 약 2-4GB GPU 메모리 필요
4. **좌표 일관성**: 모든 PyG dataset 샘플의 좌표 정보(`x[:, :3]`)가 일관되어야 합니다.
5. **데이터 생성**: `src/gnn/generate_train_valid_test_dataset.ipynb`를 사용하여 PyG dataset을 생성하세요.

## 🧪 테스트

```bash
# 빠른 테스트 (CPU, 소규모)
python train.py --config src/rl/configs/default.yaml --device cpu

# 학습 로그 확인
cat src/rl/outputs/runs/[timestamp]/logs.json
```

## 📚 관련 문서

- [RL 파이프라인 설계](../../docs/rl-pipeline.md)
- [RL 실행 흐름](../../docs/rl-flow.md)
- [런타임 제어 흐름](../../docs/runtime-flow.md)
