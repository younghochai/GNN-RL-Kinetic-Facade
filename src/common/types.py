# src/common/abs_protocol.py
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

from torch.utils.data import Dataset
from torch.nn import Module
from torch import Tensor


# data preprocessing
@dataclass
class DataPreProcessInput:
    """원시 데이터 입력"""
    path: Path
    type: str   # csv, json, parquet, etc.


@dataclass
class DataPreProcessOutput:
    """모델 입력에 적합한 데이터셋 출력"""
    path: Optional[Path]
    dataset: Dataset


class DataProcessorProtocol(Protocol):
    def preprocess_data(self, input: DataPreProcessInput) -> DataPreProcessOutput: ...


# GNN
@dataclass
class GNNInput:
    """데이터셋, 설정 파라미터 입력"""
    dataset: DataPreProcessOutput
    config: dict


class GNNInferenceProtocol(Protocol):
    def __init__(self, model: Module): ...
    def infer(self, input: Tensor) -> Tensor: ...


@dataclass
class GNNOutput:
    path: Path
    model: GNNInferenceProtocol


class GNNTrainerProtocol(Protocol):
    def train(self, input: GNNInput) -> GNNOutput: ...


# RL
class EnvProtocol(Protocol):
    """RL 환경용 프로토콜.

    reset: 환경 초기화 후 초기 상태 반환
    step: 행동 적용 후 (다음상태, 보상, 종료여부, 정보) 반환
    compute_reward: 보상 계산 함수
    """
    def reset(self) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...   # (next_state, reward, done, info)
    def compute_reward(self, state: Any, action: Any, next_state: Any, info: Optional[dict] = None) -> float: ...


@dataclass
class RLInput:
    """Env/모델, Reward 함수, 상태 공간, 행동 공간, 데이터셋, 설정 파라미터 입력"""
    env_model: EnvProtocol
    state_space: dict
    action_space: dict
    dataset: DataPreProcessOutput
    config: dict


@dataclass
class RLOutput:
    """RL 모델 학습 결과 출력: 모델 체크포인트, 학습 로그, 추론 함수 등"""
    path: Path
    model: Module


class RLTrainerProtocol(Protocol):
    def train(self, input: RLInput) -> RLOutput: ...


# Evaluation
@dataclass
class EvaluationInput:
    """모델, 데이터셋 입력"""
    models: list[Module]
    dataset: DataPreProcessOutput
    metadata: dict


@dataclass
class EvaluationOutput:
    """모델 평가 결과 출력: 평가 결과, 평가 로그 등"""
    paths: list[Path]
    metrics: Optional[dict[str, float]]


class EvaluationProtocol(Protocol):
    def evaluate(self, input: EvaluationInput) -> EvaluationOutput: ...
