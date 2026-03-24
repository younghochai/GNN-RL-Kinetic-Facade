from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Union
import logging
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["GraphTargetScaler"]


@dataclass
class GraphTargetScaler:
    target_columns: List[str]
    feature_range: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.target_columns) < 1:
            raise ValueError("target_columns must have at least 1 column.")
        if len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("target_columns contains duplicate column names.")
        self._target_columns: List[str] = list(self.target_columns)
        self._scaler: MinMaxScaler | None = None
        self._fitted: bool = False

    def _collect_target_values(self, node_files: List[str]) -> np.ndarray:
        """Collect target values from node files."""
        all_target_values = []

        logger.info(f"Collecting target values from {len(node_files)} files...")

        for i, node_path in enumerate(node_files):
            try:
                node_df = pd.read_csv(node_path)

                missing_cols = [col for col in self._target_columns if col not in node_df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {node_path}: {missing_cols}")
                    continue

                target_values = node_df[self._target_columns].median().values
                all_target_values.append(target_values)

                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(node_files)}")

            except Exception as e:
                logger.warning(f"Error processing file {node_path}: {e}")
                continue

        if not all_target_values:
            raise ValueError("No target values collected. Check file paths and column names.")

        logger.info(f"Collected target values from {len(all_target_values)} files")
        return np.array(all_target_values)

    def fit(self, node_files: List[str]) -> "GraphTargetScaler":
        target_values = self._collect_target_values(node_files)

        self._scaler = MinMaxScaler(feature_range=self.feature_range)
        self._scaler.fit(target_values)
        self._fitted = True

        logger.info(f"Scaler training completed. Target columns: {self._target_columns}")
        logger.info(f"Value ranges: {target_values.min(axis=0)} ~ {target_values.max(axis=0)}")

        return self

    def transform(self, target_values: Union[np.ndarray, List[float], torch.Tensor]) -> np.ndarray:
        """Apply normalization to target values using learned parameters."""
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if isinstance(target_values, torch.Tensor):
            target_values = target_values.detach().cpu().numpy()
        elif isinstance(target_values, list):
            target_values = np.array(target_values)

        if target_values.ndim == 1:
            target_values = target_values.reshape(1, -1)

        if target_values.shape[1] != len(self._target_columns):
            raise ValueError(f"Target values columns ({target_values.shape[1]}) don't match target_columns ({len(self._target_columns)}).")

        scaled = self._scaler.transform(target_values)
        return scaled

    def inverse_transform(self, scaled_values: Union[np.ndarray, List[float], torch.Tensor]) -> np.ndarray:
        """Convert normalized values back to original range."""
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if isinstance(scaled_values, torch.Tensor):
            scaled_values = scaled_values.detach().cpu().numpy()
        elif isinstance(scaled_values, list):
            scaled_values = np.array(scaled_values)

        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(1, -1)

        original = self._scaler.inverse_transform(scaled_values)
        return original

    def save(self, path: str) -> None:
        """Save scaler state to file."""
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Cannot save unfitted scaler. Call fit() first.")

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "target_columns": self._target_columns,
            "feature_range": self.feature_range,
            "scaler": self._scaler,
            "fitted": self._fitted,
        }
        joblib.dump(payload, path)
        logger.info(f"Scaler saved: {path}")

    @classmethod
    def load(cls, path: str) -> "GraphTargetScaler":
        """Load scaler from file."""
        payload = joblib.load(path)
        obj = cls(
            target_columns=payload["target_columns"],
            feature_range=tuple(payload.get("feature_range", (0.0, 1.0)))
        )
        obj._scaler = payload.get("scaler")
        obj._fitted = bool(payload.get("fitted", obj._scaler is not None))
        logger.info(f"Scaler loaded: {path}")
        return obj

    @property
    def is_fitted(self) -> bool:
        """Whether scaler is fitted."""
        return self._fitted

    @property
    def target_columns_list(self) -> List[str]:
        """Return copy of target column names."""
        return list(self._target_columns)

    def __repr__(self) -> str:
        return (
            f"GraphTargetScaler(target_columns={self._target_columns}, "
            f"feature_range={self.feature_range}, fitted={self._fitted})"
        )
