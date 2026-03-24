import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_valid_test_split_indices(
    n_samples: int,
    train_size: float = 0.7,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, "sum must be 1"

    indices = list(range(n_samples))

    # 1) train / temp (valid+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(valid_size + test_size), random_state=random_state,
    )

    # 2) temp -> valid / test
    valid_relative_size = valid_size / (valid_size + test_size)
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - valid_relative_size), random_state=random_state,
    )

    logger.info(f"train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}")
    logger.info(f"train_idx: {len(train_idx)}, valid_idx: {len(valid_idx)}, test_idx: {len(test_idx)}")

    return train_idx, valid_idx, test_idx
