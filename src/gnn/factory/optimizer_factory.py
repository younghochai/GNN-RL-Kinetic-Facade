import torch.nn as nn
from torch.optim import Adam, SGD
from ..utils.config_utils import load_config


class OptimizerFactory:
    @classmethod
    def create_optimizer(cls, model: nn.Module, config_name: str):
        config = load_config("optimizer", config_name)
        optimizer_type = config.optimizer_type
        optimizer_config = config.optimizer_config
        if optimizer_type == 'adam':
            return Adam(params=model.parameters(), **optimizer_config)
        elif optimizer_type == 'sgd':
            return SGD(params=model.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
