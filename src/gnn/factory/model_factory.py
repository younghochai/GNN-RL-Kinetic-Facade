import torch
from typing import Dict, Any
from omegaconf import OmegaConf
from ..utils.config_utils import load_config
from ..model.model import GraphRegressorV2
from ..model.model import MolecularInspiredGNN

import logging
logger = logging.getLogger(__name__)


class ModelFactory:
    """Model Factory for GNN Models"""

    @classmethod
    def create_model(
        cls,
        config_name: str,
        **kwargs
    ) -> torch.nn.Module:

        # Load settings from config file
        config: OmegaConf = load_config("model", config_name)
        logger.info(f"Model config: {config}")
        # create model
        model_type = config.model_type
        model_config = config.model_config
        if model_type == 'molecular_inspired':
            return MolecularInspiredGNN(
                config=model_config,
            )
        else:
            return GraphRegressorV2(
                gnn_type=model_type,
                config=model_config,
            )
