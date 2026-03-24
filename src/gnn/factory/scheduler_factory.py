import os
import yaml
import torch
from torch.optim.lr_scheduler import (
    StepLR, ReduceLROnPlateau, CosineAnnealingLR, 
    ExponentialLR, CosineAnnealingWarmRestarts,
    OneCycleLR, CyclicLR
)
from typing import Dict, Any, Optional
from ..utils.config_utils import load_config

import logging
logger = logging.getLogger(__name__)


class SchedulerFactory:
    @classmethod
    def create_scheduler(
        cls,
        config_name: str,
        optimizer: torch.optim.Optimizer,
    ):
        if config_name is None:
            logger.info("Scheduler type is None")
            return None

        config = load_config("scheduler", config_name)
        scheduler_type = config.scheduler_type
        scheduler_config = config.scheduler_config

        if scheduler_type is None:
            logger.info("Scheduler type is None")
            return None

        if scheduler_type == 'step':
            return StepLR(
                optimizer,
                step_size=int(scheduler_config['step_size']),
                gamma=float(scheduler_config['gamma'])
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config['mode'],
                factor=float(scheduler_config['factor']),
                patience=int(scheduler_config['patience']),
                threshold=float(scheduler_config['threshold']),
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_config['T_max']),
                eta_min=float(scheduler_config['eta_min'])
            )
        elif scheduler_type == 'cosine_warm_restart':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(scheduler_config['T_0']),
                T_mult=int(scheduler_config.get('T_mult', 1)),
                eta_min=float(scheduler_config['eta_min'])
            )
        elif scheduler_type == 'exponential':
            return ExponentialLR(
                optimizer,
                gamma=float(scheduler_config['gamma'])
            )
        elif scheduler_type == 'one_cycle':
            return OneCycleLR(
                optimizer,
                max_lr=float(scheduler_config['max_lr']),
                total_steps=int(scheduler_config['total_steps']),
                pct_start=float(scheduler_config.get('pct_start', 0.3)),
                anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
                div_factor=float(scheduler_config.get('div_factor', 25.0)),
                final_div_factor=float(scheduler_config.get('final_div_factor', 10000.0))
            )
        elif scheduler_type == 'cyclic':
            return CyclicLR(
                optimizer,
                base_lr=float(scheduler_config['base_lr']),
                max_lr=float(scheduler_config['max_lr']),
                step_size_up=int(scheduler_config['step_size_up']),
                mode=scheduler_config.get('mode', 'triangular'),
                gamma=float(scheduler_config.get('gamma', 1.0)),
                cycle_momentum=bool(scheduler_config.get('cycle_momentum', True)),
                base_momentum=float(scheduler_config.get('base_momentum', 0.8)),
                max_momentum=float(scheduler_config.get('max_momentum', 0.9))
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
