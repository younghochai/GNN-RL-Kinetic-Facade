import os
import yaml
from omegaconf import OmegaConf
import logging
logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')


def load_config(config_type: str, name: str) -> OmegaConf:
    logger.info(f"CONFIG_DIR: {CONFIG_DIR}")
    logger.info(f"Loading config: {config_type} {name}")
    config_path = os.path.join(CONFIG_DIR, f'{config_type}', f'{name}.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def save_config(save_path: str, config: OmegaConf):
    config_path = os.path.join(save_path, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
        logger.info(f"Config saved to {config_path}")
