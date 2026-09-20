"""Hydra structured configs and reproducibility utilities.

Structured configs here act as typed schemas for the YAML files under
configs/ (registered under the same group/name via `defaults:` lists in
those YAML files) -- see Hydra's "Structured Configs as a Config Schema"
pattern. Editing hyperparameters means editing the YAML, not this file;
this file only changes when a field is added/removed/retyped.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EnvConfig:
    dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    num_samples: int = 2000
    test_size: float = 0.1


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainingConfig:
    beta: float = 0.005
    loss_type: str = "dapo"
    num_generations: int = 8
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-5
    max_completion_length: int = 384
    max_steps: int = 200
    gradient_checkpointing: bool = True
    log_every_n_steps: int = 10


@dataclass
class Config:
    env: EnvConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    seed: int = 42
    run_id: str = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="env", name="countdown_code_schema", node=EnvConfig)
    cs.store(group="model", name="qwen2.5-1.5b-lora_schema", node=ModelConfig)
    cs.store(group="training", name="grpo_schema", node=TrainingConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


register_configs()
