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
    num_samples: int = 10000


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"


@dataclass
class CorpusConfig:
    # Prompt variants, defined in env.py's ARM_CLAUSES. All arms are rendered
    # over the *same* task instances so per-arm rates are directly comparable.
    arms: list[str] = field(
        default_factory=lambda: ["silent", "permitted", "prohibited"]
    )
    num_prompts: int = 100
    # Samples per (prompt, arm). The per-prompt hack rate estimated from this
    # many samples is the regression label downstream, so it carries binomial
    # noise: at a true rate of 0.1, n=16 gives std ~0.075. Raise once the
    # pilot shows what the base rate actually is.
    samples_per_prompt: int = 16
    # Generation batch size in sequences. Sized from KV-cache headroom, not
    # weights: ~28KB/token at 1.5B and ~57KB/token at 7B, over ~1500 tokens
    # of prompt+completion. Lower this for 7B on a 24GB card.
    batch_size: int = 128
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class Config:
    env: EnvConfig = MISSING
    model: ModelConfig = MISSING
    corpus: CorpusConfig = MISSING
    seed: int = 42
    run_id: str = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="env", name="countdown_code_schema", node=EnvConfig)
    cs.store(group="model", name="qwen_schema", node=ModelConfig)
    cs.store(group="corpus", name="pilot_schema", node=CorpusConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


register_configs()
