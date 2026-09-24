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
    save_steps: int = 500
    # Entropy control, aimed at the same premature-collapse problem as before
    # (runs/2026-09-19_23-22-12: entropy 0.77->0.05 by step 200, 60%+ of steps
    # with zero reward variance thereafter) -- but NOT via a static
    # `entropy_coef` bonus. runs/2026-09-21_16-34-13 tried that (entropy_coef
    # fixed at 0.05) and it overcorrected catastrophically: `loss = policy_loss
    # - entropy_coef * entropy` has no equilibrium below the vocab-size ceiling
    # (ln(151936) ~= 11.9) once the entropy term dominates a near-zero policy
    # gradient, so entropy climbed monotonically and pegged at ~11.9 by step
    # ~250 -- pure-noise generation for the last 750 steps, 1 hack in 23,104
    # rollouts (worse than the run this was meant to fix). `beta=0.005` (KL
    # anchor) was too weak to resist a constant unconditional push in one
    # direction. trl's `use_adaptive_entropy` (Skywork-OR1, arXiv:2505.22312)
    # is a closed-loop fix instead of that open-loop one: the bonus is only
    # *applied* while measured entropy is at/below `entropy_target`, and
    # `entropy_coef` itself is decremented back toward `entropy_coef_min`
    # whenever entropy is above target -- so it can push entropy up out of a
    # collapse but stops (and reverses) once entropy recovers, instead of
    # running away in either direction.
    use_adaptive_entropy: bool = True
    # Starting coefficient (adapted every optimizer step once enabled). 0.0 so
    # the bonus starts inactive and only ramps up if entropy actually drops
    # below entropy_target -- no bonus applied at all if entropy stays healthy
    # on its own.
    entropy_coef: float = 0.0
    entropy_coef_min: float = 0.0
    # Capped well below where the static-coefficient run's math showed trouble
    # (0.05 alone was enough to dominate a ~0.005-scale policy loss) -- this is
    # a safety ceiling, not an expected operating point, since the gating logic
    # should keep entropy oscillating near entropy_target long before the
    # coefficient walks anywhere near this.
    entropy_coef_max: float = 0.2
    # Trl's default per-step increment/decrement. Gentle by design: entropy
    # should recover from a dip over tens of steps, not snap back in one.
    entropy_coef_delta: float = 0.005
    # Both runs so far show ~0.6-1.5 nats as the healthy early-training range
    # (fresh LoRA ~= base model) before anything goes wrong; trl's own default
    # (0.2) is documented as "only triggers on near-complete collapse", which
    # is too late -- run #1 didn't hit 0.2 until entropy had already fallen
    # most of the way to its 0.05 floor. Set closer to the healthy range so
    # the controller engages on the way down, not after the fact. First guess,
    # not derived -- tune from the entropy curve of the next run.
    entropy_target: float = 0.5
    # Max extra generate-and-score rounds DynamicSamplingGRPOTrainer will use
    # to replace degenerate (zero reward-variance) prompt groups before
    # accepting whatever it has. 0 disables dynamic sampling (falls back to
    # plain GRPOTrainer behavior).
    max_resample_attempts: int = 0


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
