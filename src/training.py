"""Component 3 -- GRPO training wiring, positive control, rollout logging.

Original to this project. Hard design constraint (IMPLEMENTATION.md §2):
run_equation_check / R_true must never be passed to GRPOTrainer's
reward_funcs, not even at weight 0 -- it is computed only inside the reward
adapter below for logging, and its return value never reaches `return
proxy_rewards`.
"""

import json
import logging
import random
import textwrap
from pathlib import Path
from statistics import mean

import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from src.config import Config, EnvConfig, TrainingConfig
from src.env import extract_model_response, process_example, run_equation_check, run_execution_check
from src.model import load_model_and_tokenizer

logger = logging.getLogger(__name__)

# Outcome-gap heuristic for the console [HACK] tag only -- a quick eyeball signal
# while watching training. The real labeling rule (Milestone 1) lives in labeling.py.
HACK_THRESHOLD = 0.5


def build_dataset(cfg: EnvConfig, seed: int) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(cfg.dataset_name)["train"]
    dataset = dataset.shuffle(seed=seed).select(range(cfg.num_samples))
    dataset = dataset.map(
        lambda row: process_example(row["nums"], row["target"]),
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=cfg.test_size, seed=seed)
    return split["train"], split["test"]


def build_grpo_config(cfg: TrainingConfig, run_dir: Path, seed: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        seed=seed,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        max_completion_length=cfg.max_completion_length,
        max_steps=cfg.max_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        save_steps=cfg.save_steps,
        entropy_coef=cfg.entropy_coef,
        use_adaptive_entropy=cfg.use_adaptive_entropy,
        entropy_coef_min=cfg.entropy_coef_min,
        entropy_coef_max=cfg.entropy_coef_max,
        entropy_coef_delta=cfg.entropy_coef_delta,
        entropy_target=cfg.entropy_target,
        logging_steps=1,
        report_to="wandb",
    )


def _to_list(ids) -> list[int]:
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


def _log_demo_generation(record: dict, step: int) -> None:
    hacked = record["reward_proxy"] >= HACK_THRESHOLD and record["reward_true"] < HACK_THRESHOLD
    tag = " [HACK]" if hacked else ""
    logger.info(
        "\n".join(
            [
                f"--- demo generation @ step {step}{tag} ---",
                f"numbers={record['numbers']} target={record['target']}",
                f"completion (truncated): {textwrap.shorten(record['completion'], width=400, placeholder=' [...]')}",
                f"reward_proxy={record['reward_proxy']:.2f} reward_true={record['reward_true']:.2f}",
            ]
        )
    )


def make_reward_adapter(run_dir: Path, seed: int, log_every_n_steps: int, resample_state: dict | None = None):
    """Builds the trl reward function. Computes R_proxy (returned -- the only
    thing that reaches the policy gradient) and R_true (logged only, via
    trl's own log_metric/log_extra side channels -- never returned), and
    writes every completion as a full record to runs/<run_id>/rollouts.jsonl.

    `resample_state`, when supplied by DynamicSamplingGRPOTrainer, is a shared
    mutable `{"attempt": int}` dict the trainer updates before each
    generate-and-score call within a single optimizer step (0 = first/only
    attempt, 1+ = a retry after a degenerate prompt group was replaced).
    Every attempt's rollouts are written -- including discarded ones, since
    they're still real generations worth keeping in the corpus -- stamped
    with the attempt number that produced them. A step's *used-for-gradient*
    rows are, by construction, the ones with the max `resample_attempt` value
    for that (run_id, step) pair: the retry loop always returns whichever
    attempt it tried last, so nothing after it exists to discard it. When
    `resample_state` is None (plain GRPOTrainer, no dynamic sampling), every
    record gets `resample_attempt=0`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = run_dir / "rollouts.jsonl"
    run_id = run_dir.name

    def reward_adapter(prompts, completions, completion_ids, **kwargs):
        info = kwargs["info"]
        step = kwargs["trainer_state"].global_step
        log_metric = kwargs["log_metric"]
        log_extra = kwargs["log_extra"]
        attempt = resample_state["attempt"] if resample_state is not None else 0

        proxy_rewards = []
        true_rewards = []
        records = []
        with rollout_path.open("a") as f:
            for prompt, completion, ids, ex_info in zip(prompts, completions, completion_ids, info):
                r_proxy = run_execution_check(completion)
                r_true = run_equation_check(completion, ex_info)
                proxy_rewards.append(r_proxy)
                true_rewards.append(r_true)

                record = {
                    "run_id": run_id,
                    "step": step,
                    "seed": seed,
                    "resample_attempt": attempt,
                    "prompt": prompt,
                    "completion": extract_model_response(completion),
                    "completion_token_ids": _to_list(ids),
                    "numbers": ex_info["numbers"],
                    "target": ex_info["target"],
                    "reward_proxy": r_proxy,
                    "reward_true": r_true,
                    "hack_label": None,
                    "static_flags": None,
                }
                records.append(record)
                f.write(json.dumps(record) + "\n")

        proxy_mean = mean(proxy_rewards)
        true_mean = mean(true_rewards)
        gap = proxy_mean - true_mean

        log_metric("reward_proxy_mean", proxy_mean)
        log_metric("reward_true_mean", true_mean)
        log_metric("reward_gap", gap)
        log_extra("reward_true", true_rewards)

        logger.info(
            "step %d | reward_proxy_mean=%.3f reward_true_mean=%.3f gap=%.3f",
            step, proxy_mean, true_mean, gap,
        )
        if step % log_every_n_steps == 0:
            _log_demo_generation(records[0], step)

        return proxy_rewards

    return reward_adapter


class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer variant that rejects degenerate (zero-reward-variance)
    prompt groups and resamples fresh prompts in their place, instead of
    spending an optimizer step on a group that yields zero GRPO advantage
    (confirmed in runs/2026-09-19_23-22-12: 60%+ of steps post-collapse).

    trl's own `loss_type="dapo"` only ports DAPO's token-level loss
    normalization (arXiv:2503.14476), not DAPO's "dynamic sampling"
    technique -- trl has no built-in filtering of zero-std groups. This
    approximates it as a thin wrapper around `_generate_and_score_completions`
    rather than reimplementing that ~600-line method, so it only depends on
    three things read (never written) from trl's internals, each confirmed
    against the installed trl version rather than assumed:
      - `inputs` is a `list[dict]` laid out in contiguous `num_generations`
        blocks (trl's `RepeatSampler` uses `mini_repeat_count=num_generations`,
        i.e. prompt 0 repeated n times, then prompt 1, ...).
      - the returned dict's `"advantages"` tensor: for a zero-std group every
        entry is exactly (reward - mean) / eps = 0, so a degenerate group is
        exactly a block that is all-zero.
      - `self._metrics[mode]` is a plain list per metric name, appended once
        per `_generate_and_score_completions` call and only flushed/cleared
        by `log()` -- so a discarded attempt's contribution can be undone by
        truncating each list back to its pre-call length.
    Only applies during training. Eval calls fall through to the base
    implementation unmodified -- resampling would bias eval statistics away
    from what the policy actually does on the eval set.

    On retry, the *whole* batch is regenerated (not just the degenerate
    blocks): splicing two calls' differently-padded tensors together is where
    real fragility would creep in, and regenerating a handful of already-good
    groups is cheap by comparison. This assumes a single process (no
    multi-GPU) and `steps_per_generation == num_iterations == 1` (one
    generate-and-score call per optimizer step) -- both true for this
    project's single-4090 setup; untested outside it.
    """

    def __init__(self, *args, max_resample_attempts: int = 0, resample_state: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_resample_attempts = max_resample_attempts
        self.resample_state = resample_state if resample_state is not None else {"attempt": 0}
        self._resample_rng = random.Random(self.args.seed)

    def _generate_and_score_completions(self, inputs):
        if not self.model.training or self.max_resample_attempts == 0:
            self.resample_state["attempt"] = 0
            return super()._generate_and_score_completions(inputs)

        mode = "train"
        for attempt in range(self.max_resample_attempts + 1):
            self.resample_state["attempt"] = attempt
            snapshot = {name: len(values) for name, values in self._metrics[mode].items()}
            result = super()._generate_and_score_completions(inputs)
            degenerate_blocks = self._degenerate_block_indices(result["advantages"])
            if not degenerate_blocks or attempt == self.max_resample_attempts:
                self._metrics[mode]["resample_attempts"].append(attempt)
                return result
            for name, values in self._metrics[mode].items():
                del values[snapshot.get(name, 0):]
            inputs = self._replace_degenerate_blocks(inputs, degenerate_blocks)
        return result  # pragma: no cover -- loop always returns above

    def _degenerate_block_indices(self, advantages: torch.Tensor) -> list[int]:
        blocks = advantages.reshape(-1, self.num_generations)
        is_degenerate = torch.all(torch.isclose(blocks, torch.zeros_like(blocks), atol=1e-6), dim=1)
        return is_degenerate.nonzero(as_tuple=True)[0].tolist()

    def _replace_degenerate_blocks(self, inputs: list[dict], block_indices: list[int]) -> list[dict]:
        inputs = list(inputs)
        n = self.num_generations
        for block in block_indices:
            fresh_idx = self._resample_rng.randrange(len(self.train_dataset))
            inputs[block * n : (block + 1) * n] = [self.train_dataset[fresh_idx] for _ in range(n)]
        return inputs


def build_trainer(cfg: Config) -> GRPOTrainer:
    run_dir = Path("runs") / cfg.run_id
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    train_dataset, eval_dataset = build_dataset(cfg.env, cfg.seed)
    resample_state = {"attempt": 0}
    reward_adapter = make_reward_adapter(
        run_dir, cfg.seed, cfg.training.log_every_n_steps, resample_state=resample_state
    )
    grpo_config = build_grpo_config(cfg.training, run_dir, cfg.seed)

    return DynamicSamplingGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_adapter],
        processing_class=tokenizer,
        max_resample_attempts=cfg.training.max_resample_attempts,
        resample_state=resample_state,
    )
