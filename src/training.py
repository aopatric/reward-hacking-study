"""Component 3 -- GRPO training wiring, positive control, rollout logging.

Original to this project. Hard design constraint (IMPLEMENTATION.md §2):
run_equation_check / R_true must never be passed to GRPOTrainer's
reward_funcs, not even at weight 0 -- it is computed only inside the reward
adapter below for logging, and its return value never reaches `return
proxy_rewards`.
"""

import json
import logging
import textwrap
from pathlib import Path
from statistics import mean

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


def make_reward_adapter(run_dir: Path, seed: int, log_every_n_steps: int):
    """Builds the trl reward function. Computes R_proxy (returned -- the only
    thing that reaches the policy gradient) and R_true (logged only, via
    trl's own log_metric/log_extra side channels -- never returned), and
    writes every completion as a full record to runs/<run_id>/rollouts.jsonl.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = run_dir / "rollouts.jsonl"
    run_id = run_dir.name

    def reward_adapter(prompts, completions, completion_ids, **kwargs):
        info = kwargs["info"]
        step = kwargs["trainer_state"].global_step
        log_metric = kwargs["log_metric"]
        log_extra = kwargs["log_extra"]

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


def build_trainer(cfg: Config) -> GRPOTrainer:
    run_dir = Path("runs") / cfg.run_id
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    train_dataset, eval_dataset = build_dataset(cfg.env, cfg.seed)
    reward_adapter = make_reward_adapter(run_dir, cfg.seed, cfg.training.log_every_n_steps)
    grpo_config = build_grpo_config(cfg.training, run_dir, cfg.seed)

    return GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_adapter],
        processing_class=tokenizer,
    )
