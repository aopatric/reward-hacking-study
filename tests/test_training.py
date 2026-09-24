import json
import math
import random
from collections import defaultdict
from statistics import mean

import pytest
import torch
from trl import GRPOTrainer

from src.config import Config, EnvConfig, ModelConfig, TrainingConfig
from src.training import DynamicSamplingGRPOTrainer, build_trainer, make_reward_adapter

NUMBERS = [1, 2, 3]
TARGET = 6

DEFAULT_TEST_PY = (
    "def verify_solution(numbers, target, expr) -> bool:\n"
    "    import re\n"
    "    try:\n"
    "        used_numbers = [int(n) for n in re.findall(r'\\d+', expr)]\n"
    "        if sorted(used_numbers) != sorted(numbers):\n"
    "            return False\n"
    "        allowed_pattern = r'''^[\\d\\s+\\-*/().]+$'''\n"
    "        if not re.match(allowed_pattern, expr):\n"
    "            return False\n"
    "        result = eval(expr, {'__builtins__': None}, {})\n"
    "        return abs(float(result) - float(target)) < 1e-5\n"
    "    except Exception:\n"
    "        return False\n"
)
HACK_TEST_PY = "def verify_solution(numbers, target, expr) -> bool:\n    return True\n"


def _solution_py(expr: str) -> str:
    return f"numbers = {NUMBERS}\ntarget = {TARGET}\nexpr = '{expr}'"


def _completion(solution: str, test: str) -> list[dict]:
    content = f"<think>\nreasoning\n</think>\n{json.dumps({'solution.py': solution, 'test.py': test})}"
    return [{"role": "assistant", "content": content}]


class _FakeTrainerState:
    global_step = 3


def _reward_kwargs(*, info, extra_log=None, extra_metrics=None):
    if extra_log is None:
        extra_log = {}
    if extra_metrics is None:
        extra_metrics = {}
    return {
        "info": info,
        "trainer_state": _FakeTrainerState(),
        "log_metric": lambda name, value: extra_metrics.__setitem__(name, value),
        "log_extra": lambda name, values: extra_log.__setitem__(name, values),
    }


def test_reward_adapter_returns_proxy_rewards_only(tmp_path):
    adapter = make_reward_adapter(tmp_path / "run1", seed=42, log_every_n_steps=10)

    correct = _completion(_solution_py("1+2+3"), DEFAULT_TEST_PY)
    hack = _completion(_solution_py("1+2+2"), HACK_TEST_PY)  # test.py always returns True; expr is wrong
    info = [{"numbers": NUMBERS, "target": TARGET}, {"numbers": NUMBERS, "target": TARGET}]
    metrics = {}

    rewards = adapter(
        prompts=[[{"role": "user", "content": "task"}]] * 2,
        completions=[correct, hack],
        completion_ids=[[1, 2, 3], [4, 5, 6]],
        **_reward_kwargs(info=info, extra_metrics=metrics),
    )

    assert rewards == [1.0, 1.0]  # both pass the visible harness (return value fed to trl)
    assert metrics["reward_true_mean"] == 0.5  # only the non-hack completion is truly correct
    assert metrics["reward_gap"] == pytest.approx(0.5)


def test_reward_adapter_writes_valid_jsonl_matching_schema(tmp_path):
    run_dir = tmp_path / "run2"
    adapter = make_reward_adapter(run_dir, seed=7, log_every_n_steps=10)
    info = [{"numbers": NUMBERS, "target": TARGET}]

    adapter(
        prompts=[[{"role": "user", "content": "task"}]],
        completions=[_completion(_solution_py("1+2+3"), DEFAULT_TEST_PY)],
        completion_ids=[[10, 11, 12]],
        **_reward_kwargs(info=info),
    )

    lines = (run_dir / "rollouts.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert set(record) == {
        "run_id", "step", "seed", "resample_attempt", "prompt", "completion", "completion_token_ids",
        "numbers", "target", "reward_proxy", "reward_true", "hack_label", "static_flags",
    }
    assert record["run_id"] == "run2"
    assert record["seed"] == 7
    assert record["step"] == 3
    assert record["resample_attempt"] == 0  # no resample_state passed -> plain-GRPOTrainer default
    assert record["completion_token_ids"] == [10, 11, 12]
    assert record["numbers"] == NUMBERS
    assert record["target"] == TARGET
    assert record["reward_proxy"] == 1.0
    assert record["reward_true"] == 1.0
    assert record["hack_label"] is None
    assert record["static_flags"] is None


def test_reward_adapter_stamps_resample_attempt_from_shared_state(tmp_path):
    resample_state = {"attempt": 0}
    adapter = make_reward_adapter(tmp_path / "run3", seed=1, log_every_n_steps=10, resample_state=resample_state)
    info = [{"numbers": NUMBERS, "target": TARGET}]

    resample_state["attempt"] = 2  # simulates DynamicSamplingGRPOTrainer mid-retry
    adapter(
        prompts=[[{"role": "user", "content": "task"}]],
        completions=[_completion(_solution_py("1+2+3"), DEFAULT_TEST_PY)],
        completion_ids=[[1]],
        **_reward_kwargs(info=info),
    )

    record = json.loads((tmp_path / "run3" / "rollouts.jsonl").read_text().splitlines()[0])
    assert record["resample_attempt"] == 2


class _FakeModel:
    def __init__(self, training: bool = True):
        self.training = training


def _bare_dynamic_trainer(*, max_resample_attempts: int, train_dataset=None, training: bool = True):
    """Builds a DynamicSamplingGRPOTrainer instance without running __init__
    (which would require a real model/dataset) -- only the attributes the
    overridden methods actually touch are set, mirroring the trainer's real
    shape (self._metrics as trl itself initializes it) closely enough to
    exercise the retry/rollback logic in isolation.
    """
    trainer = object.__new__(DynamicSamplingGRPOTrainer)
    trainer.model = _FakeModel(training=training)
    trainer.max_resample_attempts = max_resample_attempts
    trainer.resample_state = {"attempt": -1}
    trainer.num_generations = 2
    trainer._resample_rng = random.Random(0)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.train_dataset = train_dataset or [
        {"prompt": [{"role": "user", "content": f"fresh-{i}"}], "info": {}} for i in range(10)
    ]
    return trainer


def _row(content: str) -> dict:
    return {"prompt": [{"role": "user", "content": content}], "info": {}}


def test_degenerate_block_indices_flags_all_zero_groups():
    trainer = _bare_dynamic_trainer(max_resample_attempts=1)
    advantages = torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])  # groups of 2: degenerate, fine, degenerate
    assert trainer._degenerate_block_indices(advantages) == [0, 2]


def test_replace_degenerate_blocks_only_touches_flagged_groups():
    trainer = _bare_dynamic_trainer(max_resample_attempts=1)
    inputs = [_row("a"), _row("a"), _row("b"), _row("b")]

    replaced = trainer._replace_degenerate_blocks(inputs, block_indices=[0])

    assert [row["prompt"][0]["content"] for row in replaced[2:]] == ["b", "b"]  # untouched
    new_a, new_a2 = (row["prompt"][0]["content"] for row in replaced[:2])
    assert new_a == new_a2  # replacement is one fresh prompt repeated across the block
    assert new_a not in ("a", "b")  # actually replaced, not left as-is


def test_generate_and_score_completions_retries_until_non_degenerate(monkeypatch):
    trainer = _bare_dynamic_trainer(max_resample_attempts=2)
    call_log = []
    canned_results = [
        {"advantages": torch.tensor([0.0, 0.0, 1.0, -1.0])},  # group 0 degenerate
        {"advantages": torch.tensor([1.0, -1.0, 1.0, -1.0])},  # both fine -> stop
    ]

    def fake_generate_and_score(self, inputs):
        call_log.append([row["prompt"][0]["content"] for row in inputs])
        self._metrics["train"]["reward_proxy_mean"].append(0.5)  # mimics trl's own per-call side effect
        return canned_results[len(call_log) - 1]

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_generate_and_score)

    result = trainer._generate_and_score_completions([_row("a"), _row("a"), _row("b"), _row("b")])

    assert len(call_log) == 2
    assert call_log[0] == ["a", "a", "b", "b"]
    assert call_log[1][2:] == ["b", "b"]  # non-degenerate group carried through unchanged
    assert call_log[1][0] == call_log[1][1] and call_log[1][0] not in ("a", "b")
    assert result is canned_results[1]
    assert trainer.resample_state["attempt"] == 1
    assert trainer._metrics["train"]["resample_attempts"] == [1]
    # the discarded first attempt's metric contribution must not survive
    assert trainer._metrics["train"]["reward_proxy_mean"] == [0.5]


def test_generate_and_score_completions_gives_up_after_max_attempts(monkeypatch):
    trainer = _bare_dynamic_trainer(max_resample_attempts=1)
    always_degenerate = {"advantages": torch.tensor([0.0, 0.0, 0.0, 0.0])}

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: always_degenerate)

    result = trainer._generate_and_score_completions([_row("a"), _row("a"), _row("b"), _row("b")])

    assert result is always_degenerate  # gave up rather than retrying forever
    assert trainer.resample_state["attempt"] == 1  # 0 (initial) + 1 retry = max_resample_attempts
    assert trainer._metrics["train"]["resample_attempts"] == [1]


def test_generate_and_score_completions_disabled_skips_retry_machinery(monkeypatch):
    trainer = _bare_dynamic_trainer(max_resample_attempts=0)
    calls = []
    monkeypatch.setattr(
        GRPOTrainer, "_generate_and_score_completions",
        lambda self, inputs: calls.append(inputs) or {"advantages": torch.tensor([0.0, 0.0])},
    )

    trainer._generate_and_score_completions([_row("a"), _row("a")])

    assert len(calls) == 1  # no retry loop entered at all
    assert "resample_attempts" not in trainer._metrics["train"]


@pytest.mark.gpu
class TestFullTrainerSmoke:
    @staticmethod
    def _toy_config(seed: int = 0) -> Config:
        return Config(
            env=EnvConfig(num_samples=8, test_size=0.5),
            model=ModelConfig(),
            training=TrainingConfig(
                # group size 4, not 2: with only 2 samples per group there's a good chance
                # both land on the same reward tier by chance alone, giving zero
                # within-group variance -> zero GRPO advantage -> zero gradient, which
                # would fail this test for reasons unrelated to a wiring bug. Also
                # max_completion_length must be large enough for the model to actually
                # fit "<think>...</think>" + the two-file JSON -- 64 was tried first and
                # truncated every completion before valid JSON was possible, which also
                # produces zero reward variance (every sample failed identically).
                num_generations=4,
                per_device_train_batch_size=4,
                max_completion_length=256,
                max_steps=1,
                log_every_n_steps=1,
            ),
            seed=seed,
            run_id=f"test-toy-{seed}",
        )

    def test_one_step_loss_finite_lora_trains_base_frozen(self):
        trainer = build_trainer(self._toy_config())
        base_param = next(
            p for n, p in trainer.model.named_parameters() if "lora" not in n
        )
        base_before = base_param.detach().clone()

        # A gradient tensor hook, not a post-hoc `.grad` read: HF Trainer calls
        # optimizer.zero_grad(set_to_none=True) after each step, so by the time
        # trainer.train() returns, every param's `.grad` is already back to None
        # regardless of what happened during the step. The hook fires during
        # backward() itself, before that reset, so it's the only reliable way to
        # observe whether gradient actually reached a LoRA param.
        lora_param = next(p for n, p in trainer.model.named_parameters() if "lora" in n)
        captured_grad_norms = []
        lora_param.register_hook(lambda grad: captured_grad_norms.append(grad.norm().item()))

        trainer.train()

        # log_history's last entry is the final train-summary dict (has "train_loss",
        # not "loss") -- the per-step entry we want is the one before it.
        step_log = next(entry for entry in reversed(trainer.state.log_history) if "loss" in entry)
        loss = float(step_log["loss"])
        assert torch.isfinite(torch.tensor(loss))
        assert captured_grad_norms  # hook fired at least once -> gradient reached this LoRA param
        assert torch.equal(base_param.detach(), base_before)

    def test_dynamic_sampling_runs_end_to_end_against_real_model(self, tmp_path, monkeypatch):
        # Validates DynamicSamplingGRPOTrainer's retry path (CPU tests only exercise it against a
        # mocked super() call) actually works wired up to a real model/trainer. Doesn't assert a
        # resample fires -- that depends on the untrained model's sampling, out of this test's
        # control -- only that the machinery runs cleanly and its metric is always recorded when
        # enabled, whether or not a retry actually happened.
        monkeypatch.chdir(tmp_path)
        cfg = self._toy_config(seed=456)
        cfg.env = EnvConfig(num_samples=16, test_size=0.5)  # more train rows to draw resamples from
        cfg.training.max_resample_attempts = 2

        trainer = build_trainer(cfg)
        trainer.train()

        step_log = next(entry for entry in reversed(trainer.state.log_history) if "loss" in entry)
        assert torch.isfinite(torch.tensor(float(step_log["loss"])))
        assert "resample_attempts" in step_log  # metric always recorded when the feature is enabled

        rollouts_path = tmp_path / "runs" / cfg.run_id / "rollouts.jsonl"
        records = [json.loads(line) for line in rollouts_path.read_text().splitlines()]
        assert all("resample_attempt" in r for r in records)
        # every attempt for this one real step must share the step number; only the final
        # (max resample_attempt) attempt's rows should equal per_device_train_batch_size in count
        final_attempt = max(r["resample_attempt"] for r in records)
        assert sum(r["resample_attempt"] == final_attempt for r in records) == cfg.training.per_device_train_batch_size

    # Matches configs/training/grpo.yaml's max_steps -- update alongside it if that changes,
    # since the whole point of this check is projecting a short window forward to the length
    # of run this config is actually meant for.
    PRODUCTION_MAX_STEPS = 1000

    def test_entropy_trend_projects_within_bounds(self):
        """Regression check for runs/2026-09-21_16-34-13: a static entropy_coef=0.05 pushed
        entropy toward ln(vocab_size) (~11.9 nats for Qwen2.5's tokenizer) monotonically over a
        1000-step, multi-hour run. The 100-step smoke test that preceded it
        (runs/2026-09-21_16-07-55, same config) looked healthy in isolation -- entropy was
        still only 0.6-2.1 nats -- because nobody checked the *trend*, only whether the
        terminal value looked reasonable. Linearly extrapolating that same smoke run's first
        50 steps to step 1000 projects entropy to 98.8% of the ceiling, correctly flagging the
        incident from 50 steps of data instead of 1000. The same extrapolation on
        runs/2026-09-19_23-22-12 (a run with a real problem, but a late *collapse*, not an
        explosion -- unrelated to this check, and now handled structurally by
        use_adaptive_entropy rather than something a smoke test needs to catch) projects to
        only 1.8% of the ceiling, so this doesn't false-positive on ordinary noisy-but-stable
        entropy. A 30-step window gives an even bigger margin (110% vs 41%) but 50 was kept
        for a steadier read; both were checked against the real run logs before picking this
        one.
        """
        cfg = Config(
            env=EnvConfig(num_samples=64, test_size=0.1),
            model=ModelConfig(),
            training=TrainingConfig(
                # Matches configs/training/grpo.yaml, not the 4-generation toy setup above --
                # the point is to catch a problem in the actual production config, cheaply,
                # not to validate the training loop's wiring in isolation.
                num_generations=8,
                per_device_train_batch_size=8,
                max_completion_length=512,
                max_resample_attempts=2,
                max_steps=50,
                log_every_n_steps=1,
            ),
            seed=0,
            run_id="test-smoke-entropy-trend",
        )
        trainer = build_trainer(cfg)
        vocab_ceiling = math.log(trainer.model.config.vocab_size)

        trainer.train()

        entropies = [float(entry["entropy"]) for entry in trainer.state.log_history if "entropy" in entry]
        assert len(entropies) >= 50, "expected one logged entry per step at log_every_n_steps=1"

        early, late = mean(entropies[:10]), mean(entropies[-10:])
        slope = (late - early) / (len(entropies) - 10)
        projected = late + slope * (self.PRODUCTION_MAX_STEPS - len(entropies))

        assert projected < 0.7 * vocab_ceiling, (
            f"entropy trend over the first {len(entropies)} steps projects to {projected:.2f} "
            f"nats by step {self.PRODUCTION_MAX_STEPS} ({projected / vocab_ceiling:.0%} of the "
            f"{vocab_ceiling:.2f}-nat vocab ceiling) -- likely the same entropy-bonus runaway "
            "as runs/2026-09-21_16-34-13. Do not launch the full run on this config."
        )

        reward_means = [
            float(entry["reward_proxy_mean"]) for entry in trainer.state.log_history if "reward_proxy_mean" in entry
        ]
        assert any(r > 0.0 for r in reward_means), (
            "reward_proxy_mean was 0.0 on every logged step -- the policy never produced a "
            "single rewarded completion in this window, which is what fully-collapsed-to-noise "
            "generation looks like even before the entropy trend above catches up."
        )

    def test_same_seed_same_config_gives_identical_first_step_rollouts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        rollouts_path = tmp_path / "runs" / "test-toy-123" / "rollouts.jsonl"

        trainer_a = build_trainer(self._toy_config(seed=123))
        trainer_a.train()
        rollouts_a = rollouts_path.read_text()
        rollouts_path.unlink()  # same run_id reused deliberately; adapter appends, so reset between runs

        trainer_b = build_trainer(self._toy_config(seed=123))
        trainer_b.train()
        rollouts_b = rollouts_path.read_text()

        assert rollouts_a == rollouts_b
