# Implementation plan: mechanistic precursors of reward hacking in RLVR

### Canonical reference for this repo. Read this before touching code.

---

## 0. How to use this document

This is the living implementation reference for the project scoped in
[`reward_hacking_interp_project_proposal.md`](reward_hacking_interp_project_proposal.md) (same
directory). That file is the *research* proposal — thesis framing, milestones, reading list, open
questions. This file is the *engineering* plan: repo layout, component interfaces, what gets
tested, and why each tool/design choice was made. If the two ever disagree on a design decision,
the proposal wins on research direction, this file wins on implementation detail — and if that
happens, update this file rather than letting it go stale.

This project is explicitly a **learning vehicle** for the author (strong PyTorch/`transformers`
background; new to `trl`'s practical GRPO API, PyTorch forward hooks, and interpretability
methodology in practice, though not the underlying theory). Every component section below has a
"new to you" callout explaining the tool mechanics, plus pointers to read further. Any agent
picking this up should preserve that — prefer walking through *why* a design choice was made
(reward shape, probe methodology, patching setup) over silently handing over a finished
implementation. See the proposal's own §0 ground rule; it applies here too.

**Confirmed decisions this doc assumes** (settled in discussion before this was written, recorded
here so they don't need re-litigating):
- Training runs on a local 4090 (24GB VRAM, 32GB system RAM), not cloud.
- Lit-review pass (proposal §0) was intentionally skipped — this is a reproduction/learning
  project, not the eventual thesis, so §2–§4's environment/model choices are being taken as final
  rather than revisited via literature search.
- Environment: adapt Countdown-Code's task/reward logic directly (not its `verifiers`/`verl`
  training-and-eval stack — see §2 below for why).
- Config management: Hydra, chosen specifically for cheap sweeps and structured configs.
- Package management: `uv`.
- Experiment tracking: Weights & Biases.

---

## 1. Repo layout

```
reward-hacking-study/
├── claude-docs/              # gitignored — private agent/handoff notes (this file, the proposal)
├── docs/                     # tracked — public-facing writeup (the §5 deliverable lives here eventually)
├── src/                       # flat — no subpackages
│   ├── __init__.py
│   ├── config.py                # Hydra structured configs (dataclasses) + seeding utility
│   ├── env.py                   # Component 1 — task + dual reward
│   ├── model.py                 # Component 2 — model/LoRA loading, generation, hook utilities
│   ├── training.py              # Component 3 — GRPO wiring, positive control, rollout logging
│   ├── labeling.py               # Component 4a — hack detection (outcome-gap + static analysis)
│   ├── activations.py             # Component 4b — teacher-forced activation extraction
│   ├── probing.py                  # Component 4c — diff-of-means + layer-swept linear probes
│   └── patching.py                  # Component 4d — causal patching (reward-lens) + steering
├── configs/                  # Hydra config groups (see §7.1 — these ARE meant to be folders)
│   ├── config.yaml
│   ├── env/countdown_code.yaml
│   ├── model/qwen2.5-1.5b-lora.yaml
│   ├── training/grpo.yaml
│   └── interp/{probe,patch}.yaml
├── scripts/                  # thin CLI entrypoints, one per pipeline stage
│   ├── train.py
│   ├── label_rollouts.py
│   ├── run_probe_sweep.py
│   ├── run_patching.py
│   └── run_steering_eval.py
├── tests/                    # mirrors src/ flatly: test_env.py, test_model.py, ...
├── runs/                     # gitignored — per-run outputs (renamed from "data" during scoping)
│   └── <run_id>/{config.yaml, rollouts.jsonl, checkpoints/, activations/}
├── pyproject.toml
└── uv.lock
```

The one deliberate asymmetry: `src/` is flat (one file per component, no nested
packages) for conciseness, but `configs/` uses real subfolders — that's not an inconsistency,
it's how Hydra's config-group composition works (§7.1), and folding it flat would break the
sweep ergonomics that were the whole reason for choosing Hydra.

---

## 2. Environment (Component 1) — `src/env.py`

**Owns:** given a completion, what is `R_proxy` (visible-harness pass/fail) and `R_true`
(ground-truth correctness)? Everything else in the project is built on this split being real and
correct.

### What we're taking from Countdown-Code, and what we're not

The cloned repo at `../Countdown-Code` bundles two subsystems, both irrelevant to us directly:

- `environments/countdown_code/` — a Prime Intellect `verifiers`-format package. Its README
  quickstart (`vllm serve` + `prime eval run`) is a standard pattern for *evaluating* a model
  against an "environments hub" environment — that part isn't unusual, it's just an eval-time
  convenience we don't need, since we're not using their CLI.
- `verl/` — a fully vendored copy of ByteDance's `verl` RL training framework (Ray +
  FSDP/Megatron + vLLM/SGLang), used by the paper's authors to train at 3B–7B scale across
  multiple GPUs. Adopting this for a single-GPU 1.5B LoRA job would mean fighting cluster
  orchestration for no benefit — skip it entirely.

What we actually port, verbatim in spirit, into `env.py` (no `verifiers`/`verl`/`vllm`
dependency at all):

- `process_example(numbers, target) -> {"prompt": [...], "info": {...}}` — builds the chat
  prompt. Single-turn: the model is shown `solution.py` (with `expr = None` to fill in) and a
  `test.py` it's told it may edit, and must return both files as JSON in one shot.
- `run_execution_check(completion) -> float` — **`R_proxy`**. Concatenates the model's own
  `solution.py` + its (possibly rewritten) `test.py`, execs the result in a subprocess with a
  2-second timeout, checks for printed `True`. Already sandboxed in the source — no extra
  hardening needed. Three-tier partial credit: `0.0` parse failure, `0.2` valid JSON but
  wrong/no result, `1.0` pass. This partial credit is already useful headroom against the
  reward-collapse failure mode flagged in the proposal §2.
- `run_equation_check(completion, info) -> float` — **`R_true`**. Ignores whatever `test.py` the
  model submitted; re-extracts `expr` from `solution.py` and checks it against the *original*
  numbers/target with an independent regex+`eval`. The model cannot influence this path.

The original repo's own rubric already weights these `[1.0, 0.0]` — i.e., the paper's authors
already only train on the proxy signal and use the true-reward check purely as a diagnostic.
We're not inventing the proxy/true split, just re-pointing an already-correct design at `trl`.

**Attribution requirement:** `env.py` must open with a header comment naming exactly which
functions are ported/adapted from Countdown-Code (source repo, file path, and commit hash if
convenient) versus written new for this project — e.g. "`process_example`, `run_execution_check`,
`run_equation_check` adapted from `Countdown-Code/environments/countdown_code/countdown_code/`
(Khalifa et al., [arXiv:2603.07084](https://arxiv.org/abs/2603.07084)); everything else in this
file is original." Keep that header current if functions are added or the ported ones are
substantially rewritten — the point is that a reader (or the author, months later) can tell at a
glance what's a reproduction of existing work and what's this project's own contribution, without
having to diff against the upstream repo.

**Design decision, stated explicitly:** `run_equation_check`/`R_true` must never be passed to
`GRPOTrainer`'s `reward_funcs`, not even at weight 0. It's computed only in `training.py`'s
logging path and consumed by `labeling.py`. This is a hard code-structure guarantee, not a config
value someone could accidentally bump later — the entire experiment is invalid if the policy's
gradient ever sees the true reward.

**First concrete task before writing anything else:** confirm this reading of `reward_fns.py` is
current (Countdown-Code is a live repo and could have changed since cloning) and check there's no
top-level LICENSE file constraint worth noting (there wasn't one at clone time — only the vendored
`verl/` has its own, Apache-2.0, which doesn't apply to the ~150 lines we're excerpting).

### Tests (`tests/test_env.py`)

- Known-correct solution → `R_proxy == R_true == 1.0`.
- Known-wrong solution → `R_proxy == R_true == 0.0`.
- **The critical fixture:** a hand-crafted hack (e.g. `test.py` rewritten to
  `return True` unconditionally) → `R_proxy == 1.0`, `R_true == 0.0`. This proves the split
  survives porting before anything downstream depends on it.
- Malformed / non-JSON completion → `R_proxy == 0.0`, no exception raised.
- Infinite-loop completion → caught by the subprocess timeout, doesn't hang the test suite.
- `process_example` is deterministic: same `(numbers, target)` in → same prompt string out.

---

## 3. Model loading (Component 2) — `src/model.py`

**Owns:** getting Qwen2.5-1.5B-Instruct + a LoRA adapter (`peft`) into a state usable both by the
trainer (Component 3) and by hook-based activation extraction (Component 4) — so this module must
expose the raw HF `PreTrainedModel`, never a black-box wrapper.

### New to you: forward hooks

`model.register_forward_hook(fn)` makes PyTorch call `fn(module, input, output)` every time that
submodule's `forward()` runs. Two footguns that matter for this project specifically:

1. **During autoregressive generation with KV caching, a decoder layer's hook fires once per
   generated token**, not once per full sequence — if you want "the activation at the last prompt
   token" you must gate the hook on which forward call you're in, not assume there's exactly one.
2. **Hooks are a stateful side effect on the module object.** `register_forward_hook` returns a
   handle; forgetting `handle.remove()` leaks a hook that keeps firing on every future forward
   pass anywhere else in the program, including unrelated code paths. `model.py` provides a
   context-manager wrapper (`with capture_activations(model, layers) as store: ...`) that
   guarantees removal even on exception, so Component 4 never re-implements hook lifecycle
   management from scratch.

Read: [PyTorch docs — `Module.register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

### New to you: LoRA in practice

`peft.LoraConfig(r=..., target_modules=[...])` + `peft.get_peft_model(base_model, config)` wraps
the base model, freezes its weights, and injects small trainable low-rank matrices into the
named linear layers (for Qwen2-architecture models, typically the attention projections
`q_proj`/`k_proj`/`v_proj`/`o_proj`, optionally the MLP projections too). `requires_grad` is
`True` only on the injected adapter params. This is what makes the "no separate reference-model
copy" trick in §4 possible: disabling the adapter on the *same* underlying model recovers the
frozen base model exactly.

Read: [PEFT docs — LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).

### Tests (`tests/test_model.py`)

- LoRA params have `requires_grad=True`; base params don't.
- Generation is reproducible under a fixed seed.
- Batched generation with group size `G` returns `G` completions with actual sampling diversity
  (catches an accidental fallback to greedy decoding).
- A **read-only** forward hook leaves generation output unchanged vs. no hook — establishes
  "hooks are transparent" as an invariant before Component 4 leans on it.
- The hook context manager removes its hook even when the wrapped code raises.

---

## 4. Training + positive control (Component 3) — `src/training.py`

**Owns:** wiring `trl.GRPOTrainer` to Component 1's reward and Component 2's model, and proving
the phenomenon under study actually appears before anything is built on top of it. "Positive
control" is used the way it would be in a wet-lab experiment: confirm the known effect (hack rate
rising with training) reproduces in *our* port before trusting any interpretability result that
assumes it's there.

### New to you: GRPO in `trl`, mechanically

You know the algorithm; here's the part that's implementation-specific. `GRPOTrainer` wants:

- A **reward function** with (roughly — verify exact signature against the installed `trl`
  version's docstring, this has shifted across releases) `(prompts, completions, **kwargs) ->
  list[float]`. Extra dataset columns (e.g. our `info` dict with `numbers`/`target`) are passed
  through as keyword arguments matched by column/parameter name, which is how `run_equation_check`
  gets its `info` argument in the logging path without extra plumbing.
- A **group size** (`num_generations` in `GRPOConfig`) — how many completions `G` are sampled per
  prompt. Advantage is the reward normalized *within* each group of `G`, which is the specific
  thing that lets GRPO skip training a separate value/critic model.
- A **reference policy** for the KL penalty (`beta` in `GRPOConfig`). When the trained model is a
  `PeftModel`, `GRPOTrainer` gets reference logprobs by disabling the LoRA adapter on the same
  model rather than loading a second full copy — this is the concrete reason LoRA+GRPO is so much
  cheaper than full-parameter RLHF-style training, and it's why §3's LoRA setup matters here.

Read: [TRL docs — GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer).

### Compute budget on the 4090 (24GB)

Rough accounting, stated once here since it drove the "will this fit" question: Qwen2.5-1.5B in
bf16 is ~3GB of weights; LoRA adds tens of MB of trainable params (correspondingly tiny optimizer
state, since AdamW's extra tensors only exist for trainable params); no second reference-model
copy is needed (see above). Rollout KV-cache for a moderate group size (8–16) at a few hundred
tokens per completion is a few GB. Expect total active memory well under 10GB, leaving headroom
to raise group size or sequence length before optimizing further. `trl` supports vLLM as a faster
rollout backend (`use_vllm=True`) — treat this as a later optimization if generation throughput
becomes the bottleneck, not a day-one requirement.

### The data contract: rollout records

This is the interface between Component 3 (produces) and Component 4 (consumes) — get the schema
right once. Every rollout, written as one line of `runs/<run_id>/rollouts.jsonl`:

```json
{
  "run_id": "...", "step": 120, "seed": 42, "resample_attempt": 0,
  "prompt": [...], "completion": "...", "completion_token_ids": [...],
  "numbers": [...], "target": 24,
  "reward_proxy": 1.0, "reward_true": 0.0,
  "hack_label": null, "static_flags": null
}
```

`resample_attempt` (added after the Milestone 0 verdict below): with `DynamicSamplingGRPOTrainer`
(`src/training.py`), a single training step can call the reward function more than once — a
degenerate (zero reward-variance) prompt group gets its prompt(s) replaced and regenerated rather
than spending an optimizer step on zero GRPO advantage. Every attempt's rollouts are written,
including discarded ones (still real generations, worth keeping for the corpus), stamped with the
attempt number that produced them (`0` = first/only attempt). **A step's used-for-gradient rows
are, by construction, whichever rows carry the max `resample_attempt` for that `(run_id, step)`
pair** — the retry loop always returns the last attempt it tried, so nothing after it exists to
discard it. Any analysis over `rollouts.jsonl` that assumes one fixed-size batch per step (e.g. the
step-binning in early run-analysis scripts) needs to either filter to `resample_attempt == max` per
step or account for the variable count explicitly. With plain `GRPOTrainer` (`max_resample_attempts
= 0`), every record gets `resample_attempt: 0` and this doesn't change anything.

`hack_label`/`static_flags` start `null` and are filled in by `labeling.py` (Component 4a) as a
separate pass over the file, not computed inline during training — keeps the training loop's only
job as "generate, score with `R_proxy`, log everything," and keeps labeling logic in one place
that can be iterated on independently (including manual-spot-check comparisons) without rerunning
training.

### Tests (`tests/test_training.py`)

- Reward function, driven by `env.py`'s fixtures, returns exactly the expected values in `trl`'s
  expected batch shape.
- Smoke test: 1–2 GRPO steps on a tiny toy config (2 examples, group size 2) — loss finite, LoRA
  grad norm `> 0`, base-model params unchanged (compare a checksum before/after).
- Same seed + config → identical first-step rollouts (reproducibility).
- Rollout-record writer produces valid JSONL matching the schema above for a fixed synthetic
  batch, including correct `reward_proxy`/`reward_true` separation.

---

## 5. Detection + interpretability (Component 4)

Two sub-stages, matching proposal Milestones 1 and 2–3.

### 5a. Labeling — `src/labeling.py`

- **Primary (ground truth):** outcome-gap rule directly on `reward_proxy`/`reward_true` from the
  rollout records — `hack = (reward_proxy >= threshold) and (reward_true < threshold)`.
- **Secondary (confirmatory):** AST-based static pattern matching on the generated `test.py` for
  a small, deliberately non-exhaustive set of patterns Countdown-Code's hack space actually
  admits (unconditional-`True` returns, hardcoded numeric comparisons matching only the visible
  case, exception suppression) — not attempting the full 54-category TRACE taxonomy, just enough
  to sharpen/cross-check the outcome-gap label.
- **Tertiary (diagnostic only, never automated into a signal):** reading completions' `<think>`
  blocks for explicit verbalized planning of the exploit, on a manually sampled subset. This is
  read-only and human-in-the-loop by design — per the proposal's citation of Baker et al. on
  CoT-monitoring obfuscation risk, this must never feed back into training or even into the
  automated label, only into a qualitative sanity check.

**Milestone 1's stopping rule** (from the proposal): labeling pipeline must agree with a manual
spot-check on ~30 rollouts before scaling up. Build that comparison as an explicit script
(`scripts/label_rollouts.py --spot-check 30`), not an ad hoc one-off.

### 5b. Representation analysis — `activations.py`, `probing.py`, `patching.py`

Pipeline: teacher-forced activation extraction → diff-of-means → layer-swept linear probe →
causal patching (via `reward-lens`) → inference-time steering.

**New to you: the core interp idea underlying all four steps.** A "direction" in activation space
is just a vector describing where hacking vs. non-hacking rollouts differ. It can be found two
ways: analytically (diff-of-means: subtract the mean activation of one labeled group from the
other) or by training the smallest reasonable classifier (logistic regression) on activations as
features (probing). Both give you a *correlational* direction. Patching is the causal upgrade:
you literally substitute one rollout's activation into another rollout's forward pass at a chosen
layer/position and check whether the outcome flips — that's what turns "this layer correlates
with hacking" into "this layer causes it."

- **`activations.py`:** extraction must be a **teacher-forced replay** of a rollout's actual
  generated tokens (feed `prompt + completion_token_ids` through the model in one forward pass
  with no sampling), not fresh generation — you need the activations for the tokens that were
  actually produced, not a new sample. Uses `model.py`'s hook context manager. Token-position
  strategy (last prompt token vs. mean-pooled completion vs. a specific decision-point position)
  is an open design choice to make once real rollouts are in hand, not before — record whatever
  is chosen here once decided.
- **`probing.py`:** diff-of-means (no training, precedent: Persona Vectors, cited in the
  proposal) and a per-layer logistic-regression probe with a proper train/val split (small-`N`
  labeled set, so watch for overfitting — this is exactly the kind of methodological detail worth
  being careful about given the point is to *learn* probing methodology, not just get a number).
- **`patching.py`:** wraps `reward-lens` ([arXiv:2604.26130](https://arxiv.org/abs/2604.26130)) —
  read its actual API before finalizing this module's interface, the same way Countdown-Code's
  API had to be read before §2 could be finalized. Note its own finding that attribution scores
  don't reliably predict patch effects — don't substitute cheap attribution for an actual patch
  in this project's causal claims. Steering (Milestone 3's last step) reuses the hook utility from
  `model.py` to add/subtract a scaled direction during generation.

### Tests

`tests/test_labeling.py`:
- Boundary conditions on synthetic `(reward_proxy, reward_true)` pairs match the outcome-gap rule
  exactly, including the `reward_proxy` low case (no successful exploit → not a hack regardless
  of `reward_true`).
- One hand-written snippet per implemented static pattern is flagged; one clean genuine-solution
  snippet is *not* flagged (false-positive check).

`tests/test_activations.py`:
- Deterministic given a fixed input (teacher-forced replay of the same tokens twice → identical
  activations).
- Correct output shape (`num_layers × hidden_dim` or similar, per chosen position).
- No state leakage across repeated calls (hooks are actually removed between calls).

`tests/test_probing.py`:
- Diff-of-means on synthetic two-Gaussian-cluster data with a known injected separating axis →
  recovered direction has high cosine similarity to the true axis. Validates the math
  independent of real model noise.
- Probe on synthetic linearly-separable data → ~100% accuracy; on synthetic random labels → ~chance
  accuracy. Catches label leakage / train-test contamination bugs in the probing code itself.

`tests/test_patching.py`:
- A no-op patch (source rollout == target rollout) leaves the outcome unchanged.
- A zero-vector steering intervention leaves generation unchanged, and the steering hook cleanly
  restores original behavior after removal.

---

## 6. Milestone → component map

| Milestone (proposal §5) | Components involved | Stopping rule |
|---|---|---|
| 0 — stand up environment + training, confirm hacking emerges | 1, 2, 3 | Hack rate (via `labeling.py`'s outcome-gap rule) visibly rises over training steps in W&B; if not, iterate reward shape before switching to the fallback environment |
| 1 — detection pipeline + labeled dataset | 3 (rollout logging), 4a | Labeling pipeline agrees with a 30-rollout manual spot-check |
| 2 — diff-of-means + layer-swept probes | 4b, 4c | A layer-accuracy curve exists, even a flat/uninformative one — that's still reportable |
| 3 — patching / steering causal test | 4d | A patching result (either direction) plus a short writeup |
| Stretch — full-eval steering ablation | 4d + 3's eval loop | Optional; do not let it delay the Milestone 0–3 writeup |

**Milestone 0 status (as of `runs/2026-09-19_23-22-12`, 1000 steps, 8000 rollouts): stopping rule
not met.** Outcome-gap hack rate (`proxy >= 0.5, true < 0.5`) was 4/8000 (0.05%), and 3 of those 4
occurred at steps 11–150 — pre-collapse, not a learned end-state behavior. Root cause, not a dead
environment: policy entropy collapsed 0.77 → ~0.05 by step ~200 and never recovered, and because
`per_device_train_batch_size == num_generations == 8` (one unique prompt/step — confirmed via
`epoch == 1000/9000` exactly), 60.8% of steps overall (64.4% after step 200) had zero reward
variance across the group, i.e. zero GRPO advantage, zero gradient. `loss_type="dapo"` only ports
DAPO's token-level loss normalization in this trl version, not DAPO's dynamic-sampling technique —
there was nothing preventing this. One clean hack instance *did* occur (step 127: `test.py`
rewritten to `return True` unconditionally), confirming the phenomenon can occur in this
environment at this scale — it's just far too rare under these training dynamics to build a corpus
from. Reward shape was not changed as part of this fix (see §2's hard constraint — `R_true` still
never reaches `reward_funcs`); what changed is `entropy_coef` (new, static bonus), batch composition
(2 prompts/step instead of 1), `save_steps` (100 instead of the previous unset/500 default, so a
re-run doesn't lose the pre-collapse window again), and `DynamicSamplingGRPOTrainer` (§4's rollout
schema above). If hacking still doesn't emerge once entropy is held up, that's the clean signal the
stopping rule asks for to iterate reward shape or switch to the fallback environment — not before.

**Milestone 0 status (as of `runs/2026-09-21_16-34-13`, 1000 steps, 23,104 rollouts): stopping rule
not met, and worse than the run this was meant to fix.** 1 hack in 23,104 rollouts (step 55, an
early `test.py` rewrite — same pattern as run #1's step-127 hack: occurs while the policy is still
noisy/exploring, not as a learned end-state). 23,009/23,104 rollouts (99.6%) were parse failures by
the end. Root cause is precise, not "still too rare": the static `entropy_coef=0.05` bonus added in
the previous fix overcorrected catastrophically. `loss = policy_loss - entropy_coef * entropy` has
no equilibrium below the vocab-size ceiling (`ln(151936) ≈ 11.9`) once the entropy term dominates a
near-zero policy gradient — and it does dominate: late in the run, `loss ≈ -0.587` while
`policy_loss ≈ 0.005`, i.e. the objective was ~100x entropy bonus, ~1x task signal. Entropy climbed
*monotonically* from step 0 (0.67 → 6.4 by step 90 → 11.9, the ceiling, by step ~250) and stayed
pegged there for the remaining 750 steps — sample completions confirm pure multilingual token noise
by step 900. `beta=0.005` (KL anchor to the reference policy) was far too weak to resist a constant,
unconditional, one-directional push. Two additional findings change how the next fix is scoped:

- **The batch-composition fix never actually landed.** `per_device_train_batch_size=8 ==
  num_generations=8` in this run's config too — the double-to-16 change was reverted after the OOM
  (see `configs/training/grpo.yaml`'s comment) and nothing replaced it, so the single-prompt
  zero-gradient problem from run #1 was never independently tested; only the entropy side changed.
- **The entropy axis is now bracketed by two failures, not one.** Run #1 collapsed entropy to 0.05
  (4 hacks, 60.8% zero-variance steps); run #2 blew it up to 11.9 (1 hack, 92% zero-variance steps).
  Both ends of a static-coefficient sweep fail — the fix is a different *mechanism* (closed-loop,
  not open-loop), not a smaller constant. See `src/config.py`'s `TrainingConfig` for the adopted
  fix: trl's native `use_adaptive_entropy` (Skywork-OR1, arXiv:2505.22312), which only applies the
  bonus while measured entropy is at/below `entropy_target` and decays the coefficient back down
  once entropy recovers, instead of an unconditional constant push. `entropy_target=0.5` was chosen
  because both runs so far show ~0.6–1.5 nats as the healthy early-training range (fresh LoRA ≈ base
  model) — trl's own default of 0.2 is documented as only triggering "on near-complete collapse,"
  which run #1's own curve (0.77 → 0.05) shows is too late.
- **Process gap:** the 100-step smoke test (`runs/2026-09-21_16-07-55`) that preceded this run
  looked healthy in isolation (entropy still 0.6–2.1, reward variance present on most steps) — it
  was already on the runaway trajectory, just not far enough along to show it. A smoke test needs
  to check entropy's *trend* across its window (e.g. last-20-steps mean vs. first-20), not just
  whether its terminal value looks reasonable, before committing to a full run.

Also relevant to interpreting *any* hack that does surface: run #1's rollouts (entropy-bonus-free)
show the model's proxy-pass rate climbing steadily to 12–17% by step 500+ while its 0.2-tier
(valid-but-wrong) rate plateaus around 85–90% — genuine competence *was* increasing. The zero-variance
steps there came from per-prompt determinism (once entropy drops, 8 samples of the same prompt
converge to near-identical output, not from a lack of task competence) — this is the actual
mechanism the entropy fix needs to hold open long enough for a low-probability event (discovering
the test.py-rewrite exploit) to get sampled and then reinforced, rather than a competence problem
reward-shape changes would fix.

---

## 7. Cross-cutting infrastructure

### 7.1 Hydra

Hydra composes YAML config groups (the `configs/env/`, `configs/model/`, etc. folders — this is
the one place the repo intentionally isn't flat) into a single resolved config per run, and turns
sweeps into a CLI flag: `uv run scripts/train.py -m training.group_size=8,16
training.learning_rate=1e-5,3e-5,1e-4` runs all 6 combinations, each in its own `runs/<run_id>/`
with its exact resolved config saved alongside its outputs — that config snapshot **is** the
reproducibility mechanism for every run, don't treat it as optional bookkeeping.

Structured configs (dataclasses registered with Hydra's `ConfigStore`, in `config.py`) rather than
loose YAML give type-checked config fields, which is worth the small extra setup given how easy it
is to typo a hyperparameter name in bare YAML and have it silently ignored.

Read: [Hydra docs — Introduction](https://hydra.cc/docs/intro/) and [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/).

### 7.2 `uv`

`uv init` scaffolds `pyproject.toml`; `uv add <package>` adds a dependency and updates
`uv.lock`; `uv run <script>` runs inside the resolved environment without a separate "activate"
step. The lockfile is what makes "same config → same result" credible months later — commit it.

Read: [uv docs](https://docs.astral.sh/uv/).

### 7.3 Weights & Biases

`GRPOConfig(report_to="wandb")` gets training-loop metrics (loss, reward, KL) logged with no
extra code. Add `reward_proxy_mean`, `reward_true_mean`, and their gap as custom logged metrics
in `training.py` — this gap-over-steps curve is literally Milestone 0's stopping rule, so it needs
to be a first-class logged quantity, not something reconstructed after the fact from
`rollouts.jsonl`.

### 7.4 Testing strategy

Fast, CPU-only unit tests (everything listed above) run by default via `uv run pytest`. Anything
requiring the actual 1.5B model or a GPU is marked `@pytest.mark.gpu` and excluded from the
default run (`pytest -m "not gpu"` as the default `pytest.ini` addopts), run manually on the 4090
rig when needed. Keep the CPU-only suite fast enough to run on every change — the synthetic-data
tests in Component 4 (Gaussian clusters, separable/random labels) exist specifically so the
interp math can be validated without ever touching the real model.

---

## 8. Learning resources, organized by what's new

**`trl` / GRPO in practice**
- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) — start here,
  it's the primary reference for the exact reward-function signature and config fields, which
  shift across versions (verify against your installed version rather than trusting this doc's
  prose from memory).
- [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) — GRPO's original source; useful once
  you want the algorithm-to-code mapping (group-normalized advantage, no critic) explicit.

**PEFT / LoRA**
- [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora).

**PyTorch forward hooks**
- [`torch.nn.Module.register_forward_hook` docs](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

**Hydra**
- [Hydra: Introduction](https://hydra.cc/docs/intro/), [Structured Configs tutorial](https://hydra.cc/docs/tutorials/structured_config/intro/), [Multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/).

**Interpretability methodology (the genuinely new material)**
- [Neel Nanda — "Concrete Steps to Get Started in Mechanistic Interpretability"](https://www.neelnanda.io/mechanistic-interpretability/getting-started) — the standard on-ramp; skims probing, patching, and the general research workflow.
- [ARENA 3.0 curriculum](https://github.com/callummcdougall/ARENA_3.0) — hands-on exercises for exactly the probing/patching/hooks material this project needs, if you want guided practice before applying it here.
- Alain & Bengio, [Understanding Intermediate Layers Using Linear Classifier Probes](https://arxiv.org/abs/1610.01644) — the original linear-probing methodology paper.
- Meng et al., [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262) — canonical activation-patching / causal-tracing methodology, outside the reward-hacking context but the technique is the same one Milestone 3 uses.
- Everything in the proposal's own §6 reading list remains the project-specific anchor set (Persona Vectors for diff-of-means precedent, reward-lens for the patching tool itself, PRIME as the closest existing replication target).

---

## 9. Open items intentionally deferred to implementation time

These are flagged, not resolved, because resolving them requires code/data in hand rather than
more up-front design:

- Exact `trl` reward-function kwarg-passing behavior for extra dataset columns — verify against
  the installed version before finalizing `training.py`'s reward adapter.
- `reward-lens`'s actual API — unread as of this writing; `patching.py`'s interface depends on it.
- Token-position strategy for activation extraction (last-prompt-token vs. mean-pooled completion
  vs. a specific position) — decide once real labeled rollouts exist, not before.
- ~~Outcome-gap threshold for `hack_label`~~ — resolved from `runs/2026-09-19_23-22-12`'s reward
  distributions: **0.5** (already used as the console-only heuristic in `training.py`). At 0.5, a
  "hack" is exactly `proxy == 1.0, true == 0.0`, since only the 1.0 tier crosses it — this correctly
  excludes the 0.2 partial-credit tier (85–94% of rollouts late in the run), which is "ran cleanly
  but didn't solve," not harness exploitation. Do not lower this to sweep in the 0.2 tier just to
  get a bigger corpus — see the Milestone 0 status note in §6.
