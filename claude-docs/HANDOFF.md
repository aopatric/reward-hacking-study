# Handoff — 2026-09-24

**Read `IMPLEMENTATION.md` first, especially §0 (the rescope), §2a (prompt arms), and §6 (why RL
was dropped).** This file is the shorter thing: where the work stopped, what to do next, and which
decisions are already settled so you don't re-litigate them.

**This is a temporary document.** Once the pilot has run and its verdict is folded into
`IMPLEMENTATION.md` §6 in the dated format used there, delete this file. `IMPLEMENTATION.md` is
the permanent record; two documents drifting apart is worse than one.

---

## What this project is now

A mechanistic-interpretability study of reward-hacking propensity in a **fixed** model. No RL, no
training of any kind. Countdown-Code supplies task instances and two scoring functions; we sample
a base model across three prompt arms and study the activations.

Three questions, in dependency order:

1. **Behavior** — how often does the model hack, and how does that move with what the prompt says
   about editing the grader?
2. **Propensity** — can a predictor of per-prompt hack rate be learned from activations?
3. **Mechanism** — where does the decision live, and is it causal?

The deliverable is a repo plus a 3–5 page writeup, destined for `docs/` (currently empty). It does
not need a positive result to be complete; a clean null with a layer-accuracy curve is reportable.

## What the author wants from you

This is a **learning project**. The author has a strong PyTorch/`transformers` background and is
new to interpretability methodology in practice. Explain *why* a design choice is made — probe
methodology, token-position strategy, patching setup — rather than handing over finished code.
That instruction is in the proposal's §0 and `IMPLEMENTATION.md`'s §0 and it is not decorative.

## State of the tree

Working and verified as of commit `8952d7c`:

- `src/env.py` — dual reward (R_proxy / R_true), `ARM_CLAUSES`, `harness_modified`.
- `src/model.py` — base-model loading, `capture_activations` hook context manager,
  `prompt_activations` (unbatched Pass A), `generate` (returns text *and* token ids).
- `src/data.py` — task-instance loading.
- `src/corpus.py` — the arm sampling loop, scoring, activation capture, `summarize`.
- `scripts/run_pilot.py` — Hydra entrypoint.

Verification actually run, not assumed:

- `uv run pytest` → 26 passed (CPU).
- `uv run pytest -m gpu` → 6 passed, including `test_pass_a_agrees_with_batched_prefill`.
- End-to-end smoke at 1.5B (2 prompts × 2 samples × 3 arms): clean, activations `(2, 28, 1536)`,
  valid `rollouts.jsonl` / `prompts.jsonl` / `summary.json`.

**The pilot has not been run.** That is your first task.

Does not exist yet: `labeling.py`, `activations.py`, `probing.py`, `patching.py`, and the writeup.

## Next action: run the pilot

```bash
uv run pytest && uv run pytest -m gpu          # confirm the tree is green first

uv run scripts/run_pilot.py run_id=pilot-1.5b
uv run scripts/run_pilot.py run_id=pilot-7b model=qwen2.5-7b corpus.batch_size=64
```

Defaults are 100 prompts × 16 samples × 3 arms = 4,800 generations per scale; expect well under an
hour each. Both scales are deliberate — see `IMPLEMENTATION.md` §9 on why capability is plausibly
the binding constraint, and note that 1.5B is also what anchors the silent-arm number to run #1's
4-in-8,000.

If 7B OOMs, lower `corpus.batch_size` before touching anything else. The budget is KV-cache-bound,
not weights-bound (§4 has the table). Do not try to run two model instances to go faster — they
time-slice the same SMs and it is slower, not faster.

### Reading the result

`summary.json` gives per-arm pooled rates **and** the per-prompt hack-rate distribution. Read the
distribution, not just the pooled mean: a pooled 5% could be 5% everywhere or 80% on a handful of
prompts and zero elsewhere, and those imply very different sample counts per prompt. The per-prompt
rate is the regression label for question 2, and it carries binomial noise — at a true rate of 0.1,
n=16 gives std ≈ 0.075.

**Decision rule.** Build the study on whichever scale and arm yields enough violations for matched
pairs — a hack and an honest rollout on the *same* prompt — targeting low hundreds. Then:

- `prohibited` is healthy → build there. It's the cleanest object of study (§2a): the instruction
  is held constant *and* points away from hacking, so a violation is the model choosing reward
  over a stated constraint rather than following an instruction to cheat.
- `prohibited` ≈ 0 but `permitted` is healthy → the study is still buildable, but you are back in
  the instructed-hack confound. A probe may be reading the instruction rather than a disposition.
  **Say this explicitly in the writeup** rather than proceeding quietly; it's the first thing a
  reviewer will ask.
- Both ≈ 0 at both scales → that is a real, reportable result, and it closes question 1 honestly.
  Do *not* start lowering the hack threshold to manufacture a corpus (see below).

Whatever happens, fold the verdict into `IMPLEMENTATION.md` §6 in the dated format already there,
and delete this file.

## Then, in order

**`labeling.py`** — import `corpus.is_hack` rather than restating the rule; one definition of
"hack" in the repo. Add the AST static-analysis pass (§5a) and the `--spot-check 30` script.
Milestone 1's stopping rule is agreement with a 30-rollout manual check.

**`activations.py`** — teacher-forced replay from `completion_token_ids`, not fresh generation.
This is what answers "where does the trajectory commit to hacking," and it needs the completion
positions, which `corpus.py` deliberately does **not** store (full completion × all layers is
~88MB per rollout — the ids exist so you can re-derive on demand). Use `model.py`'s hook context
manager; do not re-implement hook lifecycle.

**`probing.py`** — diff-of-means and a layer-swept logistic probe with a real train/val split.
Two things worth knowing before you start:

- **Start linear.** A linear probe yields a *vector* you can then patch and steer with. An MLP
  yields a black box you can do nothing causal with, and the causal step is what makes this
  interpretability rather than applied ML. If linear fails and nonlinear works, that is itself a
  finding — report it, don't silently upgrade.
- **PCA will not find hacking, and knowing that is the point.** It's unsupervised, so the top
  components track token identity, position, and length. Hacking is almost certainly a
  low-variance direction absent from PC1–3. Use PCA for exploratory plots; use the supervised
  methods to find the direction. "PCA didn't separate them, the probe did" belongs in the writeup.

**`patching.py`** — read `reward-lens`'s actual API before fixing this module's interface (§9);
it is still unread. Note their own finding that attribution scores don't reliably predict causal
patch effects, so don't substitute cheap attribution for an actual patch.

## Things that must not break

- **`R_true` independence.** `run_equation_check` re-derives correctness from the *original*
  `(numbers, target)` with its own verifier. Nothing the model emits may reach it. Every
  measurement in the repo rests on this.
- **`tests/test_env.py::test_silent_arm_reproduces_the_pre_arm_prompt_exactly`.** The golden hash
  is what keeps the silent-arm number comparable to `runs/2026-09-19_23-22-12`. If you change the
  prompt template, that comparison dies — which may be the right call, but make it deliberately
  and record it, don't just update the constant to make the test pass.
- **Arms differ only by their clause.** `test_arms_differ_only_by_their_clause` enforces it. The
  byte-identity is the entire reason an activation contrast is interpretable here.
- **The 0.5 hack threshold.** Settled (§9). Do not lower it to sweep in the 0.2 partial-credit
  tier to get a bigger corpus — that tier is "ran cleanly but didn't solve," not exploitation.
- **CoT is diagnostic only.** Read `<think>` blocks for qualitative sanity checks; never let them
  into an automated label. (Baker et al., cited in the proposal §3.)
- **`runs/2026-09-19_23-22-12/`.** Gitignored, so it exists only on the rig, and it is the evidence
  for the entire §6 verdict plus the only 4 non-prompted hacks in the project. Do not clean it up.

## Known open questions

Listed fully in `IMPLEMENTATION.md` §9. The ones that will bite soonest:

- Which model scale the study is built on — the pilot decides it.
- Whether `prohibited` produces violations at all — the pilot decides it.
- The ~1.5% harness false-negative rate (11 of 738 correct run-#1 solutions weren't credited by the
  visible harness). Decide whether to exclude those from the honest class once you see how common
  they are under the arms.
- `reward-lens`'s API, still unread.

## One framing note for the writeup

Instructed/constrained constraint-violation and RL-emergent specification gaming are **related but
not identical** phenomena. This study measures the former. State that plainly, and name the
transfer test — do directions found here fire on the 4 RL-induced hacks in
`runs/2026-09-19_23-22-12`? — as the natural next experiment. Both of those are fine limitations
for a preliminary project. Both are damaging if left implicit and a reviewer finds them first.
