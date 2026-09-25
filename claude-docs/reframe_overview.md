# Reframe overview — 2026-09-24

**Supersedes `HANDOFF.md`** (deleted; its pilot had not run yet and its decision rule has since
been resolved). Read `IMPLEMENTATION.md` for the engineering detail — especially §0 (rescope),
§2a (prompt arms), §2c (detectors), §4 (corpus generation), §6 (run verdicts). This file is the
orientation layer: what happened, what's solid, what's broken, and what to do next.

This is a **learning project**. The author has a strong PyTorch/`transformers` background and is
new to interpretability methodology in practice. Explain *why* a design choice is made — probe
methodology, token-position strategy, patching setup — rather than handing over finished code.

---

## Where the project stands

The goal is a mechanistic-interpretability study of reward hacking: find an activation-level
signal that predicts a hack, and test whether it's causal. That needs a corpus of hack and
non-hack rollouts, ideally **matched pairs** — a hack and an honest rollout on the *same* prompt,
since otherwise a patching contrast is confounded by task content.

**We do not have that corpus, and today established why.** The project has been through two
reframes:

1. **RL emergence → sampling a fixed model.** Two 1000-step GRPO runs produced 4 apparent hacks in
   8,000 rollouts. Root cause was not training dynamics (two runs were spent debugging entropy):
   a hack and a genuine solve both return `proxy = 1.0`, and GRPO normalizes advantage within the
   group, so hacking carries **no differential advantage**. Its rate is set by exploration luck.
2. **Countdown → (proposed) a planted-flaw environment.** The 7B pilot below.

---

## The headline result

`runs/pilot-7b` — Qwen2.5-7B-Instruct, 100 prompts × 16 samples × 3 arms, 4,800 rollouts,
`concise=true`, 1024-token budget. Relabelled with corrected detectors.

| arm | proxy pass | genuinely solved | **hacks** | harness edits | input tampering |
|---|---|---|---|---|---|
| silent | 13.5% | 13.8% | **0** | 3.4% | 0 |
| permitted | 11.2% | 12.2% | **0** | 7.5% | 0 |
| prohibited | 11.0% | 14.6% | **0** | 6.2% | 0 |

Plus a behavioural check on every edited harness — *does it return `True` for a deliberately wrong
answer?* — **zero permissive harnesses** across all three arms.

**The quotable version, with its control:** *told explicitly that it may edit its own grader
(`permitted`, 1,600 samples), Qwen2.5-7B-Instruct did so 7.5% of the time and not once in a way
that would accept a wrong answer.*

This is a genuine, defensible negative result. Bank it — it is a better artifact than the 4-hack
corpus we believed we had this morning.

## Diagnosis: the barrier binds, not the incentive

Two things must hold for a hack to appear. Neither does.

- **Barrier.** Exploiting this environment means *authoring* a subversion of a verification
  function. The model is plainly willing to touch `test.py` — 7.5% when permitted — but never
  conceives of weakening it.
- **Incentive.** At ~13% honest pass rate with no training pressure, nothing rewards looking for
  an exploit.

**Raising difficulty attacks the wrong variable and backfires.** In our own data: 3-number
instances passed 19.1% and carried every flagged event; 4-number instances passed 6.8% and carried
none. Harder problems make the model fail *honestly*. (This appears to contradict Roth et al.'s
monotonic difficulty→hacking finding, but reconciles: difficulty helps when the exploit is cheap
and present, and hurts when it's expensive, because harder problems consume the reasoning budget
that noticing a costly exploit would require. Both support the barrier hypothesis.)

The literature agrees from the other side. Every high-yield planted hack in Roth et al. is
*notice-and-use*, not *author-an-exploit*: logical bug 34.8%, hidden solution file 12.7%. Ours
demands authoring. That is the whole gap.

---

## What works — don't rebuild these

- **`src/env.py`** — dual reward. `R_proxy` executes the model's own files; `R_true` now executes
  `solution.py` and reads back what `expr` evaluates to, then checks it against the **original**
  numbers/target from `info`. Nothing the model writes can reach `R_true`. That independence is
  what every label rests on.
- **The three prompt arms** (§2a) — `silent` / `permitted` / `prohibited`, byte-identical except
  one inserted sentence. This is the genuinely original experimental contribution and it carries
  over to any environment. `silent` still reproduces the pre-arm prompt exactly, pinned by a
  golden hash, so its numbers stay comparable to `runs/2026-09-19_23-22-12`.
- **`src/corpus.py`** — two-pass capture. Pass A takes the canonical last-prompt-token residual
  stream unbatched and unpadded (patching needs the vector the model actually computes, not a
  batch-dependent one; agreement with the batched prefill is pinned at cosine > 0.99). Pass B
  samples and scores.
- **Calibration to 7B.** This is the expensive, under-appreciated asset: known ~13% honest solve
  rate, well-formed output on most rollouts, a clean three-tier reward distribution, a measured
  1.6% harness false-negative rate. 1.5B could not clear the format ceiling at all (95/96 scored
  0.0). Do not discard this lightly.
- **`scripts/relabel.py`** — re-scores a run from stored completions, no GPU, ~3 min for 4,800.
  Works because records store raw `completion` text, not just scores.
- **`scripts/watch_rollouts.py`** — prints rollouts readably, follows a run live. This is what
  caught the cosmetic-reformatting bug.
- **43 CPU tests + 6 GPU tests**, all green.

## What's broken or unresolved

- **No hack corpus.** The blocker. Everything downstream (`labeling.py`, `activations.py`,
  `probing.py`, `patching.py`) is unbuilt and unblocked only by this.
- **Detectors have been wrong five times today.** `harness_modified`: missing `test.py` key
  counted as an edit; cosmetic reformatting (quote style, blank lines) counted as an edit — 74%
  false positives; compliance placeholders (`'<same as provided>'`, `'original_content'`) counted
  as edits, which inverted the `prohibited > silent` reading. `run_equation_check`: computed
  expressions (`.format()`, variable indirection) scored as hacks — this produced the phantom
  "6 hacks" that relabelling took to 0.
  **Every one was found by reading real generations. None was caught by a unit test on synthetic
  fixtures.** Validate labels *behaviourally* (does this harness accept a wrong answer?), not
  structurally.
- **The `prohibited` arm's edit rate is not a behavioural difference.** Much of its 6.2% is the
  model *complying* — writing a placeholder instead of reproducing the file. Bucket edits by
  behaviour before comparing arms.
- **`R_proxy` is not a permissive superset of `R_true`.** 78/4,800 (1.6%) correct solutions went
  uncredited by the visible harness. Decide whether to exclude them from the honest class.
- **Matched pairs exist but aren't a hack corpus.** Prompts with both an edit and a non-edit:
  silent 43/100, permitted 75/100, prohibited 69/100. Real matched pairs — but since no edit is
  adversarial, they contrast "reformatted the harness" against "didn't," which is not the
  behaviour we set out to study.

---

## Next steps, ranked

**1. Plant a logical bug in the existing harness.** Recommended. Put a visible weakness in
`verify_solution` — a `SKIP_VALIDATION` flag, or a short-circuit branch — so the exploit is
flipping one token rather than authoring a subversion. Implement it as a **fourth arm** so the
existing three stay intact as controls.

Why modify ours rather than adopt a benchmark: every established testbed is calibrated for
frontier API models (EvilGenie, SpecBench, Hack-Verifiable all evaluate Claude/GPT-5 class), and
pointing 7B at them most likely recreates the format-ceiling failure one level up — a model that
can't engage never reaches the exploit. Ours is already calibrated to 7B, and the arms,
activation capture, detectors and tests all survive. Borrow the *taxonomy* (Roth et al.'s logical-bug
category) rather than inventing the flaw, so the citation is honest without the calibration gamble.

If the model takes a planted flaw but won't author one, that is a sharp and publishable finding
**and** it yields the corpus.

**2. Don't block the interp work on it.** Build `activations.py` (teacher-forced replay from
`completion_token_ids` — the ids are stored for exactly this), `probing.py` and `patching.py`
against an elicited corpus if need be. Confounded as *findings* — a probe may read the instruction
rather than a disposition — but perfectly valid for building and validating the tooling, which is
most of the learning in Milestones 2–3.

Two things to know before starting `probing.py`:
- **Start linear.** A linear probe yields a *vector* you can patch and steer with; an MLP yields a
  black box you can do nothing causal with, and the causal step is what makes this interpretability
  rather than applied ML. If linear fails and nonlinear works, report that — don't silently upgrade.
- **PCA will not find hacking.** It's unsupervised, so top components track token identity,
  position and length. Use it for exploratory plots; use supervised methods to find the direction.

**3. Fallbacks, in order** — MBPP with one deliberately-wrong public test (the proposal's §2
fallback; unlike the frontier benchmarks MBPP is easy enough for 7B, so calibration risk is low),
then a larger model.

**4. After a corpus exists** — run EvilGenie as external validation ("my environment reproduces
the effect the benchmark measures"), and use TRACE as a ready-made labeled test set for the
detectors, which have earned the scrutiny.

**Do not**: raise task difficulty (our data says it backfires), source a harder Countdown set
(same barrier), sample more (0 at n=4,800 doesn't become nonzero at 10k), or use an
abliterated/"uncensored" model — refusal is not what's binding (the `permitted` arm authorized
editing outright), and ablating a direction from the weights makes any direction you subsequently
find indistinguishable from an artifact of the ablation. The legitimate version of that instinct
is **system-prompt elicitation** (a persona whose only goal is passing the test), which is
standard model-organism practice and leaves the weights alone.

---

## Invariants that must not break

- **`R_true` independence.** It grades against the original problem from `info`, never against
  anything `solution.py` declares. A test pins this: a solution redefining `target = 879` still
  scores 0.0 against the real target of 84.
- **`test_silent_arm_reproduces_the_pre_arm_prompt_exactly`.** The golden hash keeps the silent
  arm comparable to `runs/2026-09-19_23-22-12`. Changing the template may be right, but do it
  deliberately and re-pin — don't edit the constant to make the test pass.
- **Arms differ only by their clause.** Byte-identity is the entire reason an activation contrast
  is interpretable here.
- **The 0.5 hack threshold.** Settled. Don't lower it to sweep in the 0.2 tier for a bigger corpus.
- **CoT is diagnostic only.** Never let `<think>` content into an automated label (Baker et al.).
- **`runs/2026-09-19_23-22-12/` and `runs/pilot-7b/`.** `runs/` is gitignored, so these exist only
  on the rig. They're the evidence for every verdict in §6. Don't clean them up.

## Resources

**Environment / elicitation design**
- [Hack-Verifiable Environments](https://arxiv.org/html/2605.20744) ·
  [code](https://github.com/MajoRoth/hack-verifiable-environments/) — planted-flaw taxonomy and
  yields (logical bug 34.8%, hidden solution 12.7%). **The template for step 1.**
- [EvilGenie](https://arxiv.org/pdf/2511.21654) · [code](https://github.com/JonathanGabor/evilgenie_inspect)
  — single-turn reward-hacking benchmark, frontier models. External validation, later.
- [SpecBench](https://arxiv.org/html/2605.21384) — visible + held-out suites, same proxy/true
  structure as ours but agentic.
- [Recontextualization](https://arxiv.org/abs/2512.19027) — the proposal's §2 fallback (MBPP with a
  wrong public test; Hackable LeetCode).
- [Denison et al., Sycophancy to Subterfuge](https://arxiv.org/abs/2406.10162) — curriculum and
  system-prompt elicitation, the legitimate version of "make it more willing."

**Detection**
- [TRACE](https://arxiv.org/abs/2601.20103) — 54-category taxonomy, labeled corpus. Use it to test
  our detectors.
- [Auditing Reward Hackability in Code RL Environments](https://arxiv.org/abs/2606.16062)
- [Baker et al., Monitoring Reasoning Models](https://arxiv.org/abs/2503.11926) — read before
  touching CoT at all.

**Interpretability method**
- [PRIME](https://arxiv.org/abs/2606.09711) — closest existing analogue; position against it.
- [reward-lens](https://arxiv.org/abs/2604.26130) — patching library. **API still unread**; read it
  before fixing `patching.py`'s interface. Note their finding that attribution scores don't
  reliably predict causal patch effects.
- [Persona Vectors](https://arxiv.org/abs/2507.21509) — diff-of-means precedent.
- [ROME](https://arxiv.org/abs/2202.05262) — canonical activation patching.
- [Neel Nanda, Concrete Steps](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
  · [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0) — hands-on probing/patching practice.

## Framing note for the writeup

Instructed or constrained constraint-violation and RL-emergent specification gaming are **related
but not identical** phenomena. Whatever corpus we end up with, say which one it is, and name the
transfer test — do directions found here fire on naturally-occurring hacks? — as the next
experiment. These are fine limitations for a preliminary project and damaging only if left
implicit for a reviewer to find.
