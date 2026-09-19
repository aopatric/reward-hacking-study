# Project proposal: mechanistic precursors of reward hacking in RLVR
### A preliminary interpretability project (pre-MEng, portfolio + fellowship prep)

---

## 0. Purpose of this document, and instructions for whoever picks this up next

This is a handoff brief. The immediate next step is a **literature review pass**, not implementation. Scope for that pass:
- Expand the citation graph outward from the anchor papers in §6, especially forward-citations (who has cited PRIME, Countdown-Code, and the Recontextualization paper since they were posted).
- Answer the open questions in §7.
- Flag anything that changes the environment or method choices in §2–§4 — those choices were made from search-based research, not hands-on testing, so they should be treated as a strong starting point, not a fixed spec.

**Do not** design or scope the training-intervention idea in §8 further yet — it's explicitly deferred to the actual MEng thesis, not this preliminary project. Keep this handoff's output to literature and (if asked later) environment/implementation feasibility notes.

**Ground rule for any agent that helps implement this later:** the point of this project is for the author to *learn* mechanistic interpretability hands-on, not to receive a finished artifact. Explanations of *why* a design choice is made (reward design, probe methodology, patching setup) matter as much as the code. Prefer walking through design decisions collaboratively over silently producing a complete solution.

---

## 1. One-line thesis framing

**Preliminary project question:** In a small RLVR coding environment where reward hacking is known to emerge, can we find an internal (activation-level) signal that predicts a hack *before* it manifests behaviorally, and is that signal causally implicated (not just correlated)?

**Longer-term thesis framing (not this project, but what this project is building toward):** Can that signal be turned into a training-time intervention on the *policy itself*, applied dynamically during RL, that suppresses reward hacking without merely teaching the model to hide it? (See §8 — this is the differentiator from existing work, but it is thesis-scale, not preliminary-project-scale.)

---

## 2. Environment

**Primary choice: adopt and adapt [Countdown-Code](https://github.com/zohaib-khan5040/Countdown-Code)** (Khalifa et al., [arXiv:2603.07084](https://arxiv.org/abs/2603.07084)).

Why this one instead of building from scratch: it's purpose-built for studying reward-hacking emergence under RLVR, and its "dual-access" design — the model writes code to solve a Countdown-style arithmetic task, and can also see/manipulate the test harness that grades it — gives a clean, built-in separation between:
- **Proxy reward** — did the visible test harness say pass (gameable)
- **True reward** — is the underlying solution actually correct (not gameable)

This split is the foundation for hack detection (§3), so adopting an environment that already has it cleanly built in saves real design risk. Code is open-sourced; first implementation task is porting/adapting it to run with the model/stack below.

**Fallback if Countdown-Code proves awkward to adapt:** environments from [Recontextualization Mitigates Specification Gaming](https://arxiv.org/abs/2512.19027) — either a modified MBPP setup (public test cases shown to the model, one deliberately wrong, reward = # public tests passed) or their "Hackable LeetCode" setup (model can win by overwriting a `run_tests()` function directly). Both use small, well-known datasets (MBPP), which lowers setup risk.

**Model + training stack:** Qwen2.5-1.5B-Instruct + Hugging Face `trl`'s `GRPOTrainer` + LoRA. This combination is already documented and working (HF's own GRPO quickstart, Modal's GRPO+TRL tutorial use this exact model), so there's no exotic infra risk — the open problems are the environment and the analysis, not the training loop.

**Reward-design caution, read before finalizing the reward function:** see [this short writeup](https://ishanjmukherjee.github.io/reward-hacking-grpo) of a naive length-based GRPO reward causing Qwen to collapse into repetitive nonsense in under 0.01 epochs. That's degenerate collapse, not the gradual, interpretable emergence of hacking this project needs — it's a useful sanity check for why the reward function needs enough headroom that genuine task-solving and hacking are both viable strategies the policy has to choose between.

---

## 3. Hack detection (must be automatic, not manual)

**Primary signal (ground truth):** the proxy/true reward gap built into the environment. A rollout counts as a hack when `R_proxy` is high but `R_true` is low. This is automatic, scales to however many rollouts you need, and requires no manual labeling.

**Secondary/confirmatory signal:** static-analysis pattern matching on generated code (hardcoded outputs matching only visible tests, direct modification of the harness/test-runner, suppressed exceptions). Reference taxonomy of patterns: [TRACE / Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis](https://arxiv.org/abs/2601.20103) (54-category taxonomy of code reward exploits — don't need all 54, but useful for sharpening the detector).

**Tertiary/diagnostic-only signal:** reading the chain-of-thought for explicit verbalized planning of the exploit. **Important constraint:** do not use CoT-visible hacking as a training signal (e.g., as an extra penalty) — see [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) (OpenAI, Baker et al.), which shows this teaches the model to hide the reasoning rather than stop the behavior. Use CoT only as a qualitative sanity check on a sample of labeled rollouts.

**Detection pipeline, in order:** outcome-gap label (ground truth) → static-analysis heuristic (confirms/refines) → CoT reading (qualitative spot-check only).

---

## 4. Interpretability methods (in scope for this project)

In rough order of implementation cost:

1. **Diff-of-means / concept vectors.** No training required: mean activation over hacking rollouts minus mean activation over non-hacking rollouts, per layer. Gives both a candidate detector direction and a candidate steering vector. Methodological precedent: [Persona Vectors](https://arxiv.org/abs/2507.21509) uses exactly this to extract behavior-associated directions.
2. **Linear probing, swept across layers.** Train a probe per layer on the labeled (hack vs. non-hack) activations; the layer where probe accuracy first rises sharply is a first-pass answer to "where does the divergence originate."
3. **Activation patching / causal tracing.** Swap activations between a hacking and non-hacking rollout at a given layer/position and check whether the outcome flips. This is what turns "this layer correlates with hacking" into "this layer causes it" — the key upgrade over probing alone. [reward-lens](https://github.com/suhailnadaf509/reward-lens) ([arXiv:2604.26130](https://arxiv.org/abs/2604.26130)) is a ready-made library implementing exactly this (three-mode activation patching, reward-hacking probe suite) — worth adopting rather than reimplementing, and useful if a learned reward model enters the picture later. Note their finding that attribution scores don't reliably predict causal patch effects — don't trust attribution alone as a stand-in for patching.
4. **Steering with the found direction, at inference time only.** Add/subtract the direction during generation and observe whether the hack rate changes. This is a cheap causal validation that pairs with steps 1–2, and it's the *last* step in scope for this project — no retraining loop.

**Explicitly out of scope for this project (deferred to thesis, see §8):** training an SAE from scratch, model diffing across training checkpoints, and any modification of the training loss/loop. These are either too expensive for a first pass or depend on this project's causal-validation step succeeding first.

---

## 5. Milestones (time-boxed; each has an explicit stopping rule)

| # | Milestone | What it teaches | Stopping rule |
|---|---|---|---|
| 0 | Stand up Countdown-Code (or fallback) with Qwen2.5-1.5B-Instruct + GRPOTrainer + LoRA; confirm the hack rate rises over training steps | RLVR training loop mechanics, GRPO specifics, reward-function debugging | If hacking doesn't emerge or the model degenerately collapses (see §2 caution) after reasonable reward-shape iteration, switch to the fallback environment rather than continuing to tune indefinitely |
| 1 | Build the detection pipeline (§3) and produce a labeled dataset of hack vs. non-hack rollouts | Practical eval design, the proxy/true reward distinction as a research tool, not just a training detail | Labeling pipeline agrees with a manual spot-check on ~30 rollouts before scaling up |
| 2 | Diff-of-means + layer-swept linear probes (§4.1–4.2) | Core representation-reading toolkit (activation extraction, probing methodology) | A clear layer-accuracy curve, even if the result is "no clean localization" — that's still a valid, reportable finding |
| 3 | Activation patching / steering to causally test the direction from Milestone 2 (§4.3–4.4) | The correlation-vs-causation distinction in interpretability, patching mechanics via reward-lens | A patching result (positive or negative) plus a short writeup — this is the deliverable |

**Stretch milestone (optional, only if time remains — do not let this delay the write-up above):** a no-retraining ablation where the steering vector from Milestone 3 is applied throughout a full evaluation run to see if it measurably shifts the aggregate hack rate. This is *not* the training-loss intervention from §8 — no gradient updates, just inference-time steering evaluated at scale.

**Deliverable:** a repo + a short (3–5 page or blog-post-style) writeup covering Milestones 0–3. This is the artifact for fellowship applications and advisor conversations — it doesn't need the stretch milestone to be complete and useful.

---

## 6. Reading list

**Model organism / environment design**
- Denison et al., [Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models](https://arxiv.org/abs/2406.10162) — foundational curriculum-design methodology for reward-hacking model organisms (Anthropic).
- Khalifa et al., [Countdown-Code](https://arxiv.org/abs/2603.07084) — primary environment.
- [Recontextualization Mitigates Specification Gaming without Modifying the Specification](https://arxiv.org/abs/2512.19027) — fallback environment + a mitigation baseline worth understanding.

**Detection**
- [Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis (TRACE)](https://arxiv.org/abs/2601.20103)
- [Auditing Reward Hackability in Code RL Training Environments](https://arxiv.org/abs/2606.16062)
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) — read before touching CoT at all.

**Core interpretability-of-reward-hacking (anchor papers — closest existing analogues to this project)**
- [Proxy Reward Internalization and Mechanistic Exploitation (PRIME)](https://arxiv.org/abs/2606.09711) — nearly the same experiment; this project should explicitly position itself as replicating/extending a piece of this.
- [From Reward-Hack Activations to Agentic Risk States](https://arxiv.org/abs/2606.06223)
- [reward-lens](https://arxiv.org/abs/2604.26130) — tool + paper.
- [SAFER: Probing Safety in Reward Models with Sparse Autoencoders](https://arxiv.org/abs/2507.00665)

**General mechanistic-interpretability method background**
- [Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in LLMs](https://arxiv.org/abs/2601.14004)
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)

**Signal-to-intervention prior art** (background/differentiation reading for the thesis rescope in §8 — not implementation targets for this preliminary project)
- [IR³: Contrastive Inverse RL for Interpretable Detection and Mitigation of Reward Hacking](https://arxiv.org/abs/2602.19416) — SAE-decomposes the implicit reward model, mitigates via feature-guided distillation. Closest existing "signal → intervention" work, but targets the reward model, not the policy.
- [HARVE: Hacking-Aware Reward-Head Vector Editing](https://arxiv.org/abs/2606.03131) — training-free reward-head editing.
- [Circuit-Aware Reward Training (CART)](https://arxiv.org/abs/2509.24713) — circuit-guided reward-model regularization.
- [Circuit Breakers: Improving Alignment and Robustness with Circuit Breakers](https://arxiv.org/abs/2406.04313) — the general technique for turning a bad direction into a differentiable training-time loss (representation rerouting via LoRRA), applied to harmfulness, not reward hacking.
- [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
- [BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](https://arxiv.org/abs/2602.00767) — closest structural analogue to the eventual thesis mechanism (a one-sided latent-blocking regularizer added during training), but for SFT-induced persona drift, not RL-induced reward hacking.
- [Reinforcement Learning Amplifies Emergent Misalignment from Harmless Rewards](https://arxiv.org/abs/2605.31328) — bridges the RL and emergent-misalignment literatures directly.

---

## 7. Open questions for the lit-review pass

1. What model scale(s) was Countdown-Code originally validated at? Confirm whether 1.5B is large enough to show a clean emergence curve, or whether the environment needs difficulty/reward-shape adjustment at this scale.
2. Since PRIME (§6) already runs a very similar experiment — what exact environment, model scale, and probe methodology did they use? Where does this project's setup diverge (different environment, different model, added causal-patching step), and is that divergence enough to be a genuine extension rather than a re-run?
3. Are there other small-model-friendly, open-source gameable-coding-task environments (beyond Countdown-Code and the Recontextualization MBPP/LeetCode setups) published in the last ~6 months that would be a better fit?
4. Has anyone combined reward-lens-style activation patching specifically with a Countdown-Code-style dual-reward environment? (I.e., is the Milestone 3 causal-validation step itself already done somewhere, which would change how this project should position its contribution.)
5. For the eventual thesis (§8): has anyone applied a circuit-breaker/latent-blocking-style loss *during RL* (as opposed to during SFT, which is what BLOCK-EM and Circuit Breakers do)? This is the load-bearing novelty claim and should be checked as thoroughly as possible.

---

## 8. Thesis-scale extension (context only — not this project's scope)

The rescoped thesis direction this preliminary project is building toward: characterize the hacking-precursor signal's stability/generalization across hack types (using the Milestone 3 causal-validation tooling), then test whether a circuit-breaker-style representation-rerouting loss, applied to **the policy's own activations, tracked and re-estimated dynamically during RL** (not the reward model, and not a static pre-RL direction), can suppress reward hacking — with the true-reward gap (§3), not the probe score, as the only metric that counts as success. Two known risks to design around when this is scoped for real: (a) the precursor direction likely drifts as the policy updates during RL, requiring periodic re-fitting rather than a single frozen probe; (b) penalizing a detected direction risks teaching the policy to hack via an undetected pathway instead of actually stopping — the representation-level analogue of the CoT-obfuscation risk in §3.
