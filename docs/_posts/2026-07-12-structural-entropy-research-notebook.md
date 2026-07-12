---
layout: post
title: "One Idea, Four Experiments: A Lab-Notebook Tour of Our Structural-Entropy Research"
date: 2026-07-12
description: "Structural entropy is a single information-theoretic measure of how cleanly a graph splits into a nested hierarchy of communities. We put it to work in four very different places — graph attention, reinforcement-learning state abstraction, and large-language-model confidence — and report what held up and what didn't. Headline: a task-relevant SE graph gives a real +14% return gain on one RL task (p=0.0073, 10 seeds) but does not generalize; the graph-learning result is the clean win (p<10⁻⁶); and the LLM-confidence effect is real but stubbornly gemma-family-specific. Every number here points to a committed artifact, and the corrections are reported as loudly as the wins."
---

<script>
  window.MathJax = {
    tex: { inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$'],['\\[','\\]']], processEscapes: true },
    options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

> **TL;DR.** Four research tracks, one measure. **(1) Graph attention** — the clean, significant win: a multi-level structural-entropy prior beats flat SE and modularity by 2–3.7 points of accuracy at $p<10^{-6}$, with the controls reviewers ask for. **(2) RL state abstraction (DM-SISA)** — a differentiable, GPU-native SE module that is ~16,000× cheaper than the discrete original and *matches* it on return; then a task-relevant variant that produces the project's first genuine return beat (**+14%, $p=0.0073$, 10 seeds**) on one task — but honestly *does not* generalize. **(3) LLM confidence (SeSE)** — reasoning-chain SE beats answer SE at catching a model's own errors, but only when the model is overconfident, and so far only inside the gemma model family. **(4) The benchmark itself** — the reproducible harness and provenance audit that keep all of the above honest. The single newsworthy headline: **task-relevance is what turns SE from "no effect" into a real RL gain — but that gain is task-specific, not a free lunch.**

### 60 seconds of background

**Structural entropy (SE)** is a number that measures how much uncertainty remains about "which community a random walker is in" once you know a graph's community structure. Low SE means the graph splits cleanly into groups. The idea (Li & Pan, 2016) comes in two flavors that matter enormously in practice: **flat SE** uses a single layer of communities; **multi-level SE** uses the full *encoding tree* — a nested hierarchy of communities-within-communities. Almost everything interesting below turns on that distinction, and on a second one: **is the graph you compute SE over built from raw features, or from something the task actually cares about?**

We tested SE as an inductive bias in three unrelated settings — graph neural networks, pixel-based reinforcement learning, and LLM uncertainty — because a good abstraction should help in more than one place. The discipline throughout: **numbers come only from produced artifacts; a claim without enough random seeds and a multiple-comparison correction is not a claim; and negatives and walk-backs are published, not buried.** Several apparently-promising signals in this program washed out under more seeds — we show them anyway, because that is the useful part.

---

## Track 1 — The Multi-Level Edge (graph attention)

**What it is.** Graph attention networks learn a weight for each edge but have nothing pushing those weights to respect the graph's community structure. We add a regularizer that pulls attention toward within-community edges, and we ask which *community objective* is the best regularizer: **flat SE, multi-level SE, modularity** (the classic community-quality score), or **MinCut** (a spectral clustering objective).

**Why it exists.** The whole SE program rests on a bet that the *hierarchy* — the multi-level encoding tree — carries information the flat, single-layer version cannot. Graph node-classification is the cleanest place to test that bet, because you can control for parameters and plant a known hierarchy.

**Status — shipped, and the strongest result we have.** Across **8 datasets × 15 seeds**, with Benjamini–Hochberg correction across all comparisons:

| Comparison | Effect | Significance |
|---|---|---|
| Multi-level SE vs flat SE | **+2.0 pp** accuracy | $p<10^{-6}$ |
| Multi-level SE vs modularity (raw) | **+3.7 pp** | $p<10^{-6}$ |
| Multi-level SE vs MinCut (each tuned) | +0.9 pp | $p=0.10$ (n.s.) |

Two controls make this a result rather than an artifact: a *bigger* flat-SE model is **worse** than multi-level SE (so the win is the hierarchy, not extra parameters), and on synthetic graphs with a *planted* depth-3 hierarchy the advantage peaks exactly at the true depth (L3 > L2, $p=0.0014$). **Reported nulls, at equal prominence:** the planted depth-2 effect *did not survive* 30 seeds (it looked promising at 15 and we walked it back), and a harder planted variant does not reproduce the depth result. The MinCut comparison is a *tie*, so we frame it as "tuning-free robustness," not a win.

**Feeds.** This is the lead result of the graph paper, and it supplies the "hierarchy is the lever" claim that the RL track (below) borrows to justify a multi-level SE module.

---

## Track 2 — DM-SISA: differentiable structural entropy for RL state abstraction

**What it is.** An agent learning to act from raw pixels benefits from a good internal abstraction of its states. **SISA** (from the SIDM family) builds one by minimizing SE over a graph of state transitions — but its implementation is *flat*, *discrete* (a greedy tree built on the CPU), and so slow it only runs on 1 update in 100. **DM-SISA** (ours) replaces that with a **differentiable, GPU-native, multi-level** soft-SE module that can run every update.

**Why it exists.** Two questions. First, engineering: can you make deep-tree SE differentiable and cheap enough to stop treating it as expensive preprocessing? Second, science: does an SE abstraction actually improve returns, or just add cost?

**Status — done, and it is a story of one honest win inside a larger null.**

*The engineering held cleanly.* The soft SE module matches the discrete objective to $|\Delta| = 1.15\times10^{-7}$ at hard assignments, and each abstraction call costs ≈1.3 ms versus ≈21.4 s for SISA's discrete tree — about a **16,000× speed-up**.

*The first return question was a null, honestly.* With matched hardware and seeds, DM-SISA neither beats nor loses to SISA or a no-abstraction baseline on return — the methods are statistically inseparable under high seed variance. Along the way we retracted five apparent "wins" (single-seed flukes and a two-seed "over-regularization" story that reversed at five seeds). The abstraction's *representations* did not buy return; the efficiency did.

*Then the reframe that worked.* The graph those experiments used was built from **feature cosine similarity** — it carries no task information, so its communities never helped. Rebuilding the graph to be **task-relevant** — reward-weighting the edges so that states with similar reward sit in the same community — produced the project's first genuine return beat:

| walker-walk, same hardware, 10 seeds | mean ± sd |
|---|---|
| **DM-SISA reward-graph (task-relevant)** | **756 ± 45** |
| No-abstraction baseline | 664 ± 86 |

That is **+14% return, $p=0.0073$** (two-sided rank test, under a Bonferroni threshold of 0.017 across the three graph variants tested), and the effect *strengthened* as seeds accumulated ($p$ ran 0.011 → 0.0027 → 0.0073 from 7 to 10 seeds) — the opposite of the seed-luck flukes above.

**The essential honesty — it does not generalize.** We tested the same reward-graph against the same baseline on five other tasks (3 seeds each). It is statistically indistinguishable from the baseline on two, and runs below it on three — but *none* of those differences is significant, and one apparent "−62% catastrophe" turned out to be a two-seed artifact that vanished when the third seed matched the baseline. A weaker *value*-weighted graph, tested the same way, mirrors the pattern: parity or below everywhere, nothing significant.

| Task | reward-graph vs baseline | verdict |
|---|---|---|
| **walker-walk** | **+14%** | **BEAT** ($p=0.0073$, 10 seeds) |
| cartpole / finger | −1% / −6% | parity |
| reacher / cheetah / ball-in-cup | −10% / −25% / high-variance | below, none significant |

So the defensible claim is precise: **a task-relevant SE graph produces a real, corrected-significant return gain where the feature-cosine graph and the plain baseline do not — but the benefit is task-specific, not a general property of task-relevant SE.** A scoped positive with a clear boundary, which is more useful than a false "beats the baseline everywhere."

**Feeds.** The efficiency + formulation result is a systems/workshop contribution; the walker beat + its honest generality boundary is the paper's core. It borrows Track 1's "multi-level hierarchy is the lever" and adds "and the *graph* must be task-relevant."

---

## Track 3 — SeSE: structural entropy as an LLM's self-doubt signal

**What it is.** When a language model answers a hard question, you want a confidence number that is high when it is right and low when it is wrong (so you can abstain or defer). We build that number two ways. **Answer-SE** computes structural entropy over the graph of *final answers* the model samples; **chain-SE** computes it over the graph of full *reasoning chains*. The graph edges come from a natural-language-inference model judging which answers/chains agree. The question: which SE catches the model's own errors better?

**Why it exists.** Reasoning models spend most of their tokens in the chain, not the answer. If the chain's *structure* — how much its many reasoning paths agree — carries calibration signal that the bare answer distribution misses, that is a cheap, model-internal error detector.

**Status — a real finding with a stubborn, honestly-reported boundary.** On the **gemma-3** model, chain-SE beats answer-SE at error detection — but only when the model is **overconfident** (peaked answer distribution, mediocre accuracy). Across the tested cells, the correlation between a model's overconfidence and chain-SE's advantage is strong, roughly $\mathrm{corr}\approx0.75$. That is the positive result, and it is anchored to the gemma family.

**The boundary, reported at equal prominence.** We then tested roughly ten *non-gemma* model families (Qwen, zephyr, Mistral, StableLM, and others) across grade-school-math, algebra, and word-problem benchmarks. **Every one is calibrated in the relevant sense — its answer-SE wins, not its chain-SE.** The sharpest single data point: **StableLM-2-1.6B is genuinely overconfident** (the precondition under which gemma's chain graph wins) **yet its chain graph still does not win** (answer-SE edges it out). So overconfidence *alone* is not enough — the effect appears to be a property of the gemma family specifically, not a generic consequence of miscalibration. A separate probe of a frontier model on hard math and graduate science put it firmly in the answer-wins, well-calibrated regime too.

We also debunked our own earlier cross-family "lead": a Mistral result that looked overconfident was an answer-parsing artifact (73% of answers were unparseable); fixing the extractor moved it from apparent chain-win to plain calibrated. That is exactly the kind of thing this benchmark exists to catch.

**Feeds.** The calibration paper, with its scope stated honestly ("gemma-anchored"). It also motivates the one **planned** track below.

---

## Track 4 — The benchmark and provenance audit (the honesty infrastructure)

**What it is.** A reproducible repository that pairs original method code with a shared evaluation harness, a public ledger of every experiment and lever tried, and a line-by-line audit of the survey's structural-entropy equations against their original sources.

**Why it exists.** SE has a book-length formal definition and a scattering of variant papers whose notation drifts. Before trusting any downstream result, you have to know the equations are transcribed correctly and that no experiment is quietly cherry-picked. The [equation-provenance audit]({{ site.baseurl }}/2026/06/27/se-equation-provenance.html) found and fixed two real transcription errors (an encoding-tree summation domain, and a hard-vs-soft constraint) and documented the benign abstractions.

**Status — shipped and continuously updated.** The ledger records the washed-out signals from every track above, so a future reader does not re-run a dead end.

**Feeds.** Everything. Tracks 1–3 all cite artifacts registered here; the ledger is why the nulls in this very post are traceable.

---

## Track 5 (planned) — Confidence fusion with complementary accuracy

**What it is.** Instead of *picking* answer-SE or chain-SE per model, **fuse** several confidence signals (answer-SE, chain-SE, unparseable-fraction, sample agreement) into one, weighted by which is most reliable per question. "Complementary accuracy" is the crux: fusion only helps when the signals err in *different* places, so that one covers another's mistakes.

**Why it exists.** It attacks Track 3's exact wall. A single SE signal is gemma-anchored because no one signal transfers across families. A fused signal *could* be robust across families — but only if the members are complementary rather than redundant.

**Status — planned; ~1–2 GPU-days, and mostly offline.** We already have per-question SE data saved for ~ten non-gemma families. The first experiment needs *no new sampling*: compute a weighted/learned fusion of the existing signals and check whether the **fused** error-detection score beats the **best single** signal on the calibrated families where chain-SE alone failed. If fused > best-single on held-out families, that is the cross-family result the gemma-anchored scope was missing. The honest risk: on well-calibrated models the two SE signals may be redundant (both just track the answer distribution), so fusion gains little — which we would report as a null.

**Feeds.** If it works, a stronger, less-scoped version of the Track 3 calibration paper.

---

## How the tracks connect

The through-line is one sentence: **structural entropy helps exactly when the structure it measures is both *hierarchical* and *task-relevant*, and you can only tell which by controlling for confounds and counting seeds.**

- **Track 1 (graph)** establishes the *hierarchy* half — multi-level SE beats flat SE, cleanly and significantly. This is the anchor result.
- **Track 2 (RL)** establishes the *task-relevance* half — the same multi-level SE does nothing with a feature-cosine graph, but produces a real (if task-specific) gain once the graph is reward-relevant. It inherits Track 1's "hierarchy is the lever" and adds "the graph must carry task information."
- **Track 3 (LLM)** is SE as a *confidence* signal rather than a training prior; its gemma-anchored boundary is the honest limit that **Track 5** is designed to push on, using data Track 3 already produced.
- **Track 4 (benchmark)** underwrites all of them: every number above points to an artifact it registers, and every null above is traceable because the ledger recorded it.

*Every figure in this post is drawn from a committed result artifact in the project repositories; corrections, ties, and nulls are stated with the same prominence as the wins. Read a year from now, the one durable lesson is that the wins in this program are the ones that survived more seeds and a control — and we kept the ones that didn't in plain sight.*
