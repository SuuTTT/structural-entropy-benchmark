# Response to Reviewers — and revision blueprint for the journal version

This document does two jobs at once:

1. It answers, point by point, every concern raised by the four IJCAI'25
   reviewers.
2. Each answer is a concrete, falsifiable commitment in the **journal (TGINA)
   version**, anchored to the reproducible **SE benchmark suite**
   (`benchmark/suite/`). Where a fix requires evidence, we name the experiment
   and where its result will live, rather than promising prose.

## The diagnosis the reviewers share

Read together, R1, R2, and R5 are describing one failure mode — the now-common
"broad-but-shallow survey":

- **R1:** topic is "Marginal / Redundant"; the survey **does not establish the
  *why* of SE**; theory (§3.1) is too high-level; §2.2 too long; *"it would be
  interesting to see whether code implementations are available."*
- **R2:** wants **comparison with other graph metrics**, **computational-
  complexity depth**, **more visuals**, and **limitations / failure cases**.
- **R5:** *"each section lacks a core argument explaining how SE essentially
  supports learning"*; future directions (LLM / embodied AI) **not in-depth**,
  no key scientific questions.

The single revision that resolves all of these at once is the one the journal
version is built on: **stop name-listing ~60 papers and instead run the methods,
measure when and why SE helps (and when it fails), and write the survey around
that evidence.** The benchmark suite *is* the response to the reviews. Below,
each concern is mapped to a specific element of that suite or manuscript.

---

## Reviewer #1

> **"Not establishing the *why* of structural entropy."** §3.1 gives it at a
> very high level; extend §2 on SE's properties and why it is useful; §2.2 is
> too long and hard to follow.

**Response.** Agreed; this is the central weakness and we restructure the paper
around fixing it.

- **A first-principles "why SE" section.** We replace the high-level §3.1 with a
  derivation-driven account of *what SE is a prior for*: SE is the expected
  description length of a random walk localized by a hierarchical partition —
  i.e., a **flow-based, hierarchy-aware structural prior**. We make explicit the
  three properties that motivate its use and that no single competing metric has
  together: (i) it is **hierarchical / multiscale** (an encoding tree, not a flat
  partition); (ii) it is **flow-based** (defined through the random-walk
  stationary process, so it responds to *dynamics on* the graph, not just edge
  counts); (iii) it has a **parameter-free optimum** (the resolution is selected
  by minimization rather than a free resolution parameter, unlike modularity).
- **Tighten §2.2.** The long "basic metrics" subsection is cut to the minimum
  needed (volume, cut, the 1-D = Shannon-entropy special case) and the reclaimed
  space goes to **properties and theory**: the 1-D ⇒ Shannon special case, the
  proven SE–von-Neumann-entropy gap bound [Liu et al. 2022], the resolution-limit
  comparison with modularity, and a worked small example showing *why* the
  optimal encoding tree recovers a planted hierarchy.
- **Code implementations (R1's explicit ask).** Every method in the survey is
  linked to its **original public repository, pinned by commit**, in
  `benchmark/suite/registry/methods.yaml`; 11 of the 12 core repos are already
  verified live. We additionally ship our own runnable suite. *(Correction owed
  to the record: the conference draft's benchmark text claims DeSE has "no public
  code available" — this is false; `github.com/SELGroup/DeSE` exists and will be
  reproduced.)*

> **Topic relevance "Marginal", survey "Redundant — does not situate itself
> distinctively."**

**Response.** The distinctiveness is the empirical contribution: to our
knowledge this is the **first cross-family, original-code reproduction of SE
methods under a common protocol**, including the comparisons (vs. modularity /
map-equation / spectral / learned baselines) and the failure-case analysis that
prior SE surveys — and the conference version — lack. The journal version is
positioned as a **benchmark-and-survey**, which is a distinct artifact from the
existing high-level SE overviews.

---

## Reviewer #2

> **More comparative analysis with other graph-analysis metrics.**

**Response.** This is a core deliverable, not a paragraph. The suite includes a
**cross-objective evaluation**: for every method's output we report its
modularity, map-equation codelength, and SE, so the reader sees *which objective
each method actually optimizes and whether they agree*. The community-detection
phase runs SE methods head-to-head with **Louvain, Leiden, Infomap, normalized-
cut spectral, and DMoN** across an **LFR mixing-parameter sweep** (the canonical
axis on which modularity/map-equation/SE are known to diverge). Theory side: a
dedicated subsection relates SE to modularity (resolution limit), the map
equation (both are codelength functionals — we make the relationship precise),
and spectral cuts (via the SE–VNE gap bound).

> **More technical depth on computational complexity.**

**Response.** We add a complexity-and-scalability section with (i) a corrected
complexity table cross-checked against each *original* implementation — the
conference draft asserts deDoc is "O(N³) / infeasible for N>50", which conflicts
with the original deDoc paper's near-linear claim; we resolve this empirically by
timing the real Java implementation across graph sizes — and (ii) measured
wall-clock vs. N curves on the fleet, so complexity claims are checked, not
quoted.

> **Additional visual representations of key concepts.**

**Response.** New figures: (a) a worked encoding-tree example with the SE of each
candidate tree annotated (the "why" made visual); (b) the LFR μ-sweep accuracy
curves (SE vs. baselines); (c) runtime-vs-N scalability curves; (d) a
cross-objective agreement heatmap. These replace name-only taxonomy real estate.

> **Limited discussion of limitations / failure cases of SE.**

**Response.** Failure cases become a **named section with evidence**, e.g.:
SE/topology-only methods degrade when ground-truth labels are *semantic* not
*topological* (already visible on Cora/Citeseer, where topology-only SE trails
feature-aware methods); behavior near the LFR detection threshold; the
degenerate-zero-gradient issue for continuous-relaxation SE on perfectly
disjoint cliques; and any non-reproductions we hit. Negative results are
reported, not hidden.

---

## Reviewer #3

No written comments were provided. The structural improvements above
(clearer core arguments, comparisons, visuals, complexity, reproducibility)
address the implicit "short yet illuminating" bar in the CFP.

---

## Reviewer #5

> **Each section lacks a core argument for how SE essentially supports learning
> and cross-domain applications.**

**Response.** Each technical section is rewritten to open with **one explicit,
testable thesis**, then defend it with benchmark evidence:
- *Community detection:* "SE's hierarchical, flow-based prior recovers planted
  structure competitively with modularity/map-equation and is most useful when
  the true structure is genuinely multiscale or near the detection threshold."
- *Graph learning:* "An SE-derived hierarchy is a useful inductive bias for
  pooling/structure-learning **only when** class structure aligns with
  topological community structure; we quantify when it does." (Tested in Phase 2
  vs. DiffPool/MinCutPool/TopK/SAGPool and a no-pool baseline.)
- *RL:* "SE provides an automatic, label-free hierarchy over states/actions that
  improves exploration/credit assignment; we reproduce the cleanest such claim
  per method and report effect sizes with seeds." (Phase 3.)

> **Future directions (LLM / embodied AI) not in-depth; no key scientific
> questions; unclear how SE is used in LLMs / VLMs.**

**Response.** We replace the hand-wave with **specific, grounded open problems**,
each tied to a concrete mechanism and recent work surfaced in our literature
sweep: SE for **LLM hallucination / uncertainty quantification** over claim/
semantic graphs (e.g., SeSE-style structural uncertainty), **GraphRAG retrieval
structuring** via encoding trees, **SE as a structural prior for tokenization /
representation** (SECodec), and **hierarchy discovery for embodied/long-horizon
agents** (SIDM/SI2E). Each is stated as a key scientific question
("Does minimizing SE over an LLM's retrieved-evidence graph reduce hallucination
relative to flat retrieval?") rather than a topic name — and we are honest about
which are speculative vs. already evidenced.

---

## Summary of concrete changes (conference → journal)

| # | Reviewer concern | Change | Where it lives |
|---|---|---|---|
| 1 | "Why SE" not established (R1) | First-principles prior + 3 properties + worked example | new Theory/§2 |
| 2 | §2.2 too long (R1) | Cut to essentials; space to properties/theory | §2 |
| 3 | Code availability (R1) | Verified original-repo registry + runnable suite | `suite/registry`, `suite/` |
| 4 | "Redundant" positioning (R1) | Reframe as first cross-family original-code SE benchmark | Intro/positioning |
| 5 | Comparison w/ other metrics (R2) | LFR μ-sweep + cross-objective table vs modularity/map/spectral/DMoN | Phase 1, Theory |
| 6 | Complexity depth (R2) | Corrected table + measured runtime-vs-N | Complexity §, Phase 1 |
| 7 | More visuals (R2) | Encoding-tree, μ-sweep, scalability, objective-agreement figures | Figures |
| 8 | Failure cases (R2) | Dedicated evidence-backed limitations section | Limitations § |
| 9 | Per-section core argument (R5) | Each section opens with a tested thesis | all technical § |
| 10 | LLM/embodied depth (R5) | Specific scientific questions tied to mechanisms+refs | Open Problems |
| — | Factual error | DeSE "no public code" claim is false; reproduce it | `cards/community_detection/DeSE.md` |
| — | Reproducibility | deDoc "O(N³)" claim re-checked against original Java impl | `cards/community_detection/deDoc.md` |

**Bottom line for the editor:** the conference version was a map of the field;
the journal version is a *measurement* of it. Every reviewer concern is answered
by the same shift — from listing SE methods to running them, comparing them
fairly, and reporting honestly where SE works, where it does not, and why.
