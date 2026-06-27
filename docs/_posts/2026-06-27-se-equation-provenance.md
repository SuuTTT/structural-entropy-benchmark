---
layout: post
title: "Do the Survey's Structural-Entropy Equations Match Their Sources? A Provenance Audit"
date: 2026-06-27
description: "We checked every structural-entropy equation in our survey against its original source — the SE book for the core definitions, and the cited paper for each variant. Two real errors found and fixed (the encoding-tree summation domain, and a hard-vs-soft constraint), plus a handful of benign table-level abstractions documented. This post shows, side by side, the survey form, the source form, and the verdict for all eight equations."
---

> Surveys are where definitions go to drift. A formula gets transcribed from a
> paper, simplified for a table, recombined across variants — and small index or
> constraint changes creep in. So we audited **every structural-entropy (SE)
> equation** in the survey against its origin: the **SE book** (Li, *Science of
> Artificial Intelligence*, 2024) for the core definitions, and the **cited
> paper** for each variant in the variants table. Below is the side-by-side for
> all eight, with verdicts. Two were genuinely wrong and are now fixed.

## Method

Each equation was compared against a primary source fetched and read directly —
the book's Chapter 9 for the core definitions (1D / 2D / encoding-tree SE), and
the original paper (local PDF or arXiv) for each variant. We classify each as
**MATCH** (faithful), **MINOR** (a benign table-level abstraction worth noting),
or **MISMATCH** (a real error, fixed). The convention throughout: $G=(V,E)$,
degree $d_v$, $\mathrm{vol}(G)=\sum_v d_v=2m$, encoding tree $T$ with root
$\lambda$, node $\alpha$ with parent $\alpha^-$, and $V_\alpha$ the volume
(degree mass) of the vertices under $\alpha$.

---

## Core definitions (vs the SE book)

### 1. Encoding-tree SE — **MISMATCH → fixed**

**Survey (before):**
$$\mathcal{H}^T(A) = -\!\!\sum_{\substack{\alpha\in T\\ \alpha^-\neq\lambda}} p_\alpha\log_2\frac{V_\alpha}{V_{\alpha^-}}$$

**Book (Def. 9.8 / 9.15, eqs. 9.11, 9.19):**
$$\mathcal{H}_g^T(G) = -\!\!\sum_{\substack{\alpha\in T\\ \alpha\neq\lambda}} \frac{g(T_\alpha)}{\mathrm{vol}(G)}\log_2\frac{\mathrm{vol}(\alpha)}{\mathrm{vol}(\alpha^-)}$$

The book sums over **all non-root nodes** $\alpha\neq\lambda$. The survey's
condition $\alpha^-\neq\lambda$ (parent $\neq$ root) wrongly **drops the
top-level module terms** — the children of the root. It was also internally
inconsistent: the survey's own variants table already used $\alpha\neq\lambda$.
**Fixed** to $\alpha\neq\lambda$. (Here $p_\alpha=g(T_\alpha)/\mathrm{vol}(G)$ is
the boundary-cut probability, the Markov-chain reading of the book's
modulus-function cut.)

### 2. One-dimensional SE — **MATCH**

**Survey:** $\;\mathcal{H}^{(1)}(G)=-\sum_v \frac{d_v}{\mathrm{vol}(G)}\log_2\frac{d_v}{\mathrm{vol}(G)}$

**Book (Cor. 9.13, eq. 9.17):** $\;\mathcal{H}^1(G)=-\sum_i \frac{d_i}{2m}\log_2\frac{d_i}{2m}$

Identical ($\mathrm{vol}(G)=2m$): the Shannon entropy of the degree
(stationary) distribution.

### 3. Two-dimensional SE — **MATCH**

**Survey:**
$$\mathcal{H}^{(2)}(G;\mathcal{P}) = -\sum_{X\in\mathcal{P}}\Big[\frac{g_X}{\mathrm{vol}(G)}\log_2\frac{V_X}{\mathrm{vol}(G)} + \sum_{v\in X}\frac{d_v}{\mathrm{vol}(G)}\log_2\frac{d_v}{V_X}\Big]$$

This is exactly the depth-2 specialization of the encoding-tree formula (the
standard Li–Pan 2D form): a module-entry term plus a within-module positioning
term, with a leaf's cut equal to its degree.

---

## Variants table (vs each cited paper)

### 4. Directed / flow SE — **MATCH** (minor: teleportation)

**Survey:** $\;\mathcal{H}^T(A),\quad V_\alpha=\sum_{v\in\alpha}\pi_v,\quad \boldsymbol\pi=\boldsymbol\pi P,\;\; P=D_{\mathrm{out}}^{-1}A$

Both Yao & Pan (flow-based clustering on directed graphs) and Zhang et al.
(RWE, IEEE TNSE 2024) define directed-SE volumes as sums of the **stationary
(Perron / PageRank) distribution** of the out-degree-normalized walk
$P=D_{\mathrm{out}}^{-1}A$, fed into the same encoding-tree functional —
verbatim ($p(u,v)=w(u,v)/d_{\mathrm{out}}(u)$). *Minor:* both add PageRank
teleportation ($\alpha\approx0.15$) to guarantee a unique $\pi$ on graphs with
sinks; the table shows the no-teleport core. (RWE's headline *1D* measure is a
von-Neumann/Shannon-of-stationary quantity; its *2D* module-coding form is the
one that parallels this row.)

### 5. Multi-relational SE — **MATCH** (minor: coupled fixed point)

**Survey:** $\;\mathcal{H}^T(A),\quad V_\alpha=\sum_{v\in\alpha}\pi^{\mathrm{mr}}_v,\quad \boldsymbol\pi^{\mathrm{mr}}=\boldsymbol\pi^{\mathrm{mr}}P^{\mathrm{mr}}$

Cao et al. (MrSE, UAI 2024) plug a **MultiRank** stationary law of a surfer that
*jointly* picks a node and a relation type into the same SE functional — exactly
a "multi-rank," not relation-collapsed edge-weighting. *Minor:* the source's
$\pi^{\mathrm{mr}}$ is really a **coupled pair** of fixed-point equations (a node
vector $x'$ and a relation vector $y$); the survey's single-operator
$\pi^{\mathrm{mr}}=\pi^{\mathrm{mr}}P^{\mathrm{mr}}$ is a fair shorthand.

### 6. Semi-supervised SE — **MISMATCH → fixed**

**Survey (before):** $\;\min_{\mathcal{P}}\mathcal{H}^{(2)}(G;\mathcal{P})\;\text{ s.t. }\;\mathcal{M}\subseteq\text{same},\ \mathcal{C}\subseteq\text{diff}$

**Source (Zeng et al., SSE, SDM 2024, eqs. 3.5–3.6):**
$$\min_{\mathcal{P}}\ \mathcal{H}^{(2)}(G;\mathcal{P}) + \varphi\,E_{\mathcal{P}}(G,G'),\qquad E_{\mathcal{P}}(G,G')=-\sum_{X\in\mathcal{P}}\frac{g'_X}{\mathrm{vol}(G)}\log_2\frac{V_X}{\mathrm{vol}(G)}$$

SSE does **not** hard-constrain the partition. Must-link / cannot-link are
encoded as **signed edges of a constraint graph** $G'$ and folded in as a
**soft penalty** $\varphi\,E_{\mathcal{P}}$ ($\varphi=2$) traded off against the
SE — constraints *can* be violated. The survey's "s.t." hard-constraint framing
misrepresented the mechanism. **Fixed** to the penalty form.

### 7. Incremental / dynamic SE — **MATCH** (minor: schematic $\Delta\mathcal{H}$)

**Survey:** $\;\mathcal{H}^{(2)}(G_{t+1})=\mathcal{H}^{(2)}(G_t)+\Delta\mathcal{H}(e_t)$

Yang et al. (Incre-2dSE, *Artificial Intelligence* 2024) maintain **2D** SE under
batched edge updates by tracking per-community degree/volume/cut-edge increments
and adjusting the encoding tree (naive and node-shifting strategies). The
additive schematic is directionally faithful, provided $\Delta\mathcal{H}$ is
read as a **localized incremental recomputation over affected tree nodes** (plus
tree adjustment), not a closed-form scalar. Left as-is (it is a schematic).

### 8. Continuous / differentiable SE — **MATCH** (parent index made precise)

**Survey (now):**
$$\mathcal{H}^{\mathcal{T}}(G)=\sum_{h,k}-\frac{1}{\mathrm{vol}(G)}\Big(V^{h}_{k}-\!\!\sum_{(i,j)\in E}\!S^{h}_{ik}S^{h}_{jk}w_{ij}\Big)\log_2\frac{V^{h}_{k}}{V^{h-1}_{k^-}}$$

This reproduces **LSEnet** (Sun et al., ICML 2024, Def. 4.3, eq. 4) **verbatim**:
the differentiable soft-cut $g=V^h_k-\sum_{(i,j)\in E}S^h_{ik}S^h_{jk}w_{ij}$
(volume minus internal soft edge weight) and the cumulative soft assignment
$S^h=\prod_k C^k$; DeSE (Zhang et al., KDD 2025, eq. 6) gives an equivalent
per-layer form. The one fix here: the log denominator was $V^{h-1}_{k}$ but the
source uses the **parent-indexed** $V^{h-1}_{k^-}$ (child-volume / parent-volume
across a tree edge). Corrected.

---

## Scorecard

| # | Equation | Source | Verdict |
|---|----------|--------|---------|
| 1 | Encoding-tree SE | SE book, eq. 9.11/9.19 | **MISMATCH → fixed** ($\alpha^-\neq\lambda \to \alpha\neq\lambda$) |
| 2 | 1D SE | SE book, eq. 9.17 | MATCH |
| 3 | 2D SE | SE book / Li–Pan 2016 | MATCH |
| 4 | Directed / flow SE | Yao & Pan; RWE (TNSE'24) | MATCH (teleportation noted) |
| 5 | Multi-relational SE | Cao et al., UAI 2024 | MATCH (coupled fixed point) |
| 6 | Semi-supervised SE | Zeng et al., SDM 2024 | **MISMATCH → fixed** (soft penalty) |
| 7 | Incremental / dynamic SE | Yang et al., AIJ 2024 | MATCH (schematic $\Delta\mathcal{H}$) |
| 8 | Continuous / differentiable SE | Sun et al., ICML 2024 | MATCH (parent index fixed) |

**Net:** two real errors caught and corrected against the originals; the rest
faithful, with three table-level abstractions documented rather than silently
kept. The core definitions (1D / 2D / encoding-tree) now agree with the SE book
exactly.

## References

1. A. Li. *Science of Artificial Intelligence: Mathematical Principles of Intelligence* (the "SE book"), 2024. — core definitions, Ch. 9 (eqs. 9.11, 9.17, 9.19).
2. A. Li and Y. Pan. Structural Information and Dynamical Complexity of Networks. *IEEE Trans. Information Theory*, 2016. — 1D/2D/k-D SE.
3. Y. Yao and Y. Pan. Flow-Based Clustering on Directed Graphs. — directed/flow SE volumes via PageRank stationary distribution.
4. Zhang et al. RWE: A Random-Walk Based Graph Entropy. *IEEE Trans. Network Science and Engineering*, 2024.
5. Cao et al. Multi-Relational Structural Entropy. *UAI*, 2024 (arXiv:2405.07096) — MrSE / MultiRank.
6. Zeng et al. Semi-Supervised Clustering via Structural Entropy (SSE). *SDM*, 2024 (arXiv:2312.10917) — soft-penalty constraint.
7. Yang et al. Incremental Measurement of Structural Entropy for Dynamic Graphs. *Artificial Intelligence* 334, 2024 (arXiv:2207.12653) — Incre-2dSE.
8. Sun et al. LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering. *ICML*, 2024 — differentiable SE (Def. 4.3).
9. Zhang et al. Unsupervised Graph Clustering with Deep Structural Entropy (DeSE). *KDD*, 2025 (arXiv:2505.14040).

*Companion to the structural-entropy survey. Source code and full results:
[github.com/SuuTTT/structural-entropy-benchmark](https://github.com/SuuTTT/structural-entropy-benchmark).*
