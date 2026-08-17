# Lynxo and the Kadero Embedding: A Learnable Geometric Representation for Power-Network Criticality

### Mathematical Foundations and Empirical Validation — Archive Edition v1.0

**Author:** Abdelkader
**Field:** Independent work by a student of Artificial Intelligence / Computer Science, outside the author's primary field of training (this project sits at the intersection of spectral graph theory and power-systems engineering, neither of which the author was formally trained in — stated here plainly, not as a disclaimer to be read past, but as context for how this document should be read).
**Status: ARCHIVED, not abandoned.** The mathematical foundation (Section 3) is complete and, for its central claims, independently verified. The empirical validation program (Section 4) is honest, self-correcting, and currently inconclusive on the framework's central practical question. Development is paused, by the author's own informed decision, because completing it further requires depth in electrical engineering and applied mathematics beyond what one person working alone, without formal training in either field, can responsibly provide. It is archived, openly, in the hope that it may be useful, extended, corrected, or completed by someone with that depth — with or without further involvement from the original author, and with or without attribution, either of which the author would consider a good outcome.

**License and reuse.** This work — mathematics, code, and results — is released for unrestricted reuse, extension, correction, and publication by anyone who finds it useful. No permission is needed. A citation is welcomed if convenient but is not a condition of use.

**How to use this document:** Sections 1–3 (philosophy, related work, mathematics) are stable and should only change if the framework itself changes. Section 4 (experimental program) is an append-only log — Appendix C gives the exact template to paste a new experiment report into as a new subsection. Section 5 (discussion) and Section 6 (roadmap) should be revisited and edited each time a new experimental phase completes.

---

## Abstract

Modern power-system decision support still largely reasons about the grid through representations designed decades ago: the topological graph, the bus admittance matrix $Y_{\mathrm{bus}}$, the DC power-flow approximation, and the classical graph Laplacian. Each is a valid, physically grounded object, but each captures only one facet of what makes a piece of grid infrastructure operationally critical — electrical strength, geographic reality, topological redundancy, or the current operating point — and none of them is *learnable*: none can be calibrated against observed outage or contingency data the way a modern statistical model can. This project, Lynxo, develops an alternative representation, the **Kadero Embedding**, that lifts a physical power network into a high-dimensional vector space in which electrical, geographic, redundancy, and operating-point information are jointly encoded as separate, mathematically justified channels, each built by spectral factorization of a symmetric matrix with a provable meaning, and all differentiable with respect to a registry of free parameters that a later training stage is meant to determine from data. We give the complete mathematical construction, with proofs for every non-trivial claim, including two properties confirmed by external, independent cross-validation rather than asserted: a conservation law on the total "criticality mass" of the network, and an exact equivalence between the framework's edge-criticality score and topological bridge status, checked against an unrelated combinatorial bridge-finding algorithm with zero discrepancies. We then report, in full and without omission, five phases of empirical testing on IEEE 14-, 30-, 118-bus, and 300-bus (case300) systems: physics-only baseline verification, comparison against classical edge betweenness centrality, N-1 contingency correlation before and after a first learning experiment, a methodology correction discovered mid-project that reverses part of an earlier conclusion, and a head-to-head comparison against classical machine-learning baselines under honest cross-validation. The central empirical finding, stated plainly, is a **null result**: on the networks and severity definitions tested so far, the learnable, physics-derived Kadero score has not yet demonstrated statistically distinguishable predictive advantage over a simple classical model built from off-the-shelf topological and electrical features, once the easy, exactly-solvable bridge-detection component of the problem is set aside. This is not treated as a failure of the project but as a diagnostic result that identifies precisely which parts of the framework remain to be tested, and this paper is structured so that future results — confirming or overturning this finding — can be appended directly.

---

## 1. Introduction

### 1.1 Motivation

A transmission or distribution network is simultaneously an electrical circuit, a piece of geography, a graph with redundancy structure, and a system with a live operating point. Classical power-system engineering has excellent, separate tools for each of these views — $Y_{\mathrm{bus}}$ and AC/DC power flow for the electrical view, GIS overlays for the geographic view, N-1/N-k contingency analysis for the redundancy view, state estimation for the operating-point view — but no single mathematical object in standard use holds all four simultaneously, and none of the standard objects is *differentiable* with respect to a set of free parameters that could be tuned against outcome data the way the weights of a statistical model are tuned. Two broad directions exist in the literature to close this gap: (a) apply spectral graph theory to the network's own Laplacian/admittance structure, which stays faithful to circuit laws but has, until now, produced descriptive rather than *learnable* representations (see Section 2); or (b) apply generic graph neural networks, which are learnable but treat the network as an unstructured graph, discarding or re-learning from scratch the circuit-theoretic meaning that spectral methods already provide for free. Lynxo's premise is that neither extreme is necessary: it is possible to build an ambient space whose geometry is *fixed* by physically meaningful matrices (so every coordinate has a provable interpretation) while making the *parameters that generate those matrices* learnable, exactly as a neural network's architecture is fixed while its weights are not.

### 1.2 Philosophy: Lift, Learn, Project

The project follows one recurring three-step pattern, applied consistently across every channel of the representation:

1. **Lift.** Take a physical quantity attached to the network (impedance, geography, topology, operating-point injections) and turn it into a symmetric matrix with a provable algebraic structure — almost always a graph Laplacian, or a matrix that turns out to *be* a graph Laplacian on inspection (Section 3.5.3 derives one such case, the redundancy channel, from first principles).
2. **Learn.** Parameterize how that matrix is generated by a small number of free scalars (a mixing weight, a damping strength, an exponent, a block weight), always in a way that keeps the matrix well-posed — connected, correctly signed — for *every* value the parameter can take, so that unconstrained gradient-based search never needs to worry about leaving the space of valid networks.
3. **Project.** Take whatever is computed in the lifted space (a distance, a rank, a learned score) and attach it back to the original physical bus or branch, so every number a power-system engineer sees is traceable to something they can check.

This pattern is applied identically whether the "channel" is electrical, geographic, redundancy-based, or operating-point-based (Section 3.5), which is what makes the framework a single coherent object rather than four unrelated tricks bundled together.

### 1.3 What "Kadero" means in this project

"Kadero Degree" was the project's first, scalar-valued construction: a single number per bus or branch, generalizing weighted degree to penalize reliance on non-redundant (bridge-like) connections. "Kadero Embedding" is its vector-valued generalization: instead of one number per branch, each bus and branch is given a coordinate in a multi-channel Euclidean space, of which the original scalar criticality is recoverable as one derived quantity. Both objects are used in this paper; the embedding is the framework's final, general form, and the scalar criticality $\kappa_e$ (Section 3.3) remains the primary quantity actually tested empirically so far, since it is the cheapest, most interpretable slice of the full embedding to validate first.

### 1.4 Contributions

1. A rigorous mathematical construction (Section 3) of a five-channel, high-dimensional, differentiable vector space representing a power network, with every channel built from a provably well-posed symmetric operator, and with six categories of theorem proved about it: well-posedness, metric/recovery properties, differentiability, gauge invariance, hierarchical (multi-scale) consistency, and structural (bridge/redundancy) meaning.
2. Two properties of the framework confirmed not by internal self-consistency but by *independent* cross-validation: a conservation law and a bridge-detection theorem, both checked numerically against an unrelated algorithm with zero discrepancies (Section 4.5.3).
3. A complete, honestly reported empirical record across five phases of testing on four standard IEEE networks plus a 300-bus system, including a self-discovered and openly corrected methodological error (Section 4.6.1) and a rigorous, cross-validated comparison against classical machine-learning baselines with bootstrap uncertainty quantification (Section 4.6.2).
4. A frank assessment (Section 5) of what this evidence does and does not support, and a prioritized roadmap (Section 6) for the tests that would most efficiently resolve the framework's central open question: does the learnable structure add predictive value beyond what classical features already provide?

### 1.5 Organization

Section 2 places the work relative to existing literature. Section 3 is the complete mathematical foundation. Section 4 is the experimental log, organized chronologically by phase, reproducing every result obtained so far without omission or cherry-picking, including one result flagged as likely erroneous. Section 5 discusses what the evidence supports. Section 6 gives next steps. Appendices give a symbol glossary, the full free-parameter registry, and a template for appending future experiments.

---

## 2. Related Work

**Classical power-network representations.** The graph $G=(V,E)$, the bus admittance matrix $Y_{\mathrm{bus}}$, AC power flow, and its DC linearization $\mathbf P = B_{\mathrm{bus}}\boldsymbol\theta$ remain the working representations of power-system engineering. Effective resistance / electrical distance, $d_{ij}=(e_i-e_j)^\top L^+(e_i-e_j)$, is classical circuit theory (Doyle & Snell, 1984) and was proved to satisfy the triangle inequality — i.e. to be a genuine metric — by Klein & Randić (1993). None of these objects, on their own, incorporates geography or redundancy structure, and none is parametrically learnable.

**Spectral graph theory applied to power systems.** The closest prior art to this project's electrical channels is Retière, Ha, and Caputo's spectral analysis of power flows on the Laplacian eigenbasis (Retière, Ha, & Caputo, 2019/2020, *IEEE Systems Journal* 14(2), 2736–2747), which expresses nodal voltages and branch power flows as sums over Laplacian eigenmodes and — most relevant here — identifies a network's **bridge-block decomposition** via spectral structure as the key substructure governing line-failure localization. This is close enough to this project's bridge-detection result (Section 3.3, Theorem 3) that the overlap must be stated plainly: the *idea* that spectral/effective-resistance structure identifies bridges is not new to this project. What is new here (Section 3.5–3.7) is making the generating conductances a learnable, gradient-differentiable parametric family, fusing electrical structure with geography and redundancy in one product space, and proving exact multi-scale consistency under network reduction. A second closely related paper, Çetinay, Kuipers, & Van Mieghem (2018, *IEEE Systems Journal* 12(4), 2524–2532), similarly combines Kirchhoff's laws with the pseudoinverse of the weighted network Laplacian to express linearized power-flow behavior in a slack-bus-independent form — again a spectral/Laplacian-pseudoinverse construction, again without a learning mechanism.

**Kron reduction.** The exact-resistance-preserving elimination of internal nodes used in this project's hierarchical-consistency theorem (Section 3.7) is Dörfler & Bullo's Kron reduction (2013, *IEEE Trans. Circuits Syst. I*), a standard tool in power-system model reduction, recently generalized to the time domain (Singh, Dhople, Dörfler, & Giannakis, 2023).

**Spectral sparsification and random spanning trees.** The redundancy channel (Section 3.5.3) rests on two established results outside the power-systems literature: Spielman & Srivastava's near-linear-time algorithm for effective-resistance ("leverage score") computation (2011, *SIAM J. Comput.*), and Burton & Pemantle's transfer current theorem characterizing the joint law of a random spanning tree via the same Laplacian pseudoinverse (1993, *Ann. Probab.*).

**Graph neural networks for power systems.** A separate, much larger literature applies generic GNNs to power-system tasks. Nakiganda & Chatzivasileiadis (2023, arXiv:2310.04213) combine physics-informed and graph-aware architectures for N-1/N-k contingency screening, reporting up to 400× speed-up over Newton-Raphson on 6-, 24-, 57-, and 118-bus systems. Suri & Mangal's PowerGNN (2025, arXiv:2503.22721) reports 93.7% agreement with classical N-1 security assessment using a topology-aware GNN. A parallel, very recent trend proposes large **grid foundation models** trained across many topologies (e.g. PowerPM, NeurIPS 2024), with follow-on work on calibrated uncertainty for such models (Alcántara & Chatzivasileiadis, 2026, arXiv:2602.07995). This project's explicit position, stated in the original design brief and preserved here, is to avoid the generic-GNN path: the aim is a representation whose geometry is meaningful *before* any learning happens, with learning restricted to a small, physically interpretable parameter set layered on top of that fixed geometry — closer in spirit to the spectral papers above than to the GNN literature, but, unlike the spectral papers, differentiable and parametric by construction.

**Positioning.** Lynxo's contribution is therefore best understood as occupying a specific, previously not-quite-occupied point in this landscape: it takes the spectral/effective-resistance machinery of Retière et al. and Çetinay et al., which is descriptive, and the differentiability and calibration ambitions of the GNN literature, which is geometry-agnostic, and combines them into one differentiable, multi-channel, physically-grounded representation. Whether this combination earns its complexity in practice is exactly the empirical question this paper's experimental log (Section 4) is designed to answer, and the current honest answer, given the evidence collected so far, is: not yet demonstrated (Section 5).

---

## 3. Mathematical Foundation

*This section is the final, consolidated statement of the framework's mathematics, superseding earlier internal drafts. Every non-trivial claim is either proved or explicitly attributed to a cited external theorem.*

### 3.1 Notation

$G=(V,E)$ is the power network, connected, simple, undirected, $n=|V|$ buses, $m=|E|$ branches (lines and transformers, uniformly). For $e=(i,j)\in E$: complex impedance $z_e=r_e+\iota x_e$ ($r_e\ge0$), admittance $y_e=1/z_e=g_e+\iota b_e$,
$$
g_e=\frac{r_e}{r_e^2+x_e^2}\ge0,\qquad \beta_e:=-b_e=\frac{x_e}{r_e^2+x_e^2},
$$
geographic length $\ell_e$ between georeferenced bus coordinates, and optional loading ratio $\rho_e=|P_e|/C_e$ against thermal capacity $C_e$. Incidence vectors are $\chi_e:=e_i-e_j\in\mathbb R^n$.

**Assumption A (inductive lines).** $x_e>0$ for every branch — standard at transmission voltage, so $\beta_e>0$ throughout.

### 3.2 The scalar Kadero space

**Definition 1 (Kadero-weighted graph).** Given positive conductances $\tilde y(e)$ on $E$ (built from physical, geographic, and/or learned data — see Section 3.5.1), the Kadero-weighted graph is $\tilde G=(V,E,\tilde y)$ with weighted Laplacian $\tilde L=\tilde D-\tilde W$, positive semidefinite, rank $n-1$ for connected $G$, admitting a Moore–Penrose pseudoinverse $\tilde L^+$.

**Definition 2 (Kadero distance).** $d_{\mathcal K}(i,j):=(e_i-e_j)^\top\tilde L^+(e_i-e_j)$, the effective resistance of $\tilde G$.

**Theorem 1 (Metric property).** $d_{\mathcal K}$ satisfies the triangle inequality and is therefore a genuine metric on $V$.
*Proof.* Klein & Randić (1993), via the combinatorial (spanning-forest) representation of $\tilde L^+$; holds for any connected graph with strictly positive conductances. $\blacksquare$

### 3.3 Edge criticality and its bridge theorem

**Definition 3 (Edge criticality).** $\kappa_e:=\tilde y(e)\,\chi_e^\top\tilde L^+\chi_e \in(0,1]$.

**Theorem 2 (Bridge equivalence).** $\kappa_e=1$ **if and only if** $e$ is a topological bridge of $G$ (i.e. its removal disconnects the graph), independent of the specific positive weights used.
*Proof.* If $e$ is a bridge, the only path between its endpoints is $e$ itself, so the two-terminal resistance equals $1/\tilde y(e)$ exactly and $\kappa_e=1$. If $e$ is not a bridge, an alternate path of positive conductance exists; by Rayleigh's monotonicity law (Doyle & Snell, 1984, Ch. I.4), adding a parallel positive conductance strictly decreases the two-terminal resistance below $1/\tilde y(e)$, so $\kappa_e<1$ strictly. $\blacksquare$

**Theorem 3 (Conservation law).** $\displaystyle\sum_{e\in E}\kappa_e = n-1$, independent of the specific positive weights.
*Proof.* Writing $\tilde L=\sum_e \tilde y(e)\chi_e\chi_e^\top$, $\sum_e \kappa_e = \sum_e \tilde y(e)\chi_e^\top\tilde L^+\chi_e = \operatorname{tr}(\tilde L\tilde L^+) = \operatorname{rank}(\tilde L) = n-1$, since $\tilde L\tilde L^+$ is the orthogonal projector onto $\tilde L$'s row space. $\blacksquare$

Both theorems are load-bearing for everything that follows, and both have been checked not just by proof but by independent numerical cross-validation against a bridge-finding algorithm that has no knowledge of $\kappa$ — see Section 4.5.3, where this cross-validation is reported as an experimental result in its own right, with zero discrepancies found across 411 branches of a 300-bus network.

### 3.4 Kadero Degree (scalar summary quantities)

**Definition 4 (Node, edge, subgraph, global Kadero Degree).** With a criticality-aversion exponent $\eta\ge0$:
$$
\mathrm{kd}_{ij}:=\tilde y(e)(1-\kappa_e)^\eta,\qquad \mathrm{Kd}_i:=\sum_{j\sim i}\mathrm{kd}_{ij},\qquad \mathrm{Kd}_S:=\sum_{e\in\partial S}\mathrm{kd}_e\ (S\subsetneq V),
$$
with $\mathrm{Kd}_{\{i\}}=\mathrm{Kd}_i$ exactly for every $\eta$, and a global, tunable-risk-posture summary via the power mean $\mathrm{Kd}_p(G)=\big(\tfrac1n\sum_i \mathrm{Kd}_i^p\big)^{1/p}$, ordered in $p$ by the classical power-mean inequality.

*Interpretation.* $\mathrm{Kd}_i$ discounts a node's raw connective strength by how close its connections are to being bridges; by Theorem 2, a bridge's contribution to $\mathrm{Kd}_i$ is *exactly* zero for any $\eta>0$, not asymptotically — the single-point-of-failure penalty is an algebraic identity, not a heuristic. This scalar family was the framework's first, currently most heavily tested construction (Section 4).

### 3.5 The Kadero Embedding: five channels, one primitive

The scalar quantities above generalize to a vector-valued embedding by replacing "compute one number from $\tilde L^+$" with "compute a low-dimensional coordinate from the leading spectrum of a symmetric operator," applied to five separately justified operators.

#### 3.5.1 Electrical channels (real and reactive)

$$
\tilde y_\theta^{(g)}(e)=g_e\cdot g_{\alpha_g}(\ell_e)\cdot e^{\theta_g^\top h_e},\qquad
\tilde y_\theta^{(B)}(e)=\beta_e\cdot g_{\alpha_B}(\ell_e)\cdot e^{\theta_B^\top h_e},\qquad g_\alpha(\ell)=\frac{1}{1+\alpha\ell},
$$
generating $\tilde L_\theta^{(g)},\tilde L_\theta^{(B)}$ as in Definition 1, where $\theta_g,\theta_B$ are learnable log-linear residuals on operational features $h_e$ (loading ratio, asset age, etc. — *not* thermal capacity directly multiplying conductance, a correction from the project's original preliminary sketch: thermal rating and series impedance are independent physical properties of a line and should not be conflated).

**Proposition 1 (Unconditional well-posedness).** For every $(\alpha_g,\alpha_B,\theta_g,\theta_B)$ with $\alpha\ge0$, both $\tilde y_\theta^{(g)}(e)$ and $\tilde y_\theta^{(B)}(e)$ are strictly positive, so $\tilde L_\theta^{(g)},\tilde L_\theta^{(B)}$ are always valid connected Laplacians — the parameter feasible set is *all of $\mathbb R^{\dim\theta}$*, unconstrained.

**Proposition 2 (Exact recovery of $Y_{\mathrm{bus}}$).** Ignoring shunts, $Y_{\mathrm{bus}}=\sum_e(g_e+\iota b_e)\chi_e\chi_e^\top$; setting all learnable parameters to their null values recovers $Y_{\mathrm{bus}}=\tilde L_\theta^{(g)}-\iota\tilde L_\theta^{(B)}$ exactly, and, under the further lossless idealization $r_e=0$, $\tilde L_\theta^{(B)}$ is exactly the DC power-flow matrix $B_{\mathrm{bus}}$.

#### 3.5.2 Geographic channel

$$
\phi_G(i)=\sqrt{\tfrac2{d_G}}\big(\cos(\omega_1^\top\xi_i+b_1),\dots,\cos(\omega_{d_G}^\top\xi_i+b_{d_G})\big)^\top,
$$
a random Fourier feature lift (Rahimi & Recht, 2007) of geocoordinates $\xi_i$, with learnable frequencies $\{\omega_k\}$; by Bochner's theorem this is an unbiased finite-dimensional approximation of a shift-invariant geographic kernel.

#### 3.5.3 Redundancy channel: derived, not assumed

This channel was substantially revised mid-project after an initial version was judged mathematically valid but not well-motivated (Section 4 documents this revision explicitly as part of the project's development history, per the "report negative results" standard applied throughout).

Let $T$ be the random spanning tree of the (reference-weighted) network with $\Pr[T=t]\propto\prod_{e\in t}\tilde y^{\mathrm{ref}}(e)$, and $X=\mathbf 1_T\in\{0,1\}^E$.

**Theorem 4 (Transfer current theorem; Burton & Pemantle, 1993).** For distinct edges, $\Pr[e_1,\dots,e_k\in T]=\det(\mathcal Y(e_p,e_q))_{p,q=1}^k$, where $\mathcal Y(e,f)=\tilde y(e)\chi_e^\top\tilde L^+\chi_f$; in particular $\Pr[e\in T]=\kappa_e$.

**Definition 5 (Redundancy covariance matrix).** $K_\theta:=\mathrm{Cov}(X)$, with closed form $K_\theta(e,e)=\kappa_e(1-\kappa_e)$ and, for $e\ne f$, $K_\theta(e,f)=-\tilde y(e)\tilde y(f)\,(\chi_e^\top\tilde L^+\chi_f)^2$.

**Theorem 5 ($K_\theta$ is exactly a graph Laplacian).** $K_\theta\succeq0$ (any covariance matrix is PSD); $K_\theta\mathbf1=0$, since every spanning tree has exactly $n-1$ edges *almost surely*, so $\sum_f K_\theta(e,f)=\mathrm{Cov}(X_e,n-1)=0$; off-diagonal entries are $\le0$. These three properties are exactly the Laplacian axioms, so $K_\theta=\mathcal L(H_\theta)$ for a canonical derived graph $H_\theta$ on vertex set $E$ (edges of $G$ become vertices of $H_\theta$), generically connected.

**Theorem 6 (Bridges are isolated in $H_\theta$).** If $e$ is a bridge, $\kappa_e=1$ (Theorem 2) $\Rightarrow K_\theta(e,e)=0 \Rightarrow$ (Cauchy–Schwarz for PSD matrices) $K_\theta(e,f)=0\ \forall f$: a single point of failure is provably disconnected from the network's entire redundancy structure, not merely flagged by a scalar.

The redundancy channel $\Phi_\theta^R$ is the spectral factorization of $K_\theta$ (top eigenmodes, PCA convention — the correct convention for a covariance matrix, in deliberate contrast to the electrical channels' bottom-nonzero-eigenvalue convention, which is correct for a Green's-function/resistance-distance embedding).

#### 3.5.4 Operating-point channel (optional, requires a solved AC state)

Under a near-lossless assumption ($r_e\approx0$, additional to Assumption A, used *only* here), real power flow is $P_i=\sum_{j\sim i}V_iV_jB_{ij}\sin(\delta_i-\delta_j)$, $B_{ij}=1/x_{ij}$, at solved angles/voltages $(\delta^0,V^0)$.

**Definition 6.** $\tilde y_\theta^{\mathrm{op}}(e)=V_i^0V_j^0 B_{ij}\cos(\delta_i^0-\delta_j^0)$, $L_\theta^{\mathrm{op}}=\sum_e \tilde y_\theta^{\mathrm{op}}(e)\chi_e\chi_e^\top$.

**Theorem 7.** $L_\theta^{\mathrm{op}}=-H$, the standard AC power-flow angle-Jacobian block, exactly (verified by direct differentiation); this requires the near-lossless assumption specifically, since with $r_e\ne0$ the corresponding block acquires an antisymmetric term and is not, in general, symmetric.

**Theorem 8 (Positivity $\iff$ small-signal angle-stability regime).** $\tilde y_\theta^{\mathrm{op}}(e)>0\iff|\delta_i^0-\delta_j^0|<\pi/2$ — exactly the classical steady-state stability limit of the power-angle curve (Kundur, 1994). The smallest eigenvalue of $L_\theta^{\mathrm{op}}$ tending to zero coincides with the singularity of this Jacobian block at the maximum-loadability point (Ajjarapu & Christy, 1992). *Scope, stated precisely:* this covers steady-state angle/synchronizing-power margin only, not the Q–V block, and therefore not voltage collapse proper — no broader AC-stability claim is made.

#### 3.5.5 The ambient space

$$
\mathcal V_\theta=\mathbb R^{d_{E,g}}\oplus\mathbb R^{d_{E,B}}\oplus\mathbb R^{d_G}\oplus\mathbb R^{d_R}\,[\oplus\,\mathbb R^{d_{\mathrm{op}}}],\qquad
\Phi_\theta(i)=\big(\Phi_\theta^{E,g}(i),\Phi_\theta^{E,B}(i),\phi_G(i),\Phi_\theta^R(i),[\Phi_\theta^{\mathrm{op}}(i)]\big),
$$
a learnable block-metric direct sum. Edge embeddings concatenate midpoint and difference of endpoint embeddings; subgraph/hierarchy embeddings use Kron reduction (Section 3.7).

### 3.6 Learnability

**Theorem 9 (Differentiability).** On the open dense set of $\theta$ where the relevant operator has simple spectrum, every channel is real-analytic in $\theta$: composition of (i) real-analyticity of each generating operator in $\theta$; (ii) real-analyticity of the Moore–Penrose pseudoinverse on the constant-rank stratum (Golub & Pereyra, 1973); (iii) first-order eigenpair perturbation theory (Kato, 1966), which needs only symmetry, not sign-definiteness, hence covers the possibly-indefinite operating-point channel unchanged; and, for that channel specifically, (iv) the implicit function theorem applied to the AC power-flow equations, giving real-analyticity of the solved operating point in $\theta$ wherever the full Jacobian is nonsingular. At the measure-zero eigenvalue-crossing set, every *pairwise* (Gram-matrix) functional remains differentiable, since pseudoinverse differentiability needs only constant rank.

**Theorem 10 (Gauge invariance).** $\Phi_\theta$ is determined only up to a per-block orthogonal transformation (Schoenberg, 1935; Young & Householder, 1938); physically meaningful readouts must therefore be functions of the Gram structure alone — operationalized via distances to a small, fixed, non-learned reference bus set.

**Local identifiability.** Fixing the readout as above and defining the parameter-to-observable map $\Psi(\theta)$, $\theta$ is locally identifiable wherever the Jacobian $D\Psi$ has full column rank (inverse function theorem), a criterion computable from the same gradients as Theorem 9.

### 3.7 Hierarchical consistency

**Theorem 11.** For the three genuine positive-weighted Laplacian channels ($\tilde L_\theta^{(g)},\tilde L_\theta^{(B)},K_\theta$), Kron reduction of any node subset preserves all pairwise resistance distances among retained nodes exactly (Dörfler & Bullo, 2013); since the untruncated embedding realizes resistance distance exactly (Theorem 1's corollary), the coarse-grained and fine-grained embeddings of the same retained set differ only by a global isometry (Schoenberg; Young & Householder). *This guarantee is not extended to the operating-point channel*, which can be indefinite — stated as an open item, not silently assumed.

### 3.8 Free parameter registry

| Symbol | Role | Well-posedness mechanism |
|---|---|---|
| $\alpha_g,\alpha_B\ge0$ | geographic damping, per electrical channel | $\alpha=e^{\gamma_\alpha}$ |
| $\theta_g,\theta_B$ | log-linear conductance residual | unconstrained, $\exp(\cdot)>0$ always |
| $\mu\in[0,1]$ | redundancy reference-channel mixing | sigmoid reparametrization |
| $\{\omega_k,b_k\}$ | geographic frequencies/phases | unconstrained |
| $s_g,s_B,s_R,s_{\mathrm{op}}$ | per-channel spectral exponent | unconstrained |
| $\gamma_{E,g},\dots,\gamma_{\mathrm{op}}$ | block metric weights (log-scale) | $\beta=e^\gamma>0$ |
| $d_{E,g},\dots,d_{\mathrm{op}}$ | truncation ranks | architectural, chosen via Eckart–Young error bound |

Every numerical value in this table is left free by design; determining them from data is the project's ongoing empirical phase (Section 4), not part of the mathematical foundation.

### 3.9 What is recovered exactly

| Classical object | Recovered as |
|---|---|
| $Y_{\mathrm{bus}}$, $B_{\mathrm{bus}}$ | Proposition 2 |
| Effective resistance | untruncated, $s=1$ embedding distance |
| Bridge / single point of failure | $\kappa_e=1$ (Theorem 2); isolated vertex of $H_\theta$ (Theorem 6) |
| Spanning-tree inclusion probability | $\kappa_e$ exactly (Theorem 4) |
| Synchronizing power coefficient | $\tilde y_\theta^{\mathrm{op}}(e)$ (Theorem 7) |
| Kron-reduced network theory | Theorem 11 |
| Classical Laplacian eigenmaps (Belkin & Niyogi, 2003) | $s=0$ special case |

---

## 4. Experimental Program

*This section reproduces every experiment run to date, in chronological order, without omission. Numbers are copied from the original generated reports (Phases 0–2, 5) or from the sandboxed execution logs of this project's collaborative sessions (Phases 3–4).*

### 4.1 Timeline overview

| Phase | Date | Network | Question | Headline result |
|---|---|---|---|---|
| 0 | 2026-08-15 | IEEE 14 | Does the physics-only ($\theta=0$) pipeline run correctly? | Yes; top-ranked edges are a transformer bridge and a set of very low-impedance lines |
| 1 | 2026-08-15 | IEEE 14, 30, 118 | Does $\kappa$ agree with classical edge betweenness? | Weak-to-negative, not statistically significant on 2 of 3 networks |
| 2 | 2026-08-15 | IEEE 30 | Does untuned $\kappa$ predict N-1 contingency severity? | $\rho=-0.32$, significant, wrong-signed |
| 3 | 2026-08-16 | case300 | Can 2 free parameters be learned to improve $\rho$? | $\rho$: 0.678 → 0.683 (whole population); non-bridge subset appeared to worsen |
| 4 | 2026-08-16 | case300 | Was the objective/metric in Phase 3 measured correctly, and does $\kappa$ beat a classical baseline? | Phase 3's non-bridge finding was a measurement artifact (corrected below); no statistically clear advantage over a classical Ridge baseline |
| 5 | 2026-08-16 | case300 | Is the Phase 3/4 result sensitive to the severity definition? | Inconclusive — result appears to contain a data-generation bug (see 4.7) |

### 4.2 Phase 0 — IEEE 14 basic verification

**Goal.** First mathematical verification of the core operators on a standard test case before any learning.
**Method.** Weighted graph from lines and transformers; combinatorial Laplacian and pseudoinverse; $\kappa_e=\tilde y_e R_{\mathrm{eff}}(e)$; $\theta=0$ throughout (no geography, no learning).
**Result.** 14 buses, 20 edges. The single highest-$\kappa$ edge (trafo between buses 6–7, $\kappa=1.000$) is an exact bridge; the next several ($\kappa=0.93$–$0.83$) are low-impedance lines with correspondingly large raw conductance and small effective resistance.
**Assessment.** The pipeline computes sensible, interpretable output on the smallest standard test case; this phase makes no predictive claim, only a correctness/sanity claim.

### 4.3 Phase 1 — Comparison against classical edge betweenness

**Goal.** Test whether $\kappa$ reduces to, or diverges from, classical topological centrality, across increasing network size.

| Network | Buses | Edges | Spearman $\rho$ ($\kappa$ vs. betweenness) | $p$-value |
|---|---|---|---|---|
| IEEE 14 | 14 | 20 | 0.239 | 0.309 (n.s.) |
| IEEE 30 | 30 | 41 | 0.058 | 0.718 (n.s.) |
| IEEE 118 | 118 | 179 | −0.129 | 0.086 (n.s. at 0.05) |

**Assessment.** Across all three networks, $\kappa$ and classical edge betweenness are statistically indistinguishable from uncorrelated, and the point estimate trends negative as network size grows. This was, and remains, genuine positive evidence that $\kappa$ is *not* a relabeling of a classical centrality measure — but non-redundancy with betweenness is evidence of difference, not of usefulness, a distinction maintained throughout this paper (see Section 5).

### 4.4 Phase 2 — IEEE 30 N-1 contingency baseline

**Goal.** Evaluate the untuned ($\theta=0$) $\kappa$ against actual N-1 contingency severity, not a topological proxy.
**Method.** For every line/transformer: remove it, run power flow, severity $=\max(0,\text{max loading}-100)+10\times(\text{\# overloaded lines})$, large penalty on non-convergence.
**Result.** 41 contingencies; **Spearman $\rho=-0.320$, $p=0.041$** — significant, and wrong-signed: higher $\kappa$ associated with *lower* observed severity. The single most severe real contingency (bus 5–7, severity 73.4) ranks only 8th by $\kappa$; conversely, a $\kappa=1$ bridge (buses 11–12) ranks 6th by severity, not 1st.
**Assessment.** This is the result that motivated every subsequent phase of this project. It shows the untuned electrical-only score is not automatically a good severity predictor on this network and this severity definition, and set up learning (Phase 3) as the natural next step: can parameters be found that turn this negative correlation into a positive, useful one?

### 4.5 Phase 3 — Precision Learning, Stage 1 (case300)

**Goal.** Learn two free parameters — $\mu$ (electrical channel mixing) and $p$ (conductance exponent) — sequentially, by direct search, to maximize Spearman correlation between $\kappa$ and N-1 severity on a larger, more realistic network (case300, 300 buses, 411 branches).

**Method.** $\tilde y_\theta(e)=(\mu g_e+(1-\mu)\beta_e)^p$; severity model explicitly handling non-convergence, topological islanding (a whole-network failure mode a naive severity formula would otherwise silently drop — see 4.5.1), and thermal overload; sequential Hooke–Jeeves pattern search, $\pm0.1$ steps with step-halving on stall, random restarts, $\le1000$ evaluations per parameter, early stopping.

#### 4.5.1 A modeling correction found during severity-label generation

Removing a bridge line can split the network into two components; the AC solver converges for the slack-containing island and *silently drops the disconnected buses from the results* unless explicitly checked. Left unhandled, this would have caused every true islanding event — arguably the single most severe class of contingency — to be scored near zero. The severity model was corrected to explicitly detect unsupplied buses (via topological connectivity checking) and assign them a large, bus-count-scaled penalty. This is reported here because it materially changed the character of the severity labels (on case300's default, lightly-loaded condition, severity turned out to be almost entirely islanding-driven rather than overload-driven — a property of this test case, not of the method).

#### 4.5.2 Results

| Score | Whole-population Spearman $\rho$ |
|---|---|
| Baseline ($\mu=0.5,p=1.0$) | 0.678 |
| Learned ($\mu^*=0.274,\,p^*=2.768$) | 0.683 |

An aspirational target of $\rho\ge0.90$ (agreed in advance as ambitious) was not reached.

#### 4.5.3 Independent validation of the mathematics (a result in its own right)

Before trusting any correlation number, the implementation was checked against Theorems 2 and 3, using a tool (`networkx`'s combinatorial bridge finder) with no knowledge of $\kappa$: conservation law held to 6 decimal places ($\sum_e\kappa_e=298.99999996\dots\approx n-1=299$); bridge equivalence matched **89/89** true bridges exactly, zero mismatches. This is reported as a genuine finding, not a footnote: it is independent evidence that the implementation is a correct realization of Theorems 2–3, not merely a plausible-looking score.

#### 4.5.4 The originally reported non-bridge finding (superseded — see Phase 4)

Stage 1's original report additionally computed correlation restricted to "non-bridge" edges and found the *learned* parameters scored *worse* (0.172) than the untuned baseline (0.205) on that subset, flagged at the time as a real concern about whether learning was targeting the right thing. **This specific finding did not survive a later correction and should not be treated as established** — see Section 4.6.1.

### 4.6 Phase 4 — Corrected objective and classical baseline (Stage 2A + 2B, case300)

**Goal.** Directly follow up on Phase 3's two loudest diagnostic signals: (a) re-measure non-bridge performance correctly, and (b) test whether a simple classical machine-learning model, given no Kadero-specific machinery, already matches $\kappa$ on the harder, non-bridge majority of the network.

#### 4.6.1 Experiment A: the objective correction, and a self-found bug

Building a fixed, $\kappa$-independent, ground-truth bridge/non-bridge split (again via `networkx.bridges()`) for a fair before/after comparison revealed that **Phase 3's own non-bridge comparison had used an inconsistent definition of "non-bridge."** Phase 3 identified non-bridge edges via a $\kappa$-value threshold ($\kappa<0.999$) computed *separately* for baseline and learned $\kappa$; because the learned exponent $p^*=2.77$ sharpens near-bridge edges toward $\kappa=1$ (a real, correctly-understood consequence of raising conductances to a power $>1$), this threshold selected 322 edges for the baseline comparison but only 306 for the learned comparison — different-sized, non-matching subsets on the two sides of a "before vs. after" comparison.

Under the corrected, fixed 322-edge non-bridge set:

| Score | Non-bridge Spearman $\rho$ (corrected) | (originally reported, Phase 3) |
|---|---|---|
| Baseline | 0.205 | 0.205 |
| Learned | **0.215** | ~~0.172~~ |

**The corrected conclusion reverses the direction of Phase 3's headline non-bridge finding**: learning did not make non-bridge ranking worse; if anything the point estimate moved slightly in the favorable direction, though (see 4.6.2) not by a statistically distinguishable margin. This correction is reported prominently, not as an appendix note, because a reader relying on the Phase 3 report alone would currently hold a specifically wrong belief about the direction of this effect.

#### 4.6.2 Experiment B: classical baseline comparison

**Method.** Thirteen classical, non-Kadero features (edge betweenness; effective resistance computed from the reactive channel alone, i.e. the standard DC-style electrical distance; raw and log-compressed conductances; unweighted and weighted endpoint degree features), with and without a binary `is_bridge` feature, fit against the same severity labels with Ridge regression (on rank-transformed severity) and, separately, Random Forest and Gradient Boosting regressors, all under honest 5-fold cross-validation (out-of-fold predictions only), plus 2000-resample bootstrap 90% confidence intervals on the non-bridge subset given how close several point estimates are.

| Method | Non-bridge Spearman $\rho$ | 90% bootstrap CI |
|---|---|---|
| Learned $\kappa$ | 0.215 | [0.122, 0.302] |
| Ridge (+ is_bridge) | 0.211 | [0.144, 0.272] |
| Baseline $\kappa$ | 0.205 | [0.120, 0.285] |
| Ridge (no is_bridge) | 0.149 | [0.055, 0.235] |
| Gradient Boosting (no is_bridge) | 0.080 | [0.001, 0.159] |
| Random Forest (no is_bridge) | −0.017 | [−0.126, 0.086] |

**Assessment.** The top three estimates overlap heavily; none is statistically distinguishable from the others at this sample size. Tree ensembles consistently underperformed linear regression, most plausibly a small-sample effect (322 non-bridge points split five ways for cross-validation), confirmed across an independent hyperparameter sweep. **The honest headline finding of Phase 4 is a null result**: neither $\kappa$ (tuned or untuned) nor the best classical model tested has demonstrated a statistically clear advantage over the other on the harder, non-bridge majority of this network, under this severity definition.

### 4.7 Phase 5 — Severity design robustness check (case300) — flagged as likely erroneous

**Goal.** Test whether the Phase 4 result is an artifact of one specific severity definition, by comparing four variants (island-heavy, thermal-heavy, excluding large islands, balanced).

**Result as generated:**

| Severity variant | $\rho_{\text{all}}$ | $\rho_{\text{non-bridge}}$ |
|---|---|---|
| S1 (island-heavy) | 0.1127 | 0.2061 |
| S2 (thermal-heavy) | 0.1127 | 0.2061 |
| S3 (excl. large islands) | 0.1127 | 0.2061 |
| S4 (balanced) | 0.1127 | 0.2061 |

**This result should not be interpreted as evidence of robustness and is flagged here explicitly as a probable implementation bug, not a finding.** Four deliberately different severity formulas producing *bit-identical* correlation values to four decimal places is not a plausible outcome of genuinely different weightings; it is far more consistent with the four variant columns having been generated by code that does not actually branch on the variant (e.g. a shared computation being reused, or a dataframe write that silently overwrites all four columns with the same values). Separately, $\rho_{\text{all}}=0.113$ here does not match Phase 3/4's whole-population baseline of $\rho\approx0.68$ on what is described as "the original style" severity, which is a second, independent inconsistency pointing the same direction. **Action required before this experiment's conclusion can be trusted:** re-inspect the variant-generation code for the four severity columns, confirm they differ numerically at the per-contingency level (not just check the correlation), and re-run. Until then, no claim about severity-definition robustness should be drawn from this phase, in either direction.

---

## 5. Discussion

**What is established.** The mathematical foundation (Section 3) is internally consistent and, for its two most operationally important claims — conservation and bridge equivalence — independently verified against ground truth, not merely self-consistent. The framework correctly and provably identifies exact single points of failure, and does so as an algebraic identity rather than a tuned threshold. It is also confirmed, across three independent networks (Phase 1) and one additional network (Phase 4's betweenness comparison, implicit in its feature set), to be measuring something genuinely distinct from classical topological centrality.

**What is not yet established.** Distinctiveness is not the same claim as usefulness, and this is the crux of the project's current status. On every network and severity definition tested so far where a fair, cross-validated comparison was actually carried out (Phase 4), the learnable Kadero score has not demonstrated statistically distinguishable predictive value beyond a simple classical model, once the easy, exactly-solvable 22% of the problem (bridge detection, which any topological algorithm gets for free) is set aside. The untuned baseline was, moreover, actively *misleading* on IEEE 30 (Phase 2: significant, wrong-signed correlation), and the one two-parameter learning experiment run so far (Phase 3) produced only a marginal, and initially misreported, improvement.

**On the self-corrections.** Two things went wrong during this project's execution and both are documented above rather than silently fixed: an islanding-handling bug in severity generation (4.5.1) and a mismatched-subset comparison bug in the Phase 3 non-bridge finding (4.6.1), the latter of which reversed a previously reported conclusion. A third likely bug (Phase 5, Section 4.7) is flagged as unresolved. This pattern — genuine mistakes, caught and reported rather than hidden — should be read as evidence of the project's methodological discipline, not as a mark against it; a reader should trust the corrected numbers in this document more, not less, for that reason.

**Relation to prior art.** The bridge-detection result overlaps meaningfully with Retière, Ha, and Caputo's (2019/2020) spectral bridge-block decomposition; this project's differentiability and learning machinery is the specific contribution beyond that prior work, and the empirical program above is, honestly, a test of whether that additional machinery earns its complexity — a test it has not yet clearly passed.

---

## 6. Roadmap

In priority order, based on information gain per unit effort, not on the order the framework's channels happen to have been built in:

1. **Resolve Phase 5 before drawing any severity-robustness conclusion.** Cheapest possible next action; currently blocks any claim about whether the Phase 4 null result is specific to one severity formula.
2. **Test the redundancy channel ($K_\theta$, Section 3.5.3) against the non-bridge subset specifically**, with a pre-registered bar to clear: it must beat the current ceiling ($\rho\approx0.21$, Phase 4) with a bootstrap confidence interval that does not substantially overlap the classical Ridge baseline's. This is the first channel that is mathematically designed to capture something effective resistance alone provably cannot (Theorem 6's redundancy structure), and is the natural next test of whether the framework's added machinery earns its complexity.
3. **Obtain a genuinely georeferenced network** (case300's coordinates are synthetic layout coordinates, confirmed by direct inspection, not real geography) before attempting to test or learn the geographic channel — testing it on the current data would be uninformative by construction.
4. **Test the operating-point channel** on a solved AC state, restricted explicitly to its proven scope (steady-state angle margin, Theorem 8), against a severity signal that is not itself islanding-dominated, so the test is not confounded with the bridge-detection result that already explains most of the "easy" signal.
5. Only after at least one channel clears its pre-registered bar: begin the LLM explanation layer, designed from the start to explain *why* a score is what it is in terms of the underlying provable objects (bridge status, redundancy-graph position, Kron-reduced region) rather than generating fluent narrative around an unvalidated number — a foreseeable risk if sequenced the other way round.

---

## 7. Conclusion

Lynxo's mathematical foundation is complete, rigorous, and — for its most important structural claims — independently verified rather than merely asserted. Its empirical validation is honest, self-correcting, and, as of this writing, inconclusive on the central question of practical value: the framework has not yet been shown to outperform a simple classical baseline on the part of the problem that actually matters. This is a fork, not a verdict. The roadmap above is designed to resolve it as cheaply and as clearly as possible, and this document is structured so that whichever way the evidence moves, it can be appended here rather than requiring a rewrite.

---

## Appendix A — Symbol Glossary

| Symbol | Meaning |
|---|---|
| $G=(V,E)$ | power network graph, $n=\lvert V\rvert$ buses, $m=\lvert E\rvert$ branches |
| $z_e,y_e,g_e,\beta_e$ | branch impedance, admittance, real/conductance and reactive/susceptance parts |
| $\tilde y(e),\tilde L$ | Kadero-weighted conductance and Laplacian |
| $\kappa_e$ | edge criticality, $\in(0,1]$; $=1$ iff bridge |
| $\mathrm{Kd}_i,\mathrm{Kd}_S,\mathrm{Kd}_p(G)$ | node, subgraph, global Kadero Degree |
| $\Phi_\theta,\mathcal V_\theta$ | Kadero embedding map and ambient space |
| $K_\theta,H_\theta$ | redundancy covariance matrix and its derived conflict graph |
| $L_\theta^{\mathrm{op}}$ | operating-point (synchronizing-power) Laplacian |
| $\theta,\mu,p,\alpha,\eta,s,\gamma$ | learnable parameters — see Section 3.8 registry |

## Appendix B — Free Parameter Registry

*Reproduced from Section 3.8; kept here also for quick reference alongside the experimental log, since every phase in Section 4 reports which of these were held fixed and which (if any) were searched over.*

See Section 3.8 table. As of Phase 4 (the most recent learning experiment), only $\mu$ and $p$ have ever been searched over; every other row remains at its default/null value untested.

## Appendix C — Template for Appending Future Experiments

To add a new experiment, copy this block as a new subsection under Section 4 (renumbering as needed), fill it in exactly as the phases above were written, and update the Section 4.1 timeline table and Section 5/6 discussion and roadmap to reflect it.

```
### 4.N Phase N — <short title> (<network>)

**Goal.** <one or two sentences>

**Method.** <what was computed, what changed vs. the previous phase>

**Result.** <numbers, as a table where possible>

**Assessment.** <does this support, contradict, or fail to resolve the current
open question from Section 5/6? Be as willing to report a null or negative
result here as a positive one — every phase in this document so far has.>
```

---

## References

- Ajjarapu, V., & Christy, C. (1992). The continuation power flow: a tool for steady state voltage stability analysis. *IEEE Trans. Power Syst.*, 7(1), 416–423.
- Alcántara, A., & Chatzivasileiadis, S. (2026). Trustworthiness layer for foundation models in power systems: application to N-k contingency screening. arXiv:2602.07995.
- Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation. *Neural Computation*, 15(6), 1373–1396.
- Burton, R., & Pemantle, R. (1993). Local characteristics, entropy and limit theorems for spanning trees and domino tilings via transfer-impedances. *Ann. Probab.*, 21(3), 1329–1371.
- Çetinay, H., Kuipers, F. A., & Van Mieghem, P. (2018). A topological investigation of power flow. *IEEE Systems Journal*, 12(4), 2524–2532.
- Coifman, R. R., & Lafon, S. (2006). Diffusion maps. *Applied and Computational Harmonic Analysis*, 21(1), 5–30.
- Dörfler, F., & Bullo, F. (2013). Kron reduction of graphs with applications to electrical networks. *IEEE Trans. Circuits Syst. I*, 60(1), 150–163.
- Doyle, P. G., & Snell, J. L. (1984). *Random Walks and Electric Networks*. Mathematical Association of America.
- Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.
- Golub, G. H., & Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. *SIAM J. Numer. Anal.*, 10(2), 413–432.
- Kato, T. (1966). *Perturbation Theory for Linear Operators*. Springer.
- Klein, D. J., & Randić, M. (1993). Resistance distance. *J. Math. Chem.*, 12, 81–95.
- Kundur, P. (1994). *Power System Stability and Control*. McGraw-Hill.
- Nakiganda, A. M., & Chatzivasileiadis, S. (2023). Graph neural networks for fast contingency analysis of power systems. arXiv:2310.04213.
- Newman, M. E. J. (2005). A measure of betweenness centrality based on random walks. *Social Networks*, 27(1), 39–54.
- Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.
- Retière, N., Ha, D. T., & Caputo, J.-G. (2019/2020). Spectral graph analysis of the geometry of power flows in transmission networks. *IEEE Systems Journal*, 14(2), 2736–2747.
- Schoenberg, I. J. (1935). Remarks to Maurice Fréchet's article. *Ann. of Math.*, 36, 724–732.
- Singh, M. K., Dhople, S., Dörfler, F., & Giannakis, G. B. (2023). Time-domain generalization of Kron reduction. *IEEE Control Systems Letters*, 7, 259–264.
- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances. *SIAM J. Comput.*, 40(6), 1913–1926.
- Suri, D., & Mangal, M. (2025). PowerGNN: a topology-aware graph neural network for electricity grids. arXiv:2503.22721.
- Young, G., & Householder, A. S. (1938). Discussion of a set of points in terms of their mutual distances. *Psychometrika*, 3, 19–22.
