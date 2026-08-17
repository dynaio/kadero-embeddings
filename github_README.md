# Kadero Embedding — Lynxo Project

**An original mathematical framework that lifts a power network into a learnable, high-dimensional geometric space, in order to detect operational criticality — hidden points of failure, redundancy structure, and more — that classical topological metrics miss.**

Independent research project. Archived, open, and looking for the electrical-engineering or applied-mathematics expert who wants to take it further.

---

## What is this?

Classical power-system tools — the graph, the admittance matrix $Y_{\mathrm{bus}}$, DC power flow, the graph Laplacian — are each physically grounded but each capture only one facet of what makes a piece of grid infrastructure critical, and none of them is *learnable*. The **Kadero Embedding** proposes a single, differentiable, multi-channel vector space that jointly encodes electrical strength, geography, topological redundancy, and (optionally) the live AC operating point — with every channel built from a symmetric matrix whose meaning is *proven*, not assumed, and every free parameter designed to be calibrated from data the way a model's weights are trained.

Highlights of what's proven in the full write-up:

- An edge criticality score $\kappa_e \in (0,1]$ that equals **exactly 1 if and only if** the branch is a true single point of failure — an algebraic identity, cross-validated against an independent, purely combinatorial bridge-finding algorithm with **zero discrepancies** across a 300-bus test network.
- A conservation law: the network's total "criticality mass" always sums to exactly $n-1$, regardless of how the underlying weights are chosen.
- A redundancy channel derived from the network's own random spanning-tree covariance structure, which turns out — provably — to itself be a graph Laplacian, on which true bridges appear as **isolated vertices**.
- Full differentiability of every channel with respect to its generating parameters (via classical matrix perturbation theory), meaning the entire representation is gradient-learnable, not just descriptive.

## Honest status

This is not a polished success story, and it isn't presented as one. The mathematics is complete and independently verified where verification was possible. The empirical testing — across IEEE 14-, 30-, 118-bus systems and a 300-bus system — is a genuine, self-correcting research log, including two mistakes found and openly corrected mid-project. **The current honest result is a null one**: the learnable score has not yet shown a statistically distinguishable advantage over a simple classical baseline, once true bridges (which any algorithm finds for free) are set aside — though it has repeatedly, across every network tested, shown close to zero correlation with classical edge betweenness centrality, meaning it captures something genuinely different, even if not yet shown to be better.

Full detail, all numbers, and a prioritized roadmap for what to test next are in the paper.

**[Read the full paper →](Lynxo_Kadero_Project_Report_v1.md)**

## Repository structure

```
Lynxo_Kadero_Project_Report_v1.md   the paper: philosophy, full math with
                                     proofs, complete experimental log
code/
  kadero_lib.py                     core library (electrical extraction,
                                     kappa, N-1 severity, theory validation)
  kadero_stage2_lib.py              classical-baseline comparison + bootstrap CI
  run_stage1_case300.py             executable: Precision Learning experiment
  run_stage2_case300.py             executable: corrected objective + baseline test
results/
  stage1/, stage2/                  full generated output: CSVs, JSON,
                                     figures, reports — real runs, not mockups
  pre_case300_reports/              original IEEE 14/30/118 verification reports
```

## Running it

```bash
pip install pandapower networkx scipy pandas matplotlib scikit-learn tabulate
cd code
python3 run_stage1_case300.py
python3 run_stage2_case300.py
```

Both scripts are fully self-contained and reproduce every number in the
paper from scratch (or from the cached labels included in `results/`).

## For anyone considering picking this up

Section 6 of the paper is a prioritized roadmap with a pre-registered bar
for what would count as a real result. Section 4.7 flags one experiment as
containing a likely bug — documented rather than hidden, and worth fixing
before trusting its conclusion in either direction. If you use, extend,
correct, or publish any part of this, no permission is needed — see
[LICENSE](LICENSE). A citation is welcome if convenient (see
[CITATION.cff](CITATION.cff)), not required.

## Author

Berassil Abdelkader — Final-year Computer Science Engineering student,
Artificial Intelligence specialization, University of Mustapha Stambouli,
Mascara, Algeria. This project sits outside my primary field of training,
by design and stated plainly in the paper itself — I built as far as I
responsibly could and am archiving it here in case it's useful to someone
with the domain depth to take it further.

Portfolio: https://berassil-abdelkader.web.app
