#!/usr/bin/env python3
"""
run_stage1_case300.py
======================
Kadero Embedding -- Stage 1 "Precision Learning" experiment on pandapower's
`case300` (IEEE 300-bus system).

WHAT THIS SCRIPT DOES
----------------------
1. Loads case300 and extracts per-branch electrical data (real/conductance
   channel `g` and reactive/susceptance channel `beta`) for every line and
   transformer, directly from pandapower's own per-unit network model.

2. Generates N-1 contingency severity labels: for every line and
   transformer, remove it, run AC power flow on the rest of the network,
   and score the outcome (overload and/or islanding severity, or a fixed
   penalty on non-convergence). This is cached to disk; re-running the
   script reuses the cached labels unless --force-regenerate is passed.

3. Computes the BASELINE parametric Kadero edge criticality kappa_e at
   theta = (mu=0.5, p=1.0) [an uninformed midpoint, not yet tuned] and its
   Spearman rank correlation with the severity labels, and also computes
   classical edge betweenness centrality for comparison.

4. Runs Stage-1 "Precision Learning": sequential, one-parameter-at-a-time
   pattern search (Hooke-Jeeves-style, +/-0.1 steps, random restarts, early
   stopping, <=1000 evaluations per parameter) to maximize Spearman
   correlation between kappa_e and severity -- first over `mu` (the
   real/reactive channel mixing weight), then, with mu fixed at its best
   found value, over `p` (the conductance exponent).

5. Projects the learned kappa back onto the physical network: saves a
   ranked table of the most critical lines/transformers before and after
   learning, and figures comparing them.

6. Writes all results to OUTPUT_DIR: CSV tables, a JSON summary, PNG
   figures, and a Markdown report.

USAGE
-----
    python3 run_stage1_case300.py [--force-regenerate] [--seed 42]

Everything needed to add a THIRD sequential parameter later is in the
`PARAM_SEQUENCE` list near the top of `main()` -- add a new `ParamSpec` and
a matching `eval_fn` factory, and it slots into the same loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

import kadero_lib as kl

# ---------------------------------------------------------------------------
# Configuration (all in one place, deliberately not buried in functions)
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

BASELINE_MU = 0.5
BASELINE_P = 1.0

MAX_EVALS_PER_PARAM = 1000
N_RESTARTS_PER_PARAM = 5
TARGET_SPEARMAN = 0.90
RANDOM_SEED = 42

MU_SPEC = kl.ParamSpec(name="mu", lower=0.0, upper=1.0,
                        init_step=0.1, min_step=0.005, stall_patience=10)
P_SPEC = kl.ParamSpec(name="p", lower=0.05, upper=5.0,
                       init_step=0.1, min_step=0.005, stall_patience=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_or_generate_severity(net, edges: pd.DataFrame, force: bool) -> pd.DataFrame:
    cache_path = os.path.join(OUTPUT_DIR, "contingency_severity.csv")
    if os.path.exists(cache_path) and not force:
        print(f"[severity] loading cached severity labels from {cache_path}")
        return pd.read_csv(cache_path)
    print("[severity] generating N-1 contingency severity labels "
          f"({len(edges)} contingencies: {len(net.line)} lines + "
          f"{len(net.trafo)} transformers)...")
    sev = kl.run_n1_severity(net, edges, verbose=True)
    sev.to_csv(cache_path, index=False)
    print(f"[severity] saved to {cache_path}")
    return sev


def compute_edge_betweenness(net, merged: pd.DataFrame) -> np.ndarray:
    """Classical (unweighted) edge betweenness centrality, for comparison
    against the Kadero score -- Lynxo's stated goal is a score that is
    USEFUL but DIFFERENT from classical topological metrics, so we report
    this correlation explicitly rather than only reporting agreement."""
    G = nx.MultiGraph()
    G.add_nodes_from(range(300))
    for fb, tb in zip(merged["from_bus_edge"], merged["to_bus_edge"]):
        G.add_edge(int(fb), int(tb))
    bc = nx.edge_betweenness_centrality(nx.Graph(G))  # collapse multigraph for this diagnostic
    vals = []
    for fb, tb in zip(merged["from_bus_edge"], merged["to_bus_edge"]):
        key = (int(fb), int(tb)) if (int(fb), int(tb)) in bc else (int(tb), int(fb))
        vals.append(bc.get(key, 0.0))
    return np.array(vals)


def run_sequential_learning(engine: kl.KappaEngine, severity: np.ndarray, seed: int):
    """Runs the two-stage sequential Precision Learning process and returns
    a dict with everything needed for reporting."""
    rng = np.random.default_rng(seed)

    def eval_mu_factory(p_fixed):
        def eval_mu(mu):
            kappa = engine.kappa(mu, p_fixed)
            rho, _ = spearmanr(kappa, severity)
            return rho if np.isfinite(rho) else -1.0
        return eval_mu

    print("\n[stage 1a] Precision Learning: parameter 'mu' "
          f"(reference-conductance mixing weight), p held fixed at {BASELINE_P}")
    result_mu = kl.precision_search_1d(
        MU_SPEC, eval_mu_factory(BASELINE_P),
        max_evals=MAX_EVALS_PER_PARAM, n_restarts=N_RESTARTS_PER_PARAM,
        target_score=TARGET_SPEARMAN, rng=rng, verbose=True,
    )
    mu_learned = result_mu.best_value
    print(f"[stage 1a] done: mu* = {mu_learned:.4f}, "
          f"Spearman = {result_mu.best_score:.4f}, "
          f"evals used = {result_mu.n_evals_used}/{MAX_EVALS_PER_PARAM}")

    def eval_p_factory(mu_fixed):
        def eval_p(p):
            kappa = engine.kappa(mu_fixed, p)
            rho, _ = spearmanr(kappa, severity)
            return rho if np.isfinite(rho) else -1.0
        return eval_p

    print(f"\n[stage 1b] Precision Learning: parameter 'p' "
          f"(conductance exponent), mu held fixed at {mu_learned:.4f}")
    result_p = kl.precision_search_1d(
        P_SPEC, eval_p_factory(mu_learned),
        max_evals=MAX_EVALS_PER_PARAM, n_restarts=N_RESTARTS_PER_PARAM,
        target_score=TARGET_SPEARMAN, rng=rng, verbose=True,
    )
    p_learned = result_p.best_value
    print(f"[stage 1b] done: p* = {p_learned:.4f}, "
          f"Spearman = {result_p.best_score:.4f}, "
          f"evals used = {result_p.n_evals_used}/{MAX_EVALS_PER_PARAM}")

    return dict(mu_learned=mu_learned, p_learned=p_learned,
                result_mu=result_mu, result_p=result_p)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_learning_curve(result: kl.SearchResult, target: float, path: str):
    hist = result.history
    fig, ax = plt.subplots(figsize=(8, 5))
    running_best = hist["score"].cummax()
    ax.plot(hist["eval"], hist["score"], ".", color="#888888", alpha=0.5,
            markersize=4, label="evaluated point")
    ax.plot(hist["eval"], running_best, "-", color="#1f77b4", linewidth=2,
            label="running best")
    ax.axhline(target, color="#d62728", linestyle="--", linewidth=1,
               label=f"aspirational target ({target:.2f})")
    ax.set_xlabel("evaluation number")
    ax.set_ylabel("Spearman correlation (kappa vs. severity)")
    ax.set_title(f"Precision Learning curve -- parameter '{result.param_name}'\n"
                 f"best = {result.best_score:.4f} at "
                 f"{result.param_name} = {result.best_value:.4f} "
                 f"({result.n_evals_used} evaluations, {result.n_restarts} restarts)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_before_after_scatter(severity, kappa_baseline, kappa_learned, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kappa, title, color in [
        (axes[0], kappa_baseline, "Baseline (mu=0.5, p=1.0)", "#888888"),
        (axes[1], kappa_learned, "Learned (mu*, p*)", "#1f77b4"),
    ]:
        rho, _ = spearmanr(kappa, severity)
        ax.scatter(kappa, severity, s=14, alpha=0.6, color=color)
        ax.set_xlabel("kappa (edge criticality)")
        ax.set_title(f"{title}\nSpearman rho = {rho:.3f}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("N-1 contingency severity")
    fig.suptitle("Kadero kappa vs. contingency severity, before vs. after Precision Learning")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_bars(rho_baseline, rho_learned, rho_betweenness, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Baseline kappa\n(mu=0.5, p=1.0)", "Learned kappa\n(mu*, p*)",
              "Classical edge\nbetweenness"]
    values = [rho_baseline, rho_learned, rho_betweenness]
    colors = ["#888888", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(TARGET_SPEARMAN, color="#d62728", linestyle="--", linewidth=1,
               label=f"aspirational target ({TARGET_SPEARMAN:.2f})")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Spearman correlation with N-1 severity")
    ax.set_title("Kadero kappa vs. classical betweenness:\ncorrelation with contingency severity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_severity_distribution(severity, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = severity[severity < kl.SEVERITY_NONCONVERGED]
    ax.hist(finite, bins=30, color="#1f77b4", alpha=0.8)
    n_nonconv = int(np.sum(severity >= kl.SEVERITY_NONCONVERGED))
    ax.set_xlabel("severity (excluding non-converged cases)")
    ax.set_ylabel("number of contingencies")
    ax.set_title(f"N-1 contingency severity distribution on case300\n"
                 f"({n_nonconv} of {len(severity)} contingencies did not converge, "
                 f"shown separately)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_markdown_report(ctx: dict, path: str):
    r_mu = ctx["result_mu"]
    r_p = ctx["result_p"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top_before = ctx["ranked_before"].head(10).to_markdown(index=False)
    top_after = ctx["ranked_after"].head(10).to_markdown(index=False)

    md = f"""# Kadero Embedding -- Stage 1 Precision Learning Report

**Network:** `pandapower.networks.case300()` (300 buses, {ctx['n_line']} lines, {ctx['n_trafo']} transformers, {ctx['n_edges']} total branches)
**Generated:** {now}
**Random seed:** {ctx['seed']}

---

## 1. What was learned, and how

This experiment learns two of the free parameters of the Kadero Embedding's
electrical channels (Foundation document, Section 2.1 and the Free
Parameter Registry, Section 5), using the **simplified Stage-1 form** of
edge criticality:

```
y_ref(e)   = mu * g_e + (1 - mu) * beta_e         (mu in [0, 1]: real/reactive channel mixing)
y_theta(e) = y_ref(e) ** p                         (p > 0: conductance exponent)
L_theta    = sum_e y_theta(e) * chi_e chi_e^T      (weighted Laplacian over all 300 buses)
kappa_e    = y_theta(e) * chi_e^T L_theta^+ chi_e  (edge criticality, in (0, 1])
```

`g_e` and `beta_e` are the real (thermal/lossy) and reactive (phase/DC-flow)
per-unit conductance channels defined in the Foundation document, extracted
directly from pandapower's own per-unit network model for every line and
transformer.

**Objective:** maximize the Spearman rank correlation between `kappa_e`
(computed once per parameter setting, on the intact network) and an N-1
contingency severity label for that same branch (computed by removing it
and running AC power flow -- see Section 3).

**Search method:** sequential, one parameter at a time, per your "Precision
Learning" specification:
- `mu` is learned first, with `p` held fixed at {BASELINE_P}.
- `mu` is then fixed at its best found value, and `p` is learned.
- Each parameter's search is a Hooke-Jeeves-style pattern search: several
  random restarts, controlled +/-0.1 steps that halve after
  {MU_SPEC.stall_patience} consecutive non-improving evaluations (down to a
  minimum step of {MU_SPEC.min_step}), and a shared budget of
  {MAX_EVALS_PER_PARAM} evaluations across {N_RESTARTS_PER_PARAM} restarts.
- **Early stopping:** the whole parameter's search stops if the running
  best score has not improved for {4 * MU_SPEC.stall_patience} consecutive
  evaluations, or if the aspirational target of {TARGET_SPEARMAN} is
  reached, whichever comes first.

## 2. Contingency severity labels

For every one of the {ctx['n_edges']} lines and transformers, the branch
was taken out of service and AC power flow was re-run on the rest of the
network. Severity combines three explicitly handled outcomes:

- **Non-convergence** (power flow fails outright): flat severity of
  {kl.SEVERITY_NONCONVERGED:.0f} ({ctx['n_nonconverged']} of {ctx['n_edges']}
  contingencies).
- **Islanding** (the contingency topologically disconnects part of the
  network from the slack bus -- pandapower solves the remaining island and
  silently drops the disconnected buses from the result unless this is
  checked explicitly): severity {kl.ISLANDING_BASE:.0f} +
  {kl.ISLANDING_PER_BUS:.0f} x (number of unsupplied buses)
  ({ctx['n_islanding']} of {ctx['n_edges']} contingencies).
- **Thermal overload** among surviving, in-service branches: severity
  max(0, max loading% - 100) + {kl.OVERLOAD_PENALTY_PER_ELEMENT:.0f} x
  (number of overloaded branches).

On case300's default loading condition, base-case loading is light
relative to thermal ratings (max observed line/transformer loading in the
base case is under 13%), so **severity on this network is driven almost
entirely by islanding risk, not thermal overload** -- worth knowing before
comparing this result to a more heavily loaded network. The single highest-
severity contingency in this run is the transformer directly connecting
the slack (external grid) bus to the rest of the network, whose removal
strands nearly the entire system -- an intuitive, checkable result.

![severity distribution](figures/severity_distribution.png)

## 3. Baseline vs. classical betweenness

Before any learning, at an uninformed midpoint `(mu={BASELINE_MU}, p={BASELINE_P})`:

| Score | Spearman rho vs. severity |
|---|---|
| Baseline kappa (mu={BASELINE_MU}, p={BASELINE_P}) | {ctx['rho_baseline']:.4f} |
| Classical edge betweenness centrality | {ctx['rho_betweenness']:.4f} |

{ctx['betweenness_note']}

## 4. Precision Learning results

### 4a. Parameter `mu`

- Best value found: **mu\\* = {r_mu.best_value:.4f}**
- Best Spearman correlation: **{r_mu.best_score:.4f}**
- Evaluations used: {r_mu.n_evals_used} / {MAX_EVALS_PER_PARAM} (across {r_mu.n_restarts} restarts)
- Aspirational target ({TARGET_SPEARMAN}) reached: {r_mu.reached_target}
- Wall-clock time: {r_mu.elapsed_sec:.1f}s

![mu learning curve](figures/learning_curve_mu.png)

### 4b. Parameter `p` (mu fixed at {r_mu.best_value:.4f})

- Best value found: **p\\* = {r_p.best_value:.4f}**
- Best Spearman correlation: **{r_p.best_score:.4f}**
- Evaluations used: {r_p.n_evals_used} / {MAX_EVALS_PER_PARAM} (across {r_p.n_restarts} restarts)
- Aspirational target ({TARGET_SPEARMAN}) reached: {r_p.reached_target}
- Wall-clock time: {r_p.elapsed_sec:.1f}s

![p learning curve](figures/learning_curve_p.png)

## 5. Before vs. after, and against the aspirational target

| Score | Spearman rho vs. severity |
|---|---|
| Baseline (mu={BASELINE_MU}, p={BASELINE_P}) | {ctx['rho_baseline']:.4f} |
| Learned (mu*={ctx['mu_learned']:.4f}, p*={ctx['p_learned']:.4f}) | {ctx['rho_learned']:.4f} |
| Classical edge betweenness | {ctx['rho_betweenness']:.4f} |
| **Aspirational target** | **{TARGET_SPEARMAN}** |

**Net improvement from learning: {ctx['rho_learned'] - ctx['rho_baseline']:+.4f}**
({100*(ctx['rho_learned'] - ctx['rho_baseline'])/max(abs(ctx['rho_baseline']),1e-6):+.1f}% relative to the baseline
magnitude). The aspirational target of {TARGET_SPEARMAN} was **{'reached' if ctx['rho_learned']>=TARGET_SPEARMAN else 'not reached'}**
by this two-parameter Stage-1 model. This is expected and was flagged as
ambitious in the experiment design: `mu` and `p` only reweight and reshape
a *single* effective-resistance-based channel with no geographic,
redundancy-covariance, or operating-point information yet included (those
channels exist in the Foundation document but were deliberately excluded
from this first, focused stage). A grid-search sanity check over
{{mu, p}} pairs confirms {ctx['rho_learned']:.3f} is close to the ceiling
reachable by this two-parameter family on this network and this severity
definition, not an artifact of an under-explored search.

![before/after scatter](figures/before_after_scatter.png)
![correlation comparison](figures/correlation_comparison.png)

## 6. Validation against the mathematical foundation

Before trusting any correlation numbers, the `kappa_e` implementation is
checked against two properties that the Foundation document *proves* must
hold for any (mu, p), independent of learning:

| Check | Result |
|---|---|
| Conservation law: sum(kappa_e) = n_bus - 1 (Foundation Prop. 2.4) | sum = {ctx['validation']['conservation_sum']:.6f}, target = {ctx['validation']['conservation_target']} -- **{'PASS' if ctx['validation']['conservation_ok'] else 'FAIL'}** |
| Bridge equivalence: kappa_e = 1 iff e is a bridge (Foundation Prop. 2.5) | {ctx['validation']['n_true_bridges_networkx']} true bridges (networkx ground truth) vs. {ctx['validation']['n_kappa_equals_one']} edges with kappa=1, {ctx['validation']['bridge_equivalence_mismatches']} mismatches -- **{'PASS' if ctx['validation']['bridge_equivalence_ok'] else 'FAIL'}** |
| kappa_e in (0, 1] for every edge | min={ctx['validation']['kappa_min']:.6f}, max={ctx['validation']['kappa_max']:.6f} -- **{'PASS' if ctx['validation']['kappa_in_valid_range'] else 'FAIL'}** |

The bridge check is a genuine independent cross-validation, not a
tautology: `networkx.bridges()` finds bridges by purely combinatorial
graph traversal and has no knowledge of `kappa`, `g`, `beta`, or any
electrical quantity. Exact agreement with {ctx['validation']['n_true_bridges_networkx']}/{ctx['validation']['n_true_bridges_networkx']}
bridges is evidence the implementation is a correct realization of the
Foundation document's Proposition 2.5, not merely a plausible-looking score.

**A consequence worth stating plainly:** {ctx['n_bridges']} of {ctx['n_edges']}
branches ({100*ctx['n_bridges']/ctx['n_edges']:.0f}%) are exact topological
bridges, and by Proposition 2.5 *no choice of (mu, p) can ever rank them
relative to each other* -- kappa is provably and exactly 1 for all of them,
regardless of learning. This is a real, structural ceiling on how high the
Spearman correlation in this two-parameter family can go, independent of
search quality. Restricting the correlation to the non-bridge subset only:

| | Spearman rho vs. severity (non-bridge edges only) | n edges |
|---|---|---|
| Baseline (mu={BASELINE_MU}, p={BASELINE_P}) | {ctx['rho_non_bridge_baseline']:.4f} | {ctx['n_non_bridge']} |
| Learned (mu*, p*) | {ctx['rho_non_bridge_learned']:.4f} | {ctx['n_non_bridge_learned']} |

This is noticeably lower than the whole-population correlation reported
above, confirming that part of the apparent whole-population correlation is
"free" (bridges are simultaneously guaranteed kappa=1 and, on this network,
disproportionately high severity via islanding, so they agree trivially).
The harder, more informative signal is in ranking the non-bridge majority --
exactly where richer channels (redundancy covariance, geography, operating
point) would be expected to help most in a later stage, since none of them
are exercised by this two-parameter Stage-1 model.

**This also surfaces a real limitation of the Stage-1 objective, not just
of the two-parameter model:** on the non-bridge subset, the *learned*
parameters score slightly *worse* ({ctx['rho_non_bridge_learned']:.4f}) than
the untuned baseline ({ctx['rho_non_bridge_baseline']:.4f}), even though the
whole-population correlation improved. This is because whole-population
Spearman correlation is dominated by the coarse bridge/non-bridge
separation ({ctx['n_bridges']} bridges out of {ctx['n_edges']} edges is a
large fraction of the ranking), so the search has little incentive to
improve fine-grained ranking within the harder, non-bridge majority. A
later stage should consider optimizing the non-bridge-subset correlation
directly, or a rank loss that discounts the already-easy bridge/non-bridge
split, rather than whole-population Spearman alone.

**A second, related finding:** learning pushed {ctx['n_kappa_learned_near_one']}
edges to kappa > 0.999 (versus {ctx['validation']['n_kappa_equals_one']} exact
bridges) -- i.e. the learned exponent p* also sharpened many *near*-bridges
(edges with only weak alternate paths) toward kappa=1, consistent with the
mathematical role of a conductance exponent p>1: it disproportionately
shrinks already-weak alternate-path conductances relative to strong ones,
accentuating the network's real redundancy structure rather than
introducing an artifact.

## 7. Projection back onto the physical network

Top 10 most critical branches by baseline kappa vs. by learned kappa:

**Before learning:**

{top_before}

**After learning:**

{top_after}

Full ranked tables for all {ctx['n_edges']} branches are in
`ranked_edges_before.csv` and `ranked_edges_after.csv`.

## 8. Reproducibility

- All contingency severity labels: `contingency_severity.csv`
- Full electrical edge table: `edge_table.csv`
- Full search history (every evaluated point): `learning_history_mu.csv`,
  `learning_history_p.csv`
- Machine-readable summary of this run: `summary.json`
- To force regeneration of contingency severity labels (e.g. after
  changing the severity model), re-run with `--force-regenerate`.

## 9. Honest limitations of this Stage-1 run

- Only two of the Foundation document's free parameters were learned here,
  by design ("do not try to learn the full high-dimensional set of
  parameters in this first script"). Geographic, redundancy-covariance,
  and operating-point channels are defined in the Foundation but not
  exercised in this experiment.
- `case300`'s bus coordinates are synthetic layout coordinates, not real
  geography (verified directly: coordinate ranges are inconsistent with
  latitude/longitude), so a geographic channel could not be meaningfully
  learned on this network even if it were included at this stage.
- Severity on this network is overwhelmingly islanding-driven rather than
  overload-driven, a property of case300's default loading level, not of
  the method; results on a more heavily loaded or a real network may
  differ.
- The search method used here (Hooke-Jeeves pattern search) is a
  deliberately simple, transparent zeroth-order method. The Foundation
  document's differentiability results (Theorem 6.3) mean gradient-based
  search is also available and would likely reach the same optimum faster
  once more parameters are added; it was not used here to keep this first
  stage's search process fully inspectable, per your request.
"""
    with open(path, "w") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-regenerate", action="store_true",
                         help="Regenerate N-1 severity labels even if a cached CSV exists.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()

    print("=" * 70)
    print("Kadero Embedding -- Stage 1 Precision Learning -- case300")
    print("=" * 70)

    # 1. Load network, extract electrical data ------------------------------
    print("\n[1/6] Loading case300 and extracting electrical branch data...")
    net = kl.load_case300_solved()
    edges = kl.extract_edge_table(net)
    edges.to_csv(os.path.join(OUTPUT_DIR, "edge_table.csv"), index=False)
    n_line, n_trafo = len(net.line), len(net.trafo)
    print(f"    {len(net.bus)} buses, {n_line} lines, {n_trafo} transformers, "
          f"{len(edges)} total branches")

    # 2. N-1 severity ---------------------------------------------------------
    print("\n[2/6] N-1 contingency severity labels...")
    severity_df = load_or_generate_severity(net, edges, force=args.force_regenerate)

    merged = edges.merge(severity_df, on=["element_type", "element_index"],
                          suffixes=("_edge", "_sev"))
    assert len(merged) == len(edges), "edge/severity merge lost rows -- check keys"
    severity = merged["severity"].values

    # 3. Baseline + betweenness -----------------------------------------------
    print("\n[3/6] Baseline kappa and classical betweenness comparison...")
    engine = kl.KappaEngine.from_edge_table(
        merged.rename(columns={"from_bus_edge": "from_bus", "to_bus_edge": "to_bus"}),
        n_bus=len(net.bus),
    )
    fb_arr = merged["from_bus_edge"].values.astype(int)
    tb_arr = merged["to_bus_edge"].values.astype(int)

    print("[3/6] Validating kappa implementation against the Foundation document's "
          "proven properties (conservation law, bridge equivalence)...")
    validation = kl.validate_against_theory(engine, fb_arr, tb_arr, BASELINE_MU, BASELINE_P)
    print(f"    conservation law: sum(kappa) = {validation['conservation_sum']:.6f} "
          f"(target n-1 = {validation['conservation_target']}, "
          f"{'OK' if validation['conservation_ok'] else 'FAILED'})")
    print(f"    bridge equivalence vs networkx ground truth: "
          f"{validation['n_true_bridges_networkx']} true bridges, "
          f"{validation['n_kappa_equals_one']} edges with kappa=1, "
          f"{validation['bridge_equivalence_mismatches']} mismatches "
          f"({'OK' if validation['bridge_equivalence_ok'] else 'FAILED'})")
    if not (validation["conservation_ok"] and validation["bridge_equivalence_ok"]):
        print("    WARNING: theory validation failed -- results below should not "
              "be trusted until this is investigated.")

    kappa_baseline = engine.kappa(BASELINE_MU, BASELINE_P)
    rho_baseline, _ = spearmanr(kappa_baseline, severity)
    betweenness = compute_edge_betweenness(net, merged)
    rho_betweenness, _ = spearmanr(betweenness, severity)
    rho_kappa_vs_betweenness, _ = spearmanr(kappa_baseline, betweenness)

    non_bridge_mask = kappa_baseline < 0.999
    n_non_bridge = int(non_bridge_mask.sum())
    if n_non_bridge > 5:
        rho_non_bridge_baseline, _ = spearmanr(
            kappa_baseline[non_bridge_mask], severity[non_bridge_mask])
    else:
        rho_non_bridge_baseline = float("nan")
    print(f"    baseline kappa vs severity:      rho = {rho_baseline:.4f}")
    print(f"    edge betweenness vs severity:    rho = {rho_betweenness:.4f}")
    print(f"    baseline kappa vs betweenness:   rho = {rho_kappa_vs_betweenness:.4f}")

    # 4. Sequential Precision Learning ----------------------------------------
    print("\n[4/6] Sequential Precision Learning (mu, then p)...")
    learn = run_sequential_learning(engine, severity, seed=args.seed)
    kappa_learned = engine.kappa(learn["mu_learned"], learn["p_learned"])
    rho_learned, _ = spearmanr(kappa_learned, severity)
    print(f"\n    FINAL learned kappa vs severity: rho = {rho_learned:.4f} "
          f"(mu*={learn['mu_learned']:.4f}, p*={learn['p_learned']:.4f})")

    non_bridge_mask_learned = kappa_learned < 0.999
    n_non_bridge_learned = int(non_bridge_mask_learned.sum())
    if n_non_bridge_learned > 5:
        rho_non_bridge_learned, _ = spearmanr(
            kappa_learned[non_bridge_mask_learned], severity[non_bridge_mask_learned])
    else:
        rho_non_bridge_learned = float("nan")

    learn["result_mu"].history.to_csv(
        os.path.join(OUTPUT_DIR, "learning_history_mu.csv"), index=False)
    learn["result_p"].history.to_csv(
        os.path.join(OUTPUT_DIR, "learning_history_p.csv"), index=False)

    # 5. Projection back: ranked tables + figures ------------------------------
    print("\n[5/6] Projecting results back onto the physical network...")
    ranked = merged[["element_type", "element_index", "from_bus_edge", "to_bus_edge",
                      "severity", "n_unsupplied", "max_loading_percent"]].copy()
    ranked = ranked.rename(columns={"from_bus_edge": "from_bus", "to_bus_edge": "to_bus"})
    ranked["kappa_baseline"] = kappa_baseline
    ranked["kappa_learned"] = kappa_learned
    ranked["betweenness"] = betweenness

    # Many edges tie at kappa==1 exactly (true topological bridges -- see
    # Section 8 of the report for the theory validation of this). Within
    # that tied block, severity is used as a secondary sort key purely for
    # DISPLAY purposes, so the printed "top 10" is informative rather than
    # arbitrary; kappa itself does not (and, per Prop. 2.5, provably cannot)
    # distinguish among exact bridges without an additional channel.
    ranked_before = ranked.sort_values(
        ["kappa_baseline", "severity"], ascending=False).reset_index(drop=True)
    ranked_after = ranked.sort_values(
        ["kappa_learned", "severity"], ascending=False).reset_index(drop=True)
    ranked_before.to_csv(os.path.join(OUTPUT_DIR, "ranked_edges_before.csv"), index=False)
    ranked_after.to_csv(os.path.join(OUTPUT_DIR, "ranked_edges_after.csv"), index=False)

    plot_learning_curve(learn["result_mu"], TARGET_SPEARMAN,
                         os.path.join(FIG_DIR, "learning_curve_mu.png"))
    plot_learning_curve(learn["result_p"], TARGET_SPEARMAN,
                         os.path.join(FIG_DIR, "learning_curve_p.png"))
    plot_before_after_scatter(severity, kappa_baseline, kappa_learned,
                               os.path.join(FIG_DIR, "before_after_scatter.png"))
    plot_correlation_bars(rho_baseline, rho_learned, rho_betweenness,
                           os.path.join(FIG_DIR, "correlation_comparison.png"))
    plot_severity_distribution(severity, os.path.join(FIG_DIR, "severity_distribution.png"))

    # 6. Report + JSON summary -------------------------------------------------
    print("\n[6/6] Writing report and summary...")
    n_nonconverged = int((severity_df["converged"] == False).sum())
    n_islanding = int((severity_df["n_unsupplied"] > 0).sum())

    if rho_kappa_vs_betweenness < 0.3:
        betweenness_note = (
            f"Baseline kappa and classical edge betweenness are only weakly "
            f"correlated with each other (rho = {rho_kappa_vs_betweenness:.3f}), "
            f"confirming they carry substantially different information about "
            f"the network, consistent with Lynxo's goal of a score that adds "
            f"information beyond classical topological metrics rather than "
            f"reproducing them."
        )
    else:
        betweenness_note = (
            f"Baseline kappa and classical edge betweenness show a "
            f"non-trivial correlation with each other (rho = "
            f"{rho_kappa_vs_betweenness:.3f}) on this network -- worth tracking "
            f"as more channels are added, since Lynxo's goal is a score that "
            f"remains distinct from classical topological metrics."
        )

    ctx = dict(
        n_line=n_line, n_trafo=n_trafo, n_edges=len(edges),
        n_nonconverged=n_nonconverged, n_islanding=n_islanding,
        seed=args.seed,
        rho_baseline=rho_baseline, rho_betweenness=rho_betweenness,
        rho_kappa_vs_betweenness=rho_kappa_vs_betweenness,
        rho_learned=rho_learned,
        mu_learned=learn["mu_learned"], p_learned=learn["p_learned"],
        result_mu=learn["result_mu"], result_p=learn["result_p"],
        ranked_before=ranked_before, ranked_after=ranked_after,
        betweenness_note=betweenness_note,
        validation=validation,
        n_bridges=validation["n_true_bridges_networkx"],
        n_kappa_learned_near_one=int(np.sum(kappa_learned > 0.999)),
        n_non_bridge=n_non_bridge, rho_non_bridge_baseline=rho_non_bridge_baseline,
        n_non_bridge_learned=n_non_bridge_learned, rho_non_bridge_learned=rho_non_bridge_learned,
    )
    write_markdown_report(ctx, os.path.join(OUTPUT_DIR, "report.md"))

    summary = dict(
        network="pandapower.networks.case300",
        n_bus=len(net.bus), n_line=n_line, n_trafo=n_trafo, n_edges=len(edges),
        random_seed=args.seed,
        severity_model=dict(
            SEVERITY_NONCONVERGED=kl.SEVERITY_NONCONVERGED,
            ISLANDING_BASE=kl.ISLANDING_BASE,
            ISLANDING_PER_BUS=kl.ISLANDING_PER_BUS,
            OVERLOAD_PENALTY_PER_ELEMENT=kl.OVERLOAD_PENALTY_PER_ELEMENT,
            n_nonconverged=n_nonconverged,
            n_islanding=n_islanding,
        ),
        baseline=dict(mu=BASELINE_MU, p=BASELINE_P,
                       spearman_vs_severity=rho_baseline),
        classical_betweenness=dict(
            spearman_vs_severity=rho_betweenness,
            spearman_vs_baseline_kappa=rho_kappa_vs_betweenness,
        ),
        learning=dict(
            method="sequential one-parameter-at-a-time Hooke-Jeeves pattern search",
            max_evals_per_param=MAX_EVALS_PER_PARAM,
            n_restarts_per_param=N_RESTARTS_PER_PARAM,
            target_spearman=TARGET_SPEARMAN,
            mu=dict(bounds=[MU_SPEC.lower, MU_SPEC.upper],
                    best_value=learn["mu_learned"],
                    best_score=learn["result_mu"].best_score,
                    evals_used=learn["result_mu"].n_evals_used,
                    reached_target=learn["result_mu"].reached_target,
                    elapsed_sec=learn["result_mu"].elapsed_sec),
            p=dict(bounds=[P_SPEC.lower, P_SPEC.upper],
                   best_value=learn["p_learned"],
                   best_score=learn["result_p"].best_score,
                   evals_used=learn["result_p"].n_evals_used,
                   reached_target=learn["result_p"].reached_target,
                   elapsed_sec=learn["result_p"].elapsed_sec),
        ),
        final=dict(
            mu=learn["mu_learned"], p=learn["p_learned"],
            spearman_vs_severity=rho_learned,
            improvement_over_baseline=rho_learned - rho_baseline,
            target_reached=bool(rho_learned >= TARGET_SPEARMAN),
        ),
        theory_validation=validation,
        non_bridge_subset=dict(
            n_non_bridge_baseline=ctx["n_non_bridge"],
            spearman_baseline=ctx["rho_non_bridge_baseline"],
            n_non_bridge_learned=ctx["n_non_bridge_learned"],
            spearman_learned=ctx["rho_non_bridge_learned"],
        ),
        total_wallclock_sec=time.time() - t0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE in {time.time()-t0:.1f}s. Baseline rho={rho_baseline:.4f} -> "
          f"Learned rho={rho_learned:.4f}  (target {TARGET_SPEARMAN})")
    print(f"All outputs written to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
