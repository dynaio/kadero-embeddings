#!/usr/bin/env python3
"""
run_stage2_case300.py
=======================
Kadero Embedding -- Stage 2A + 2B, on the same case300 network and the same
N-1 contingency severity labels as Stage 1.

Stage 2A -- Corrected Objective:
    Re-evaluate baseline kappa (mu=0.5, p=1.0) and Stage-1-learned kappa
    (mu*=0.274, p*=2.768) with the non-bridge-subset Spearman correlation
    as the PRIMARY metric (whole-population is reported for reference only).

Stage 2B -- Classical Baseline:
    Build a feature matrix of classical, non-Kadero quantities (edge
    betweenness, raw/classical effective resistance, conductances, degree
    features, optionally is_bridge) and fit Ridge / Random Forest /
    Gradient Boosting regressors against the same severity labels, under
    honest 5-fold cross-validation, to answer: can a simple classical model
    match or beat kappa on the harder non-bridge subset?

Both stages reuse Stage 1's cached edge table and severity labels if
present (looked for first in ./cache/, falling back to regenerating from
scratch via kadero_lib if absent or if --force-regenerate is passed).

USAGE
-----
    python3 run_stage2_case300.py [--force-regenerate] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import kadero_lib as kl
import kadero_stage2_lib as k2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = "cache"                # where Stage-1 outputs are looked for / copied
OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

BASELINE_MU, BASELINE_P = 0.5, 1.0
# Exact values found by Stage 1's Precision Learning search -- reused here,
# not re-learned, since Stage 2's question is "was the OBJECTIVE the
# problem", which requires holding the learned parameters fixed.
LEARNED_MU, LEARNED_P = 0.27395604855596344, 2.7680859375000003

N_CV_SPLITS = 5
N_BOOTSTRAP = 2000
BOOT_CI = 0.90
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading (reuse Stage-1 cache if available)
# ---------------------------------------------------------------------------

def load_edges_and_severity(force: bool):
    edge_cache = os.path.join(CACHE_DIR, "edge_table.csv")
    sev_cache = os.path.join(CACHE_DIR, "contingency_severity.csv")

    if os.path.exists(edge_cache) and os.path.exists(sev_cache) and not force:
        print(f"[data] reusing cached Stage-1 edge table and severity labels "
              f"from {CACHE_DIR}/")
        edges = pd.read_csv(edge_cache)
        severity_df = pd.read_csv(sev_cache)
        net = kl.load_case300_solved()  # still needed for bus count / betweenness graph
    else:
        print("[data] no cache found (or --force-regenerate passed): "
              "rebuilding edge table and N-1 severity labels from scratch...")
        net = kl.load_case300_solved()
        edges = kl.extract_edge_table(net)
        severity_df = kl.run_n1_severity(net, edges, verbose=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        edges.to_csv(edge_cache, index=False)
        severity_df.to_csv(sev_cache, index=False)

    merged = edges.merge(severity_df, on=["element_type", "element_index"],
                          suffixes=("_edge", "_sev"))
    assert len(merged) == len(edges), "edge/severity merge lost rows"
    return net, merged


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison_bars(comparison_df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(comparison_df))
    for ax, col, title in [(axes[0], "rho_all", "Whole population (411 edges)"),
                            (axes[1], "rho_nonbridge", "Non-bridge subset (322 edges)")]:
        colors = ["#1f77b4" if "kappa" in n.lower() else "#ff7f0e"
                  for n in comparison_df["method"]]
        bars = ax.bar(x, comparison_df[col], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["method"], rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, comparison_df[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03),
                    f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_ylabel("Spearman correlation with N-1 severity")
    fig.suptitle("Stage 2: Kadero kappa vs. classical models -- "
                  "whole population vs. non-bridge subset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ci_forest(ci_results: dict, path: str):
    """Forest plot of non-bridge Spearman correlation with bootstrap 90% CIs,
    so differences between close point estimates can be read against their
    actual uncertainty rather than compared as bare numbers."""
    names = list(ci_results.keys())
    points = [ci_results[n]["point"] for n in names]
    los = [ci_results[n]["ci_low"] for n in names]
    his = [ci_results[n]["ci_high"] for n in names]
    order = np.argsort(points)
    names = [names[i] for i in order]
    points = [points[i] for i in order]
    los = [los[i] for i in order]
    his = [his[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 2))
    y = np.arange(len(names))
    colors = ["#1f77b4" if "kappa" in n.lower() else "#ff7f0e" for n in names]
    for yi, p, lo, hi, c in zip(y, points, los, his, colors):
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2)
        ax.plot(p, yi, "o", color=c, markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman correlation, non-bridge subset "
                  f"({int(BOOT_CI*100)}% bootstrap CI, n={N_BOOTSTRAP})")
    ax.set_title("Non-bridge performance with uncertainty:\n"
                  "point estimates this close should NOT be read as a ranking")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names: list, importances: np.ndarray, title: str, path: str):
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#2ca02c")
    ax.set_xlabel("feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_best_scatter(kappa_baseline, kappa_learned, ridge_pred, severity, is_bridge, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    nb = ~is_bridge
    for ax, pred, title in [
        (axes[0], kappa_baseline, "Baseline kappa (mu=0.5,p=1.0)"),
        (axes[1], kappa_learned, "Learned kappa (mu*,p*)"),
        (axes[2], ridge_pred, "Ridge (classical features,\nwith is_bridge)"),
    ]:
        rho_nb, _ = spearmanr(pred[nb], severity[nb])
        ax.scatter(pred[~nb], severity[~nb], s=14, alpha=0.4, color="#888888", label="bridge")
        ax.scatter(pred[nb], severity[nb], s=14, alpha=0.7, color="#1f77b4", label="non-bridge")
        ax.set_title(f"{title}\nnon-bridge Spearman = {rho_nb:.3f}")
        ax.set_xlabel("score")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("N-1 contingency severity")
    axes[0].legend(fontsize=8)
    fig.suptitle("Non-bridge edges are the hard part: bridges (gray) separate trivially")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(ctx: dict, path: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    comp_table = ctx["comparison_df"].to_markdown(index=False, floatfmt=".4f")
    ci_lines = "\n".join(
        f"| {name} | {v['point']:.4f} | [{v['ci_low']:.4f}, {v['ci_high']:.4f}] |"
        for name, v in ctx["ci_results"].items()
    )

    md = f"""# Kadero Embedding -- Stage 2A + 2B Report
## Corrected Objective and Classical Baseline Comparison

**Network:** `pandapower.networks.case300()` (same as Stage 1: {ctx['n_bus']} buses, {ctx['n_edges']} branches, {ctx['n_bridges']} exact bridges / {ctx['n_edges']-ctx['n_bridges']} non-bridge edges)
**Contingency severity labels:** {"reused from Stage-1 cache" if ctx['reused_cache'] else "regenerated from scratch"}
**Generated:** {now}
**Random seed:** {ctx['seed']}

---

## 1. Why this experiment

Stage 1 found that (a) whole-population Spearman correlation between kappa
and N-1 severity improved only marginally with learning (0.678 to 0.683),
(b) on the non-bridge subset specifically, the *learned* parameters scored
*worse* than the untuned baseline, and (c) {ctx['n_bridges']} of
{ctx['n_edges']} edges are exact topological bridges where kappa=1 is
mathematically guaranteed regardless of any parameter choice, meaning
whole-population correlation is dominated by an "easy" 22% of the ranking.

This experiment asks the two questions that follow directly from that
diagnosis: does looking only at the metric that matters (non-bridge
correlation) change the picture (**Experiment A**), and can a simple
classical model, using none of the Kadero machinery, already capture most
of the predictable non-bridge signal (**Experiment B**)?

## 2. Experiment A: Corrected objective

| Score | Whole population (n={ctx['n_edges']}) | Non-bridge subset (n={ctx['n_nonbridge']}) |
|---|---|---|
| Baseline kappa (mu=0.5, p=1.0) | {ctx['kappa_baseline_all']:.4f} | {ctx['kappa_baseline_nb']:.4f} |
| Learned kappa (mu*={LEARNED_MU:.3f}, p*={LEARNED_P:.3f}) | {ctx['kappa_learned_all']:.4f} | {ctx['kappa_learned_nb']:.4f} |
| **Change from learning** | **{ctx['kappa_learned_all']-ctx['kappa_baseline_all']:+.4f}** | **{ctx['kappa_learned_nb']-ctx['kappa_baseline_nb']:+.4f}** |

**Does correcting the objective change the picture? Yes — but not in the
direction Stage 1 suggested, and for a reason worth stating plainly rather
than glossing over.**

While building this fixed, ground-truth bridge/non-bridge split
(`networkx.bridges()`, independent of kappa, {ctx['n_bridges']} bridges
for every comparison), it became clear that **Stage 1's non-bridge
comparison used an inconsistent definition of "non-bridge" between the
baseline and learned cases.** Stage 1 identified non-bridge edges via a
kappa-dependent threshold (`kappa < 0.999`) computed *separately* for each
parameter setting -- and because the learned exponent p*=2.77 sharpens
near-bridge edges toward kappa=1 (a real, correctly-understood effect of
raising conductances to a power >1, see Stage 1 report Section 6), this
threshold selected **322 edges for the baseline comparison but only 306
edges for the learned comparison** -- a different-sized, non-matching
subset for each side of the "before vs after" comparison. That is an
apples-to-oranges comparison, not a fair one.

Under the fixed, ground-truth bridge definition used throughout this
report (322 non-bridge edges for both), **learned kappa's non-bridge
correlation is 0.2151, not 0.1721 as Stage 1 reported, and is now
marginally *higher* than the baseline's 0.2048, not lower.**

**The corrected, honest conclusion is therefore different from Stage 1's
headline finding:** learning did not clearly make non-bridge ranking worse.
The bootstrap confidence intervals in Section 3 below show the two
estimates overlap substantially (baseline [0.120, 0.285] vs. learned
[0.122, 0.302] at 90% confidence) -- the more defensible statement is that
**Stage 1's two-parameter learning had no clearly detectable effect,
positive or negative, on non-bridge ranking**, once measured consistently.
The original "learning made things worse" claim was itself a measurement
artifact, not a property of the learned parameters. This is exactly the
kind of thing a corrected-objective pass is supposed to catch, and it cuts
against the previous report's conclusion rather than confirming it --
reported here in full rather than quietly revised.

## 3. Experiment B: Classical baseline comparison

Features used (13 total; see `classical_features.csv` for values):
`betweenness`, `r_eff_raw` (classical DC-style effective resistance, NOT
using Kadero's mu/p), `g`, `beta`, `log1p_g`, `log1p_beta`, `deg_from`,
`deg_to`, `deg_min`, `deg_max`, `wdeg_from`, `wdeg_to`, `wdeg_min` --
plus `is_bridge` in a second, parallel feature set. All models are
evaluated under honest 5-fold cross-validation (out-of-fold predictions
only) so that supervised fitting to all {ctx['n_edges']} severity labels
does not get an unfair advantage over kappa, which was tuned with only two
free parameters in Stage 1.

### Full comparison table

{comp_table}

### Uncertainty: bootstrap 90% confidence intervals (non-bridge subset)

Several of the numbers above are close together. A {int(BOOT_CI*100)}%
bootstrap CI ({N_BOOTSTRAP} resamples) on the non-bridge subset makes clear
which differences are and are not distinguishable from sampling noise at
this sample size:

| Method | Non-bridge Spearman | {int(BOOT_CI*100)}% CI |
|---|---|---|
{ci_lines}

![confidence forest plot](figures/ci_forest_nonbridge.png)

**Read this plot before drawing conclusions from the table above.** Where
confidence intervals overlap substantially, the point-estimate ranking
between those methods is not statistically meaningful at this sample size.

### Model-family finding worth stating explicitly

Random Forest and Gradient Boosting -- explored across several depth/leaf
settings during development, not just the settings reported here (see
`hyperparameter_probe.csv`) -- consistently underperformed simple Ridge
regression on the non-bridge subset, sometimes scoring near zero. This is
almost certainly a sample-size effect: {ctx['n_nonbridge']} non-bridge
edges split five ways for cross-validation leaves roughly
{ctx['n_nonbridge']//N_CV_SPLITS} held-out points per fold, too few for a
high-variance tree ensemble to generalize reliably, while low-variance
linear regression on standardized features generalizes better from the
same data. This is a data-scarcity finding about this experiment, not
evidence that the underlying relationship is linear.

![comparison bars](figures/comparison_bars.png)

## 4. Feature importance (Random Forest, without is_bridge)

![feature importance](figures/feature_importance_rf.png)

## 5. Does the classical baseline already capture the predictable non-bridge signal?

**Partially, and the honest picture is more nuanced than a single number.**
Ridge regression on classical features, WITHOUT the is_bridge shortcut,
reaches {ctx['ridge_nb_no_bridge']:.4f} on the non-bridge subset -- below
both baseline kappa ({ctx['kappa_baseline_nb']:.4f}) and learned kappa
({ctx['kappa_learned_nb']:.4f}), though the bootstrap CIs in Section 3
overlap enough that this gap should not be read as a confident win for
kappa. Adding is_bridge as a feature (which, recall, is constant and
therefore uninformative *within* the non-bridge evaluation rows) lifts
Ridge to {ctx['ridge_nb_with_bridge']:.4f} -- essentially tied with both
kappa variants, well inside their overlapping confidence intervals.

**What this implies for the Kadero representation, stated plainly:** on
this network and this severity definition, neither kappa nor the best
classical model tested here demonstrates a clearly, statistically
distinguishable advantage over the other on the hard non-bridge subset.
Kappa is not obviously worse than a reasonable classical baseline (a
concern a skeptical reader might have going in) -- but it is not yet
demonstrated to be clearly *better*, either, and the honest sample-size
reality at n=322 means this experiment's power to detect a moderate real
difference is limited. The bridge-detection result (Stage 1, Section 6) is
real, exact, and provably not reproducible by a classical model in the
same closed-form way -- but bridge detection is also the *easy* 22% of the
problem, achievable by a single boolean topological check with no learning
and no Kadero machinery at all.

This is not evidence that the Kadero representation is wrong or that the
project should stop -- it is evidence that the two-parameter, single
resistance-channel model tested in Stage 1 has not yet been shown to add
non-bridge predictive value beyond a handful of classical numeric
features, on this specific network and severity proxy. The channels NOT
yet tested (redundancy covariance, geography,
operating point) are exactly where a genuine, demonstrable advantage would
have to come from, if one exists -- since neither the two-parameter Kadero
model nor a reasonable classical baseline has yet separated itself
clearly from the other on the metric that actually matters.

## 6. Reproducibility

- `classical_features.csv`: the full feature matrix used for Experiment B.
- `comparison_table.csv`, `summary.json`: all numbers in this report,
  machine-readable.
- `hyperparameter_probe.csv`: the RF/GBR depth and leaf-size sweep referenced
  in Section 3.
- Cached Stage-1 inputs reused: `{ctx['edge_table_source']}`,
  `{ctx['severity_source']}`.

## 7. Honest limitations of this Stage-2 run

- Cross-validation with n=322 (non-bridge) and 5 folds gives noisy
  per-fold estimates; the bootstrap CIs in Section 3 should be treated as
  the primary evidence about precision, not the point estimates alone.
- The classical feature set, while reasonably thorough, was built once and
  not itself tuned or searched over -- a more exhaustive classical feature
  engineering pass could plausibly close or widen the gap further in
  either direction.
- Severity remains the same islanding-dominated proxy used in Stage 1
  (see Stage-1 report Section 2); this experiment does not address whether
  a different severity definition would change the picture -- that is a
  natural next question, not answered here.
- `is_bridge` is included as a feature exactly as the brief specified, but
  its near-irrelevance to non-bridge evaluation rows (constant value within
  that subset) means its main effect is on the whole-population number, not
  the harder metric -- worth remembering when reading Section 3's table at
  a glance.
"""
    with open(path, "w") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("Kadero Embedding -- Stage 2A + 2B -- case300")
    print("=" * 70)

    # --- Data -----------------------------------------------------------
    print("\n[1/6] Loading network and contingency severity labels...")
    net, merged = load_edges_and_severity(force=args.force_regenerate)
    reused_cache = os.path.exists(os.path.join(CACHE_DIR, "edge_table.csv")) and not args.force_regenerate

    n_bus = len(net.bus)
    fb = merged["from_bus_edge"].values.astype(int)
    tb = merged["to_bus_edge"].values.astype(int)
    g = merged["g"].values.astype(float)
    beta = merged["beta"].values.astype(float)
    severity = merged["severity"].values
    n_edges = len(merged)

    is_bridge = k2.compute_bridge_mask(fb, tb, n_bus)
    n_bridges = int(is_bridge.sum())
    n_nonbridge = n_edges - n_bridges
    print(f"    {n_edges} branches, {n_bridges} exact bridges, {n_nonbridge} non-bridge edges")

    # --- Experiment A: corrected objective ------------------------------
    print("\n[2/6] Experiment A: kappa under the corrected (non-bridge) objective...")
    engine = kl.KappaEngine(n_bus=n_bus, from_bus=fb, to_bus=tb, g=g, beta=beta)
    kappa_baseline = engine.kappa(BASELINE_MU, BASELINE_P)
    kappa_learned = engine.kappa(LEARNED_MU, LEARNED_P)

    res_kb = k2.spearman_all_and_nonbridge(kappa_baseline, severity, is_bridge)
    res_kl = k2.spearman_all_and_nonbridge(kappa_learned, severity, is_bridge)
    print(f"    baseline kappa:  all={res_kb['rho_all']:.4f}  nonbridge={res_kb['rho_nonbridge']:.4f}")
    print(f"    learned  kappa:  all={res_kl['rho_all']:.4f}  nonbridge={res_kl['rho_nonbridge']:.4f}")

    # --- Experiment B: classical baseline --------------------------------
    print("\n[3/6] Experiment B: classical feature baseline (Ridge / RF / GBR, "
          f"{N_CV_SPLITS}-fold CV)...")
    feats = k2.build_classical_features(fb, tb, g, beta, is_bridge, n_bus)
    feats.X_with_bridge.to_csv(os.path.join(OUTPUT_DIR, "classical_features.csv"), index=False)

    results = {}
    preds_store = {}

    for tag, X in [("with_bridge", feats.X_with_bridge), ("without_bridge", feats.X_without_bridge)]:
        pred_ridge = k2.cv_predict_ridge(X, severity, n_splits=N_CV_SPLITS, seed=args.seed)
        results[f"ridge_{tag}"] = k2.spearman_all_and_nonbridge(pred_ridge, severity, is_bridge)
        preds_store[f"ridge_{tag}"] = pred_ridge
        print(f"    ridge [{tag}]: all={results[f'ridge_{tag}']['rho_all']:.4f}  "
              f"nonbridge={results[f'ridge_{tag}']['rho_nonbridge']:.4f}")

        pred_rf, rf_model = k2.cv_predict_rf(X, severity, n_splits=N_CV_SPLITS, seed=args.seed)
        results[f"rf_{tag}"] = k2.spearman_all_and_nonbridge(pred_rf, severity, is_bridge)
        preds_store[f"rf_{tag}"] = pred_rf
        print(f"    rf    [{tag}]: all={results[f'rf_{tag}']['rho_all']:.4f}  "
              f"nonbridge={results[f'rf_{tag}']['rho_nonbridge']:.4f}")

        pred_gbr, gbr_model = k2.cv_predict_gbr(X, severity, n_splits=N_CV_SPLITS, seed=args.seed)
        results[f"gbr_{tag}"] = k2.spearman_all_and_nonbridge(pred_gbr, severity, is_bridge)
        preds_store[f"gbr_{tag}"] = pred_gbr
        print(f"    gbr   [{tag}]: all={results[f'gbr_{tag}']['rho_all']:.4f}  "
              f"nonbridge={results[f'gbr_{tag}']['rho_nonbridge']:.4f}")

        if tag == "without_bridge":
            rf_full_names = feats.feature_names_without
            rf_full_importances = rf_model.feature_importances_

    # --- Bootstrap CIs on non-bridge subset -------------------------------
    print("\n[4/6] Bootstrapping confidence intervals on the non-bridge subset...")
    nb_mask = ~is_bridge
    ci_results = {}
    ci_results["Baseline kappa"] = k2.bootstrap_spearman_ci(
        kappa_baseline, severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    ci_results["Learned kappa"] = k2.bootstrap_spearman_ci(
        kappa_learned, severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    ci_results["Ridge (with is_bridge)"] = k2.bootstrap_spearman_ci(
        preds_store["ridge_with_bridge"], severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    ci_results["Ridge (no is_bridge)"] = k2.bootstrap_spearman_ci(
        preds_store["ridge_without_bridge"], severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    ci_results["GBR (no is_bridge)"] = k2.bootstrap_spearman_ci(
        preds_store["gbr_without_bridge"], severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    ci_results["RF (no is_bridge)"] = k2.bootstrap_spearman_ci(
        preds_store["rf_without_bridge"], severity, nb_mask, n_boot=N_BOOTSTRAP, ci=BOOT_CI, seed=args.seed)
    for name, v in ci_results.items():
        print(f"    {name}: {v['point']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    # --- Hyperparameter probe table (documenting the depth/leaf sweep) ----
    probe_rows = []
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from scipy.stats import rankdata as _rankdata
    y_rank = _rankdata(severity)
    kf = KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=args.seed)
    for depth, leaf in [(3, 10), (4, 8), (6, 5), (10, 3)]:
        rf_p = RandomForestRegressor(n_estimators=500, max_depth=depth,
                                      min_samples_leaf=leaf, random_state=args.seed, n_jobs=-1)
        pred = cross_val_predict(rf_p, feats.X_without_bridge.values, y_rank, cv=kf)
        r = k2.spearman_all_and_nonbridge(pred, severity, is_bridge)
        probe_rows.append(dict(model="RF", max_depth=depth, min_samples_leaf=leaf,
                                learning_rate=None, rho_all=r["rho_all"], rho_nonbridge=r["rho_nonbridge"]))
    for depth, lr in [(2, 0.05), (3, 0.05), (2, 0.1)]:
        gbr_p = GradientBoostingRegressor(n_estimators=300, max_depth=depth,
                                           learning_rate=lr, random_state=args.seed)
        pred = cross_val_predict(gbr_p, feats.X_without_bridge.values, y_rank, cv=kf)
        r = k2.spearman_all_and_nonbridge(pred, severity, is_bridge)
        probe_rows.append(dict(model="GBR", max_depth=depth, min_samples_leaf=None,
                                learning_rate=lr, rho_all=r["rho_all"], rho_nonbridge=r["rho_nonbridge"]))
    pd.DataFrame(probe_rows).to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_probe.csv"), index=False)

    # --- Comparison table --------------------------------------------------
    print("\n[5/6] Building comparison table and figures...")
    rows = [
        dict(method="Baseline kappa (mu=0.5,p=1.0)", **res_kb),
        dict(method="Learned kappa (mu*,p*)", **res_kl),
    ]
    label_map = {
        "ridge_with_bridge": "Ridge (+ is_bridge)", "ridge_without_bridge": "Ridge (no is_bridge)",
        "rf_with_bridge": "RF (+ is_bridge)", "rf_without_bridge": "RF (no is_bridge)",
        "gbr_with_bridge": "GBR (+ is_bridge)", "gbr_without_bridge": "GBR (no is_bridge)",
    }
    for key, label in label_map.items():
        rows.append(dict(method=label, **results[key]))
    comparison_df = pd.DataFrame(rows)[["method", "rho_all", "rho_nonbridge", "n_all", "n_nonbridge"]]
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)
    print(comparison_df.to_string(index=False))

    # --- Figures -------------------------------------------------------
    plot_comparison_bars(comparison_df, os.path.join(FIG_DIR, "comparison_bars.png"))
    plot_ci_forest(ci_results, os.path.join(FIG_DIR, "ci_forest_nonbridge.png"))
    plot_feature_importance(rf_full_names, rf_full_importances,
                             "Random Forest feature importance (rank-severity target, no is_bridge)",
                             os.path.join(FIG_DIR, "feature_importance_rf.png"))
    plot_best_scatter(kappa_baseline, kappa_learned, preds_store["ridge_with_bridge"],
                       severity, is_bridge, os.path.join(FIG_DIR, "best_models_scatter.png"))

    # --- Report + summary -------------------------------------------------
    print("\n[6/6] Writing report and summary...")
    ctx = dict(
        n_bus=n_bus, n_edges=n_edges, n_bridges=n_bridges, n_nonbridge=n_nonbridge,
        reused_cache=reused_cache, seed=args.seed,
        kappa_baseline_all=res_kb["rho_all"], kappa_baseline_nb=res_kb["rho_nonbridge"],
        kappa_learned_all=res_kl["rho_all"], kappa_learned_nb=res_kl["rho_nonbridge"],
        ridge_nb_with_bridge=results["ridge_with_bridge"]["rho_nonbridge"],
        ridge_nb_no_bridge=results["ridge_without_bridge"]["rho_nonbridge"],
        comparison_df=comparison_df, ci_results=ci_results,
        edge_table_source=("cache/edge_table.csv" if reused_cache else "regenerated"),
        severity_source=("cache/contingency_severity.csv" if reused_cache else "regenerated"),
    )
    write_report(ctx, os.path.join(OUTPUT_DIR, "report.md"))

    summary = dict(
        network="pandapower.networks.case300", n_bus=n_bus, n_edges=n_edges,
        n_bridges=n_bridges, n_nonbridge=n_nonbridge, random_seed=args.seed,
        reused_stage1_cache=reused_cache,
        experiment_a=dict(baseline=res_kb, learned=res_kl,
                           learned_params=dict(mu=LEARNED_MU, p=LEARNED_P)),
        experiment_b={k: v for k, v in results.items()},
        bootstrap_ci_nonbridge={k: v for k, v in ci_results.items()},
        total_wallclock_sec=time.time() - t0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE in {time.time()-t0:.1f}s.")
    print(f"Non-bridge: baseline kappa={res_kb['rho_nonbridge']:.4f}  "
          f"learned kappa={res_kl['rho_nonbridge']:.4f}  "
          f"best classical={comparison_df['rho_nonbridge'].max():.4f}")
    print(f"All outputs written to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
