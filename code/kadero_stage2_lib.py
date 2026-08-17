"""
kadero_stage2_lib.py
=====================
Stage-2-specific helpers, kept separate from the reusable core `kadero_lib`
so Stage 1's library stays untouched and reusable as-is.

Contains:
  - ground-truth bridge detection (independent of kappa, via networkx)
  - the classical baseline feature matrix (betweenness, raw effective
    resistance, conductances, degree features)
  - a cross-validated evaluation harness shared by every model (Ridge and
    Random Forest, with and without the is_bridge feature) so that kappa
    and the classical models are scored by an identical, honest protocol
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, rankdata

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---------------------------------------------------------------------------
# Ground-truth bridge mask (independent of any kappa computation)
# ---------------------------------------------------------------------------

def compute_bridge_mask(from_bus: np.ndarray, to_bus: np.ndarray, n_bus: int) -> np.ndarray:
    """
    True topological bridges via networkx's purely combinatorial bridge
    detection, exactly as cross-validated in Stage 1 (kadero_lib.
    validate_against_theory). An edge is a bridge in the underlying
    MULTIgraph iff (a) its bus pair is a bridge of the simple graph AND
    (b) that bus pair is not duplicated by a parallel branch (a parallel
    line/transformer between the same two buses is never itself a bridge,
    since the parallel partner is an alternate path).
    """
    pair_counts = Counter()
    for a, b in zip(from_bus, to_bus):
        pair_counts[tuple(sorted((int(a), int(b))))] += 1

    G = nx.Graph()
    G.add_nodes_from(range(n_bus))
    for a, b in zip(from_bus, to_bus):
        G.add_edge(int(a), int(b))
    bridge_pairs = set(tuple(sorted(e)) for e in nx.bridges(G))

    return np.array([
        (tuple(sorted((int(a), int(b)))) in bridge_pairs
         and pair_counts[tuple(sorted((int(a), int(b))))] == 1)
        for a, b in zip(from_bus, to_bus)
    ])


# ---------------------------------------------------------------------------
# Classical feature matrix
# ---------------------------------------------------------------------------

@dataclass
class ClassicalFeatures:
    X_with_bridge: pd.DataFrame
    X_without_bridge: pd.DataFrame
    feature_names_with: list
    feature_names_without: list


def build_classical_features(from_bus: np.ndarray, to_bus: np.ndarray,
                              g: np.ndarray, beta: np.ndarray,
                              is_bridge: np.ndarray, n_bus: int) -> ClassicalFeatures:
    """
    Builds the "strong, transparent classical baseline" feature set
    requested for Stage 2, Experiment B:

      - edge betweenness centrality (unweighted, standard networkx)
      - raw effective resistance (classical electrical distance, computed
        from the reactive/susceptance channel ONLY -- beta_e -- which is
        the standard DC-flow-style electrical distance from the power
        systems literature, deliberately NOT using the Kadero-specific
        (mu, p) mixing, so this is a genuinely independent classical
        quantity, not a relabeled Kadero output)
      - g_e, beta_e (raw) and log1p(g_e), log1p(beta_e) (both channels are
        heavy-tailed -- beta_e alone spans nearly 4 orders of magnitude on
        case300 -- so both raw and log-compressed versions are offered and
        the model is free to use whichever helps)
      - unweighted topological degree of each endpoint bus, and their
        min/max
      - weighted degree of each endpoint bus (sum of g+beta over incident
        branches -- the classical, unlearned analogue of the Kadero
        "strength" quantity)
      - is_bridge (0/1): included in one feature set, excluded in a second,
        parallel feature set, exactly as requested, since bridge status
        alone is known (Stage 1) to explain much of the whole-population
        signal trivially.
    """
    n = len(from_bus)

    # Betweenness (unweighted, simple graph -- parallel branches between the
    # same two buses share the same betweenness value by construction).
    G = nx.Graph()
    G.add_nodes_from(range(n_bus))
    for a, b in zip(from_bus, to_bus):
        G.add_edge(int(a), int(b))
    bc = nx.edge_betweenness_centrality(G)
    betweenness = np.array([
        bc.get((int(a), int(b)), bc.get((int(b), int(a)), 0.0))
        for a, b in zip(from_bus, to_bus)
    ])

    # Raw (classical, non-Kadero) effective resistance from beta alone.
    y_beta = np.clip(beta, 1e-10, None)
    L = np.zeros((n_bus, n_bus))
    np.add.at(L, (from_bus, from_bus), y_beta)
    np.add.at(L, (to_bus, to_bus), y_beta)
    np.add.at(L, (from_bus, to_bus), -y_beta)
    np.add.at(L, (to_bus, from_bus), -y_beta)
    L_pinv = np.linalg.pinv(L, hermitian=True)
    Rii = np.diag(L_pinv)
    r_eff_raw = Rii[from_bus] + Rii[to_bus] - 2.0 * L_pinv[from_bus, to_bus]

    # Degree features.
    deg_count = np.zeros(n_bus, dtype=float)
    for a, b in zip(from_bus, to_bus):
        deg_count[a] += 1
        deg_count[b] += 1
    deg_from, deg_to = deg_count[from_bus], deg_count[to_bus]

    w = g + beta
    wdeg = np.zeros(n_bus, dtype=float)
    np.add.at(wdeg, from_bus, w)
    np.add.at(wdeg, to_bus, w)
    wdeg_from, wdeg_to = wdeg[from_bus], wdeg[to_bus]

    base = pd.DataFrame({
        "betweenness": betweenness,
        "r_eff_raw": r_eff_raw,
        "g": g,
        "beta": beta,
        "log1p_g": np.log1p(g),
        "log1p_beta": np.log1p(beta),
        "deg_from": deg_from,
        "deg_to": deg_to,
        "deg_min": np.minimum(deg_from, deg_to),
        "deg_max": np.maximum(deg_from, deg_to),
        "wdeg_from": wdeg_from,
        "wdeg_to": wdeg_to,
        "wdeg_min": np.minimum(wdeg_from, wdeg_to),
    })

    names_without = list(base.columns)
    with_bridge = base.copy()
    with_bridge["is_bridge"] = is_bridge.astype(float)
    names_with = list(with_bridge.columns)

    return ClassicalFeatures(
        X_with_bridge=with_bridge, X_without_bridge=base,
        feature_names_with=names_with, feature_names_without=names_without,
    )


# ---------------------------------------------------------------------------
# Cross-validated model evaluation (shared, honest protocol)
# ---------------------------------------------------------------------------

def cv_predict_ridge(X: pd.DataFrame, severity: np.ndarray, n_splits: int = 5,
                      seed: int = 42) -> np.ndarray:
    """
    Ridge regression on RANK-transformed severity (a standard trick for
    optimizing something aligned with Spearman correlation using a simple
    linear model), evaluated via 5-fold cross-validation so that reported
    performance reflects genuine held-out generalization, not in-sample
    fit -- this matters especially for the `is_bridge` feature, which
    could otherwise trivially "explain" 22% of the ranking.
    Returns out-of-fold predictions, aligned with the input row order.
    """
    y_rank = rankdata(severity)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(model, X.values, y_rank, cv=kf)


def cv_predict_rf(X: pd.DataFrame, severity: np.ndarray, n_splits: int = 5,
                   seed: int = 42) -> tuple[np.ndarray, RandomForestRegressor]:
    """
    Random Forest regression on rank-transformed severity, evaluated via
    the same 5-fold cross-validation protocol as Ridge. Also fits one
    final model on the FULL dataset (not used for evaluation, only for
    reporting feature importances) since per-fold importances would be
    five separate, harder-to-summarize sets of numbers.
    """
    y_rank = rankdata(severity)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=5,
        random_state=seed, n_jobs=-1,
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(rf, X.values, y_rank, cv=kf)

    rf_full = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=5,
        random_state=seed, n_jobs=-1,
    )
    rf_full.fit(X.values, y_rank)
    return preds, rf_full


def cv_predict_gbr(X: pd.DataFrame, severity: np.ndarray, n_splits: int = 5,
                    seed: int = 42) -> tuple[np.ndarray, GradientBoostingRegressor]:
    """
    Gradient Boosting regression on rank-transformed severity, same
    protocol as Ridge/RF. Included alongside Random Forest per the brief
    ("small Gradient Boosting / Random Forest regressor") -- in initial
    hyperparameter probing (see report) GBR with shallow trees (depth 2)
    consistently outperformed RF on the non-bridge subset, though neither
    tree ensemble matched Ridge; this function uses those probed settings
    rather than the RF defaults, documented explicitly rather than
    silently tuned.
    """
    y_rank = rankdata(severity)
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.1, random_state=seed,
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(gbr, X.values, y_rank, cv=kf)

    gbr_full = GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.1, random_state=seed,
    )
    gbr_full.fit(X.values, y_rank)
    return preds, gbr_full


def spearman_all_and_nonbridge(pred: np.ndarray, severity: np.ndarray,
                                is_bridge: np.ndarray) -> dict:
    rho_all, p_all = spearmanr(pred, severity)
    nb = ~is_bridge
    if nb.sum() > 5:
        rho_nb, p_nb = spearmanr(pred[nb], severity[nb])
    else:
        rho_nb, p_nb = float("nan"), float("nan")
    return dict(rho_all=float(rho_all), p_all=float(p_all),
                rho_nonbridge=float(rho_nb), p_nonbridge=float(p_nb),
                n_all=len(pred), n_nonbridge=int(nb.sum()))


def bootstrap_spearman_ci(pred: np.ndarray, severity: np.ndarray, mask: np.ndarray = None,
                           n_boot: int = 2000, ci: float = 0.90, seed: int = 0) -> dict:
    """
    Nonparametric bootstrap confidence interval for the Spearman correlation
    between `pred` and `severity`, restricted to `mask` if given.

    This matters here specifically because several of the headline numbers
    in this report are close together (e.g. kappa non-bridge = 0.205 vs.
    Ridge-with-bridge non-bridge = 0.211): a bare point estimate invites
    over-reading a difference that may not be distinguishable from noise
    at n ~ 300-400. Resampling edges with replacement and recomputing
    Spearman each time gives an honest sense of that noise floor.
    """
    rng = np.random.default_rng(seed)
    if mask is not None:
        pred, severity = pred[mask], severity[mask]
    n = len(pred)
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(pred[idx], severity[idx])
        boot_rhos[b] = r
    lo = float(np.nanpercentile(boot_rhos, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(boot_rhos, (1 - (1 - ci) / 2) * 100))
    point, _ = spearmanr(pred, severity)
    return dict(point=float(point), ci_low=lo, ci_high=hi, ci_level=ci, n=n)
