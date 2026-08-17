"""
kadero_lib.py
=============
Core reusable library for the Kadero Embedding — Stage 1 Precision Learning
experiment (case300).

This module implements, in simplified/reduced form, the pieces of the Kadero
mathematical foundation needed for this first learning stage:

  1. Extraction of per-unit electrical branch data (real/thermal channel `g`
     and reactive/phase channel `beta`) directly from pandapower's internal
     per-unit network model, for every line AND transformer, uniformly.

  2. The reference conductance and edge criticality functional kappa_e,
     following the Foundation document (Section 2.3):

         y_ref(e)  = mu * g_e + (1 - mu) * beta_e            (mixing)
         y_theta(e) = y_ref(e) ** p                           (exponent)
         L_theta    = sum_e y_theta(e) * chi_e chi_e^T         (weighted Laplacian)
         R_eff(e)   = chi_e^T L_theta^+ chi_e                  (effective resistance)
         kappa_e    = y_theta(e) * R_eff(e)                    (edge criticality, in (0,1])

     This is the SIMPLIFIED, two-free-parameter (mu, p) form of kappa used for
     Stage 1. It is a genuine special case of the full framework (mu, p are
     two of the entries in the Free Parameter Registry of the Foundation
     document), not a different, ad hoc score.

  3. N-1 contingency severity generation via pandapower AC power flow,
     with explicit handling of non-convergence and islanding (topological
     disconnection of part of the network from the slack bus), since these
     are common and highly significant outcomes on a real transmission
     topology and must not be silently dropped.

  4. A simple, documented "precision" direct-search (Hooke-Jeeves-style
     pattern search with fixed-then-shrinking step, multiple random
     restarts, and early stopping) for one-parameter-at-a-time learning,
     exactly as specified for this Stage 1 experiment.

Every numerical design choice (severity weights, search bounds, patience,
etc.) is a named constant at the top of the relevant function or in
`config.py`-style dictionaries passed in by the caller — nothing is buried.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

warnings.filterwarnings("ignore")  # silence pandapower's numba-not-installed notice etc.


# ---------------------------------------------------------------------------
# 1. Electrical data extraction
# ---------------------------------------------------------------------------

def extract_edge_table(net: "pp.pandapowerNet") -> pd.DataFrame:
    """
    Build the per-branch electrical data table (one row per line AND per
    transformer) directly from pandapower's internal per-unit network model
    (net._ppc['branch']), which pandapower itself uses to run power flow.

    Using pandapower's own per-unit conversion (rather than re-deriving
    transformer impedance from vk_percent / vkr_percent by hand) is a
    deliberate robustness choice: case300's transformer table contains some
    unusual vk_percent values, and reusing pandapower's already-validated
    internal conversion avoids re-implementing (and possibly getting wrong)
    the transformer per-unit impedance formula for this specific dataset.

    Returns a DataFrame with columns:
        element_type   'line' or 'trafo'
        element_index  index into net.line / net.trafo
        from_bus, to_bus   original pandapower bus indices
        r_pu, x_pu     per-unit series resistance / reactance
        g, beta        real (conductance) and reactive (susceptance-magnitude)
                       Kadero channel weights, g_e = r/(r^2+x^2),
                       beta_e = |x|/(r^2+x^2)   [Foundation Sec. 2.1]

    Requires that `pp.runpp(net)` has already been called at least once so
    that `net._ppc` is populated.
    """
    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X

    if "_ppc" not in net or net._ppc is None:
        raise RuntimeError(
            "net._ppc is not populated. Call pp.runpp(net) once before "
            "extract_edge_table()."
        )

    branch = net._ppc["branch"]
    lookup = net._pd2ppc_lookups["bus"]  # original pandapower bus idx -> ppc internal idx
    inv_lookup = {ppc_idx: orig_idx for orig_idx, ppc_idx in enumerate(lookup)}

    n_line = len(net.line)
    n_trafo = len(net.trafo)
    n_branch = branch.shape[0]
    if n_branch != n_line + n_trafo:
        raise RuntimeError(
            f"Unexpected branch count: ppc has {n_branch} branches but "
            f"net has {n_line} lines + {n_trafo} trafos = {n_line + n_trafo}. "
            "The assumption that ppc branches are ordered [lines..., trafos...] "
            "in table order (verified for pandapower 3.x on case300) may not "
            "hold for this pandapower version/network."
        )

    rows = []
    for row in range(n_branch):
        fb = inv_lookup[int(branch[row, F_BUS])]
        tb = inv_lookup[int(branch[row, T_BUS])]
        r = float(branch[row, BR_R])
        x = float(branch[row, BR_X])
        x_abs = abs(x)  # Assumption A (Foundation Sec. 1) requires x_e > 0;
        # case300 contains exactly one line (a modeled series-capacitor
        # branch) with negative reactance. We take |x| for that branch and
        # flag it explicitly in the returned table via `x_was_negative`.
        z2 = r * r + x_abs * x_abs
        g = r / z2 if z2 > 0 else 0.0
        beta = x_abs / z2 if z2 > 0 else 0.0

        is_line = row < n_line
        rows.append(
            dict(
                element_type="line" if is_line else "trafo",
                element_index=row if is_line else row - n_line,
                from_bus=fb,
                to_bus=tb,
                r_pu=r,
                x_pu=x,
                g=g,
                beta=beta,
                x_was_negative=bool(x < 0),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Parametric Kappa (simplified 2-parameter Stage-1 form)
# ---------------------------------------------------------------------------

@dataclass
class KappaEngine:
    """
    Precomputes fixed arrays (bus indices, g, beta) once, then evaluates
    kappa_e(mu, p) cheaply and repeatedly -- this is the object the search
    loop calls on every evaluation.
    """
    n_bus: int
    from_bus: np.ndarray
    to_bus: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    eps: float = 1e-10

    @classmethod
    def from_edge_table(cls, edge_df: pd.DataFrame, n_bus: int) -> "KappaEngine":
        return cls(
            n_bus=n_bus,
            from_bus=edge_df["from_bus"].values.astype(int),
            to_bus=edge_df["to_bus"].values.astype(int),
            g=edge_df["g"].values.astype(float),
            beta=edge_df["beta"].values.astype(float),
        )

    def y_theta(self, mu: float, p: float) -> np.ndarray:
        """y_ref(e) = mu*g_e + (1-mu)*beta_e ;  y_theta(e) = y_ref(e)**p"""
        y_ref = np.clip(mu * self.g + (1.0 - mu) * self.beta, self.eps, None)
        return y_ref ** p

    def kappa(self, mu: float, p: float) -> np.ndarray:
        """
        Returns kappa_e for every edge, in the same row order as the edge
        table this engine was built from.

        kappa_e = y_theta(e) * R_eff(e),  R_eff(e) = chi_e^T L_theta^+ chi_e

        Sanity property (checked in tests/validate_foundation.py): for any
        (mu, p), sum_e kappa_e == n_bus - 1 exactly (Foundation Prop. 2.4 /
        Companion Prop. 3.3, the conservation law), and kappa_e in (0, 1].
        """
        yth = self.y_theta(mu, p)
        L = np.zeros((self.n_bus, self.n_bus))
        np.add.at(L, (self.from_bus, self.from_bus), yth)
        np.add.at(L, (self.to_bus, self.to_bus), yth)
        np.add.at(L, (self.from_bus, self.to_bus), -yth)
        np.add.at(L, (self.to_bus, self.from_bus), -yth)
        L_pinv = np.linalg.pinv(L, hermitian=True)
        Rii = np.diag(L_pinv)
        R_eff = Rii[self.from_bus] + Rii[self.to_bus] - 2.0 * L_pinv[self.from_bus, self.to_bus]
        return yth * R_eff


# ---------------------------------------------------------------------------
# 3. N-1 contingency severity generation
# ---------------------------------------------------------------------------

# Severity model constants (documented, not buried):
SEVERITY_NONCONVERGED = 1000.0   # power flow failed to converge: worst tier
ISLANDING_BASE = 300.0           # flat penalty once ANY bus loses supply
ISLANDING_PER_BUS = 5.0          # additional penalty per unsupplied bus
OVERLOAD_PENALTY_PER_ELEMENT = 10.0  # penalty per element pushed over 100% loading


def run_n1_severity(net_template: "pp.pandapowerNet", edge_df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    """
    For every line and transformer in `edge_df`, take it out of service, run
    AC power flow on the rest of the network, and compute a severity score.

    Severity design (all three failure modes are treated explicitly, since
    on case300's default loading condition, thermal overload is rare but
    topological islanding is common -- see the accompanying report):

      - Power flow raises an exception or fails to converge:
            severity = SEVERITY_NONCONVERGED   (flat, worst tier)
      - Power flow converges but leaves some buses topologically unsupplied
        (disconnected from the slack bus -- pandapower's `check_connectivity`
        silently excludes these from the solved system rather than raising):
            severity = ISLANDING_BASE + ISLANDING_PER_BUS * n_unsupplied
                       + overload_component (see below), i.e. islanding cases
            still also pick up the overload term from the surviving island.
      - Otherwise (no islanding, converged):
            overload_component = max(0, max_loading_percent - 100)
                                   + OVERLOAD_PENALTY_PER_ELEMENT * n_overloaded
            severity = overload_component

    A single, reused pandapower net object is used for all contingencies
    (toggling `in_service` and restoring it after each run) since this is
    roughly 7x faster than reconstructing the case from scratch every time,
    with no difference in results.

    Returns a DataFrame with one row per (element_type, element_index),
    matching the rows of `edge_df`.
    """
    net = net_template  # reused in place; caller should pass a fresh object it doesn't need after
    n_line = len(net.line)
    n_trafo = len(net.trafo)

    records = []
    t_start = time.time()
    total = n_line + n_trafo
    done = 0

    for etype, table in (("line", net.line), ("trafo", net.trafo)):
        n_elem = len(table)
        for idx in range(n_elem):
            table.at[idx, "in_service"] = False
            converged = False
            n_unsupplied = 0
            max_loading = np.nan
            n_overloaded = 0
            try:
                pp.runpp(net, init="auto", enforce_q_lims=False,
                         check_connectivity=True, numba=False)
                converged = bool(net["converged"])
                if not converged:
                    severity = SEVERITY_NONCONVERGED
                else:
                    unsupplied = top.unsupplied_buses(net)
                    n_unsupplied = len(unsupplied)

                    loadings = [net.res_line.loading_percent[net.line.in_service.values].dropna()]
                    if n_trafo > 0:
                        loadings.append(
                            net.res_trafo.loading_percent[net.trafo.in_service.values].dropna()
                        )
                    all_load = pd.concat(loadings) if loadings else pd.Series(dtype=float)
                    max_loading = float(all_load.max()) if len(all_load) else 0.0
                    n_overloaded = int((all_load > 100.0).sum())

                    overload_component = (
                        max(0.0, max_loading - 100.0)
                        + OVERLOAD_PENALTY_PER_ELEMENT * n_overloaded
                    )
                    islanding_component = (
                        ISLANDING_BASE + ISLANDING_PER_BUS * n_unsupplied
                        if n_unsupplied > 0 else 0.0
                    )
                    severity = overload_component + islanding_component
            except Exception:
                severity = SEVERITY_NONCONVERGED
                converged = False
            finally:
                table.at[idx, "in_service"] = True

            fb = int(table.at[idx, "from_bus"] if etype == "line" else table.at[idx, "hv_bus"])
            tb = int(table.at[idx, "to_bus"] if etype == "line" else table.at[idx, "lv_bus"])
            records.append(dict(
                element_type=etype, element_index=idx, from_bus=fb, to_bus=tb,
                converged=converged, n_unsupplied=n_unsupplied,
                max_loading_percent=max_loading, n_overloaded=n_overloaded,
                severity=severity,
            ))
            done += 1
            if verbose and done % 100 == 0:
                print(f"    ... {done}/{total} contingencies done "
                      f"({time.time() - t_start:.1f}s elapsed)")

    if verbose:
        print(f"    N-1 severity generation finished: {total} contingencies "
              f"in {time.time() - t_start:.1f}s")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Precision Learning: one-parameter-at-a-time pattern search
# ---------------------------------------------------------------------------

def validate_against_theory(engine: "KappaEngine", from_bus: np.ndarray, to_bus: np.ndarray,
                             mu: float, p: float) -> dict:
    """
    Cross-checks the numerical kappa implementation against two provable
    properties from the Foundation document, independent of any learning:

      1. Conservation law (Foundation Prop. 2.4 / Companion Prop. 3.3):
         sum_e kappa_e == n_bus - 1 exactly, for ANY (mu, p).
      2. Bridge equivalence (Foundation Prop. 2.5 / Companion Prop. 3.2):
         kappa_e == 1 if and only if e is a topological bridge -- checked
         here against networkx's independent, purely combinatorial bridge
         detection algorithm as ground truth (not derived from kappa at all).

    Returns a dict summarizing both checks; raises no exception even on
    failure (the caller decides how to report it), except that a hard
    numerical bug (e.g. negative kappa) is flagged explicitly.
    """
    import networkx as nx
    from collections import Counter

    kappa = engine.kappa(mu, p)

    conservation_target = engine.n_bus - 1
    conservation_error = float(abs(kappa.sum() - conservation_target))

    pair_counts = Counter()
    for fb, tb in zip(from_bus, to_bus):
        pair_counts[tuple(sorted((int(fb), int(tb))))] += 1

    G = nx.Graph()
    G.add_nodes_from(range(engine.n_bus))
    for fb, tb in zip(from_bus, to_bus):
        G.add_edge(int(fb), int(tb))
    nx_bridge_pairs = set(tuple(sorted(e)) for e in nx.bridges(G))

    true_bridge_mask = np.array([
        (tuple(sorted((int(fb), int(tb)))) in nx_bridge_pairs
         and pair_counts[tuple(sorted((int(fb), int(tb))))] == 1)
        for fb, tb in zip(from_bus, to_bus)
    ])
    kappa_is_one = np.abs(kappa - 1.0) < 1e-6
    bridge_mismatches = int(np.sum(true_bridge_mask != kappa_is_one))

    return dict(
        n_edges=len(kappa),
        kappa_min=float(kappa.min()),
        kappa_max=float(kappa.max()),
        kappa_in_valid_range=bool(kappa.min() > -1e-6 and kappa.max() < 1 + 1e-6),
        conservation_target=conservation_target,
        conservation_sum=float(kappa.sum()),
        conservation_error=conservation_error,
        conservation_ok=bool(conservation_error < 1e-3),
        n_true_bridges_networkx=int(true_bridge_mask.sum()),
        n_kappa_equals_one=int(kappa_is_one.sum()),
        bridge_equivalence_mismatches=bridge_mismatches,
        bridge_equivalence_ok=bool(bridge_mismatches == 0),
    )


@dataclass
class ParamSpec:
    """Specification of one learnable parameter for the sequential search."""
    name: str
    lower: float
    upper: float
    init_step: float = 0.1
    min_step: float = 0.005
    stall_patience: int = 12   # consecutive non-improving evals before halving the step


@dataclass
class SearchResult:
    param_name: str
    best_value: float
    best_score: float
    history: pd.DataFrame          # columns: eval, restart, value, score, step
    n_evals_used: int
    n_restarts: int
    reached_target: bool
    elapsed_sec: float


def precision_search_1d(
    spec: ParamSpec,
    eval_fn: Callable[[float], float],
    max_evals: int = 1000,
    n_restarts: int = 5,
    target_score: Optional[float] = 0.90,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> SearchResult:
    """
    "Precision Learning" search for a single scalar parameter.

    Method: Hooke-Jeeves-style direct pattern search, run from several
    random initial points (random restarts), with a fixed evaluation
    budget shared across restarts, and early stopping.

    Why this method: the brief specifically asks for a deliberate,
    interpretable, "controlled ±0.1 step" search rather than a generic
    gradient-based optimizer (gradients through kappa ARE available per
    the Foundation document, but this Stage-1 experiment deliberately
    uses a simpler, fully transparent search so every evaluated point and
    every accept/reject decision is inspectable). This is a well known,
    classical zeroth-order method (Hooke & Jeeves, 1961), not an ad hoc
    loop.

    How it works, precisely:
      1. `max_evals` total function evaluations are split evenly across
         `n_restarts` random starting points in [spec.lower, spec.upper].
      2. From the current point x with score f(x), try x+step and x-step
         (clipped to bounds). If either improves on f(x), move there
         immediately (first-improvement, not best-improvement -- keeps the
         search fast and matches "controlled changes" rather than an
         exhaustive local scan).
      3. If neither neighbour improves, that counts as one "stall". After
         `spec.stall_patience` consecutive stalls, the step is halved
         (down to `spec.min_step`); this is what turns the initial
         "learning-rate style step of 0.1" into a genuine convergence
         criterion rather than a fixed grid.
      4. A restart ends when its step falls below `spec.min_step`, or the
         shared evaluation budget is exhausted.
      5. Early stopping of the WHOLE parameter's search: if the running
         best score has not improved for `4 * spec.stall_patience`
         evaluations across ALL restarts combined, the search stops even
         if budget remains -- "no meaningful improvement for a patience
         window" as specified.
      6. If `target_score` is reached at any point, the search stops
         immediately and reports success.

    `eval_fn(value) -> score` where higher score is better (this module
    always uses Spearman correlation with the severity labels as the
    score, so higher = better).
    """
    if rng is None:
        rng = np.random.default_rng()

    t_start = time.time()
    rows = []
    evals_used = 0
    budget_per_restart = max(1, max_evals // n_restarts)

    global_best_value = None
    global_best_score = -np.inf
    global_no_improve_streak = 0
    global_patience = 4 * spec.stall_patience
    reached_target = False

    for restart_id in range(n_restarts):
        if evals_used >= max_evals or reached_target:
            break

        x = float(rng.uniform(spec.lower, spec.upper))
        step = spec.init_step
        fx = eval_fn(x)
        evals_used += 1
        rows.append(dict(eval=evals_used, restart=restart_id, value=x, score=fx, step=step))
        if fx > global_best_score:
            global_best_score, global_best_value = fx, x
            global_no_improve_streak = 0
        else:
            global_no_improve_streak += 1

        if verbose:
            print(f"    [{spec.name}] restart {restart_id}: start x={x:.4f} score={fx:.4f}")

        stall_count = 0
        restart_evals = 1

        while (evals_used < max_evals and restart_evals < budget_per_restart
               and step >= spec.min_step and not reached_target
               and global_no_improve_streak < global_patience):

            improved_this_round = False
            for cand in (x + step, x - step):
                if evals_used >= max_evals or restart_evals >= budget_per_restart:
                    break
                cand = float(np.clip(cand, spec.lower, spec.upper))
                f_cand = eval_fn(cand)
                evals_used += 1
                restart_evals += 1
                rows.append(dict(eval=evals_used, restart=restart_id,
                                  value=cand, score=f_cand, step=step))

                if f_cand > fx + 1e-9:
                    x, fx = cand, f_cand
                    improved_this_round = True

                if f_cand > global_best_score:
                    global_best_score, global_best_value = f_cand, cand
                    global_no_improve_streak = 0
                else:
                    global_no_improve_streak += 1

                if target_score is not None and global_best_score >= target_score:
                    reached_target = True
                    break

                if improved_this_round:
                    break  # first-improvement: accept and re-center immediately

            if improved_this_round:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= spec.stall_patience:
                    step = step / 2.0
                    stall_count = 0

        if verbose:
            print(f"    [{spec.name}] restart {restart_id} done: "
                  f"local best score={fx:.4f} at x={x:.4f} "
                  f"(evals so far: {evals_used})")

    history = pd.DataFrame(rows)
    result = SearchResult(
        param_name=spec.name,
        best_value=global_best_value,
        best_score=global_best_score,
        history=history,
        n_evals_used=evals_used,
        n_restarts=n_restarts,
        reached_target=reached_target,
        elapsed_sec=time.time() - t_start,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Convenience: fresh case300 net, run once, ready for extraction
# ---------------------------------------------------------------------------

def load_case300_solved() -> "pp.pandapowerNet":
    """Load case300 and run one base-case power flow so net._ppc is populated."""
    net = pn.case300()
    pp.runpp(net, init="auto", enforce_q_lims=False, numba=False)
    if not net["converged"]:
        raise RuntimeError("Base-case power flow for case300 did not converge.")
    return net
