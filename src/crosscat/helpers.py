"""Helper wrappers that mimic a small subset of the original LocalEngine API."""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import constants as const
from .gibbs import GibbsConfig, _sbp_weights_from_v, run_gibbs_mcmc
from .inference import infer_mixed_sbp_multiview_table

Query = Tuple[int, int]
Condition = Tuple[int, int, float]


def posterior_from_observations(
    observed_cont: jnp.ndarray,
    observed_cat: jnp.ndarray,
    *,
    key: Optional[jax.Array] = None,
):
    """Fit a posterior trace from observed tables."""
    if key is None:
        key = jax.random.key(0)
    return infer_mixed_sbp_multiview_table(observed_cont, observed_cat, key=key)


@partial(jax.jit, static_argnames=["num_iters"])
def run_gibbs_iterations(
    trace,
    *,
    key: Optional[jax.Array] = None,
    num_iters: int = 1,
    alpha_view: float = const.ALPHA_VIEW,
    alpha_cluster: float = const.ALPHA_CLUSTER,
):
    """JIT-friendly helper that mirrors LocalEngine.analyze."""
    if num_iters <= 0:
        raise ValueError("num_iters must be > 0.")
    if key is None:
        key = jax.random.key(0)
    state = run_gibbs_mcmc(
        key,
        trace,
        alpha_view,
        alpha_cluster,
        cfg=GibbsConfig(num_iters=num_iters),
    )
    return state.key, state


def predictive_samples(
    trace,
    queries: Sequence[Query],
    *,
    key: Optional[jax.Array] = None,
    num_samples: int = 1,
    conditions: Optional[Sequence[Condition]] = None,
):
    """Sample queried cells conditioned on optional row-wise evidence."""
    if len(queries) == 0:
        raise ValueError("queries must contain at least one entry.")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")
    if key is None:
        key = jax.random.key(0)

    args = trace.get_args()
    n_rows = args[0].unwrap()
    n_cont_cols = args[1].unwrap()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()
    total_cols = n_cont_cols + n_cat_cols

    choices = trace.get_choices()
    retval = trace.get_retval()

    view_idx = jnp.asarray(choices["views", "view_idx"], dtype=jnp.int32)
    row_clusters = jnp.asarray(choices["row_clusters", "idx"], dtype=jnp.int32)
    sticks = jnp.asarray(choices["cluster_weights", "v"])
    cluster_weights = jax.vmap(_sbp_weights_from_v)(sticks)
    if n_cont_cols > 0:
        clusters_cont = jnp.asarray(choices["clusters_cont", "mean"]).reshape(
            (n_views, n_clusters, n_cont_cols)
        )
        clusters_prec = jnp.asarray(choices["clusters_cont", "tau"]).reshape(
            (n_views, n_clusters, n_cont_cols)
        )
    else:
        clusters_cont = jnp.zeros((n_views, n_clusters, 0))
        clusters_prec = jnp.zeros((n_views, n_clusters, 0))
    if n_cat_cols > 0:
        clusters_cat = jnp.asarray(choices["clusters_cat", "probs"]).reshape(
            (n_views, n_clusters, n_cat_cols, const.NUM_CATEGORIES)
        )
    else:
        clusters_cat = None

    query_map = _group_queries_by_row(queries, total_cols)
    cond_map = _group_conditions_by_row(conditions or [], total_cols)
    _ensure_disjoint_cells(query_map, cond_map)

    samples: List[List[float]] = []
    curr_key = key
    for _ in range(num_samples):
        sample_vals = [0.0] * len(queries)
        for row_id, row_queries in query_map.items():
            clusters, curr_key = _clusters_for_row(
                row_id,
                n_rows,
                row_clusters,
                cluster_weights,
                view_idx,
                n_cont_cols,
                cond_map.get(row_id, []),
                clusters_cont,
                clusters_prec,
                clusters_cat,
                curr_key,
            )
            curr_key, row_values = _sample_row_entries(
                row_id,
                row_queries,
                clusters,
                view_idx,
                n_cont_cols,
                clusters_cont,
                clusters_prec,
                clusters_cat,
                curr_key,
            )
            for (pos, _), value in zip(row_queries, row_values):
                sample_vals[pos] = value
        samples.append(sample_vals)

    return curr_key, jnp.asarray(samples)


def impute(
    trace,
    queries: Sequence[Query],
    *,
    key: Optional[jax.Array] = None,
    num_samples: int = 100,
    conditions: Optional[Sequence[Condition]] = None,
):
    """Return posterior mean imputations for the provided query cells."""
    key, draws = predictive_samples(
        trace,
        queries,
        key=key,
        num_samples=num_samples,
        conditions=conditions,
    )
    estimates = jnp.mean(draws, axis=0)
    return key, estimates


def _group_queries_by_row(
    queries: Sequence[Query], total_cols: int
) -> Dict[int, List[Tuple[int, int]]]:
    row_map: Dict[int, List[Tuple[int, int]]] = {}
    for idx, (row, col) in enumerate(queries):
        if row < 0:
            raise ValueError(f"Row index {row} must be non-negative.")
        if col < 0 or col >= total_cols:
            raise ValueError(f"Column index {col} outside [0, {total_cols}).")
        row_map.setdefault(row, []).append((idx, col))
    return row_map


def _group_conditions_by_row(
    conditions: Sequence[Condition], total_cols: int
) -> Dict[int, List[Tuple[int, float]]]:
    cond_map: Dict[int, List[Tuple[int, float]]] = {}
    for row, col, value in conditions:
        if row < 0:
            raise ValueError(f"Row index {row} must be non-negative.")
        if col < 0 or col >= total_cols:
            raise ValueError(f"Column index {col} outside [0, {total_cols}).")
        cond_map.setdefault(row, []).append((col, value))
    return cond_map


def _ensure_disjoint_cells(
    query_map: Dict[int, List[Tuple[int, int]]],
    cond_map: Dict[int, List[Tuple[int, float]]],
):
    cond_set = {
        (row, col) for row, entries in cond_map.items() for col, _ in entries
    }
    for row, entries in query_map.items():
        for _, col in entries:
            if (row, col) in cond_set:
                raise ValueError(
                    f"Query cell ({row}, {col}) also present in conditions."
                )


def _split_conditions_by_view(
    row_conditions: List[Tuple[int, float]],
    n_cont_cols: int,
    view_idx: jnp.ndarray,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, int]]]]:
    cont_by_view: Dict[int, List[Tuple[int, float]]] = {}
    cat_by_view: Dict[int, List[Tuple[int, int]]] = {}
    for col, value in row_conditions:
        v = int(view_idx[col])
        if col < n_cont_cols:
            cont_by_view.setdefault(v, []).append((col, float(value)))
        else:
            cat_col = col - n_cont_cols
            cat_by_view.setdefault(v, []).append((cat_col, int(value)))
    return cont_by_view, cat_by_view


def _log_normal_pdf(value: float, mean: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """Return log N(value | mean, tau^-1) for each cluster."""
    tau_safe = jnp.clip(tau, 1e-8)
    return 0.5 * (
        jnp.log(tau_safe) - jnp.log(2.0 * jnp.pi) - tau_safe * (value - mean) ** 2
    )


def _clusters_for_row(
    row_id: int,
    n_rows: int,
    row_clusters: jnp.ndarray,
    cluster_weights: jnp.ndarray,
    view_idx: jnp.ndarray,
    n_cont_cols: int,
    row_conditions: List[Tuple[int, float]],
    clusters_cont: jnp.ndarray,
    clusters_prec: jnp.ndarray,
    clusters_cat: Optional[jnp.ndarray],
    key: jax.Array,
) -> Tuple[jnp.ndarray, jax.Array]:
    if row_id < n_rows:
        return jnp.asarray(row_clusters[:, row_id], dtype=jnp.int32), key

    cond_cont, cond_cat = _split_conditions_by_view(row_conditions, n_cont_cols, view_idx)

    assignments = []
    key_local = key
    n_views = cluster_weights.shape[0]
    for v in range(n_views):
        logits = jnp.log(jnp.clip(cluster_weights[v], 1e-20, jnp.inf))
        for col_idx, val in cond_cont.get(v, []):
            mu = clusters_cont[v, :, col_idx]
            tau = clusters_prec[v, :, col_idx]
            logits = logits + _log_normal_pdf(val, mu, tau)
        for cat_idx, cat_val in cond_cat.get(v, []):
            if clusters_cat is None:
                continue
            probs = clusters_cat[v, :, cat_idx, :]
            logits = logits + jnp.log(
                jnp.clip(probs[:, cat_val], 1e-20, jnp.inf)
            )
        key_local, subkey = jax.random.split(key_local)
        cluster = jax.random.categorical(subkey, logits)
        assignments.append(cluster)

    return jnp.asarray(assignments, dtype=jnp.int32), key_local


def _sample_row_entries(
    row_id: int,
    row_queries: List[Tuple[int, int]],
    view_clusters: jnp.ndarray,
    view_idx: jnp.ndarray,
    n_cont_cols: int,
    clusters_cont: jnp.ndarray,
    clusters_prec: jnp.ndarray,
    clusters_cat: Optional[jnp.ndarray],
    key: jax.Array,
) -> Tuple[jax.Array, List[float]]:
    values: List[float] = []
    key_local = key
    for _, col in row_queries:
        view = int(view_idx[col])
        cluster = int(view_clusters[view])
        if col < n_cont_cols:
            mu = clusters_cont[view, cluster, col]
            tau = clusters_prec[view, cluster, col]
            tau_safe = jnp.clip(tau, 1e-8)
            std = jnp.sqrt(1.0 / tau_safe)
            key_local, subkey = jax.random.split(key_local)
            draw = jax.random.normal(subkey) * std + mu
            values.append(float(draw))
        else:
            if clusters_cat is None:
                raise ValueError("No categorical parameters available.")
            cat_idx = col - n_cont_cols
            probs = clusters_cat[view, cluster, cat_idx, :]
            key_local, subkey = jax.random.split(key_local)
            cat_draw = jax.random.categorical(
                subkey, jnp.log(jnp.clip(probs, 1e-20, 1.0))
            )
            values.append(float(cat_draw))
    return key_local, values
