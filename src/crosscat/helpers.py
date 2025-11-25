"""Helper wrappers that mimic a small subset of the legacy LocalEngine API."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from genjax import ChoiceMapBuilder as C  # type: ignore

from .constants import ALPHA_CLUSTER, ALPHA_VIEW
from .gibbs import GibbsConfig, run_gibbs_mcmc
from .inference import infer_mixed_sbp_multiview_table
from .model import mixed_sbp_multiview_table

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


def run_gibbs_iterations(
    trace,
    *,
    key: Optional[jax.Array] = None,
    num_iters: int = 1,
    alpha_view: float = ALPHA_VIEW,
    alpha_cluster: float = ALPHA_CLUSTER,
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
    """Sample queried cells conditioned on the remaining observed entries."""
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

    query_cont, query_cat = _split_queries(
        queries, n_rows, n_cont_cols, n_cat_cols
    )
    cond_cont, cond_cat = _split_conditions(
        conditions or [], n_rows, n_cont_cols, n_cat_cols
    )
    _ensure_disjoint(query_cont, query_cat, cond_cont, cond_cat)

    cm = _build_constraints(trace, query_cont, query_cat, cond_cont, cond_cat)
    args = trace.get_args()

    samples: List[List[float]] = []
    curr_key = key
    for _ in range(num_samples):
        curr_key, subkey = jax.random.split(curr_key)
        sample_trace, _ = mixed_sbp_multiview_table.importance(subkey, cm, args)
        samples.append(_extract_query_values(sample_trace, queries, n_cont_cols))

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


def _split_queries(
    entries: Sequence[Query],
    n_rows: int,
    n_cont_cols: int,
    n_cat_cols: int,
):
    total_cols = n_cont_cols + n_cat_cols
    cont: set[Tuple[int, int]] = set()
    cat: set[Tuple[int, int]] = set()
    for row, col in entries:
        if row < 0 or row >= n_rows:
            raise ValueError(f"Row index {row} outside [0, {n_rows}).")
        if col < 0 or col >= total_cols:
            raise ValueError(f"Column index {col} outside [0, {total_cols}).")
        if col < n_cont_cols:
            cont.add((row, col))
        else:
            cat.add((row, col - n_cont_cols))
    return cont, cat


def _split_conditions(
    entries: Sequence[Condition],
    n_rows: int,
    n_cont_cols: int,
    n_cat_cols: int,
):
    total_cols = n_cont_cols + n_cat_cols
    cont: Dict[Tuple[int, int], float] = {}
    cat: Dict[Tuple[int, int], int] = {}
    for row, col, value in entries:
        if row < 0 or row >= n_rows:
            raise ValueError(f"Row index {row} outside [0, {n_rows}).")
        if col < 0 or col >= total_cols:
            raise ValueError(f"Column index {col} outside [0, {total_cols}).")
        if col < n_cont_cols:
            key = (row, col)
            if key in cont:
                raise ValueError(f"Duplicate condition for cell {key}.")
            cont[key] = float(value)
        else:
            key = (row, col - n_cont_cols)
            if key in cat:
                raise ValueError(f"Duplicate condition for cell {key}.")
            cat[key] = int(value)
    return cont, cat


def _ensure_disjoint(
    query_cont: Iterable[Tuple[int, int]],
    query_cat: Iterable[Tuple[int, int]],
    cond_cont: Dict[Tuple[int, int], float],
    cond_cat: Dict[Tuple[int, int], int],
):
    for idx in query_cont:
        if idx in cond_cont:
            raise ValueError(f"Query cell {idx} also present in conditions.")
    for idx in query_cat:
        if idx in cond_cat:
            raise ValueError(f"Query cell {idx} also present in conditions.")


def _build_constraints(
    trace,
    query_cont: Iterable[Tuple[int, int]],
    query_cat: Iterable[Tuple[int, int]],
    cond_cont: Dict[Tuple[int, int], float],
    cond_cat: Dict[Tuple[int, int], int],
):
    choices = trace.get_choices()
    retval = trace.get_retval()
    cm = C.n()

    rows_cont = np.asarray(choices["rows_cont"])
    rows_cat = np.asarray(retval["cat"])

    for (row, col), value in cond_cont.items():
        cm = cm | C["rows_cont", row, col].set(value)
    for (row, col), value in cond_cat.items():
        cm = cm | C["rows_cat", row, col].set(value)

    cont_mask = np.ones_like(rows_cont, dtype=bool)
    for idx in query_cont:
        cont_mask[idx] = False
    for idx in cond_cont:
        cont_mask[idx] = False
    for row, col in np.argwhere(cont_mask):
        cm = cm | C["rows_cont", int(row), int(col)].set(float(rows_cont[row, col]))

    cat_mask = np.ones_like(rows_cat, dtype=bool)
    for idx in query_cat:
        cat_mask[idx] = False
    for idx in cond_cat:
        cat_mask[idx] = False
    for row, col in np.argwhere(cat_mask):
        cm = cm | C["rows_cat", int(row), int(col)].set(int(rows_cat[row, col]))

    return cm


def _extract_query_values(trace, queries: Sequence[Query], n_cont_cols: int) -> List[float]:
    """Collect scalar samples for the flattened column index queries."""
    choices = trace.get_choices()
    retval = trace.get_retval()
    rows_cont = choices["rows_cont"]
    rows_cat = retval["cat"]

    values: List[float] = []
    for row, col in queries:
        if col < n_cont_cols:
            values.append(float(rows_cont[row, col]))
        else:
            cat_idx = col - n_cont_cols
            values.append(float(rows_cat[row, cat_idx]))
    return values
