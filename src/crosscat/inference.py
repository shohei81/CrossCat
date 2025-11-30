"""Posterior inference helpers for mixed SBP multi-view CrossCat."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from genjax import ChoiceMapBuilder as C  # type: ignore

from . import constants as const
from .model import _default_args, mixed_sbp_multiview_table


def infer_mixed_sbp_multiview_table(
    observed_cont: jnp.ndarray,
    observed_cat: jnp.ndarray,
    *,
    key: Optional[jax.Array] = None,
):
    """Infer a posterior trace given observed continuous and categorical tables."""
    n_rows, n_cont_cols = observed_cont.shape
    n_rows_cat, n_cat_cols = observed_cat.shape

    assert n_rows == const.NUM_ROWS
    assert n_rows_cat == const.NUM_ROWS
    assert n_cont_cols == const.NUM_CONT_COLS
    assert n_cat_cols == const.NUM_CAT_COLS

    if key is None:
        key = jax.random.key(1)
    constraints = C["rows_cont"].set(observed_cont)
    constraints = constraints | C["rows_cat"].set(observed_cat)

    trace, _ = mixed_sbp_multiview_table.importance(
        key, constraints, _default_args()
    )
    return trace


def _simulate_posterior_trace(key: jax.Array):
    """Simulate prior table and construct a posterior trace given its observations."""
    key_prior, key_post = jax.random.split(key)
    prior_trace = mixed_sbp_multiview_table.simulate(key_prior, _default_args())
    prior_retval = prior_trace.get_retval()
    observed_cont = prior_retval["cont"]
    observed_cat = prior_retval["cat"]
    posterior_trace = infer_mixed_sbp_multiview_table(
        observed_cont, observed_cat, key=key_post
    )
    return posterior_trace, observed_cont, observed_cat


def _extract_hyperparams_from_trace(trace):
    """Return current column-wise hyperparameters stored in the trace choices."""
    args = trace.get_args()
    n_cont_cols = args[1].unwrap()
    n_cat_cols = args[2].unwrap()
    choices = trace.get_choices()

    if n_cont_cols > 0:
        mu0_vec = choices["hyper_cont", "mu0"]
        kappa0_vec = choices["hyper_cont", "kappa0"]
        ng_alpha0_vec = choices["hyper_cont", "alpha0"]
        ng_beta0_vec = choices["hyper_cont", "beta0"]
    else:
        zero = jnp.zeros((0,), dtype=jnp.float32)
        mu0_vec = zero
        kappa0_vec = zero
        ng_alpha0_vec = zero
        ng_beta0_vec = zero

    if n_cat_cols > 0:
        alpha_cat_vec = choices["hyper_cat", "alpha_cat"]
    else:
        alpha_cat_vec = jnp.zeros((0,), dtype=jnp.float32)

    return mu0_vec, kappa0_vec, ng_alpha0_vec, ng_beta0_vec, alpha_cat_vec
