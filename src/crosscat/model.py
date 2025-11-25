"""Generative model pieces for the mixed-type SBP multi-view CrossCat example."""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit

from genjax import beta, categorical, dirichlet, gen, normal, gamma  # type: ignore
from genjax._src.core.pytree import Const

from .constants import (
    ALPHA_CAT,
    ALPHA_CAT_A,
    ALPHA_CAT_B,
    ALPHA_CLUSTER,
    ALPHA_VIEW,
    MU0_PRIOR_MEAN,
    MU0_PRIOR_VAR,
    KAPPA0_A,
    KAPPA0_B,
    NG_ALPHA0_A,
    NG_ALPHA0_B,
    NG_BETA0_A,
    NG_BETA0_B,
    NUM_CATEGORIES,
    NUM_CAT_COLS,
    NUM_CLUSTERS,
    NUM_CONT_COLS,
    NUM_ROWS,
    NUM_VIEWS,
)


def _default_args():
    """Default argument tuple for mixed_sbp_multiview_table."""
    return (
        Const(NUM_ROWS),
        Const(NUM_CONT_COLS),
        Const(NUM_CAT_COLS),
        Const(NUM_VIEWS),
        Const(NUM_CLUSTERS),
        ALPHA_VIEW,
        ALPHA_CLUSTER,
        ALPHA_CAT,
    )


@gen
def sbp_weights(alpha: float, n_components: Const[int]):  # type: ignore
    k = n_components.unwrap()
    if k <= 1:
        return jnp.ones((1,))
    conc0 = jnp.ones(k - 1)
    conc1 = alpha * jnp.ones(k - 1)
    v = beta(conc0, conc1) @ ("v",)
    v_ext = jnp.concatenate([v, jnp.array([1.0])], axis=0)
    one_minus = 1.0 - v_ext
    prefix = jnp.concatenate(
        [jnp.array([1.0]), jnp.cumprod(one_minus[:-1], axis=0)], axis=0
    )
    return v_ext * prefix


@gen
def sbp_column_views(  # type: ignore
    n_cols: Const[int], n_views: Const[int], alpha: float
):
    weights = sbp_weights(alpha, n_views) @ ("view_weights",)
    view_idx = categorical(logits=jnp.log(weights), sample_shape=n_cols) @ ("view_idx",)
    return view_idx


@gen
def cont_column_hyperparams():  # type: ignore
    mu0 = normal(MU0_PRIOR_MEAN, MU0_PRIOR_VAR) @ ("mu0",)
    kappa0 = gamma(KAPPA0_A, KAPPA0_B) @ ("kappa0",)
    alpha0 = gamma(NG_ALPHA0_A, NG_ALPHA0_B) @ ("alpha0",)
    beta0 = gamma(NG_BETA0_A, NG_BETA0_B) @ ("beta0",)
    return {"mu0": mu0, "kappa0": kappa0, "alpha0": alpha0, "beta0": beta0}


@gen
def cat_column_hyperparams():  # type: ignore
    alpha_cat = gamma(ALPHA_CAT_A, ALPHA_CAT_B) @ ("alpha_cat",)
    return {"alpha_cat": alpha_cat}


@gen
def cluster_cont_params(  # type: ignore
    prior_mean: jnp.ndarray,
    kappa0: jnp.ndarray,
    alpha0: jnp.ndarray,
    beta0: jnp.ndarray,
    n_cont_cols: Const[int],
):
    k = n_cont_cols.unwrap()
    if k == 0:
        zeros = jnp.zeros((0,), dtype=jnp.float32)
        return {"mean": zeros, "tau": zeros}

    tau = gamma(alpha0, beta0) @ ("tau",)

    # Mean | tau ~ Normal(prior_mean, (kappa0 * tau)^(-1))
    var_means = 1.0 / jnp.clip(kappa0 * tau, 1e-20)
    means = normal(prior_mean, var_means) @ ("mean",)
    return {"mean": means, "tau": tau}


@gen
def cluster_cat_params(  # type: ignore
    alpha_vec: jnp.ndarray,
    n_cat_cols: Const[int],
    n_categories: Const[int],
):
    k = n_cat_cols.unwrap()
    if k == 0:
        return {"probs": jnp.zeros((0, n_categories.unwrap()))}

    n_cats = n_categories.unwrap()
    alphas = jnp.broadcast_to(alpha_vec[:, None], (k, n_cats))
    probs = dirichlet(alphas) @ ("probs",)
    return {"probs": probs}


@gen
def mixed_sbp_multiview_table(  # type: ignore
    n_rows: Const[int],
    n_cont_cols: Const[int],
    n_cat_cols: Const[int],
    n_views: Const[int],
    n_clusters: Const[int],
    alpha_view: float,
    alpha_cluster: float,
    alpha_cat: float,
) -> dict:
    """Mixed-type SBP multi-view CrossCat generative model."""
    num_rows = n_rows.unwrap()
    num_cont_cols = n_cont_cols.unwrap()
    num_cat_cols = n_cat_cols.unwrap()
    num_views = n_views.unwrap()
    num_clusters = n_clusters.unwrap()

    total_cols = num_cont_cols + num_cat_cols

    if num_cont_cols > 0:
        hyper_cont = (
            cont_column_hyperparams.repeat(n=num_cont_cols)() @ ("hyper_cont",)
        )
        mu0_vec = hyper_cont["mu0"]
        kappa0_vec = hyper_cont["kappa0"]
        ng_alpha0_vec = hyper_cont["alpha0"]
        ng_beta0_vec = hyper_cont["beta0"]
    else:
        zeros = jnp.zeros((0,), dtype=jnp.float32)
        mu0_vec = zeros
        kappa0_vec = zeros
        ng_alpha0_vec = zeros
        ng_beta0_vec = zeros

    if num_cat_cols > 0:
        hyper_cat = (
            cat_column_hyperparams.repeat(n=num_cat_cols)() @ ("hyper_cat",)
        )
        alpha_cat_vec = hyper_cat["alpha_cat"]
    else:
        alpha_cat_vec = jnp.zeros((0,), dtype=jnp.float32)

    view_idx = sbp_column_views(Const(total_cols), n_views, alpha_view) @ ("views",)

    cluster_weights = sbp_weights.repeat(n=num_views)(alpha_cluster, n_clusters) @ (
        "cluster_weights",
    )

    clusters_cont_params_flat = (
        cluster_cont_params.repeat(n=num_views * num_clusters)(
            mu0_vec,
            kappa0_vec,
            ng_alpha0_vec,
            ng_beta0_vec,
            n_cont_cols,
        )
        @ ("clusters_cont",)
    )
    clusters_cont = clusters_cont_params_flat["mean"].reshape(
        (num_views, num_clusters, num_cont_cols)
    )
    clusters_prec = clusters_cont_params_flat["tau"].reshape(
        (num_views, num_clusters, num_cont_cols)
    )

    cat_params_flat = (
        cluster_cat_params.repeat(n=num_views * num_clusters)(
            alpha_cat_vec, n_cat_cols, Const(NUM_CATEGORIES)
        )
        @ ("clusters_cat",)
    )
    cat_params = cat_params_flat["probs"].reshape(
        (num_views, num_clusters, num_cat_cols, NUM_CATEGORIES)
    )

    logits_clusters = jnp.log(cluster_weights + 1e-20)
    logits_clusters = logits_clusters[:, None, :].repeat(num_rows, axis=1)
    row_clusters = categorical(logits=logits_clusters) @ ("row_clusters", "idx")

    if num_cont_cols > 0:
        view_idx_cont = view_idx[:num_cont_cols].astype(jnp.int32)
        row_clusters_T = row_clusters.T
        views_cont_2d = jnp.broadcast_to(
            view_idx_cont[None, :], (num_rows, num_cont_cols)
        )
        z_cont = jnp.take_along_axis(row_clusters_T, views_cont_2d, axis=1)

        col_indices = jnp.broadcast_to(
            jnp.arange(num_cont_cols, dtype=jnp.int32)[None, :],
            (num_rows, num_cont_cols),
        )

        mean_cont_table = clusters_cont[views_cont_2d, z_cont, col_indices]
        prec_cont_table = clusters_prec[views_cont_2d, z_cont, col_indices]
    else:
        mean_cont_table = jnp.zeros((num_rows, 0), dtype=jnp.float32)
        prec_cont_table = jnp.zeros((num_rows, 0), dtype=jnp.float32)

    obs_var_table = 1.0 / jnp.clip(prec_cont_table, 1e-20)
    table_cont = normal(mean_cont_table, obs_var_table) @ ("rows_cont",)

    if num_cat_cols == 0:
        table_cat = jnp.zeros((num_rows, 0), dtype=jnp.int32)
    else:
        col_idx_cat = num_cont_cols + jnp.arange(num_cat_cols)
        views_cat = view_idx[col_idx_cat].astype(jnp.int32)

        row_idx = jnp.arange(num_rows, dtype=jnp.int32)
        z_cat = row_clusters[views_cat[:, None], row_idx[None, :]]
        z_cat = z_cat.T

        views_cat_2d = jnp.broadcast_to(views_cat, (num_rows, num_cat_cols))
        col_indices = jnp.broadcast_to(
            jnp.arange(num_cat_cols, dtype=jnp.int32), (num_rows, num_cat_cols)
        )

        probs_cat = cat_params[views_cat_2d, z_cat, col_indices, :]
        logits_cat = jnp.log(probs_cat + 1e-20)

        table_cat = categorical(logits=logits_cat) @ ("rows_cat",)

    return {"cont": table_cont, "cat": table_cat}


jitted_update = jit(mixed_sbp_multiview_table.update)
