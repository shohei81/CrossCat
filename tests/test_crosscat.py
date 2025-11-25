"""Tests and helper kernels for a mixed-type SBP multi-view CrossCat model."""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gammaln

import genjax
from genjax import ChoiceMapBuilder as C  # type: ignore
from genjax import beta, categorical, dirichlet, gen, normal, gamma  # type: ignore
from genjax._src.core.pytree import Const


# Mixed-type SBP multi-view CrossCat (generative model, shape tests, and update kernels)
NUM_ROWS = 10
NUM_CONT_COLS = 1
NUM_CAT_COLS = 1
NUM_VIEWS = 2
NUM_CLUSTERS = 3
ALPHA_VIEW = 1.0
ALPHA_CLUSTER = 1.0
PRIOR_MEAN = 0.0
NUM_CATEGORIES = 3
ALPHA_CAT = 1.0

# Hyperpriors for SBP concentration parameters (Gamma(a, b))
ALPHA_VIEW_A = 1.0
ALPHA_VIEW_B = 1.0
ALPHA_CLUSTER_A = 1.0
ALPHA_CLUSTER_B = 1.0
ALPHA_CAT_A = 1.0
ALPHA_CAT_B = 1.0

# Hyperparameters for Normal-Gamma prior on continuous cluster parameters
NG_KAPPA0 = 1.0
NG_ALPHA0 = 1.0
NG_BETA0 = 1.0
NG_ALPHA0_A = 1.0
NG_ALPHA0_B = 1.0
NG_BETA0_A = 1.0
NG_BETA0_B = 1.0

# Hyperpriors for column-wise Normal-Gamma location hyperparameters
MU0_PRIOR_MEAN = 0.0
MU0_PRIOR_VAR = 10.0
KAPPA0_A = 1.0
KAPPA0_B = 1.0


def _normal_gamma_marginal_loglik_from_stats(
    n: jnp.ndarray,
    sum_y: jnp.ndarray,
    sumsq_y: jnp.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> jnp.ndarray:
    """Normal-Gamma marginal log-likelihood log p(y | hyper) from sufficient stats.

    Works element-wise over `n`, `sum_y`, `sumsq_y`; for n == 0, returns 0.0.
    """
    n = n.astype(jnp.float32)
    n_pos = n > 0.0
    n_safe = jnp.where(n_pos, n, 1.0)

    y_bar = sum_y / n_safe
    s2 = sumsq_y - (sum_y**2) / n_safe

    kappa_n = kappa0 + n
    alpha_n = alpha0 + 0.5 * n
    beta_n = (
        beta0
        + 0.5 * s2
        + (kappa0 * n * (y_bar - mu0) ** 2) / (2.0 * kappa_n)
    )

    loglik = (
        0.5 * (jnp.log(kappa0) - jnp.log(kappa_n))
        + gammaln(alpha_n)
        - gammaln(alpha0)
        + alpha0 * jnp.log(beta0)
        - alpha_n * jnp.log(beta_n)
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )

    return jnp.where(n_pos, loglik, 0.0)


def _normal_gamma_predictive_loglik_from_stats(
    n: jnp.ndarray,
    sum_y: jnp.ndarray,
    sumsq_y: jnp.ndarray,
    y_new: jnp.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> jnp.ndarray:
    """Predictive log-likelihood log p(y_new | y_old, hyper) from sufficient stats."""
    log_old = _normal_gamma_marginal_loglik_from_stats(
        n,
        sum_y,
        sumsq_y,
        mu0,
        kappa0,
        alpha0,
        beta0,
    )

    n_new = n + 1.0
    sum_y_new = sum_y + y_new
    sumsq_y_new = sumsq_y + y_new**2

    log_new = _normal_gamma_marginal_loglik_from_stats(
        n_new,
        sum_y_new,
        sumsq_y_new,
        mu0,
        kappa0,
        alpha0,
        beta0,
    )
    return log_new - log_old


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
    view_idx = categorical(jnp.log(weights), sample_shape=n_cols) @ ("view_idx",)
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
    """Mixed-type SBP multi-view CrossCat generative model.

    Returns a dict with:
      - "cont": continuous table of shape (num_rows, num_cont_cols)
      - "cat": categorical table of shape (num_rows, num_cat_cols)
    """
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
    )  # (num_views, num_clusters)

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

    # Row clusters per view, stored as a single array-valued choice
    logits_clusters = jnp.log(cluster_weights + 1e-20)  # (num_views, num_clusters)
    logits_clusters = logits_clusters[:, None, :].repeat(num_rows, axis=1)
    row_clusters = categorical(logits_clusters) @ ("row_clusters", "idx")

    # Continuous part
    if num_cont_cols > 0:
        view_idx_cont = view_idx[:num_cont_cols].astype(jnp.int32)
        row_clusters_T = row_clusters.T  # (num_rows, num_views)
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

    # Observation variance is the inverse of precision.
    obs_var_table = 1.0 / jnp.clip(prec_cont_table, 1e-20)
    table_cont = normal(mean_cont_table, obs_var_table) @ ("rows_cont",)

    # Categorical part
    if num_cat_cols == 0:
        table_cat = jnp.zeros((num_rows, 0), dtype=jnp.int32)
    else:
        col_idx_cat = num_cont_cols + jnp.arange(num_cat_cols)
        views_cat = view_idx[col_idx_cat].astype(jnp.int32)

        row_idx = jnp.arange(num_rows, dtype=jnp.int32)
        z_cat = row_clusters[views_cat[:, None], row_idx[None, :]]
        z_cat = z_cat.T  # (num_rows, num_cat_cols)

        views_cat_2d = jnp.broadcast_to(views_cat, (num_rows, num_cat_cols))
        col_indices = jnp.broadcast_to(
            jnp.arange(num_cat_cols, dtype=jnp.int32), (num_rows, num_cat_cols)
        )

        probs_cat = cat_params[views_cat_2d, z_cat, col_indices, :]
        logits_cat = jnp.log(probs_cat + 1e-20)

        table_cat = categorical(logits_cat) @ ("rows_cat",)

    return {"cont": table_cont, "cat": table_cat}


# Jitted kernel that applies a single GenJAX model update step.
jitted_update = jit(mixed_sbp_multiview_table.update)


def test_model_table_shapes():
    rng_key = jax.random.key(0)
    trace = mixed_sbp_multiview_table.simulate(rng_key, _default_args())
    retval = trace.get_retval()
    choices = trace.get_choices()

    table_cont = retval["cont"]
    table_cat = retval["cat"]

    assert table_cont.shape == (NUM_ROWS, NUM_CONT_COLS)
    assert table_cat.shape == (NUM_ROWS, NUM_CAT_COLS)

    total_cols = NUM_CONT_COLS + NUM_CAT_COLS
    view_idx = choices["views", "view_idx"]
    assert view_idx.shape == (total_cols,)
    assert jnp.all(view_idx >= 0)
    assert jnp.all(view_idx < NUM_VIEWS)

    row_clusters = choices["row_clusters", "idx"]
    assert row_clusters.shape == (NUM_VIEWS, NUM_ROWS)
    assert jnp.all(row_clusters >= 0)
    assert jnp.all(row_clusters < NUM_CLUSTERS)

    assert jnp.all(table_cat >= 0)
    assert jnp.all(table_cat < NUM_CATEGORIES)


def _sbp_weights_from_v(v: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct finite SBP weights from Beta sticks v (shape (k-1,))."""
    k_minus_1 = v.shape[0]
    if k_minus_1 == 0:
        return jnp.ones((1,))
    v_ext = jnp.concatenate([v, jnp.array([1.0], dtype=v.dtype)], axis=0)
    one_minus = 1.0 - v_ext
    prefix = jnp.concatenate(
        [jnp.array([1.0], dtype=v.dtype), jnp.cumprod(one_minus[:-1], axis=0)], axis=0
    )
    return v_ext * prefix


def infer_mixed_sbp_multiview_table(
    observed_cont: jnp.ndarray, observed_cat: jnp.ndarray
):
    n_rows, n_cont_cols = observed_cont.shape
    n_rows_cat, n_cat_cols = observed_cat.shape

    assert n_rows == NUM_ROWS
    assert n_rows_cat == NUM_ROWS
    assert n_cont_cols == NUM_CONT_COLS
    assert n_cat_cols == NUM_CAT_COLS

    key = jax.random.key(1)
    constraints = C["rows_cont"].set(observed_cont)
    constraints = constraints | C["rows_cat"].set(observed_cat)

    trace, _ = mixed_sbp_multiview_table.importance(
        key, constraints, _default_args()
    )
    return trace


def _simulate_posterior_trace(key: jax.Array):
    """Simulate prior table and construct a posterior trace given its observations."""
    prior_trace = mixed_sbp_multiview_table.simulate(key, _default_args())
    prior_retval = prior_trace.get_retval()
    observed_cont = prior_retval["cont"]
    observed_cat = prior_retval["cat"]
    posterior_trace = infer_mixed_sbp_multiview_table(observed_cont, observed_cat)
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


# ======================
# Trace-based Gibbs updates (using jitted_update)
# ======================


def gibbs_update_row_clusters(
    key,
    trace,
    mu0_vec: jax.Array,
    kappa0_vec: jax.Array,
    ng_alpha0: jax.Array,
    ng_beta0: jax.Array,
    alpha_cat_vec: jax.Array,
):
    """Gibbs update for row cluster assignments in this model."""
    args = trace.get_args()
    n_rows = args[0].unwrap()
    n_cont_cols = args[1].unwrap()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    choices = trace.get_choices()
    retval = trace.get_retval()

    rows_cont = choices["rows_cont"]  # (n_rows, n_cont_cols)
    rows_cat = retval["cat"]  # (n_rows, n_cat_cols)

    view_idx = choices["views", "view_idx"]  # (n_cont_cols + n_cat_cols,)
    view_idx_cont = view_idx[:n_cont_cols]
    view_idx_cat = view_idx[n_cont_cols:]

    # Build collapsed categorical sufficient statistics from the current row_clusters.
    row_clusters = choices["row_clusters", "idx"]  # (n_views, n_rows)
    v_cluster = choices["cluster_weights", "v"]  # (n_views, n_clusters-1)

    # ===== Prepare per-view masks and weights =====
    views_arange = jnp.arange(n_views)[:, None]  # (n_views, 1)

    # (n_views, n_cont_cols): whether view v owns continuous column j
    mask_cont_all = (views_arange == view_idx_cont[None, :]).astype(jnp.float32)

    # (n_views, n_cat_cols): whether view v owns categorical column j
    # If n_cat_cols == 0, this becomes (n_views, 0) and contributes zero downstream.
    if n_cat_cols > 0:
        mask_cat_all = (views_arange == view_idx_cat[None, :]).astype(jnp.float32)
    else:
        mask_cat_all = jnp.zeros((n_views, 0), dtype=jnp.float32)

    # SBP weights w_v for each view (n_views, n_clusters).
    weights_all = jax.vmap(_sbp_weights_from_v)(v_cluster)

    # RNG keys for each view.
    key, subkey_views = jax.random.split(key)
    keys_v = jax.random.split(subkey_views, n_views)  # (n_views,)

    # ===== Define a kernel that updates row_clusters for a single view =====
    def _update_rows_one_view(
        key_v: jax.Array,
        weights_v: jax.Array,  # (n_clusters,)
        mask_cont_v: jax.Array,  # (n_cont_cols,)
        mask_cat_v: jax.Array,  # (n_cat_cols,)
        row_clusters_v: jax.Array,  # (n_rows,)
        alpha_cat_vec_full: jax.Array,  # (n_cat_cols,)
    ) -> jax.Array:
        """Collapsed Gibbs updates for one view's row clusters."""

        # Continuous sufficient statistics per (cluster, cont_col) under current z.
        masked_cont = rows_cont * mask_cont_v[None, :]  # (n_rows, n_cont_cols)
        one_hot = jax.nn.one_hot(
            row_clusters_v, n_clusters, dtype=jnp.float32
        )  # (n_rows, n_clusters)
        n_k = jnp.sum(one_hot, axis=0)  # (n_clusters,)
        sum_y = jnp.einsum("rn,rc->nc", one_hot, masked_cont)  # (n_clusters, n_cont)
        sumsq_y = jnp.einsum(
            "rn,rc->nc", one_hot, masked_cont**2
        )  # (n_clusters, n_cont)

        # Initial categorical counts (cluster, column, category).
        y_all = rows_cat.astype(jnp.int32)  # (n_rows, n_cat_cols)
        idx_flat = (
            row_clusters_v[:, None] * (n_cat_cols * NUM_CATEGORIES)
            + jnp.arange(n_cat_cols)[None, :] * NUM_CATEGORIES
            + y_all
        )
        counts_flat = jnp.bincount(
            idx_flat.reshape(-1),
            length=n_clusters * n_cat_cols * NUM_CATEGORIES,
            minlength=n_clusters * n_cat_cols * NUM_CATEGORIES,
        ).astype(jnp.int32)
        counts_cat = counts_flat.reshape((n_clusters, n_cat_cols, NUM_CATEGORIES))

        def one_row(carry, r_idx):
            key_local, row_clusters_curr, counts_curr, n_k_curr, sum_y_curr, sumsq_y_curr = carry

            # Continuous part: Student-t predictive log-likelihood from Normal-Gamma.
            x_r_cont = rows_cont[r_idx] * mask_cont_v  # (n_cont_cols,)
            old_k = row_clusters_curr[r_idx]

            one_hot_old = jax.nn.one_hot(
                old_k, n_clusters, dtype=jnp.float32
            )  # (n_clusters,)
            n_minus = n_k_curr - one_hot_old  # (n_clusters,)
            sum_y_minus = sum_y_curr - one_hot_old[:, None] * x_r_cont[None, :]
            sumsq_y_minus = sumsq_y_curr - one_hot_old[:, None] * (
                x_r_cont**2
            )[None, :]

            if n_cont_cols > 0:
                y_new_mat = jnp.broadcast_to(
                    x_r_cont[None, :], sum_y_minus.shape
                )  # (n_clusters, n_cont_cols)
                n_broadcast = jnp.broadcast_to(
                    n_minus[:, None], sum_y_minus.shape
                )
                log_pred_mat = _normal_gamma_predictive_loglik_from_stats(
                    n_broadcast,
                    sum_y_minus,
                    sumsq_y_minus,
                    y_new_mat,
                    mu0_vec,
                    kappa0_vec,
                    ng_alpha0,
                    ng_beta0,
                )  # (n_clusters, n_cont_cols)
                loglik_cont = jnp.sum(
                    log_pred_mat * mask_cont_v[None, :], axis=1
                )  # (n_clusters,)
            else:
                loglik_cont = jnp.zeros((n_clusters,))

            # Categorical part (collapsed Dirichlet-Multinomial).
            if n_cat_cols > 0:
                y_r = rows_cat[r_idx].astype(jnp.int32)  # (n_cat_cols,)
                old_k = row_clusters_curr[r_idx]

                # Temporarily remove the contribution of row r.
                counts_minus = counts_curr
                counts_minus = counts_minus.at[
                    old_k, jnp.arange(n_cat_cols), y_r
                ].add(-1)

                counts_k = counts_minus  # (n_clusters, n_cat_cols, NUM_CATEGORIES)
                n_sum = jnp.sum(counts_k, axis=2)  # (n_clusters, n_cat_cols)

                y_idx = y_r[None, :, None]  # (1, n_cat_cols, 1)
                counts_y = jnp.take_along_axis(counts_k, y_idx, axis=2)[
                    :, :, 0
                ]  # (n_clusters, n_cat_cols)

                alpha_c = alpha_cat_vec_full  # (n_cat_cols,)
                numer = alpha_c[None, :] + counts_y
                denom = alpha_c[None, :] * NUM_CATEGORIES + n_sum

                loglik_cat_mat = jnp.log(jnp.clip(numer, 1e-20, jnp.inf)) - jnp.log(
                    jnp.clip(denom, 1e-20, jnp.inf)
                )
                loglik_cat = jnp.sum(loglik_cat_mat * mask_cat_v[None, :], axis=1)
            else:
                loglik_cat = jnp.zeros((n_clusters,))

            # prior + likelihood
            log_prior = jnp.log(jnp.clip(weights_v, 1e-20, jnp.inf))
            logits = loglik_cont + loglik_cat + log_prior

            key_local, subkey = jax.random.split(key_local)
            new_k = jax.random.categorical(subkey, logits)

            # Add the new cluster assignment back into counts.
            counts_new = counts_minus.at[new_k, jnp.arange(n_cat_cols), y_r].add(1)
            row_clusters_curr = row_clusters_curr.at[r_idx].set(new_k)
            one_hot_new = jax.nn.one_hot(
                new_k, n_clusters, dtype=jnp.float32
            )  # (n_clusters,)
            n_k_new = n_minus + one_hot_new
            sum_y_new = sum_y_minus + one_hot_new[:, None] * x_r_cont[None, :]
            sumsq_y_new = sumsq_y_minus + one_hot_new[:, None] * (
                x_r_cont**2
            )[None, :]

            return (
                key_local,
                row_clusters_curr,
                counts_new,
                n_k_new,
                sum_y_new,
                sumsq_y_new,
            ), None

        init = (key_v, row_clusters_v, counts_cat, n_k, sum_y, sumsq_y)
        (key_v_out, row_clusters_new, _, _, _, _), _ = lax.scan(
            one_row, init, jnp.arange(n_rows)
        )
        return row_clusters_new

    # Use vmap over views; each view runs its own scan over rows.
    new_row_clusters = jax.vmap(
        _update_rows_one_view,
        in_axes=(0, 0, 0, 0, 0, None),
        out_axes=0,
    )(
        keys_v,
        weights_all,
        mask_cont_all,
        mask_cat_all,
        row_clusters,
        alpha_cat_vec,
    )
    # new_row_clusters: (n_views, n_rows)

    # Apply the updated row clusters back to the GenJAX trace.
    cm = C["row_clusters", "idx"].set(new_row_clusters)
    argdiffs = genjax.Diff.no_change(trace.args)
    key, subkey_update = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey_update, trace, cm, argdiffs)
    return key, new_trace


def gibbs_update_column_views(
    key,
    trace,
    mu0_vec: jax.Array,
    kappa0_vec: jax.Array,
    ng_alpha0: jax.Array,
    ng_beta0: jax.Array,
    alpha_cat_vec: jax.Array,
):
    """Gibbs update for column view assignments in mixed SBP model (collapsed cat)."""
    args = trace.get_args()
    n_rows = args[0].unwrap()
    n_cont_cols = args[1].unwrap()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    choices = trace.get_choices()
    retval = trace.get_retval()

    table_cont = choices["rows_cont"]
    table_cat = retval["cat"]

    view_idx = choices["views", "view_idx"]

    row_clusters = choices["row_clusters", "idx"]

    v_views = choices["views", "view_weights", "v"]  # (n_views-1,)
    view_weights = _sbp_weights_from_v(v_views)

    total_cols = n_cont_cols + n_cat_cols
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, total_cols)
    new_view_idx = []

    def _loglik_cont_for_view(
        table_cont_col: jax.Array,
        row_clusters_v: jax.Array,
        mu0_c: float,
        kappa0_c: float,
        alpha0_c: float,
        beta0_c: float,
    ) -> jax.Array:
        """Collapsed Normal-Gamma log-likelihood for one view and one cont column."""
        y = table_cont_col
        z = row_clusters_v.astype(jnp.int32)  # (n_rows,)

        one_hot = jax.nn.one_hot(z, n_clusters, dtype=jnp.float32)  # (n_rows, n_k)
        n_k = jnp.sum(one_hot, axis=0)  # (n_clusters,)
        sum_y = jnp.einsum("rn,r->n", one_hot, y)  # (n_clusters,)
        sumsq_y = jnp.einsum("rn,r->n", one_hot, y**2)  # (n_clusters,)

        log_marg_per_cluster = _normal_gamma_marginal_loglik_from_stats(
            n_k,
            sum_y,
            sumsq_y,
            mu0_c,
            kappa0_c,
            alpha0_c,
            beta0_c,
        )
        return jnp.sum(log_marg_per_cluster)

    def _loglik_cat_collapsed_for_view(
        table_cat_col: jax.Array,
        row_clusters_v: jax.Array,
        alpha_c: float,
    ) -> jax.Array:
        """Collapsed Dirichlet-Multinomial log-likelihood for one view and one column."""
        y = table_cat_col.astype(jnp.int32)  # (n_rows,)
        z = row_clusters_v.astype(jnp.int32)  # (n_rows,)

        idx = z * NUM_CATEGORIES + y  # (n_rows,)
        counts_flat = jnp.bincount(
            idx,
            length=n_clusters * NUM_CATEGORIES,
            minlength=n_clusters * NUM_CATEGORIES,
        ).astype(jnp.float32)
        counts = counts_flat.reshape((n_clusters, NUM_CATEGORIES))

        n_k = jnp.sum(counts, axis=1)  # (n_clusters,)

        alpha0 = alpha_c * NUM_CATEGORIES
        # log p(data | alpha) = sum_k [ log Gamma(alpha0) - log Gamma(n_k + alpha0)
        #                              + sum_t (log Gamma(n_{k,t}+alpha_c) - log Gamma(alpha_c)) ]
        term1 = gammaln(alpha0) - gammaln(n_k + alpha0)  # (n_clusters,)
        term2 = jnp.sum(
            gammaln(counts + alpha_c) - gammaln(alpha_c), axis=1
        )  # (n_clusters,)
        return jnp.sum(term1 + term2)

    for c in range(total_cols):
        if c < n_cont_cols:
            col_idx = c
            table_cont_col = table_cont[:, col_idx]  # (n_rows,)
            mu0_c = mu0_vec[col_idx]
            kappa0_c = kappa0_vec[col_idx]
            alpha0_c = ng_alpha0[col_idx]
            beta0_c = ng_beta0[col_idx]
            loglik_vec = jax.vmap(
                _loglik_cont_for_view,
                in_axes=(None, 0, None, None, None, None),
                out_axes=0,
            )(
                table_cont_col,
                row_clusters,
                mu0_c,
                kappa0_c,
                alpha0_c,
                beta0_c,
            )  # (n_views,)
        else:
            c_cat = c - n_cont_cols
            table_cat_col = table_cat[:, c_cat]  # (n_rows,)
            alpha_c = alpha_cat_vec[c_cat]
            loglik_vec = jax.vmap(
                _loglik_cat_collapsed_for_view,
                in_axes=(None, 0, None),
                out_axes=0,
            )(table_cat_col, row_clusters, alpha_c)  # (n_views,)

        logits = loglik_vec + jnp.log(view_weights + 1e-20)
        v_new = jax.random.categorical(keys[c], logits)
        new_view_idx.append(v_new)

    new_view_idx_arr = jnp.stack(new_view_idx, axis=0)

    cm = C["views", "view_idx"].set(new_view_idx_arr)
    argdiffs = genjax.Diff.no_change(trace.args)
    key, subkey = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey, trace, cm, argdiffs)
    return key, new_trace


def gibbs_update_cluster_params(
    key,
    trace,
    alpha_cat_vec: jax.Array,
    mu0_vec: jax.Array,
    kappa0_vec: jax.Array,
    ng_alpha0: jax.Array,
    ng_beta0: jax.Array,
):
    """Gibbs update for continuous means and categorical probs in mixed SBP model."""
    args = trace.get_args()
    n_rows = args[0].unwrap()
    n_cont_cols = args[1].unwrap()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    choices = trace.get_choices()
    retval = trace.get_retval()

    table_cont = choices["rows_cont"]
    table_cat = retval["cat"]

    view_idx = choices["views", "view_idx"]
    view_idx_cont = view_idx[:n_cont_cols]
    view_idx_cat = view_idx[n_cont_cols:]

    clusters_cont_flat = choices["clusters_cont", "mean"]
    clusters_cont = clusters_cont_flat.reshape((n_views, n_clusters, n_cont_cols))
    clusters_prec_flat = choices["clusters_cont", "tau"]
    clusters_prec = clusters_prec_flat.reshape((n_views, n_clusters, n_cont_cols))

    cat_params_flat = choices["clusters_cat", "probs"]
    cat_params = cat_params_flat.reshape(
        (n_views, n_clusters, n_cat_cols, NUM_CATEGORIES)
    )

    row_clusters = choices["row_clusters", "idx"]

    # Continuous parameters (Normal-Gamma conjugate update for each (view, cluster, col))
    updated_cont = clusters_cont
    updated_prec = clusters_prec

    def _cont_body(carry, c_idx):
        key_local, cont_arr, prec_arr = carry
        v_c = view_idx_cont[c_idx]
        obs_col = table_cont[:, c_idx]
        row_clusters_v = row_clusters[v_c]

        m0_c = mu0_vec[c_idx]
        kappa0_c = kappa0_vec[c_idx]
        alpha0_c = ng_alpha0[c_idx]
        beta0_c = ng_beta0[c_idx]

        counts = jnp.bincount(
            row_clusters_v, length=n_clusters, minlength=n_clusters
        ).astype(table_cont.dtype)
        sum_obs = jnp.bincount(
            row_clusters_v, weights=obs_col, length=n_clusters
        )
        sum_obs_sq = jnp.bincount(
            row_clusters_v, weights=obs_col**2, length=n_clusters
        )

        n_k = counts
        mean_data = sum_obs / jnp.where(n_k == 0, 1.0, n_k)
        sse = sum_obs_sq - n_k * mean_data**2
        sse = jnp.where(n_k == 0, 0.0, sse)

        kappa_n = kappa0_c + n_k
        alpha_n = alpha0_c + 0.5 * n_k
        beta_n = beta0_c + 0.5 * sse + 0.5 * (kappa0_c * n_k / kappa_n) * (
            mean_data - m0_c
        ) ** 2

        key_local, subkey_tau, subkey_mu = jax.random.split(key_local, 3)
        # tau ~ Gamma(alpha_n, beta_n) using JAX's gamma (shape-scale) with scale=1/beta_n
        tau_sample = jax.random.gamma(subkey_tau, alpha_n) / jnp.clip(beta_n, 1e-20)

        std_mu = jnp.sqrt(1.0 / (kappa_n * tau_sample))
        eps = jax.random.normal(subkey_mu, shape=(n_clusters,))
        mu_num = kappa0_c * m0_c + n_k * mean_data
        mu_den = jnp.where(kappa_n == 0.0, 1.0, kappa_n)
        mu_sample = mu_num / mu_den + std_mu * eps

        mask = n_k > 0.0

        old_mu = cont_arr[v_c, :, c_idx]
        old_tau = prec_arr[v_c, :, c_idx]
        new_mu = jnp.where(mask, mu_sample, old_mu)
        new_tau = jnp.where(mask, tau_sample, old_tau)

        cont_arr = cont_arr.at[v_c, :, c_idx].set(new_mu)
        prec_arr = prec_arr.at[v_c, :, c_idx].set(new_tau)

        return (key_local, cont_arr, prec_arr), None

    if n_cont_cols > 0:
        (key, updated_cont, updated_prec), _ = lax.scan(
            _cont_body, (key, updated_cont, updated_prec), jnp.arange(n_cont_cols)
        )

    clusters_cont = updated_cont
    clusters_prec = updated_prec

    # Categorical probs
    updated_cat = cat_params

    def _cat_body(carry, c_idx):
        key_local, cat_arr = carry
        v_c = view_idx_cat[c_idx]
        y_col = table_cat[:, c_idx].astype(int)
        row_clusters_v = row_clusters[v_c]

        idx = row_clusters_v * NUM_CATEGORIES + y_col
        counts_flat = jnp.bincount(
            idx,
            length=n_clusters * NUM_CATEGORIES,
            minlength=n_clusters * NUM_CATEGORIES,
        ).astype(table_cat.dtype)
        counts = counts_flat.reshape((n_clusters, NUM_CATEGORIES))

        alpha_prior = alpha_cat_vec[c_idx] * jnp.ones((n_clusters, NUM_CATEGORIES))
        alpha_post = alpha_prior + counts

        key_local, subkey = jax.random.split(key_local)
        keys_k = jax.random.split(subkey, n_clusters)

        def _sample_dirichlet(k_, a_):
            return jax.random.dirichlet(k_, a_)

        probs_post = jax.vmap(_sample_dirichlet)(keys_k, alpha_post)
        cat_arr = cat_arr.at[v_c, :, c_idx].set(probs_post)
        return (key_local, cat_arr), None

    if n_cat_cols > 0:
        (key, updated_cat), _ = lax.scan(
            _cat_body, (key, updated_cat), jnp.arange(n_cat_cols)
        )

    new_cont_flat = clusters_cont.reshape(clusters_cont_flat.shape)
    new_prec_flat = clusters_prec.reshape(clusters_prec_flat.shape)
    new_cat_flat = updated_cat.reshape(cat_params_flat.shape)

    cm = C["clusters_cont", "mean"].set(new_cont_flat)
    cm = cm.at["clusters_cont", "tau"].set(new_prec_flat)
    cm = cm.at["clusters_cat", "probs"].set(new_cat_flat)

    argdiffs = genjax.Diff.no_change(trace.args)
    key, subkey = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey, trace, cm, argdiffs)
    return key, new_trace


def gibbs_update_cluster_sticks(key, trace, alpha_cluster):
    """Gibbs update for cluster-level SBP sticks."""
    args = trace.get_args()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    choices = trace.get_choices()
    row_clusters = choices["row_clusters", "idx"]

    counts = jax.vmap(
        lambda rv: jnp.bincount(rv, length=n_clusters, minlength=n_clusters).astype(
            jnp.float32
        )
    )(row_clusters)  # (n_views, n_clusters)

    n_k = counts[:, :-1]
    suffix_sums = jnp.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
    n_gt = suffix_sums[:, 1:]
    a_post = 1.0 + n_k
    b_post = alpha_cluster + n_gt

    key, subkey = jax.random.split(key)
    new_v = jax.random.beta(subkey, a_post, b_post)

    cm = C["cluster_weights", "v"].set(new_v)
    argdiffs = genjax.Diff.no_change(trace.args)
    key, subkey2 = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey2, trace, cm, argdiffs)
    return key, new_trace


def gibbs_update_view_sticks(key, trace, alpha_view):
    """Gibbs update for view-level SBP sticks."""
    args = trace.get_args()
    n_views = args[3].unwrap()

    choices = trace.get_choices()
    view_idx = choices["views", "view_idx"]
    v_sticks = choices["views", "view_weights", "v"]

    counts = jnp.bincount(view_idx, length=n_views, minlength=n_views).astype(
        jnp.float32
    )

    m_k = counts[:-1]
    suffix_sums = jnp.cumsum(counts[::-1])[::-1]
    m_gt = suffix_sums[1:]
    a_post = 1.0 + m_k
    b_post = alpha_view + m_gt

    key, subkey = jax.random.split(key)
    new_v = jax.random.beta(subkey, a_post, b_post)
    new_v = v_sticks.at[:].set(new_v)

    cm = C["views", "view_weights", "v"].set(new_v)
    argdiffs = genjax.Diff.no_change(trace.args)
    key, subkey2 = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey2, trace, cm, argdiffs)
    return key, new_trace


def _sample_alpha_from_sticks_sbp(
    key: jax.Array,
    v_sticks: jax.Array,
    alpha_current: float,
    a_prior: float,
    b_prior: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for SBP concentration given Beta(1, alpha) sticks."""
    v_flat = v_sticks.reshape(-1)

    # Summed log(1 - v) term; clip for numerical stability.
    sum_log1mv = jnp.sum(jnp.log(jnp.clip(1.0 - v_flat, 1e-20, 1.0)))
    num_sticks = v_flat.size

    # Build a log-spaced grid around current alpha (multiplicative window).
    num_grid = 32
    alpha_safe = jnp.clip(alpha_current, 1e-2, 1e2)
    log_alpha0 = jnp.log(alpha_safe)
    offsets = jnp.linspace(-2.0, 2.0, num_grid)
    alpha_grid = jnp.exp(log_alpha0 + offsets)

    # Prior: Gamma(a_prior, b_prior) with density proportional to alpha^{a-1} exp(-b * alpha).
    log_prior = (a_prior - 1.0) * jnp.log(alpha_grid) - b_prior * alpha_grid

    # Likelihood from independent Beta(1, alpha) sticks:
    # log p(v | alpha) = num_sticks * log alpha + (alpha - 1) * sum_log1mv  (up to const.).
    log_lik = num_sticks * jnp.log(alpha_grid) + (alpha_grid - 1.0) * sum_log1mv

    log_post = log_prior + log_lik

    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    alpha_new = alpha_grid[idx]
    return key, alpha_new


def _sample_alpha_from_dirichlet_grid(
    key: jax.Array,
    probs: jnp.ndarray,
    alpha_current: float,
    a_prior: float,
    b_prior: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for Dirichlet concentration from probs samples.

    Assumes probs[..., j] ~ Dirichlet(alpha * 1_K) i.i.d. over leading axes.
    """
    k = probs.shape[-1]
    probs_flat = probs.reshape(-1, k)
    logp = jnp.log(jnp.clip(probs_flat, 1e-20, 1.0))
    sum_logp = jnp.sum(logp, axis=-1)
    S = jnp.sum(sum_logp)
    m = probs_flat.shape[0]

    num_grid = 32
    alpha_safe = jnp.clip(alpha_current, 1e-2, 1e2)
    log_alpha0 = jnp.log(alpha_safe)
    offsets = jnp.linspace(-2.0, 2.0, num_grid)
    alpha_grid = jnp.exp(log_alpha0 + offsets)

    # Gamma(a, b) prior on alpha
    log_prior = (a_prior - 1.0) * jnp.log(alpha_grid) - b_prior * alpha_grid

    # Dirichlet likelihood for probs under concentration alpha_grid.
    # For each alpha: m * (gammaln(k*alpha) - k*gammaln(alpha)) + (alpha-1)*S
    log_lik = (
        m * (gammaln(k * alpha_grid) - k * gammaln(alpha_grid))
        + (alpha_grid - 1.0) * S
    )

    log_post = log_prior + log_lik

    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    alpha_new = alpha_grid[idx]
    return key, alpha_new


def _sample_ng_alpha0_from_prec_grid(
    key: jax.Array,
    prec: jnp.ndarray,
    alpha0_current: float,
    beta0_current: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for NG_ALPHA0 given Gamma(alpha0, beta0) precisions."""
    prec_flat = prec.reshape(-1)
    log_prec = jnp.log(jnp.clip(prec_flat, 1e-20, jnp.inf))
    s_log_prec = jnp.sum(log_prec)
    s_prec = jnp.sum(prec_flat)
    m = prec_flat.size

    num_grid = 32
    alpha_safe = jnp.clip(alpha0_current, 1e-2, 1e2)
    log_alpha0 = jnp.log(alpha_safe)
    offsets = jnp.linspace(-2.0, 2.0, num_grid)
    alpha_grid = jnp.exp(log_alpha0 + offsets)

    # Prior on alpha0: Gamma(NG_ALPHA0_A, NG_ALPHA0_B)
    log_prior = (NG_ALPHA0_A - 1.0) * jnp.log(alpha_grid) - NG_ALPHA0_B * alpha_grid

    # Likelihood of prec under Gamma(alpha0, beta0_current)
    # sum_j [ (alpha0-1)log tau_j - beta0*tau_j + alpha0*log beta0 - gammaln(alpha0) ]
    log_lik = (
        (alpha_grid - 1.0) * s_log_prec
        - beta0_current * s_prec
        + m * (alpha_grid * jnp.log(beta0_current) - gammaln(alpha_grid))
    )

    log_post = log_prior + log_lik
    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    alpha_new = alpha_grid[idx]
    return key, alpha_new


def _sample_ng_beta0_from_prec_grid(
    key: jax.Array,
    prec: jnp.ndarray,
    alpha0_current: float,
    beta0_current: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for NG_BETA0 given Gamma(alpha0, beta0) precisions."""
    prec_flat = prec.reshape(-1)
    log_prec = jnp.log(jnp.clip(prec_flat, 1e-20, jnp.inf))
    s_log_prec = jnp.sum(log_prec)
    s_prec = jnp.sum(prec_flat)
    m = prec_flat.size

    num_grid = 32
    beta_safe = jnp.clip(beta0_current, 1e-2, 1e2)
    log_beta0 = jnp.log(beta_safe)
    offsets = jnp.linspace(-2.0, 2.0, num_grid)
    beta_grid = jnp.exp(log_beta0 + offsets)

    # Prior on beta0: Gamma(NG_BETA0_A, NG_BETA0_B)
    log_prior = (NG_BETA0_A - 1.0) * jnp.log(beta_grid) - NG_BETA0_B * beta_grid

    # Likelihood of prec under Gamma(alpha0_current, beta0)
    # sum_j [ (alpha0-1)log tau_j - beta*tau_j + alpha0*log beta - gammaln(alpha0) ]
    log_lik = (
        (alpha0_current - 1.0) * s_log_prec
        - beta_grid * s_prec
        + m * (alpha0_current * jnp.log(beta_grid) - gammaln(alpha0_current))
    )

    log_post = log_prior + log_lik
    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    beta_new = beta_grid[idx]
    return key, beta_new


def _sample_mu0_from_means_grid(
    key: jax.Array,
    means: jnp.ndarray,
    prec: jnp.ndarray,
    kappa0_c: float,
    mu0_current: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for mu0 given cluster means and precisions."""
    means_flat = means.reshape(-1)
    prec_flat = prec.reshape(-1)

    num_grid = 32
    # Center grid around current mu0, with a moderate window.
    width = 3.0
    mu_grid = jnp.linspace(mu0_current - width, mu0_current + width, num_grid)

    # Likelihood: mu_k ~ Normal(mu0, (kappa0 * tau_k)^(-1)).
    # log p(means | mu0) = -0.5 * sum_k kappa0 * tau_k * (mu_k - mu0)^2  (up to const.)
    diff = means_flat[None, :] - mu_grid[:, None]  # (G, M)
    weighted_sq = kappa0_c * prec_flat[None, :] * diff**2
    log_lik = -0.5 * jnp.sum(weighted_sq, axis=1)

    # Prior: mu0 ~ Normal(MU0_PRIOR_MEAN, MU0_PRIOR_VAR)
    log_prior = -0.5 * (mu_grid - MU0_PRIOR_MEAN) ** 2 / MU0_PRIOR_VAR

    log_post = log_lik + log_prior
    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    mu0_new = mu_grid[idx]
    return key, mu0_new


def _sample_kappa0_from_means_grid(
    key: jax.Array,
    means: jnp.ndarray,
    prec: jnp.ndarray,
    mu0_c: float,
    kappa0_current: float,
) -> tuple[jax.Array, jax.Array]:
    """Grid-based Gibbs update for kappa0 given cluster means and precisions."""
    means_flat = means.reshape(-1)
    prec_flat = prec.reshape(-1)

    diff = means_flat - mu0_c
    S = jnp.sum(prec_flat * diff**2)
    m = means_flat.size

    num_grid = 32
    kappa_safe = jnp.clip(kappa0_current, 1e-2, 1e2)
    log_kappa0 = jnp.log(kappa_safe)
    offsets = jnp.linspace(-2.0, 2.0, num_grid)
    kappa_grid = jnp.exp(log_kappa0 + offsets)

    # Likelihood: mu_k ~ Normal(mu0, (kappa0 * tau_k)^(-1))
    # log p(means | kappa0) = 0.5 * m * log kappa0 - 0.5 * kappa0 * S + const.
    log_lik = 0.5 * m * jnp.log(kappa_grid) - 0.5 * kappa_grid * S

    # Prior: kappa0 ~ Gamma(KAPPA0_A, KAPPA0_B)
    log_prior = (KAPPA0_A - 1.0) * jnp.log(kappa_grid) - KAPPA0_B * kappa_grid

    log_post = log_lik + log_prior
    key, subkey = jax.random.split(key)
    idx = jax.random.categorical(subkey, log_post)
    kappa0_new = kappa_grid[idx]
    return key, kappa0_new


def gibbs_update_alphas(
    key: jax.Array,
    trace,
    alpha_view: float,
    alpha_cluster: float,
):
    """Gibbs update for alpha_view and alpha_cluster from SBP sticks, and trace args."""
    choices = trace.get_choices()
    v_views = choices["views", "view_weights", "v"]  # (n_views-1,)
    v_clusters = choices["cluster_weights", "v"]  # (n_views, n_clusters-1)

    key, alpha_view_new = _sample_alpha_from_sticks_sbp(
        key, v_views, alpha_view, ALPHA_VIEW_A, ALPHA_VIEW_B
    )
    key, alpha_cluster_new = _sample_alpha_from_sticks_sbp(
        key, v_clusters, alpha_cluster, ALPHA_CLUSTER_A, ALPHA_CLUSTER_B
    )

    # Update trace args so that subsequent uses of mixed_sbp_multiview_table
    # see the new alpha_view / alpha_cluster.
    args = trace.get_args()
    assert len(args) == 8

    argdiffs = (
        genjax.Diff.no_change(args[0]),
        genjax.Diff.no_change(args[1]),
        genjax.Diff.no_change(args[2]),
        genjax.Diff.no_change(args[3]),
        genjax.Diff.no_change(args[4]),
        genjax.Diff.unknown_change(alpha_view_new),
        genjax.Diff.unknown_change(alpha_cluster_new),
        genjax.Diff.no_change(args[7]),
    )

    cm_empty = C.n()
    key, subkey = jax.random.split(key)
    new_trace, _, _, _ = jitted_update(subkey, trace, cm_empty, argdiffs)

    return key, new_trace, alpha_view_new, alpha_cluster_new


def gibbs_update_alpha_cat(
    key: jax.Array,
    trace,
    alpha_cat_vec: jax.Array,
):
    """Gibbs update for per-column alpha_cat_vec from Dirichlet cluster_cat_params."""
    choices = trace.get_choices()
    cat_params_flat = choices["clusters_cat", "probs"]

    args = trace.get_args()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    cat_params = cat_params_flat.reshape(
        (n_views, n_clusters, n_cat_cols, NUM_CATEGORIES)
    )

    def body(carry, c_idx):
        key_local, alpha_vec = carry
        probs_col = cat_params[:, :, c_idx, :]
        alpha_c = alpha_vec[c_idx]
        key_local, alpha_new = _sample_alpha_from_dirichlet_grid(
            key_local, probs_col, alpha_c, ALPHA_CAT_A, ALPHA_CAT_B
        )
        alpha_vec = alpha_vec.at[c_idx].set(alpha_new)
        return (key_local, alpha_vec), None

    if n_cat_cols > 0:
        (key, alpha_cat_new), _ = lax.scan(
            body, (key, alpha_cat_vec), jnp.arange(n_cat_cols)
        )
        cm = C["hyper_cat", "alpha_cat"].set(alpha_cat_new)
        argdiffs = genjax.Diff.no_change(trace.args)
        key, subkey = jax.random.split(key)
        trace, _, _, _ = jitted_update(subkey, trace, cm, argdiffs)
    else:
        alpha_cat_new = alpha_cat_vec

    return key, trace, alpha_cat_new


def gibbs_update_ng_hyperparams(
    key: jax.Array,
    trace,
    ng_alpha0: jax.Array,
    ng_beta0: jax.Array,
):
    """Gibbs update for NG_ALPHA0 and NG_BETA0 from continuous precisions."""
    choices = trace.get_choices()
    clusters_prec_flat = choices["clusters_cont", "tau"]

    args = trace.get_args()
    n_cont_cols = args[1].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    clusters_prec = clusters_prec_flat.reshape((n_views, n_clusters, n_cont_cols))

    def body(carry, c_idx):
        key_local, alpha_vec, beta_vec = carry
        prec_col = clusters_prec[:, :, c_idx]
        alpha_c = alpha_vec[c_idx]
        beta_c = beta_vec[c_idx]

        key_local, alpha_new = _sample_ng_alpha0_from_prec_grid(
            key_local, prec_col, alpha_c, beta_c
        )
        key_local, beta_new = _sample_ng_beta0_from_prec_grid(
            key_local, prec_col, alpha_new, beta_c
        )

        alpha_vec = alpha_vec.at[c_idx].set(alpha_new)
        beta_vec = beta_vec.at[c_idx].set(beta_new)
        return (key_local, alpha_vec, beta_vec), None

    if n_cont_cols > 0:
        (key, ng_alpha0_new, ng_beta0_new), _ = lax.scan(
            body,
            (key, ng_alpha0, ng_beta0),
            jnp.arange(n_cont_cols),
        )
        cm = C["hyper_cont", "alpha0"].set(ng_alpha0_new)
        cm = cm.at["hyper_cont", "beta0"].set(ng_beta0_new)
        argdiffs = genjax.Diff.no_change(trace.args)
        key, subkey = jax.random.split(key)
        trace, _, _, _ = jitted_update(subkey, trace, cm, argdiffs)
    else:
        ng_alpha0_new = ng_alpha0
        ng_beta0_new = ng_beta0

    return key, trace, ng_alpha0_new, ng_beta0_new


def gibbs_update_mu0_kappa0_hyperparams(
    key: jax.Array,
    trace,
    mu0_vec: jax.Array,
    kappa0_vec: jax.Array,
):
    """Gibbs updates for mu0_vec and kappa0_vec from cluster means and precisions."""
    choices = trace.get_choices()
    clusters_cont_flat = choices["clusters_cont", "mean"]
    clusters_prec_flat = choices["clusters_cont", "tau"]

    args = trace.get_args()
    n_cont_cols = args[1].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    clusters_cont = clusters_cont_flat.reshape((n_views, n_clusters, n_cont_cols))
    clusters_prec = clusters_prec_flat.reshape((n_views, n_clusters, n_cont_cols))

    def body(carry, c_idx):
        key_local, mu0_arr, kappa0_arr = carry
        means_col = clusters_cont[:, :, c_idx]
        prec_col = clusters_prec[:, :, c_idx]

        mu0_c = mu0_arr[c_idx]
        kappa0_c = kappa0_arr[c_idx]

        key_local, mu0_new = _sample_mu0_from_means_grid(
            key_local, means_col, prec_col, kappa0_c, mu0_c
        )
        key_local, kappa0_new = _sample_kappa0_from_means_grid(
            key_local, means_col, prec_col, mu0_new, kappa0_c
        )

        mu0_arr = mu0_arr.at[c_idx].set(mu0_new)
        kappa0_arr = kappa0_arr.at[c_idx].set(kappa0_new)
        return (key_local, mu0_arr, kappa0_arr), None

    if n_cont_cols > 0:
        (key, mu0_new, kappa0_new), _ = lax.scan(
            body, (key, mu0_vec, kappa0_vec), jnp.arange(n_cont_cols)
        )
        cm = C["hyper_cont", "mu0"].set(mu0_new)
        cm = cm.at["hyper_cont", "kappa0"].set(kappa0_new)
        argdiffs = genjax.Diff.no_change(trace.args)
        key, subkey = jax.random.split(key)
        trace, _, _, _ = jitted_update(subkey, trace, cm, argdiffs)
    else:
        mu0_new = mu0_vec
        kappa0_new = kappa0_vec

    return key, trace, mu0_new, kappa0_new


def gibbs_sweep(
    key,
    trace,
    alpha_view,
    alpha_cluster,
    alpha_cat_vec,
    mu0_vec,
    kappa0_vec,
    ng_alpha0,
    ng_beta0,
):
    """1 Gibbs sweep (trace-based, each block update internally uses jitted_update)."""
    # Row clusters
    key, trace = gibbs_update_row_clusters(
        key, trace, mu0_vec, kappa0_vec, ng_alpha0, ng_beta0, alpha_cat_vec
    )

    # Cluster params (uses Normal-Gamma and Dirichlet hyperparameters)
    key, trace = gibbs_update_cluster_params(
        key,
        trace,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )

    # SBP sticks
    key, trace = gibbs_update_cluster_sticks(key, trace, alpha_cluster)
    key, trace = gibbs_update_view_sticks(key, trace, alpha_view)

    # Hyperparameters alpha_view / alpha_cluster (SBP concentration)
    key, trace, alpha_view, alpha_cluster = gibbs_update_alphas(
        key, trace, alpha_view, alpha_cluster
    )

    # Hyperparameter alpha_cat (Dirichlet concentration for categorical params, per column)
    key, trace, alpha_cat_vec = gibbs_update_alpha_cat(key, trace, alpha_cat_vec)

    # Hyperparameters for Normal-Gamma prior (continuous scale)
    key, trace, ng_alpha0, ng_beta0 = gibbs_update_ng_hyperparams(
        key, trace, ng_alpha0, ng_beta0
    )

    # Hyperparameters for Normal-Gamma prior (continuous location)
    key, trace, mu0_vec, kappa0_vec = gibbs_update_mu0_kappa0_hyperparams(
        key, trace, mu0_vec, kappa0_vec
    )

    # Column views
    key, trace = gibbs_update_column_views(
        key, trace, mu0_vec, kappa0_vec, ng_alpha0, ng_beta0, alpha_cat_vec
    )

    return (
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )


class GibbsConfig(NamedTuple):
    num_iters: int


class GibbsState(NamedTuple):
    key: jax.Array
    trace: object
    alpha_view: float
    alpha_cluster: float
    alpha_cat_vec: jax.Array
    mu0_vec: jax.Array
    kappa0_vec: jax.Array
    ng_alpha0: jax.Array
    ng_beta0: jax.Array


def gibbs_step(state: GibbsState) -> GibbsState:
    (
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    ) = state
    (
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    ) = gibbs_sweep(
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )
    return GibbsState(
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )


def run_gibbs_mcmc(
    key: jax.Array,
    trace,
    alpha_view: float,
    alpha_cluster: float,
    cfg: GibbsConfig,
    ng_alpha0: float = NG_ALPHA0,
    ng_beta0: float = NG_BETA0,
) -> GibbsState:
    """Trace-based MCMC driver.

    Internally uses jitted_update, but this function itself is kept non-jitted.
    """
    (
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    ) = _extract_hyperparams_from_trace(trace)

    state0 = GibbsState(
        key,
        trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
    )

    def one_step(st, _):
        st = gibbs_step(st)
        return st, None

    state_T, _ = lax.scan(one_step, state0, xs=jnp.arange(cfg.num_iters))
    return state_T


# Backwards-compatible alias (avoids jitting a huge function directly).
run_gibbs_mcmc_jit = run_gibbs_mcmc


def test_update_row_and_column_gibbs_preserves_observations():
    key = jax.random.key(10)
    posterior_trace, _, _ = _simulate_posterior_trace(key)
    original_choices = posterior_trace.get_choices()
    original_retval = posterior_trace.get_retval()
    original_cont = original_choices["rows_cont"]
    original_cat = original_retval["cat"]

    (
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    ) = _extract_hyperparams_from_trace(posterior_trace)

    key, subkey = jax.random.split(key)
    key, trace = gibbs_update_row_clusters(
        subkey,
        posterior_trace,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )
    key, subkey = jax.random.split(key)
    key, trace = gibbs_update_column_views(
        subkey,
        trace,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )

    updated_choices = trace.get_choices()
    updated_retval = trace.get_retval()
    updated_cont = updated_choices["rows_cont"]
    updated_cat = updated_retval["cat"]

    assert jnp.allclose(updated_cont, original_cont)
    assert jnp.allclose(updated_cat, original_cat)


def test_gibbs_sweep_preserves_observations_and_shapes():
    key = jax.random.key(2026)
    posterior_trace, _, _ = _simulate_posterior_trace(key)
    original_choices = posterior_trace.get_choices()
    original_retval = posterior_trace.get_retval()
    original_cont = original_choices["rows_cont"]
    original_cat = original_retval["cat"]

    alpha_view = ALPHA_VIEW
    alpha_cluster = ALPHA_CLUSTER
    (
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
        alpha_cat_vec,
    ) = _extract_hyperparams_from_trace(posterior_trace)

    (
        key,
        updated_trace,
        alpha_view_new,
        alpha_cluster_new,
        alpha_cat_vec_new,
        mu0_vec_new,
        kappa0_vec_new,
        ng_alpha0_new,
        ng_beta0_new,
    ) = gibbs_sweep(
        key,
        posterior_trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )
    updated_choices = updated_trace.get_choices()
    updated_retval = updated_trace.get_retval()
    updated_cont = updated_choices["rows_cont"]
    updated_cat = updated_retval["cat"]

    assert jnp.allclose(updated_cont, original_cont)
    assert jnp.allclose(updated_cat, original_cat)

    total_cols = NUM_CONT_COLS + NUM_CAT_COLS
    assert updated_choices["views", "view_idx"].shape == (total_cols,)
    row_clusters = updated_choices["row_clusters", "idx"]
    assert row_clusters.shape == (NUM_VIEWS, NUM_ROWS)

    assert alpha_view_new > 0.0
    assert alpha_cluster_new > 0.0
    assert jnp.all(alpha_cat_vec_new > 0.0)
    assert jnp.all(ng_alpha0_new > 0.0)
    assert jnp.all(ng_beta0_new > 0.0)


def test_run_gibbs_multiple_iterations_preserves_observations():
    key = jax.random.key(2027)
    posterior_trace, _, _ = _simulate_posterior_trace(key)
    original_choices = posterior_trace.get_choices()
    original_retval = posterior_trace.get_retval()
    original_cont = original_choices["rows_cont"]
    original_cat = original_retval["cat"]

    alpha_view = ALPHA_VIEW
    alpha_cluster = ALPHA_CLUSTER
    cfg = GibbsConfig(num_iters=5)

    key, subkey = jax.random.split(key)
    state_T = run_gibbs_mcmc_jit(
        subkey, posterior_trace, alpha_view, alpha_cluster, cfg=cfg
    )

    final_trace = state_T.trace
    final_choices = final_trace.get_choices()
    final_retval = final_trace.get_retval()
    final_cont = final_choices["rows_cont"]
    final_cat = final_retval["cat"]

    assert jnp.allclose(final_cont, original_cont)
    assert jnp.allclose(final_cat, original_cat)

    assert state_T.alpha_view > 0.0
    assert state_T.alpha_cluster > 0.0


def test_run_gibbs_mcmc_jit_preserves_observations():
    # Same as above, but explicitly exercises the "jit" alias.
    key = jax.random.key(2028)
    posterior_trace, _, _ = _simulate_posterior_trace(key)
    original_choices = posterior_trace.get_choices()
    original_retval = posterior_trace.get_retval()
    original_cont = original_choices["rows_cont"]
    original_cat = original_retval["cat"]

    alpha_view = ALPHA_VIEW
    alpha_cluster = ALPHA_CLUSTER
    cfg = GibbsConfig(num_iters=5)

    key, subkey = jax.random.split(key)
    state_T = run_gibbs_mcmc_jit(
        subkey, posterior_trace, alpha_view, alpha_cluster, cfg=cfg
    )

    final_trace = state_T.trace
    final_choices = final_trace.get_choices()
    final_retval = final_trace.get_retval()
    final_cont = final_choices["rows_cont"]
    final_cat = final_retval["cat"]

    assert jnp.allclose(final_cont, original_cont)
    assert jnp.allclose(final_cat, original_cat)

    assert state_T.alpha_view > 0.0
    assert state_T.alpha_cluster > 0.0


def test_timing_gibbs_update_kernels_jit():
    """Measure compile vs step time for jitted update kernels (informational)."""
    key = jax.random.key(3030)
    # Set up a posterior trace as in other tests.
    posterior_trace, observed_cont, observed_cat = _simulate_posterior_trace(key)

    # JIT row-cluster update
    jitted_row = jax.jit(gibbs_update_row_clusters)

    (
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    ) = _extract_hyperparams_from_trace(posterior_trace)

    key_row = key
    t0 = time.perf_counter()
    key_row, trace_row = jitted_row(
        key_row,
        posterior_trace,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )
    t1 = time.perf_counter()
    key_row, trace_row = jitted_row(
        key_row,
        trace_row,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )
    t2 = time.perf_counter()

    print(
        f"[timing] gibbs_update_row_clusters: "
        f"compile+first = {t1 - t0:.4f}s, second = {t2 - t1:.4f}s"
    )

    # JIT column-view update
    jitted_col = jax.jit(gibbs_update_column_views)

    key_col = key
    t3 = time.perf_counter()
    key_col, trace_col = jitted_col(
        key_col,
        posterior_trace,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )
    t4 = time.perf_counter()
    key_col, trace_col = jitted_col(
        key_col,
        trace_col,
        mu0_vec,
        kappa0_vec,
        ng_alpha0_vec,
        ng_beta0_vec,
        alpha_cat_vec,
    )
    t5 = time.perf_counter()

    print(
        f"[timing] gibbs_update_column_views: "
        f"compile+first = {t4 - t3:.4f}s, second = {t5 - t4:.4f}s"
    )

    # Basic sanity: observations should still match constraints after updates.
    choices_row = trace_row.get_choices()
    retval_row = trace_row.get_retval()
    assert choices_row["rows_cont"].shape == observed_cont.shape
    assert retval_row["cat"].shape == observed_cat.shape

    choices_col = trace_col.get_choices()
    retval_col = trace_col.get_retval()
    assert choices_col["rows_cont"].shape == observed_cont.shape
    assert retval_col["cat"].shape == observed_cat.shape


def test_timing_gibbs_sweep_jit():
    """Measure compile vs step time for jitted gibbs_sweep (informational)."""
    key = jax.random.key(4040)
    # Set up posterior as in other tests.
    posterior_trace, observed_cont, observed_cat = _simulate_posterior_trace(key)

    alpha_view = ALPHA_VIEW
    alpha_cluster = ALPHA_CLUSTER
    (
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
        alpha_cat_vec,
    ) = _extract_hyperparams_from_trace(posterior_trace)

    # Explicitly separate compilation and execution.
    t0 = time.perf_counter()
    compiled_sweep = jax.jit(gibbs_sweep).lower(
        key,
        posterior_trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    ).compile()
    t1 = time.perf_counter()

    (
        key1,
        trace1,
        alpha_view1,
        alpha_cluster1,
        alpha_cat_vec1,
        mu0_vec1,
        kappa0_vec1,
        ng_alpha0_1,
        ng_beta0_1,
    ) = compiled_sweep(
        key,
        posterior_trace,
        alpha_view,
        alpha_cluster,
        alpha_cat_vec,
        mu0_vec,
        kappa0_vec,
        ng_alpha0,
        ng_beta0,
    )
    t2 = time.perf_counter()
    (
        key2,
        trace2,
        alpha_view2,
        alpha_cluster2,
        alpha_cat_vec2,
        mu0_vec2,
        kappa0_vec2,
        ng_alpha0_2,
        ng_beta0_2,
    ) = compiled_sweep(
        key1,
        trace1,
        alpha_view1,
        alpha_cluster1,
        alpha_cat_vec1,
        mu0_vec1,
        kappa0_vec1,
        ng_alpha0_1,
        ng_beta0_1,
    )
    t3 = time.perf_counter()
    (
        key3,
        trace3,
        alpha_view3,
        alpha_cluster3,
        alpha_cat_vec3,
        mu0_vec3,
        kappa0_vec3,
        ng_alpha0_3,
        ng_beta0_3,
    ) = compiled_sweep(
        key2,
        trace2,
        alpha_view2,
        alpha_cluster2,
        alpha_cat_vec2,
        mu0_vec2,
        kappa0_vec2,
        ng_alpha0_2,
        ng_beta0_2,
    )
    t4 = time.perf_counter()

    print(
        f"[timing] gibbs_sweep (jitted compiled): "
        f"compile_only = {t1 - t0:.4f}s, "
        f"first = {t2 - t1:.4f}s, "
        f"second = {t3 - t2:.4f}s, "
        f"third = {t4 - t3:.4f}s"
    )

    # Observations should be preserved and shapes consistent.
    choices3 = trace3.get_choices()
    retval3 = trace3.get_retval()
    assert jnp.allclose(choices3["rows_cont"], observed_cont)
    assert jnp.allclose(retval3["cat"], observed_cat)
    assert alpha_view3 > 0.0
    assert alpha_cluster3 > 0.0
