"""Trace-based Gibbs update kernels for the CrossCat example."""

from __future__ import annotations

from typing import NamedTuple

import genjax
import jax
import jax.numpy as jnp
from genjax import ChoiceMapBuilder as C  # type: ignore
from jax import lax
from jax.scipy.special import gammaln

from . import constants as const
from .inference import _extract_hyperparams_from_trace
from .model import jitted_update
from .stats import (
    _normal_gamma_marginal_loglik_from_stats,
    _normal_gamma_predictive_loglik_from_stats,
)


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


# ======================
# Trace-based Gibbs updates (using jitted_update)
# ======================


def gibbs_update_row_clusters(
    key,
    trace,
    # 以下の十分統計量(mu0_vecなど)はUncollapsedでは直接使いませんが
    # 引数インターフェースを変えないならそのままでもOKです。
    # 代わりに、Traceから現在のクラスタパラメータを取得します。
    *args, 
):
    """
    Blocked Gibbs update for row cluster assignments.
    Collapsed (積分消去) ではなく、現在のクラスタパラメータ theta を条件として
    全行を一括並列更新 (vmap) します。
    """
    args_trace = trace.get_args()
    n_rows = args_trace[0].unwrap()
    n_cont_cols = args_trace[1].unwrap()
    n_cat_cols = args_trace[2].unwrap()
    n_views = args_trace[3].unwrap()
    n_clusters = args_trace[4].unwrap()

    choices = trace.get_choices()
    retval = trace.get_retval()

    # --- 1. データの取得 ---
    rows_cont = choices["rows_cont"]  # (n_rows, n_cont_cols)
    rows_cat = retval["cat"]          # (n_rows, n_cat_cols)

    # --- 2. 現在のパラメータ(theta)の取得 ---
    # これがあるのが Uncollapsed の特徴です
    
    # クラスタ重み (n_views, n_clusters)
    v_cluster = choices["cluster_weights", "v"]
    weights_all = jax.vmap(_sbp_weights_from_v)(v_cluster)

    # 連続変数のパラメータ (n_views, n_clusters, n_cont_cols)
    if n_cont_cols > 0:
        mus = choices["clusters_cont", "mean"].reshape(n_views, n_clusters, n_cont_cols)
        taus = choices["clusters_cont", "tau"].reshape(n_views, n_clusters, n_cont_cols)
    else:
        mus = None
        taus = None

    # カテゴリ変数のパラメータ (n_views, n_clusters, n_cat_cols, n_categories)
    if n_cat_cols > 0:
        probs = choices["clusters_cat", "probs"].reshape(
            n_views, n_clusters, n_cat_cols, const.NUM_CATEGORIES
        )
    else:
        probs = None

    # View割り当て (n_cols,)
    view_idx = choices["views", "view_idx"]
    view_idx_cont = view_idx[:n_cont_cols]
    view_idx_cat = view_idx[n_cont_cols:]

    # --- 3. Viewごとのマスク作成 ---
    views_arange = jnp.arange(n_views)[:, None]
    mask_cont = (views_arange == view_idx_cont[None, :]) # (n_views, n_cont_cols)
    if n_cat_cols > 0:
        mask_cat = (views_arange == view_idx_cat[None, :])   # (n_views, n_cat_cols)
    else:
        mask_cat = jnp.zeros((n_views, 0), dtype=bool)

    # --- 4. 並列計算ロジック (vmap over views) ---
    
    # 1つのViewについて、全行・全クラスタの対数尤度を計算する関数
    def _compute_logits_one_view(
        weights_v, # (n_clusters,)
        mus_v,     # (n_clusters, n_cont_cols) 
        taus_v,    # (n_clusters, n_cont_cols)
        probs_v,   # (n_clusters, n_cat_cols, n_cats)
        mask_cont_v, # (n_cont_cols,)
        mask_cat_v,  # (n_cat_cols,)
    ):
        # 連続変数の対数尤度: Sum over columns (masked)
        # rows_cont: (n_rows, n_cont)
        # mus_v: (n_clusters, n_cont)
        # logpdf: (n_rows, n_clusters, n_cont) -> sum -> (n_rows, n_clusters)
        if n_cont_cols > 0:
            # ブロードキャスト: (n_rows, 1, n_cont) vs (1, n_clusters, n_cont)
            diff = rows_cont[:, None, :] - mus_v[None, :, :]
            # Normal log-pdf: 0.5 * (log(tau) - log(2pi) - tau * diff^2)
            log_pdf_cont = 0.5 * (jnp.log(jnp.clip(taus_v, 1e-20)) - jnp.log(2 * jnp.pi) - taus_v * diff**2)
            # マスクされている列だけ足す
            log_lik_cont = jnp.sum(log_pdf_cont * mask_cont_v[None, None, :], axis=2)
        else:
            log_lik_cont = jnp.zeros((n_rows, n_clusters))

        # カテゴリ変数の対数尤度
        if n_cat_cols > 0:
            # rows_cat: (n_rows, n_cat) -> 値自体がインデックス
            # probs_v: (n_clusters, n_cat, n_vocab)
            
            # 各行・各列がどの確率値を使うかを取得
            # jax.vmap等を使っても良いですが、take_along_axis的な処理が必要です。
            # ここでは簡易的に、OneHot的な計算あるいはgatherを行うイメージです。
            
            # (n_rows, 1, n_cat, 1)
            row_vals_exp = rows_cat[:, None, :, None].astype(jnp.int32)
            
            # (1, n_clusters, n_cat, n_vocab) -> take -> (n_rows, n_clusters, n_cat, 1)
            selected_probs = jnp.take_along_axis(
                probs_v[None, ...], row_vals_exp, axis=3
            ).squeeze(-1)
            
            log_pdf_cat = jnp.log(jnp.clip(selected_probs, 1e-20))
            log_lik_cat = jnp.sum(log_pdf_cat * mask_cat_v[None, None, :], axis=2)
        else:
            log_lik_cat = jnp.zeros((n_rows, n_clusters))
            
        # 事前分布 (log weights)
        log_prior = jnp.log(jnp.clip(weights_v, 1e-20)) # (n_clusters,)
        
        # 合計対数尤度 (n_rows, n_clusters)
        logits = log_lik_cont + log_lik_cat + log_prior[None, :]
        return logits

    # vmapで全View一括計算
    # mus, taus, probs はView次元(0軸)を持つので、そのままマップされます
    logits_all_views = jax.vmap(
        _compute_logits_one_view,
        in_axes=(0, 0, 0, 0, 0, 0)
    )(
        weights_all,
        mus if mus is not None else jnp.zeros((n_views, n_clusters, 0)),
        taus if taus is not None else jnp.zeros((n_views, n_clusters, 0)),
        probs if probs is not None else jnp.zeros((n_views, n_clusters, 0, 0)),
        mask_cont,
        mask_cat
    )
    # logits_all_views: (n_views, n_rows, n_clusters)

    # --- 5. サンプリング ---
    key, subkey = jax.random.split(key)
    # 全View, 全Rowを一括サンプリング
    new_row_clusters = jax.random.categorical(subkey, logits_all_views, axis=2)
    # (n_views, n_rows)

    # --- 6. Traceの更新 ---
    cm = C["row_clusters", "idx"].set(new_row_clusters.astype(jnp.int32))
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

        idx = z * const.NUM_CATEGORIES + y  # (n_rows,)
        counts_flat = jnp.bincount(
            idx,
            length=n_clusters * const.NUM_CATEGORIES,
            minlength=n_clusters * const.NUM_CATEGORIES,
        ).astype(jnp.float32)
        counts = counts_flat.reshape((n_clusters, const.NUM_CATEGORIES))

        n_k = jnp.sum(counts, axis=1)  # (n_clusters,)

        alpha0 = alpha_c * const.NUM_CATEGORIES
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

    if n_cat_cols > 0:
        cat_params_flat = choices["clusters_cat", "probs"]
        cat_params = cat_params_flat.reshape(
            (n_views, n_clusters, n_cat_cols, const.NUM_CATEGORIES)
        )
    else:
        cat_params_flat = jnp.zeros((0,), dtype=table_cont.dtype)
        cat_params = jnp.zeros(
            (n_views, n_clusters, 0, const.NUM_CATEGORIES), dtype=table_cont.dtype
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

        idx = row_clusters_v * const.NUM_CATEGORIES + y_col
        counts_flat = jnp.bincount(
            idx,
            length=n_clusters * const.NUM_CATEGORIES,
            minlength=n_clusters * const.NUM_CATEGORIES,
        ).astype(table_cat.dtype)
        counts = counts_flat.reshape((n_clusters, const.NUM_CATEGORIES))

        alpha_prior = alpha_cat_vec[c_idx] * jnp.ones((n_clusters, const.NUM_CATEGORIES))
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
    if n_cat_cols > 0:
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
    log_prior = (const.NG_ALPHA0_A - 1.0) * jnp.log(alpha_grid) - const.NG_ALPHA0_B * alpha_grid

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
    log_prior = (const.NG_BETA0_A - 1.0) * jnp.log(beta_grid) - const.NG_BETA0_B * beta_grid

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
    log_prior = -0.5 * (mu_grid - const.MU0_PRIOR_MEAN) ** 2 / const.MU0_PRIOR_VAR

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
    log_prior = (const.KAPPA0_A - 1.0) * jnp.log(kappa_grid) - const.KAPPA0_B * kappa_grid

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
        key, v_views, alpha_view, const.ALPHA_VIEW_A, const.ALPHA_VIEW_B
    )
    key, alpha_cluster_new = _sample_alpha_from_sticks_sbp(
        key, v_clusters, alpha_cluster, const.ALPHA_CLUSTER_A, const.ALPHA_CLUSTER_B
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
    args = trace.get_args()
    n_cat_cols = args[2].unwrap()
    n_views = args[3].unwrap()
    n_clusters = args[4].unwrap()

    if n_cat_cols == 0:
        return key, trace, alpha_cat_vec

    choices = trace.get_choices()
    cat_params_flat = choices["clusters_cat", "probs"]

    cat_params = cat_params_flat.reshape(
        (n_views, n_clusters, n_cat_cols, const.NUM_CATEGORIES)
    )

    def body(carry, c_idx):
        key_local, alpha_vec = carry
        probs_col = cat_params[:, :, c_idx, :]
        alpha_c = alpha_vec[c_idx]
        key_local, alpha_new = _sample_alpha_from_dirichlet_grid(
            key_local, probs_col, alpha_c, const.ALPHA_CAT_A, const.ALPHA_CAT_B
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
    ng_alpha0: float = const.NG_ALPHA0,
    ng_beta0: float = const.NG_BETA0,
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
