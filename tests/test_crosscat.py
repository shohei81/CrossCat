"""Tests and helper kernels for a mixed-type SBP multi-view CrossCat model."""

import time

import jax
import jax.numpy as jnp

from crosscat import helpers
from crosscat.constants import (
    ALPHA_CLUSTER,
    ALPHA_VIEW,
    NUM_CAT_COLS,
    NUM_CATEGORIES,
    NUM_CLUSTERS,
    NUM_CONT_COLS,
    NUM_ROWS,
    NUM_VIEWS,
)
from crosscat.gibbs import (
    GibbsConfig,
    gibbs_sweep,
    gibbs_update_column_views,
    gibbs_update_row_clusters,
    run_gibbs_mcmc_jit,
)
from crosscat.inference import (
    _extract_hyperparams_from_trace,
    _simulate_posterior_trace,
)
from crosscat.model import _default_args, mixed_sbp_multiview_table


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


def test_helper_posteriors_and_gibbs_wrappers():
    key = jax.random.key(5050)
    posterior_trace, observed_cont, observed_cat = _simulate_posterior_trace(key)

    helper_trace = helpers.posterior_from_observations(
        observed_cont, observed_cat, key=jax.random.key(123)
    )
    helper_choices = helper_trace.get_choices()
    assert jnp.allclose(helper_choices["rows_cont"], observed_cont)
    helper_retval = helper_trace.get_retval()
    assert jnp.allclose(helper_retval["cat"], observed_cat)

    print(
        "[helpers] posterior rows_cont[0]:",
        jnp.asarray(helper_choices["rows_cont"])[0],
    )
    if NUM_CAT_COLS > 0:
        print(
            "[helpers] posterior cat row0:",
            jnp.asarray(helper_retval["cat"])[0],
        )

    _, state = helpers.run_gibbs_iterations(
        helper_trace, key=jax.random.key(456), num_iters=3
    )
    final_choices = state.trace.get_choices()
    final_retval = state.trace.get_retval()
    assert final_choices["rows_cont"].shape == observed_cont.shape
    assert final_retval["cat"].shape == observed_cat.shape
    assert state.alpha_view > 0.0
    assert state.alpha_cluster > 0.0


def test_helper_predictive_and_impute_samples():
    key = jax.random.key(6060)
    posterior_trace, observed_cont, observed_cat = _simulate_posterior_trace(key)

    queries: list[tuple[int, int]] = []
    if NUM_CONT_COLS > 0:
        queries.append((0, 0))
    if NUM_CAT_COLS > 0:
        queries.append((0, NUM_CONT_COLS))

    key, samples = helpers.predictive_samples(
        posterior_trace,
        queries,
        key=key,
        num_samples=5,
    )
    assert samples.shape == (5, len(queries))
    print("[helpers] predictive samples:", samples)

    if NUM_CAT_COLS > 0:
        cat_draws = samples[:, -1]
        assert jnp.all(cat_draws >= 0)
        assert jnp.all(cat_draws < NUM_CATEGORIES)
    if NUM_CONT_COLS > 0:
        assert jnp.var(samples[:, 0]) > 0.0

    key, imputations = helpers.impute(
        posterior_trace, queries, key=key, num_samples=10
    )
    assert imputations.shape == (len(queries),)
    print("[helpers] imputed means:", imputations)

    # Original trace should remain untouched by helper sampling.
    original_choices = posterior_trace.get_choices()
    original_retval = posterior_trace.get_retval()
    assert jnp.allclose(original_choices["rows_cont"], observed_cont)
    assert jnp.allclose(original_retval["cat"], observed_cat)
