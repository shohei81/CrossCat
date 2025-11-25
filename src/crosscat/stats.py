"""Probability helper utilities shared across CrossCat components."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln


def _normal_gamma_marginal_loglik_from_stats(
    n: jnp.ndarray,
    sum_y: jnp.ndarray,
    sumsq_y: jnp.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> jnp.ndarray:
    """Normal-Gamma marginal log-likelihood log p(y | hyper) from sufficient stats."""
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
