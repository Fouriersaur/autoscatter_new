"""
gaussian_states.py
===================
Pure-Gaussian-state structure: Siegel-matrix parametrisation, dark-state
nullifiers, and the "Approach-A" state-level ceiling optimisation.

Implements gaussian_autoscatter_algorithm.md §1.3, §1.4, §5, §7. A pure
Gaussian state is the dark state of N nullifiers c_k in ker(sigma + iOmega),
a Lagrangian plane coordinatised by the Siegel matrix Z = X + iY
(complex symmetric, Im Z > 0):

    sigma = [[ Y^-1, Y^-1 X ], [ X Y^-1, Y + X Y^-1 X ]]     (xx..pp order)

No AutoScatter analogue -- AutoScatter targets a scattering behaviour, not
a density matrix, so it has no "pure state" concept.

Convention: this package uses vacuum = 1/2 (sigma_vac = I/2), not the doc's
vacuum = 1. All formulas below are the 1/2-convention versions (extra
factor of 1/2 on sigma, kernel of sigma + (i/2)Omega instead of sigma+iOmega)
so they stay consistent with targets.py/covariance_physics.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt

from reservoir_engineering.covariance_physics import symplectic_form

jax.config.update("jax_enable_x64", True)


# sigma_from_siegel(X, Y) -> pure-state covariance for Z = X + iY.
# Pure by construction for any symmetric X and any Y > 0 (§7: purity
# enforced structurally here, not via a penalty).
# Doc uses xx..pp block ordering; this package uses per-mode (x0,p0,x1,p1,...)
# so we build in xx..pp then permute.

def sigma_from_siegel(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    N = X.shape[0]
    Yinv = jnp.linalg.inv(Y)
    top    = jnp.concatenate([Yinv,        Yinv @ X], axis=1)
    bottom = jnp.concatenate([X @ Yinv,    Y + X @ Yinv @ X], axis=1)
    sigma_block = 0.5 * jnp.concatenate([top, bottom], axis=0)   # xx..pp order

    perm = []
    for i in range(N):
        perm += [i, N + i]
    perm = jnp.array(perm)
    return sigma_block[jnp.ix_(perm, perm)]


# dark_state_nullifiers(sigma) -> ker(sigma + (i/2)Omega), the N ideal
# dissipator directions (§1.3). M is Hermitian; for an exact pure state its
# N smallest eigenvalues are 0. Returns the corresponding eigenvectors (also
# a good approximate answer when sigma is only nearly pure).

def dark_state_nullifiers(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma)
    N = sigma.shape[0] // 2
    Omega = np.asarray(symplectic_form(N))
    M = sigma + 0.5j * Omega
    _, eigvecs = np.linalg.eigh(M)   # ascending, M Hermitian
    return eigvecs[:, :N].T          # (N, 2N) complex


# optimise_state(N, loss_fn) -> Approach-A ceiling (§5, §7): optimise
# directly over the Siegel matrix (X, Y), before any discrete (G,C) search.
# Purity is automatic (sigma_from_siegel), so unlike the §3 inner loop no
# purity/stability term is needed -- just the target-predicate loss.
#
# loss_fn(sigma) -> jax scalar, e.g. any combination of the §6 functionals.
# Free params are unconstrained reals: X from its upper triangle, Y = LL^T
# via a Cholesky factor L (diagonal passed through softplus to stay > 0).
#
# Returns dict with 'Z', 'sigma', 'loss' (the ceiling), and 'nullifiers'
# (the ideal dissipators to seed/sparsify the discrete search).

def _params_to_XY(params: jnp.ndarray, N: int):
    n_tri = N * (N + 1) // 2
    x_params, l_params = params[:n_tri], params[n_tri:]

    rows, cols = np.triu_indices(N)
    X = jnp.zeros((N, N))
    X = X.at[rows, cols].set(x_params)
    X = X + X.T - jnp.diag(jnp.diag(X))

    L = jnp.zeros((N, N))
    L = L.at[rows, cols].set(l_params)   # upper-tri; used as L^T below
    L = L.T
    diag_idx = jnp.arange(N)
    L = L.at[diag_idx, diag_idx].set(jax.nn.softplus(L[diag_idx, diag_idx]) + 1e-6)
    Y = L @ L.T + 1e-9 * jnp.eye(N)
    return X, Y


def optimise_state(
    N: int,
    loss_fn,
    num_restarts: int = 8,
    maxiter: int = 500,
    init_range: float = 1.0,
) -> dict:
    n_tri = N * (N + 1) // 2
    n_params = 2 * n_tri

    def total_loss(params):
        X, Y = _params_to_XY(params, N)
        sigma = sigma_from_siegel(X, Y)
        return loss_fn(sigma)

    loss_jit = jax.jit(total_loss)
    grad_jit = jax.jit(jax.grad(total_loss))

    best = None
    for _ in range(num_restarts):
        x0 = np.random.uniform(-init_range, init_range, n_params)
        result = sciopt.minimize(
            fun=lambda p: float(loss_jit(jnp.array(p, dtype=float))),
            jac=lambda p: np.array(grad_jit(jnp.array(p, dtype=float)), dtype=float),
            x0=x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter},
        )
        if best is None or result.fun < best.fun:
            best = result

    X_star, Y_star = _params_to_XY(jnp.array(best.x), N)
    sigma_star = np.array(sigma_from_siegel(X_star, Y_star))
    Z_star = np.array(X_star) + 1j * np.array(Y_star)

    return {
        'Z': Z_star,
        'X': np.array(X_star),
        'Y': np.array(Y_star),
        'sigma': sigma_star,
        'loss': float(best.fun),
        'nullifiers': dark_state_nullifiers(sigma_star),
    }
