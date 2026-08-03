"""
targets.py
==========
Standard target covariance matrices for Gaussian quantum state engineering.
Provides the RIGHT-HAND SIDE of the optimisation problem: given one of
these target covariances, the rest of the pipeline discovers which circuit
topology achieves it.

Conventions: real quadrature basis q=(x_0,p_0,x_1,p_1,...), sigma_ij =
1/2<{q_i,q_j}>. Vacuum sigma_vac = 1/2*I2 per mode. Uncertainty principle:
sigma + i/2*Omega >= 0, equivalently all symplectic eigenvalues >= 1/2 —
unphysical targets will confuse the optimizer.

Reference target: squeezed_vacuum(r) for a single mechanical mode, matched
by a 2-mode (cavity+mechanics) system with BS+TMS coupling (Kronwald
topology) — the primary validation target throughout this package.
"""

import numpy as np


# Single-mode squeezed vacuum: sigma = 1/2*diag(e^-2r, e^2r). r=0 is vacuum;
# x squeezed below 1/2 for r>0. Squeezing in dB ~ 8.686*r (r=1.0 -> ~8.7dB).
def squeezed_vacuum(r: float) -> np.ndarray:
    return 0.5 * np.diag([np.exp(-2*r), np.exp(2*r)])


# Two-mode squeezed vacuum (TMSV/EPR state) for modes 0,1: as r->inf,
# approaches perfect x-x/p-p correlation (parametric down-conversion).
#   sigma = 1/2*[[cosh(2r)I2, -sinh(2r)sz], [-sinh(2r)sz, cosh(2r)I2]]
# Dissipative-engineering (Woolley-Clerk) convention: x0+x1, p0-p1 are the
# squeezed quadratures (equivalent to the PDC convention up to a
# pi-rotation on mode 0). Log negativity = r.
def two_mode_squeezed(r: float) -> np.ndarray:
    c  = np.cosh(2*r)
    s  = np.sinh(2*r)
    sz = np.diag([1., -1.])
    I2 = np.eye(2)
    return 0.5 * np.block([[c*I2, -s*sz],
                            [-s*sz, c*I2]])


# Ground state, sigma = 1/2*I. Sanity-check target: a dissipative system at
# T=0 with no active squeezing should always reach this.
def vacuum(n_modes: int = 1) -> np.ndarray:
    return 0.5 * np.eye(2 * n_modes)


# Thermal state at occupation n_bar, sigma = (n_bar+1/2)*I.
def thermal(n_bar: float, n_modes: int = 1) -> np.ndarray:
    return (n_bar + 0.5) * np.eye(2 * n_modes)


# Gaussian cluster (graph) state for measurement-based quantum computation:
# each mode individually p-squeezed (delta -> 0 is the ideal infinite-
# squeezing limit), then CZ gates applied per the adjacency matrix
# (S_CZ = [[I,0],[Gamma,I]]). Default adjacency is a linear chain.
# Which topology dissipatively ENGINEERS this state is an open question —
# that's what the searcher is for.
def cluster_state(n_modes: int, delta: float, adjacency: np.ndarray = None) -> np.ndarray:
    if adjacency is None:
        adjacency = np.zeros((n_modes, n_modes))
        for i in range(n_modes - 1):
            adjacency[i, i+1] = 1.0
            adjacency[i+1, i] = 1.0

    sigma_product = np.zeros((2*n_modes, 2*n_modes))
    for i in range(n_modes):
        sigma_product[2*i,   2*i  ] = 0.5 * np.exp(+2*delta)
        sigma_product[2*i+1, 2*i+1] = 0.5 * np.exp(-2*delta)

    S = np.eye(2*n_modes)
    for i in range(n_modes):
        for j in range(i+1, n_modes):
            if adjacency[i, j] != 0:
                g = float(adjacency[i, j])
                S[2*i+1, 2*j  ] += g   # p_i gets x_j
                S[2*j+1, 2*i  ] += g   # p_j gets x_i

    return S @ sigma_product @ S.T


# Variance of each cluster nullifier f_j = p_j - sum_k A[j,k]*x_k. Ideal
# cluster state: all variances -> e^{-2*delta}/2 as delta -> 0.
def nullifier_variances(sigma: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    N = sigma.shape[0] // 2
    variances = np.zeros(N)
    for j in range(N):
        v = sigma[2*j+1, 2*j+1]
        for k in range(N):
            if adjacency[j, k] != 0:
                v -= 2.0 * adjacency[j, k] * sigma[2*j+1, 2*k]
                for l in range(N):
                    if adjacency[j, l] != 0:
                        v += adjacency[j, k] * adjacency[j, l] * sigma[2*k, 2*l]
        variances[j] = v
    return variances


# True iff sigma + i/2*Omega >= 0, i.e. all symplectic eigenvalues >= 1/2.
# Validate user-defined targets before running the optimizer — an
# unphysical target produces meaningless results.
def is_physical(sigma):
    return bool(np.all(symplectic_eigenvalues(sigma) >= 0.5 - 1e-10))


# The N symplectic eigenvalues of a 2N x 2N covariance matrix — physical
# invariants under symplectic transformations. nu_k >= 1/2 always (physical);
# nu_k == 1/2 means mode k is pure; nu_k > 1/2 means mixed/thermal.
# Eigenvalues of i*Omega*sigma come in +-nu_k pairs; take abs and dedupe.
def symplectic_eigenvalues(sigma: np.ndarray) -> np.ndarray:
    N = sigma.shape[0] // 2
    Omega = np.zeros_like(sigma)
    for i in range(N):
        Omega[2*i, 2*i+1] = 1.
        Omega[2*i+1, 2*i] = -1.
    M = 1j * Omega @ sigma
    eigs = np.linalg.eigvals(M)
    nus = np.sort(np.abs(eigs.real))
    return nus[::2]   # [nu1,nu1,nu2,nu2,...] -> take every other


# Squeezing in dB for one mode's x-quadrature: S = -10*log10(2*sigma_xx).
# S>0 = squeezed below shot noise, S=0 = vacuum. Diagonalise the mode's 2x2
# block instead if you need the maximally-squeezed quadrature, not just x.
def squeezing_db(sigma: np.ndarray, mode_id: int = 0) -> float:
    sigma_xx = sigma[2*mode_id, 2*mode_id]
    return float(-10 * np.log10(2 * sigma_xx))


# Duan-Simon inseparability criterion for a 2-mode state (4x4 sigma): True
# iff entangled. Checks both EPR orientations (u=x0-x1,v=p0+p1 and
# u=x0+x1,v=p0-p1) since the two are convention-dependent (PDC vs
# dissipative/Woolley-Clerk); separable requires Var(u)+Var(v) >= 1 in both.
def duan_criterion(sigma: np.ndarray) -> bool:
    sum1 = (sigma[0,0] + sigma[2,2] - 2*sigma[0,2]
            + sigma[1,1] + sigma[3,3] + 2*sigma[1,3])
    sum2 = (sigma[0,0] + sigma[2,2] + 2*sigma[0,2]
            + sigma[1,1] + sigma[3,3] - 2*sigma[1,3])
    return bool(sum1 < 1.0 or sum2 < 1.0)


# Logarithmic negativity for a 2-mode state (4x4 sigma only): partial-
# transpose (flip mode-1's p), take its smallest symplectic eigenvalue
# nu_min, E_N = max(0, -log2(2*nu_min)). 0 = separable, >0 = entangled.
def log_negativity(sigma: np.ndarray) -> float:
    if sigma.shape != (4, 4):
        raise ValueError("log_negativity requires a 4x4 (2-mode) covariance matrix")
    T = np.diag([1., 1., 1., -1.])
    sigma_pt = T @ sigma @ T
    nus = symplectic_eigenvalues(sigma_pt)
    nu_min = float(np.min(nus))
    return float(max(0., -np.log2(2 * nu_min)))


# Purity mu = 1/sqrt(det(2*sigma)); 1 for pure, <1 for mixed. On a reduced
# (signal-modes-only) sigma this is the physically relevant purity of the
# engineered state — mixedness comes from thermal bath noise and residual
# entanglement with the auxiliary modes at finite cooperativity (mu -> 1 as
# C -> infinity in the resolved-sideband limit).
def purity(sigma: np.ndarray) -> float:
    return float(1.0 / np.sqrt(np.linalg.det(2 * sigma)))


# §6 target-quantity reference, remaining functionals (the rest are above:
# symplectic_eigenvalues, squeezing_db, log_negativity, duan_criterion, purity).

# Mean energy per mode: Tr(sigma)/(2N).
def mean_energy(sigma: np.ndarray) -> float:
    return float(np.trace(sigma) / sigma.shape[0])


# u^T sigma u for an arbitrary collective quadrature vector u.
def collective_quadrature_variance(sigma: np.ndarray, u: np.ndarray) -> float:
    u = np.asarray(u)
    return float(u @ sigma @ u)


# Fidelity between an achieved covariance sigma_a and a PURE reference
# state sigma_b: F = 1/sqrt(det(sigma_a+sigma_b)) (exact whenever one state
# is pure — Marian&Marian 2012 / Banchi-Braunstein-Pirandola PRL 115,
# 260501 (2015), specialised to this package's vacuum=1/2 convention;
# verified by hand for the single-mode minimum-uncertainty case: F =
# 2*sqrt(ab)/(a+b), which reduces to F=1 for sigma_a==sigma_b==vacuum).
# Raises if sigma_b isn't (numerically) pure.
def fidelity(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    sigma_a = np.asarray(sigma_a)
    sigma_b = np.asarray(sigma_b)
    if sigma_a.shape != sigma_b.shape:
        raise ValueError("fidelity requires sigma_a and sigma_b of the same shape")
    if abs(purity(sigma_b) - 1.0) > 1e-6:
        raise ValueError("fidelity(sigma_a, sigma_b) requires sigma_b to be a pure "
                          "reference state (purity(sigma_b) == 1); got "
                          f"purity={purity(sigma_b):.6f}")
    denom = np.linalg.det(sigma_a + sigma_b)
    return float(1.0 / np.sqrt(denom))


# ───────────────────────────────────────────────────────────────────────────
# §2.1 target predicate: {(Q_i, q_i, relation, modes)}
# ───────────────────────────────────────────────────────────────────────────
# The doc's target is a list of scalar quantities Q_i matched to target
# values q_i, each on its own mode subset — NOT necessarily a full target
# covariance matrix (that's the special case of "specific correlations":
# matching every entry of a block at once, still supported directly via
# CovarianceOptimizer's sigma_target argument). TargetTerm is one such
# (Q_i, q_i, weight, modes) entry; CovarianceOptimizer(target_predicate=
# [...]) accepts a list of these (alongside or instead of sigma_target).
#
# fn must be JAX-differentiable (called on a jnp array inside the §3 loss)
# — use the jnp_* functionals in covariance_physics.py, or
# target_quadratic_form below for anything expressible as Tr(Q sigma).

class TargetTerm:
    def __init__(self, fn, q, weight: float = 1.0, modes=None, name: str = None):
        self.fn = fn
        self.q = float(q)
        self.weight = float(weight)
        self.modes = list(modes) if modes is not None else None
        self.name = name or getattr(fn, '__name__', 'Q')

    def __repr__(self):
        return f'TargetTerm({self.name}, q={self.q}, weight={self.weight}, modes={self.modes})'


# Any quantity linear in sigma (Tr(Q sigma) = sum(Q * sigma)) as a §2.1 term
# — covers a single quadrature variance (Q = u u^T), an EPR/Duan-style sum
# Var(u)+Var(v) (Q = u u^T + v v^T), and individual correlations sigma_ij
# (Q = symmetrised e_i e_j^T) all with the one primitive (§6 "collective-
# quadrature squeezing" and "specific correlations").
def target_quadratic_form(q, Q, modes=None, weight: float = 1.0, name: str = 'quadratic_form') -> TargetTerm:
    import jax.numpy as jnp
    Q_arr = jnp.array(Q)
    fn = lambda sigma: jnp.sum(Q_arr * sigma)
    return TargetTerm(fn, q, weight, modes, name=name)


# u^T sigma u target (single collective quadrature, §6) — the Q=u u^T case
# of target_quadratic_form, exposed directly since it's the common case
# (single-mode squeezing: u = one-hot on x or p).
def target_collective_quadrature(q, u, modes=None, weight: float = 1.0, name: str = 'quadrature_variance') -> TargetTerm:
    import jax.numpy as jnp
    u_arr = jnp.array(u)
    fn = lambda sigma: u_arr @ sigma @ u_arr
    return TargetTerm(fn, q, weight, modes, name=name)


# Logarithmic negativity target (§6), 2-mode (4x4) block.
def target_log_negativity(q, modes, weight: float = 1.0) -> TargetTerm:
    from reservoir_engineering.covariance_physics import jnp_log_negativity
    return TargetTerm(jnp_log_negativity, q, weight, modes, name='log_negativity')


# Purity target (§6): q=1.0 for an exactly pure target block.
def target_purity(q, modes=None, weight: float = 1.0) -> TargetTerm:
    from reservoir_engineering.covariance_physics import jnp_purity
    return TargetTerm(jnp_purity, q, weight, modes, name='purity')
