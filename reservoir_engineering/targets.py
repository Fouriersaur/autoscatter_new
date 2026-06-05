"""
targets.py
==========
Standard target covariance matrices for Gaussian quantum state engineering.

This module provides the RIGHT-HAND SIDE of the optimisation problem:
given one of these target covariances, the rest of the pipeline discovers
WHICH circuit topology achieves it.

═══════════════════════════════════════════════════════════════════════════════
CONVENTIONS
═══════════════════════════════════════════════════════════════════════════════

Real quadrature basis:  q = (x_0, p_0, x_1, p_1, ...)
Covariance matrix:      σ_ij = ½ ⟨{q_i, q_j}⟩

Vacuum (ground state) noise level:  σ_vac = ½ · I₂ per mode.
  x-quadrature variance = ½,  p-quadrature variance = ½.

Squeezed state:  one quadrature below ½ (squeezed), the other above ½.
Entangled state: off-diagonal correlations between modes.

Uncertainty principle (symplectic):  σ + i/2 · Ω ≥ 0
  where Ω = block-diag([[0, 1], [-1, 0]]) is the symplectic form.
  Equivalently: all symplectic eigenvalues of σ must be ≥ ½.
  Physical states satisfy this; unphysical targets will confuse the optimizer.

═══════════════════════════════════════════════════════════════════════════════
REFERENCE TARGET: KRONWALD SCHEME
═══════════════════════════════════════════════════════════════════════════════

The primary validation target throughout this package is:
    squeezed_vacuum(r)  for a single mechanical mode
This should be matched by a 2-mode system (cavity + mechanics) with BS + TMS
coupling in the Kronwald topology.  See kronwald_optimizer.py for the analytic
reference.  Use squeezed_vacuum(r) as the target for all validation tests.
"""

import numpy as np


# ───────────────────────────────────────────────────────────────────────────
# squeezed_vacuum(r) → np.ndarray shape (2, 2)
# ───────────────────────────────────────────────────────────────────────────
# Single-mode squeezed vacuum covariance.
#
# Physical meaning:
#   A single bosonic mode prepared in a vacuum-squeezed state with squeezing
#   parameter r.  The x-quadrature is squeezed below the vacuum level;
#   the p-quadrature is anti-squeezed above it.
#
# Formula:
#   σ = ½ · diag(e^{-2r},  e^{+2r})
#
#   r = 0:  vacuum state,  σ = ½ · I₂
#   r > 0:  squeezed in x,  σ_xx = ½ e^{-2r} < ½
#   r < 0:  squeezed in p,  σ_pp = ½ e^{+2r} < ½  (just relabelling)
#
# Squeezing in dB:  S = -10 log10(2 σ_xx) = 20 r / ln(10) ≈ 8.686 · r  dB
#
# Typical values used in experiments:
#   r = 0.5  →  ~4.3 dB squeezing
#   r = 1.0  →  ~8.7 dB squeezing
#   r = 1.5  →  ~13  dB squeezing  (challenging)

def squeezed_vacuum(r: float) -> np.ndarray:
    return 0.5 * np.diag([np.exp(-2*r), np.exp(2*r)])

# ───────────────────────────────────────────────────────────────────────────
# two_mode_squeezed(r) → np.ndarray shape (4, 4)
# ───────────────────────────────────────────────────────────────────────────
# Two-mode squeezed vacuum (TMSV / EPR state) covariance for modes 0 and 1.
#
# Physical meaning:
#   Two modes in a maximally entangled Gaussian state. As r → ∞ this
#   approaches the ideal Einstein–Podolsky–Rosen (EPR) state with perfect
#   position–position and momentum–momentum correlations.
#   Created by a parametric down-conversion process (TMS interaction).
#
# Formula (block structure in the (x_0, p_0, x_1, p_1) basis):
#
#   σ = ½ · [[ cosh(2r) · I₂    sinh(2r) · σ_z  ]
#             [ sinh(2r) · σ_z   cosh(2r) · I₂   ]]
#
#   where σ_z = diag(+1, -1) and I₂ = identity(2).
#
#   Explicitly:
#   σ_00 = σ_11 = ½ cosh(2r)  (both modes equally noisy individually)
#   σ_01[x,x] = +½ sinh(2r)   (positive x-x correlation)
#   σ_01[p,p] = -½ sinh(2r)   (negative p-p correlation → anti-correlated)
#
# Entanglement:
#   Duan criterion: violated (state is entangled) iff  r > 0.
#   Log negativity = r  (monotonically increasing with squeezing).
#
# Usage: target for 3-mode systems with 2 signal modes (cavity mediates).

def two_mode_squeezed(r: float) -> np.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# vacuum(n_modes=1) → np.ndarray shape (2*n_modes, 2*n_modes)
# ───────────────────────────────────────────────────────────────────────────
# Ground (vacuum) state covariance for n_modes uncorrelated modes.
#
# Formula:  σ = ½ · I_{2n_modes}
#
# This is the trivial target used for sanity-check optimisations:
# any dissipative system at T=0 without active squeezing should reach vacuum.
# If the optimizer cannot achieve this target, there is a bug in the physics.

def vacuum(n_modes: int = 1) -> np.ndarray:
    return 0.5 * np.eye(2 * n_modes)

# ───────────────────────────────────────────────────────────────────────────
# thermal(n_bar, n_modes=1) → np.ndarray shape (2*n_modes, 2*n_modes)
# ───────────────────────────────────────────────────────────────────────────
# Thermal state covariance for n_modes uncorrelated modes at occupation n̄.
#
# Formula:  σ = (n̄ + ½) · I_{2n_modes}
#
# n̄ = 0:  vacuum,  σ = ½ · I
# n̄ > 0:  added thermal noise
#
# Usage: used as the initial (before optimisation) state of the mechanical
# mode, and as a sanity-check target for the mechanical bath alone.

def thermal(n_bar: float, n_modes: int = 1) -> np.ndarray:
    return (n_bar + 0.5) * np.eye(2 * n_modes)

# ───────────────────────────────────────────────────────────────────────────
# cluster_state(n_modes, delta) → np.ndarray shape (2*n_modes, 2*n_modes)
# ───────────────────────────────────────────────────────────────────────────
# Approximate cluster (graph) state covariance for quantum computation.
# Only well-defined for a specific graph structure (chain, grid, etc.).
#
# Physical meaning:
#   Cluster states are universal resources for measurement-based quantum
#   computation. Gaussian cluster states are generated by CZ gates applied
#   to momentum-squeezed input modes. Here delta = squeezing parameter;
#   delta → 0 gives the ideal (infinite squeezing) cluster state.
#
# Construction for a LINEAR CHAIN of n_modes:
#   Start with each mode individually squeezed: σ_i = ½ diag(e^{+2δ}, e^{-2δ})
#   (squeezing in p, i.e. delta = r > 0 gives σ_pp < ½)
#   Apply CZ gates between neighbouring modes i and i+1:
#       CZ gate: symplectic matrix S_CZ = [[I, 0], [Γ, I]] where Γ is the
#       adjacency matrix of the graph (1 for connected pairs).
#   Final covariance:  σ_cluster = S_CZ · σ_product · S_CZ^T
#
# This is an advanced target — implement after the simpler squeezed_vacuum
# and two_mode_squeezed cases are working.
#
# Note: the circuit topology needed to ENGINEER a cluster state dissipatively
# is an interesting open question — this is what the searcher is for.

def cluster_state(n_modes: int, delta: float) -> np.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# is_physical(sigma) → bool
# ───────────────────────────────────────────────────────────────────────────
# Check whether a covariance matrix is physically valid.
#
# A covariance matrix σ is physical iff:
#     σ + i/2 · Ω ≥ 0    (all eigenvalues ≥ 0)
# where Ω is the 2N × 2N symplectic form:
#     Ω = block-diag([[0, 1], [-1, 0]])   (one 2×2 block per mode)
#
# Equivalently (and numerically more stable): compute the symplectic
# eigenvalues of σ and check all are ≥ ½.
# Symplectic eigenvalues = eigenvalues of |iΩσ| (absolute values of
# eigenvalues of the antisymmetric matrix iΩσ).
#
# Returns True if physical, False otherwise.
# Use this to validate user-defined target covariances before running
# the optimizer — an unphysical target will produce meaningless results.

def is_physical(sigma):
    return bool(np.all(symplectic_eigenvalues(sigma) >= 0.5 - 1e-10))

# ───────────────────────────────────────────────────────────────────────────
# symplectic_eigenvalues(sigma) → np.ndarray shape (N,)
# ───────────────────────────────────────────────────────────────────────────
# Compute the N symplectic eigenvalues of a 2N × 2N covariance matrix.
#
# These are the physical invariants of a Gaussian state — invariant under
# symplectic (canonical) transformations (rotations, squeezing in phase space).
#
# Method:
#   1. Build the symplectic form Ω (block-diag of [[0,1],[-1,0]]).
#   2. Compute the matrix product M = iΩσ.
#   3. Eigenvalues of M come in ±ν_k pairs (real ν_k > 0).
#   4. Return the N positive values ν_k, sorted ascending.
#
# Physical interpretation:
#   ν_k ≥ ½ for all k  ↔  state is physical (uncertainty principle)
#   ν_k = ½            ↔  mode k is a pure state (minimum uncertainty)
#   ν_k > ½            ↔  mode k is mixed (thermal or non-minimum uncertainty)

def symplectic_eigenvalues(sigma: np.ndarray) -> np.ndarray:
    N = sigma.shape[0] // 2
    Omega = np.zeros_like(sigma)
    # Generate the omega matrix
    for i in range(N):
        Omega[2*i, 2*i+1] = 1.
        Omega[2*i+1, 2*i] = -1.
    M = 1j * Omega @ sigma
    eigs = np.linalg.eigvals(M)

    return np.sprt(np.abs(eigs.real))[N:]

# ───────────────────────────────────────────────────────────────────────────
# squeezing_db(sigma, mode_id=0) → float
# ───────────────────────────────────────────────────────────────────────────
# Squeezing in decibels for a single mode.
#
# Formula:  S = -10 · log10(2 · σ_xx)
#   where σ_xx = sigma[2*mode_id, 2*mode_id]  (x-quadrature variance)
#
# S > 0 dB  →  squeezed below shot noise (σ_xx < ½)
# S = 0 dB  →  shot noise level (vacuum)
# S < 0 dB  →  above shot noise (thermal or anti-squeezed quadrature)
#
# Note: this measures squeezing in the x-quadrature specifically.
# To find the maximally squeezed quadrature, diagonalise the 2×2 block
# of the mode and take the minimum eigenvalue.

def squeezing_db(sigma: np.ndarray, mode_id: int = 0) -> float:
    pass


# ───────────────────────────────────────────────────────────────────────────
# log_negativity(sigma) → float
# ───────────────────────────────────────────────────────────────────────────
# Logarithmic negativity for a 2-mode Gaussian state (N=2 only).
# Quantifies the amount of entanglement.
#
# Method (Simon 2000 / Adesso et al.):
#   1. Compute the partial-transpose covariance matrix σ^PT by flipping
#      the sign of the p-quadrature of one mode:
#          σ^PT = T · σ · T    where T = diag(1, 1, 1, -1)
#   2. Find the symplectic eigenvalues ν̃_1 ≤ ν̃_2 of σ^PT.
#   3. If ν̃_1 < ½: the state is entangled (PPT criterion violated).
#   4. Log negativity: E_N = max(0,  -log2(2 ν̃_1))
#
# Returns:
#   E_N ≥ 0  (0 = separable, > 0 = entangled)
#
# Only defined for 2-mode states (4×4 covariance matrices).
# Raise ValueError for other sizes.

def log_negativity(sigma: np.ndarray) -> float:
    pass


# ───────────────────────────────────────────────────────────────────────────
# duan_criterion(sigma) → bool
# ───────────────────────────────────────────────────────────────────────────
# Duan–Simon inseparability criterion for 2-mode Gaussian states.
# Returns True if the state is entangled (criterion violated).
#
# The criterion: define the EPR-like operators
#     u = x_0 − x_1      v = p_0 + p_1
# The state is separable only if:
#     Var(u) + Var(p) ≥ 1   (in natural units where vacuum = ½ per quadrature)
#     i.e.  σ_xx(mode 0) + σ_xx(mode 1) - 2σ_{x0,x1}
#          + σ_pp(mode 0) + σ_pp(mode 1) + 2σ_{p0,p1}  ≥  1
#
# If this sum is < 1: state is entangled.
#
# Only valid for 2-mode states (4×4 covariance matrix).
# Simpler to compute than log_negativity; use as a quick entanglement check.

def duan_criterion(sigma: np.ndarray) -> bool:
    pass
