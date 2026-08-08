"""
linear_oracle.py
================
The certifying continuous oracle of state_stabilization_algorithm.md §2.1/§4.

Replaces the gradient-descent inner loop of covariance_optimizer.py with an
exact linear solve. The whole point: fix the target covariance V and the
stationarity equation A V + V A^T + D = 0 becomes LINEAR in the unknown
Hamiltonian matrix G and dissipation matrix Upsilon = C^dag C. A linear
problem is solved globally by one SVD — no initial guess, no restarts, no
local minima, and (unlike the old optimiser) a failure is a CERTIFICATE that
no Hamiltonian on this graph works, not a report that a search gave up.

INPUT IS A COVARIANCE MATRIX ONLY.
There is no target_predicate here, and this is structural rather than a
missing feature. The method's exactness comes from V entering (**) as a
KNOWN CONSTANT MATRIX; a scalar functional like log-negativity or purity
pins one number, not the matrix, so (**) stays underdetermined in V and the
unknown entries multiply the unknown G — bilinear, and every local minimum
the reformulation exists to remove comes straight back. Predicate targets
therefore need a different (bilinear/BMI) treatment; see the note at the
bottom of this docstring.

The equations (§1, §2 of the doc; identical to covariance_physics.py's
(G,C) machine, in this package's vacuum=1/2 convention):

    A = Omega (G + Im Upsilon)          D = Omega (Re Upsilon) Omega^T
    (**)  Omega(G + Im Y) V - V(G - Im Y) Omega + Omega (Re Y) Omega^T = 0

Every term is linear in G, Re Y, Im Y.

Pipeline per graph (see `decide`):

  1. STATE COMPLETION (§2.1). (**) needs the FULL 2n x 2n V, but the caller
     only ever specifies the signal block. `complete_covariance` recovers
     the rest exactly, for pure targets:
       - a pure marginal is uncorrelated with its complement, so V_TA = 0;
       - a local symplectic on the aux modes is a gauge (it redefines G and
         Upsilon, both of which are being solved for), so V_AA is Williamson
         diagonal; and a drain relaxing to its own (possibly squeezed)
         vacuum is pure, so every nu_m = 1/2.
     Hence V = V_target (+) (1/2) I. Not an assumption — see the doc.
     Mixed targets and Gamma_signal > 0 break this (V_TA != 0) and are out
     of scope; complete_covariance raises rather than silently linearising
     something that isn't linear.

  2. GENERATOR BASIS (§4). "G lives on the graph" is a LINEAR SUBSPACE
     S_G. Crucially the basis is over generator BLOCKS, not matrix entries:
     a beamsplitter and a two-mode-squeezing edge occupy the same (i,j)
     support but are different 2x2 blocks, so an entry-support subspace
     could not tell them apart. See `hamiltonian_basis`.

  3. LINEAR SOLVE (§4A). Assemble the design matrix column-by-column
     (M_b = Omega G_b V - V G_b Omega for each basis element), solve
     least squares by SVD. Nonzero residual => no Hamiltonian on this graph
     realises the target with this dissipator structure.

  4. ATTRACTIVITY FILTER (§4C). Residual zero is necessary, NOT sufficient:
     the solution set is affine and can contain frozen/dark-mode members
     whose A is only marginally stable (the minimum-norm point is often
     exactly such a spurious solution — G = 0, mode simply undriven).
     Search the solution set for a strictly Hurwitz member.

  5. THREE-VALUED VERDICT (§4C). VALID (explicit witness), INVALID
     (certificate), UNDECIDED (solutions exist but no Hurwitz member
     found). UNDECIDED propagates NOWHERE — this is what stops the one
     remaining nonconvex step (4) from manufacturing false invalids and
     poisoning a whole down-set.

Joint (G, Upsilon) mode — §4B without an SDP solver:
  `decide(..., joint=True)` makes Upsilon a free Hermitian unknown on the
  auxiliary block instead of fixing vacuum baths. That covers every
  Gaussian bath (thermal, squeezed, cross-correlated between drains) in one
  linear space. The PSD constraint Upsilon >= 0 is NOT imposed by the
  linear solve, which makes it a RELAXATION — and that is exactly what
  makes a nonzero residual a sound INVALID certificate, since a relaxation
  that is infeasible implies the true problem is infeasible. On the VALID
  side nothing is lost either: witnesses are verified numerically
  (Upsilon >= 0 AND A Hurwitz AND forward Lyapunov solve reproduces V), so
  a returned VALID is always genuine. cvxpy only sharpens the boundary
  between INVALID and UNDECIDED; if it is installed, `use_sdp=True` runs
  the real §4B feasibility program.

Not implemented here (deliberate, see doc §6): limit-only targets. At
Gamma_signal = 0 with a pure target this oracle returns the exact finite
witness that schemes like Kronwald only approximate; the asymptotic module
is a separate concern.
"""

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sciopt
from typing import List, Dict, Optional, Tuple

from reservoir_engineering.topology_search import (
    NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING, PARAMETRIC,
    BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)

# Verdict constants (§4C). Only VALID and INVALID may propagate through the
# search's up-/down-sets; UNDECIDED is reported and otherwise inert.
VALID     = 'VALID'
INVALID   = 'INVALID'
UNDECIDED = 'UNDECIDED'

# 2x2 blocks. Every real symmetric 2x2 is spanned by {I2, sx, sz}; every
# real 2x2 by {I2, J2, sx, sz}. J2 is the one-mode symplectic form.
_I2 = np.eye(2)
_J2 = np.array([[0., 1.], [-1., 0.]])
_SX = np.array([[0., 1.], [1., 0.]])
_SZ = np.array([[1., 0.], [0., -1.]])

# Residual is judged relative to the scale of the problem, never absolutely:
# V's entries scale like e^{2r} for a squeezing-r target, so a fixed 1e-14
# would read as "nonzero" on a strongly squeezed target purely from
# conditioning. See `_residual_is_zero`.
RESIDUAL_RTOL_DEFAULT = 1e-9

# Strictly-Hurwitz means max Re lambda < -MARGIN_TOL, not < 0: a dark mode
# sits at exactly 0 and floating point will place it at +-1e-16 either side.
# Anything inside the band is treated as marginal (i.e. NOT attractive).
MARGIN_TOL_DEFAULT = 1e-9


# symplectic_form(n): Omega = block-diag(J2), per-mode (xpxp) ordering.
# Duplicated from covariance_physics rather than imported so this module
# stays numpy-only (no JAX import cost on the search's hot path).
def symplectic_form(n: int) -> np.ndarray:
    Omega = np.zeros((2 * n, 2 * n))
    for i in range(n):
        Omega[2 * i, 2 * i + 1] = 1.
        Omega[2 * i + 1, 2 * i] = -1.
    return Omega


# ───────────────────────────────────────────────────────────────────────────
# §2.1 state completion
# ───────────────────────────────────────────────────────────────────────────
# complete_covariance(sigma_target, target_mode_ids, num_modes) -> full V.
#
# The caller specifies sigma_target on the signal modes only; (**) needs the
# whole 2n x 2n matrix. Recovered exactly (doc §2.1):
#   V_TA = 0     because a PURE marginal is uncorrelated with everything
#                else (rho_T pure => rho_TA = rho_T (x) rho_A, true even
#                when the global state is mixed)
#   V_AA         a PURE state of the auxiliary modes, since a drain
#                relaxing to its own vacuum/squeezed-vacuum bath is pure.
#                Defaults to the vacuum (1/2)I.
#
# CORRECTION to state_stabilization_algorithm.md §2.1 Step 2(i), found by
# running this oracle. The doc argues that a local symplectic on the aux
# modes is a gauge, so V_AA may be taken Williamson-diagonal and hence
# (1/2)I. THAT ARGUMENT IS ONLY VALID FOR THE UNCONSTRAINED PROBLEM. Under
# a graph constraint the aux symplectic does not preserve S_G: squeezing
# auxiliary mode m turns a pure beamsplitter edge (0,i) into a
# beamsplitter+two-mode-squeezing edge, i.e. it moves you to a DIFFERENT
# graph. So V_AA = (1/2)I is a gauge FIXING, not a free choice, and it is
# only complete up to that relabelling.
#
# Both readings are legitimate and this function supports both:
#   aux_squeezing = None (default) — every drain unsqueezed, V_AA = (1/2)I.
#     Complete: every physical scheme still appears somewhere in the
#     search, but in the frame where drains carry no squeezing. A passive
#     Zippilli-Vitali scheme (BS-only network + squeezed drain) therefore
#     shows up as its gauge image, with BS+TMS on the drain edges.
#     Numerically confirmed on the two-mode-squeezed target: the passive
#     triangle {(0,1)BS,(0,2)BS,(1,2)BS} with a squeezed drain and
#     {(0,1)BS+TMS,(0,2)BS+TMS,(1,2)BS} with a vacuum drain are the same
#     scheme seen in two gauges.
#   aux_squeezing = [(r_m, theta_m), ...] — drain m prepared in a squeezed
#     vacuum. Use this to recover the Zippilli-Vitali LABELLING, in which
#     the passive network is the answer and the squeezing is supplied by
#     the bath. `scan_aux_squeezing` sweeps it for you; the drain squeezing
#     that works is not arbitrary (for a squeezing-r target it is r), so
#     the sweep is short and sharply peaked.
#
# nu_aux is the doc's separate fallback for when the drain-PURITY argument
# is unavailable: pass symplectic eigenvalues nu_m >= 1/2 for mixed drains.
# Verdicts are then pointwise in nu — a grid-wide failure is UNDECIDED,
# never INVALID.
#
# Everything here is general in the number of auxiliary modes, with ONE
# caveat about coverage that only bites for two or more drains. The
# (nu_aux, aux_squeezing) parametrisation builds a PRODUCT state over the
# drains: 2 free parameters each, 2m in total. Pure Gaussian states of m
# modes form Sp(2m,R)/U(m), of dimension m(m+1) — equal to 2m only at
# m = 1. So for m >= 2 this parametrisation misses drain-drain CORRELATED
# states (m(m-1) directions, e.g. two drains sharing a two-mode squeezed
# state). Pass aux_cov to supply such a state directly.
#
# This is a coverage gap in the aux-state SCAN only, never in the default
# search: at aux_squeezing=None the gauge-fixing argument is what carries
# completeness, and it holds for any m — every pure V_AA equals S S^T/2 for
# some symplectic S on the auxiliary modes, and applying S^-1 maps it to
# (1/2)I while relabelling the graph. So an arbitrary-m search with the
# default drains still sees every scheme, just in the unsqueezed-drain
# frame.
#
# aux_cov: explicit 2m x 2m auxiliary block, overriding nu_aux and
# aux_squeezing. Use for correlated or otherwise non-product drain states.
#
# Raises on a mixed target: with V_TT impure, V_TA != 0 and (**) is
# bilinear, so linearising would be silently wrong rather than merely
# approximate.
def complete_covariance(
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    num_modes: int,
    nu_aux: Optional[List[float]] = None,
    aux_squeezing: Optional[List[Tuple[float, float]]] = None,
    aux_cov: Optional[np.ndarray] = None,
    purity_tol: float = 1e-8,
) -> np.ndarray:
    sigma_target = np.asarray(sigma_target, dtype=float)
    n_sig = len(target_mode_ids)
    if sigma_target.shape != (2 * n_sig, 2 * n_sig):
        raise ValueError(f"sigma_target is {sigma_target.shape}, expected "
                         f"{(2 * n_sig, 2 * n_sig)} for {n_sig} target modes")

    # Purity in this package's convention: pure <=> det(2 sigma) = 1.
    det2 = float(np.linalg.det(2 * sigma_target))
    if abs(det2 - 1.) > purity_tol:
        raise ValueError(
            f"linear_oracle requires a PURE target (det(2 sigma) = 1); got "
            f"det(2 sigma) = {det2:.12g}, purity = {1/np.sqrt(det2):.12g}. "
            "A mixed target has V_TA != 0, so the full covariance is not "
            "determined and (**) is bilinear, not linear — see "
            "state_stabilization_algorithm.md §2.1 'Scope boundary'. Use "
            "covariance_optimizer.CovarianceOptimizer for mixed targets.")

    aux_ids = [i for i in range(num_modes) if i not in target_mode_ids]
    if nu_aux is None:
        nu_aux = [0.5] * len(aux_ids)
    if len(nu_aux) != len(aux_ids):
        raise ValueError(f"nu_aux has {len(nu_aux)} entries, expected {len(aux_ids)}")
    if any(nu < 0.5 - 1e-12 for nu in nu_aux):
        raise ValueError("nu_aux entries must be >= 1/2 (uncertainty principle)")
    if aux_squeezing is None:
        aux_squeezing = [(0.0, 0.0)] * len(aux_ids)
    if len(aux_squeezing) != len(aux_ids):
        raise ValueError(f"aux_squeezing has {len(aux_squeezing)} entries, expected {len(aux_ids)}")

    V = np.zeros((2 * num_modes, 2 * num_modes))
    idx = []
    for m in target_mode_ids:
        idx += [2 * m, 2 * m + 1]
    V[np.ix_(idx, idx)] = sigma_target

    if aux_cov is not None:
        # Explicit auxiliary block — the escape hatch for drain states this
        # product parametrisation cannot express (correlated drains, m >= 2).
        aux_cov = np.asarray(aux_cov, dtype=float)
        d = 2 * len(aux_ids)
        if aux_cov.shape != (d, d):
            raise ValueError(f"aux_cov is {aux_cov.shape}, expected {(d, d)} "
                             f"for {len(aux_ids)} auxiliary modes")
        aidx = []
        for m in aux_ids:
            aidx += [2 * m, 2 * m + 1]
        V[np.ix_(aidx, aidx)] = aux_cov
        return V

    for slot, m in enumerate(aux_ids):
        r, th = aux_squeezing[slot]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        V[2 * m:2 * m + 2, 2 * m:2 * m + 2] = (
            nu_aux[slot] * R @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ R.T)
    return V


# ───────────────────────────────────────────────────────────────────────────
# §4 generator bases (the admissible subspaces S_G, S_Upsilon)
# ───────────────────────────────────────────────────────────────────────────
# hamiltonian_basis(triu_array, num_modes, ...) -> list of (label, G_b).
#
# S_G is spanned by GENERATOR BLOCKS, not by matrix entries. This matters:
# a beamsplitter edge and a two-mode-squeezing edge on the same pair (i,j)
# have identical entry support but different 2x2 blocks, so an
# entry-support subspace could not distinguish them and the graph encoding
# of topology_search.py would collapse.
#
# The blocks are derived from covariance_physics.build_hamiltonian_matrix,
# which returns K = Omega G rather than G; inverting with Omega^-1 = -Omega:
#     BS  real    K[i,j] =  a J2, K[j,i] =  a J2   ->  G blocks  I2,  I2
#     BS  imag    K[i,j] =  b I2, K[j,i] = -b I2   ->  G blocks -J2,  J2
#     TMS         K[i,j] = -g sx, K[j,i] = -g sx   ->  G blocks  sz,  sz
#     parametric  K[i,i] =  x sz                   ->  G block   sx
#     detuning    K[i,i] =  d J2                   ->  G block   I2
# each of which is symmetric as G must be.
#
# Two deliberate generalisations over the old optimiser, both instances of
# the doc's "let the Hamiltonian search range freely":
#   - TMS gets a free PHASE (a second generator, sx), exactly as
#     beamsplitters already do in covariance_optimizer. {sz, sx} spans the
#     ACTIVE part of an off-diagonal block and {I2, J2} the PASSIVE part
#     ([G, Omega] = 0 iff spanned by {I2, J2} — checkable directly), so
#     BS + TMS on a pair spans the FULL 2x2 block, 4 real dof.
#   - parametric gets both squeezing quadratures {sx, sz}; with the
#     detuning I2 that spans every symmetric 2x2, i.e. the full single-mode
#     quadratic Hamiltonian.
# Set allow_phases=False to pin both back to the old real-only generators.
#
# Sanity check on completeness: for the fully connected graph on n modes
# the basis has 4*C(n,2) + 3n = n(2n+1) elements, which is exactly
# dim{symmetric 2n x 2n} — as it must be, since a fully connected graph
# with detunings imposes no restriction on G at all.
def hamiltonian_basis(
    triu_array,
    num_modes: int,
    include_detunings: bool = True,
    allow_phases: bool = True,
) -> List[Tuple[str, np.ndarray]]:
    rows, cols = np.triu_indices(num_modes)
    basis = []

    def add(label, blocks):
        G = np.zeros((2 * num_modes, 2 * num_modes))
        for (a, b, blk) in blocks:
            G[2 * a:2 * a + 2, 2 * b:2 * b + 2] += blk
        basis.append((label, G))

    if include_detunings:
        for i in range(num_modes):
            add(f'delta_{i}', [(i, i, _I2)])

    for k, (i, j) in enumerate(zip(rows, cols)):
        val = int(triu_array[k])
        if val == NO_COUPLING:
            continue

        if i == j:
            if val == PARAMETRIC:
                add(f'par_{i}', [(i, i, _SX)])
                if allow_phases:
                    add(f'par90_{i}', [(i, i, _SZ)])
            continue

        wants_bs  = val in (BEAMSPLITTER, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)
        wants_tms = val in (TWO_MODE_SQUEEZING, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)

        if wants_bs:
            add(f'bs_{i}{j}', [(i, j, _I2), (j, i, _I2)])
            if allow_phases:
                add(f'bs90_{i}{j}', [(i, j, -_J2), (j, i, _J2)])
        if wants_tms:
            add(f'tms_{i}{j}', [(i, j, _SZ), (j, i, _SZ)])
            if allow_phases:
                add(f'tms90_{i}{j}', [(i, j, _SX), (j, i, _SX)])

    return basis


# dissipation_basis(aux_node_ids, num_modes, coupled_drains) -> list of
# (label, Re_b, Im_b), a real basis for Hermitian Upsilon supported on the
# auxiliary block.
#
# Upsilon = C^dag C is Hermitian PSD; splitting Upsilon = S + iT gives S
# real symmetric and T real antisymmetric, so a real basis of the Hermitian
# matrices on a d-dimensional support has d(d+1)/2 + d(d-1)/2 = d^2
# elements. Spanning ALL of it (not just rank-1 vacuum rows) is what makes
# the joint mode cover every Gaussian bath at once — thermal, squeezed,
# and cross-correlated between drains — with no discrete palette to choose
# from and no separate squeeze variables.
#
# The support is the AUXILIARY modes only: §2.1's completion assumes the
# signal modes have no bath of their own (Gamma_signal = 0), which is the
# same idealisation that makes V_TA = 0 and the whole method exact.
#
# coupled_drains=False (default) restricts to one independent bath per
# auxiliary mode (block-diagonal Upsilon) — the physically ordinary case,
# each drain wired to its own reservoir. True allows a shared reservoir
# correlating different drains, a strictly larger subspace.
def dissipation_basis(
    aux_node_ids: List[int],
    num_modes: int,
    coupled_drains: bool = False,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    d = 2 * num_modes
    support = []
    for m in aux_node_ids:
        support += [2 * m, 2 * m + 1]

    def blocks_of(a, b):
        return (a // 2, b // 2)

    basis = []
    for ia, a in enumerate(support):
        for b in support[ia:]:
            if not coupled_drains and blocks_of(a, b)[0] != blocks_of(a, b)[1]:
                continue
            S = np.zeros((d, d)); T = np.zeros((d, d))
            S[a, b] = 1.; S[b, a] = 1.
            basis.append((f'ReY_{a}{b}', S.copy(), T.copy()))
            if a != b:
                S2 = np.zeros((d, d)); T2 = np.zeros((d, d))
                T2[a, b] = 1.; T2[b, a] = -1.
                basis.append((f'ImY_{a}{b}', S2, T2))
    return basis


# vacuum_dissipation(aux_node_ids, num_modes, rate) -> a FIXED Upsilon: one
# plain vacuum-loss channel per auxiliary mode.
#
# From covariance_physics.build_jump_matrix, a cavity vacuum bath is
# c = sqrt(kappa/2)[1, i], so its block of Upsilon = sum_mu c* c^T is
# (kappa/2) [[1, i], [-i, 1]] — Hermitian, rank 1 (eigenvalues 0 and
# kappa), and reproducing the familiar -kappa/2 decay and kappa/2 vacuum
# diffusion.
#
# `rate` is arbitrary and defaults to 1: by §1(b) the physics is invariant
# under the JOINT rescaling (G, Upsilon) -> (sG, sUpsilon), so fixing the
# drain rate merely fixes that gauge and the solved G comes out scaled to
# match. It is NOT the case that V ignores absolute rates — rescaling
# Upsilon alone would move V.
def vacuum_dissipation(aux_node_ids: List[int], num_modes: int,
                        rate: float = 1.0) -> np.ndarray:
    d = 2 * num_modes
    Y = np.zeros((d, d), dtype=complex)
    blk = (rate / 2.) * np.array([[1., 1j], [-1j, 1.]])
    for m in aux_node_ids:
        Y[2 * m:2 * m + 2, 2 * m:2 * m + 2] += blk
    return Y


# ───────────────────────────────────────────────────────────────────────────
# §4A the linear system
# ───────────────────────────────────────────────────────────────────────────
# lyapunov_residual_matrix(G, Y, V): the left-hand side of (**). Zero iff
# (G, Upsilon) makes V stationary. Used both to assemble the design matrix
# (one basis element at a time) and to check a finished witness.
def lyapunov_residual_matrix(G: np.ndarray, Y: np.ndarray, V: np.ndarray) -> np.ndarray:
    n = V.shape[0] // 2
    Om = symplectic_form(n)
    ReY, ImY = np.real(Y), np.imag(Y)
    return (Om @ (G + ImY) @ V - V @ (G - ImY) @ Om + Om @ ReY @ Om.T)


# drift_matrix(G, Y): A = Omega (G + Im Upsilon).
def drift_matrix(G: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = G.shape[0] // 2
    return symplectic_form(n) @ (G + np.imag(Y))


# diffusion_matrix(Y): D = Omega (Re Upsilon) Omega^T.
def diffusion_matrix(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0] // 2
    Om = symplectic_form(n)
    return Om @ np.real(Y) @ Om.T


# build_linear_system(V, h_basis, d_basis, Y_fixed) -> (M, b, split).
#
# Assembles (**) as M z = b with z the stacked coefficients of the
# Hamiltonian basis and (if d_basis is given) the dissipation basis.
#
# Built COLUMN BY COLUMN — one call to lyapunov_residual_matrix per basis
# element — rather than via the Kronecker form 𝕃 = (V (x) Omega) +
# (Omega (x) V) of the doc. Algebraically identical, but it sidesteps the
# row-major/column-major vec convention entirely (numpy flattens row-major,
# the Kronecker identity assumes column-major) and it restricts to S_G
# automatically instead of needing a separate selection matrix P.
#
# All (2n)^2 entries are kept as rows even though (**) is symmetric and only
# n(2n+1) of them are independent. The duplicates are harmless for least
# squares (they reweight the off-diagonals uniformly) and cost nothing at
# these sizes.
def build_linear_system(
    V: np.ndarray,
    h_basis: List[Tuple[str, np.ndarray]],
    d_basis: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
    Y_fixed: Optional[np.ndarray] = None,
):
    d = V.shape[0]
    zero_Y = np.zeros((d, d), dtype=complex)
    cols = []

    for _, Gb in h_basis:
        cols.append(lyapunov_residual_matrix(Gb, zero_Y, V).ravel())
    n_h = len(h_basis)

    n_d = 0
    if d_basis:
        for _, Sb, Tb in d_basis:
            cols.append(lyapunov_residual_matrix(np.zeros((d, d)), Sb + 1j * Tb, V).ravel())
        n_d = len(d_basis)

    M = np.column_stack(cols) if cols else np.zeros((d * d, 0))

    # Everything not carried by a free basis element goes to the RHS.
    if Y_fixed is not None:
        b = -lyapunov_residual_matrix(np.zeros((d, d)), Y_fixed, V).ravel()
    else:
        b = np.zeros(d * d)

    return M, b, (n_h, n_d)


# _residual_is_zero: scale-aware test. V's entries grow like e^{2r} for a
# squeezing-r target, so an absolute threshold would flag strongly squeezed
# targets as infeasible purely from conditioning.
def _residual_is_zero(resid_norm: float, M: np.ndarray, b: np.ndarray,
                       rtol: float = RESIDUAL_RTOL_DEFAULT) -> bool:
    scale = max(np.linalg.norm(b), np.linalg.norm(M), 1.0)
    return resid_norm <= rtol * scale


# solve_stationarity(V, h_basis, d_basis, Y_fixed) -> dict describing the
# affine solution set of (**) restricted to the admissible subspace:
#   'feasible'   — residual is zero (a stationary solution exists)
#   'residual'   — the least-squares residual norm (the §4A certificate)
#   'z0'         — minimum-norm particular solution
#   'nullspace'  — orthonormal basis of the remaining freedom
# The SVD makes this global: there is no initial guess and no local minimum
# to get stuck in, which is the entire reason for the reformulation.
def solve_stationarity(
    V: np.ndarray,
    h_basis,
    d_basis=None,
    Y_fixed=None,
    rtol: float = RESIDUAL_RTOL_DEFAULT,
) -> Dict:
    M, b, split = build_linear_system(V, h_basis, d_basis, Y_fixed)

    if M.shape[1] == 0:
        resid = float(np.linalg.norm(b))
        return {'feasible': _residual_is_zero(resid, M, b, rtol), 'residual': resid,
                'z0': np.zeros(0), 'nullspace': np.zeros((0, 0)), 'split': split,
                'M': M, 'b': b}

    z0, _, _, sv = np.linalg.lstsq(M, b, rcond=None)
    resid = float(np.linalg.norm(M @ z0 - b))

    # Nullspace from the SVD, with a rank tolerance tied to the largest
    # singular value (numpy's lstsq residuals array is unreliable here since
    # it is empty for rank-deficient systems, which is exactly our case).
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    tol = max(M.shape) * np.finfo(float).eps * (s[0] if s.size else 0.)
    rank = int(np.sum(s > tol))
    N = Vt[rank:].T if rank < M.shape[1] else np.zeros((M.shape[1], 0))

    return {'feasible': _residual_is_zero(resid, M, b, rtol), 'residual': resid,
            'z0': z0, 'nullspace': N, 'split': split, 'M': M, 'b': b,
            'singular_values': sv}


# ───────────────────────────────────────────────────────────────────────────
# §4C attractivity filter and the three-valued verdict
# ───────────────────────────────────────────────────────────────────────────
# _assemble(z, h_basis, d_basis, Y_fixed) -> (G, Upsilon) from coefficients.
def _assemble(z, h_basis, d_basis, Y_fixed, dim):
    n_h = len(h_basis)
    G = np.zeros((dim, dim))
    for k, (_, Gb) in enumerate(h_basis):
        G = G + z[k] * Gb
    Y = np.zeros((dim, dim), dtype=complex) if Y_fixed is None else Y_fixed.copy()
    if d_basis:
        for k, (_, Sb, Tb) in enumerate(d_basis):
            Y = Y + z[n_h + k] * (Sb + 1j * Tb)
    return G, Y


# normalised_margin(A): -max Re lambda(A) / ||A||.
#
# Must be normalised, not raw. In joint mode the solution set is a CONE —
# by §1(b) the physics is invariant under the joint rescaling
# (G, Upsilon) -> (sG, sUpsilon) — so the raw margin can be driven to
# anything by inflating ||z||, and an unnormalised search happily returns
# margins of 1e159 that mean nothing. Dividing by ||A|| makes the criterion
# scale-free, so the filter answers the question that actually matters
# ("is this direction Hurwitz?") rather than "how big can I make it?".
def normalised_margin(A: np.ndarray) -> float:
    nrm = float(np.linalg.norm(A))
    if nrm < 1e-300:
        return 0.0
    return stability_margin(A) / nrm


# stability_margin(A): -max Re lambda(A). Positive iff strictly Hurwitz;
# by §3 a genuine solution can never have this NEGATIVE (a positive-definite
# V solving the Lyapunov equation forces Re lambda <= 0 for every
# eigenvalue), so the only real risk is a margin of exactly zero — a dark
# mode. The filter is therefore a search for strict negativity among
# candidates already known to be at worst marginal.
def stability_margin(A: np.ndarray) -> float:
    return float(-np.max(np.real(np.linalg.eigvals(A))))


# _is_psd(Y): Upsilon >= 0, the physical-realisability condition (any
# Hermitian PSD Upsilon factorises as C^dag C, and its rank is the channel
# count). Trivially true when Upsilon is fixed to vacuum rows; a real check
# in joint mode, where the linear solve does not impose it.
def _is_psd(Y: np.ndarray, tol: float = 1e-9) -> bool:
    w = np.linalg.eigvalsh((Y + Y.conj().T) / 2)
    return bool(np.min(w) >= -tol * max(1.0, float(np.max(np.abs(w)))))


# attractivity_filter(sol, ...) -> (found, z, margin), §4C / §8 Move 1.
#
# Searches the affine solution set z0 + null(M) for a member with strictly
# Hurwitz A (and, in joint mode, Upsilon >= 0). Strategy: try the
# minimum-norm point, then K random draws, then a direct maximisation of
# the stability margin over the nullspace coefficients.
#
# A FAILURE HERE IS NOT EVIDENCE OF INVALIDITY. Callers must map it to
# UNDECIDED, never INVALID — that asymmetry is what confines this one
# nonconvex step to a completeness gap on a single graph instead of letting
# it poison the whole down-set through the search's propagation rules.
#
# `seed` is derived from the graph by the caller so the verdict is
# reproducible: §5(iii)'s bidirectional-agreement assertion compares
# partitions across two traversals and needs the filter to be deterministic.
#
# maximise=False (default) is the DECISION mode: return as soon as any
# member clears margin_tol, since one witness settles validity and stopping
# early is the whole point of an existential test.
#
# maximise=True is §8 Move 1's GAP MAXIMISER,
#     gamma* = max over the feasible set of [ -max_i Re lambda_i(A) ],
# which does not stop at the first witness but keeps ascending the spectral
# abscissa. Two reasons the doc gives for wanting it, both real:
#   (a) COMPLETENESS. If a graph is valid its Hurwitz members form a
#       relatively open, positive-measure subset of the solution set (A
#       depends continuously on (G,Upsilon) and the Hurwitz set is open), so
#       a genuine interior ascent finds a witness whenever one exists, while
#       an early-exit sampler can miss a thin feasible cone. This is the
#       move that converts UNDECIDED graphs to VALID.
#   (b) FIGURE OF MERIT. gamma* IS the dissipative gap, so the oracle that
#       decides validity also returns the relaxation rate the resource-vs-gap
#       comparison needs — no separate optimisation.
# The prediction in (a) is empirically untested at scale; it is flagged as
# such in the doc and should be reported as a proposal, not a theorem about
# this implementation.
def attractivity_filter(
    sol: Dict,
    h_basis,
    d_basis,
    Y_fixed,
    dim: int,
    num_samples: int = 32,
    sample_scale: float = 1.0,
    seed: int = 0,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    require_psd: bool = False,
    n_polish: int = 4,
    maximise: bool = False,
) -> Tuple[bool, Optional[np.ndarray], float]:
    z0, N = sol['z0'], sol['nullspace']
    n_null = N.shape[1] if N.size else 0

    # The ACCEPTANCE test: strictly Hurwitz, and physically realisable.
    def margin_of(z):
        G, Y = _assemble(z, h_basis, d_basis, Y_fixed, dim)
        if require_psd and not _is_psd(Y):
            return -np.inf
        return normalised_margin(drift_matrix(G, Y))

    # The SEARCH objective, which must not be the acceptance test. Rejecting
    # a PSD-violating point with -inf makes the whole infeasible region flat,
    # so a simplex method started outside Upsilon >= 0 has no signal telling
    # it which way the feasible set is and simply stalls. Penalising the
    # violation continuously instead lets the search WALK INTO the feasible
    # region from outside.
    #
    # This is not a nicety. On the passive Zippilli-Vitali topologies —
    # a beamsplitter-only network stabilising a cluster state through one
    # squeezed drain — the physical region is a thin cone inside the
    # solution space: random Gaussian sampling of a 7-dimensional nullspace
    # scored 0 PSD hits out of 2000. With -inf rejection the filter reported
    # UNDECIDED; with the penalty it finds a witness at normalised margin
    # 0.0279, better than any active scheme found by descent. A whole class
    # of schemes was invisible for want of this.
    psd_weight = 1e3

    def score(z):
        G, Y = _assemble(z, h_basis, d_basis, Y_fixed, dim)
        m = normalised_margin(drift_matrix(G, Y))
        if require_psd:
            w = np.linalg.eigvalsh((Y + Y.conj().T) / 2)
            scale_w = max(1.0, float(np.max(np.abs(w))))
            m = m - psd_weight * max(0., -float(np.min(w))) / scale_w
        return m

    best_z, best_m = z0, margin_of(z0)
    if best_m > margin_tol and not maximise:
        return True, z0, best_m

    if n_null == 0:
        return (best_m > margin_tol), best_z, best_m

    rng = np.random.default_rng(seed)
    scale = sample_scale * max(1.0, float(np.linalg.norm(z0)))
    best_score, best_score_z = score(z0), z0
    for _ in range(num_samples):
        z = z0 + N @ rng.normal(0., scale, n_null)
        m = margin_of(z)
        if m > margin_tol and not maximise:
            return True, z, m
        if m > best_m:
            best_z, best_m = z, m
        s = score(z)
        if s > best_score:
            best_score, best_score_z = s, z

    # Nonconvex polish over the nullspace coefficients (small: n_null is
    # typically a handful). Nelder-Mead because the margin is a max over
    # eigenvalue real parts and so only piecewise smooth. Restarted from
    # several seeds, since the feasible cone can be missed from a single
    # start even with the penalty guiding the way in.
    def neg_score(alpha):
        s = score(z0 + N @ alpha)
        return 1e9 if not np.isfinite(s) else -s

    starts = [np.linalg.lstsq(N, best_score_z - z0, rcond=None)[0]]
    if maximise:
        # Also restart from the best MARGIN point, not only the best
        # penalised score: once the ascent is allowed to continue past the
        # first witness these are different points, and the gap maximiser
        # wants the one already highest on the abscissa.
        starts.append(np.linalg.lstsq(N, best_z - z0, rcond=None)[0])
    starts += [rng.normal(0., 1., n_null) for _ in range(max(0, n_polish - 1))]
    for alpha0 in starts:
        res = sciopt.minimize(neg_score, alpha0, method='Nelder-Mead',
                              options={'maxiter': 300 * max(1, n_null),
                                       'xatol': 1e-10, 'fatol': 1e-12})
        z = z0 + N @ res.x
        m = margin_of(z)
        if m > best_m:
            best_z, best_m = z, m
        if m > margin_tol and not maximise:
            return True, z, m
    return (best_m > margin_tol), best_z, best_m


# ───────────────────────────────────────────────────────────────────────────
# structural certificates — the sound INVALID verdicts
# ───────────────────────────────────────────────────────────────────────────
# AMENDMENT to state_stabilization_algorithm.md §4A/§4B, found by running
# the oracle: the doc's residual-based INVALID certificate CANNOT FIRE at
# Gamma_signal = 0, so on its own it prunes nothing.
#
# Why. With no bath on the signal modes, (G, Upsilon) = (0, 0) gives A = 0
# and D = 0, so (**) holds for ANY V whatsoever — the "everything frozen"
# configuration is stationary for every target on every graph. In the joint
# path this is worse than a nuisance: the linear system is HOMOGENEOUS
# (b = 0, since nothing is moved to the right-hand side), so z = 0 solves
# it exactly and the least-squares residual is identically zero no matter
# what the graph or target are. Verified numerically. The doc notices the
# frozen point in §4A — it is the "minimum-norm least-squares point is
# G = 0" example — but does not notice that its existence makes the
# residual test vacuous as a GRAPH-level verdict.
#
# The correct reading: a zero residual carries no information here, and the
# real question is entirely the attractivity one — does the solution set
# contain a Hurwitz member? That is not a least-squares residual and not an
# SDP; deciding whether an affine matrix family contains a Hurwitz member
# is hard in general. So INVALID has to come from somewhere else.
#
# Below are two certificates that are cheap, exact, and provable. They are
# graph-structural rather than numerical, which is why they can be trusted
# to prune. Everything they do not catch is UNDECIDED — the search is then
# slower but never wrong, since UNDECIDED propagates nowhere.
#
# structural_certificate(...) -> None if nothing is proved, else a dict
# describing the proof.
#
# (a) UNDAMPED COMPONENT. All dissipation lives on the auxiliary modes
#     (Upsilon is supported there — this is the same Gamma_signal = 0
#     idealisation that makes §2.1's state completion exact). If a
#     connected component of the coupling graph contains a signal mode but
#     no auxiliary mode, then on that component D = 0 and A = Omega G with
#     G symmetric, so tr A = tr(Omega G) = 0 — the trace of an
#     antisymmetric times a symmetric matrix. Eigenvalues summing to zero
#     cannot all have negative real part, so A is not Hurwitz for ANY
#     admissible G. No Hamiltonian on this graph stabilises anything there.
#
# (b) CORRELATION ACROSS A CUT. Distinct components have no coupling and
#     (with one independent reservoir per drain) no shared bath, so their
#     dynamics and noise are independent and the steady state factorises:
#     the cross-covariance between modes in different components is zero.
#     If V_target demands a nonzero correlation between two signal modes
#     that this graph places in different components, no solution exists.
#     This is the certificate that does the real pruning work on entangled
#     targets — EPR, cluster states — where most sparse graphs fail it.
#
#     Gated on coupled_drains=False: a reservoir shared between two drains
#     could correlate their components, and then the argument fails.
# _orthonormal / _nullspace / _restrict: small subspace helpers for the
# undamped-subspace certificate below. Columns are always kept orthonormal
# so that C @ C.T is the orthogonal projector onto the subspace.
# `ref` is the scale the singular values are judged against. It MUST be
# passed whenever the matrix can legitimately be zero, which is exactly the
# case in the invariant-subspace iterations below: there the block being
# nulled is (I - P) M W, and it vanishes precisely when W is already
# invariant under M — the success case. Judging it against its OWN largest
# singular value then compares 1e-16 with 1e-16, declares full rank, and
# collapses the subspace to nothing, so the certificate silently fails on
# the graphs it is meant to catch.
#
# That is not hypothetical: with the tolerance self-referenced, the same
# physical problem gave 156 INVALID with the drain labelled mode 2 and only
# 80 with it labelled mode 0 — 92 graphs whose verdict flipped under a
# relabelling that cannot change the physics. Passing the generator norm as
# `ref` makes the test scale-correct and the verdict permutation-invariant.
def _orthonormal(X: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    if X.size == 0:
        return X
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    rank = int(np.sum(s > rtol * max(float(s[0]), 1e-300)))
    return U[:, :rank]


def _nullspace(M: np.ndarray, rtol: float = 1e-10,
                ref: Optional[float] = None) -> np.ndarray:
    if M.size == 0:
        return np.eye(M.shape[1] if M.ndim == 2 else 0)
    _, s, Vt = np.linalg.svd(M)
    scale = float(s[0]) if (ref is None and s.size) else (ref or 0.0)
    rank = int(np.sum(s > rtol * max(scale, 1e-300)))
    return Vt[rank:].T


# undamped_subspace(triu, num_modes, aux_ids) -> basis for the largest
# subspace W such that, for EVERY admissible (G, Upsilon) on this graph,
#     A(G,Upsilon)^T W subset W    and    D(Upsilon) W = 0.
#
# §4A's "a mode (or Bogoliubov combination) left undamped — unreachable by
# any dissipative channel through the graph", in its general form.
#
# Why a nonzero W certifies invalidity. Pick w in W an eigenvector of the
# restriction of A^T to W (it exists over C). Then w is a left eigenvector
# of A, and D w = 0, so by §3's identity 2 Re(lambda) w^dag V w =
# -w^dag D w = 0, giving Re lambda = 0. A is therefore non-Hurwitz for
# every admissible choice, so no attractive solution exists on this graph.
# Sound with no numerical solve at all — it is a property of the generator
# subspaces, not of any particular solution.
#
# The computation is the standard largest-invariant-subspace iteration:
# start from the directions no diffusion generator can reach, then
# repeatedly discard anything the drift generators push out, until stable.
#
# This SUBSUMES the whole-mode connectivity argument (a disconnected
# signal component's coordinates form such a W) while also catching
# collective/Bogoliubov dark modes that no per-mode reachability test can
# see. Measured on the two-mode-squeezed target it certifies exactly the
# same 80 graphs as the connectivity argument — there the dark modes of
# the remaining graphs are parameter-DEPENDENT, so no common W exists —
# but it is the correct general statement and costs microseconds.
def undamped_subspace(triu_array, num_modes: int, aux_ids: List[int],
                       coupled_drains: bool = False, rtol: float = 1e-9) -> np.ndarray:
    dim = 2 * num_modes
    h_basis = hamiltonian_basis(triu_array, num_modes,
                                include_detunings=True, allow_phases=True)
    d_basis = dissipation_basis(aux_ids, num_modes, coupled_drains=coupled_drains)
    Om = symplectic_form(num_modes)

    # W must be annihilated by every diffusion generator D_c = Om Re(Y_c) Om^T.
    Ds = [Om @ Sc @ Om.T for _, Sc, _ in d_basis]
    W = _nullspace(np.vstack(Ds), rtol) if Ds else np.eye(dim)

    # A^T = -(G - Im Y) Om, so its generators are -G_b Om and +Im(Y_c) Om.
    gens = [-Gb @ Om for _, Gb in h_basis] + [Tc @ Om for _, _, Tc in d_basis]

    for _ in range(dim + 2):
        if W.shape[1] == 0:
            break
        cur = W
        for M in gens:
            if cur.shape[1] == 0:
                break
            P = cur @ cur.T                      # cur is orthonormal
            # ref = ||M||: the block below is zero exactly when cur is
            # already M-invariant, so it has no meaningful scale of its own.
            K = _nullspace((np.eye(dim) - P) @ M @ cur, rtol,
                            ref=float(np.linalg.norm(M, 2)))
            cur = _orthonormal(cur @ K) if K.shape[1] else np.zeros((dim, 0))
        if cur.shape[1] == W.shape[1]:
            W = cur
            break
        W = cur
    return W


# solution_set_dark_subspace(triu, V, ...) -> dimension of the largest
# subspace W with, for EVERY solution (G, Upsilon) of (**) at this V,
#     A(G,Upsilon)^T W subset W    and    D(Upsilon) W = 0.
#
# Nonzero => every point of the solution set carries a left eigenvector with
# Re(lambda) = 0, so no attractive solution exists AT THIS V. A certificate,
# not a failed search.
#
# This is undamped_subspace's condition applied to the SOLUTION SET rather
# than to the whole admissible set S_G x S_Upsilon, and that distinction is
# the entire content. Quantifying over every setting the graph permits is
# far too strong: a generic admissible point is not dark (measured
# ||w^T A|| = 0.560 on a graph whose solution set is uniformly dark), so
# undamped_subspace returns 0 and certifies nothing. Restricted to the
# solutions — the only settings that produce the target at all — the dark
# subspace is the SAME for every point, with only the oscillation frequency
# varying (measured: six sampled solutions, dark subspaces agreeing to
# ||P_i - P_0|| ~ 4e-16).
#
# Measured effect on the two-mode-squeezed target, all 512 graphs, r = 0.5,
# one auxiliary drain, fixed vacuum frame:
#     connectivity certificates alone     :  80 INVALID, 376 UNDECIDED
#     with this one                       : 456 INVALID,   0 UNDECIDED
# i.e. the partition is COMPLETE — every graph is decided, and the 56 VALID
# graphs are exactly the complement. Audited three ways: no graph carrying a
# Hurwitz witness is condemned (0 of 56); the verdict is invariant under
# relabelling the drain; and on 456 graphs x 5 independently drawn solution
# points the universal claim holds numerically, ||D W|| = 0 exactly and the
# A^T-invariance defect <= 8.4e-15.
#
# Earlier revisions of this file reported 156 INVALID / 300 UNDECIDED here.
# That figure was a TOLERANCE ARTEFACT, not a weaker certificate: the rank
# test inside the iteration judged the block (I - P) M W against its own
# largest singular value, which is ~1e-16 exactly when the subspace IS
# invariant, so the success case was read as full rank and the subspace
# collapsed. See _nullspace's header. The give-away was that 156 depended on
# which mode the drain was called (92 graphs flipped verdict under a
# relabelling); scale-correct tolerances removed the dependence and the
# undecided stratum with it.
#
# SCOPE: this certifies the graph AT THE GIVEN V. It is frame-dependent —
# the drain state enters through the solution set — so a graph certified
# here at a vacuum drain may still be valid with a squeezed one. That is
# not hypothetical: the two-beamsplitter scheme is exactly such a graph. So
# `decide` (fixed V) may propagate this verdict, while `optimise_aux_state`
# (which ranges over drain states) must not treat it as graph-level.
#
# return_basis=True returns the (complex, generally non-orthonormal) basis
# itself instead of its dimension, which is what pbh_certificate needs to
# exhibit the offending direction rather than merely count it.
def solution_set_dark_subspace(triu_array, V: np.ndarray, num_modes: int,
                                aux_ids: List[int], coupled_drains: bool = False,
                                rtol: float = 1e-9, return_basis: bool = False):
    dim = 2 * num_modes
    h_basis = hamiltonian_basis(triu_array, num_modes,
                                include_detunings=True, allow_phases=True)
    d_basis = dissipation_basis(aux_ids, num_modes, coupled_drains=coupled_drains)
    sol = solve_stationarity(V, h_basis, d_basis, None, rtol=rtol)
    N = sol['nullspace']
    if N.shape[1] == 0:
        return np.zeros((dim, 0), dtype=complex) if return_basis else 0

    # Complex counterparts of _nullspace / _orthonormal, with the same
    # scale-reference discipline: `ref` must be supplied wherever the block
    # can legitimately be zero (see _nullspace's header for what goes wrong
    # otherwise — a verdict that depends on which mode the drain is called).
    def null_c(M, ref=None):
        if M.size == 0:
            return np.eye(M.shape[1] if M.ndim == 2 else 0, dtype=complex)
        _, s, Vh = np.linalg.svd(M)
        scale = float(s[0]) if (ref is None and s.size) else (ref or 0.0)
        rank = int(np.sum(s > rtol * max(scale, 1e-300)))
        return Vh[rank:].conj().T

    def orth_c(X):
        if X.size == 0:
            return X
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        rank = int(np.sum(s > rtol * max(float(s[0]), 1e-300)))
        return U[:, :rank]

    As, Ds = [], []
    for k in range(N.shape[1]):
        G, Y = _assemble(N[:, k], h_basis, d_basis, None, dim)
        As.append(drift_matrix(G, Y).T.astype(complex))
        Ds.append(diffusion_matrix(Y).astype(complex))

    W = orth_c(null_c(np.vstack(Ds)))
    for _ in range(dim + 2):
        if W.shape[1] == 0:
            break
        cur = W
        for M in As:
            if cur.shape[1] == 0:
                break
            # cur is kept orthonormal, so P is the projector and ||M|| is the
            # right scale for the residual block (I - P) M cur.
            P = cur @ cur.conj().T
            K = null_c((np.eye(dim) - P) @ M @ cur,
                        ref=float(np.linalg.norm(M, 2)))
            cur = orth_c(cur @ K) if K.shape[1] else np.zeros((dim, 0), dtype=complex)
        if cur.shape[1] == W.shape[1]:
            W = cur
            break
        W = cur
    return W if return_basis else W.shape[1]


# pbh_certificate(...) -> None, or a dict exhibiting an UNCONTROLLABLE
# IMAGINARY MODE. §8 Move 2, and the auditable form of the two dark-subspace
# certificates above.
#
# The logic chain the doc sets out, in one place:
#   §3      every solution of (**) with V > 0 already has Re lambda <= 0, so
#           "not Hurwitz" can only mean an eigenvalue exactly ON the
#           imaginary axis — there is nothing to check off it.
#   PBH     (Popov-Belevitch-Hautus) the pair (A, B) with D = B B^T is
#           uncontrollable at lambda iff rank[A - lambda I | B] < 2n, and the
#           deficiency is witnessed by a left null vector w:
#               w^T (A - lambda I) = 0   and   w^T B = 0.
#   here    D w = B B^T w = 0 is EQUIVALENT to w^T B = 0, so "dark direction"
#           and "uncontrollable mode" are the same object. A dark mode is an
#           uncontrollable imaginary mode.
#
# So the subspace computations above (undamped_subspace over the whole
# admissible family; solution_set_dark_subspace over the solution set) do not
# merely suggest invalidity — each hands over a w that FAILS the PBH rank
# test, which is a single numerical quantity anyone can recompute. This
# function produces that number, together with the lambda it sits at.
#
# scope='admissible' quantifies over every (G, Upsilon) the graph permits and
# is therefore FRAME-INDEPENDENT — safe to propagate as a graph-level verdict
# regardless of drain state. scope='solution_set' quantifies only over the
# solutions at this V; it is far stronger (80 -> 456 of 512 on the
# two-mode-squeezed target, emptying the undecided stratum) but
# frame-DEPENDENT, so a caller that ranges over drain states must not treat
# it as graph-level. Both are returned; callers pick by scope.
def pbh_certificate(triu_array, V: np.ndarray, num_modes: int,
                     aux_ids: List[int], coupled_drains: bool = False,
                     rtol: float = 1e-9, scope: str = 'solution_set'
                     ) -> Optional[Dict]:
    dim = 2 * num_modes
    h_basis = hamiltonian_basis(triu_array, num_modes,
                                include_detunings=True, allow_phases=True)
    d_basis = dissipation_basis(aux_ids, num_modes, coupled_drains=coupled_drains)

    if scope == 'admissible':
        W = undamped_subspace(triu_array, num_modes, aux_ids,
                              coupled_drains=coupled_drains, rtol=rtol
                              ).astype(complex)
    elif scope == 'solution_set':
        W = solution_set_dark_subspace(triu_array, V, num_modes, aux_ids,
                                        coupled_drains=coupled_drains,
                                        rtol=rtol, return_basis=True)
    else:
        raise ValueError("scope must be 'admissible' or 'solution_set'")
    if W.shape[1] == 0:
        return None

    # A representative point of the set being quantified over. The subspace
    # claim is that EVERY point is dark on W, so any representative exhibits
    # the failure — but the representative must be PHYSICAL, or the PBH rank
    # test degenerates. A point with Upsilon not PSD has a diffusion matrix
    # with negative eigenvalues, D = B B^T has no real factor, B comes back
    # empty, and rank[A - lambda I | B] < 2n reduces to "A - lambda I is
    # singular" — true by construction and therefore proving nothing. So
    # draw until Upsilon >= 0 and D != 0, and record whether that succeeded.
    rng = np.random.default_rng(12345)
    if scope == 'solution_set':
        sol = solve_stationarity(V, h_basis, d_basis, None, rtol=rtol)
        N = sol['nullspace']
        draw = lambda: N @ rng.normal(0., 1., N.shape[1])
    else:
        nz = len(h_basis) + len(d_basis)
        draw = lambda: rng.normal(0., 1., nz)

    G = Y = None
    psd_point = False
    for _ in range(64):
        z = draw()
        Gc, Yc = _assemble(z, h_basis, d_basis, None, dim)
        if G is None:
            G, Y = Gc, Yc                    # fall-back representative
        if _is_psd(Yc) and np.linalg.norm(diffusion_matrix(Yc)) > 1e-12:
            G, Y, psd_point = Gc, Yc, True
            break

    A = drift_matrix(G, Y)
    D = diffusion_matrix(Y)
    # D = B B^T with B from the PSD square root; only its column space
    # matters to the rank test, so the symmetric root is as good as any
    # Cholesky factor and does not need D to be nonsingular.
    ew, ev = np.linalg.eigh((D + D.T) / 2)
    keep = ew > rtol * max(float(np.max(np.abs(ew))) if ew.size else 1.0, 1e-300)
    B = (ev[:, keep] * np.sqrt(np.clip(ew[keep], 0., None))) if np.any(keep) \
        else np.zeros((dim, 0))

    # Restrict A^T to W and diagonalise: every eigenvalue of the restriction
    # is an eigenvalue of A whose left eigenvector lies in W, hence is dark.
    Q, _ = np.linalg.qr(W)
    M = Q.conj().T @ A.T.astype(complex) @ Q
    lam, vec = np.linalg.eig(M)

    best = None
    for k in range(len(lam)):
        w = Q @ vec[:, k]
        nw = float(np.linalg.norm(w))
        if nw < 1e-300:
            continue
        w = w / nw
        eig_res = float(np.linalg.norm(A.T.astype(complex) @ w - lam[k] * w))
        dark_res = float(np.linalg.norm(B.T.astype(complex) @ w)) if B.size \
            else 0.0
        stacked = np.hstack([A.astype(complex) - lam[k] * np.eye(dim),
                             B.astype(complex)]) if B.size else \
            (A.astype(complex) - lam[k] * np.eye(dim))
        s = np.linalg.svd(stacked, compute_uv=False)
        rank = int(np.sum(s > rtol * max(float(s[0]), 1e-300)))
        cand = {'kind': 'pbh_uncontrollable', 'scope': scope,
                'dim': int(W.shape[1]), 'lambda': complex(lam[k]),
                'pbh_rank': rank, 'full_rank': dim,
                'eigen_residual': eig_res, 'darkness_residual': dark_res,
                'psd_point': psd_point, 'n_channels': int(B.shape[1]) if B.size else 0,
                'frame_dependent': (scope == 'solution_set'),
                'note': ('w^T A = lambda w^T with Re(lambda) = {:.2e} and w^T B = 0, '
                          'so rank[A - lambda I | B] = {} < {}: the mode is '
                          'uncontrollable, hence undamped, for every point of the '
                          '{} set — A can never be Hurwitz there'
                          .format(float(np.real(lam[k])), rank, dim,
                                  'solution' if scope == 'solution_set' else 'admissible'))}
        if rank >= dim:
            continue                       # not actually rank-deficient; skip
        if best is None or abs(np.real(lam[k])) < abs(np.real(best['lambda'])):
            best = cand
    return best


def structural_certificate(
    triu_array,
    V: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    coupled_drains: bool = False,
    corr_tol: float = 1e-10,
    include_solution_set: bool = True,
    attach_pbh: bool = True,
) -> Optional[Dict]:
    n = len(node_types)
    rows, cols = np.triu_indices(n)
    aux = set(i for i in range(n) if i not in target_mode_ids)

    # Connectivity is over Hamiltonian edges AND shared dissipators (§4A:
    # "the graph's connectivity (Hamiltonian edges and shared dissipators)").
    # With coupled_drains the auxiliary modes may share one reservoir, which
    # correlates them and therefore bridges components the edges alone leave
    # separate — so those links must be added before any cut argument.
    adj = [[] for _ in range(n)]
    for k, (i, j) in enumerate(zip(rows, cols)):
        if i != j and int(triu_array[k]) != NO_COUPLING:
            adj[i].append(j)
            adj[j].append(i)
    if coupled_drains:
        aux_list = sorted(aux)
        for a, b in zip(aux_list, aux_list[1:]):
            adj[a].append(b)
            adj[b].append(a)

    comp = [-1] * n
    for start in range(n):
        if comp[start] != -1:
            continue
        stack, cid = [start], start
        while stack:
            u = stack.pop()
            if comp[u] != -1:
                continue
            comp[u] = cid
            stack.extend(v for v in adj[u] if comp[v] == -1)

    # (a) undamped component containing a signal mode
    for c in set(comp):
        members = [i for i in range(n) if comp[i] == c]
        if any(i in target_mode_ids for i in members) and not any(i in aux for i in members):
            return {'kind': 'undamped_component', 'component': members,
                    'note': ('signal modes {} lie in a component with no auxiliary mode, so '
                              'D = 0 there and tr A = tr(Omega G) = 0; eigenvalues summing to '
                              'zero cannot all have Re < 0, so A is not Hurwitz for any G'
                              .format([i for i in members if i in target_mode_ids]))}

    # (a2) undamped SUBSPACE — the general form of (a), covering collective
    # and Bogoliubov dark modes that a per-mode connectivity test cannot
    # see. Checked after (a) only because (a) is cheaper and yields a more
    # readable certificate when it applies; (a2) subsumes it.
    W = undamped_subspace(triu_array, n, sorted(aux), coupled_drains=coupled_drains)
    if W.shape[1] > 0:
        cert = {'kind': 'undamped_subspace', 'dim': int(W.shape[1]),
                'basis': W,
                'note': ('a {}-dimensional subspace is annihilated by every admissible '
                          'diffusion generator AND invariant under every admissible drift '
                          'generator, so it carries a left eigenvector with Re(lambda) = 0 '
                          'for EVERY (G, Upsilon) on this graph; A can never be Hurwitz'
                          .format(W.shape[1]))}
        # §8 Move 2: restate the subspace claim as a PBH rank deficiency, the
        # form that can be recomputed and audited as one number.
        if attach_pbh:
            cert['pbh'] = pbh_certificate(triu_array, V, n, sorted(aux),
                                           coupled_drains=coupled_drains,
                                           scope='admissible')
        return cert

    # (b) target demands correlation across a cut
    if not coupled_drains:
        for a in target_mode_ids:
            for b in target_mode_ids:
                if a >= b or comp[a] == comp[b]:
                    continue
                blk = V[2 * a:2 * a + 2, 2 * b:2 * b + 2]
                if np.linalg.norm(blk) > corr_tol:
                    return {'kind': 'correlation_across_cut', 'modes': (a, b),
                            'block_norm': float(np.linalg.norm(blk)),
                            'note': ('target requires correlation between signal modes {} and {}, '
                                      'but this graph puts them in disconnected components with '
                                      'independent baths, so their steady state factorises'
                                      .format(a, b))}

    # (c) uniformly dark solution set — the general certificate, and the one
    # that does the most work (80 -> 456 of 512 on the two-mode-squeezed
    # target, leaving nothing undecided there).
    # FRAME-DEPENDENT: it certifies this graph at THIS V only, so a caller
    # ranging over drain states must not propagate it as a graph-level
    # verdict — see solution_set_dark_subspace's scope note.
    if include_solution_set:
        d = solution_set_dark_subspace(triu_array, V, n, sorted(aux),
                                        coupled_drains=coupled_drains)
        if d > 0:
            cert = {'kind': 'solution_set_dark', 'dim': int(d), 'frame_dependent': True,
                    'note': ('every solution of (**) at this drain state shares a '
                              '{}-dimensional dark subspace, so all of them have '
                              'Re(lambda) = 0 and none is attractive'.format(d))}
            if attach_pbh:
                cert['pbh'] = pbh_certificate(triu_array, V, n, sorted(aux),
                                               coupled_drains=coupled_drains,
                                               scope='solution_set')
            return cert
    return None


# ───────────────────────────────────────────────────────────────────────────
# §4B dissipator factorisation: reading the reservoir off the solution
# ───────────────────────────────────────────────────────────────────────────
# factor_dissipator(Upsilon, num_modes) -> one entry per dissipative channel.
#
# The doc's §4B readout. Rather than deciding vacuum-vs-squeezed by varying
# the drain STATE and re-solving, take the Upsilon a solve already returned,
# factor it as C^dag C, and read the reservoir directly off each jump vector:
#
#   L = c_x x + c_p p,  and with x = (a + a^dag)/sqrt2, p = (a - a^dag)/(i sqrt2),
#       L = alpha a + beta a^dag,   alpha = (c_x - i c_p)/sqrt2
#                                   beta  = (c_x + i c_p)/sqrt2
#
# beta is the ANOMALOUS (a^dag) weight. beta = 0 is a plain-loss channel — a
# vacuum reservoir — and beta != 0 is a Bogoliubov channel, i.e. a squeezed
# reservoir with
#       squeezing  s = arctanh(|beta| / |alpha|)
#       phase      = arg(beta) - arg(alpha), up to convention.
#
# Checked against the two closed forms in covariance_physics.build_jump_matrix:
#   vacuum   c = sqrt(k/2)[1, i]            -> beta = 0,          s = 0
#   squeezed c = sqrt(k/2)[e^r, i e^{-r}]   -> |beta|/|alpha| = tanh r, s = r
#
# This is strictly better than the V_aux scan for REPORTING: it costs one
# eigendecomposition of a matrix already in hand, needs no extra solves, and
# works for any number of channels at once. It does NOT replace the scan for
# SEARCHING, because which schemes are reachable still depends on the frame
# the graph constraint is written in — see optimise_aux_state.
def factor_dissipator(Upsilon: np.ndarray, num_modes: int,
                       tol: float = 1e-9) -> List[Dict]:
    Y = (Upsilon + Upsilon.conj().T) / 2
    w, U = np.linalg.eigh(Y)
    scale = max(float(np.max(np.abs(w))), 1e-300)

    channels = []
    for k in range(len(w) - 1, -1, -1):
        if w[k] <= tol * scale:
            continue
        # Upsilon = sum_mu conj(c_mu) c_mu^T, so a positive eigenpair
        # (w, u) contributes c = sqrt(w) * conj(u).
        c = np.sqrt(w[k]) * np.conj(U[:, k])
        per_mode = []
        for m in range(num_modes):
            cx, cp = c[2 * m], c[2 * m + 1]
            if abs(cx) < 1e-14 and abs(cp) < 1e-14:
                continue
            alpha = (cx - 1j * cp) / np.sqrt(2)
            beta = (cx + 1j * cp) / np.sqrt(2)
            ratio = abs(beta) / max(abs(alpha), 1e-300)
            per_mode.append({
                'mode': m, 'alpha': complex(alpha), 'beta': complex(beta),
                'rate': float(abs(alpha) ** 2 - abs(beta) ** 2),
                'squeezing': float(np.arctanh(min(ratio, 1 - 1e-15))) if ratio < 1 else np.inf,
                'phase': float(np.angle(beta) - np.angle(alpha)),
            })
        channels.append({'eigenvalue': float(w[k]), 'modes': per_mode,
                         'squeezed': any(p['squeezing'] > 1e-6 for p in per_mode)})
    return channels


# reservoir_summary(Upsilon, num_modes): one-line classification of what
# bath the solution actually calls for — 'vacuum' when every channel is
# plain loss, 'squeezed' when any carries an anomalous weight.
def reservoir_summary(Upsilon: np.ndarray, num_modes: int) -> Dict:
    ch = factor_dissipator(Upsilon, num_modes)
    sq = [p for c in ch for p in c['modes'] if p['squeezing'] > 1e-6]
    return {'channels': len(ch), 'kind': 'squeezed' if sq else 'vacuum',
            'max_squeezing': max([p['squeezing'] for p in sq], default=0.0),
            'detail': ch}


# physical_parameters(info, triu, node_types, target_mode_ids) -> the
# experimentally meaningful numbers behind a witness.
#
# The witness comes back as matrices (G, Upsilon); this turns them into the
# quantities somebody actually sets on a bench:
#
#   H = sum_i Delta_i a_i^dag a_i
#       + sum_<ij> ( J_ij a_i^dag a_j + h.c. )          beamsplitter edges
#       + sum_<ij> ( nu_ij a_i^dag a_j^dag + h.c. )     two-mode-squeezing edges
#       + sum_i   ( chi_i a_i^dag^2 + h.c. )            parametric self-edges
#   L_m = sqrt(kappa_m) ( cosh(s_m) a_m + e^{i theta_m} sinh(s_m) a_m^dag )
#
# EVERYTHING IS REPORTED AS A RATIO TO THE DRAIN RATE. The solution set is
# a cone — (G, Upsilon) -> (sG, sUpsilon) leaves the steady state untouched
# (§1(b)) — so an absolute coupling strength is meaningless and the raw
# numbers a solve returns can come back at any scale (they routinely arrive
# around 1e-5). Normalising to kappa fixes that gauge and makes two schemes
# comparable.
#
# Gauge warning on phases: a local rotation a_i -> e^{i phi} a_i shifts
# arg(J_ij) without changing any physics, so an individual coupling phase
# is NOT an observable. Magnitudes, detuning patterns and bath squeezings
# are. Phases are reported for completeness, flagged as gauge-dependent.
def physical_parameters(info: Dict, triu_array, node_types: List[str],
                         target_mode_ids: List[int]) -> Dict:
    n = len(node_types)
    aux_ids = [i for i in range(n) if i not in target_mode_ids]
    G, Y = info['G'], info['Upsilon']

    basis = hamiltonian_basis(triu_array, n, include_detunings=True, allow_phases=True)
    cols = np.column_stack([Gb.ravel() for _, Gb in basis])
    coeffs, *_ = np.linalg.lstsq(cols, G.ravel(), rcond=None)
    c = {lbl: float(v) for (lbl, _), v in zip(basis, coeffs)}

    # Drain rate sets the scale. Im(Upsilon) on a drain block is (kappa/2) J2.
    rates = {}
    for m in aux_ids:
        rates[m] = 2.0 * float(np.imag(Y[2 * m, 2 * m + 1]))
    kappa_ref = max((abs(v) for v in rates.values()), default=0.0)
    if kappa_ref < 1e-300:
        kappa_ref = 1.0

    detunings = {i: c.get(f'delta_{i}', 0.0) / kappa_ref for i in range(n)}

    couplings = {}
    rows, cols_i = np.triu_indices(n)
    for k, (i, j) in enumerate(zip(rows, cols_i)):
        if int(np.asarray(triu_array)[k]) == NO_COUPLING:
            continue
        if i == j:
            re, im = c.get(f'par_{i}', 0.0), c.get(f'par90_{i}', 0.0)
            if abs(re) + abs(im) > 1e-12:
                z = (re + 1j * im) / kappa_ref
                couplings[f'chi_{i}'] = {'kind': 'parametric', 'magnitude': abs(z),
                                          'phase': float(np.angle(z))}
            continue
        for pref, kind in (('bs', 'beamsplitter'), ('tms', 'two_mode_squeezing')):
            re, im = c.get(f'{pref}_{i}{j}', 0.0), c.get(f'{pref}90_{i}{j}', 0.0)
            if abs(re) + abs(im) > 1e-12:
                z = (re + 1j * im) / kappa_ref
                sym = 'J' if pref == 'bs' else 'nu'
                couplings[f'{sym}_{i}{j}'] = {'kind': kind, 'magnitude': abs(z),
                                               'phase': float(np.angle(z))}

    drains = {}
    for chan in factor_dissipator(Y, n):
        for p in chan['modes']:
            m = p['mode']
            drains.setdefault(m, []).append({
                'kappa': abs(rates.get(m, 0.0)) / kappa_ref,
                'squeezing': p['squeezing'], 'phase': p['phase'],
            })

    return {'kappa_reference': kappa_ref, 'detunings': detunings,
            'couplings': couplings, 'drains': drains,
            'channels': info.get('upsilon_rank'),
            'note': 'all rates in units of the drain rate; coupling phases are gauge-dependent'}


# format_parameters(...): the above, as a readable block.
def format_parameters(params: Dict, indent: str = '    ') -> str:
    out = []
    out.append(f'{indent}H = sum_i Delta_i a_i+ a_i'
               ' + (J_ij a_i+ a_j + nu_ij a_i+ a_j+ + chi_i a_i+^2 + h.c.)')
    out.append(f'{indent}L_m = sqrt(kappa_m)( cosh(s) a_m + e^{{i.theta}} sinh(s) a_m+ )')
    out.append(f'{indent}[rates in units of the drain rate; phases are gauge-dependent]')
    for name, d in sorted(params['couplings'].items(),
                          key=lambda kv: -kv[1]['magnitude']):
        out.append(f'{indent}  |{name}| = {d["magnitude"]:.6f}   '
                   f'arg = {d["phase"]:+.4f} rad   ({d["kind"]})')
    nz = {i: v for i, v in params['detunings'].items() if abs(v) > 1e-9}
    if nz:
        out.append(f'{indent}  detunings: ' +
                   ',  '.join(f'Delta_{i} = {v:+.6f}' for i, v in sorted(nz.items())))
    else:
        out.append(f'{indent}  detunings: all zero (resonant)')
    for m, chans in sorted(params['drains'].items()):
        for ch in chans:
            tag = ('plain vacuum' if ch['squeezing'] < 1e-6
                   else f'SQUEEZED s = {ch["squeezing"]:.6f}, theta = {ch["phase"]:+.4f} rad')
            out.append(f'{indent}  drain {m}: kappa = {ch["kappa"]:.6f}, {tag}')
    return '\n'.join(out)


# ───────────────────────────────────────────────────────────────────────────
# §2.1 frame change: vacuum-frame solution -> physical squeezed-drain frame
# ───────────────────────────────────────────────────────────────────────────
# apply_aux_symplectic(G, Upsilon, S_aux, aux_ids, num_modes).
#
# The doc's §2.1 bijection. An auxiliary-local symplectic S = I_T (+) S_A
# leaves V_TT alone (because V_TA = 0) and only reshapes V_AA, so it relates
# the convenient vacuum frame to the physical one by
#       G_phys = S^T G' S,      Upsilon_phys = S^T Upsilon' S,
# preserving symmetry, positivity and rank (channel count).
#
# The caveat the doc flags is the important one in practice: graph sparsity
# is a PHYSICAL-frame statement, and this map does not preserve it —
# squeezing a drain turns a beamsplitter edge into beamsplitter+two-mode-
# squeezing. So use this to translate a FOUND solution between frames, never
# to argue that searching one frame covers the other.
def apply_aux_symplectic(G, Upsilon, S_aux, aux_ids, num_modes):
    S = np.eye(2 * num_modes)
    for slot, m in enumerate(aux_ids):
        S[2 * m:2 * m + 2, 2 * m:2 * m + 2] = S_aux[2 * slot:2 * slot + 2,
                                                     2 * slot:2 * slot + 2]
    return S.T @ G @ S, S.T @ Upsilon @ S


# squeeze_symplectic(r, theta): the single-mode symplectic whose action on
# the vacuum produces a squeezed vacuum of parameter (r, theta) — the S_A
# to feed apply_aux_symplectic for a one-drain frame change.
def squeeze_symplectic(r: float, theta: float = 0.0) -> np.ndarray:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ np.diag([np.exp(-r), np.exp(r)]) @ R.T


# ───────────────────────────────────────────────────────────────────────────
# §4B optional SDP path (requires cvxpy; not a hard dependency)
# ───────────────────────────────────────────────────────────────────────────
# sdp_feasibility(V, h_basis, d_basis) -> (status, G, Upsilon).
#
# The real §4B: find G in S_G, Upsilon Hermitian PSD in S_Upsilon satisfying
# (**). The Hurwitz condition is deliberately ABSENT from the program — with
# V fixed it is equivalent to the absence of a dark mode, i.e. a
# rank/detectability condition and not a linear matrix inequality, so
# including it would destroy both convexity and the dual certificate that
# makes an INVALID verdict sound. It is recovered afterwards by
# attractivity_filter.
#
# Returns 'infeasible' (sound INVALID, with a dual certificate available
# from the solver), 'feasible' (a CANDIDATE only — still needs the filter),
# or 'unavailable' when cvxpy is not installed.
#
# MEASURED LIMITATION — the doc's §4B expects the infeasible branch to be
# the main source of INVALID certificates. It cannot fire in this regime,
# and the reason is structural rather than numerical.
#
# With lossless signal modes there is a point available on EVERY graph for
# EVERY pure target: damp the drain with plain loss and leave the signal
# modes entirely uncoupled (G = 0). The drain relaxes to its own vacuum,
# the signal modes are frozen at whatever V_TT says, and (**) is satisfied
# EXACTLY. It is PSD, and it survives any normalisation (its tr(Re Y) can
# be scaled to 1), so adding tr(Upsilon) = 1 to exclude Upsilon = 0 does
# not remove it either. Verified: residual 0.00e+00, PSD True, margin 0.
#
# So the feasible set is never empty, cvxpy returns 'optimal' for every
# graph tested — passive, active, and the empty graph alike — and no dual
# certificate is ever produced. This is the same frozen solution that makes
# §4A's least-squares residual identically zero; the SDP inherits it
# because it is a relaxation of the same equations.
#
# The consequence is that the doc's two certificate routes (§4A residual,
# §4B duality) both collapse at Gamma_signal = 0, and INVALID has to come
# from the structural certificates instead — see structural_certificate.
# What the SDP still provides is §4B's OTHER contribution: a feasible point
# whose Upsilon can be factored to read off the reservoir (see
# factor_dissipator), which works exactly as described.
def sdp_feasibility(V: np.ndarray, h_basis, d_basis):
    try:
        import cvxpy as cp
    except ImportError:
        return 'unavailable', None, None

    dim = V.shape[0]
    g = cp.Variable(len(h_basis)) if h_basis else None
    y = cp.Variable(len(d_basis)) if d_basis else None

    G = sum((g[k] * h_basis[k][1] for k in range(len(h_basis))),
            np.zeros((dim, dim))) if h_basis else np.zeros((dim, dim))
    ReY = sum((y[k] * d_basis[k][1] for k in range(len(d_basis))),
              np.zeros((dim, dim))) if d_basis else np.zeros((dim, dim))
    ImY = sum((y[k] * d_basis[k][2] for k in range(len(d_basis))),
              np.zeros((dim, dim))) if d_basis else np.zeros((dim, dim))

    Om = symplectic_form(dim // 2)
    eq = Om @ (G + ImY) @ V - V @ (G - ImY) @ Om + Om @ ReY @ Om.T

    # Upsilon >= 0 for the Hermitian S + iT is equivalent to the real
    # symmetric 2d x 2d matrix [[S, -T], [T, S]] being PSD — the standard
    # complex-to-real lifting, used so this works with any real SDP solver.
    lift = cp.bmat([[ReY, -ImY], [ImY, ReY]])
    prob = cp.Problem(cp.Minimize(0), [eq == 0, lift >> 0])
    try:
        prob.solve()
    except Exception:
        return 'unavailable', None, None

    if prob.status in ('infeasible', 'infeasible_inaccurate'):
        return 'infeasible', None, None
    if prob.status in ('optimal', 'optimal_inaccurate'):
        Gv = sum(float(g.value[k]) * h_basis[k][1] for k in range(len(h_basis))) \
             if h_basis else np.zeros((dim, dim))
        Yv = sum(float(y.value[k]) * (d_basis[k][1] + 1j * d_basis[k][2])
                 for k in range(len(d_basis))) if d_basis else np.zeros((dim, dim), complex)
        return 'feasible', Gv, Yv
    return 'unavailable', None, None


# ───────────────────────────────────────────────────────────────────────────
# the oracle
# ───────────────────────────────────────────────────────────────────────────
# decide(triu_array, V, target_mode_ids, node_types, ...) -> dict with
# 'verdict' in {VALID, INVALID, UNDECIDED} plus the witness or certificate.
#
# Pipeline (doc §4):
#   structural — the sound INVALID certificates (see structural_certificate;
#                these REPLACE the doc's residual-based certificate, which
#                is provably vacuous at Gamma_signal = 0).
#   fast path  — fix vacuum baths on the drains, solve linearly for G (4A),
#                then filter for a Hurwitz member.
#   joint path — free Hermitian Upsilon on the auxiliary block (4B), which
#                covers every Gaussian bath (thermal, squeezed,
#                cross-correlated) in one linear space; filter for a member
#                that is simultaneously PSD and Hurwitz.
#   gap max    — §8 Move 1: before conceding UNDECIDED, stop asking whether
#                the feasible set contains a Hurwitz point and MAXIMISE the
#                spectral gap over it instead. Costs gap_effort x the
#                sampling budget, but only on the graphs that would
#                otherwise be undecided.
#   verdict    — VALID only with a fully verified witness (residual zero,
#                Upsilon >= 0, A Hurwitz, and the FORWARD Lyapunov solve
#                reproducing V). Feasible-but-unwitnessed is UNDECIDED,
#                never INVALID.
def decide(
    triu_array,
    V: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    include_detunings: bool = True,
    allow_phases: bool = True,
    coupled_drains: bool = False,
    joint: bool = True,
    use_sdp: bool = False,
    num_samples: int = 32,
    seed: Optional[int] = None,
    rtol: float = RESIDUAL_RTOL_DEFAULT,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    include_solution_set: bool = True,
    gap_effort: int = 8,
) -> Dict:
    num_modes = len(node_types)
    dim = 2 * num_modes
    aux_node_ids = [i for i in range(num_modes) if i not in target_mode_ids]

    # Deterministic per-graph seed: §5(iii)'s bidirectional agreement check
    # compares partitions across two traversals, so the same graph must give
    # the same verdict regardless of when it is visited.
    if seed is None:
        seed = int(abs(hash(tuple(int(x) for x in np.asarray(triu_array)))) % (2 ** 31))

    h_basis = hamiltonian_basis(triu_array, num_modes,
                                include_detunings=include_detunings,
                                allow_phases=allow_phases)

    out = {'triu': np.asarray(triu_array), 'n_hamiltonian_dof': len(h_basis),
           'seed': seed, 'path': None}

    # ---- structural certificates: the sound INVALID verdicts --------------
    cert = structural_certificate(triu_array, V, target_mode_ids, node_types,
                                   coupled_drains=coupled_drains,
                                   include_solution_set=include_solution_set)
    if cert is not None:
        out['verdict'] = INVALID
        out['certificate'] = cert
        out['path'] = 'structural'
        return out

    # ---- fast path (§4A): fixed vacuum drains, solve for G alone ----------
    Y_vac = vacuum_dissipation(aux_node_ids, num_modes, rate=1.0)
    sol_fast = solve_stationarity(V, h_basis, None, Y_vac, rtol=rtol)
    out['residual_fixed_upsilon'] = sol_fast['residual']

    if sol_fast['feasible']:
        ok, z, margin = attractivity_filter(
            sol_fast, h_basis, None, Y_vac, dim, num_samples=num_samples,
            seed=seed, margin_tol=margin_tol, require_psd=False)
        if ok:
            G, Y = _assemble(z, h_basis, None, Y_vac, dim)
            out.update(_witness(G, Y, V, margin, 'fixed_upsilon'))
            return out

    if not joint:
        # Without the joint path a nonzero residual only rules out THIS
        # dissipator structure, so the honest verdict is undecided.
        out['verdict'] = UNDECIDED
        out['reason'] = ('no attractive solution with fixed vacuum drains; '
                          'joint=False so the graph itself was not decided')
        return out

    # ---- joint path (§4B): free Hermitian Upsilon on the aux block --------
    d_basis = dissipation_basis(aux_node_ids, num_modes, coupled_drains=coupled_drains)
    out['n_dissipation_dof'] = len(d_basis)
    # NOTE this system is HOMOGENEOUS (b = 0), so z = 0 always solves it and
    # sol['residual'] is identically zero — see structural_certificate's
    # header. It is recorded for diagnostics only and must never be read as
    # a verdict. The information is entirely in the nullspace: the graph is
    # valid iff that nullspace contains a PSD, Hurwitz member.
    sol = solve_stationarity(V, h_basis, d_basis, None, rtol=rtol)
    out['residual_joint'] = sol['residual']
    out['solution_dim'] = int(sol['nullspace'].shape[1])
    out['path'] = 'joint'

    ok, z, margin = attractivity_filter(
        sol, h_basis, d_basis, None, dim, num_samples=num_samples,
        seed=seed, margin_tol=margin_tol, require_psd=True)
    if ok:
        G, Y = _assemble(z, h_basis, d_basis, None, dim)
        out.update(_witness(G, Y, V, margin, 'joint'))
        return out

    # ---- §8 Move 1: gap maximisation before conceding UNDECIDED ------------
    # The cheap decision-mode filter stops at the first witness and can miss
    # a thin feasible cone entirely. The doc's Move 1 is to replace the
    # feasibility question with the OPTIMISATION
    #     gamma* = max over the feasible set of [-max_i Re lambda_i(A)],
    # on the grounds that a valid graph's Hurwitz members form a relatively
    # open, positive-measure subset, so a genuine interior ascent finds one
    # whenever it exists. Run at gap_effort x the sampling budget, with no
    # early exit, and only on graphs about to be filed UNDECIDED — so the
    # cost lands exactly on the stratum it is meant to shrink and nowhere
    # else. gamma* > 0 is a sound VALID witness; gamma* = 0 still proves
    # nothing (Move 1 attacks the VALID side only).
    if gap_effort and gap_effort > 0 and sol['nullspace'].shape[1] > 0:
        # Budget split: samples scale with gap_effort (each costs one
        # eigenvalue decomposition) while the Nelder-Mead restarts, which are
        # three orders of magnitude dearer, grow only additively. The
        # restarts are what actually walks into a thin feasible cone, so they
        # cannot be dropped — but 2 + effort of them is already well past the
        # point where extra ones find anything new.
        ok, z, margin = attractivity_filter(
            sol, h_basis, d_basis, None, dim,
            num_samples=num_samples * int(gap_effort),
            seed=seed + 1, margin_tol=margin_tol, require_psd=True,
            n_polish=2 + int(gap_effort), maximise=True)
        out['gap_star'] = margin
        if ok:
            G, Y = _assemble(z, h_basis, d_basis, None, dim)
            out.update(_witness(G, Y, V, margin, 'joint_gap_max'))
            return out

    if use_sdp:
        status, G, Y = sdp_feasibility(V, h_basis, d_basis)
        out['sdp_status'] = status
        if status == 'infeasible':
            out['verdict'] = INVALID
            out['certificate'] = {'kind': 'sdp_dual', 'note':
                                   'SDP feasibility program infeasible (§4B)'}
            return out

    # Solutions exist but none of them was shown to be attractive. NOT
    # invalid — the search must not prune on this (§4C).
    # Feasible, no Hurwitz member found, no impossibility certificate: §4C's
    # third outcome. The true category exists — the graph objectively either
    # has a Hurwitz member or does not — but the procedure has not computed
    # it, so it must NOT be pruned on (§4C, §5(i)).
    out['verdict'] = UNDECIDED
    out['reason'] = ('feasible (stationary solutions exist) but no strictly Hurwitz, '
                      'PSD member was found and no universal certificate applies; '
                      'raise gap_effort/num_samples, or decide it with §8 Move 3')
    out['best_margin'] = margin
    return out


# scan_aux_squeezing(...): run `decide` over a grid of drain squeezings.
#
# For the Zippilli-Vitali LABELLING (passive network, squeezing supplied by
# the drain's bath), which the default V_AA = (1/2)I gauge fixing hides —
# see complete_covariance's header for why that gauge is a relabelling
# rather than a loss of completeness.
#
# The grid is short because the working drain squeezing is not arbitrary:
# the drain supplies the target's squeezing, so for a squeezing-r target
# the solution sits at r_aux = r, and the solution-space dimension jumps
# sharply there (numerically: dim 3 -> 5 at exactly r_aux = r for the
# two-mode-squeezed target on the Vitali triangle). Verdicts are pointwise
# in the grid: a hit is a genuine VALID witness, but a grid-wide miss is
# UNDECIDED, never INVALID, since the grid proves nothing between points.
#
# MULTIPLE DRAINS. The default grid is the SHARED-squeezing slice — every
# drain given the same (r, theta) — because an independent per-drain grid
# costs (|r_grid|*|theta_grid|)^m and is unusable past m = 2. That slice is
# a heuristic, not a cover, so for m >= 2 a miss is weak evidence: it says
# nothing about unequal or correlated drain states. Two ways out:
#   - pass `configs`, an explicit iterable of per-drain configurations
#     (each a list of m (r, theta) pairs, or an m-drain covariance block
#     accepted by complete_covariance's aux_cov), to scan whatever slice
#     you actually care about;
#   - or just use the default gauge (aux_squeezing=None) in `decide`, which
#     needs no scan at all and is complete for any m — the drain squeezing
#     is then absorbed into the graph labelling.
# The second is the right default; this function exists to recover the
# Zippilli-Vitali NAMING, not to extend the search's reach.
def scan_aux_squeezing(
    triu_array,
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    r_grid=None,
    theta_grid=None,
    configs=None,
    **kwargs,
) -> Dict:
    num_modes = len(node_types)
    n_aux = num_modes - len(target_mode_ids)

    if configs is None:
        if r_grid is None:
            r_grid = np.linspace(0., 1.5, 31)
        if theta_grid is None:
            theta_grid = [0., np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        configs = [[(float(r), float(th))] * n_aux
                   for r in r_grid for th in theta_grid]

    best = None
    for cfg in configs:
        cfg_arr = np.asarray(cfg, dtype=float)
        if cfg_arr.ndim == 2 and cfg_arr.shape == (2 * n_aux, 2 * n_aux):
            V = complete_covariance(sigma_target, target_mode_ids, num_modes,
                                     aux_cov=cfg_arr)
        else:
            V = complete_covariance(sigma_target, target_mode_ids, num_modes,
                                     aux_squeezing=[tuple(c) for c in cfg])
        out = decide(triu_array, V, target_mode_ids, node_types, **kwargs)
        out['aux_squeezing'] = cfg[0] if (n_aux == 1 or
                                           all(tuple(c) == tuple(cfg[0]) for c in cfg)) else cfg
        if out['verdict'] == VALID:
            return out
        if best is None or out.get('best_margin', -np.inf) > best.get('best_margin', -np.inf):
            best = out
    # A grid-wide miss decides nothing between grid points.
    best['verdict'] = UNDECIDED if best['verdict'] != INVALID else INVALID
    return best


# ───────────────────────────────────────────────────────────────────────────
# deciding vacuum vs squeezed reservoir, without being told
# ───────────────────────────────────────────────────────────────────────────
# optimise_aux_state(...): find the drain STATE the graph needs, by
# continuous optimisation rather than a grid.
#
# This is the analogue of covariance_optimizer's squeezable_aux_ids /
# squeeze_ab free variables, where the gradient optimiser decided for itself
# whether and how much to squeeze each drain and (a_,b_) = (0,0) recovered
# plain vacuum. The linear oracle needs the same freedom, but cannot simply
# make the drain state another unknown of the linear system: V multiplies G
# in (**), so an unknown V_AA would make the problem bilinear and destroy
# the one-shot global solve (see complete_covariance's header).
#
# The resolution is that the drain state is a LOW-DIMENSIONAL OUTER
# variable — 2 parameters (r, theta) per drain — around an inner problem
# that stays exactly linear. What makes this practical rather than a blind
# search is that the inner solve provides a sharp, cheap detector of the
# right drain state:
#
#   A solution exists exactly where the design matrix M(r, theta) DROPS
#   RANK. Its nullspace has some generic dimension d0 for a typical drain
#   state; at the drain state the scheme actually needs, extra solutions
#   appear and the (d0+1)-th smallest singular value collapses to ~1e-16.
#
# Measured, on the passive Zippilli-Vitali star for the z=0.6 Cayley
# cluster (sigma_4, normalised):
#     r_aux   0.00    0.55    0.58    0.60    0.62    0.90
#     sigma   0.955   0.225   0.093   6e-16   0.096   0.955
# and on the Vitali two-mode-squeezed target, the same V-shape bottoming at
# r_aux = 0.40 = r_target. The minimum is sharp but the approach to it is
# smooth, so a derivative-free minimiser walks straight in — no grid, and
# no risk of stepping over the exact value the way a fixed grid does.
#
# Reading the answer:
#   r* ~ 0  -> a PLAIN VACUUM reservoir suffices for this graph
#   r* > 0  -> a SQUEEZED reservoir is REQUIRED, and r* is how much
# so the vacuum-vs-squeezed question is decided by the search, not declared
# by the user. Returns the optimised drain states plus the verdict there.
def optimise_aux_state(
    triu_array,
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    n_starts: int = 6,
    r_max: float = 2.0,
    seed: int = 0,
    rank_drop_tol: float = 1e-9,
    squeeze_cost: float = 1e-3,
    prefer_vacuum: bool = False,
    **decide_kwargs,
) -> Dict:
    num_modes = len(node_types)
    aux_ids = [i for i in range(num_modes) if i not in target_mode_ids]
    n_aux = len(aux_ids)

    # prefer_vacuum is an optional SHORT CUT, not the mechanism. With it the
    # vacuum case is settled by one solve instead of an optimisation; with
    # it off (the default) nothing is privileged and the squeezing-cost term
    # in `objective` below is what drives r to 0 when no squeezing is
    # needed. Both give the same answer — the flag only trades a little
    # compute for having one fewer hand-written rule in the loop.
    # The solution-set certificate is FRAME-DEPENDENT, and this function
    # ranges over drain states, so it must not be allowed to stamp a
    # graph-level INVALID. Only the frame-independent structural
    # certificates (connectivity) may do that here.
    decide_kwargs = dict(decide_kwargs, include_solution_set=False)
    V_vac = complete_covariance(sigma_target, target_mode_ids, num_modes)
    out_vac = decide(triu_array, V_vac, target_mode_ids, node_types, **decide_kwargs)
    if prefer_vacuum and out_vac['verdict'] == VALID:
        out_vac['aux_squeezing'] = [(0.0, 0.0)] * n_aux
        out_vac['reservoir'] = 'vacuum'
        out_vac['rank_drop_score'] = 0.0
        return out_vac
    h_basis = hamiltonian_basis(
        triu_array, num_modes,
        include_detunings=decide_kwargs.get('include_detunings', True),
        allow_phases=decide_kwargs.get('allow_phases', True))
    d_basis = dissipation_basis(aux_ids, num_modes,
                                coupled_drains=decide_kwargs.get('coupled_drains', False))

    def svs(params):
        aux = [(float(params[2 * k]), float(params[2 * k + 1])) for k in range(n_aux)]
        V = complete_covariance(sigma_target, target_mode_ids, num_modes,
                                 aux_squeezing=aux)
        M, _, _ = build_linear_system(V, h_basis, d_basis, None)
        s = np.linalg.svd(M, compute_uv=False)
        return np.sort(s) / max(float(s[0]) if s.size else 1.0, 1e-300)

    rng = np.random.default_rng(seed)

    # Generic nullity d0: the rank deficiency present at a TYPICAL drain
    # state, which carries no information. Only deficiency BEYOND it marks
    # a drain state the graph can actually use.
    tol = 1e-10
    d0 = min(int(np.sum(svs(rng.uniform(0.1, 1.0, 2 * n_aux)) <= tol)) for _ in range(5))

    # Objective = rank drop + a small squeezing cost.
    #
    # The rank-drop term alone is degenerate whenever squeezing is not
    # needed: if the graph works at every drain state, the term is ~1e-16
    # everywhere and the minimiser wanders off to wherever it happens to
    # land, then reports that arbitrary point as "the required squeezing".
    # Adding lambda*sum(r^2) breaks the tie the way the physics wants — among
    # all drain states that work, prefer the LEAST squeezed one — so r
    # relaxes to 0 exactly when a vacuum reservoir suffices. This is the
    # continuous analogue of covariance_optimizer's free (a_,b_) variables
    # settling at (0,0), and it replaces the discrete "try vacuum first"
    # rule with something the optimiser decides for itself.
    #
    # lambda must be small enough not to move a genuine r*. It does not: the
    # rank-drop term rises from ~1e-16 to O(0.1) within ~0.02 of the correct
    # squeezing, so a penalty of order 1e-3 cannot shift the minimum, and
    # measured r* stays 0.400000 / 0.600000 / 1.000000 to six decimals.
    def objective(params):
        p = np.array(params, dtype=float)
        if np.any(p[0::2] < -1e-9) or np.any(p[0::2] > r_max):
            return 1e3
        s = svs(p)
        drop = float(s[d0]) if d0 < len(s) else 0.0
        return drop + squeeze_cost * float(np.sum(p[0::2] ** 2))

    # COARSE SCAN, then local refine. Pure multistart is not reliable here:
    # the detector has a sharp zero at the drain state that works, but it
    # ALSO decays slowly at large squeezing, so a random start in that tail
    # slides monotonically to the bound and never sees the real well. On the
    # two-mode-squeezed target the detector reads 0.23 at r = 0.7 and 0.09 at
    # r = 1.2 while the true zero sits at r = 0.50 — a start above ~0.7 is
    # simply lost. Random multistart found it only by luck.
    #
    # The objective costs one SVD (sub-millisecond), so a coarse sweep is
    # cheap insurance: the well is ~0.1 wide, so a step of 0.05 cannot skip
    # it, and the best grid points then seed a local refine that recovers
    # full precision. This is ordinary global-optimisation hygiene, not
    # knowledge about the target — the grid is over the SEARCH variable and
    # says nothing about which value is expected.
    #
    # With several drains a full product grid is exponential, so the coarse
    # pass sweeps the shared-squeezing diagonal only and the refine explores
    # the full 2*n_aux space from there (plus random restarts).
    coarse_r = np.arange(0.0, r_max + 1e-9, 0.05)
    scan = [(objective(np.concatenate([[r, 0.0]] * n_aux)), r) for r in coarse_r]
    scan.sort()
    seeds = [np.concatenate([[r, 0.0]] * n_aux) for _, r in scan[:max(3, n_starts // 2)]]

    best = (np.inf, np.zeros(2 * n_aux))
    starts = [np.zeros(2 * n_aux)] + seeds               # vacuum, then best grid points
    starts += [np.concatenate([[rng.uniform(0, r_max), rng.uniform(0, np.pi)]
                               for _ in range(n_aux)]) for _ in range(n_starts)]
    for p0 in starts:
        res = sciopt.minimize(objective, p0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-16})
        if res.fun < best[0]:
            best = (float(res.fun), res.x)
    _, p_star = best

    # Threshold the RAW rank drop, never the penalised objective. The
    # squeezing cost is deliberately nonzero at any genuine r* (at r* = 0.4
    # it contributes 1e-3 * 0.16 = 1.6e-4, far above rank_drop_tol), so
    # testing the objective would reject every squeezed solution as "not a
    # real rank drop" — the penalty exists to break ties, not to judge.
    s_star = svs(np.maximum(p_star, 0.0))
    score = float(s_star[d0]) if d0 < len(s_star) else 0.0

    # The rank drop must actually REACH zero. A minimiser that merely drifts
    # with a drop of ~1e-2 has found no extra solution at all, and reporting
    # its landing point as "the required squeezing" would be reading signal
    # out of noise. Below tolerance the drain state is a genuine requirement
    # of the graph; above it, there is none to find.
    if score > rank_drop_tol:
        # No drain state is SPECIAL for this graph. Two very different
        # situations share that signature, and they are told apart by
        # whether the graph works at a generic drain state at all:
        #   valid at vacuum   -> it works for every drain state, so no
        #                        squeezing is required; the answer is vacuum
        #                        and the flat objective was flat because
        #                        there was nothing to find, not because the
        #                        search failed
        #   not valid there   -> no reservoir makes this graph work
        out_vac['aux_squeezing'] = [(0.0, 0.0)] * n_aux
        out_vac['rank_drop_score'] = score
        out_vac['generic_nullity'] = d0
        if out_vac['verdict'] == VALID:
            out_vac['reservoir'] = 'vacuum'
            out_vac['reason'] = (
                'no drain state is distinguished (no rank drop anywhere), and the graph '
                'is valid with an unsqueezed drain — squeezing is not required')
        else:
            out_vac['reservoir'] = 'none found'
            out_vac['reason'] = (
                f'no drain state produced a rank drop (best {score:.2e} > {rank_drop_tol:.0e}) '
                'and the graph is not valid with vacuum either')
        return out_vac

    aux_star = [(max(0.0, float(p_star[2 * k])), float(p_star[2 * k + 1]))
                for k in range(n_aux)]
    V = complete_covariance(sigma_target, target_mode_ids, num_modes,
                             aux_squeezing=aux_star)
    out = decide(triu_array, V, target_mode_ids, node_types, **decide_kwargs)
    out['aux_squeezing'] = aux_star
    out['rank_drop_score'] = score
    out['generic_nullity'] = d0
    out['reservoir'] = ('vacuum' if all(r < 1e-3 for r, _ in aux_star)
                        else 'squeezed')
    return out


# _witness(G, Y, V, margin, path): package a solution and VERIFY it end to
# end. A witness is only accepted if it survives every independent check:
# the stationarity residual, Upsilon >= 0, strict Hurwitz-ness, and the
# FORWARD Lyapunov solve reproducing V. The last is the important one — it
# is computed by a completely different route (solve_continuous_lyapunov on
# the assembled A and D) and so catches sign or convention errors that the
# residual, built from the same expression as the design matrix, cannot.
def _witness(G, Y, V, margin, path):
    A = drift_matrix(G, Y)
    D = diffusion_matrix(Y)
    resid = float(np.linalg.norm(lyapunov_residual_matrix(G, Y, V)))
    try:
        V_fwd = sla.solve_continuous_lyapunov(A, -D)
        fwd_err = float(np.linalg.norm(V_fwd - V))
    except Exception:
        V_fwd, fwd_err = None, np.inf
    # Channel count must use a RELATIVE tolerance. The solution set is a
    # cone, so a witness may come back with ||Upsilon|| tiny in absolute
    # terms while being a perfectly good rank-1 or rank-2 dissipator; an
    # absolute tol=1e-8 then reports "0 dissipative channels", which is not
    # merely wrong but physically impossible — zero channels means D = 0 and
    # no Hurwitz A can exist at all (§3).
    w = np.linalg.svd(Y, compute_uv=False)
    rank = int(np.sum(w > 1e-9 * max(float(w[0]) if w.size else 1.0, 1e-300)))
    # §4B readout: what reservoir does this solution actually call for?
    # Taken from the returned Upsilon by factorisation, so it is a property
    # of the witness rather than of whatever frame the search happened to
    # run in — validated to recover s = 0.4 / 0.6 / 1.0 exactly on the
    # closed-form squeezed baths.
    res = reservoir_summary(Y, G.shape[0] // 2)
    return {'verdict': VALID, 'G': G, 'Upsilon': Y, 'A': A, 'D': D,
            # 'gap' and 'stability_margin' are the same number under two
            # names: §8 Move 1 observes that the quantity deciding validity
            # IS the dissipative gap, so the oracle returns the figure of
            # merit for the resource-vs-gap comparison for free.
            'gap': margin,
            'stability_margin': margin, 'stationarity_residual': resid,
            'forward_error': fwd_err, 'V_forward': V_fwd, 'path': path,
            'upsilon_psd': _is_psd(Y), 'upsilon_rank': rank,
            'reservoir_kind': res['kind'], 'reservoir_squeezing': res['max_squeezing'],
            'channels': res['detail']}
