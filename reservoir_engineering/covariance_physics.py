"""
covariance_physics.py
=====================
Low-level physics engine for open quantum systems in the real quadrature basis.

Everything here must be JAX-differentiable so gradients flow back to coupling
strengths and the outer optimiser can use jax.grad.

═══════════════════════════════════════════════════════════════════════════════
REAL QUADRATURE BASIS
═══════════════════════════════════════════════════════════════════════════════

For N bosonic modes, the state vector is:

    q = (x_0, p_0, x_1, p_1, ..., x_{N-1}, p_{N-1})     shape: (2N,)

Mode i occupies rows/columns  2i  and  2i+1  in ALL matrices.
All matrices (A, D, σ) are real-valued 2N × 2N.
Vacuum noise level = ½ per quadrature (convention: [x, p] = i).

═══════════════════════════════════════════════════════════════════════════════
OPEN SYSTEM DYNAMICS  (Lyapunov equation)
═══════════════════════════════════════════════════════════════════════════════

The system evolves under the linear quantum Langevin equation:

    dq/dt = A · q + noise

In steady state, the covariance matrix σ (where σ_ij = ½⟨{q_i,q_j}⟩)
satisfies the continuous-time Lyapunov equation:

    A · σ + σ · Aᵀ + D = 0

where:
  A  — drift matrix   (2N × 2N, real) — encodes coherent couplings + decay
  D  — diffusion matrix (2N × 2N, real, diagonal) — encodes bath noise

Solution via Kronecker-product vectorisation:

    vec(σ) = -(I ⊗ A + A ⊗ I)⁻¹ vec(D)

This is a (2N)² × (2N)² linear system, solvable by jnp.linalg.solve.
jnp.linalg.solve is fully supported by jax.grad, so the whole pipeline is
differentiable with respect to the coupling strengths that enter A.

Stability requirement:
    All eigenvalues of A must have strictly negative real parts.
    (Violated → no steady state → Lyapunov solution is meaningless.)
    Enforced by the optimiser via reparametrisation or penalty.

═══════════════════════════════════════════════════════════════════════════════
NODE TYPES  (what kind of physical mode is each graph node?)
═══════════════════════════════════════════════════════════════════════════════

'cavity'      — optical or microwave resonator coupled to a zero-temperature
                (vacuum) input bath.
                Decay rate : kappa  (κ)
                Bath noise : D_ii = κ/2  per quadrature  (vacuum noise only)

'mechanical'  — mechanical oscillator (or magnon mode) coupled to a thermal
                phonon bath at occupation n_th.
                Decay rate : gamma  (γ)
                Thermal occ: n_th   (n̄)
                Bath noise : D_ii = γ(n̄ + ½)  per quadrature

Node dict format used throughout this package:
    {
        'id'   : int,            # unique node index 0..N-1
        'type' : 'cavity' | 'mechanical',
        'kappa': float,          # used if type == 'cavity'
        'gamma': float,          # used if type == 'mechanical'
        'n_th' : float,          # used if type == 'mechanical'
    }

═══════════════════════════════════════════════════════════════════════════════
EDGE TYPES  (what kind of coupling connects two graph nodes?)
═══════════════════════════════════════════════════════════════════════════════

'beamsplitter'  (BS)  — Hamiltonian H = g (a†b + ab†)
    Energy-conserving swap interaction (two-tone red sideband drive).
    In the real quadrature basis:
        A[2i:2i+2, 2j:2j+2] +=  g · I₂     (i → j block)
        A[2j:2j+2, 2i:2i+2] +=  g · I₂     (j → i block)
    Both off-diagonal 2×2 blocks are  +g · I₂.
    Physical effect: swaps state between modes i and j.

'two_mode_squeezing'  (TMS)  — H = ν (a†b† + ab)
    Parametric amplification interaction (two-tone blue sideband drive).
    Uses σ_z = diag(+1, −1):
        A[2i:2i+2, 2j:2j+2] +=  +ν · σ_z   (i → j block)
        A[2j:2j+2, 2i:2i+2] +=  −ν · σ_z   (j → i block)
    Physical effect: generates squeezing and entanglement between i and j.

'parametric'  (SQ)  — H = χ (a² + a†²)    (single-mode, i == j only)
    Single-mode squeezing via a degenerate OPA.
    Acts on the diagonal 2×2 block of mode i:
        A[2i:2i+2, 2i:2i+2] +=  χ · σ_z
    Physical effect: squeezes one quadrature of mode i.

Edge dict format used throughout this package:
    {
        'i'       : int,    # index of first mode
        'j'       : int,    # index of second mode (== i for parametric)
        'type'    : 'beamsplitter' | 'two_mode_squeezing' | 'parametric',
        'strength': float,  # coupling constant g, ν, or χ (NOT used by
                            # build_drift_matrix — it reads coupling_strengths
                            # array instead, so the function stays differentiable)
    }

═══════════════════════════════════════════════════════════════════════════════
KRONWALD VALIDATION
═══════════════════════════════════════════════════════════════════════════════

The canonical test case is a 2-mode system (1 cavity + 1 mechanical) with:
    edges = [BS(g),  TMS(ν)]

The achieved mechanical squeezing should satisfy  r = atanh(ν/g)  as γ → 0.
Compare against kronwald_optimizer.py which hard-codes this topology.
If these two give the same σ_mech, the physics engine is correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict

jax.config.update("jax_enable_x64", True)


# ───────────────────────────────────────────────────────────────────────────
# quadrature_slice(mode_id)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Return the Python slice that selects mode `mode_id` from the (2N,)
#   quadrature vector (or the corresponding rows/columns of a 2N×2N matrix).
#
# Rule:  mode i → indices  2i  and  2i+1
#
# Examples:
#   quadrature_slice(0) → slice(0, 2)      ← cavity (x_cav, p_cav)
#   quadrature_slice(1) → slice(2, 4)      ← mechanics (x_mech, p_mech)
#   quadrature_slice(2) → slice(4, 6)      ← third mode
#
# This helper is called by build_drift_matrix and build_diffusion_matrix
# to locate the 2×2 block for each node and each edge.

def quadrature_slice(mode_id: int):
    pass


# ───────────────────────────────────────────────────────────────────────────
# build_drift_matrix(nodes, edges, coupling_strengths)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Assemble the 2N × 2N real drift matrix A from the graph description.
#   Must be JAX-differentiable with respect to `coupling_strengths`.
#
# Parameters:
#   nodes              — list of N node dicts (see NODE TYPES above)
#   edges              — list of E edge dicts (types only; strengths NOT read
#                        from 'strength' field — use coupling_strengths array)
#   coupling_strengths — jnp.ndarray shape (E,), one value per edge in order
#
# Returns:
#   A — jnp.ndarray shape (2N, 2N), real
#
# Construction (add each contribution to a zero 2N×2N matrix):
#
#   Step 1 — diagonal decay blocks (one per node):
#     For node i of type 'cavity':     A[s_i, s_i] += −(κ_i / 2) · I₂
#     For node i of type 'mechanical': A[s_i, s_i] += −(γ_i / 2) · I₂
#     where s_i = quadrature_slice(i) and I₂ = identity(2)
#
#   Step 2 — off-diagonal coupling blocks (one per edge):
#     Let g = coupling_strengths[edge_idx], s_i = quadrature_slice(edge.i),
#     s_j = quadrature_slice(edge.j), σ_z = diag(+1, -1)
#
#     'beamsplitter':
#         A[s_i, s_j] +=  g · I₂
#         A[s_j, s_i] +=  g · I₂
#
#     'two_mode_squeezing':
#         A[s_i, s_j] +=  g · σ_z
#         A[s_j, s_i] += -g · σ_z
#
#     'parametric' (i == j):
#         A[s_i, s_i] +=  g · σ_z
#
# JAX differentiability note:
#   Do NOT use plain Python index assignment (A[s] = ...) on jnp arrays —
#   that is not differentiable. Instead build A using jnp.zeros + a series
#   of jnp operations, OR use lax.dynamic_update_slice / index_update with
#   jax.ops. The standard approach is to accumulate contributions via
#   repeated addition on a base-zero jnp array.

def build_drift_matrix(
    nodes: List[Dict],
    edges: List[Dict],
    coupling_strengths: jnp.ndarray,
) -> jnp.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# build_diffusion_matrix(nodes)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Assemble the 2N × 2N diagonal diffusion matrix D.
#   D does NOT depend on coupling_strengths — it is a constant for a fixed
#   set of nodes — so it can be computed once and reused inside jit loops.
#
# Parameters:
#   nodes — list of N node dicts
#
# Returns:
#   D — jnp.ndarray shape (2N, 2N), real, diagonal
#
# Construction:
#   For cavity node i:
#       D[2i, 2i]     = κ_i / 2
#       D[2i+1, 2i+1] = κ_i / 2
#   For mechanical node i:
#       D[2i, 2i]     = γ_i · (n̄_i + 0.5)
#       D[2i+1, 2i+1] = γ_i · (n̄_i + 0.5)
#
# Can be built with jnp.diag(jnp.array([...])) or np.diag + jnp.array cast.

def build_diffusion_matrix(nodes: List[Dict]) -> jnp.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# solve_lyapunov_kronecker(A, D)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Solve  A σ + σ Aᵀ + D = 0  for the steady-state covariance matrix σ.
#   Fully JAX-differentiable (jnp.linalg.solve is AD-supported).
#
# Parameters:
#   A — jnp.ndarray (2N, 2N) drift matrix
#   D — jnp.ndarray (2N, 2N) diffusion matrix
#
# Returns:
#   σ — jnp.ndarray (2N, 2N), the steady-state covariance matrix
#
# Method — Kronecker vectorisation:
#   1. n = A.shape[0]
#   2. I = jnp.eye(n)
#   3. M = jnp.kron(I, A) + jnp.kron(A, I)     shape (n², n²)
#      Note: uses the identity  (I⊗A + A⊗I) vec(σ) = −vec(D)
#      where vec(M) = M.flatten() in row-major (C) order.
#   4. vec_sigma = jnp.linalg.solve(M, −D.flatten())
#   5. return vec_sigma.reshape(n, n)
#
# This is identical to kronwald_optimizer.solve_lyapunov — just
# generalised from hard-coded n=4 to arbitrary n=2*N_modes.
#
# Numerical note:
#   M is singular iff A has two eigenvalues λ_i, λ_j with λ_i + λ_j = 0.
#   This happens at stability boundaries.  Always ensure A is Hurwitz before
#   calling this.

def solve_lyapunov_kronecker(A: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# get_mode_covariance(sigma, mode_ids)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Extract the 2M × 2M covariance submatrix for a chosen subset of M modes.
#   Used to compare only the "signal" modes against the target covariance,
#   ignoring auxiliary cavity modes that are not part of the target state.
#
# Parameters:
#   sigma    — jnp.ndarray (2N, 2N), full system covariance
#   mode_ids — list of int, e.g. [1] for the mechanical mode in a 2-mode system
#
# Returns:
#   sigma_sub — jnp.ndarray (2M, 2M)  where M = len(mode_ids)
#
# Construction:
#   Build index list: idx = [2i, 2i+1  for i in mode_ids]
#   Return sigma[np.ix_(idx, idx)]   (rows and columns indexed by idx)
#
# Example (2-mode system, N=2):
#   get_mode_covariance(sigma, [1])  →  sigma[2:4, 2:4]  (mechanical block)
#   This matches kronwald_optimizer.mechanical_cov(sigma) = sigma[2:, 2:].

def get_mode_covariance(
    sigma: jnp.ndarray,
    mode_ids: List[int],
) -> jnp.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# covariance_loss(coupling_strengths, nodes, edges, target_cov, target_mode_ids)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   The scalar loss function for parameter optimisation.
#   Measures how far the achieved steady-state covariance is from the target.
#   Designed to be JIT-compiled and differentiated via jax.grad.
#
# Parameters:
#   coupling_strengths — jnp.ndarray (E,)  — the optimisation variables
#   nodes              — list of node dicts
#   edges              — list of edge dicts (types only)
#   target_cov         — jnp.ndarray (2M, 2M) — desired covariance
#   target_mode_ids    — list of int — which modes to compare
#
# Returns:
#   scalar jnp float:  ½ · ‖ σ_sub − σ_target ‖²_F
#   (Frobenius norm squared, halved — so gradient = σ_sub − σ_target)
#
# Steps:
#   1. A        = build_drift_matrix(nodes, edges, coupling_strengths)
#   2. D        = build_diffusion_matrix(nodes)    ← constant; precompute outside JIT
#   3. sigma    = solve_lyapunov_kronecker(A, D)
#   4. sigma_sub = get_mode_covariance(sigma, target_mode_ids)
#   5. diff     = sigma_sub − target_cov
#   6. return   jnp.sum(diff ** 2) / 2.
#
# Usage pattern:
#   loss_jit  = jax.jit(covariance_loss, static_argnums=(1,2,4))
#   grad_loss = jax.jit(jax.grad(covariance_loss, argnums=0), static_argnums=(1,2,4))
#   value = loss_jit(coupling_strengths, nodes, edges, target_cov, target_mode_ids)
#   grads = grad_loss(coupling_strengths, nodes, edges, target_cov, target_mode_ids)
#
# Note: D does not depend on coupling_strengths, so pass it as a separate
#       argument rather than recomputing it inside the JIT-compiled function.
#       This avoids redundant computation across restarts.

def covariance_loss(
    coupling_strengths: jnp.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    target_cov: jnp.ndarray,
    target_mode_ids: List[int],
) -> jnp.ndarray:
    pass
