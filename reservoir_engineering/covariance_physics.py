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
FROM HAMILTONIAN TO DRIFT MATRIX  (how A is constructed)
═══════════════════════════════════════════════════════════════════════════════

The drift matrix A is built directly from the system Hamiltonian H via the
quantum Langevin equations (Heisenberg-Langevin formalism):

    ȧ_i = -i [a_i, H] - (κ_i / 2) a_i + sqrt(κ_i) a_in,i

Converting each mode operator to real quadratures  x = (a+a†)/√2,
p = -i(a-a†)/√2  yields a LINEAR equation of motion in vector form:

    dq/dt = A · q + noise

The drift matrix A is assembled block-by-block from H:

  Hamiltonian term            →  block of A
  ──────────────────────────────────────────────────────────────────
  (system-bath coupling)      →  -κ_i/2 · I₂  (diagonal decay block)
  Δ_i a_i†a_i  (detuning)   →  +Δ_i · J₂    (diagonal rotation block)
  g(a†b + ab†) (beamsplitter) →  g · J₂       (off-diagonal block, both sides symmetric)
  ν(a†b† + ab) (TMS)         →  -ν · σ_x    (both off-diagonal blocks, same sign, symmetric)

where J₂ = [[0,1],[-1,0]], σ_x = [[0,1],[1,0]].

The key distinction between BS and TMS comes from the commutator:
  [a_i, g(a_i†a_j + h.c.)] = g a_j    → ȧ_i = -ig·a_j
    ẋ_i = +g·p_j,  ṗ_i = -g·x_j      → g · J₂ block (antisymmetric rotation)
  [a_i, ν(a_i†a_j† + h.c.)] = ν a_j† → ȧ_i = -iν·a_j†
    ẋ_i = -ν·p_j,  ṗ_i = -ν·x_j      → -ν · σ_x block (SAME sign in both off-diagonal blocks)

A is split into TWO parts — exactly mirroring AutoScatter's (-iH - κ/2):

    A  =  H_quad  +  A_decay
          └─ from H ─┘   └─ from dissipation ─┘

  H_quad  (2N × 2N real) — the EXPLICIT Hamiltonian matrix in the quadrature basis.
                            The direct analogue of AutoScatter's complex N×N coupling matrix H.
                            Contains ONLY the coherent (Hamiltonian) terms:
                              diagonal blocks:    Δ_i · J₂         (from H = Δ_i a_i†a_i)
                              BS off-diagonal:   +g · J₂, +g · J₂    (from H = g(a†b + ab†), symmetric)
                              TMS off-diagonal:  -g · σ_x, -g · σ_x  (from H = g(a†b† + ab), symmetric)

  A_decay (2N × 2N real) — the dissipation part.  NOT from H.  From system-bath coupling.
                            diagonal blocks only:  -decay_i/2 · I₂  per mode i

  build_hamiltonian_matrix(nodes, edges, coupling_strengths) → H_quad
  build_drift_matrix(nodes, edges, coupling_strengths)       → A = H_quad + A_decay

AutoScatter analogy:
  AutoScatter:         -iH  (complex N×N, from Hamiltonian)  -  κ/2  (decay, separate)
  Reservoir Eng:    H_quad  (real 2N×2N, from Hamiltonian)   + A_decay (decay, separate)
  Both: the Hamiltonian and the decay are SEPARATE contributions combined into one matrix.

═══════════════════════════════════════════════════════════════════════════════
OPEN SYSTEM DYNAMICS  (Lyapunov equation)
═══════════════════════════════════════════════════════════════════════════════

The system evolves under the linear quantum Langevin equation:

    dq/dt = A · q + noise        where A = H_quad + A_decay

In steady state, the covariance matrix σ (where σ_ij = ½⟨{q_i,q_j}⟩)
satisfies the continuous-time Lyapunov equation:

    A · σ + σ · Aᵀ + D = 0

where:
  A  — drift matrix   (2N × 2N, real) — A = H_quad + A_decay
                       H_quad encodes the coherent Hamiltonian (couplings + detunings)
                       A_decay encodes the dissipation (-decay/2 per mode)
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
        'delta': float,          # effective detuning in the rotating frame (default 0.0)
                                 # appears in A as a rotation on the diagonal 2×2 block:
                                 #   A[s_i, s_i] += delta_i · [[0, 1], [-1, 0]]
                                 # = 0 for all modes in the standard doubly-rotating frame
                                 #   (Kronwald, Wang-Clerk: all delta_i = 0).
                                 # ≠ 0 when two or more modes cannot simultaneously be
                                 #   in their own rotating frames — e.g. two mechanical
                                 #   modes with different frequencies ω_m1 ≠ ω_m2: one is
                                 #   in its rotating frame (delta=0), the other has a
                                 #   residual detuning delta = ω_m2 − ω_m1.
                                 # Also ≠ 0 if the drive is intentionally off-resonance
                                 #   (e.g. to improve stability or modify squeezing direction).
    }

═══════════════════════════════════════════════════════════════════════════════
EDGE TYPES  (what kind of coupling connects two graph nodes?)
═══════════════════════════════════════════════════════════════════════════════

'beamsplitter'  (BS)  — Hamiltonian H = g (a†b + ab†)
    Energy-conserving swap interaction (two-tone red sideband drive).
    In the real quadrature basis (J₂ = [[0,1],[-1,0]]):
        H_quad[2i:2i+2, 2j:2j+2] +=  g · J₂     (i → j block)
        H_quad[2j:2j+2, 2i:2i+2] +=  g · J₂     (j → i block, symmetric — same matrix)
    [a,H] = g·b → ȧ = -ig·b → ẋ_i = g·p_j, ṗ_i = -g·x_j → g·J₂ block.
    Physical effect: swaps state between modes i and j.

'two_mode_squeezing'  (TMS)  — H = ν (a†b† + ab)
    Parametric amplification interaction (two-tone blue sideband drive).
    In the real quadrature basis (σ_x = [[0,1],[1,0]]):
        H_quad[2i:2i+2, 2j:2j+2] +=  -ν · σ_x   (i → j block)
        H_quad[2j:2j+2, 2i:2i+2] +=  -ν · σ_x   (j → i block, same sign — symmetric)
    [a,H] = ν·b† → ȧ = -iν·b† → ẋ_i = -ν·p_j, ṗ_i = -ν·x_j → -ν·σ_x block.
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

    return slice (2*mode_id, 2 * mode_id + 2)


# ───────────────────────────────────────────────────────────────────────────
# build_hamiltonian_matrix(nodes, edges, coupling_strengths)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Assemble the 2N × 2N real Hamiltonian matrix H_quad from the system
#   Hamiltonian H in the real quadrature basis.
#
#   H_quad is the EXPLICIT Hamiltonian matrix — the direct analogue of
#   AutoScatter's complex N×N coupling matrix H.
#   It contains ONLY the coherent (Hamiltonian) contributions.
#   Decay terms (-κ/2 · I₂) are NOT included here — they are added separately
#   in build_drift_matrix to form A = H_quad + A_decay.
#
#   AutoScatter analogy:
#     AutoScatter:       H (complex N×N)   — explicit Hamiltonian matrix
#     Reservoir Eng: H_quad (real 2N×2N)  — explicit Hamiltonian matrix
#     Both used as:  S = f(-iH, -κ/2)  ↔  A = H_quad + A_decay
#
# Parameters:
#   nodes              — list of N node dicts (reads 'delta' field per node)
#   edges              — list of E edge dicts (reads 'type' field per edge)
#   coupling_strengths — jnp.ndarray shape (E,), one value per edge
#
# Returns:
#   H_quad — jnp.ndarray shape (2N, 2N), real
#            Contains detuning and coupling terms only (no decay).
#
# Construction:
#
#   Step 1 — diagonal detuning blocks (one per node, from H = Δ_i a_i†a_i):
#     For every node i:
#         J₂ = [[0, 1], [-1, 0]]
#         H_quad[s_i, s_i] += delta_i · J₂
#     If delta_i = 0 (standard doubly-rotating frame): this block is zero.
#     If delta_i ≠ 0: the mode precesses in phase space at rate delta_i.
#
#   Step 2 — off-diagonal coupling blocks (one per edge, from H coupling terms):
#     Let g = coupling_strengths[edge_idx], s_i = quadrature_slice(edge.i),
#             s_j = quadrature_slice(edge.j)
#
#     'beamsplitter' (H = g(a†b + ab†)):
#         [a_i, H] = g·a_j  →  ȧ_i = -ig·a_j
#         ẋ_i = +g·p_j,  ṗ_i = -g·x_j  →  block = g · J₂
#         H_quad[s_i, s_j] += g · J₂
#         H_quad[s_j, s_i] += g · J₂         (same matrix both sides — symmetric)
#
#     'two_mode_squeezing' (H = g(a†b† + ab)):
#         [a_i, H] = g·a_j†  →  ȧ_i = -ig·a_j†
#         ẋ_i = -g·p_j,  ṗ_i = -g·x_j  →  block = -g · σ_x
#         H_quad[s_i, s_j] += -g · σ_x
#         H_quad[s_j, s_i] += -g · σ_x       (same sign in both blocks — symmetric)
#
#     'parametric' (H = χ(a² + a†²), single-mode):
#         H_quad[s_i, s_i] += χ · σ_z
#
# Why H_quad is real (not complex like AutoScatter's H):
#   In the complex basis, H[i,j] = g·e^{iφ} (complex — encodes phase).
#   In the real quadrature basis, the phase is replaced by the choice of block
#   structure (I₂ = BS, σ_z = TMS). No complex numbers needed.
#   This is why there is no phase optimisation variable in this project.

def build_hamiltonian_matrix(
    nodes: List[Dict],
    edges: List[Dict],
    coupling_strengths: jnp.ndarray,) -> jnp.ndarray:

    N  = len(nodes)
    H  = jnp.zeros((2 * N, 2 * N))
    J2 = jnp.array([[0.,  1.], [-1., 0.]])  # [[0,1],[-1,0]] — used for detuning AND BS
    sx = jnp.array([[0.,  1.], [ 1., 0.]])  # σ_x = [[0,1],[1,0]] — used for TMS
    sz = jnp.array([[1.,  0.], [ 0., -1.]]) # σ_z = [[1,0],[0,-1]] — used for parametric

    # Step 1 — diagonal detuning blocks (from H = Δ_i a_i†a_i)
    # [a_i, Δ_i a_i†a_i] = Δ_i a_i  →  ȧ_i = -iΔ_i a_i
    # → ẋ_i = +Δ_i p_i,  ṗ_i = -Δ_i x_i  →  Δ_i · J₂
    for i, node in enumerate(nodes):
        s     = quadrature_slice(i)
        delta = node.get('delta', 0.0)
        H     = H.at[s, s].add(delta * J2)

    # Step 2 — off-diagonal coupling blocks (from H coupling terms)
    for k, edge in enumerate(edges):
        si = quadrature_slice(edge['i'])
        sj = quadrature_slice(edge['j'])
        g  = coupling_strengths[k]

        if edge['type'] == 'beamsplitter':
            # H = g(a†b + ab†)  →  [a,H] = g·b  →  ȧ = -ig·b
            # ẋ_i = +g·p_j,  ṗ_i = -g·x_j  →  block = g·J₂
            # ẋ_j = +g·p_i,  ṗ_j = -g·x_i  →  block = g·J₂  (same)
            H = H.at[si, sj].add( g * J2)   # C_{i,j} = +g·J₂
            H = H.at[sj, si].add( g * J2)   # C_{j,i} = +g·J₂  (symmetric)

        elif edge['type'] == 'two_mode_squeezing':
            # H = g(a†b† + ab)  →  [a,H] = g·b†  →  ȧ = -ig·b†
            # ẋ_i = -g·p_j,  ṗ_i = -g·x_j  →  block = -g·σ_x
            # ẋ_j = -g·p_i,  ṗ_j = -g·x_i  →  block = -g·σ_x  (same sign, symmetric)
            H = H.at[si, sj].add(-g * sx)   # C_{i,j} = -g·σ_x
            H = H.at[sj, si].add(-g * sx)   # C_{j,i} = -g·σ_x  (same sign)

        elif edge['type'] == 'parametric':
            # H = χ(a² + a†²), single-mode squeezing
            H = H.at[si, si].add(g * sz)    # diagonal σ_z block

    return H

# ───────────────────────────────────────────────────────────────────────────
# build_drift_matrix(nodes, edges, coupling_strengths)
# ───────────────────────────────────────────────────────────────────────────
# Purpose:
#   Assemble the full 2N × 2N drift matrix A = H_quad + A_decay.
#
#   H_quad  = build_hamiltonian_matrix(nodes, edges, coupling_strengths)
#             (coherent Hamiltonian part — couplings + detunings)
#   A_decay = diagonal -decay_i/2 · I₂ blocks
#             (dissipation part — from system-bath coupling, NOT from H)
#
#   This separation mirrors AutoScatter exactly:
#     AutoScatter uses:  -iH  (Hamiltonian)  +  -κ/2  (decay)  in S-matrix
#     Here:           H_quad  (Hamiltonian)  +  A_decay (decay) = A
#
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
# Construction:
#   A = H_quad + A_decay
#
#   Part 1 — H_quad  (call build_hamiltonian_matrix):
#     Step 1a — diagonal detuning blocks (from H = Δ_i a_i†a_i):
#       For every node i: H_quad[s_i, s_i] += delta_i · J₂
#     Step 1b — off-diagonal coupling blocks (from H coupling terms):
#       BS:  H_quad[s_i,s_j] += +g·I₂,  H_quad[s_j,s_i] += +g·I₂
#       TMS: H_quad[s_i,s_j] += +g·σ_z, H_quad[s_j,s_i] += -g·σ_z
#
#   Part 2 — A_decay  (dissipation, NOT from H):
#     Step 2 — diagonal decay blocks (one per node):
#     For node i of type 'cavity':     A_decay[s_i, s_i] = −(κ_i / 2) · I₂
#     For node i of type 'mechanical': A_decay[s_i, s_i] = −(γ_i / 2) · I₂
#     where s_i = quadrature_slice(i) and I₂ = identity(2)
#
#   Return: A = H_quad + A_decay
#
#   (Kept as a single function for JAX efficiency — avoids building two
#    separate matrices and adding them. Internally assembles A in one pass.)
#
#   Step 1a — diagonal detuning rotation blocks (one per node, if delta != 0):
#     For every node i with detuning delta_i = node['delta'] (default 0):
#         J₂ = [[0, 1], [-1, 0]]    (2×2 antisymmetric rotation generator)
#         A[s_i, s_i] += delta_i · J₂
#
#     Physical meaning of delta_i:
#       In the rotating frame, delta_i is the mode's frequency offset from the
#       frame rotation frequency. It causes the quadratures to rotate at rate delta_i
#       (analogous to a free-running oscillator precessing in phase space).
#       For a cavity at detuning delta_i:
#           A_ii = [[-κ/2,    delta_i],
#                   [-delta_i, -κ/2  ]]
#       For delta_i = 0: purely decaying (Kronwald standard case).
#       For delta_i ≠ 0: the squeezed quadrature rotates, affecting the steady-state
#       squeezing angle. The optimizer can use delta_i to align squeezing direction
#       with the target covariance.
#
#     When to set delta_i ≠ 0:
#       (a) Multi-mechanical-frequency systems: only one ω_m can be removed by the
#           rotating frame; other mechanical modes get delta = ω_m_other − ω_m_ref.
#       (b) When the scheme requires an off-resonance drive to achieve the target.
#           (AutoScatter's Δ_i variables play this exact role.)
#       (c) When the squeezing direction of the target covariance is rotated relative
#           to the natural x-quadrature — a non-zero delta_i rotates the squeezing.
#
#   Step 2 — off-diagonal coupling blocks (one per edge):
#     Let g = coupling_strengths[edge_idx], s_i = quadrature_slice(edge.i),
#     s_j = quadrature_slice(edge.j), σ_z = diag(+1, -1)
#
#     'beamsplitter':
#         H_quad[s_i, s_j] +=  g · J₂
#         H_quad[s_j, s_i] +=  g · J₂        (symmetric — same block both sides)
#
#     'two_mode_squeezing':
#         H_quad[s_i, s_j] += -g · σ_x
#         H_quad[s_j, s_i] += -g · σ_x       (same sign — symmetric)
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
    
    H_quad = build_hamiltonian_matrix(nodes=nodes, edges=edges, coupling_strengths=coupling_strengths)
    N = len(nodes)
    A = H_quad
    I2 = jnp.eye(2)

    for i, node in enumerate(nodes):
        s = quadrature_slice(i)
        decay = node["kappa"] if node["type"] == "cavity" else node["gamma"]
        A = A.at[s, s].add((-decay/2) * I2)
    
    return A

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
# 
# Example of nodes dictionary
# nodes = [
# {'id': 0, 'type': 'cavity',     'kappa': 1.0, 'delta': 0.0},
# {'id': 1, 'type': 'mechanical', 'gamma': 0.01, 'n_th': 0.0, 'delta': 0.0}]

def build_diffusion_matrix(nodes: List[Dict]) -> jnp.ndarray: 

    N = len(nodes)

    diag = []
    for node in nodes: 
        if node["type"] == "cavity":
            noise = node["kappa"] / 2
        else:
            noise = node["gamma"] * (node["n_th"] + 1/2)
        diag += [noise, noise]

    D = jnp.diag(jnp.array(diag))

    return D

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

    n = A.shape[0]
    I = jnp.eye(n)
    M = jnp.kron(I,A) + jnp.kron(A,I)
    vec_sigma = jnp.linalg.solve(M, -D.flatten())

    return vec_sigma.reshape(n,n)

# ───────────────────────────────────────────────────────────────────────────
# check_stability(A) → bool
# ───────────────────────────────────────────────────────────────────────────
# Purpose (Stage 1 of the main algorithm):
#   Check whether the drift matrix A is Hurwitz — all eigenvalues have
#   strictly negative real parts.  Called BEFORE any gradient optimisation.
#   If False, the system has no steady state and the topology is discarded
#   immediately without touching the Lyapunov solver.
#
# Parameters:
#   A — np.ndarray or jnp.ndarray (2N, 2N) drift matrix
#
# Returns:
#   bool — True if all Re(λ_k) < 0 (stable), False otherwise
#
# Method:
#   eigenvalues = np.linalg.eigvals(np.asarray(A))
#   return bool(np.all(np.real(eigenvalues) < 0))
#
# Usage in Stage 1:
#   Called by check_stability_unit_cooperativity() in CovarianceOptimizer,
#   which builds A with every cooperativity = 1.  A topology that is unstable
#   at unit cooperativity cannot be stabilised by changing coupling ratios —
#   the instability is structural (determined by edge TYPES and graph shape,
#   not coupling magnitudes).  TMS edges on a cycle with no BS edges are the
#   archetypal structurally-unstable case.
#
# Note:
#   Uses numpy (not JAX) because it is called outside the differentiable
#   computation graph — purely as a discrete topology filter.  The actual
#   loss functions (covariance_loss, covariance_loss_from_ratios) remain
#   fully JAX-differentiable.

def check_stability(A, tolerance: float = 1e-10) -> bool:
    eigs = np.linalg.eigvals(np.asarray(A))
    return bool(np.all(np.real(eigs) < tolerance))


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
# => return the relavant submatrix that is the target state
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
    
    idx = []
    for m in mode_ids:
        idx += [2*m, 2*m + 1]

    return sigma[np.ix_(idx, idx)]

# ───────────────────────────────────────────────────────────────────────────
# build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
# ───────────────────────────────────────────────────────────────────────────
# Purpose (Stages 2 and 3 of the main algorithm):
#   Build the drift matrix A using the cooperativity reparametrisation.
#   Instead of optimising coupling strengths g_{ij} directly, the optimiser
#   works with dimensionless coupling RATIOS C̃_i = exp(u_i) at a fixed
#   large scale λ = lambda_scale.
#
#   The mapping from ratios to coupling strengths is:
#
#       C_{ij}  = lambda_scale  ×  ratios[k]              (cooperativity)
#       g_{ij}  = sqrt( C_{ij} × decay_i × decay_j / 4 ) (coupling strength)
#
#   where:
#       decay_i = kappa_i  if node i is a cavity
#       decay_i = gamma_i  if node i is mechanical
#
#   In the log parametrisation used by the optimiser:
#       u_k   = log(C̃_k)         — the actual free variable (unconstrained real)
#       ratios[k] = exp(u_k)      — caller computes exp before passing in
#       C_k   = lambda_scale · exp(u_k)
#       g_k   = sqrt(lambda_scale · exp(u_k) · decay_i · decay_j / 4)
#
#   This function converts ratios → coupling strengths and calls build_drift_matrix.
#   Fully JAX-differentiable with respect to `ratios`.
#
# Parameters:
#   nodes        — list of N node dicts (same format as build_drift_matrix)
#   edges        — list of E edge dicts (types only; edge k ↔ ratios[k])
#   ratios       — jnp.ndarray shape (E,), C̃_i > 0 (pass jnp.exp(u) from optimiser)
#   lambda_scale — float, the fixed scale (default LAMBDA_SCALE_DEFAULT = 1000).
#                  Must satisfy lambda_scale >> 1 to be in the strong-coupling limit.
#                  Also used at λ=10,100 during Stage 2 convergence test.
#
# Returns:
#   A — jnp.ndarray (2N, 2N), drift matrix built from the converted g values
#
# Construction:
#   For each edge k with nodes i and j:
#       decay_i = node[i]['kappa'] if 'cavity' else node[i]['gamma']
#       decay_j = node[j]['kappa'] if 'cavity' else node[j]['gamma']
#       C_k  = lambda_scale * ratios[k]
#       g_k  = jnp.sqrt(C_k * decay_i * decay_j / 4.)
#   coupling_strengths = jnp.array([g_0, g_1, ..., g_{E-1}])
#   return build_drift_matrix(nodes, edges, coupling_strengths)
#
# Why this reparametrisation?
#   At large lambda_scale, g_{ij} ~ sqrt(lambda_scale) can be very large.
#   C̃_i = C_i / lambda_scale is O(1) regardless of lambda, so the gradient
#   landscape is well-conditioned numerically.
#   The log parametrisation u_k = log(C̃_k) makes C̃_k automatically positive
#   and gives multiplicative gradient steps — no lower bounds needed.

"""
def build_drift_matrix(
    nodes: List[Dict],
    edges: List[Dict],
    coupling_strengths: jnp.ndarray,

"""

#
# Differentiability:
#   All operations (jnp.sqrt, multiplication, build_drift_matrix) are differentiable.
#   jax.grad(covariance_loss_from_ratios, argnums=0) flows through this function.
#   Enforce ratios > 0 via L-BFGS-B lower bounds to avoid sqrt singularity.

def build_drift_matrix_from_ratios(
    nodes: List[Dict],
    edges: List[Dict],
    ratios: jnp.ndarray,
    lambda_scale: float,
) -> jnp.ndarray:

    coupling_strengths = []

    for k, edge in enumerate(edges):
        i, j = edge["i"], edge["j"]
        decay_i = nodes[i]["kappa"] if nodes[i]["type"] == "cavity" else nodes[i]["gamma"]
        decay_j = nodes[j]["kappa"] if nodes[j]["type"] == "cavity" else nodes[j]["gamma"]

        C_k = lambda_scale * ratios[k]
        g_k = jnp.sqrt(C_k * decay_i * decay_j / 4)
        coupling_strengths.append(g_k)

    return build_drift_matrix(nodes, edges, jnp.array(coupling_strengths))
    
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
    
    A = build_drift_matrix(nodes=nodes, edges=edges, coupling_strengths=coupling_strengths)
    D = build_diffusion_matrix(nodes)

    sigma = solve_lyapunov_kronecker(A,D)
    sigma_sub = get_mode_covariance(sigma, target_mode_ids)
    diff = sigma_sub - target_cov
    loss = jnp.sum(diff**2)/2

    return loss

# ───────────────────────────────────────────────────────────────────────────
# covariance_loss_from_ratios(log_ratios, nodes, edges, D, lambda_scale,
#                              target_cov, target_mode_ids)
# ───────────────────────────────────────────────────────────────────────────
# Purpose (Stages 2 and 3 of the main algorithm):
#   The scalar loss function for continuous optimisation over coupling ratios.
#   The optimiser works in LOG SPACE: free variable u_k = log(C̃_k).
#   Passing log_ratios = u_k, this function computes ratios = exp(u_k) then
#   calls build_drift_matrix_from_ratios. Fully JAX-differentiable.
#
# Parameters:
#   log_ratios      — jnp.ndarray (E,), u_k = log(C̃_k), unconstrained reals.
#                     C̃_k = exp(u_k) > 0 automatically.
#   nodes           — list of N node dicts (types, decay rates)
#   edges           — list of E edge dicts (types only)
#   D               — jnp.ndarray (2N, 2N), precomputed diffusion matrix (constant).
#                     Pass D in from outside the JIT-compiled call for efficiency.
#   lambda_scale    — float, fixed scale (LAMBDA_SCALE_DEFAULT=1000 for Stage 3;
#                     smaller values [10, 100] used in Stage 2 convergence test)
#   target_cov      — jnp.ndarray (2M, 2M), target covariance for signal modes
#   target_mode_ids — list of int, which modes to compare
#
# Returns:
#   scalar jnp float:  ½ · ‖ σ_sub − σ_target ‖²_F
#
# Steps:
#   1. ratios    = jnp.exp(log_ratios)                          ← log → linear
#   2. A         = build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
#   3. sigma     = solve_lyapunov_kronecker(A, D)
#   4. sigma_sub = get_mode_covariance(sigma, target_mode_ids)
#   5. diff      = sigma_sub − target_cov
#   6. return    jnp.sum(diff ** 2) / 2.
#
# Log parametrisation benefits:
#   • C̃_k = exp(u_k) > 0 always — no bounds or clipping needed
#   • Gradient steps are multiplicative (10% change regardless of scale)
#   • No runaway: instability raises the loss before exp(u_k) diverges
#
# Usage pattern:
#   loss_jit  = jax.jit(covariance_loss_from_ratios, static_argnums=(1,2,5,6))
#   grad_loss = jax.jit(jax.grad(covariance_loss_from_ratios, argnums=0),
#                        static_argnums=(1,2,5,6))
#   value = loss_jit(log_ratios, nodes, edges, D, lambda_scale, sigma_target, mode_ids)
#   grads = grad_loss(log_ratios, nodes, edges, D, lambda_scale, sigma_target, mode_ids)

def covariance_loss_from_ratios(
    log_ratios: jnp.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    D: jnp.ndarray,
    lambda_scale: float,
    target_cov: jnp.ndarray,
    target_mode_ids: List[int],
) -> jnp.ndarray:

    ratios = jnp.exp(log_ratios)
    A = build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
    D = build_diffusion_matrix(nodes)

    sigma = solve_lyapunov_kronecker(A,D)
    sigma_sub = get_mode_covariance(sigma, target_mode_ids)
    diff = sigma_sub - target_cov
    loss = jnp.sum(diff**2)/2

    return loss
