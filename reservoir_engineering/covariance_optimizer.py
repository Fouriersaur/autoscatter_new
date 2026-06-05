"""
covariance_optimizer.py
=======================
Main optimiser class for covariance-matrix targeting.

MIRRORS: autoscatter/architecture_optimizer.py  (1-to-1 conceptual mapping)

═══════════════════════════════════════════════════════════════════════════════
ROLE IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════

This file contains CovarianceOptimizer, the central class of the package.
It mirrors Architecture_Optimizer in AutoScatter essentially one-to-one:

    AutoScatter                          Reservoir Engineering
    ─────────────────────────────────    ─────────────────────────────────
    Architecture_Optimizer               CovarianceOptimizer
    S_target (sympy Matrix)          →   sigma_target (numpy array, 2M×2M)
    mode_types (list of bool)        →   node_types (list of 'cavity'|'mechanical')
    num_auxiliary_modes (int)        →   num_auxiliary_modes (int)
    gabs, gphases, Deltas (complex)  →   C̃_k (ratios), Δ_i (detunings) — real
    gauge phases (complex)           →   (none — σ is gauge-invariant)
    coupling_matrix H (complex N×N)  →   H_quad (real 2N×2N) — EXPLICIT Hamiltonian matrix
                                         (analogue of H: detuning + coupling blocks, no decay)
    decay separate: -κ/2 in S-matrix →   A_decay separate: -decay/2·I₂ diagonal blocks
    A = -iH - κ/2  (combined)        →   A = H_quad + A_decay  (combined drift matrix)
    kappa_int_matrix                 →   (absorbed in D via γ, n_th)
    S = I + (-iH - κ/2)⁻¹           →   Aσ + σAᵀ + D = 0
    Frobenius loss on S              →   Frobenius loss on σ_sub
    C_{i,j} = 4|g_{i,j}|²          →   C_{i,j} = 4g_{i,j}²/(κ_i·κ_j)

The class has TWO responsibilities (exactly as in AutoScatter):
  1. INNER LOOP — parameter optimisation for a fixed topology.
     Given a condition list (which edges are absent / what type),
     find coupling_strengths that minimise the covariance loss.
     Methods: optimize_given_conditions, repeated_optimization.

  2. OUTER LOOP — topology discovery via breadth-first search.
     Enumerate all possible topologies (by complexity level),
     test each with the inner loop, prune using subgraph rules.
     Methods: prepare_all_possible_combinations, find_valid_combinations,
              identify_potential_combinations, cleanup_valid_combinations,
              perform_breadth_first_search.

═══════════════════════════════════════════════════════════════════════════════
KEY SIMPLIFICATIONS VS AUTOSCATTER
═══════════════════════════════════════════════════════════════════════════════

1. NO SYMPY — AutoScatter builds a symbolic H matrix (sympy) and lambdifies
   it to JAX. Here, A is built directly in JAX from edge dicts. No sympy needed.
   Simpler, but loses the ability to have free symbols in sigma_target.

2. REAL COUPLING STRENGTHS — In AutoScatter, couplings are complex (gabs + gphase).
   In the real quadrature basis, A is real, so all couplings are real numbers.
   Fewer optimisation variables: one scalar per edge instead of two (no phase).

3. NO GAUGE FREEDOM — AutoScatter optimises over detector-position phases γ.
   Covariance matrices are gauge-invariant: σ doesn't depend on the reference phase.
   No gauge phases needed.

4. STABILITY ENFORCEMENT — AutoScatter doesn't need this (passive systems are
   always stable). Here, TMS edges can destabilise A. Handled via a penalty
   term (Constraint_stability) added to the loss, or via log-reparametrisation.

5. DIFFUSION MATRIX D IS FIXED — D depends only on node types and physical
   parameters (κ, γ, n_th), NOT on coupling strengths. Precomputed once.
   In AutoScatter, the kappa_int_matrix can optionally be optimised over
   (port_intrinsic_losses). Here, decay rates are fixed inputs.

═══════════════════════════════════════════════════════════════════════════════
COOPERATIVITY  (mirrors extract_cooperativities in Architecture_Optimizer)
═══════════════════════════════════════════════════════════════════════════════

In AutoScatter:   C_{i,j} = 4 |g_{i,j}|²   (with κ_ext = 1 normalised away)
Here:             C_{i,j} = 4 g_{i,j}² / (κ_i · κ_j)

where κ_i, κ_j are the decay rates of modes i and j.
For a cavity-mechanical pair: C = 4g² / (κ · γ) — the optomechanical cooperativity.

Physical meaning: C > 1 means strong coupling (interaction faster than decay).
Kronwald condition: g > ν > 0 with C_g - C_ν > some threshold for stability.

The cooperativity dict is returned alongside the solution dict in the
result of optimize_given_conditions, exactly as in AutoScatter.

═══════════════════════════════════════════════════════════════════════════════
THREE-STAGE ALGORITHM  (the full pipeline inside find_valid_combinations)
═══════════════════════════════════════════════════════════════════════════════

For every candidate topology produced by the BFS outer loop, the algorithm
runs three internal stages before accepting or rejecting that topology.

STAGE 1 — Stability check at unit cooperativity  (fast discrete filter)
    Build A from H with all cooperativities C_{ij} = 1.
    "Unit cooperativity" means each Hamiltonian coupling g_{ij} is set to
    the geometric-mean threshold: g_{ij} = sqrt(decay_i × decay_j / 4).
    A is assembled via the H → A map (build_drift_matrix): one block per
    Hamiltonian term.  Then check whether all eigenvalues of A have strictly
    negative real part.
    If unstable → discard immediately.  Do NOT proceed to Stage 2 or 3.
    No gradient computation, no Lyapunov solve — just an eigenvalue check.
    Method: check_stability_unit_cooperativity(triu_array) → bool

STAGE 2 — Find scaling exponents {β_i}  (combinatorial, no gradient)
    Set C_i = λ^{β_i} (placeholder ratios C̃_i = 1) and solve the Lyapunov
    equation at λ ∈ [10, 100, 1000].  Extract V_system(λ) each time.
    Convergence criterion: V_system(λ=100) ≈ V_system(λ=1000) to within 5%.
    Start with same-scaling: all β_i = 1.  If convergence → done.
    If not: search over β_i ∈ EXPONENT_SEARCH_GRID_1 = {0.5, 1.0, 2.0}.
    If still no convergence: try EXPONENT_SEARCH_GRID_2 = {0.5, 1.0, 1.5, 2.0, 3.0}.
    If no combination converges → topology cannot realise a pure Gaussian state.
    Discard and add to self.invalid_combinations.
    Method: find_scaling_exponents(triu_array) → (betas, success)

STAGE 3 — Find coupling ratios {C̃_i} AND detunings {Δ_i}  (continuous optimisation)
    Fix λ = LAMBDA_SCALE_DEFAULT = 1000.
    Free variables — TWO GROUPS (mirrors AutoScatter's gabs + Deltas):
      Group 1 — coupling ratios:  C̃_i ∈ (0, ∞), one per active edge.
          Actual cooperativity:  C_i = λ^{β_i} · C̃_i
          Coupling strength:     g_i = sqrt(λ^{β_i} · C̃_i · decay_i · decay_j / 4)
          Initial guess:         C̃_i ~ Uniform(INIT_RATIO_RANGE_DEFAULT)
          Bounds:                C̃_i > 0 (enforced via L-BFGS-B lower bound)
      Group 2 — mode detunings:  Δ_i ∈ (−∞, +∞), one per mode i.
          Enter A as:            A[s_i, s_i] += Δ_i · [[0,1],[−1,0]]
          Initial guess:         Δ_i = 0 (resonant driving — zero detuning start)
          Bounds:                unbounded (Δ_i can be positive or negative)
    Loss: ½‖V_system − V_target‖²_F  (Frobenius on signal modes only)
    Optimiser: L-BFGS-B + JAX autodiff (same as AutoScatter).
    Multiple random initialisations for Group 1; Group 2 always starts at 0.
    Method: optimize_given_conditions(conditions, betas, lambda_scale) → (success, info_out)

    Why include detunings?
      AutoScatter optimises over Δ_i explicitly (they are in all_variables_list).
      Here, Δ_i are needed whenever:
        (a) The target covariance has a squeezed quadrature at a non-zero angle
            (Δ_i rotates the squeezing direction).
        (b) Multiple mechanical modes have different frequencies (only one can be in
            its own rotating frame; the others need residual detuning Δ ≠ 0).
        (c) Off-resonance driving improves stability or squeezing magnitude.
      For Kronwald / Wang-Clerk: Stage 3 optimizer will converge to Δ_i ≈ 0
      (confirming the resonant condition). This is a non-trivial output.

    Stage 2 uses Δ_i = 0 for all modes (resonant assumption).
    Stage 1 uses Δ_i = 0 for all modes (resonant stability check).

OUTPUT per successful topology:
    topology:          graph (nodes, edges, edge types)
    scaling_exponents: {β_i} per active edge  (from Stage 2)
    coupling_ratios:   {C̃_i} per active edge (from Stage 3, Group 1)
    detunings:         {Δ_i} per mode         (from Stage 3, Group 2)
    lambda_scale:      λ = 1000 (fixed)
    physical_formula:  g_i = sqrt(λ^{β_i} · C̃_i · decay_i · decay_j / 4)
    cooperativities:   C_i = λ^{β_i} · C̃_i  (the hardware requirement)

INTERPRETATION:
    If all β_i are equal (typically β_i = 1):
        Single overall scale knob — take all C_i >> 1 simultaneously.
        Squeezing / entanglement is set purely by the RATIOS {C̃_i}.
    If β_i differ across edges:
        Edges with larger β_i must be made parametrically stronger as
        the overall drive scale grows.  The physical coupling on edge i is
        g_i = sqrt(λ^{β_i} · C̃_i · κ_i · γ_i).
    If all Δ_i ≈ 0:  resonant driving suffices (Kronwald, Wang-Clerk cases).
    If some Δ_i ≠ 0: the scheme requires off-resonance driving — this is a
        non-trivial prediction about the laser frequency required.

No AutoScatter analogue for Stage 2.  AutoScatter optimises directly over
complex coupling magnitudes and phases; the scaling structure is implicit.
Here we make it explicit and discover it automatically in Stage 2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt
from tqdm import trange, tqdm
from itertools import product as itertools_product
from typing import List, Optional

jax.config.update("jax_enable_x64", True)

# ── Gradient method constants  (mirrors architecture_optimizer.py) ─────────
AUTODIFF_FORWARD  = 'autodiff_forward'
AUTODIFF_REVERSE  = 'autodiff_reverse'
DIFFERENCE_QUOTIENT = '2-point'

# ── Default optimisation hyperparameters (mirrors architecture_optimizer.py) ─
INIT_STRENGTH_RANGE_DEFAULT  = [0.01, 3.0]   # initial coupling strengths [g_lo, g_hi]
BOUNDS_STRENGTH_DEFAULT      = [0., np.inf]   # lower bound = 0 (strengths are non-negative)

# ── Stage 2 & 3 hyperparameters (no AutoScatter analogue) ─────────────────
LAMBDA_SCALE_DEFAULT    = 1000.               # fixed large scale λ for Stage 3
INIT_RATIO_RANGE_DEFAULT = [0.1, 5.0]         # initial C̃_i draw range for Stage 3
BOUNDS_RATIO_DEFAULT    = [1e-6, np.inf]      # Stage 3 bounds: C̃_i > 0 strictly

# Exponent search grids for Stage 2 (β_i values to try per edge):
EXPONENT_SEARCH_GRID_1 = [0.5, 1.0, 2.0]            # first pass (3 values, fast)
EXPONENT_SEARCH_GRID_2 = [0.5, 1.0, 1.5, 2.0, 3.0]  # expanded (5 values, slower)
# Stage 2 convergence tolerance: relative change in V_system between λ=100 and λ=1000
CONVERGENCE_TOL_DEFAULT = 0.05   # 5% relative tolerance


# ───────────────────────────────────────────────────────────────────────────
# STANDALONE FUNCTION: find_minimum_number_auxiliary_modes(...)
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: find_minimum_number_auxiliary_modes(S_target, start_value, max_value,
#          allow_squeezing, **kwargs_optimizer)
#          in autoscatter/architecture_optimizer.py  (EXACT analogue)
#
# Purpose:
#   Identify the MINIMUM number of auxiliary cavity modes required to realise
#   the target covariance sigma_target, starting from zero auxiliaries.
#
# Parameters:
#   sigma_target       : np.ndarray (2M, 2M) — target covariance for signal modes
#   target_mode_ids    : list of int — which modes are signal modes (e.g. mechanical)
#   node_types_signal  : list of str — types of the signal modes (e.g. ['mechanical'])
#   start_value        : int — start search from this many auxiliaries (default 0)
#   max_value          : int — stop search at this many auxiliaries (default 5)
#   **kwargs_optimizer : passed to CovarianceOptimizer.__init__
#
# Algorithm:
#   for num_aux in range(start_value, max_value+1):
#       print('testing %i auxiliary modes' % num_aux)
#       Build node_types = node_types_signal + ['cavity'] * num_aux
#       Build optimizer = CovarianceOptimizer(sigma_target, target_mode_ids,
#                                             node_types, num_auxiliary_modes=num_aux,
#                                             make_initial_test=False, ...)
#       success, _, _ = optimizer.repeated_optimization(conditions=[], ...)
#       if success:
#           print('minimum auxiliary modes: %i' % num_aux)
#           return optimizer
#   return None
#
# Returns:
#   CovarianceOptimizer instance configured for the minimum found, or None.
#
# Example:
#   # Discover how many cavities are needed to squeeze a mechanical mode
#   optimizer = find_minimum_number_auxiliary_modes(
#       sigma_target    = squeezed_vacuum(r=1.0),
#       target_mode_ids = [0],
#       node_types_signal = ['mechanical'],
#       start_value=0, max_value=3,
#   )
#   # Should return an optimizer with 1 auxiliary cavity (Kronwald topology)

def find_minimum_number_auxiliary_modes(
    sigma_target,
    target_mode_ids: List[int],
    node_types_signal: List[str],
    start_value: int = 0,
    max_value: int = 5,
    **kwargs_optimizer,
):
    pass


# ───────────────────────────────────────────────────────────────────────────
# CLASS: CovarianceOptimizer
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: Architecture_Optimizer in autoscatter/architecture_optimizer.py
#
# Central class of the package. Combines inner-loop parameter optimisation
# and outer-loop topology search, exactly as Architecture_Optimizer does.

class CovarianceOptimizer:

    # -----------------------------------------------------------------------
    # __init__(sigma_target, target_mode_ids, node_types,
    #          num_auxiliary_modes=0,
    #          gradient_method=AUTODIFF_REVERSE,
    #          kwargs_optimization={},
    #          solver_options={},
    #          enforced_constraints=[],
    #          make_initial_test=True)
    # -----------------------------------------------------------------------
    # MIRRORS: Architecture_Optimizer.__init__(S_target, num_auxiliary_modes,
    #          num_far_detuned_modes, mode_types, gradient_method,
    #          kwargs_optimization, solver_options, enforced_constraints,
    #          make_initial_test, ...)
    #
    # Parameters:
    #   sigma_target       : np.ndarray (2M, 2M) — target covariance for signal modes.
    #                        Mirrors S_target (sympy Matrix) in AutoScatter, but
    #                        purely numerical (no free symbols).
    #   target_mode_ids    : list of int — which modes to compare to sigma_target.
    #                        E.g. [1] means compare mode 1 to sigma_target.
    #                        Mirrors num_port_modes (AutoScatter knows S is for port modes).
    #   node_types         : list of str, length N — 'cavity' or 'mechanical'.
    #                        Mirrors mode_types (list of bool) in AutoScatter.
    #                        Length = num_signal_modes + num_auxiliary_modes.
    #   num_auxiliary_modes: int — how many nodes in node_types are auxiliary.
    #                        Mirrors num_auxiliary_modes in AutoScatter.
    #   gradient_method    : AUTODIFF_FORWARD | AUTODIFF_REVERSE | DIFFERENCE_QUOTIENT
    #                        Same as AutoScatter.
    #   kwargs_optimization: dict — mirrors Architecture_Optimizer.kwargs_optimization:
    #                          num_tests              (default 10) ← AutoScatter default
    #                          verbosity              (default 0)
    #                          init_strength_range    (default [0.01, 3.0])
    #                          max_violation_success  (default 1e-8)
    #                          interrupt_if_successful (default True)
    #   solver_options     : dict — passed to scipy.optimize.minimize:
    #                          maxiter  (default 2000 for L-BFGS-B)
    #                          ftol, gtol (default 0, 1e-12)
    #   enforced_constraints: list of Base_Constraint objects —
    #                        Added to the loss function for all topologies.
    #                        Constraint_stability() should always be included.
    #                        Mirrors enforced_constraints in AutoScatter (e.g.
    #                        MinimalAddedInputNoise for quantum-limited amplifiers).
    #   make_initial_test  : bool — if True, test the fully-connected graph immediately.
    #                        Raise Exception if even the dense graph fails.
    #                        Mirrors make_initial_test in AutoScatter exactly.
    #
    # Internally sets up:
    #   self.nodes, self.edges  ← default node/edge dicts from node_types
    #   self.D                  ← precomputed diffusion matrix (constant, from nodes)
    #   self.num_modes          ← len(node_types)
    #   self.num_signal_modes   ← len(target_mode_ids)
    #   self.num_auxiliary_modes← num_auxiliary_modes
    #   self.all_possible_edges ← list of all edge slots (i,j) with j>=i
    #   self.conditions_func    ← JIT-compiled loss + gradient (like AutoScatter)
    #   self.jacobian           ← JIT-compiled gradient function
    #   self.valid_combinations    ← list of valid triu_arrays found (like AutoScatter)
    #   self.invalid_combinations  ← list of invalid triu_arrays found
    #   self.tested_complexities   ← list of complexity levels tested
    #   self.num_tested_graphs     ← count per level
    #   self.num_tested_invalid_graphs ← count per level

    def __init__(
        self,
        sigma_target,
        target_mode_ids: List[int],
        node_types: List[str],
        num_auxiliary_modes: int = 0,
        gradient_method: str = AUTODIFF_REVERSE,
        kwargs_optimization: dict = {},
        solver_options: dict = {},
        enforced_constraints: list = [],
        make_initial_test: bool = True,
    ):
        pass

    # -----------------------------------------------------------------------
    # __setup_all_constraints__()
    # -----------------------------------------------------------------------
    # MIRRORS: __setup_all_constraints__ in Architecture_Optimizer
    #
    # Enumerate ALL possible architectural constraints for the N-mode system.
    # Stored in self.all_possible_constraints as a flat list.
    #
    # For each upper-triangle pair (i, j):
    #   Append Constraint_coupling_absent(i, j)
    #   If i != j:
    #       Append Constraint_coupling_beamsplitter(i, j)
    #       (TMS is the default unconstrained edge type — absent means no edge)
    # For each diagonal (i, i):
    #   Append Constraint_coupling_absent(i, i)  ← no single-mode squeezing
    #
    # Used by check_all_constraints to discover which constraints are
    # "accidentally" satisfied in a dense-graph solution.

    def __setup_all_constraints__(self):
        pass

    # -----------------------------------------------------------------------
    # __initialize_conditions_func__()
    # -----------------------------------------------------------------------
    # MIRRORS: __initialize_conditions_func__ in Architecture_Optimizer
    #
    # Build and JIT-compile the loss function and its gradient.
    # This is called once in __init__ and reused for all topology tests.
    #
    # The loss function (mirroring calc_conditions in AutoScatter):
    #
    #   def calc_conditions(coupling_strengths_free, conditions):
    #       # 1. Expand free params into full coupling_strengths array
    #       #    (absent edges fixed to 0, free edges from the input)
    #       full_cs = expand_to_full(coupling_strengths_free, conditions)
    #
    #       # 2. Build drift matrix A = H_quad + A_decay.
    #       #    H_quad = build_hamiltonian_matrix(nodes, edges, full_cs)
    #       #      — the EXPLICIT Hamiltonian matrix (analogue of AutoScatter's H)
    #       #      — contains ONLY coherent terms (couplings + detunings):
    #       #          BS  edge k:  H += g_k(a†b + h.c.)  →  H_quad block += g_k · I₂
    #       #          TMS edge k:  H += g_k(a†b† + h.c.) →  H_quad block += ±g_k · σ_z
    #       #          Detuning:    H += Δ_i a_i†a_i      →  H_quad diag += Δ_i · J₂
    #       #    A_decay = diagonal -decay_i/2 · I₂ blocks (dissipation, NOT from H)
    #       #    A = H_quad + A_decay   (mirrors AutoScatter: -iH + (-κ/2))
    #       A = build_drift_matrix(self.nodes, self.edges, full_cs)
    #
    #       # 3. Solve Lyapunov for steady-state covariance
    #       sigma = solve_lyapunov_kronecker(A, self.D)
    #
    #       # 4. Extract signal modes submatrix
    #       sigma_sub = get_mode_covariance(sigma, self.target_mode_ids)
    #
    #       # 5. Frobenius loss (same form as AutoScatter's S-matrix Frobenius loss)
    #       diff = sigma_sub - jnp.array(self.sigma_target)
    #       frobenius_loss = jnp.sum(jnp.abs(diff)**2) / 2.
    #
    #       # 6. Add enforced constraint penalties (like MinimalAddedInputNoise)
    #       penalty = sum(c(A, sigma) for c in self.enforced_constraints)
    #
    #       total_loss = frobenius_loss + penalty
    #       return total_loss, {'A': A, 'sigma': sigma}
    #
    # Returns: self.conditions_func (JIT-compiled), sets self.jacobian.
    # The jacobian is obtained via jax.jacrev or jax.jacfwd depending on
    # gradient_method — same pattern as Architecture_Optimizer.

    def __initialize_conditions_func__(self):
        pass

    # -----------------------------------------------------------------------
    # check_stability_unit_cooperativity(triu_array) → bool
    # -----------------------------------------------------------------------
    # STAGE 1 of the main algorithm.  Fast discrete topology filter — no
    # gradient computation, no Lyapunov solve.
    #
    # Build the drift matrix A with ALL cooperativities C_{ij} = 1 for every
    # active edge in the topology, then check that A is Hurwitz.
    #
    # "Unit cooperativity" means:
    #   g_{ij} = sqrt( decay_i × decay_j / 4 )
    #   where decay_i = kappa_i for cavity nodes, gamma_i for mechanical nodes.
    #
    # Parameters:
    #   triu_array — 1D int array encoding the topology (see topology_search.py)
    #
    # Returns:
    #   bool — True if A is Hurwitz (stable), False if any eigenvalue has Re ≥ 0
    #
    # Steps:
    #   1. Convert triu_array to nodes, edges via TopologyGraph(...).to_nodes_edges_dicts().
    #   2. For each active (non-zero) edge k between nodes i and j:
    #        decay_i = self.nodes[i]['kappa'] if cavity, else self.nodes[i]['gamma']
    #        decay_j = self.nodes[j]['kappa'] if cavity, else self.nodes[j]['gamma']
    #        g_unit_k = np.sqrt(decay_i * decay_j / 4.)
    #   3. Build unit_g_array = np.array([g_unit_0, ..., g_unit_{E-1}])
    #   4. A = build_drift_matrix(nodes, edges, jnp.array(unit_g_array))
    #   5. return check_stability(A)       ← from covariance_physics.py
    #
    # If this returns False:
    #   Log the topology as structurally unstable.
    #   Append triu_array to self.invalid_combinations.
    #   Do NOT call find_scaling_exponents or optimize_given_conditions.
    #
    # Physical intuition:
    #   TMS (blue-sideband) couplings on a cycle with no stabilising BS
    #   (red-sideband) couplings produce a runaway parametric amplifier.
    #   This is structural — no choice of coupling magnitude fixes it.
    #   Only a different edge type or graph topology resolves it.
    #
    # No AutoScatter analogue (AutoScatter's passive cavities are always stable).

    def check_stability_unit_cooperativity(self, triu_array) -> bool:
        pass

    # -----------------------------------------------------------------------
    # find_scaling_exponents(triu_array, lambda_values=None, beta_grid_1=None,
    #                         beta_grid_2=None, convergence_tol=None) → (betas, success)
    # -----------------------------------------------------------------------
    # STAGE 2 of the main algorithm.  Discover the scaling exponents {β_i}.
    # No gradient computation — only forward Lyapunov solves.
    #
    # For each candidate exponent combination {β_i}, set C_i = λ^{β_i} × 1
    # (unit ratios C̃_i = 1) and solve the Lyapunov equation at three values
    # of λ.  Extract V_system(λ) each time.  Check whether V_system converges:
    # all entries approach finite, non-zero values as λ grows.
    #
    # Parameters:
    #   triu_array      — 1D int array encoding the topology
    #   lambda_values   — list of 3 floats, default [10., 100., 1000.].
    #                     Must be increasing; convergence checked at last two values.
    #   beta_grid_1     — first search grid, default EXPONENT_SEARCH_GRID_1 = [0.5,1.0,2.0]
    #   beta_grid_2     — expanded grid, default EXPONENT_SEARCH_GRID_2 = [0.5,1.0,1.5,2.0,3.0]
    #   convergence_tol — float, relative tolerance for convergence, default CONVERGENCE_TOL_DEFAULT = 0.05
    #
    # Returns:
    #   (betas, success):
    #     betas   — list of E floats (β_i per active edge) if found, else None
    #     success — bool: True if a valid exponent combination was found
    #
    # Algorithm:
    #   Step A — same-scaling test (β_i = 1 for all edges):
    #       Set C_i = λ^1 for all edges.
    #       For λ in lambda_values:
    #           g_k = np.sqrt(λ * decay_k_i * decay_k_j / 4.)
    #           Build A, solve Lyapunov, extract V_system = get_mode_covariance(σ, target_mode_ids)
    #       convergence = (max|V_sys(λ_mid) − V_sys(λ_hi)| / max|V_sys(λ_hi)| < tol)
    #                   AND max|V_sys(λ_hi)| > 1e-12      ← not identically zero
    #       If converges: return ([1.0]*E, True)
    #
    #   Step B — grid search over beta_grid_1^E (first pass):
    #       For each combination (β_0, β_1, ..., β_{E-1}) in itertools.product(beta_grid_1, repeat=E):
    #           For λ in lambda_values:
    #               g_k = np.sqrt(λ^{β_k} * decay_k_i * decay_k_j / 4.)
    #               Build A.
    #               If not check_stability(A): skip this (β, λ) pair immediately.
    #               Else: solve Lyapunov, extract V_system.
    #           If converges: return (list(combination), True)
    #
    #   Step C — expanded grid search over beta_grid_2^E:
    #       Same as Step B but with the larger grid.
    #       (Only reached if Step B exhausts without convergence.)
    #
    #   Step D — failure:
    #       return (None, False)
    #       Log: 'topology has no finite pure-Gaussian steady state'
    #
    # Convergence check (used in Steps A, B, C):
    #   V_mid  = V_system at second-to-last λ (e.g. 100)
    #   V_hi   = V_system at last λ (e.g. 1000)
    #   scale  = max(np.abs(V_hi))
    #   converged = (max(np.abs(V_mid - V_hi)) / scale < convergence_tol)
    #               AND scale > 1e-12
    #
    # Complexity (E = number of active edges):
    #   Step A: 3 Lyapunov solves → negligible
    #   Step B: |grid_1|^E × 3 solves.  For E=2: 9×3=27.  For E=4: 81×3=243.
    #   Step C: |grid_2|^E × 3 solves.  For E=2: 25×3=75.  For E=6: ~47k (slow).
    #   Add early stopping: break as soon as convergence is found.
    #
    # Called by: find_valid_combinations, AFTER Stage 1 (stability check) passes.
    # If success=False: add triu_array to self.invalid_combinations, skip Stage 3.
    # If success=True:  pass betas to optimize_given_conditions for Stage 3.
    #
    # No AutoScatter analogue — unique to Reservoir Engineering.
    # Corresponds to discovering the "scaling structure" of the solution
    # before committing to continuous optimisation.

    def find_scaling_exponents(
        self,
        triu_array,
        lambda_values=None,
        beta_grid_1=None,
        beta_grid_2=None,
        convergence_tol: float = None,
    ) -> tuple:
        pass

    # -----------------------------------------------------------------------
    # give_free_variable_idxs(conditions) → list of int
    # -----------------------------------------------------------------------
    # MIRRORS: give_free_variable_idxs(conditions) in Architecture_Optimizer
    #
    # Given a list of constraints, return the indices into the full coupling
    # strengths array that are FREE (not fixed to zero by absent constraints).
    #
    # Build full_indices = list(range(len(all_edges)))
    # For each Constraint_coupling_absent(i,j) in conditions:
    #     Find the index k such that all_edges[k] == (i,j)
    #     Remove k from free_indices
    # Return free_indices
    #
    # Used by create_initial_guess, setup_bounds, and
    # give_conditions_func_with_conditions to handle topology constraints.

    def give_free_variable_idxs(self, conditions: list) -> list:
        pass

    # -----------------------------------------------------------------------
    # give_conditions_func_with_conditions(conditions) → (loss_fn, grad_fn, _)
    # -----------------------------------------------------------------------
    # MIRRORS: give_conditions_func_with_conditions(conditions)
    #          in Architecture_Optimizer  (EXACT analogue)
    #
    # Wrap self.conditions_func to only operate on the FREE coupling strengths
    # (those not fixed to 0 by Constraint_coupling_absent).
    #
    # Returns:
    #   calc_conditions_constrained(partial_cs) → (loss, aux_dict)
    #     partial_cs has shape (num_free_edges,), not (num_all_edges,).
    #     Internally pads with zeros for absent edges, then calls conditions_func.
    #   calc_jacobian_constrained(partial_cs) → gradient array, shape (num_free_edges,)
    #   _ (placeholder for Hessian, not implemented)
    #
    # This wrapping is EXACTLY what AutoScatter does: it constructs a full
    # parameter array padded with zeros for constrained variables, then
    # extracts the relevant gradient components.

    def give_conditions_func_with_conditions(self, conditions: list):
        pass

    # -----------------------------------------------------------------------
    # create_initial_guess(conditions=[], betas=None, optimize_detunings=True)
    #     → (initial_x, free_idxs)
    # -----------------------------------------------------------------------
    # MIRRORS: create_initial_guess(conditions, init_abs_range, ...)
    #          in Architecture_Optimizer
    #
    # Sample the initial parameter vector x = [C̃_i..., Δ_i...] for Stage 3.
    # Two groups of variables (mirrors AutoScatter's gabs + Deltas initial guess):
    #
    # Group 1 — coupling ratios (E_free entries):
    #   Draw C̃_i uniformly in INIT_RATIO_RANGE_DEFAULT = [0.1, 5.0].
    #   (Mirrors AutoScatter's uniform draw of |g_{ij}| in [init_abs_range[0], init_abs_range[1]].)
    #   Note: AutoScatter also draws gphases ∈ [-π, π]; here no phases needed
    #         (real quadrature basis = real A matrix, no phase degree of freedom).
    #
    # Group 2 — mode detunings (N entries, one per node):
    #   Δ_i = 0 for ALL modes (always start at resonance).
    #   Rationale: resonant driving is the physical default.  The optimizer
    #   will move Δ_i away from 0 if the target requires off-resonance driving.
    #   For Kronwald / Wang-Clerk: expect Δ_i ≈ 0 at convergence (confirming resonance).
    #   (Mirrors AutoScatter: Deltas initialised to 0 or to known resonance values.)
    #
    # Returns:
    #   initial_x  : np.ndarray (E_free + N,) — concatenated [C̃..., Δ...]
    #   free_idxs  : list of int — indices of free edge slots (Group 1)
    #
    # If optimize_detunings=False: return only Group 1 (C̃_i only), shape (E_free,).

    def create_initial_guess(
        self,
        conditions: list = [],
        betas=None,
        optimize_detunings: bool = True,
    ):
        pass

    # -----------------------------------------------------------------------
    # setup_bounds(conditions) → np.ndarray or None
    # -----------------------------------------------------------------------
    # MIRRORS: setup_bounds(bounds_intrinsic_loss, free_idxs)
    #          in Architecture_Optimizer
    #
    # Build the bounds array for L-BFGS-B.
    # Each free coupling strength is bounded below by 0 (physical: non-negative).
    # No upper bound (inf).
    # Returns array of shape (num_free_edges, 2) with columns [lower, upper].
    #
    # In AutoScatter, bounds are set for intrinsic_loss variables.
    # Here, bounds enforce non-negativity of all coupling strengths.

    def setup_bounds(self, conditions: list):
        pass

    # -----------------------------------------------------------------------
    # complete_variable_arrays_with_zeros(partial_cs, conditions) → np.ndarray
    # -----------------------------------------------------------------------
    # MIRRORS: complete_variable_arrays_with_zeros in Architecture_Optimizer
    #
    # Pad a partial (free-edges-only) coupling array with zeros for absent edges.
    # Returns a full coupling array of shape (num_all_edges,).
    # Used to recover the full solution from the optimiser output.

    def complete_variable_arrays_with_zeros(self, partial_cs, conditions: list) -> np.ndarray:
        pass

    # -----------------------------------------------------------------------
    # optimize_given_conditions(conditions, ..., betas, lambda_scale) → (success, info_out)
    # -----------------------------------------------------------------------
    # MIRRORS: optimize_given_conditions(conditions, triu_matrix, verbosity, ...)
    #          in Architecture_Optimizer  (EXACT analogue — this is the core method)
    #
    # STAGE 3 of the main algorithm.
    # Run ONE optimisation (single random start) for a FIXED topology and
    # FIXED scaling exponents {β_i}.
    # The independent variables are:
    #   Group 1 — coupling RATIOS C̃_i > 0 (one per active edge)
    #   Group 2 — mode DETUNINGS Δ_i ∈ ℝ (one per mode)
    # These are concatenated into a single vector for scipy: [C̃_0,...,C̃_E, Δ_0,...,Δ_N].
    # Mirrors AutoScatter's (gabs, Deltas) as the two groups of free variables.
    #
    # Parameters:
    #   conditions               : list of constraint objects encoding the topology.
    #                              If None, use translate_triu_to_conditions(triu_array).
    #   triu_array               : alternative to conditions (1D encoding).
    #   betas                    : list of E floats, scaling exponents from Stage 2.
    #                              Default: [1.0] * num_free_edges  (same scaling).
    #                              Passed from find_scaling_exponents result.
    #   lambda_scale             : float, fixed large scale. Default LAMBDA_SCALE_DEFAULT=1000.
    #                              All cooperativities are C_i = lambda_scale^{β_i} · C̃_i.
    #   optimize_detunings       : bool, default True.
    #                              If True, Δ_i are free variables (initialised to 0).
    #                              If False, all Δ_i = 0 fixed (resonant case — faster,
    #                              sufficient for Kronwald/Wang-Clerk in sideband limit).
    #   verbosity                : print progress if True.
    #   init_ratio_range         : [lo, hi] for initial C̃_i draw (default [0.1, 5.0]).
    #   max_violation_success    : success threshold on loss (default 1e-8).
    #   calc_conditions_and_gradients: pre-computed (loss_fn, grad_fn, _) to reuse.
    #   method                   : scipy optimizer method (default 'L-BFGS-B').
    #   **kwargs_solver          : passed to scipy.optimize.minimize options.
    #
    # Steps (Stage 3 — mirrors Architecture_Optimizer.optimize_given_conditions):
    #   1. Build loss and gradient functions:
    #      The free variable vector is x = [C̃_0,...,C̃_{E-1}, Δ_0,...,Δ_{N-1}].
    #      The loss function:
    #        a. Extract ratios = x[:E], detunings = x[E:]
    #        b. Set node['delta'] = detunings[i] for each node i
    #        c. A = build_drift_matrix_from_ratios(nodes, edges, ratios, betas, lambda_scale)
    #           This builds A from the Hamiltonian H with coupling strengths
    #           g_k = sqrt(λ^{β_k} · C̃_k · decay_i · decay_j / 4) per edge k,
    #           and detuning rotation Δ_i · J₂ on each diagonal block.
    #           A is the "coupling matrix" — it encodes all of H in real quadrature form.
    #        d. sigma = solve_lyapunov_kronecker(A, D)   ← A σ + σ Aᵀ + D = 0
    #        e. return ½‖get_mode_covariance(sigma, target_mode_ids) − sigma_target‖²_F
    #      Gradient via jax.grad argnums=0 (w.r.t. full x vector).
    #   2. Sample initial guess:
    #        C̃_i ~ Uniform(init_ratio_range[0], init_ratio_range[1]) for free edges
    #        Δ_i = 0 for all modes  (always start at resonance)
    #   3. Bounds:
    #        C̃_i ∈ [BOUNDS_RATIO_DEFAULT[0], ∞]  (positive)
    #        Δ_i ∈ (−∞, +∞)                       (unbounded)
    #   4. Run scipy.optimize.minimize(fun=loss, jac=grad, bounds=bounds,
    #                                  method='L-BFGS-B', callback=callback, ...).
    #   5. Build solution dict and info dict.
    #
    # Returns:
    #   success : bool — loss < max_violation_success
    #   info_out : dict — mirrors AutoScatter's info_out, extended with Stage 2+3:
    #     {
    #       'initial_guess'       : full x vector at start [C̃_i..., Δ_i...]
    #       'free_idxs'           : list of free edge indices
    #       'solution'            : full x vector at end
    #       'coupling_ratios'     : dict {'C̃_{i,j}': float} — Group 1 output
    #       'detunings'           : dict {'Δ_i': float} — Group 2 output
    #                               ≈ 0 for Kronwald/Wang-Clerk; ≠ 0 for multi-ω_m schemes
    #       'scaling_exponents'   : dict {'β_{i,j}': float} — from Stage 2 input
    #       'lambda_scale'        : float — λ used in Stage 3
    #       'coupling_strengths'  : np.ndarray (E,) — recovered g values:
    #                               g_k = sqrt(λ^{β_k} · C̃_k · decay_i · decay_j / 4)
    #       'parameters_for_analysis': cooperativities dict — C_i = λ^{β_i} · C̃_i
    #       'physical_coupling_formula': str — 'g_k = sqrt(λ^β · C̃ · decay_i · decay_j/4)'
    #       'final_cost'          : float
    #       'success'             : bool
    #       'optimizer_message'   : str from scipy
    #       'A'                   : achieved drift matrix (2N×2N)
    #       'sigma_full'          : achieved full covariance (2N×2N)
    #       'sigma_achieved'      : achieved covariance for signal modes (2M×2M)
    #       'sigma_target'        : self.sigma_target
    #       'nit'                 : number of iterations
    #       'loss_history'        : list of loss values per callback step
    #       'bounds'              : bounds array used
    #     }
    #
    # Relationship to AutoScatter's optimize_given_conditions:
    #   AutoScatter free variables: gabs (|g_{ij}|) + gphases (arg(g_{ij})) + Deltas (Δ_i)
    #   Here:                       C̃_i (ratios, replaces gabs)  +  Δ_i (detunings)
    #   No phases needed (real quadrature basis = real A matrix).
    #   The betas + lambda_scale parametrisation is new; detunings are the same concept.
    #   AutoScatter note: Δ_i = ±1 for far-detuned modes (Schur complement).
    #   Here: Δ_i = 0 for most modes; non-zero only for multi-ω_m schemes.

    def optimize_given_conditions(
        self,
        conditions: list = None,
        triu_array=None,
        betas=None,
        lambda_scale: float = None,
        verbosity: bool = False,
        init_ratio_range=None,
        max_violation_success: float = 1e-8,
        calc_conditions_and_gradients=None,
        method: str = 'L-BFGS-B',
        **kwargs_solver,
    ):
        pass

    # -----------------------------------------------------------------------
    # repeated_optimization(num_tests, conditions, ...) → (success, infos, where)
    # -----------------------------------------------------------------------
    # MIRRORS: repeated_optimization(num_tests, conditions, ...) in Architecture_Optimizer
    #          (EXACT analogue — runs multiple random restarts)
    #
    # Run optimize_given_conditions num_tests times from different random starts.
    # Stop early if interrupt_if_successful=True AND a success is found.
    # Returns the BEST result (lowest loss) across all restarts.
    #
    # Steps:
    #   1. Pre-build calc_conditions_and_gradients ONCE (avoid retracing).
    #      (Same optimisation as AutoScatter: build JIT outside the loop.)
    #   2. For _ in range(num_tests):
    #          success, info = self.optimize_given_conditions(conditions, ...)
    #          Record success, info.
    #          If success and interrupt_if_successful: break.
    #   3. Return np.any(successes), list_of_infos, np.where(successes)
    #
    # The return format mirrors AutoScatter's repeated_optimization exactly:
    #   (bool, list_of_info_dicts, np.where_array)

    def repeated_optimization(
        self,
        num_tests: int,
        conditions: list = None,
        triu_array=None,
        verbosity: bool = False,
        init_strength_range=None,
        max_violation_success: float = 1e-8,
        interrupt_if_successful: bool = True,
        **kwargs_solver,
    ):
        pass

    # -----------------------------------------------------------------------
    # check_all_constraints(A, loss) → triu_array
    # -----------------------------------------------------------------------
    # MIRRORS: check_all_constraints(coupling_matrix, kappa_int_matrix, max_violation)
    #          in Architecture_Optimizer  (EXACT analogue)
    #
    # After finding a solution for the fully-connected graph, discover which
    # architectural constraints are "accidentally" satisfied (nearly zero).
    # This reveals the MINIMAL topology that the solution actually uses.
    #
    # For each constraint c in self.all_possible_constraints:
    #   residual = c(A, sigma)    (from the achieved solution)
    #   If |residual| < threshold:
    #       The constraint is satisfied → this edge is effectively absent or constrained.
    #       Add c to fulfilled_constraints.
    #
    # Convert fulfilled_constraints back to a triu_array.
    # This is the minimal topology that should be re-optimised and stored.
    #
    # AutoScatter uses this to reduce discovered solutions to their minimal
    # topology. Same role here: find the sparsest graph that actually works.

    def check_all_constraints(self, A, sigma, threshold=None) -> np.ndarray:
        pass

    # -----------------------------------------------------------------------
    # prepare_all_possible_combinations()
    # -----------------------------------------------------------------------
    # MIRRORS: prepare_all_possible_combinations() in Architecture_Optimizer
    #          (EXACT analogue — populates self.list_of_triu_arrays,
    #           self.complexity_levels, self.unique_complexity_levels)
    #
    # Enumerate ALL possible topologies for the N-mode system.
    # Uses itertools.product over the possible edge types for each upper-triangle slot.
    #
    # For each upper-triangle entry (i, j):
    #   If Constraint_coupling_absent(i,j) in self.enforced_constraints:
    #       allowed_entries = [NO_COUPLING]   ← cannot have this edge
    #   Elif i == j:
    #       allowed_entries = [NO_COUPLING, PARAMETRIC]
    #   Else:
    #       allowed_entries = [NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING]
    #
    # Compute:
    #   self.possible_entry_lists   = list of allowed_entries per slot
    #   self.list_of_triu_arrays    = list of all possible triu_arrays (all combinations)
    #   self.complexity_levels      = sum(triu_array) for each
    #   self.unique_complexity_levels = sorted unique complexity values
    #   self.num_possible_graphs    = len(list_of_triu_arrays)
    #
    # For N=2 modes (1 cavity + 1 mechanical): 3 off-diagonal choices × 2 diagonal choices
    # × 2 diagonal choices = 3 × 4 = 12 possible topologies. Manageable.
    # For N=3: 3³ × 2² × 2 = 27 × 4 × 2 = 216. Still tractable with BFS pruning.
    # For N=4: 3^6 × 2^4 = 729 × 16 = 11664. Expensive but feasible.
    #
    # This method is the SAME ALGORITHM as AutoScatter's prepare_all_possible_combinations
    # which uses itertools.product over possible coupling matrix entries.

    def prepare_all_possible_combinations(self):
        pass

    # -----------------------------------------------------------------------
    # identify_potential_combinations(complexity_level,
    #                                  skip_check_for_valid_subgraphs=False)
    # -----------------------------------------------------------------------
    # MIRRORS: identify_potential_combinations in Architecture_Optimizer  (EXACT)
    #
    # Filter the full list of topologies at a given complexity level to only
    # those worth testing (the "potential combinations").
    #
    # A topology is a potential combination if:
    #   cond1: it is NOT a subgraph of any INVALID combination discovered so far.
    #          (If a simpler graph failed, this more complex one would also fail.)
    #          check_if_subgraph_triu(self.invalid_combinations, triu_array)
    #   cond2: no VALID combination is a subgraph of it.
    #          (A simpler valid graph already exists — this one is redundant.)
    #          check_if_subgraph_triu(triu_array, self.valid_combinations)
    #
    # These two pruning rules are IDENTICAL to AutoScatter's:
    #   cond1: not check_if_subgraph_upper_triangle(invalid_combos, combo)
    #   cond2: not check_if_subgraph_upper_triangle(combo, valid_combos)
    #
    # This is the key efficiency win: most topologies are pruned before testing.
    # The BFS visits topologies in order of increasing complexity, so valid
    # simple topologies prune large parts of the search space at higher complexity.

    def identify_potential_combinations(
        self,
        complexity_level: int,
        skip_check_for_valid_subgraphs: bool = False,
    ) -> list:
        pass

    # -----------------------------------------------------------------------
    # find_valid_combinations(complexity_level, combinations_to_test=None,
    #                          perform_graph_reduction=True)
    # -----------------------------------------------------------------------
    # MIRRORS: find_valid_combinations(complexity_level, ...) in Architecture_Optimizer
    #          (EXACT analogue, EXTENDED with 3-stage pipeline)
    #
    # Test all candidate topologies at a given complexity level.
    # Runs the THREE-STAGE ALGORITHM (Stage 1→2→3) for each candidate.
    #
    # Parameters:
    #   complexity_level          : int — which layer of the BFS to test
    #   combinations_to_test      : optional override list (instead of identify_potential)
    #   perform_graph_reduction   : if True, run check_all_constraints on each
    #                               successful Stage 3 solution to find its minimal topology.
    #
    # Steps (3-stage extension of AutoScatter's find_valid_combinations):
    #
    #   potential = identify_potential_combinations(complexity_level)
    #   For each triu_array in potential:
    #
    #       ── STAGE 1: Stability at unit cooperativity ────────────────────
    #       stable = check_stability_unit_cooperativity(triu_array)
    #       If not stable:
    #           self.invalid_combinations.append(triu_array)
    #           continue   ← skip to next topology immediately
    #
    #       ── STAGE 2: Find scaling exponents ─────────────────────────────
    #       betas, stage2_ok = find_scaling_exponents(triu_array)
    #       If not stage2_ok:
    #           self.invalid_combinations.append(triu_array)
    #           continue   ← topology cannot realise a finite Gaussian state
    #
    #       ── STAGE 3: Optimise coupling ratios ───────────────────────────
    #       conditions = translate_triu_to_conditions(triu_array)
    #       If check_if_subgraph_triu(triu_array, newly_added_combos): skip
    #       success, all_infos, _ = repeated_optimization(
    #           conditions=conditions, betas=betas, lambda_scale=LAMBDA_SCALE_DEFAULT, ...)
    #
    #       If success:
    #           If perform_graph_reduction:
    #               minimal_triu = check_all_constraints(all_infos[-1]['A'], ...)
    #           Else:
    #               minimal_triu = triu_array
    #           self.valid_combinations.append(minimal_triu)
    #           Store best_info (coupling_ratios, betas, cooperativities, etc.)
    #           newly_added_combos.append(minimal_triu)
    #       Else:
    #           self.invalid_combinations.append(triu_array)
    #
    #   Update counters (num_tested_graphs, num_tested_invalid_graphs, etc.)
    #
    # Key extension over AutoScatter:
    #   AutoScatter's find_valid_combinations goes directly from topology → optimizer.
    #   Here, Stages 1 and 2 are added BEFORE the optimizer call.
    #   Stage 1 and Stage 2 can prune topologies without any gradient computation,
    #   making the BFS substantially faster for large topology searches.
    #
    # Note on Stage 2 results storage:
    #   self.valid_combinations stores (triu_array, betas) pairs so that the
    #   scaling exponents are preserved alongside each valid topology.
    #   This differs from AutoScatter which stores only triu_arrays.

    def find_valid_combinations(
        self,
        complexity_level: int,
        combinations_to_test=None,
        perform_graph_reduction: bool = True,
    ):
        pass

    # -----------------------------------------------------------------------
    # cleanup_valid_combinations()
    # -----------------------------------------------------------------------
    # MIRRORS: cleanup_valid_combinations() in Architecture_Optimizer  (EXACT)
    #
    # Remove REDUNDANT valid combinations from self.valid_combinations.
    # A combination is redundant if another valid combination is a subgraph of it
    # (the simpler graph is sufficient — no need to keep the complex one).
    #
    # Algorithm:
    #   Deduplicate valid_combinations (np.unique on triu arrays).
    #   For each valid_combo in deduplicated list:
    #       Check if any OTHER valid combo is a subgraph of valid_combo.
    #       If yes: valid_combo is redundant → remove it.
    #   Keep only irreducible (minimal) valid combinations.
    #
    # Called after each find_valid_combinations to keep the valid list clean.
    # Mirrors AutoScatter's cleanup_valid_combinations exactly.

    def cleanup_valid_combinations(self):
        pass

    # -----------------------------------------------------------------------
    # perform_breadth_first_search() → np.ndarray of valid triu_arrays
    # -----------------------------------------------------------------------
    # MIRRORS: perform_breadth_first_search() in Architecture_Optimizer  (EXACT)
    #
    # THE MAIN OUTER LOOP — discovers all minimal valid topologies.
    #
    # Algorithm:
    #   1. prepare_all_possible_combinations()
    #      Print: '%i graphs identified' % num_possible_graphs
    #   2. For c in self.unique_complexity_levels  (ascending order):
    #          Print: 'test all graphs with %i degrees of freedom' % c
    #          find_valid_combinations(c)
    #          cleanup_valid_combinations()
    #   3. Return np.array(self.valid_combinations, dtype=int8)
    #
    # This is IDENTICAL to AutoScatter's perform_breadth_first_search.
    # The outer loop iterates over complexity levels (sparse → dense).
    # Pruning rules in identify_potential_combinations make it tractable.
    #
    # For the Kronwald validation case (N=2, 1 cavity + 1 mechanical):
    #   Expected: exactly ONE valid minimal topology found at complexity 3
    #   (1×BEAMSPLITTER + 1×TWO_MODE_SQUEEZING = complexity 1+2=3).
    #   The algorithm must NOT find valid topologies at lower complexities
    #   (complexity 1 or 2) because neither BS alone nor TMS alone works.

    def perform_breadth_first_search(self) -> np.ndarray:
        pass

    # -----------------------------------------------------------------------
    # count_valid_invalid_graphs_layers() → (complexities, valid_counts, invalid_counts)
    # -----------------------------------------------------------------------
    # MIRRORS: count_valid_invalid_graphs_layers() in Architecture_Optimizer
    #
    # Count how many valid/invalid graphs exist at each complexity level.
    # Used for plotting the "landscape" of the search space.
    # Returns arrays of counts per complexity level, for analysis.py to plot.

    def count_valid_invalid_graphs_layers(self):
        pass

    # -----------------------------------------------------------------------
    # extract_cooperativities(conditions, solution_ratios, betas=None,
    #                          lambda_scale=None) → dict
    # -----------------------------------------------------------------------
    # MIRRORS: extract_cooperativities_and_human_defined_parameters(conditions, solution_dict)
    #          in Architecture_Optimizer  (EXACT analogue, EXTENDED for Stage 2+3)
    #
    # Compute and report all physically relevant quantities for the solution.
    # Accepts coupling RATIOS C̃_i (Stage 3 output) rather than raw g values.
    #
    # Parameters:
    #   conditions     : list of constraint objects (defines which edges are active)
    #   solution_ratios: np.ndarray (E_free,), the Stage 3 solution C̃_i per free edge
    #   betas          : list of floats, scaling exponents β_i from Stage 2 (or None if
    #                    not run, in which case defaults to β_i = 1 for all edges)
    #   lambda_scale   : float, the λ used in Stage 3 (default LAMBDA_SCALE_DEFAULT)
    #
    # Returns:
    #   dict with the following keys per active edge (i, j):
    #     'C̃_{i,j}'    : float — dimensionless coupling ratio (Stage 3 output)
    #     'β_{i,j}'    : float — scaling exponent (Stage 2 output)
    #     'C_{i,j}'    : float — actual cooperativity C = λ^β · C̃
    #     'g_{i,j}'    : float — physical coupling strength g = sqrt(C · κ_i · γ_j / 4)
    #     'formula'    : str — 'g = sqrt(λ^β · C̃ · decay_i · decay_j / 4)'
    #
    # For the Kronwald scheme (β_g = β_ν = 1):
    #   C̃_g, C̃_ν are the Stage 3 output (dimensionless ratios)
    #   C_g = λ · C̃_g,  C_ν = λ · C̃_ν  (with λ=1000, these are the actual cooperativities)
    #   squeezing r = atanh(sqrt(C_ν/C_g)) = atanh(sqrt(C̃_ν/C̃_g))  ← ratio is λ-independent
    #   g = sqrt(λ · C̃_g · κ · γ / 4)  ← the required coupling strength for given κ, γ
    #
    # AutoScatter uses:  C_{i,j} = 4 * |g_{i,j}|²  (κ_ext normalised to 1).
    # Here:              C_{i,j} = λ^{β_i} · C̃_i = 4 g² / (decay_i · decay_j).
    #
    # INTERPRETATION OF ALL-EQUAL-β CASE (most common):
    #   If all β_i = 1: single scale knob — to realise the scheme experimentally,
    #   take λ >> 1 (strong cooperativity).  The ratios C̃_i determine the squeezing.
    # INTERPRETATION OF UNEQUAL-β CASE:
    #   Edge with β=2 must have cooperativity growing as λ² (parametrically stronger
    #   than β=1 edges).  The experiment has two independently adjustable drives.

    def extract_cooperativities(
        self,
        conditions: list,
        solution_ratios,
        betas=None,
        lambda_scale=None,
    ) -> dict:
        pass

    # -----------------------------------------------------------------------
    # dict_extract_relevant_information(solution_cs, conditions) → dict
    # -----------------------------------------------------------------------
    # MIRRORS: dict_extract_relevant_information in Architecture_Optimizer
    #
    # Build a human-readable dict of the solution:
    #   {edge_label: coupling_strength}
    # for all FREE (non-absent) edges.
    # edge_label format: 'g_{i,j}' for BS, 'nu_{i,j}' for TMS.
    #
    # Called inside optimize_given_conditions to build solution_dict.

    def dict_extract_relevant_information(self, solution_cs, conditions: list) -> dict:
        pass
