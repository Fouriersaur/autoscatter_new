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
TWO-STAGE ALGORITHM  (prototype — Stage 2 scaling discovery commented out)
═══════════════════════════════════════════════════════════════════════════════

PROTOTYPE SIMPLIFICATION:
    Stage 2 (convergence test / scaling discovery) is currently disabled.
    All cooperativity ratios C̃_k are assumed equal at initialisation (u_k=0).
    Stage 3 optimises from this flat starting point directly.
    Stage 2 should be re-enabled once the prototype recovers the Kronwald scheme.

For every candidate topology produced by the BFS outer loop, the algorithm
runs two internal stages before accepting or rejecting that topology.

STAGE 1 — Stability check at unit cooperativity  (fast discrete filter)
    Build A from H with all cooperativities C_{ij} = 1.
    "Unit cooperativity" means each Hamiltonian coupling g_{ij} is set to
    the geometric-mean threshold: g_{ij} = sqrt(decay_i × decay_j / 4).
    A is assembled via the H → A map (build_drift_matrix): one block per
    Hamiltonian term.  Then check whether all eigenvalues of A have strictly
    negative real part.
    If unstable → SKIP this topology (do not proceed to Stage 2 or 3).
    No gradient computation, no Lyapunov solve — just an eigenvalue check.
    Method: check_stability_unit_cooperativity(triu_array) → bool

    CRITICAL — Stage 1 failures do NOT go into self.invalid_combinations.
    Reason: a topology that is unstable at unit cooperativity may become
    stable at a different coupling ratio. Adding MORE edges (a supergraph)
    can add stabilising BS couplings that fix the instability.
    Example: TMS-only is marginally unstable at C=1, but BS+TMS (Kronwald)
    is stable. If TMS-only were added to invalid_combinations, its superset
    BS+TMS would be pruned and never found.
    Stage 1 is a SPEED FILTER only — it avoids expensive Stage 2/3 computation
    for topologies that cannot even be stable at unit cooperativity, but it
    does not make any claim about whether their supersets are valid.

# ── STAGE 2 DISABLED (prototype) ─────────────────────────────────────────────
# STAGE 2 — Convergence test  (short optimisation at multiple scales)
#     For λ in [10, 100, 1000]:
#         Run a SHORT optimisation (~30 steps of L-BFGS-B) over log_ratios u_k:
#             u_k* = argmin_{u_k} loss(λ, u_k)
#             where C̃_k = exp(u_k),  g_k = sqrt(λ · C̃_k · decay_i · decay_j / 4)
#         Record L*(λ) = loss at the short optimum.
#     Accept topology if:
#         L*(10) > L*(100) > L*(1000)   (loss strictly decreasing — converging)
#         AND L*(1000) < CONVERGENCE_LOSS_THRESHOLD
#     If accepted: warm-start Stage 3 from u_k* found at λ=1000.
#     If rejected: add to self.invalid_combinations.
#         Stage 2 failures CAN prune supersets — structural incompatibility
#         with σ_target is not fixed by adding more edges.
#     Method: check_convergence(triu_array) → (u_warm, success)
#
#     Key design choice: tests convergence at the OPTIMAL ratio for each λ,
#     not an arbitrary placeholder (e.g. C̃_k=1 would give wrong ratios and
#     incorrectly reject valid topologies like Kronwald).
# ─────────────────────────────────────────────────────────────────────────────

STAGE 2 (prototype) — Optimise log coupling ratios {u_k} AND detunings {Δ_i}
    Fix λ = LAMBDA_SCALE_DEFAULT = 1000.
    Free variables — TWO GROUPS (mirrors AutoScatter's gabs + Deltas):
      Group 1 — LOG coupling ratios:  u_k = log(C̃_k) ∈ (−∞, +∞), one per active edge.
          C̃_k = exp(u_k) > 0 automatically (no lower bound needed)
          Cooperativity:   C_k = λ · exp(u_k)
          Coupling:        g_k = sqrt(λ · exp(u_k) · decay_i · decay_j / 4)
          Initial guess:   u_k = 0 for all k  (C̃_k = 1, all ratios equal — prototype)
                           [future: warm-start from Stage 2 result u_k* at λ=1000]
          Bounds:          none (log space handles positivity)
      Group 2 — mode detunings:  Δ_i ∈ (−∞, +∞), one per mode i.
          Enter A as:  A[s_i, s_i] += Δ_i · J₂   where J₂ = [[0,1],[-1,0]]
          Initial guess: Δ_i = 0 (resonant driving start)
          Bounds:        none (unbounded)
    Loss: ½‖σ_system − σ_target‖²_F  (Frobenius on signal modes only)
    Gradient flows: loss → σ_sub → Lyapunov solve → A → g_k → exp(u_k)
    Optimiser: L-BFGS-B + JAX autodiff (same as AutoScatter).
    Multiple random restarts for Group 1; Group 2 always starts at 0.
    Method: optimize_given_conditions(conditions, lambda_scale) → (success, info_out)

    Why log parametrisation?
      exp(u_k) > 0 automatically — no bounds needed.
      Gradient steps are multiplicative: Δu_k=0.1 → 10% change at any scale.
      Runaway prevented naturally: instability increases loss before exp(u_k) → ∞.

    Why include detunings?
      AutoScatter optimises over Δ_i explicitly (they are in all_variables_list).
      Here, Δ_i are needed whenever:
        (a) The target covariance has a squeezed quadrature at a non-zero angle
            (Δ_i rotates the squeezing direction in phase space).
        (b) Multiple mechanical modes have different frequencies (only one can be in
            its own rotating frame; the others need residual detuning Δ ≠ 0).
        (c) Off-resonance driving improves stability or squeezing magnitude.
      For Kronwald / Wang-Clerk: Stage 2 optimizer converges to Δ_i ≈ 0
      (confirming the resonant condition). This is a non-trivial output.

    Stage 1 uses Δ_i = 0 for all modes (resonant stability check).

OUTPUT per successful topology:
    topology:        graph (nodes, edges, edge types)
    log_ratios:      {u_k} per active edge      (Stage 2 Group 1 raw output)
    coupling_ratios: {C̃_k = exp(u_k)} per edge (Stage 2 Group 1 interpreted)
    detunings:       {Δ_i} per mode             (Stage 2 Group 2 output)
    lambda_scale:    λ = 1000 (fixed)
    physical_formula: g_k = sqrt(λ · exp(u_k) · decay_i · decay_j / 4)
    cooperativities: C_k = λ · exp(u_k)          (the hardware requirement)

INTERPRETATION:
    All cooperativities are large (C_k = λ · C̃_k with λ=1000).
    The RATIOS {C̃_k} determine which state is produced.
    The absolute scale λ=1000 ensures the strong-coupling limit is reached.
    If all Δ_i ≈ 0:  resonant driving suffices (Kronwald, Wang-Clerk cases).
    If some Δ_i ≠ 0: off-resonance driving is required — a non-trivial prediction.
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
LAMBDA_SCALE_DEFAULT         = 1000.          # fixed large scale λ for optimisation
INIT_LOG_RATIO_RANGE_DEFAULT = [-1.0, 1.0]   # initial u_k = log(C̃_k) draw range
#                                              (corresponds to C̃_k ∈ [e^{-1}, e^1] ≈ [0.37, 2.72])

# ── Stage 2 scaling-discovery constants (disabled in prototype) ────────────
# Re-enable when implementing check_convergence for automated scale finding.
# LAMBDA_VALUES_STAGE2       = [10., 100., 1000.]  # scales for convergence test
# STAGE2_SHORT_OPT_ITER      = 30                  # L-BFGS-B steps per scale
# CONVERGENCE_LOSS_THRESHOLD = 1e-4                # accept if L*(λ=1000) < this


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
    for num_aux in range(start_value, max_value + 1):
        print(f'testing {num_aux} auxiliary modes')
        node_types = list(node_types_signal) + ['cavity'] * num_aux
        try:
            optimizer = CovarianceOptimizer(
                sigma_target=sigma_target,
                target_mode_ids=target_mode_ids,
                node_types=node_types,
                num_auxiliary_modes=num_aux,
                make_initial_test=True,
                **kwargs_optimizer,
            )
            print(f'minimum auxiliary modes: {num_aux}')
            return optimizer
        except Exception:
            continue
    return None


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
        make_initial_test: bool = True,):

            
        from reservoir_engineering.covariance_physics import build_diffusion_matrix

        self.sigma_target         = np.array(sigma_target) # 2M x 2M target covariance matrix
        self.target_mode_ids      = list(target_mode_ids) # which modes are the squeezed modes ("Mechanical modes")
        self.node_types           = list(node_types) 
        self.num_modes            = len(node_types)
        self.num_auxiliary_modes  = num_auxiliary_modes
        self.gradient_method      = gradient_method
        self.enforced_constraints = list(enforced_constraints)

        # Optimization hyperparameters

        self.kwargs_optimization = dict(num_tests=10, verbosity=0,
                                    max_violation_success=1e-8,
                                    interrupt_if_successful=True)
        self.kwargs_optimization.update(kwargs_optimization)

        self.solver_options = dict(maxiter=2000, ftol=0, gtol=1e-12)
        self.solver_options.update(solver_options)

        # Initilise the node dictionary
        self.nodes = []
        for i, t in enumerate(node_types):
            if t == 'cavity':
                self.nodes.append({'id': i, 'type': 'cavity',
                                'kappa': 1.0, 'delta': 0.0})
            else:
                self.nodes.append({'id': i, 'type': 'mechanical',
                                'gamma': 0.01, 'n_th': 0.0, 'delta': 0.0})

        self.D = build_diffusion_matrix(self.nodes)

        # All the upper-triangle edge slots 
        self.all_possible_edges = [(i, j)
                               for i in range(self.num_modes)
                               for j in range(i, self.num_modes)]
        
        # BFS state
        self.valid_combinations         = []
        self.invalid_combinations       = []
        self.best_info_list             = []
        self.tested_complexities        = []
        self.num_tested_graphs          = []
        self.num_tested_invalid_graphs  = []

        # initilise all constraints on all the modes

        self.__setup_all_constraints__()

        # See if the fully-connected graph can achieve sigma_target
        # if failed -> the subgraphs of the fully connected graph cannot either

        if make_initial_test:
            from reservoir_engineering.topology_search import TopologyGraph
            from reservoir_engineering.topology_search import translate_triu_to_conditions

            full_triu = TopologyGraph.fully_connected(node_types).triu_array

            # Optimization
            conditions_full = translate_triu_to_conditions(full_triu, node_types)
            success, _, _ = self.repeated_optimization(
                num_tests=self.kwargs_optimization['num_tests'],
                conditions=conditions_full,
                lambda_scale=LAMBDA_SCALE_DEFAULT,
            )
            if not success:
                raise Exception(
                    "Fully-connected graph failed to achieve sigma_target. "
                    "Check that the target is physically reachable with these node types.")
            

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
        
        from reservoir_engineering.constraints import (
            Constraint_coupling_absent, Constraint_coupling_beamsplitter)
        
        self.all_possible_constraints = []
        
        for i in range(self.num_modes):
            for j in range(i, self.num_modes):
                # There is no self-coupling within a mode i,i
                self.all_possible_constraints.append(Constraint_coupling_absent(i, j))
                if i != j:
                    # Initialising all the off-digonal interactions i,j to be BS 
                    self.all_possible_constraints.append(Constraint_coupling_beamsplitter(i, j))

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
    #   Do NOT append to self.invalid_combinations.
    #   Do NOT call find_scaling_exponents or optimize_given_conditions.
    #   Just continue to the next topology (skip silently).
    #
    #   WHY NOT add to invalid_combinations?
    #   Adding more edges (a supergraph) can STABILISE a previously unstable
    #   topology. Example: TMS-only has a zero eigenvalue at unit cooperativity
    #   (marginally unstable), but adding a BS edge (→ BS+TMS = Kronwald) makes
    #   it stable. If TMS-only were added to invalid_combinations, the BFS prune
    #   rule "skip supersets of invalid topologies" would skip BS+TMS entirely,
    #   and the algorithm would NEVER find the Kronwald topology.
    #   Stage 1 is a SPEED FILTER only, not a structural impossibility proof.
    #
    # No AutoScatter analogue (AutoScatter's passive cavities are always stable).


    # == To Check if Matrix A satisfies the stability condition under unit coopertivity == 
    def check_stability_unit_cooperativity(self, triu_array) -> bool:
        from reservoir_engineering.topology_search import TopologyGraph
        from reservoir_engineering.covariance_physics import build_drift_matrix, check_stability

        default_kappa = next((n['kappa'] for n in self.nodes if n['type'] == 'cavity'), 1.0)
        default_gamma = next((n['gamma'] for n in self.nodes if n['type'] == 'mechanical'), 0.01)
        
        # Creates Nodes List: {'id': 0, 'type': 'cavity', 'kappa': 1.0}
        # Creates Edges List: {'i': 0, 'j': 1, 'type': 'beamsplitter'}
        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts(
            default_kappa=default_kappa, default_gamma=default_gamma,)

        unit_gs = [] # Coupling strength given unit coopertivity

        for edge in edges:
            # Obtain the two nodes for each edge
            ni, nj = nodes[edge["i"]], nodes[edge["j"]]

            # Decay rates of the two modes with the edge connected
            di = ni.get('kappa', ni.get('gamma'))
            dj = nj.get('kappa', nj.get('gamma'))

            unit_gs.append(float(np.sqrt(di * dj / 4.)))
        
        A = build_drift_matrix(nodes, edges, jnp.array(unit_gs))
        
        return check_stability(A)

    # -----------------------------------------------------------------------
    # check_convergence(triu_array, lambda_values=None, short_iter=None,
    #                   loss_threshold=None) → (u_warm, success)
    # -----------------------------------------------------------------------
    # STAGE 2 of the main algorithm.  Test whether this topology can approach
    # σ_target as the coupling scale λ grows, using short optimisations at
    # multiple scales rather than a fixed placeholder ratio.
    #
    # Parameters:
    #   triu_array      — 1D int array encoding the topology
    #   lambda_values   — list of 3 increasing floats, default LAMBDA_VALUES_STAGE2 = [10, 100, 1000]
    #   short_iter      — int, L-BFGS-B iterations per scale, default STAGE2_SHORT_OPT_ITER = 30
    #   loss_threshold  — float, accept if L*(λ_max) < threshold,
    #                     default CONVERGENCE_LOSS_THRESHOLD = 1e-4
    #
    # Returns:
    #   (u_warm, success):
    #     u_warm   — jnp.ndarray (E,), log_ratios at λ_max after short optimisation
    #                (warm-start for Stage 3, or None on failure)
    #     success  — bool: True if loss is decreasing AND below threshold
    #
    # Algorithm:
    #   losses = []
    #   u_current = jnp.zeros(E)   ← initial log_ratios = 0 → C̃_k = 1
    #   for λ in lambda_values:
    #       u_opt = short_optimise(covariance_loss_from_ratios,
    #                              init=u_current, lambda_scale=λ,
    #                              max_iter=short_iter)
    #       losses.append(covariance_loss_from_ratios(u_opt, ..., λ, ...))
    #       u_current = u_opt          ← warm-start each scale from the previous
    #
    #   converging = losses[0] > losses[1] > losses[2]  AND losses[2] < loss_threshold
    #   if converging: return (u_current, True)
    #   else:          return (None, False)
    #
    # Why this works:
    #   Tests convergence at the OPTIMAL ratio for each λ, not an arbitrary 1:1 ratio.
    #   Correctly handles topologies where the convergent direction requires specific
    #   coupling ratios — e.g. Kronwald needs ν < g, not ν = g.
    #   With C̃_k=1 (old approach), Kronwald would have σ_pm diverging and be rejected.
    #   With this approach, the short optimiser finds ν/g ≈ tanh(r) at each λ and
    #   confirms the loss is decreasing.
    #
    # Cost:
    #   3 × short_iter gradient steps = 90 gradient evaluations total.
    #   Each gradient evaluation ≈ 1 Lyapunov solve + 1 backprop.
    #   Comparable to the old β-grid search (which did 3^E × 3 forward solves)
    #   but correct for all topologies including Kronwald.
    #
    # Called by: find_valid_combinations, AFTER Stage 1 (stability check) passes.
    # If success=False: add triu_array to self.invalid_combinations, skip Stage 3.
    # If success=True:  pass u_warm to optimize_given_conditions as warm start.
    #
    # No AutoScatter analogue — unique to Reservoir Engineering.

    # ── PROTOTYPE: check_convergence is disabled ──────────────────────────────
    # Stage 2 (automated scaling discovery) is not used in the prototype.
    # find_valid_combinations skips directly from Stage 1 to optimize_given_conditions.
    # Re-enable by uncommenting the body below and restoring the Stage 2 constants.
    #
    # def check_convergence(
    #     self,
    #     triu_array,
    #     lambda_values=None,        # default: LAMBDA_VALUES_STAGE2 = [10, 100, 1000]
    #     short_iter: int = None,    # default: STAGE2_SHORT_OPT_ITER = 30
    #     loss_threshold: float = None,  # default: CONVERGENCE_LOSS_THRESHOLD = 1e-4
    # ) -> tuple:
    #     lambda_values  = lambda_values  or LAMBDA_VALUES_STAGE2
    #     short_iter     = short_iter     or STAGE2_SHORT_OPT_ITER
    #     loss_threshold = loss_threshold or CONVERGENCE_LOSS_THRESHOLD
    #
    #     nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts()
    #     E = len(edges)
    #     losses = []
    #     u_current = jnp.zeros(E)
    #
    #     for lam in lambda_values:
    #         def loss_fn(u):
    #             return covariance_loss_from_ratios(
    #                 u, nodes, edges, self.D, lam,
    #                 jnp.array(self.sigma_target), self.target_mode_ids)
    #         grad_fn = jax.jit(jax.grad(loss_fn))
    #         result = sciopt.minimize(
    #             fun=jax.jit(loss_fn), jac=grad_fn,
    #             x0=np.array(u_current), method='L-BFGS-B',
    #             options={'maxiter': short_iter})
    #         u_current = jnp.array(result.x)
    #         losses.append(float(loss_fn(u_current)))
    #
    #     converging = (losses[0] > losses[1] > losses[2]) and (losses[2] < loss_threshold)
    #     if converging:
    #         return (u_current, True)
    #     else:
    #         return (None, False)

    def check_convergence(
        self,
        triu_array,
        lambda_values=None,
        short_iter: int = None,
        loss_threshold: float = None,
    ) -> tuple:
        # PROTOTYPE: Stage 2 disabled. Always returns (None, True) to skip straight to Stage 3.
        # No filtering on convergence — every Stage-1-stable topology proceeds to optimisation.
        return (None, True)

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

    # == see which edges are free to optimize? == 
    def give_free_variable_idxs(self, conditions: list) -> list:
        from reservoir_engineering.constraints import Constraint_coupling_absent
        
        # Check which edges are absent 
        absent = {tuple(c.idxs) for c in conditions
                  if isinstance(c, Constraint_coupling_absent)}
        
        # return a list of indicies k which output edges which are present
        return [k for k, (i, j) in enumerate(self.all_possible_edges) if (min(i,j), max(i,j)) not in absent]

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

    # == Returns the loss function and its gradient for a specific toplogy found == 

    def give_conditions_func_with_conditions(self, conditions: list):
        from reservoir_engineering.topology_search import translate_conditions_to_triu, TopologyGraph
        from reservoir_engineering.covariance_physics import (
            build_drift_matrix_from_ratios, solve_lyapunov_kronecker, get_mode_covariance)
        
        # == When topology is found -> what is the triu array, nodes and edges of the topology == 
        triu_array = translate_conditions_to_triu(conditions, self.num_modes, self.node_types)
        default_kappa = next((n['kappa'] for n in self.nodes if n['type'] == 'cavity'), 1.0)
        default_gamma = next((n['gamma'] for n in self.nodes if n['type'] == 'mechanical'), 0.01)
        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts(
            default_kappa=default_kappa, default_gamma=default_gamma)
        
        E = len(edges)
        N = self.num_modes
        D = self.D
        sigma_target_jnp = jnp.array(self.sigma_target)
        target_mode_ids = self.target_mode_ids
        enforced_constraints = self.enforced_constraints
        J2 = jnp.array([[0., 1.], [-1., 0.]])
        lambda_scale = LAMBDA_SCALE_DEFAULT

        def loss_fn(x):
            """
            x is a flat vector of all free parameters:
            x = [u_0, u_1, ..., u_{E-1}, Δ_0, Δ_1, ..., Δ_{N-1}]
                ←── log coupling ratios ──→  ←──── detunings ────→
            """
            log_ratios = x[:E]
            detunings_x = x[E:]
            ratios = jnp.exp(log_ratios)

            A = build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
            
            for i in range(N):
                s = slice(2 * i, 2 * i + 2)
                A = A.at[s, s].add(detunings_x[i] * J2)
            
            # construct the cov matrix -> loss function
            sigma = solve_lyapunov_kronecker(A, D)
            sigma_sub = get_mode_covariance(sigma, target_mode_ids)
            loss = jnp.sum((sigma_sub - sigma_target_jnp) ** 2) / 2.
            
            # Add the contraints f into the loss function
            for c in enforced_constraints:
                loss = loss + c(A, sigma)
            
            return loss

        loss_jit = jax.jit(loss_fn)
        grad_jit = jax.jit(jax.grad(loss_fn))
        return loss_jit, grad_jit, None

    # -----------------------------------------------------------------------
    # create_initial_guess(conditions=[], u_warm=None, optimize_detunings=True)
    #     → (initial_x, free_idxs)
    # -----------------------------------------------------------------------
    # MIRRORS: create_initial_guess(conditions, init_abs_range, ...)
    #          in Architecture_Optimizer
    #
    # Sample the initial parameter vector x = [u_k..., Δ_i...] for Stage 3.
    # Two groups of variables (mirrors AutoScatter's gabs + Deltas initial guess):
    #
    # Group 1 — LOG coupling ratios (E_free entries):
    #   If u_warm provided (from Stage 2): use u_warm as initial guess.
    #   Otherwise: draw u_k uniformly in INIT_LOG_RATIO_RANGE_DEFAULT = [-1.0, 1.0].
    #   (Corresponds to C̃_k = exp(u_k) ∈ [e^{-1}, e^1] ≈ [0.37, 2.72].)
    #   Note: AutoScatter draws gphases ∈ [-π, π]; here no phases needed
    #         (real quadrature basis = real A matrix, no phase degree of freedom).
    #
    # Group 2 — mode detunings (N entries, one per node):
    #   Δ_i = 0 for ALL modes (always start at resonance).
    #   The optimizer moves Δ_i away from 0 if the target requires off-resonance driving.
    #   For Kronwald / Wang-Clerk: expect Δ_i ≈ 0 at convergence.
    #
    # Returns:
    #   initial_x  : np.ndarray (E_free + N,) — concatenated [u_k..., Δ_i...]
    #   free_idxs  : list of int — indices of free edge slots (Group 1)
    #
    # If optimize_detunings=False: return only Group 1 (u_k only), shape (E_free,).

    # === Create initial guesses for the 1. log ratios and 2. mode detunings ===

    def create_initial_guess(
        self,
        conditions: list = [],
        betas=None,
        optimize_detunings: bool = True,
    ):
        
        # Give the indices of the free variables that need optimising
        free_idxs = self.give_free_variable_idxs(conditions)
        E = len(free_idxs)

        lo, hi = INIT_LOG_RATIO_RANGE_DEFAULT
        u_init = np.random.uniform(lo, hi, E).astype(float)
        
        if optimize_detunings:
            x0 = np.concatenate([u_init, np.zeros(self.num_modes)])
        else:
            x0 = u_init
        
        return x0, free_idxs

    # -----------------------------------------------------------------------
    # setup_bounds(conditions) → np.ndarray or None
    # -----------------------------------------------------------------------
    # MIRRORS: setup_bounds(bounds_intrinsic_loss, free_idxs)
    #          in Architecture_Optimizer
    #
    # Build the bounds array for L-BFGS-B.
    # With the log parametrisation, ALL variables are unconstrained:
    #   u_k = log(C̃_k) ∈ (−∞, +∞)  — exp(u_k) > 0 automatically
    #   Δ_i             ∈ (−∞, +∞)  — detuning is unbounded
    # Returns None (no bounds needed) or an array of (None, None) pairs.
    #
    # Contrast with AutoScatter: gabs ≥ 0 required explicit lower bounds.
    # Here the log reparametrisation eliminates that requirement entirely.

    def setup_bounds(self, conditions: list):
        return None  # log-space: u_k and Δ_i are both unconstrained reals

    # -----------------------------------------------------------------------
    # complete_variable_arrays_with_zeros(partial_cs, conditions) → np.ndarray
    # -----------------------------------------------------------------------
    # MIRRORS: complete_variable_arrays_with_zeros in Architecture_Optimizer
    #
    # Pad a partial (free-edges-only) coupling array with zeros for absent edges.
    # Returns a full coupling array of shape (num_all_edges,).
    # Used to recover the full solution from the optimiser output.

    def complete_variable_arrays_with_zeros(self, partial_cs, conditions: list) -> np.ndarray:
        free_idxs = self.give_free_variable_idxs(conditions)
        full = np.zeros(len(self.all_possible_edges))
        for k, idx in enumerate(free_idxs):
            full[idx] = partial_cs[k]
        return full

    # -----------------------------------------------------------------------
    # optimize_given_conditions(conditions, ..., lambda_scale, u_warm) → (success, info_out)
    # -----------------------------------------------------------------------
    # MIRRORS: optimize_given_conditions(conditions, triu_matrix, verbosity, ...)
    #          in Architecture_Optimizer  (EXACT analogue — this is the core method)
    #
    # STAGE 3 of the main algorithm.
    # Run ONE optimisation (single random start) for a FIXED topology.
    # The independent variables are:
    #   Group 1 — LOG coupling ratios u_k = log(C̃_k) ∈ (−∞, +∞), one per active edge
    #   Group 2 — mode DETUNINGS Δ_i ∈ ℝ, one per mode
    # Concatenated into a single vector: x = [u_0,...,u_{E-1}, Δ_0,...,Δ_{N-1}].
    # Mirrors AutoScatter's (gabs, Deltas) as the two groups of free variables.
    #
    # Parameters:
    #   conditions               : list of constraint objects encoding the topology.
    #                              If None, use translate_triu_to_conditions(triu_array).
    #   triu_array               : alternative to conditions (1D encoding).
    #   lambda_scale             : float, fixed large scale. Default LAMBDA_SCALE_DEFAULT=1000.
    #                              All cooperativities are C_k = lambda_scale · exp(u_k).
    #   u_warm                   : jnp.ndarray (E,), warm-start log_ratios from Stage 2.
    #                              If None, draw u_k ~ Uniform(INIT_LOG_RATIO_RANGE_DEFAULT).
    #   optimize_detunings       : bool, default True.
    #                              If True, Δ_i are free variables (initialised to 0).
    #                              If False, all Δ_i = 0 fixed (resonant case — faster).
    #   verbosity                : print progress if True.
    #   max_violation_success    : success threshold on loss (default 1e-8).
    #   calc_conditions_and_gradients: pre-computed (loss_fn, grad_fn, _) to reuse.
    #   method                   : scipy optimizer method (default 'L-BFGS-B').
    #   **kwargs_solver          : passed to scipy.optimize.minimize options.
    #
    # Steps (Stage 3):
    #   1. Build loss and gradient functions.
    #      Free variable vector x = [u_0,...,u_{E-1}, Δ_0,...,Δ_{N-1}].
    #      The loss function:
    #        a. Extract log_ratios = x[:E], detunings = x[E:]
    #        b. Set node['delta'] = detunings[i] for each node i
    #        c. ratios = jnp.exp(log_ratios)   ← C̃_k = exp(u_k) > 0 always
    #        d. A = build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
    #           Builds A from H with g_k = sqrt(λ · C̃_k · decay_i · decay_j / 4)
    #           and detuning rotation Δ_i · J₂ on each diagonal block.
    #        e. sigma = solve_lyapunov_kronecker(A, D)
    #        f. return ½‖get_mode_covariance(sigma, target_mode_ids) − sigma_target‖²_F
    #      Gradient via jax.grad argnums=0 (w.r.t. full x vector).
    #   2. Initial guess:
    #        u_k = u_warm[k] if provided, else u_k ~ Uniform(INIT_LOG_RATIO_RANGE_DEFAULT)
    #        Δ_i = 0 for all modes (always start at resonance)
    #   3. Bounds: NONE — x is unconstrained.
    #        u_k ∈ (−∞, +∞): exp(u_k) > 0 automatically
    #        Δ_i ∈ (−∞, +∞): unbounded
    #   4. Run scipy.optimize.minimize(fun=loss, jac=grad, method='L-BFGS-B', ...).
    #   5. Build solution dict and info dict.
    #
    # Returns:
    #   success : bool — loss < max_violation_success
    #   info_out : dict — mirrors AutoScatter's info_out:
    #     {
    #       'initial_guess'       : full x vector at start [u_k..., Δ_i...]
    #       'free_idxs'           : list of free edge indices
    #       'solution'            : full x vector at end
    #       'log_ratios'          : np.ndarray (E,) — u_k values (Group 1 raw)
    #       'coupling_ratios'     : dict {'C̃_{i,j}': exp(u_k)} — Group 1 interpreted
    #       'detunings'           : dict {'Δ_i': float} — Group 2 output
    #                               ≈ 0 for Kronwald/Wang-Clerk; ≠ 0 for multi-ω_m schemes
    #       'lambda_scale'        : float — λ used in Stage 3
    #       'coupling_strengths'  : np.ndarray (E,) — physical g values:
    #                               g_k = sqrt(λ · exp(u_k) · decay_i · decay_j / 4)
    #       'cooperativities'     : dict — C_k = λ · exp(u_k) per edge
    #       'physical_formula'    : 'g_k = sqrt(λ · C̃_k · decay_i · decay_j / 4)'
    #       'final_cost'          : float
    #       'success'             : bool
    #       'optimizer_message'   : str from scipy
    #       'A'                   : achieved drift matrix (2N×2N)
    #       'sigma_full'          : achieved full covariance (2N×2N)
    #       'sigma_achieved'      : achieved covariance for signal modes (2M×2M)
    #       'sigma_target'        : self.sigma_target
    #       'nit'                 : number of iterations
    #       'loss_history'        : list of loss values per callback step
    #     }
    #
    # Relationship to AutoScatter's optimize_given_conditions:
    #   AutoScatter free variables: gabs (|g_{ij}|) + gphases (arg(g_{ij})) + Deltas (Δ_i)
    #   Here: u_k = log(C̃_k) (replaces gabs, no phases) + Δ_i (detunings, same concept)
    #   No phases needed — real quadrature basis means A is real (no phase freedom).

    def optimize_given_conditions(
        self,
        conditions: list = None,
        triu_array=None,
        lambda_scale: float = None,
        u_warm=None,
        optimize_detunings: bool = True,
        verbosity: bool = False,
        max_violation_success: float = 1e-8,
        calc_conditions_and_gradients=None,
        method: str = 'L-BFGS-B',
        **kwargs_solver,
    ):
        from reservoir_engineering.topology_search import (
            TopologyGraph, translate_triu_to_conditions, translate_conditions_to_triu)
        from reservoir_engineering.covariance_physics import (
            build_drift_matrix_from_ratios, solve_lyapunov_kronecker, get_mode_covariance)

        # Resolve: need both conditions and triu_array
        if triu_array is not None and conditions is None:
            conditions = translate_triu_to_conditions(triu_array, self.node_types)
        elif conditions is not None and triu_array is None:
            triu_array = translate_conditions_to_triu(conditions, self.num_modes, self.node_types)
        elif conditions is None and triu_array is None:
            raise ValueError("Must provide conditions or triu_array")

        if lambda_scale is None:
            lambda_scale = LAMBDA_SCALE_DEFAULT

        default_kappa = next((n['kappa'] for n in self.nodes if n['type'] == 'cavity'), 1.0)
        default_gamma = next((n['gamma'] for n in self.nodes if n['type'] == 'mechanical'), 0.01)
        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts(
            default_kappa=default_kappa, default_gamma=default_gamma)
        E = len(edges)
        N = self.num_modes

        sigma_target_jnp = jnp.array(self.sigma_target)
        D = self.D
        target_mode_ids = self.target_mode_ids
        enforced_constraints = self.enforced_constraints
        J2 = jnp.array([[0., 1.], [-1., 0.]])

        # Build loss: x = [u_0,...,u_{E-1}, Δ_0,...,Δ_{N-1}]
        def loss_fn(x):
            log_ratios = x[:E]
            detunings_x = x[E:]
            ratios = jnp.exp(log_ratios)
            A = build_drift_matrix_from_ratios(nodes, edges, ratios, lambda_scale)
            for i in range(N):
                s = slice(2 * i, 2 * i + 2)
                A = A.at[s, s].add(detunings_x[i] * J2)
            sigma = solve_lyapunov_kronecker(A, D)
            sigma_sub = get_mode_covariance(sigma, target_mode_ids)
            loss = jnp.sum((sigma_sub - sigma_target_jnp) ** 2) / 2.
            for c in enforced_constraints:
                loss = loss + c(A, sigma)
            return loss

        if calc_conditions_and_gradients is not None:
            loss_jit, grad_jit, _ = calc_conditions_and_gradients
        else:
            loss_jit = jax.jit(loss_fn)
            grad_jit = jax.jit(jax.grad(loss_fn))

        # Initial guess
        if u_warm is not None:
            u_init = np.array(u_warm[:E], dtype=float)
        else:
            lo, hi = INIT_LOG_RATIO_RANGE_DEFAULT
            u_init = np.random.uniform(lo, hi, E).astype(float)

        if optimize_detunings:
            x0 = np.concatenate([u_init, np.zeros(N)])
        else:
            x0 = u_init

        loss_history = []
        def callback(x):
            loss_history.append(float(loss_jit(jnp.array(x, dtype=float))))

        solver_opts = dict(self.solver_options)
        solver_opts.update(kwargs_solver)

        result = sciopt.minimize(
            fun=lambda x: float(loss_jit(jnp.array(x, dtype=float))),
            jac=lambda x: np.array(grad_jit(jnp.array(x, dtype=float)), dtype=float),
            x0=x0,
            method=method,
            bounds=None,
            options=solver_opts,
            callback=callback,
        )

        x_sol = result.x
        log_ratios_sol = x_sol[:E]
        detunings_sol = x_sol[E:] if optimize_detunings else np.zeros(N)
        final_loss = float(loss_jit(jnp.array(x_sol, dtype=float)))
        success = final_loss < max_violation_success

        # Reconstruct A and sigma at solution
        ratios_sol = jnp.exp(jnp.array(log_ratios_sol))
        A_sol = build_drift_matrix_from_ratios(nodes, edges, ratios_sol, lambda_scale)
        for i in range(N):
            s = slice(2 * i, 2 * i + 2)
            A_sol = A_sol.at[s, s].add(float(detunings_sol[i]) * J2)
        sigma_full_sol = solve_lyapunov_kronecker(A_sol, D)
        sigma_achieved = get_mode_covariance(sigma_full_sol, target_mode_ids)

        # Physical quantities per edge
        coupling_strengths_sol = []
        cooperativities = {}
        coupling_ratios_dict = {}
        for k, edge in enumerate(edges):
            ni, nj = nodes[edge['i']], nodes[edge['j']]
            di = ni.get('kappa', ni.get('gamma'))
            dj = nj.get('kappa', nj.get('gamma'))
            u_k = float(log_ratios_sol[k])
            C_tilde = float(np.exp(u_k))
            C_k = lambda_scale * C_tilde
            g_k = float(np.sqrt(max(C_k * di * dj / 4., 0.)))
            coupling_strengths_sol.append(g_k)
            label = f"({edge['i']},{edge['j']},{edge['type'][:3]})"
            cooperativities[label] = C_k
            coupling_ratios_dict[f'C~_{label}'] = C_tilde

        if verbosity:
            print(f'  loss={final_loss:.3e}  success={success}  nit={result.nit}')

        info_out = {
            'initial_guess'      : x0,
            'free_idxs'          : self.give_free_variable_idxs(conditions),
            'solution'           : x_sol,
            'log_ratios'         : log_ratios_sol,
            'coupling_ratios'    : coupling_ratios_dict,
            'detunings'          : {f'Delta_{i}': float(detunings_sol[i]) for i in range(N)},
            'lambda_scale'       : lambda_scale,
            'coupling_strengths' : np.array(coupling_strengths_sol),
            'cooperativities'    : cooperativities,
            'physical_formula'   : 'g_k = sqrt(lambda * C~_k * decay_i * decay_j / 4)',
            'final_cost'         : final_loss,
            'success'            : success,
            'optimizer_message'  : result.message,
            'A'                  : np.array(A_sol),
            'sigma_full'         : np.array(sigma_full_sol),
            'sigma_achieved'     : np.array(sigma_achieved),
            'sigma_target'       : self.sigma_target,
            'nit'                : result.nit,
            'loss_history'       : loss_history,
        }
        return success, info_out

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
        lambda_scale: float = None,
        u_warm=None,
        verbosity: bool = False,
        max_violation_success: float = 1e-8,
        interrupt_if_successful: bool = True,
        **kwargs_solver,
    ):
        if lambda_scale is None:
            lambda_scale = LAMBDA_SCALE_DEFAULT

        # Build the JIT-compiled functions once, reuse across restarts
        from reservoir_engineering.topology_search import translate_triu_to_conditions
        if conditions is not None:
            calc_cag = self.give_conditions_func_with_conditions(conditions)
        elif triu_array is not None:
            conds = translate_triu_to_conditions(triu_array, self.node_types)
            calc_cag = self.give_conditions_func_with_conditions(conds)
        else:
            calc_cag = None

        successes = []
        infos = []
        for _ in range(num_tests):
            success, info = self.optimize_given_conditions(
                conditions=conditions,
                triu_array=triu_array,
                lambda_scale=lambda_scale,
                u_warm=u_warm,
                verbosity=verbosity,
                max_violation_success=max_violation_success,
                calc_conditions_and_gradients=calc_cag,
                **kwargs_solver,
            )
            successes.append(success)
            infos.append(info)
            if success and interrupt_if_successful:
                break

        return bool(np.any(successes)), infos, np.where(successes)

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
        from reservoir_engineering.topology_search import translate_conditions_to_triu

        if threshold is None:
            threshold = 1e-6

        fulfilled = []
        A_jnp = jnp.array(A)
        sigma_jnp = jnp.array(sigma)
        for c in self.all_possible_constraints:
            try:
                residual = float(c(A_jnp, sigma_jnp))
                if abs(residual) < threshold:
                    fulfilled.append(c)
            except Exception:
                pass

        return translate_conditions_to_triu(fulfilled, self.num_modes, self.node_types)

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
    #       allowed_entries = [NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
    #                          BEAMSPLITTER_AND_TWO_MODE_SQUEEZING]
    #       Four off-diagonal choices. BEAMSPLITTER_AND_TWO_MODE_SQUEEZING(4)
    #       represents both drives simultaneously (e.g. Kronwald topology).
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
        from reservoir_engineering.constraints import Constraint_coupling_absent
        from reservoir_engineering.topology_search import (
            NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
            PARAMETRIC, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)

        possible_entry_lists = []
        for i, j in self.all_possible_edges:
            forced_absent = any(
                isinstance(c, Constraint_coupling_absent) and
                c.idxs == [min(i, j), max(i, j)]
                for c in self.enforced_constraints)
            if forced_absent:
                possible_entry_lists.append([NO_COUPLING])
            elif i == j:
                possible_entry_lists.append([NO_COUPLING, PARAMETRIC])
            else:
                possible_entry_lists.append([
                    NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
                    BEAMSPLITTER_AND_TWO_MODE_SQUEEZING])

        self.possible_entry_lists = possible_entry_lists
        self.list_of_triu_arrays = [
            np.array(combo, dtype=np.int8)
            for combo in itertools_product(*possible_entry_lists)]
        self.complexity_levels = [int(np.sum(t)) for t in self.list_of_triu_arrays]
        self.unique_complexity_levels = sorted(set(self.complexity_levels))
        self.num_possible_graphs = len(self.list_of_triu_arrays)

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
        from reservoir_engineering.topology_search import check_if_subgraph_triu

        potential = []
        for triu, c in zip(self.list_of_triu_arrays, self.complexity_levels):
            if c != complexity_level:
                continue
            # Skip if this triu is a superset of a known invalid
            if self.invalid_combinations:
                if check_if_subgraph_triu([triu], self.invalid_combinations):
                    continue
            # Skip if a known valid is already a subgraph of this triu (redundant)
            if not skip_check_for_valid_subgraphs and self.valid_combinations:
                if check_if_subgraph_triu([triu], self.valid_combinations):
                    continue
            potential.append(triu)
        return potential

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
    #           continue   ← skip silently. DO NOT add to invalid_combinations.
    #           Reason: a supergraph (more edges) can be stable even if this
    #           topology is not. Adding to invalid_combinations would prune
    #           valid supersets (e.g., TMS-only unstable → would prune BS+TMS).
    #
    #       ── STAGE 2: Convergence test (DISABLED in prototype) ────────────
    #       # u_warm, stage2_ok = check_convergence(triu_array)
    #       # In prototype: check_convergence always returns (None, True).
    #       # u_warm = None → Stage 3 starts from random initialisation.
    #       # No pruning based on convergence in the prototype.
    #       u_warm, stage2_ok = check_convergence(triu_array)   # returns (None, True)
    #       # stage2_ok is always True in prototype — no topology is pruned here.
    #
    #       ── STAGE 3: Optimise log coupling ratios ────────────────────────
    #       conditions = translate_triu_to_conditions(triu_array)
    #       If check_if_subgraph_triu(triu_array, newly_added_combos): skip
    #       success, all_infos, _ = repeated_optimization(
    #           conditions=conditions, lambda_scale=LAMBDA_SCALE_DEFAULT,
    #           u_warm=u_warm, ...)
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
        from reservoir_engineering.topology_search import (
            translate_triu_to_conditions, check_if_subgraph_triu)

        if combinations_to_test is None:
            combinations_to_test = self.identify_potential_combinations(complexity_level)

        newly_added = []
        num_tested = 0
        num_skipped = 0

        for triu_array in combinations_to_test:

            # STAGE 1: stability at unit cooperativity (fast filter, no gradient)
            if not self.check_stability_unit_cooperativity(triu_array):
                num_skipped += 1
                continue  # DO NOT add to invalid_combinations

            # STAGE 2: convergence test (prototype: always returns (None, True))
            u_warm, stage2_ok = self.check_convergence(triu_array)
            if not stage2_ok:
                self.invalid_combinations.append(triu_array)
                num_skipped += 1
                continue

            # Skip if a newly-found valid is already a subgraph of this triu
            if newly_added and check_if_subgraph_triu([triu_array], newly_added):
                num_skipped += 1
                continue

            # STAGE 3: gradient optimisation of log coupling ratios
            conditions = translate_triu_to_conditions(triu_array, self.node_types)
            success, infos, _ = self.repeated_optimization(
                num_tests=self.kwargs_optimization['num_tests'],
                conditions=conditions,
                lambda_scale=LAMBDA_SCALE_DEFAULT,
                u_warm=u_warm,
                max_violation_success=self.kwargs_optimization['max_violation_success'],
                interrupt_if_successful=self.kwargs_optimization['interrupt_if_successful'],
            )
            num_tested += 1

            if success:
                best_info = min(infos, key=lambda x: x['final_cost'])
                if perform_graph_reduction:
                    try:
                        minimal_triu = self.check_all_constraints(
                            best_info['A'], best_info['sigma_full'])
                    except Exception:
                        minimal_triu = triu_array
                else:
                    minimal_triu = triu_array
                self.valid_combinations.append(minimal_triu)
                self.best_info_list.append(best_info)
                newly_added.append(minimal_triu)
            else:
                self.invalid_combinations.append(triu_array)
                num_skipped += 1

        self.tested_complexities.append(complexity_level)
        self.num_tested_graphs.append(num_tested)
        self.num_tested_invalid_graphs.append(num_skipped)

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
        from reservoir_engineering.topology_search import check_if_subgraph_triu

        if not self.valid_combinations:
            return

        # Deduplicate
        seen = set()
        unique = []
        for t in self.valid_combinations:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique.append(t)

        # Remove any triu that has a simpler valid triu as a subgraph (redundant)
        kept = []
        for i, t in enumerate(unique):
            others = unique[:i] + unique[i + 1:]
            if others and check_if_subgraph_triu([t], others):
                continue  # a simpler valid topology is contained in t → discard t
            kept.append(t)

        self.valid_combinations = kept

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
        self.prepare_all_possible_combinations()
        print(f'{self.num_possible_graphs} graphs identified')

        for c in self.unique_complexity_levels:
            print(f'testing complexity {c}')
            self.find_valid_combinations(c)
            self.cleanup_valid_combinations()
            if self.valid_combinations:
                print(f'  found {len(self.valid_combinations)} valid topology/topologies')

        if self.valid_combinations:
            return np.array(self.valid_combinations, dtype=np.int8)
        return np.array([], dtype=np.int8)

    # -----------------------------------------------------------------------
    # count_valid_invalid_graphs_layers() → (complexities, valid_counts, invalid_counts)
    # -----------------------------------------------------------------------
    # MIRRORS: count_valid_invalid_graphs_layers() in Architecture_Optimizer
    #
    # Count how many valid/invalid graphs exist at each complexity level.
    # Used for plotting the "landscape" of the search space.
    # Returns arrays of counts per complexity level, for analysis.py to plot.

    def count_valid_invalid_graphs_layers(self):
        complexities = getattr(self, 'unique_complexity_levels', [])
        valid_counts = [
            sum(1 for t in self.valid_combinations if int(np.sum(t)) == c)
            for c in complexities]
        invalid_counts = [
            sum(1 for t in self.invalid_combinations if int(np.sum(t)) == c)
            for c in complexities]
        return complexities, valid_counts, invalid_counts

    # -----------------------------------------------------------------------
    # extract_cooperativities(conditions, solution_log_ratios, lambda_scale=None) → dict
    # -----------------------------------------------------------------------
    # MIRRORS: extract_cooperativities_and_human_defined_parameters(conditions, solution_dict)
    #          in Architecture_Optimizer  (EXACT analogue)
    #
    # Compute and report all physically relevant quantities for the solution.
    # Accepts LOG coupling ratios u_k (Stage 3 raw output) and converts to
    # physical quantities.
    #
    # Parameters:
    #   conditions          : list of constraint objects (defines which edges are active)
    #   solution_log_ratios : np.ndarray (E_free,), u_k = log(C̃_k) per free edge
    #   lambda_scale        : float, the λ used in Stage 3 (default LAMBDA_SCALE_DEFAULT)
    #
    # Returns:
    #   dict with the following keys per active edge (i, j):
    #     'u_{i,j}'     : float — log ratio u_k (raw Stage 3 output)
    #     'C̃_{i,j}'    : float — dimensionless ratio C̃_k = exp(u_k)
    #     'C_{i,j}'     : float — actual cooperativity C_k = λ · C̃_k
    #     'g_{i,j}'     : float — physical coupling g_k = sqrt(C_k · decay_i · decay_j / 4)
    #     'formula'     : str — 'g = sqrt(λ · exp(u) · decay_i · decay_j / 4)'
    #
    # For the Kronwald scheme:
    #   C̃_g = exp(u_BS),  C̃_ν = exp(u_TMS)
    #   C_g = λ · C̃_g,   C_ν = λ · C̃_ν   (both large at λ=1000)
    #   squeezing r = atanh(sqrt(C̃_ν / C̃_g))  ← ratio is λ-independent
    #   g = sqrt(λ · C̃_g · κ · γ / 4)          ← physical coupling for given κ, γ
    #
    # AutoScatter uses:  C_{i,j} = 4 * |g_{i,j}|²  (κ_ext normalised to 1).
    # Here:              C_{i,j} = λ · C̃_i = λ · exp(u_i).

    def extract_cooperativities(
        self,
        conditions: list,
        solution_log_ratios,
        lambda_scale=None,
    ) -> dict:
        from reservoir_engineering.topology_search import translate_conditions_to_triu, TopologyGraph

        if lambda_scale is None:
            lambda_scale = LAMBDA_SCALE_DEFAULT

        triu_array = translate_conditions_to_triu(conditions, self.num_modes, self.node_types)
        default_kappa = next((n['kappa'] for n in self.nodes if n['type'] == 'cavity'), 1.0)
        default_gamma = next((n['gamma'] for n in self.nodes if n['type'] == 'mechanical'), 0.01)
        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts(
            default_kappa=default_kappa, default_gamma=default_gamma)

        result = {}
        for k, edge in enumerate(edges):
            ni, nj = nodes[edge['i']], nodes[edge['j']]
            di = ni.get('kappa', ni.get('gamma'))
            dj = nj.get('kappa', nj.get('gamma'))
            u_k = float(solution_log_ratios[k])
            C_tilde = float(np.exp(u_k))
            C_k = lambda_scale * C_tilde
            g_k = float(np.sqrt(max(C_k * di * dj / 4., 0.)))
            label = f"({edge['i']},{edge['j']},{edge['type'][:3]})"
            result[f'u_{label}']  = u_k
            result[f'C~_{label}'] = C_tilde
            result[f'C_{label}']  = C_k
            result[f'g_{label}']  = g_k
        result['formula'] = 'g_k = sqrt(lambda * C~_k * decay_i * decay_j / 4)'
        return result

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
        free_idxs = self.give_free_variable_idxs(conditions)
        result = {}
        for k, idx in enumerate(free_idxs):
            i, j = self.all_possible_edges[idx]
            result[f'g_{i}{j}'] = float(solution_cs[k])
        return result
