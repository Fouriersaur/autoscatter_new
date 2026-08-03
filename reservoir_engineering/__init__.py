"""
reservoir_engineering
=====================
Automated discovery of dissipative quantum state engineering topologies.
MIRRORS: autoscatter/__init__.py in structure and philosophy.

AutoScatter targets a SCATTERING MATRIX (S-matrix, frequency domain) via
input-output theory. This package targets a COVARIANCE MATRIX (sigma, time
domain) via the Lyapunov equation. The pipeline is otherwise identical.

File structure (mirrors autoscatter/ file-by-file):
  scattering.py            -> covariance_physics.py   (S-matrix -> Lyapunov A sigma+sigma A^T+D=0)
  constraints.py            -> constraints.py           (same two-level constraint hierarchy)
  architecture.py            -> topology_search.py       (triu encoding, subgraph checks, TopologyGraph)
  architecture_optimizer.py -> covariance_optimizer.py  (Architecture_Optimizer -> CovarianceOptimizer)
  symbolic.py / jax_functions.py -> not needed (real quadrature basis, no sympy)
  (no analogue)              -> targets.py               (standard target covariances + §6 functionals)
  (no analogue)              -> benchmarks.py             (Kronwald/Wang-Clerk/Woolley-Clerk/... schemes)
  (partial analogue)         -> analysis.py               (visualisation/analysis tools)
  (no analogue)               -> gaussian_states.py        (Siegel matrix, dark-state kernel, Approach-A ceiling)

Typical workflow:
  1. Define target state             — targets.py (e.g. squeezed_vacuum(r=1.0))
  2. Find minimum auxiliary modes     — find_minimum_number_auxiliary_modes(...)
  3. Run breadth-first topology search — optimizer.perform_breadth_first_search()
       (per candidate: Stage 1 stability filter, Stage 2 gradient
        optimisation of the (G-tilde,C-tilde) coupling ratios — see
        covariance_optimizer.py's module docstring for the full two-stage
        algorithm)
  4. Inspect results                  — constraints.plot_graph, analysis.py tools
  5. Validate against known benchmarks — benchmarks.run_benchmark('kronwald', ...)
  6. Apply to unknown targets          — e.g. targets.cluster_state(...)

Key differences from AutoScatter: couplings are real, not complex (real
quadrature basis has no phase degree of freedom, so fewer optimisation
variables and no gauge-phase optimisation — sigma is gauge-invariant);
stability isn't automatic (TMS edges can destabilise A, unlike AutoScatter's
always-stable passive systems); sigma_target is for SIGNAL modes while
AutoScatter's S-matrix is for port modes (auxiliary modes are the
dissipative resource here, hidden ports there).

Alignment with gaussian_autoscatter_algorithm.md:
  §2.1 target predicate {(Q_i, q_i, weight, modes)}
        -> targets.py: TargetTerm, target_quadratic_form,
        target_collective_quadrature, target_log_negativity, target_purity;
        CovarianceOptimizer(target_predicate=[...]) — sigma_target remains
        supported as the convenience "match every entry at once" case, and
        may be combined with target_predicate.
  §1.2 (G,C) machine, A=K+Omega Im(C^dag C), D=Omega Re(C^dag C) Omega^T
        -> covariance_physics.py: build_hamiltonian_matrix (=K=Omega*G),
        build_jump_matrix (=C), build_drift_matrix_from_GC, build_diffusion_matrix_from_C
  §1.3/§1.4 Siegel matrix Z, dark-state kernel, Approach-A ceiling
        -> gaussian_states.py: sigma_from_siegel, dark_state_nullifiers, optimise_state
  §2.6 constraint registry (structural/domain/soft)
        -> constraints.py: Base_Constraint.slot, ConstraintRegistry,
        Constraint_passivity, Constraint_cooperativity_cap
        -> covariance_optimizer.CovarianceOptimizer(squeezable_aux_ids=[...]):
        §2.6(b) domain palette declaring which aux cavity nodes MAY use a
        squeezed (Bogoliubov) bath instead of plain vacuum; the optimizer
        (not the human) decides whether/how much — see
        covariance_physics.build_jump_matrix and info_out['bath_squeezing']
  §3 inner-loop loss (target + stability + optional purity/reg + soft terms)
        -> covariance_optimizer.py: CovarianceOptimizer(lambda_pure=,
        lambda_reg=, normalize_targets=), optimize_given_conditions
  §4 discrete outer loop (enumerate -> test -> prune -> canonicalise -> reduce)
        -> covariance_optimizer.py: prepare_all_possible_combinations,
        find_valid_combinations, cleanup_valid_combinations,
        perform_breadth_first_search; topology_search.canonicalise_by_gauge
  §2.5 cost metric / ranking -> covariance_optimizer.rank_by_cost
  §5 full algorithm (ceiling -> for M in M_min..M_max -> rank)
        -> covariance_optimizer.run_algorithm
  §6 target-quantity reference -> targets.py (+ covariance_physics.purity_violation)
"""

# Physics layer — mirrors autoscatter/scattering.py
from reservoir_engineering.covariance_physics import (
    build_hamiltonian_matrix,        # K = Omega*G, the coherent drift generator (no decay)
    build_drift_matrix,              # A = K + Omega Im(C^dag C)
    build_diffusion_matrix,          # D = Omega Re(C^dag C) Omega^T
    solve_lyapunov_kronecker,        # solve A sigma + sigma A^T + D = 0
    get_mode_covariance,             # extract sigma_sub for target modes
    check_stability,                 # Stage 1: is A Hurwitz?
    build_drift_diffusion_from_GC_tilde,  # §1.5 (G-tilde,C-tilde), works at decay=0
    covariance_loss,                 # 1/2||sigma_sub - target||_F^2
    symplectic_form,                 # Omega
    build_jump_matrix,               # C, the jump-operator matrix (§1.2)
    build_drift_matrix_from_GC,      # general (G,C) machine (§1.2)
    build_diffusion_matrix_from_C,   # general (G,C) machine (§1.2)
    purity_violation,                # optional §3 purity loss term
    jnp_symplectic_eigenvalues,      # §6 functionals, JAX-differentiable
    jnp_purity,
    jnp_log_negativity,
)

# Pure-state / Siegel-matrix layer — no AutoScatter analogue (§1.3/§1.4/§5/§7)
from reservoir_engineering.gaussian_states import (
    sigma_from_siegel,          # Z=X+iY -> pure-state covariance
    dark_state_nullifiers,      # ker(sigma+i/2 Omega) -> ideal dissipator directions
    optimise_state,             # Approach-A state-level ceiling
)

# Constraint objects — mirrors autoscatter/constraints.py
from reservoir_engineering.topology_search import (
    BEAMSPLITTER_AND_TWO_MODE_SQUEEZING, # edge type 4: both drives on one pair
)
from reservoir_engineering.constraints import (
    NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING, PARAMETRIC,
    Constraint_coupling_absent,
    Constraint_coupling_beamsplitter,
    Constraint_coupling_two_mode_squeezing,
    Constraint_coupling_symmetric,      # ties two edges' coupling strengths together
    Constraint_stability,
    Constraint_physical_state,
    Constraint_target_squeezing,
    Constraint_entanglement,
    Constraint_passivity,               # §2.6(b) domain: passive-only modes
    Constraint_cooperativity_cap,       # §2.6(c) soft: hinge penalty on G_tilde_max
    ConstraintRegistry,                 # §2.6: structural/domain/soft classifier
    setup_constraints,
    plot_graph,
    plot_list_of_graphs,
)

# Graph encoding utilities — mirrors autoscatter/architecture.py
from reservoir_engineering.topology_search import (
    TopologyGraph,
    triu_to_edge_matrix,
    edge_matrix_to_triu,
    check_if_subgraph,
    check_if_subgraph_triu,
    translate_triu_to_conditions,
    translate_conditions_to_triu,
    characterize_topology,
    find_min_number_pump_tones,
    calc_number_of_possibilities,
    canonicalise_by_gauge,              # §4 step 3 / §5: residual gauge quotient
)

# Main optimiser — mirrors autoscatter/architecture_optimizer.py
from reservoir_engineering.covariance_optimizer import (
    CovarianceOptimizer,
    find_minimum_number_auxiliary_modes,
    rank_by_cost,                           # §2.5 cost metric ranking
    run_algorithm,                          # §5 full-algorithm top-level driver
    AUTODIFF_FORWARD,
    AUTODIFF_REVERSE,
    KAPPA_0_DEFAULT,                        # §1.5 reference rate
    DIRECT_LOG_BOUND_DEFAULT,
    SQUEEZE_AB_BOUND_DEFAULT,                # §2.6(b) squeezed-bath palette bound
)

# Target covariance matrices — no AutoScatter analogue
from reservoir_engineering.targets import (
    squeezed_vacuum, two_mode_squeezed, vacuum, thermal, cluster_state,
    is_physical, symplectic_eigenvalues, squeezing_db, log_negativity,
    duan_criterion, purity,
    mean_energy, collective_quadrature_variance, fidelity,   # §6
    TargetTerm, target_quadratic_form, target_collective_quadrature,  # §2.1 target predicate
    target_log_negativity, target_purity,
)

# Benchmarks — no AutoScatter analogue
from reservoir_engineering.benchmarks import (
    ALL_BENCHMARKS,
    get_benchmark,
    make_benchmark_optimizer,
    run_benchmark,
    run_all_benchmarks,
    print_benchmark_summary,
)

# Analysis and visualisation — partial analogue (AutoScatter puts
# visualisation in constraints.py; this package also has a dedicated module)
from reservoir_engineering.analysis import (
    compare_covariance,
    print_topology_summary,
    plot_optimization_history,
    summarise_search_results,
    plot_squeezing_vs_complexity,
    validate_kronwald,
)

__all__ = [
    # physics
    'build_drift_matrix', 'build_diffusion_matrix',
    'solve_lyapunov_kronecker', 'get_mode_covariance', 'covariance_loss',
    'symplectic_form', 'build_jump_matrix',
    'build_drift_matrix_from_GC', 'build_diffusion_matrix_from_C',
    'purity_violation',
    'jnp_symplectic_eigenvalues', 'jnp_purity', 'jnp_log_negativity',
    # pure-state / Siegel layer (§1.3/§1.4/§5/§7)
    'sigma_from_siegel', 'dark_state_nullifiers', 'optimise_state',
    # constraints
    'NO_COUPLING', 'BEAMSPLITTER', 'TWO_MODE_SQUEEZING', 'PARAMETRIC',
    'BEAMSPLITTER_AND_TWO_MODE_SQUEEZING',
    'Constraint_coupling_absent', 'Constraint_coupling_beamsplitter',
    'Constraint_coupling_two_mode_squeezing', 'Constraint_coupling_symmetric',
    'Constraint_stability', 'Constraint_physical_state',
    'Constraint_target_squeezing', 'Constraint_entanglement',
    'Constraint_passivity', 'Constraint_cooperativity_cap', 'ConstraintRegistry',
    'setup_constraints', 'plot_graph', 'plot_list_of_graphs',
    # graph utilities
    'TopologyGraph', 'triu_to_edge_matrix', 'edge_matrix_to_triu',
    'check_if_subgraph', 'check_if_subgraph_triu',
    'translate_triu_to_conditions', 'translate_conditions_to_triu',
    'characterize_topology', 'find_min_number_pump_tones',
    'calc_number_of_possibilities', 'canonicalise_by_gauge',
    # physics helpers (Stage 1–2)
    'build_hamiltonian_matrix',
    'check_stability', 'build_drift_diffusion_from_GC_tilde',
    # optimiser
    'CovarianceOptimizer', 'find_minimum_number_auxiliary_modes',
    'rank_by_cost', 'run_algorithm',
    'AUTODIFF_FORWARD', 'AUTODIFF_REVERSE',
    'KAPPA_0_DEFAULT', 'DIRECT_LOG_BOUND_DEFAULT', 'SQUEEZE_AB_BOUND_DEFAULT',
    # targets
    'squeezed_vacuum', 'two_mode_squeezed', 'vacuum', 'thermal',
    'cluster_state', 'is_physical', 'symplectic_eigenvalues',
    'squeezing_db', 'log_negativity', 'duan_criterion', 'purity',
    'mean_energy', 'collective_quadrature_variance', 'fidelity',
    'TargetTerm', 'target_quadratic_form', 'target_collective_quadrature',
    'target_log_negativity', 'target_purity',
    # benchmarks
    'ALL_BENCHMARKS', 'get_benchmark', 'make_benchmark_optimizer',
    'run_benchmark', 'run_all_benchmarks', 'print_benchmark_summary',
    # analysis
    'compare_covariance', 'print_topology_summary',
    'plot_optimization_history', 'summarise_search_results',
    'plot_squeezing_vs_complexity', 'validate_kronwald',
]
