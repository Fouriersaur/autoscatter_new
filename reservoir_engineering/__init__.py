"""
reservoir_engineering
=====================
Automated discovery of dissipative quantum state engineering topologies.

MIRRORS: autoscatter/__init__.py in structure and philosophy.

AutoScatter targets a SCATTERING MATRIX (S-matrix, frequency domain) via
input-output theory.  This package targets a COVARIANCE MATRIX (σ, time
domain) via the Lyapunov equation.  The pipeline is otherwise identical.

═══════════════════════════════════════════════════════════════════════════════
FILE STRUCTURE  (mirrors autoscatter/ file-by-file)
═══════════════════════════════════════════════════════════════════════════════

autoscatter/                    reservoir_engineering/
────────────────────────────    ────────────────────────────────────────────
scattering.py                → covariance_physics.py
  S-matrix via input-output      σ-matrix via Lyapunov equation
  S = I + (-iH - κ/2)⁻¹         Aσ + σAᵀ + D = 0

constraints.py               → constraints.py
  Constraint_coupling_zero       Constraint_coupling_absent
  Constraint_coupling_phase_zero Constraint_coupling_beamsplitter
  MinimalAddedInputNoise         Constraint_stability
  MinimalAddedOutputNoise        Constraint_physical_state
  plot_graph                     plot_graph (same API)

architecture.py              → topology_search.py
  triu_matrix encoding           triu_array encoding (NO_COUPLING=0,
  NO_COUPLING=0                    BEAMSPLITTER=1,
  COUPLING_WITHOUT_PHASE=1         TWO_MODE_SQUEEZING=2,
  COUPLING_WITH_PHASE=2            PARAMETRIC=3)
  find_min_number_of_pumps       find_min_number_pump_tones
  check_if_subgraph_upper_tri    check_if_subgraph_triu
  Architecture class             TopologyGraph class

architecture_optimizer.py    → covariance_optimizer.py
  Architecture_Optimizer         CovarianceOptimizer
  S_target (sympy Matrix)        sigma_target (numpy array)
  mode_types (list of bool)      node_types (list of str)
  gabs, gphases, Deltas          coupling_strengths (real array, no phases)
  gauge phases                   (none — σ is gauge-invariant)
  C_{i,j} = 4|g_{i,j}|²         C_{i,j} = 4g_{i,j}²/(κ_i·κ_j)
  perform_breadth_first_search   perform_breadth_first_search (same algorithm)

symbolic.py                  → (not needed — real quadrature basis, no sympy)
jax_functions.py             → (absorbed in covariance_physics.py)

─── no AutoScatter analogue ─   targets.py
                                  Standard target covariances
                                  (squeezed_vacuum, two_mode_squeezed, cluster_state...)

─── no AutoScatter analogue ─   benchmarks.py
                                  Known schemes for validation:
                                  Kronwald, Wang-Clerk, Woolley-Clerk,
                                  three-mode cluster, etc.

─── partial analogue ────────   analysis.py
  (visualization in constraints.py)  Extended visualization + analysis tools

═══════════════════════════════════════════════════════════════════════════════
TYPICAL WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

Step 1 — Define target state (targets.py):
    from reservoir_engineering.targets import squeezed_vacuum
    sigma_target = squeezed_vacuum(r=1.0)

Step 2 — Find minimum auxiliary modes (covariance_optimizer.py):
    from reservoir_engineering.covariance_optimizer import find_minimum_number_auxiliary_modes
    optimizer = find_minimum_number_auxiliary_modes(
        sigma_target       = sigma_target,
        target_mode_ids    = [0],           # mechanical mode is signal
        node_types_signal  = ['mechanical'],
        start_value=0, max_value=3,
    )
    # Discovers: 1 auxiliary cavity is needed (Kronwald topology)

Step 3 — Run breadth-first topology search (covariance_optimizer.py):
    valid_topologies = optimizer.perform_breadth_first_search()
    # For EACH candidate topology the BFS runs THREE internal stages:
    #
    # STAGE 1 — Stability at unit cooperativity (fast, no gradient):
    #     Build A from H with all C_{ij}=1 (g_{ij} = sqrt(decay_i·decay_j/4)).
    #     A is the real 2N×2N "coupling matrix" derived from H via quantum
    #     Langevin equations.  Discard if any Re(eigenvalue of A) ≥ 0.
    #
    # STAGE 2 — Find scaling exponents {β_i} (combinatorial, no gradient):
    #     Set C_i = λ^{β_i}, rebuild A from H at λ=10,100,1000, test V_system
    #     convergence.  Search β_i ∈ {0.5,1,2} per edge.  Discard if none converge.
    #     Kronwald expected result: β_g = β_ν = 1 (same-scaling passes first).
    #
    # STAGE 3 — Optimise coupling ratios {C̃_i} at fixed λ=1000 (gradient):
    #     Rebuild A from H at each step: g_k = sqrt(λ^{β_k}·C̃_k·decay_i·decay_j/4).
    #     Minimise ½‖V_system − V_target‖²_F over C̃_i > 0 using L-BFGS-B + JAX.
    #     Actual cooperativity: C_i = λ^{β_i} · C̃_i.
    #     Physical coupling: g_i = sqrt(λ^{β_i} · C̃_i · decay_i · decay_j / 4).
    #
    # Final output per valid topology:
    #     topology (graph), scaling_exponents {β_i}, coupling_ratios {C̃_i}, λ=1000

Step 4 — Inspect results (constraints.py, analysis.py):
    from reservoir_engineering.constraints import plot_graph
    from reservoir_engineering.analysis import (compare_covariance,
        summarise_search_results, plot_scaling_exponents, plot_cooperativity_ratios)
    plot_graph(...)
    plot_scaling_exponents(best_info)      # Stage 2 output: β_i per edge
    plot_cooperativity_ratios(best_info)   # Stage 3 output: C̃_i per edge
    summarise_search_results(valid_topologies)

Step 5 — Validate against known benchmarks (benchmarks.py):
    from reservoir_engineering.benchmarks import run_benchmark
    results = run_benchmark('kronwald', r_or_param=1.0)
    print_benchmark_summary(results)

Step 6 — Apply to unknown targets (cluster states):
    from reservoir_engineering.targets import cluster_state
    sigma_cluster = cluster_state(n_modes=3, delta=0.3)
    # Run the same pipeline — discovers the topology for you

═══════════════════════════════════════════════════════════════════════════════
KEY DIFFERENCE FROM AUTOSCATTER
═══════════════════════════════════════════════════════════════════════════════

AutoScatter optimises COMPLEX coupling strengths (|g| and phase arg(g)).
Here, all coupling strengths are REAL (no phase degree of freedom in the
real quadrature basis). This means:
  • Fewer optimisation variables per edge (one real number, not two).
  • No gauge-phase optimisation needed.
  • The cooperativity C_{i,j} = 4g²/(κ_i·κ_j) is purely real.

AutoScatter targets the S-matrix for PORT modes only; auxiliary modes are
hidden. Here, sigma_target is for SIGNAL modes (e.g. mechanical modes);
auxiliary modes (cavities) are the dissipative resource.
"""

# ── Physics layer ─────────────────────────────────────────────────────────
# Mirrors: autoscatter/scattering.py (S-matrix physics)
from reservoir_engineering.covariance_physics import (
    build_hamiltonian_matrix,        # H_quad (2N×2N real) — explicit Hamiltonian matrix
                                     # analogue of AutoScatter's complex N×N coupling matrix H
                                     # contains ONLY coherent terms (couplings + detunings, no decay)
    build_drift_matrix,              # A = H_quad + A_decay  (full drift matrix)
    build_diffusion_matrix,          # build D from nodes (constant, no coupling dependence)
    solve_lyapunov_kronecker,        # solve Aσ + σAᵀ + D = 0 via Kronecker trick
    get_mode_covariance,             # extract σ_sub for target modes
    check_stability,                 # Stage 1: check A is Hurwitz (all Re(λ) < 0)
    build_drift_matrix_from_ratios,  # Stage 3: A from coupling ratios C̃_i + exponents β_i
    covariance_loss,                 # ½‖σ_sub - σ_target‖²_F (JAX-differentiable)
    covariance_loss_from_ratios,     # Stage 3 loss: same but parametrised by C̃_i
)

# ── Constraint objects ────────────────────────────────────────────────────
# Mirrors: autoscatter/constraints.py (constraint objects + graph plotting)
from reservoir_engineering.constraints import (
    NO_COUPLING,                        # edge type constant = 0
    BEAMSPLITTER,                       # edge type constant = 1
    TWO_MODE_SQUEEZING,                 # edge type constant = 2
    PARAMETRIC,                         # edge type constant = 3
    Constraint_coupling_absent,         # mirrors Constraint_coupling_zero
    Constraint_coupling_beamsplitter,   # mirrors Constraint_coupling_phase_zero
    Constraint_coupling_two_mode_squeezing,
    Constraint_stability,               # mirrors MinimalAddedInputNoise
    Constraint_physical_state,          # mirrors MinimalAddedOutputNoise
    Constraint_target_squeezing,
    Constraint_entanglement,
    setup_constraints,                  # mirrors setup_constraints
    plot_graph,                         # mirrors plot_graph
    plot_list_of_graphs,                # mirrors plot_list_of_graphs
)

# ── Graph encoding utilities ──────────────────────────────────────────────
# Mirrors: autoscatter/architecture.py (graph data structures)
from reservoir_engineering.topology_search import (
    TopologyGraph,                      # mirrors Architecture class
    triu_to_edge_matrix,                # mirrors triu_to_adjacency_matrix
    edge_matrix_to_triu,                # mirrors adjacency_to_triu_matrix
    check_if_subgraph,                  # mirrors check_if_subgraph
    check_if_subgraph_triu,             # mirrors check_if_subgraph_upper_triangle
    translate_triu_to_conditions,       # mirrors translate_upper_triangle_coupling_matrix_to_conditions
    translate_conditions_to_triu,       # mirrors translate_conditions_to_upper_triangle_coupling_matrix
    characterize_topology,              # mirrors characterize_architecture
    find_min_number_pump_tones,         # mirrors find_min_number_of_pumps
    calc_number_of_possibilities,       # mirrors calc_number_of_possibilities
)

# ── Main optimiser ────────────────────────────────────────────────────────
# Mirrors: autoscatter/architecture_optimizer.py (Architecture_Optimizer)
from reservoir_engineering.covariance_optimizer import (
    CovarianceOptimizer,                    # mirrors Architecture_Optimizer
    find_minimum_number_auxiliary_modes,    # mirrors find_minimum_number_auxiliary_modes
    AUTODIFF_FORWARD,
    AUTODIFF_REVERSE,
    LAMBDA_SCALE_DEFAULT,                   # Stage 3 fixed scale λ = 1000
    EXPONENT_SEARCH_GRID_1,                 # Stage 2 β search: {0.5, 1.0, 2.0}
    EXPONENT_SEARCH_GRID_2,                 # Stage 2 β search: {0.5, 1.0, 1.5, 2.0, 3.0}
)

# ── Target covariance matrices ────────────────────────────────────────────
# No AutoScatter analogue (target is given by user as S_target sympy matrix)
from reservoir_engineering.targets import (
    squeezed_vacuum,            # ½ diag(e^{-2r}, e^{+2r})
    two_mode_squeezed,          # EPR / TMSV state
    vacuum,                     # ½ I (ground state)
    thermal,                    # (n̄+½) I
    cluster_state,              # N-mode Gaussian cluster state
    is_physical,                # check uncertainty principle
    symplectic_eigenvalues,     # physical invariants of Gaussian state
    squeezing_db,               # squeezing in decibels
    log_negativity,             # entanglement measure
    duan_criterion,             # inseparability criterion
)

# ── Benchmarks ────────────────────────────────────────────────────────────
# No AutoScatter analogue
from reservoir_engineering.benchmarks import (
    ALL_BENCHMARKS,
    get_benchmark,
    make_benchmark_optimizer,
    run_benchmark,
    run_all_benchmarks,
    print_benchmark_summary,
)

# ── Analysis and visualisation ────────────────────────────────────────────
# Partial analogue: AutoScatter puts visualisation in constraints.py;
# here we have a dedicated analysis.py with extended tools.
from reservoir_engineering.analysis import (
    compare_covariance,             # heatmap of achieved vs target
    print_topology_summary,         # human-readable topology description
    plot_optimization_history,      # loss vs iteration
    summarise_search_results,       # ranked table of discovered topologies
    plot_squeezing_vs_complexity,   # squeezing dB vs num_edges scatter plot
    validate_kronwald,              # ground-truth validation check
)

__all__ = [
    # physics
    'build_drift_matrix', 'build_diffusion_matrix',
    'solve_lyapunov_kronecker', 'get_mode_covariance', 'covariance_loss',
    # constraints
    'NO_COUPLING', 'BEAMSPLITTER', 'TWO_MODE_SQUEEZING', 'PARAMETRIC',
    'Constraint_coupling_absent', 'Constraint_coupling_beamsplitter',
    'Constraint_coupling_two_mode_squeezing',
    'Constraint_stability', 'Constraint_physical_state',
    'Constraint_target_squeezing', 'Constraint_entanglement',
    'setup_constraints', 'plot_graph', 'plot_list_of_graphs',
    # graph utilities
    'TopologyGraph', 'triu_to_edge_matrix', 'edge_matrix_to_triu',
    'check_if_subgraph', 'check_if_subgraph_triu',
    'translate_triu_to_conditions', 'translate_conditions_to_triu',
    'characterize_topology', 'find_min_number_pump_tones',
    'calc_number_of_possibilities',
    # physics helpers (Stage 1–3)
    'build_hamiltonian_matrix',
    'check_stability', 'build_drift_matrix_from_ratios', 'covariance_loss_from_ratios',
    # optimiser
    'CovarianceOptimizer', 'find_minimum_number_auxiliary_modes',
    'AUTODIFF_FORWARD', 'AUTODIFF_REVERSE',
    'LAMBDA_SCALE_DEFAULT', 'EXPONENT_SEARCH_GRID_1', 'EXPONENT_SEARCH_GRID_2',
    # targets
    'squeezed_vacuum', 'two_mode_squeezed', 'vacuum', 'thermal',
    'cluster_state', 'is_physical', 'symplectic_eigenvalues',
    'squeezing_db', 'log_negativity', 'duan_criterion',
    # benchmarks
    'ALL_BENCHMARKS', 'get_benchmark', 'make_benchmark_optimizer',
    'run_benchmark', 'run_all_benchmarks', 'print_benchmark_summary',
    # analysis
    'compare_covariance', 'print_topology_summary',
    'plot_optimization_history', 'summarise_search_results',
    'plot_squeezing_vs_complexity', 'validate_kronwald',
]
