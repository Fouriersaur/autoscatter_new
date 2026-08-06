"""
covariance_optimizer.py
=======================
Main optimiser class for covariance-matrix targeting.
MIRRORS: autoscatter/architecture_optimizer.py

CovarianceOptimizer has two responsibilities, exactly like Architecture_Optimizer:
  1. INNER LOOP — parameter optimisation for a fixed topology (which edges
     are absent / what type): find coupling strengths minimising the
     covariance loss. optimize_given_conditions, repeated_optimization.
  2. OUTER LOOP — topology discovery via breadth-first search: enumerate
     topologies by complexity, test each with the inner loop, prune via
     subgraph rules. prepare_all_possible_combinations,
     find_valid_combinations, identify_potential_combinations,
     cleanup_valid_combinations, perform_breadth_first_search.

Differences from AutoScatter: no sympy (A built directly in JAX from edge
dicts); couplings are real, not complex (real quadrature basis, no phase
variable); no gauge freedom (sigma is gauge-invariant); stability isn't
automatic (TMS edges can destabilise A, handled via Constraint_stability);
D is fixed (depends only on node types/kappa/gamma/n_th, not couplings).

Coupling parametrisation — §1.5's literal (G-tilde, C-tilde): every
coherent edge k gets a free log-ratio u_k with g_k = kappa0*exp(u_k)
(kappa0 = a single fixed reference rate, default 1.0); every AUXILIARY
node (every mode not in target_mode_ids) gets a free log-rate v_m with
decay_m = kappa0*exp(v_m). Signal-mode decay rates stay fixed inputs (may
be exactly 0 — the idealized §1.6 limit; nothing here divides by them, so
gamma=0 and gamma!=0 both just work — see
covariance_physics.build_drift_diffusion_from_GC_tilde). No lambda, no
per-edge decay_i*decay_j reconstruction, no reference-edge pinning (kappa0
already fixes the overall-scale gauge — see that function's docstring).

Two-stage algorithm run per candidate topology in the BFS outer loop:
  Stage 1 (check_stability_unit_cooperativity) — fast filter, no gradient:
      build A at unit cooperativity (g_ij = sqrt(decay_i*decay_j/4), a
      fixed reference coupling used ONLY for this cheap pre-filter, not
      tied to how Stage 2 later parametrises the real search), check
      Hurwitz. Failures are NOT added to invalid_combinations, since a
      supergraph can add stabilising edges (e.g. TMS-only is marginally
      unstable, but BS+TMS/Kronwald is stable) — this is a speed filter
      only, not a structural impossibility proof.
  Stage 2 (optimize_given_conditions) — the real test: optimise the
      coherent log-ratios u_k and auxiliary log-rates v_m (+ optionally
      detunings Delta_i) against the target loss (§3, target term + soft
      constraints + optional purity/regulariser terms). Success if loss <
      max_violation_success.

Output per successful topology: G_tilde/C_tilde_aux (exp(u_k)/exp(v_m)),
detunings, physical coupling strengths g_k=kappa0*exp(u_k), auxiliary decay
rates, and a post-hoc 'cooperativities' dict (C_k=4*g_k^2/(decay_i*decay_j),
computed from the achieved solution for reporting/rank_by_cost — not the
search variable). All Delta_i ~ 0 means resonant driving suffices
(Kronwald/Wang-Clerk); nonzero means off-resonance is required.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt
from tqdm import trange, tqdm
from itertools import product as itertools_product
from typing import List, Optional

jax.config.update("jax_enable_x64", True)

AUTODIFF_FORWARD  = 'autodiff_forward'
AUTODIFF_REVERSE  = 'autodiff_reverse'
DIFFERENCE_QUOTIENT = '2-point'

INIT_STRENGTH_RANGE_DEFAULT  = [0.01, 3.0]   # initial coupling strengths [g_lo, g_hi]
BOUNDS_STRENGTH_DEFAULT      = [0., np.inf]   # strengths are non-negative

INIT_LOG_RATIO_RANGE_DEFAULT = [-1.0, 1.0]    # initial u_k/v_m draw range (G~_k in [e^-1,e^1])
DETUNING_BOUND_DEFAULT       = 20.0

# Detunings must NOT be initialised at exactly 0. At Delta=0 a mode can be
# left completely undamped (a "dark" collective mode decoupled from every
# dissipative channel), so A has eigenvalues with Re=0 and the Lyapunov
# equation is SINGULAR: no steady state exists, the loss is undefined, and
# its gradient blows up (~1e18 in practice). L-BFGS-B's line search then
# fails on the very first step and the detunings never leave zero — every
# restart returns Delta=0 exactly. This is not a flat/stationary point the
# optimiser could climb out of; it is a pole of the objective.
#
# Woolley-Clerk-type schemes live exactly here: their Omega*(a^dag a -
# b^dag b) term exists precisely to split the otherwise DEGENERATE Bogoliubov
# modes (one combination of which is dark and never cooled), so Delta=0 is
# the one point at which the scheme is undefined — cf. Woolley & Clerk PRA
# 89, 063805 (2014) eq. (22a), Omega >> gamma. Starting there makes such
# topologies unreachable, and the search instead invents a spurious weak
# coupling whose only job is to break the dark mode and regularise the solve.
# Each restart therefore draws Delta_i ~ U(-scale, +scale); the jitter also
# supplies the asymmetry these schemes need (Delta_1 = -Delta_2 != 0).
# Override via kwargs_optimization['init_detuning_scale'] (0.0 restores the
# old always-zero behaviour). Note a finite mechanical gamma>0 damps the dark
# mode and removes the singularity independently of this.
INIT_DETUNING_SCALE_DEFAULT  = 1.0

# §1.5 (G-tilde, C-tilde) parametrisation: no lambda, no decay_i*decay_j
# reconstruction — g_k=kappa0*exp(u_k) directly, works identically at
# decay=0 or decay>0 (see covariance_physics.build_drift_diffusion_from_GC_tilde).
# KAPPA_0_DEFAULT is just a unit choice (§1.5's reference rate);
# DIRECT_LOG_BOUND_DEFAULT caps |u_k|,|v_m| so g_k/kappa0 stays in [1e-4,1e4].
KAPPA_0_DEFAULT           = 1.0
DIRECT_LOG_BOUND_DEFAULT  = np.log(1e4)   # ~9.21

# §2.6(b) domain palette: squeezed-bath (Bogoliubov) dissipator on a cavity
# aux node, in place of plain vacuum. Free vars (a_,b_) = Re/Im of
# M=sinh(r)e^{i theta} (see covariance_physics.build_jump_matrix) — an
# unconstrained 2D reparametrisation of (r,theta) with no periodic-phase
# wraparound, and (a_,b_)=(0,0) exactly recovers plain vacuum, so declaring
# a node squeezable OFFERS the resource without FORCING it: the optimizer
# finds both whether and how much to squeeze. SQUEEZE_AB_BOUND_DEFAULT caps
# |a_|,|b_| (i.e. |M|<=~50, r up to ~asinh(50)~4.6) for the same reason
# DIRECT_LOG_BOUND_DEFAULT caps u_k/v_m: keep the search finite.
SQUEEZE_AB_BOUND_DEFAULT  = 50.0

# §2.6(b) domain palette: beamsplitter couplings are COMPLEX (carry a free
# phase theta_k) by default — see covariance_physics.build_hamiltonian_matrix's
# phase generalisation and Zippilli & Vitali PRL 126, 020402 (2021) eq.
# S.27-S.29 (their Lemma: the passivity-preserving coupling phase is
# generically nonzero, fixed by the squeezing phases of whatever's coupled
# to that edge — forcing real, as this package used to do unconditionally,
# is a restriction, not a baseline). Use constraints.Constraint_real_coupling
# to opt a specific edge back into real-only. One period is enough range
# (theta is genuinely periodic, unlike the log-space u_k/v_m).
PHASE_BOUND_DEFAULT      = np.pi


# Find the minimum number of auxiliary cavity modes needed to satisfy the
# target predicate, by trying num_aux = start_value, start_value+1, ... and
# returning the first CovarianceOptimizer whose fully-connected graph
# succeeds (make_initial_test=True raises if it can't). sigma_target may be
# None if a target_predicate (§2.1) is passed instead via **kwargs_optimizer.
def find_minimum_number_auxiliary_modes(
    sigma_target=None,
    target_mode_ids: List[int] = None,
    node_types_signal: List[str] = None,
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


# §2.5 cost metric: rank results by (num_auxiliary_modes, num_couplings,
# num_active_couplings [drive-tone proxy], max_cooperativity) ascending —
# smaller is "more minimal". results: list of dicts with keys
# 'num_auxiliary_modes', 'triu_array', 'node_types', 'info' (best_info or None).
def rank_by_cost(results: list) -> list:
    from reservoir_engineering.topology_search import characterize_topology

    def cost_key(r):
        charac = characterize_topology(r['triu_array'], r['node_types'])
        coops = (r['info'] or {}).get('cooperativities', {})
        max_coop = max(coops.values()) if coops else 0.0
        return (r['num_auxiliary_modes'], charac['num_couplings'],
                charac['num_active_couplings'], max_coop)

    return sorted(results, key=cost_key)


# §5 "full algorithm" top-level driver:
#   optional Approach-A ceiling (gaussian_states.optimise_state) for diagnostics
#   for M in M_min..M_max: build a CovarianceOptimizer with M auxiliary
#       modes, run perform_breadth_first_search, canonicalise the results by
#       residual gauge symmetry (canonicalise_by_gauge)
#   return results ranked by cost (rank_by_cost)
#
# approach_a_loss_fn: optional callable(sigma)->jax scalar, the target
# predicate evaluated on a pure N_signal-mode covariance; if given, runs the
# ceiling first purely as a diagnostic (doesn't currently seed Stage 2 —
# wiring the ideal dark-state nullifiers into a warm start is a natural
# follow-up). stop_at_first_M=True mirrors find_minimum_number_auxiliary_modes
# (stop at the first M with any valid scheme); set False to keep searching
# larger M for possibly-cheaper alternatives.
def run_algorithm(
    sigma_target=None,
    target_mode_ids: List[int] = None,
    node_types_signal: List[str] = None,
    M_min: int = 0,
    M_max: int = 5,
    approach_a_loss_fn=None,
    kwargs_optimizer: dict = None,
    stop_at_first_M: bool = True,
    verbosity: bool = True,
) -> dict:
    from reservoir_engineering.topology_search import canonicalise_by_gauge

    kwargs_optimizer = kwargs_optimizer or {}

    ceiling = None
    if approach_a_loss_fn is not None:
        from reservoir_engineering.gaussian_states import optimise_state
        ceiling = optimise_state(len(target_mode_ids), approach_a_loss_fn)
        if verbosity:
            print(f'Approach-A ceiling: loss={ceiling["loss"]:.3e} '
                  f'(ideal N={len(target_mode_ids)} dark-state dissipators found)')

    all_results = []
    for M in range(M_min, M_max + 1):
        if verbosity:
            print(f'=== run_algorithm: searching with {M} auxiliary modes ===')
        node_types = list(node_types_signal) + ['cavity'] * M
        try:
            optimizer = CovarianceOptimizer(
                sigma_target=sigma_target,
                target_mode_ids=target_mode_ids,
                node_types=node_types,
                num_auxiliary_modes=M,
                make_initial_test=True,
                **kwargs_optimizer,
            )
        except Exception as exc:
            if verbosity:
                print(f'  M={M}: infeasible even fully-connected ({exc}); skipping')
            continue

        valid = optimizer.perform_breadth_first_search()
        if len(valid) == 0:
            continue

        canon = canonicalise_by_gauge(list(valid), node_types, fixed_mode_ids=target_mode_ids)
        for t in canon:
            idx = next((k for k, v in enumerate(optimizer.valid_combinations)
                        if np.array_equal(v, t)), None)
            info = optimizer.best_info_list[idx] if idx is not None else None
            all_results.append({
                'num_auxiliary_modes': M,
                'triu_array': t,
                'node_types': node_types,
                'info': info,
                'optimizer': optimizer,
            })

        if stop_at_first_M:
            break

    return {'ceiling': ceiling, 'results': rank_by_cost(all_results)}


# Central class: combines inner-loop parameter optimisation and outer-loop
# topology search. MIRRORS: Architecture_Optimizer.
class CovarianceOptimizer:

    # sigma_target: (2M,2M) target covariance for target_mode_ids (the
    # convenience case of §2.1's target predicate: matching every entry of
    # the target block at once). target_predicate: §2.1's general form —
    # a list of TargetTerm(fn, q, weight, modes) entries, each a scalar
    # quantity Q_i(sigma) matched to a target value q_i (e.g.
    # targets.target_log_negativity, target_purity, target_quadratic_form).
    # At least one of sigma_target/target_predicate must be given; both may
    # be given together (their residuals simply add in the §3 loss).
    # node_types: length N list of 'cavity'|'mechanical' (signal + auxiliary
    # modes). enforced_constraints: Base_Constraint objects added to the
    # loss for every topology (Constraint_stability should always be
    # included). lambda_pure/lambda_reg/normalize_targets: optional §3 loss
    # terms (all off by default). make_initial_test: if True, verify the
    # fully-connected graph can satisfy the target predicate at all (raises
    # if not) before anything else.
    #
    # kappa: fixed decay rate for cavity-type nodes (used as-is for signal
    #   cavities; overwritten per-solve for auxiliary cavities — see
    #   aux_node_ids below). gamma: same, for mechanical-type nodes; may be
    #   exactly 0.0 (the idealized §1.6 limit — see module docstring).
    # kappa0: §1.5 reference rate for the free log-ratios/log-rates.
    # aux_node_ids (computed automatically) = every mode NOT in
    #   target_mode_ids — only these get a free decay rate; signal-mode
    #   decay rates stay at the fixed kappa/gamma given above.
    # squeezable_aux_ids: §2.6(b) domain palette — human-declared subset of
    #   the auxiliary CAVITY modes allowed a squeezed (Bogoliubov) bath
    #   instead of plain vacuum (see module docstring / SQUEEZE_AB_BOUND_DEFAULT).
    #   The optimizer decides whether/how much to squeeze each one; omit a
    #   node here and it stays plain vacuum, no exceptions.
    def __init__(
        self,
        sigma_target=None,
        target_mode_ids: List[int] = None,
        node_types: List[str] = None,
        num_auxiliary_modes: int = 0,
        gradient_method: str = AUTODIFF_REVERSE,
        kwargs_optimization: dict = {},
        solver_options: dict = {},
        enforced_constraints: list = [],
        make_initial_test: bool = True,
        kappa: float = 1.0,
        gamma: float = 0.01,
        lambda_pure: float = 0.0,
        lambda_reg: float = 0.0,
        normalize_targets: bool = False,
        target_predicate: list = None,
        kappa0: float = KAPPA_0_DEFAULT,
        squeezable_aux_ids: List[int] = None,):


        from reservoir_engineering.covariance_physics import build_diffusion_matrix

        if sigma_target is None and not target_predicate:
            raise ValueError(
                "Must provide sigma_target and/or target_predicate (§2.1: at "
                "least one target-predicate term is required).")

        self.sigma_target         = None if sigma_target is None else np.array(sigma_target)
        self.target_predicate     = list(target_predicate or [])   # §2.1: [(Q_i, q_i, weight, modes), ...]
        self.target_mode_ids      = list(target_mode_ids) # which modes are the squeezed modes ("Mechanical modes")
        self.node_types           = list(node_types)
        self.num_modes            = len(node_types)
        self.num_auxiliary_modes  = num_auxiliary_modes
        self.gradient_method      = gradient_method
        self.kappa0                   = kappa0
        # every mode NOT a signal/target mode is auxiliary — only these get
        # a free decay rate (see class docstring above)
        self.aux_node_ids             = [i for i in range(self.num_modes) if i not in self.target_mode_ids]
        self.enforced_constraints = list(enforced_constraints)

        # §2.6(b) real-coupling palette: beamsplitter edges are complex
        # (free phase) by default — this set records which specific edges
        # the user has restricted back to real-only via
        # Constraint_real_coupling, so no phase variable gets created for
        # them (structural, not a soft penalty — see that class's docstring).
        from reservoir_engineering.constraints import Constraint_real_coupling
        self.real_coupling_edges = {tuple(c.idxs) + (c.edge_type[:3],)
                                     for c in self.enforced_constraints
                                     if isinstance(c, Constraint_real_coupling)}

        # §2.6(b) squeezed-bath palette (see class docstring): must be a
        # subset of aux_node_ids, and cavity type (squeezed-bath formula is
        # only defined for the vacuum-based cavity row, not the thermal
        # loss/gain rows).
        self.squeezable_aux_ids = list(squeezable_aux_ids or [])
        for nid in self.squeezable_aux_ids:
            if nid not in self.aux_node_ids:
                raise ValueError(f"squeezable_aux_ids: node {nid} is not an auxiliary mode "
                                  f"(aux_node_ids={self.aux_node_ids})")
            if self.node_types[nid] != 'cavity':
                raise ValueError(f"squeezable_aux_ids: node {nid} is type "
                                  f"'{self.node_types[nid]}', must be 'cavity'")

        # §3 optional loss terms (off by default — see class docstring)
        self.lambda_pure       = lambda_pure
        self.lambda_reg        = lambda_reg
        self.normalize_targets = normalize_targets

        self.kwargs_optimization = dict(num_tests=10, verbosity=0,
                                    max_violation_success=1e-8,
                                    interrupt_if_successful=True,
                                    optimize_detunings=False)
        self.kwargs_optimization.update(kwargs_optimization)

        self.solver_options = dict(maxiter=2000, ftol=0, gtol=1e-12)
        self.solver_options.update(solver_options)

        self.nodes = []
        for i, t in enumerate(node_types):
            if t == 'cavity':
                self.nodes.append({'id': i, 'type': 'cavity',
                                'kappa': kappa, 'delta': 0.0})
            else:
                self.nodes.append({'id': i, 'type': 'mechanical',
                                'gamma': gamma, 'n_th': 0.0, 'delta': 0.0})

        self.D = build_diffusion_matrix(self.nodes)

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

        self.__setup_all_constraints__()

        # Fully-connected graph must satisfy the target predicate, or no subgraph can either.
        if make_initial_test:
            from reservoir_engineering.topology_search import TopologyGraph
            from reservoir_engineering.topology_search import translate_triu_to_conditions

            full_triu = TopologyGraph.fully_connected(node_types).triu_array
            conditions_full = translate_triu_to_conditions(full_triu, node_types)
            success, _, _ = self.repeated_optimization(
                num_tests=self.kwargs_optimization['num_tests'],
                conditions=conditions_full,
                max_violation_success=self.kwargs_optimization['max_violation_success'],
                optimize_detunings=self.kwargs_optimization['optimize_detunings'],
            )
            if not success:
                raise Exception(
                    "Fully-connected graph failed to satisfy the target predicate. "
                    "Check that the target is physically reachable with these node types.")


    # All possible architectural constraints for the N-mode system (every
    # edge absent, every off-diagonal edge BS) — used by check_all_constraints
    # to discover which are "accidentally" satisfied in a dense-graph solution.
    def __setup_all_constraints__(self):

        from reservoir_engineering.constraints import (
            Constraint_coupling_absent, Constraint_coupling_beamsplitter)

        self.all_possible_constraints = []

        for i in range(self.num_modes):
            for j in range(i, self.num_modes):
                self.all_possible_constraints.append(Constraint_coupling_absent(i, j))
                if i != j:
                    self.all_possible_constraints.append(Constraint_coupling_beamsplitter(i, j))

    def __initialize_conditions_func__(self):
        pass

    # §2.1/§3 target-predicate residual: Σ_i w_i (Q_i(σ)/q_i - 1)² over
    # self.target_predicate, each term evaluated on the sub-block given by
    # its own `modes` (full sigma if modes is None). Combine with the
    # optional sigma_target Frobenius term (below) to get the full §3
    # equality-target contribution to L. sigma is the FULL system
    # covariance (not sigma_sub) — each term picks its own modes.
    #
    # q_i == 0 (e.g. targeting a zero correlation) can't be relative-error
    # normalised — falls back to plain squared error in that case.
    def _predicate_residual(self, sigma):
        from reservoir_engineering.covariance_physics import get_mode_covariance
        residual = 0.0
        for term in self.target_predicate:
            sub = sigma if term.modes is None else get_mode_covariance(sigma, term.modes)
            val = term.fn(sub)
            if abs(term.q) > 1e-12:
                residual = residual + term.weight * (val / term.q - 1.) ** 2
            else:
                residual = residual + term.weight * val ** 2
        return residual

    # Full §3 equality-target term: optional sigma_target Frobenius match
    # (on sigma_sub = target_mode_ids block) plus the target_predicate sum
    # (each term on its own modes, from the full sigma). target_scale
    # normalises the Frobenius term when self.normalize_targets is set.
    def _target_residual(self, sigma, sigma_sub, sigma_target_jnp, target_scale=None):
        residual = 0.0
        if sigma_target_jnp is not None:
            matrix_term = jnp.sum((sigma_sub - sigma_target_jnp) ** 2) / 2.
            if target_scale is not None and target_scale > 0:
                matrix_term = matrix_term / target_scale
            residual = residual + matrix_term
        residual = residual + self._predicate_residual(sigma)
        return residual

    # Stage 1: fast discrete filter, no gradient. Build A at unit
    # cooperativity (g_ij = sqrt(decay_i*decay_j/4) — a fixed reference
    # coupling used ONLY for this cheap pre-filter, unrelated to how Stage 2
    # parametrises the real search) and check Hurwitz. False does NOT go
    # into invalid_combinations — a supergraph (e.g. BS+TMS/Kronwald) can be
    # stable even when this topology (e.g. TMS-only) isn't; this is a speed
    # filter, not a structural impossibility proof.
    def check_stability_unit_cooperativity(self, triu_array) -> bool:
        from reservoir_engineering.topology_search import TopologyGraph
        from reservoir_engineering.covariance_physics import build_drift_matrix, check_stability

        default_kappa = next((n['kappa'] for n in self.nodes if n['type'] == 'cavity'), 1.0)
        default_gamma = next((n['gamma'] for n in self.nodes if n['type'] == 'mechanical'), 0.01)

        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts(
            default_kappa=default_kappa, default_gamma=default_gamma,)

        unit_gs = []
        for edge in edges:
            ni, nj = nodes[edge["i"]], nodes[edge["j"]]
            di = ni.get('kappa', ni.get('gamma'))
            dj = nj.get('kappa', nj.get('gamma'))
            unit_gs.append(float(np.sqrt(di * dj / 4.)))

        A = build_drift_matrix(nodes, edges, jnp.array(unit_gs))
        return check_stability(A)

    # Indices into all_possible_edges that are FREE (not fixed to 0 by a
    # Constraint_coupling_absent in conditions).
    def give_free_variable_idxs(self, conditions: list) -> list:
        from reservoir_engineering.constraints import Constraint_coupling_absent

        absent = {tuple(c.idxs) for c in conditions
                  if isinstance(c, Constraint_coupling_absent)}

        return [k for k, (i, j) in enumerate(self.all_possible_edges) if (min(i,j), max(i,j)) not in absent]

    # Indices into `edges` (this topology's edge list) that get a free
    # phase variable: beamsplitter edges NOT restricted to real-only via
    # Constraint_real_coupling (self.real_coupling_edges — see __init__).
    # Non-beamsplitter edges never get a phase (build_hamiltonian_matrix's
    # phase generalisation only covers that edge type so far).
    def _phase_edge_idxs(self, edges) -> list:
        return [k for k, e in enumerate(edges)
                if e['type'] == 'beamsplitter'
                and (min(e['i'], e['j']), max(e['i'], e['j']), e['type'][:3]) not in self.real_coupling_edges]

    # Build (u:E, v:n_aux, sq:2*n_sq, theta:P) coupling variables. theta is
    # length E (padded with zeros at non-phase-free indices) so it plugs
    # directly into build_drift_diffusion_from_GC_tilde's coherent_phases.
    def _unpack_coupling_vars(self, x, E, n_aux, n_sq, phase_idxs):
        u = x[:E]
        v = x[E:E + n_aux]
        sq = x[E + n_aux:E + n_aux + 2 * n_sq].reshape(n_sq, 2)
        P = len(phase_idxs)
        theta_free = x[E + n_aux + 2 * n_sq:E + n_aux + 2 * n_sq + P]
        theta = jnp.zeros(E).at[jnp.array(phase_idxs, dtype=int)].set(theta_free) if P > 0 else jnp.zeros(E)
        return u, v, sq, theta

    # Build the (loss, grad) pair for a fixed topology's conditions: free
    # vars x = [u_0,...,u_{E-1}, v_0,...,v_{A-1}] — ALL E coherent edges (no
    # reference-edge pinning: kappa0 already fixes the overall-scale gauge,
    # see build_drift_diffusion_from_GC_tilde's docstring) plus one log-rate
    # per auxiliary node (self.aux_node_ids), plus the §2.6(c) soft
    # constraints. Signal-mode decay rates come from self.nodes as given
    # (may be exactly 0). Used by repeated_optimization to avoid rebuilding
    # the JIT-compiled loss/grad on every restart.
    def give_conditions_func_with_conditions(self, conditions: list):
        from reservoir_engineering.topology_search import translate_conditions_to_triu, TopologyGraph
        from reservoir_engineering.covariance_physics import (
            build_drift_diffusion_from_GC_tilde, solve_lyapunov_kronecker, get_mode_covariance)

        triu_array = translate_conditions_to_triu(conditions, self.num_modes, self.node_types)
        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts()
        for i in range(self.num_modes):
            if i not in self.aux_node_ids:
                nodes[i] = dict(self.nodes[i])   # signal modes keep their real (possibly gamma=0) rates

        E = len(edges)
        n_aux = len(self.aux_node_ids)
        n_sq = len(self.squeezable_aux_ids)
        phase_idxs = self._phase_edge_idxs(edges)
        sigma_target_jnp = None if self.sigma_target is None else jnp.array(self.sigma_target)
        target_mode_ids = self.target_mode_ids
        aux_node_ids = self.aux_node_ids
        squeezable_aux_ids = self.squeezable_aux_ids
        kappa0 = self.kappa0

        from reservoir_engineering.constraints import Constraint_coupling_symmetric
        sym_constraints   = [c for c in self.enforced_constraints
                             if isinstance(c, Constraint_coupling_symmetric)]
        other_constraints = [c for c in self.enforced_constraints
                             if not isinstance(c, Constraint_coupling_symmetric)]

        def loss_fn(x):
            u, v, s, theta = self._unpack_coupling_vars(x, E, n_aux, n_sq, phase_idxs)
            A, D = build_drift_diffusion_from_GC_tilde(
                nodes, edges, u, v, aux_node_ids, kappa0, squeezable_aux_ids, s, theta)

            sigma = solve_lyapunov_kronecker(A, D)
            sigma_sub = get_mode_covariance(sigma, target_mode_ids)
            loss = self._target_residual(sigma, sigma_sub, sigma_target_jnp)

            for c in other_constraints:
                loss = loss + c(A, sigma)

            for c in sym_constraints:
                loss = loss + c(u, edges)

            return loss

        loss_jit = jax.jit(loss_fn)
        grad_jit = jax.jit(jax.grad(loss_fn))
        return loss_jit, grad_jit, None

    # Sample the initial parameter vector: E+A log-ratios/log-rates ~
    # Uniform(INIT_LOG_RATIO_RANGE_DEFAULT), plus N detunings drawn from
    # Uniform(-init_detuning_scale, +init_detuning_scale) if optimize_detunings.
    # The detunings are deliberately NOT started at resonance — Delta=0 can be a
    # singular point of the Lyapunov solve (undamped dark mode), see
    # INIT_DETUNING_SCALE_DEFAULT.
    def create_initial_guess(
        self,
        conditions: list = [],
        betas=None,
        optimize_detunings: bool = False,
    ):
        free_idxs = self.give_free_variable_idxs(conditions)
        E = len(free_idxs)

        lo, hi = INIT_LOG_RATIO_RANGE_DEFAULT
        u_init = np.random.uniform(lo, hi, E).astype(float)

        if optimize_detunings:
            d_scale = self.kwargs_optimization.get('init_detuning_scale',
                                                    INIT_DETUNING_SCALE_DEFAULT)
            d_init = np.random.uniform(-d_scale, d_scale, self.num_modes)
            x0 = np.concatenate([u_init, d_init])
        else:
            x0 = u_init

        return x0, free_idxs

    # Log parametrisation means every variable is unconstrained (u_k, v_m,
    # Delta_i in (-inf,inf)); no bounds needed here (unlike AutoScatter's
    # gabs >= 0).
    def setup_bounds(self, conditions: list):
        return None

    # Pad a free-edges-only coupling array with zeros for absent edges.
    def complete_variable_arrays_with_zeros(self, partial_cs, conditions: list) -> np.ndarray:
        free_idxs = self.give_free_variable_idxs(conditions)
        full = np.zeros(len(self.all_possible_edges))
        for k, idx in enumerate(free_idxs):
            full[idx] = partial_cs[k]
        return full

    # Stage 2, the core method: one optimisation run (single random start)
    # for a fixed topology. Free variables x = [u_0,...,u_{E-1},
    # v_0,...,v_{A-1}] (+ Delta_0,...,Delta_{N-1} if optimize_detunings) —
    # see build_drift_diffusion_from_GC_tilde and the class docstring. Loss
    # = §3's L: target term (_target_residual) + soft constraints
    # (enforced_constraints, incl. stability) + optional purity/regulariser
    # terms (self.lambda_pure/self.lambda_reg). success = final loss <
    # max_violation_success. Returns (success, info_out) with
    # G_tilde/C_tilde_aux, detunings, physical coupling strengths, a
    # post-hoc 'cooperativities' dict, achieved sigma, loss history, etc. —
    # see the dict below for exact keys.
    def optimize_given_conditions(
        self,
        conditions: list = None,
        triu_array=None,
        optimize_detunings: bool = False,
        verbosity: bool = False,
        max_violation_success: float = 1e-8,
        calc_conditions_and_gradients=None,
        method: str = 'L-BFGS-B',
        **kwargs_solver,
    ):
        from reservoir_engineering.topology_search import (
            TopologyGraph, translate_triu_to_conditions, translate_conditions_to_triu)
        from reservoir_engineering.covariance_physics import (
            build_drift_diffusion_from_GC_tilde, solve_lyapunov_kronecker, get_mode_covariance,
            purity_violation, check_stability)

        if triu_array is not None and conditions is None:
            conditions = translate_triu_to_conditions(triu_array, self.node_types)
        elif conditions is not None and triu_array is None:
            triu_array = translate_conditions_to_triu(conditions, self.num_modes, self.node_types)
        elif conditions is None and triu_array is None:
            raise ValueError("Must provide conditions or triu_array")

        nodes, edges = TopologyGraph(self.node_types, triu_array).to_nodes_edges_dicts()
        for i in range(self.num_modes):
            if i not in self.aux_node_ids:
                nodes[i] = dict(self.nodes[i])   # signal modes keep their real (possibly gamma=0) rates
        E = len(edges)
        N = self.num_modes
        n_aux = len(self.aux_node_ids)
        n_sq = len(self.squeezable_aux_ids)
        aux_node_ids = self.aux_node_ids
        squeezable_aux_ids = self.squeezable_aux_ids
        kappa0 = self.kappa0
        phase_idxs = self._phase_edge_idxs(edges)
        n_phase = len(phase_idxs)

        sigma_target_jnp = None if self.sigma_target is None else jnp.array(self.sigma_target)
        target_mode_ids = self.target_mode_ids
        J2 = jnp.array([[0., 1.], [-1., 0.]])

        # Constraint_coupling_symmetric/_cooperativity_cap are dispatched on
        # (log_ratios, edges), not (A, sigma) like every other constraint —
        # filter them out here the same way give_conditions_func_with_conditions
        # does, or they crash when called with the wrong signature.
        from reservoir_engineering.constraints import Constraint_coupling_symmetric, Constraint_cooperativity_cap
        u_constraints  = [c for c in self.enforced_constraints
                          if isinstance(c, (Constraint_coupling_symmetric, Constraint_cooperativity_cap))]
        other_constraints = [c for c in self.enforced_constraints if c not in u_constraints]

        target_scale = (float(np.sum(np.asarray(self.sigma_target) ** 2))
                         if self.normalize_targets and self.sigma_target is not None else None)

        def loss_fn(x):
            u, v, sq, theta = self._unpack_coupling_vars(x, E, n_aux, n_sq, phase_idxs)
            A, D = build_drift_diffusion_from_GC_tilde(
                nodes, edges, u, v, aux_node_ids, kappa0, squeezable_aux_ids, sq, theta)

            if optimize_detunings:
                detunings_x = x[E + n_aux + 2 * n_sq + n_phase:]
                for i in range(N):
                    s = slice(2 * i, 2 * i + 2)
                    A = A.at[s, s].add(detunings_x[i] * J2)

            sigma = solve_lyapunov_kronecker(A, D)
            sigma_sub = get_mode_covariance(sigma, target_mode_ids)

            loss = self._target_residual(sigma, sigma_sub, sigma_target_jnp, target_scale)

            for c in other_constraints:
                loss = loss + c(A, sigma)

            for c in u_constraints:
                loss = loss + c(u, edges)

            if self.lambda_pure > 0:
                loss = loss + self.lambda_pure * purity_violation(sigma_sub)

            if self.lambda_reg > 0:
                loss = loss + self.lambda_reg * jnp.sum(jnp.concatenate([u, v]) ** 2)

            return loss

        if calc_conditions_and_gradients is not None:
            loss_jit, grad_jit, _ = calc_conditions_and_gradients
        else:
            loss_jit = jax.jit(loss_fn)
            grad_jit = jax.jit(jax.grad(loss_fn))

        bound       = self.kwargs_optimization.get('direct_log_bound', DIRECT_LOG_BOUND_DEFAULT)
        sq_bound    = self.kwargs_optimization.get('squeeze_ab_bound', SQUEEZE_AB_BOUND_DEFAULT)
        phase_bound = self.kwargs_optimization.get('phase_bound', PHASE_BOUND_DEFAULT)
        d_bound     = self.kwargs_optimization.get('detuning_bound',  DETUNING_BOUND_DEFAULT)

        lo, hi = INIT_LOG_RATIO_RANGE_DEFAULT
        n_coupling_vars = E + n_aux + 2 * n_sq + n_phase
        # per-variable final bound: (u,v) get `bound`, squeeze (a_,b_) get
        # `sq_bound`, phase (theta) gets `phase_bound` (one period — see
        # PHASE_BOUND_DEFAULT).
        final_bounds = np.concatenate([np.full(E + n_aux, bound), np.full(2 * n_sq, sq_bound),
                                        np.full(n_phase, phase_bound)])

        # Decay-rate-aware init for the coherent edges (using the SAME
        # unit-cooperativity formula as Stage 1's fast filter,
        # g_unit=sqrt(decay_i*decay_j/4)): a flat Uniform(lo,hi) center for
        # every edge is fine when all edges touch similar decay rates (e.g.
        # Kronwald's single cavity-mech pair), but with several MODE PAIRS
        # of very different natural scale in one topology (e.g. cavity-mech
        # AND mech-mech edges, kappa/gamma ~ 100x apart), a flat center
        # leaves some edges' initial physical g_k orders of magnitude away
        # from what they need, and L-BFGS-B can get stuck. Centering each
        # edge's init on its own unit-cooperativity scale removes that
        # cross-edge conditioning gap; the uniform range still supplies the
        # randomness multi-restart relies on. Auxiliary log-rates (v) and
        # squeeze params (a_,b_) have no natural per-edge scale of their own
        # — left centered at 0 (aux decay = kappa0; squeeze = plain vacuum).
        # Phase (theta) is centered at 0 too but jittered over its FULL
        # period (Uniform(-pi,pi), not the log-ratio jitter range) — it's
        # periodic, not a magnitude, so a wide initial spread matters more
        # than a small perturbation around a single guess.
        u_center = []
        for edge in edges:
            ni, nj = nodes[edge['i']], nodes[edge['j']]
            di = ni.get('kappa', ni.get('gamma'))
            dj = nj.get('kappa', nj.get('gamma'))
            g_unit = np.sqrt(max(di * dj, 0.) / 4.)
            u_center.append(float(np.log(g_unit / kappa0)) if g_unit > 0 else 0.0)
        centers = np.array(u_center + [0.0] * n_aux + [0.0] * (2 * n_sq) + [0.0] * n_phase)
        jitter = np.concatenate([
            np.random.uniform(lo, hi, E + n_aux + 2 * n_sq),
            np.random.uniform(-phase_bound, phase_bound, n_phase),
        ])
        x0_coupling = np.clip(centers + jitter, -final_bounds, final_bounds).astype(float)

        # Progressive coupling-scale warm-up (generalises the old
        # cooperativity mode's Stage-2 multi-scale lambda continuation to
        # (u,v) together): a few short solves at increasing bounds, each
        # warm-starting the next from the previous solution. A single
        # fixed-bound solve from a smart-but-static init can still get
        # stuck in a bad local minimum on topologies with several very
        # different edge scales in one graph (e.g. cavity-mech AND
        # mech-mech edges together — heterogeneous decay rates, not fixed
        # by the per-edge init alone); ramping the reachable scale up
        # gradually escapes that far more reliably than jumping straight to
        # the full bound. Skipped when there's nothing to warm up.
        # Detuning init: jittered, never all-zero — see INIT_DETUNING_SCALE_DEFAULT
        # (Delta=0 is a singular point of the Lyapunov solve, not merely a poor
        # guess). The warm-up below holds the detunings FIXED at this draw while
        # it ramps the coupling scale, so it must use the same nonzero values —
        # pinning them back to 0 there would put the warm-up itself inside the
        # singular region.
        if optimize_detunings:
            d_scale = self.kwargs_optimization.get('init_detuning_scale',
                                                    INIT_DETUNING_SCALE_DEFAULT)
            d_init = np.random.uniform(-d_scale, d_scale, N)
            d_init = np.clip(d_init, -d_bound, d_bound)
        else:
            d_init = np.zeros(N)

        if n_coupling_vars > 0:
            zeros_tail = d_init if optimize_detunings else np.zeros(0)

            def pad(yy):
                return np.concatenate([yy, zeros_tail]) if optimize_detunings else yy

            y = x0_coupling
            for stage_scalar in (2.3, 4.6, 6.9):
                stage_bounds = np.minimum(stage_scalar, final_bounds)
                if np.all(stage_bounds >= final_bounds):
                    continue
                y = np.clip(y, -stage_bounds, stage_bounds)
                wres = sciopt.minimize(
                    fun=lambda x: float(loss_jit(jnp.array(pad(x), dtype=float))),
                    jac=lambda x: np.array(grad_jit(jnp.array(pad(x), dtype=float)), dtype=float)[:n_coupling_vars],
                    x0=y, method=method,
                    bounds=list(zip(-stage_bounds, stage_bounds)),
                    options={'maxiter': 60},
                )
                y = wres.x
            x0_coupling = y

        x0 = np.concatenate([x0_coupling, d_init]) if optimize_detunings else x0_coupling

        loss_history = []
        def callback(x):
            loss_history.append(float(loss_jit(jnp.array(x, dtype=float))))

        solver_opts = dict(self.solver_options)
        solver_opts.update(kwargs_solver)

        if optimize_detunings:
            bounds = list(zip(-final_bounds, final_bounds)) + ([(-d_bound, d_bound)] * N)
        else:
            bounds = list(zip(-final_bounds, final_bounds))

        n_vars = len(x0)
        if n_vars == 0:
            x_sol = x0
        else:
            result = sciopt.minimize(
                fun=lambda x: float(loss_jit(jnp.array(x, dtype=float))),
                jac=lambda x: np.array(grad_jit(jnp.array(x, dtype=float)), dtype=float),
                x0=x0,
                method=method,
                bounds=bounds,
                options=solver_opts,
                callback=callback,
            )
            x_sol = result.x

        u_sol, v_sol, sq_sol, theta_sol = self._unpack_coupling_vars(x_sol, E, n_aux, n_sq, phase_idxs)
        u_sol, v_sol, sq_sol, theta_sol = np.array(u_sol), np.array(v_sol), np.array(sq_sol), np.array(theta_sol)
        detunings_sol = x_sol[E + n_aux + 2 * n_sq + n_phase:] if optimize_detunings else np.zeros(N)
        final_loss = float(loss_jit(jnp.array(x_sol, dtype=float)))

        A_sol, D_sol = build_drift_diffusion_from_GC_tilde(
            nodes, edges, jnp.array(u_sol), jnp.array(v_sol), aux_node_ids, kappa0,
            squeezable_aux_ids, jnp.array(sq_sol), jnp.array(theta_sol))
        for i in range(N):
            s = slice(2 * i, 2 * i + 2)
            A_sol = A_sol.at[s, s].add(float(detunings_sol[i]) * J2)
        sigma_full_sol = solve_lyapunov_kronecker(A_sol, D_sol)
        sigma_achieved = get_mode_covariance(sigma_full_sol, target_mode_ids)

        # Constraint_stability is a SOFT penalty (§2.6(c)) — it shapes the
        # landscape but a low loss doesn't guarantee it actually WON at the
        # optimum found. An unstable A makes solve_lyapunov_kronecker solve
        # a near-singular/wrong-sign system, silently producing an
        # unphysical sigma (e.g. negative variances) that can coincidentally
        # score a low target residual. Cross-check Hurwitz explicitly so
        # `success` never reports a machine that doesn't actually have the
        # claimed steady state.
        success = (final_loss < max_violation_success) and check_stability(A_sol)

        G_tilde = {}
        coherent_phases = {}
        coupling_strengths_sol = []
        for k, edge in enumerate(edges):
            g_k = float(kappa0 * np.exp(u_sol[k]))   # magnitude only
            coupling_strengths_sol.append(g_k)
            label = f"({edge['i']},{edge['j']},{edge['type'][:3]})"
            G_tilde[f'G~_{label}'] = float(np.exp(u_sol[k]))
            if k in phase_idxs:
                coherent_phases[f'theta_{label}'] = float(theta_sol[k])

        C_tilde_aux, aux_decay_sol = {}, {}
        for slot, node_id in enumerate(aux_node_ids):
            c_val = float(kappa0 * np.exp(v_sol[slot]))
            aux_decay_sol[node_id] = c_val
            C_tilde_aux[f'C~_aux_{node_id}'] = float(np.exp(v_sol[slot]))

        # post-hoc (r,theta) from the achieved (a_,b_) — see
        # covariance_physics.build_jump_matrix / SQUEEZE_AB_BOUND_DEFAULT.
        # M=a_+i*b_=sinh(r)e^{i theta}; r=0 (a_=b_=0) means the search found
        # plain vacuum was better/sufficient for that node, even though it
        # was offered a squeezed bath.
        bath_squeezing = {}
        for slot, node_id in enumerate(squeezable_aux_ids):
            a_val, b_val = float(sq_sol[slot, 0]), float(sq_sol[slot, 1])
            bath_squeezing[node_id] = {
                'a': a_val, 'b': b_val,
                'r': float(np.arcsinh(np.hypot(a_val, b_val))),
                'theta': float(np.arctan2(b_val, a_val)),
            }

        # post-hoc cooperativities (reporting only — not the search variable
        # here); inf where a signal mode's fixed decay is exactly 0.
        cooperativities = {}
        for k, edge in enumerate(edges):
            ni_id, nj_id = edge['i'], edge['j']
            di = aux_decay_sol.get(ni_id, nodes[ni_id].get('kappa', nodes[ni_id].get('gamma')))
            dj = aux_decay_sol.get(nj_id, nodes[nj_id].get('kappa', nodes[nj_id].get('gamma')))
            g_k = coupling_strengths_sol[k]
            label = f"({edge['i']},{edge['j']},{edge['type'][:3]})"
            cooperativities[label] = float(4 * g_k ** 2 / (di * dj)) if (di > 0 and dj > 0) else float('inf')

        opt_nit = result.nit if n_vars > 0 else 0
        opt_msg = result.message if n_vars > 0 else 'no free variables — direct evaluation'

        if verbosity:
            print(f'  loss={final_loss:.3e}  success={success}  nit={opt_nit}')

        info_out = {
            'initial_guess'       : x0,
            'solution'            : x_sol,
            'coherent_log_ratios' : u_sol,
            'aux_log_rates'       : v_sol,
            'G_tilde'             : G_tilde,
            'C_tilde_aux'         : C_tilde_aux,
            'bath_squeezing'      : bath_squeezing,
            'coherent_phases'     : coherent_phases,
            'detunings'           : {f'Delta_{i}': float(detunings_sol[i]) for i in range(N)},
            'kappa0'              : kappa0,
            'coupling_strengths'  : np.array(coupling_strengths_sol),
            'aux_decay_rates'     : aux_decay_sol,
            'cooperativities'     : cooperativities,
            'physical_formula'    : ('g_k = kappa0*exp(u_k)*exp(i*theta_k) (theta free for beamsplitter '
                                      'edges unless Constraint_real_coupling pins it to 0);  '
                                      'aux_decay_m = kappa0*exp(v_m)'),
            'final_cost'          : final_loss,
            'success'             : success,
            'optimizer_message'   : opt_msg,
            'A'                   : np.array(A_sol),
            'sigma_full'          : np.array(sigma_full_sol),
            'sigma_achieved'      : np.array(sigma_achieved),
            'sigma_target'        : self.sigma_target,
            'nit'                 : opt_nit,
            'loss_history'        : loss_history,
        }
        return success, info_out

    # Run optimize_given_conditions num_tests times from different random
    # starts; stop early on success if interrupt_if_successful. Returns
    # (any_success, list_of_infos, indices_of_successes).
    def repeated_optimization(
        self,
        num_tests: int,
        conditions: list = None,
        triu_array=None,
        verbosity: bool = False,
        max_violation_success: float = 1e-8,
        interrupt_if_successful: bool = True,
        **kwargs_solver,
    ):
        # Build the JIT-compiled loss/grad once, reuse across restarts. NOT
        # valid when optimize_detunings=True: give_conditions_func_with_conditions's
        # loss_fn never applies the per-mode detuning rotation to A (that
        # logic only exists in optimize_given_conditions's own inline
        # loss_fn), so reusing it here would silently make the detuning
        # variables dead weight (zero gradient, no effect) instead of
        # actually searching over them. Fall back to rebuilding loss/grad
        # fresh each restart in that case — slower (no JIT reuse) but correct.
        from reservoir_engineering.topology_search import translate_triu_to_conditions
        optimize_detunings = kwargs_solver.get('optimize_detunings', False)
        if optimize_detunings:
            calc_cag = None
        elif conditions is not None:
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

    # After solving the fully-connected graph, read off which constraints
    # are "accidentally" satisfied (block ~ 0/pure-BS/pure-TMS) to recover
    # the minimal topology the solution actually uses.
    def check_all_constraints(self, A, sigma, threshold=None) -> np.ndarray:
        from reservoir_engineering.constraints import (
            Constraint_coupling_absent,
            Constraint_coupling_beamsplitter,
            Constraint_coupling_two_mode_squeezing)
        from reservoir_engineering.topology_search import (
            NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING, PARAMETRIC,
            BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)

        if threshold is None:
            # Must sit ABOVE the smallest representable coupling. With
            # g_k = kappa0*exp(u_k) and |u_k| <= direct_log_bound, a coupling
            # the optimiser has fully suppressed cannot reach 0 — it parks at
            # the floor kappa0*exp(-bound) (=1e-4 for the defaults). The old
            # hard-coded 1e-6 sat 100x BELOW that floor, so a maximally
            # suppressed edge always read as "present" and graph reduction
            # could never prune it, systematically over-reporting edge count.
            # Take a margin above the floor instead.
            bound = self.kwargs_optimization.get('direct_log_bound',
                                                  DIRECT_LOG_BOUND_DEFAULT)
            threshold = max(1e-6, 10.0 * self.kappa0 * float(np.exp(-bound)))

        A_jnp = jnp.array(A)
        sigma_jnp = jnp.array(sigma)
        N = self.num_modes
        rows, cols = np.triu_indices(N)
        triu = np.zeros(len(rows), dtype=int)

        for k, (i, j) in enumerate(zip(rows, cols)):
            if i == j:
                # Parametric drive splits x/p decay rates: without it
                # A[2i,2i]==A[2i+1,2i+1] (both -decay/2); with it they differ.
                diag_diff = abs(float(A_jnp[2*i, 2*i]) - float(A_jnp[2*i+1, 2*i+1]))
                triu[k] = PARAMETRIC if diag_diff > threshold else NO_COUPLING
            else:
                bs_c  = Constraint_coupling_beamsplitter(i, j)
                tms_c = Constraint_coupling_two_mode_squeezing(i, j)
                absent_c = Constraint_coupling_absent(i, j)
                absent_val   = float(absent_c(A_jnp, sigma_jnp))
                bs_residual  = float(bs_c(A_jnp, sigma_jnp))
                tms_residual = float(tms_c(A_jnp, sigma_jnp))
                if abs(absent_val) < threshold:
                    triu[k] = NO_COUPLING
                elif abs(bs_residual) < threshold:
                    triu[k] = BEAMSPLITTER
                elif abs(tms_residual) < threshold:
                    triu[k] = TWO_MODE_SQUEEZING
                else:
                    triu[k] = BEAMSPLITTER_AND_TWO_MODE_SQUEEZING

        return triu

    # Enumerate every possible topology for the N-mode system (itertools
    # product over each slot's allowed values: {0} if forced absent, {0,3}
    # on the diagonal, {0,1,2,4} off-diagonal). Populates
    # list_of_triu_arrays/complexity_levels/unique_complexity_levels — the
    # search space the BFS walks low->high complexity.
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

    # Filter to topologies at complexity_level worth testing: not a
    # supergraph of a known invalid (would also fail) and not already
    # covered by a known valid subgraph (redundant). The key BFS speedup.
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
            if self.invalid_combinations:
                if check_if_subgraph_triu([triu], self.invalid_combinations):
                    continue
            if not skip_check_for_valid_subgraphs and self.valid_combinations:
                if check_if_subgraph_triu([triu], self.valid_combinations):
                    continue
            potential.append(triu)
        return potential

    # Run Stage 1->2 on every candidate at complexity_level. Stage 1
    # failures are NOT added to invalid_combinations (see class docstring —
    # a supergraph can add stabilising edges); only a Stage-2 success is
    # recorded as valid (optionally reduced to its minimal topology via
    # check_all_constraints).
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

            if not self.check_stability_unit_cooperativity(triu_array):
                num_skipped += 1
                continue

            if newly_added and check_if_subgraph_triu([triu_array], newly_added):
                num_skipped += 1
                continue

            conditions = translate_triu_to_conditions(triu_array, self.node_types)
            success, infos, _ = self.repeated_optimization(
                num_tests=self.kwargs_optimization['num_tests'],
                conditions=conditions,
                max_violation_success=self.kwargs_optimization['max_violation_success'],
                interrupt_if_successful=self.kwargs_optimization['interrupt_if_successful'],
                optimize_detunings=self.kwargs_optimization['optimize_detunings'],
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
            # Stage 2 failures are NOT added to invalid_combinations either:
            # a simpler failing topology doesn't imply its supersets fail
            # (BS-only can't squeeze, but BS+TMS/Kronwald can).

        self.tested_complexities.append(complexity_level)
        self.num_tested_graphs.append(num_tested)
        self.num_tested_invalid_graphs.append(num_skipped)

    # Drop redundant valid_combinations — any topology that has another
    # valid topology as a subgraph is superseded by the simpler one.
    def cleanup_valid_combinations(self):
        from reservoir_engineering.topology_search import check_if_subgraph_triu

        if not self.valid_combinations:
            return

        seen = set()
        unique = []
        for t in self.valid_combinations:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique.append(t)

        kept = []
        for i, t in enumerate(unique):
            others = unique[:i] + unique[i + 1:]
            if others and check_if_subgraph_triu([t], others):
                continue
            kept.append(t)

        self.valid_combinations = kept

    # The main outer loop: enumerate all topologies, then walk complexity
    # levels low->high running find_valid_combinations + cleanup at each.
    # For Kronwald (1 cavity + 1 mechanical): expect exactly one valid
    # minimal topology, at complexity 3 (BS+TMS) — neither BS alone nor TMS
    # alone should succeed at lower complexity.
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

    # Valid/invalid topology counts per complexity level, for analysis.py's
    # search-space plots.
    def count_valid_invalid_graphs_layers(self):
        complexities = getattr(self, 'unique_complexity_levels', [])
        valid_counts = [
            sum(1 for t in self.valid_combinations if int(np.sum(t)) == c)
            for c in complexities]

        invalid_counts = [
            sum(1 for t in self.invalid_combinations if int(np.sum(t)) == c)
            for c in complexities]

        return complexities, valid_counts, invalid_counts

    # {edge_label: coupling_strength} dict for the free edges in conditions.
    def dict_extract_relevant_information(self, solution_cs, conditions: list) -> dict:
        free_idxs = self.give_free_variable_idxs(conditions)
        result = {}
        for k, idx in enumerate(free_idxs):
            i, j = self.all_possible_edges[idx]
            result[f'g_{i}{j}'] = float(solution_cs[k])
        return result
