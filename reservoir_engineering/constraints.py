"""
constraints.py
==============
Constraint objects for encoding graph topology and physics requirements.
MIRRORS: autoscatter/constraints.py

A "condition" both ENCODES a structural fact about the graph (which edges
exist, what type) AND EVALUATES how much that fact is violated for a given
parameter set, e.g. [Constraint_coupling_absent(0,1), Constraint_coupling_beamsplitter(0,2)]
means "edge (0,1) absent; edge (0,2) is a beamsplitter, not TMS".

CovarianceOptimizer uses these to: (1) decide which coupling strengths are
free vs fixed to 0, (2) add penalty terms to the loss for physics
requirements (stability, entanglement, ...), (3) after optimising the
fully-connected graph, read off which constraints are "accidentally"
satisfied to infer the minimal topology.

Two-level hierarchy (mirrors AutoScatter):
  Coupling_Constraint subclasses — architectural: which edges exist and
    what type (Constraint_coupling_absent, _beamsplitter, ...).
  Base_Constraint subclasses — physics requirements added as a loss
    residual (Constraint_stability, _physical_state, ...).

Edge type constants: BS and TMS are both off-diagonal but have different
block structure in A (I2 vs sigma_z), so unlike AutoScatter's single
COUPLING_WITH_PHASE they need distinct codes; PARAMETRIC is diagonal-only.

Constraint registry (gaussian_autoscatter_algorithm.md §2.6): every
constraint carries a `.slot` tag routing it to one of three places:
  'structural' (a) — prunes the palette before enumeration. Free, exact.
      e.g. Constraint_coupling_absent.
  'domain'     (b) — restricts edge TYPES / parameterisation. Free, exact.
      e.g. Constraint_coupling_beamsplitter/_two_mode_squeezing, Constraint_passivity.
  'soft'       (c) — a weighted penalty term added to the §3 loss.
      e.g. Constraint_stability, _physical_state, _target_squeezing,
      _entanglement, Constraint_cooperativity_cap.
Rule: route to the earliest slot that fits (a before b before c) — (a)/(b)
are free, (c) enlarges the non-convex landscape. `ConstraintRegistry` below
just classifies a flat constraint list into these three slots.
"""

import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Edge type constants (upper-triangle encoding):
#   0 = no coupling
#   1 = beamsplitter (BS): A-block = g*I2 (energy-conserving)
#   2 = two-mode squeezing (TMS): A-block = nu*sigma_z (parametric)
#   3 = parametric / single-mode squeezing (diagonal only)
NO_COUPLING = 0
BEAMSPLITTER = 1
TWO_MODE_SQUEEZING = 2
PARAMETRIC = 3

EDGETYPE_ABSENT = None
EDGETYPE_BEAMSPLITTER = 'beamsplitter'
EDGETYPE_TWO_MODE_SQUEEZING = 'two_mode_squeezing'
EDGETYPE_PARAMETRIC = 'parametric'


# Abstract base: every constraint implements __call__(A, sigma) -> jax
# scalar residual (0 = satisfied). slot: 'structural'|'domain'|'soft',
# read by ConstraintRegistry; must be set on each subclass.
class Base_Constraint:
    slot = None

    def __call__(self, A, sigma):
        raise NotImplementedError


# Base class for architectural (edge-topology) constraints. idxs stored
# canonically as [min(i,j), max(i,j)] so (0,1) == (1,0).
class Coupling_Constraint(Base_Constraint):
    def __init__(self, i, j):
        self.idxs = [min(i, j), max(i, j)]

    def __eq__(self, other):
        return type(self) == type(other) and self.idxs == other.idxs

    def __hash__(self):
        return hash((type(self).__name__, self.idxs[0], self.idxs[1]))


# No edge between modes i,j: residual = ||A[s_i,s_j]||, zero iff block is zero.
class Constraint_coupling_absent(Coupling_Constraint):
    slot = 'structural'   # §2.6(a): prunes this edge slot out of enumeration

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import quadrature_slice
        except ImportError:
            from covariance_physics import quadrature_slice
        si = quadrature_slice(self.idxs[0])
        sj = quadrature_slice(self.idxs[1])
        return jnp.sum(jnp.abs(A[si, sj]))

    def __str__(self):
        return f'No coupling between mode {self.idxs[0]} and mode {self.idxs[1]}'


# If edge (i,j) exists it must be a beamsplitter: block ~ g*I2, so the
# symmetric (TMS-like) part of the block must vanish.
class Constraint_coupling_beamsplitter(Coupling_Constraint):
    slot = 'domain'   # §2.6(b): fixes the TYPE of an active edge

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import quadrature_slice
        except ImportError:
            from covariance_physics import quadrature_slice
        si = quadrature_slice(self.idxs[0])
        sj = quadrature_slice(self.idxs[1])
        block = A[si, sj]
        sym = (block + block.T) / 2        # TMS component — must be zero for pure BS
        return jnp.sum(sym ** 2)


# If edge (i,j) exists it must be TMS: block ~ nu*sigma_z, so the
# antisymmetric (BS-like) part of the block must vanish.
class Constraint_coupling_two_mode_squeezing(Coupling_Constraint):
    slot = 'domain'   # §2.6(b): fixes the TYPE of an active edge

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import quadrature_slice
        except ImportError:
            from covariance_physics import quadrature_slice
        si = quadrature_slice(self.idxs[0])
        sj = quadrature_slice(self.idxs[1])
        block = A[si, sj]
        asym = (block - block.T) / 2       # BS component — must be zero for pure TMS
        return jnp.sum(asym ** 2)

    def __str__(self):
        return f'Edge ({self.idxs[0]},{self.idxs[1]}) is two-mode squeezing'


# A must be Hurwitz (all eigenvalues Re<0) for a steady state to exist.
# No AutoScatter analogue in the simple case — its passive systems are
# automatically stable, but TMS interactions here can destabilise A.
# penalty_strength acts like a Lagrange multiplier pushing away from
# instability; zero when stable.
class Constraint_stability(Base_Constraint):
    slot = 'soft'   # §2.6(c): lambda_stab*stab(A) term — must dominate the loss

    def __init__(self, penalty_strength=50.0):
        self.penalty_strength = penalty_strength

    def __call__(self, A, sigma):
        eigs = jnp.linalg.eigvals(A)
        return self.penalty_strength * jnp.sum(jnp.maximum(0., jnp.real(eigs)))


# sigma_sub must satisfy the uncertainty principle (symplectic eigenvalues
# >= 1/2). Automatically true once the Lyapunov solve succeeds for a stable
# A — this is mostly a diagnostic guard against numerical issues near
# marginal stability.
class Constraint_physical_state(Base_Constraint):
    slot = 'soft'   # §2.6(c): diagnostic penalty term

    def __init__(self, target_mode_ids):
        self.target_mode_ids = target_mode_ids

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import get_mode_covariance
        except ImportError:
            from covariance_physics import get_mode_covariance
        s = get_mode_covariance(sigma, self.target_mode_ids)
        N = s.shape[0] // 2
        Omega = jnp.zeros_like(s)
        for i in range(N):
            Omega = Omega.at[2*i, 2*i+1].set(1.)
            Omega = Omega.at[2*i+1, 2*i].set(-1.)
        eigs = jnp.linalg.eigvals(1j * Omega @ s)
        nu_min = jnp.min(jnp.abs(jnp.real(eigs)))
        return jnp.maximum(0., 0.5 - nu_min)


# mode_id must reach at least squeezing r_min (a "floor" constraint used to
# steer the search toward genuinely squeezed solutions).
class Constraint_target_squeezing(Base_Constraint):
    slot = 'soft'   # §2.6(c): penalty coupled to the value of sigma

    def __init__(self, mode_id, r_min):
        self.mode_id = mode_id
        self.r_min = r_min

    def __call__(self, A, sigma):
        sigma_xx = sigma[2*self.mode_id, 2*self.mode_id]
        r_achieved = -0.5 * jnp.log(2. * sigma_xx)
        return jnp.maximum(0., self.r_min - r_achieved)


# The two modes in mode_ids must be entangled per the Duan criterion
# (Var(x0-x1) + Var(p0+p1) < 1). Zero when entangled, positive when separable.
class Constraint_entanglement(Base_Constraint):
    slot = 'soft'   # §2.6(c): penalty coupled to the value of sigma

    def __init__(self, mode_ids):
        self.mode_ids = mode_ids

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import get_mode_covariance
        except ImportError:
            from covariance_physics import get_mode_covariance
        s = get_mode_covariance(sigma, self.mode_ids)
        duan_sum = (s[0,0] + s[2,2] - 2*s[0,2] +
                    s[1,1] + s[3,3] + 2*s[1,3])
        return jnp.maximum(0., duan_sum - 1.0)


# Infer the edge type between modes i,j from A's block: near-zero -> absent;
# else compare the symmetric (BS) vs antisymmetric (TMS) component. Diagonal
# (i==j): unequal x/p decay rates -> parametric squeezing present.
def return_edge_type(A, i, j, threshold=1e-4):
    try:
        from reservoir_engineering.covariance_physics import quadrature_slice
    except ImportError:
        from covariance_physics import quadrature_slice
    si = quadrature_slice(i)
    sj = quadrature_slice(j)
    if i == j:
        diff = abs(float(A[2*i, 2*i]) - float(A[2*i+1, 2*i+1]))
        return EDGETYPE_PARAMETRIC if diff > threshold else None
    block = np.array(A[si, sj])
    if np.linalg.norm(block, 'fro') < threshold:
        return EDGETYPE_ABSENT
    sym  = (block + block.T) / 2
    asym = (block - block.T) / 2
    if np.linalg.norm(sym) > np.linalg.norm(asym):
        return EDGETYPE_TWO_MODE_SQUEEZING
    return EDGETYPE_BEAMSPLITTER


# Penalises |u_a - u_b|^2 to tie two edges' coupling strengths together.
# edge_type: 'bea'|'two' (first 3 chars). Dispatched specially (called with
# (log_ratios, edges) rather than (A, sigma)) inside
# give_conditions_func_with_conditions.
class Constraint_coupling_symmetric:
    # Implemented as a soft hinge, not a hard tie of the two variables into
    # one (which would be the stricter §2.6(b)-domain approach) — tagged
    # 'soft' to reflect the mechanism actually used.
    slot = 'soft'

    def __init__(self, i1, j1, edge_type1, i2, j2, edge_type2,
                 penalty_strength=10.0, guard_edges=None):
        self.edge_a = (min(i1, j1), max(i1, j1), edge_type1[:3])
        self.edge_b = (min(i2, j2), max(i2, j2), edge_type2[:3])
        self.penalty_strength = penalty_strength
        # guard_edges: (i,j,type_prefix) that must ALL be present for the penalty to fire
        self.guard_edges = [(min(i, j), max(i, j), t[:3]) for i, j, t in (guard_edges or [])]

    def __call__(self, log_ratios, edges):
        present = set()
        idx_a = idx_b = None
        for k, e in enumerate(edges):
            key = (min(e['i'], e['j']), max(e['i'], e['j']), e.get('type', '')[:3])
            present.add(key)
            if key == self.edge_a:
                idx_a = k
            if key == self.edge_b:
                idx_b = k
        if idx_a is None or idx_b is None:
            return 0.0
        if any(g not in present for g in self.guard_edges):
            return 0.0
        return self.penalty_strength * (log_ratios[idx_a] - log_ratios[idx_b]) ** 2


# §2.6(b) domain: restricts one beamsplitter edge's coupling to be REAL
# (phase pinned to 0), i.e. opts INTO the old always-real behaviour for
# that specific edge. General beamsplitter couplings are complex by
# default (see covariance_physics.build_hamiltonian_matrix's phase
# generalisation, and Zippilli & Vitali PRL 126, 020402 (2021) eq.
# S.27-S.29 — the passivity-preserving phase is generically nonzero, so
# forcing real is a restriction the user opts into, not the baseline).
# Detected structurally by CovarianceOptimizer.__init__ (like
# Constraint_coupling_absent) — the phase variable for a real-pinned edge
# is never created, not soft-penalized to zero, so this is free/exact.
class Constraint_real_coupling(Coupling_Constraint):
    slot = 'domain'   # §2.6(b): fixes the coupling's phase, not just its type

    def __init__(self, i, j, edge_type='beamsplitter'):
        super().__init__(i, j)
        self.edge_type = edge_type

    def __call__(self, A, sigma):
        # Structurally enforced (no phase variable exists for this edge) —
        # nothing left to penalise in the loss.
        return 0.0

    def __str__(self):
        return f'Edge ({self.idxs[0]},{self.idxs[1]}) coupling is real-only'


# §2.6 domain example "passivity": bars TMS/squeezing/gain on the given
# modes. A fully passive network fed by vacuum can't create entanglement or
# squeezing, so the target's non-passive resource must live elsewhere
# (typically auxiliary modes). __call__ is a soft-penalty fallback for
# callers that add this directly to enforced_constraints instead of routing
# it through ConstraintRegistry's palette masking.
class Constraint_passivity(Base_Constraint):
    slot = 'domain'

    def __init__(self, mode_ids):
        self.mode_ids = list(mode_ids)

    def __call__(self, A, sigma):
        try:
            from reservoir_engineering.covariance_physics import quadrature_slice
        except ImportError:
            from covariance_physics import quadrature_slice
        penalty = 0.0
        N = A.shape[0] // 2
        for i in self.mode_ids:
            si = quadrature_slice(i)
            for j in range(N):
                if j == i:
                    continue
                sj = quadrature_slice(j)
                block = A[si, sj]
                sym = (block + block.T) / 2   # TMS/squeezing component
                penalty = penalty + jnp.sum(sym ** 2)
        return penalty

    def __str__(self):
        return f'Modes {self.mode_ids} must be passive (beamsplitter-only, no gain)'


# §2.6 soft example "coupling-strength cap": hinge penalty when one edge's
# dimensionless coupling G~_k=exp(u_k) (i.e. g_k/kappa0 — see
# covariance_optimizer's (G-tilde,C-tilde) parametrisation) exceeds
# G_tilde_max. (A hard bound belongs in slot (a) — see
# DIRECT_LOG_BOUND_DEFAULT in covariance_optimizer.py — use this only when a
# soft, per-edge trade-off is preferred over one global cap.) Caps the
# dimensionless ratio directly, not a literal cooperativity C=4g^2/(kappa*gamma)
# — that would need both endpoints' decay rates, which this per-edge,
# decay-blind constraint doesn't have; use the achieved solution's
# post-hoc 'cooperativities' dict (in optimize_given_conditions's info_out)
# to check the real cooperativity after the fact instead.
# Dispatched like Constraint_coupling_symmetric: called as (u, edges) where
# u are the coherent log-ratios only (not the auxiliary log-rates).
class Constraint_cooperativity_cap:
    slot = 'soft'

    def __init__(self, i, j, edge_type, G_tilde_max, penalty_strength=1.0):
        self.edge = (min(i, j), max(i, j), edge_type[:3])
        self.G_tilde_max = G_tilde_max
        self.penalty_strength = penalty_strength

    def __call__(self, log_ratios, edges):
        for k, e in enumerate(edges):
            key = (min(e['i'], e['j']), max(e['i'], e['j']), e.get('type', '')[:3])
            if key == self.edge:
                G_tilde_k = jnp.exp(log_ratios[k])
                return self.penalty_strength * jnp.maximum(0., G_tilde_k - self.G_tilde_max) ** 2
        return 0.0


# Classifies a flat constraint list into the three §2.6 slots. Thin,
# non-mutating — structural/domain constraints already flow through
# translate_triu_to_conditions/give_free_variable_idxs as slots (a)/(b)
# describe; this just lets a caller inspect where each requirement landed.
class ConstraintRegistry:
    def __init__(self, constraints=None):
        self.constraints = list(constraints or [])

    @property
    def structural(self):
        return [c for c in self.constraints if getattr(c, 'slot', None) == 'structural']

    @property
    def domain(self):
        return [c for c in self.constraints if getattr(c, 'slot', None) == 'domain']

    @property
    def soft(self):
        return [c for c in self.constraints if getattr(c, 'slot', None) == 'soft']

    def add(self, constraint):
        self.constraints.append(constraint)
        return self

    def __iter__(self):
        return iter(self.constraints)

    def __repr__(self):
        return (f'ConstraintRegistry(structural={len(self.structural)}, '
                f'domain={len(self.domain)}, soft={len(self.soft)})')


# Factory: build a constraint list from index lists, e.g.
# setup_constraints(edges_absent=[(0,1)], edges_beamsplitter=[(0,2)]).
def setup_constraints(edges_absent=None, edges_beamsplitter=None, edges_tms=None):
    conditions = []
    for (i,j) in (edges_absent or []):
        conditions.append(Constraint_coupling_absent(i, j))
    for (i,j) in (edges_beamsplitter or []):
        conditions.append(Constraint_coupling_beamsplitter(i, j))
    for (i,j) in (edges_tms or []):
        conditions.append(Constraint_coupling_two_mode_squeezing(i, j))
    return conditions


# Raise if a constraint list is contradictory (same edge marked both absent
# and typed). Used to fail fast on bad input.
def check_overlapping_constraints(list_of_constraints):
    from collections import defaultdict
    by_edge = defaultdict(list)
    for c in list_of_constraints:
        if isinstance(c, Coupling_Constraint):
            by_edge[tuple(c.idxs)].append(type(c).__name__)
    for edge, types in by_edge.items():
        if 'Constraint_coupling_absent' in types and len(types) > 1:
            raise ValueError(f"Contradictory constraints on edge {edge}: absent + type")


# Drop redundant type constraints on edges already marked absent.
def cleanup_list_of_constraints(combo):
    absent = {tuple(c.idxs) for c in combo if isinstance(c, Constraint_coupling_absent)}
    return [c for c in combo
            if not (isinstance(c, (Constraint_coupling_beamsplitter,
                                   Constraint_coupling_two_mode_squeezing))
                    and tuple(c.idxs) in absent)]


# Draw the graph via networkx+matplotlib: BS edges green, TMS blue,
# parametric red self-loops; edge width proportional to coupling_strengths
# if given. Not yet implemented.
def plot_graph(
    nodes=None,
    edges_or_triu=None,
    coupling_strengths=None,
    node_colors=None,
    mode_types=None,
    positions=None,
    ax=None,
    edge_width=2,
    color_beamsplitter='green',
    color_two_mode_squeezing='blue',
    color_parametric='red',
):
    pass


# Grid of plot_graph panels, one per topology in list_of_triu_arrays. Not
# yet implemented.
def plot_list_of_graphs(
    list_of_triu_arrays,
    node_types,
    node_colors,
    positions=None,
    architectures_per_row=5,
    size_per_column=2.5,
    size_per_row=2.5,
    **kwargs,
):
    pass
