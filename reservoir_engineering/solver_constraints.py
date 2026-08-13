"""
solver_constraints.py
=====================
The constraint menu of solver_constraints.md, wired INTO the exact solve of
linear_oracle.py rather than applied as a filter afterwards.

Why "during the solve" is the whole point. linear_oracle decides a graph by
one global SVD on the admissible subspaces S_G (Hamiltonians the graph
permits) and S_Upsilon (dissipators the drains permit). Almost everything on
the physics menu — passivity, reality, vacuum vs squeezed reservoir,
reciprocity, geometry masks, equivariance — is a LINEAR condition on those
same objects, so it is not an extra test at all: it shrinks S_G / S_Upsilon
to a subspace and the SAME single solve then runs inside it. Nothing about
the method degrades. In particular the INVALID certificates stay sound,
because every certificate here is of the form "no member of this affine
family is Hurwitz", and a constraint only makes the family smaller.

The three tiers of solver_constraints.md, and where each enters:

  LINEAR     -> basis restriction, before any solve. `keep_G`/`keep_Y` drop
                whole generators; `G_maps`/`Y_maps` impose a linear residual
                L(G) = 0 / L(Upsilon) = 0 and the basis is replaced by a
                nullspace basis of it. One sound solve, no false negatives.
  CONVEX     -> a penalty on the search score plus a hard acceptance test on
                the returned witness (bounded coupling, bounded power, l1).
                The solve stays linear; only the point CHOSEN inside the
                solution set changes.
  NONCONVEX  -> acceptance tests over the solution set (gap floor, channel
                count). These are exactly the nonconvexity the reformulation
                removed, so they are hard filters over the Move-1 interior
                ascent — never conditions on bare feasibility.

The one asymmetry to keep in mind, straight from the doc: linear constraints
preserve FEASIBILITY exactly but say nothing about ATTRACTIVITY, which is
open and nonconvex. Imposing sparsity or equivariance on bare feasibility can
hand back the frozen G = 0 point; that is why every constraint here is
evaluated over the Hurwitz search (`constrained_attractivity_filter`) and
never over `solve_stationarity` alone.

Scale gauge. The physics is invariant under (G, Upsilon) -> (sG, sUpsilon)
(linear_oracle §1(b)), so any bound on a coupling STRENGTH is vacuous until
the scale is pinned. Every magnitude constraint here is therefore evaluated
on the gauge-fixed pair (G, Upsilon)/tr(Re Upsilon) — i.e. in units of the
total drain rate, the same normalisation `physical_parameters` reports in.

Typical use:

    from reservoir_engineering import solver_constraints as sc
    from reservoir_engineering.linear_oracle import complete_covariance
    from reservoir_engineering.targets import two_mode_squeezed

    V = complete_covariance(two_mode_squeezed(0.5), [0, 1], 3)
    cons = [sc.passive_hamiltonian(), sc.VacuumReservoir()]
    out = sc.constrained_decide(triu, V, [0, 1], ['cavity']*3, constraints=cons)
    out['verdict']              # VALID / INVALID / UNDECIDED, under the constraints
    out['constraint_report']    # per-constraint satisfaction of the witness

and for a whole sweep, `ConstrainedSearch` is `certified_search.CertifiedSearch`
with the constrained oracle substituted in.

Soundness notes, in one place:
  - INVALID from `structural_certificate` is computed on the UNCONSTRAINED
    admissible set. Sound here (smaller family, same conclusion), just not as
    sharp as it could be — which is why the constrained solution set gets its
    own dark-subspace and Routh-Hurwitz certificates below.
  - `admits_graph` (degree bounds and friends) excludes a graph by fiat. That
    verdict is reported with propagates=False, because it is NOT monotone in
    the subgraph order the search propagates along: a graph can violate a
    degree bound while its subgraphs satisfy it. Prefer masking the search's
    alphabet to using it as an oracle verdict.
  - Reservoir constraints are FRAME statements. `VacuumReservoir` means
    vacuum in the frame V was completed in, so it belongs with
    complete_covariance(..., aux_squeezing=None); `SqueezedReservoir(r, th)`
    belongs with the matching aux_squeezing. See complete_covariance's header
    for why that frame is a labelling and not a loss of generality.
"""

import numpy as np
import scipy.optimize as sciopt
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from reservoir_engineering.topology_search import NO_COUPLING
# VALID is re-exported rather than used directly: witnesses are stamped VALID
# by linear_oracle._witness, which is the only place allowed to mint one.
from reservoir_engineering.linear_oracle import (
    VALID, INVALID, UNDECIDED, MARGIN_TOL_DEFAULT, RESIDUAL_RTOL_DEFAULT,
    squeeze_symplectic, hamiltonian_basis, dissipation_basis,
    vacuum_dissipation, solve_stationarity, drift_matrix, diffusion_matrix,
    normalised_margin, structural_certificate, factor_dissipator,
    _assemble, _is_psd, _witness, _hurwitz_minors)


# ───────────────────────────────────────────────────────────────────────────
# context and gauge fixing
# ───────────────────────────────────────────────────────────────────────────
# ConstraintContext: everything a constraint may need that is not (G, Upsilon).
# Passed to every hook so constraints stay stateless in the graph and can be
# reused across a whole sweep.
class ConstraintContext:
    def __init__(self, num_modes: int, target_mode_ids: Sequence[int],
                 node_types: Optional[Sequence[str]] = None,
                 V: Optional[np.ndarray] = None):
        self.num_modes = int(num_modes)
        self.target_mode_ids = list(target_mode_ids)
        self.aux_ids = [i for i in range(self.num_modes)
                        if i not in self.target_mode_ids]
        self.node_types = list(node_types) if node_types is not None else None
        self.V = V
        self.dim = 2 * self.num_modes


# _gauge_normalise(G, Y): pin the scale gauge before judging any magnitude.
#
# tr(Re Upsilon) is the total drain rate (a vacuum drain of rate kappa
# contributes kappa/2 to each of its two diagonal entries), it is strictly
# positive for any PSD dissipator that actually damps, and it is linear — so
# it is the cheapest gauge fixing available and the one the doc suggests.
# Falling back to a norm keeps the function total on degenerate input rather
# than dividing by zero.
def _gauge_normalise(G: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = float(np.real(np.trace(np.real(Y))))
    if not np.isfinite(k) or abs(k) < 1e-30:
        k = max(float(np.linalg.norm(Y)), float(np.linalg.norm(G)), 1e-30)
    return G / k, Y / k


# _head(label): the generator TYPE of a basis label, i.e. the part before the
# mode indices. One of delta / par / par90 / bs / bs90 / tms / tms90 (see
# linear_oracle.hamiltonian_basis). Splitting on '_' is unambiguous because
# the type never contains one, while parsing the indices out of 'bs_110'
# would not be.
def _head(label: str) -> str:
    return label.split('_')[0]


# _nullspace_columns(C): orthonormal basis of {v : C v = 0}, with the rank
# tolerance tied to C's own largest singular value.
def _nullspace_columns(C: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    if C.size == 0:
        return np.eye(C.shape[1]) if C.ndim == 2 else np.zeros((0, 0))
    _, s, Vt = np.linalg.svd(C)
    scale = float(s[0]) if s.size else 0.0
    rank = int(np.sum(s > rtol * max(scale, 1e-300)))
    return Vt[rank:].T


# _stack_real(x): a complex residual becomes two real ones. Coefficients in
# both bases are REAL, so a complex-valued linear condition is two real
# conditions and must be stacked as such — imposing only the real part would
# silently leave the imaginary half of the constraint unenforced.
def _stack_real(x) -> np.ndarray:
    a = np.asarray(x).ravel()
    if np.iscomplexobj(a):
        return np.concatenate([a.real, a.imag])
    return a.astype(float)


# ───────────────────────────────────────────────────────────────────────────
# the constraint interface
# ───────────────────────────────────────────────────────────────────────────
# SolverConstraint: every hook is optional, and a constraint uses only the
# ones matching its tier.
#
#   admits_graph(triu, ctx)      combinatorial gate on the graph itself
#   mask_graph(triu, ctx)        edit the graph before the basis is built
#                                (support masks: geometry, lattices)
#   keep_G / keep_Y              drop whole generators from the basis
#   G_maps / Y_maps              linear residuals L(.) = 0, imposed by
#                                replacing the basis with a nullspace basis
#   fixed_Y(ctx)                 supply the fast path's fixed dissipator
#   violation(G, Y, ctx)         >= 0, must reach 0; HARD (convex/nonconvex)
#   cost(G, Y, ctx)              a soft objective term, never an acceptance
#                                test (l1 sparsity, power minimisation)
#
# violation/cost are always handed the GAUGE-FIXED pair — see _gauge_normalise
# — so a bound on a coupling means "in units of the total drain rate".
class SolverConstraint:
    name = 'constraint'
    kind = 'linear'                 # linear | convex | nonconvex | graph
    weight = 1e3                    # penalty weight in the search score
    tol = 1e-8                      # acceptance tolerance on `violation`

    def admits_graph(self, triu, ctx: ConstraintContext) -> bool:
        return True

    def mask_graph(self, triu, ctx: ConstraintContext):
        return triu

    def keep_G(self, label: str, Gb: np.ndarray, ctx: ConstraintContext) -> bool:
        return True

    def keep_Y(self, label: str, Sb: np.ndarray, Tb: np.ndarray,
               ctx: ConstraintContext) -> bool:
        return True

    def G_maps(self, ctx: ConstraintContext) -> List[Callable]:
        return []

    def Y_maps(self, ctx: ConstraintContext) -> List[Callable]:
        return []

    def fixed_Y(self, ctx: ConstraintContext) -> Optional[np.ndarray]:
        return None

    def violation(self, G: np.ndarray, Y: np.ndarray,
                  ctx: ConstraintContext) -> float:
        return 0.0

    def cost(self, G: np.ndarray, Y: np.ndarray,
             ctx: ConstraintContext) -> float:
        return 0.0

    def __repr__(self):
        return f'{self.name}[{self.kind}]'


# ConstraintSet: applies a list of constraints, in the order basis -> solve ->
# witness. Intersecting linear constraints is just stacking their residual
# maps and filters, so the set is itself a linear constraint.
class ConstraintSet:
    def __init__(self, constraints: Iterable[SolverConstraint] = ()):
        self.items: List[SolverConstraint] = list(constraints or [])

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return 'ConstraintSet(' + ', '.join(repr(c) for c in self.items) + ')'

    def admits_graph(self, triu, ctx) -> Tuple[bool, Optional[SolverConstraint]]:
        for c in self.items:
            if not c.admits_graph(triu, ctx):
                return False, c
        return True, None

    def mask_graph(self, triu, ctx):
        out = np.array(triu, dtype=int, copy=True)
        for c in self.items:
            out = np.array(c.mask_graph(out, ctx), dtype=int, copy=True)
        return out

    # restrict_hamiltonian: filters first (cheap, exact, preserves the
    # physical labels), then the linear residual maps, whose nullspace basis
    # is a set of MIXED generators — a combination of the graph's generators
    # is still a Hamiltonian the graph permits, so nothing is lost, but the
    # per-edge labels no longer survive. physical_parameters recovers them
    # afterwards by least squares against the full graph basis.
    def restrict_hamiltonian(self, h_basis, ctx, rtol: float = 1e-10):
        basis = [(lbl, Gb) for (lbl, Gb) in h_basis
                 if all(c.keep_G(lbl, Gb, ctx) for c in self.items)]
        maps = [m for c in self.items for m in c.G_maps(ctx)]
        if not maps or not basis:
            return basis
        C = np.column_stack([
            np.concatenate([_stack_real(m(Gb)) for m in maps])
            for _, Gb in basis])
        N = _nullspace_columns(C, rtol)
        out = []
        for k in range(N.shape[1]):
            G = np.zeros_like(basis[0][1])
            for b, (_, Gb) in enumerate(basis):
                G = G + N[b, k] * Gb
            out.append((f'gmix{k}', G))
        return out

    def restrict_dissipation(self, d_basis, ctx, rtol: float = 1e-10):
        basis = [(lbl, Sb, Tb) for (lbl, Sb, Tb) in d_basis
                 if all(c.keep_Y(lbl, Sb, Tb, ctx) for c in self.items)]
        maps = [m for c in self.items for m in c.Y_maps(ctx)]
        if not maps or not basis:
            return basis
        C = np.column_stack([
            np.concatenate([_stack_real(m(Sb + 1j * Tb)) for m in maps])
            for _, Sb, Tb in basis])
        N = _nullspace_columns(C, rtol)
        out = []
        for k in range(N.shape[1]):
            S = np.zeros_like(basis[0][1])
            T = np.zeros_like(basis[0][2])
            for b, (_, Sb, Tb) in enumerate(basis):
                S = S + N[b, k] * Sb
                T = T + N[b, k] * Tb
            out.append((f'ymix{k}', S, T))
        return out

    # fixed_Y: the fast path (§4A) pins Upsilon and solves for G alone. A
    # reservoir constraint may pin it differently; the last one wins, and a
    # candidate that violates any Y-linear constraint is rejected so the fast
    # path is SKIPPED rather than run in a frame the constraints forbid.
    def fixed_Y(self, ctx) -> Optional[np.ndarray]:
        Y = None
        for c in self.items:
            cand = c.fixed_Y(ctx)
            if cand is not None:
                Y = cand
        return Y

    def accepts_fixed_Y(self, Y: np.ndarray, ctx, rtol: float = 1e-9) -> bool:
        maps = [m for c in self.items for m in c.Y_maps(ctx)]
        scale = max(float(np.linalg.norm(Y)), 1e-300)
        for m in maps:
            if float(np.linalg.norm(_stack_real(m(Y)))) > rtol * scale:
                return False
        return all(c.keep_Y('fixed', np.real(Y), np.imag(Y), ctx)
                   for c in self.items)

    # violations / costs: evaluated on the GAUGE-FIXED pair, always.
    def violations(self, G, Y, ctx) -> Dict[str, float]:
        Gn, Yn = _gauge_normalise(G, Y)
        return {c.name: float(c.violation(Gn, Yn, ctx)) for c in self.items}

    def penalty(self, G, Y, ctx) -> float:
        Gn, Yn = _gauge_normalise(G, Y)
        p = 0.0
        for c in self.items:
            v = float(c.violation(Gn, Yn, ctx))
            if v > 0:
                p += c.weight * v
            p += float(c.cost(Gn, Yn, ctx))
        return p

    def satisfied(self, G, Y, ctx) -> bool:
        Gn, Yn = _gauge_normalise(G, Y)
        return all(float(c.violation(Gn, Yn, ctx)) <= c.tol for c in self.items)

    def report(self, G, Y, ctx) -> List[Dict]:
        Gn, Yn = _gauge_normalise(G, Y)
        return [{'name': c.name, 'kind': c.kind,
                 'violation': float(c.violation(Gn, Yn, ctx)),
                 'cost': float(c.cost(Gn, Yn, ctx)),
                 'satisfied': bool(float(c.violation(Gn, Yn, ctx)) <= c.tol)}
                for c in self.items]


def as_constraint_set(constraints) -> ConstraintSet:
    if isinstance(constraints, ConstraintSet):
        return constraints
    if constraints is None:
        return ConstraintSet()
    if isinstance(constraints, SolverConstraint):
        return ConstraintSet([constraints])
    return ConstraintSet(list(constraints))


# ───────────────────────────────────────────────────────────────────────────
# Hamiltonian structure — constraints on G           (all LINEAR)
# ───────────────────────────────────────────────────────────────────────────
# GeneratorPalette: keep or drop whole generator TYPES.
#
# This is the cheapest possible form of a linear constraint — it deletes
# basis columns instead of intersecting with a nullspace — and it is exact
# rather than a surrogate, because linear_oracle's basis is over generator
# BLOCKS. "Passive" is not "some entries vanish", it is "the tms and par
# blocks are absent", which is precisely a palette statement.
#
# Cross-check available for free: passivity in the block sense is [G, Omega]
# = 0, and {I2, J2} (delta, bs, bs90) is exactly the commuting span while
# {sx, sz} (tms, par) is exactly the anticommuting one — so the palette and
# the commutator condition define the same subspace.
class GeneratorPalette(SolverConstraint):
    kind = 'linear'

    def __init__(self, allowed: Optional[Iterable[str]] = None,
                 forbidden: Optional[Iterable[str]] = None,
                 name: str = 'generator_palette'):
        self.allowed = set(allowed) if allowed is not None else None
        self.forbidden = set(forbidden) if forbidden is not None else set()
        self.name = name

    def keep_G(self, label, Gb, ctx):
        h = _head(label)
        if h in self.forbidden:
            return False
        return True if self.allowed is None else (h in self.allowed)


def passive_hamiltonian() -> GeneratorPalette:
    """Number-conserving H: no anomalous (squeezing) terms anywhere in G."""
    return GeneratorPalette(forbidden={'tms', 'tms90', 'par', 'par90'},
                            name='passive_hamiltonian')


def real_couplings() -> GeneratorPalette:
    """h_ij real: no synthetic gauge flux, no free phase on any generator.

    Drops the 90-degree partner of every generator (bs90 / tms90 / par90),
    which is exactly `hamiltonian_basis(allow_phases=False)` — offered here as
    a constraint so it composes with the rest instead of being a separate
    argument threaded through the search.
    """
    return GeneratorPalette(forbidden={'bs90', 'tms90', 'par90'},
                            name='real_couplings')


def zero_detunings() -> GeneratorPalette:
    """Forbid the diagonal of G — stabilise without tuning mode frequencies."""
    return GeneratorPalette(forbidden={'delta'}, name='zero_detunings')


def no_single_mode_squeezing() -> GeneratorPalette:
    """No degenerate parametric drive (no self-edges in H)."""
    return GeneratorPalette(forbidden={'par', 'par90'},
                            name='no_single_mode_squeezing')


def only_single_mode_squeezing() -> GeneratorPalette:
    """Parametric drive may be degenerate only — no two-mode squeezing."""
    return GeneratorPalette(forbidden={'tms', 'tms90'},
                            name='only_single_mode_squeezing')


def no_two_mode_squeezing() -> GeneratorPalette:
    return GeneratorPalette(forbidden={'tms', 'tms90'},
                            name='no_two_mode_squeezing')


# _is_passive_generator(Gb): [G, Omega] = 0, the exact number-conserving test.
#
# Used instead of reading the generator's name because it is the definition
# rather than a proxy: {I2, J2} (detuning, bs, bs90) commutes with Omega and
# {sx, sz} (par, par90, tms, tms90) anticommutes, so the commutator sorts the
# basis into passive and anomalous with no label parsing — which also keeps it
# correct if the basis ever grows a generator this file does not know about.
def _is_passive_generator(Gb: np.ndarray, rtol: float = 1e-12) -> bool:
    n = Gb.shape[0] // 2
    Om = np.zeros((2 * n, 2 * n))
    for i in range(n):
        Om[2 * i, 2 * i + 1] = 1.0
        Om[2 * i + 1, 2 * i] = -1.0
    C = Gb @ Om - Om @ Gb
    return float(np.linalg.norm(C)) <= rtol * max(float(np.linalg.norm(Gb)), 1e-300)


# _generator_support(Gb): the modes a basis element touches.
def _generator_support(Gb: np.ndarray) -> set:
    n = Gb.shape[0] // 2
    return {i for i in range(n)
            if np.any(np.abs(Gb[2 * i:2 * i + 2, :]) > 0)
            or np.any(np.abs(Gb[:, 2 * i:2 * i + 2]) > 0)}


# PartialPassivity: passive WITHIN a set of modes, active everywhere else.
#
# The interesting middle ground between `passive_hamiltonian` (nothing
# anomalous anywhere, which for a squeezed target with vacuum drains is
# INVALID by construction) and the free search. Default `modes` is the SIGNAL
# block, so the question becomes:
#
#     may the signal modes be coupled to each other only by beamsplitters,
#     with every parametric resource pushed onto the signal-drain edges (and
#     the drain-drain ones)?
#
# That is exactly the structural question behind the reservoir-engineering
# schemes where the target manifold is built by the dissipative sector while
# the coherent network over the signal modes stays number-conserving. It is
# also the constraint an experiment usually faces: two-mode squeezing between
# two signal cavities needs a pump bridging their sum frequency, while a
# signal-to-drain blue sideband is the drive the drain is already wired for.
#
# A generator is dropped iff it is anomalous AND its support lies entirely
# inside `modes` — so a TMS between two signal modes goes, a parametric
# self-drive on a signal mode goes (single-mode squeezing acts within one
# signal mode, hence within the block), and a signal-drain TMS survives.
#
# LINEAR, and strictly weaker than passive_hamiltonian, so anything valid
# under full passivity is valid under this.
class PartialPassivity(SolverConstraint):
    kind = 'linear'

    def __init__(self, modes: Optional[Iterable[int]] = None,
                 name: Optional[str] = None):
        self.modes = None if modes is None else set(int(m) for m in modes)
        self._name = name

    @property
    def name(self):
        if self._name:
            return self._name
        return ('passive_target_block' if self.modes is None
                else f'passive_within{sorted(self.modes)}')

    def keep_G(self, label, Gb, ctx):
        block = set(ctx.target_mode_ids) if self.modes is None else self.modes
        if _is_passive_generator(Gb):
            return True
        return not _generator_support(Gb).issubset(block)


def passive_target_block() -> PartialPassivity:
    """Beamsplitter-only coupling AMONG the signal modes; anomalous terms are
    still allowed on the signal-drain and drain-drain edges."""
    return PartialPassivity()


# ───────────────────────────────────────────────────────────────────────────
# Connectivity / geometry — support masks              (LINEAR / combinatorial)
# ───────────────────────────────────────────────────────────────────────────
# EdgeMask: a fixed hardware mask, imposed by editing the GRAPH before the
# basis is built rather than by filtering the basis afterwards.
#
# Editing the graph is the right place for it: the search encodes topology in
# `triu`, so masking there keeps the graph the oracle reports identical to the
# graph it actually solved, and the subgraph propagation of certified_search
# stays meaningful. Self-edges (parametric) are left alone — use
# no_single_mode_squeezing for those.
class EdgeMask(SolverConstraint):
    kind = 'linear'

    def __init__(self, allowed_pairs: Optional[Iterable[Tuple[int, int]]] = None,
                 forbidden_pairs: Optional[Iterable[Tuple[int, int]]] = None,
                 name: str = 'edge_mask'):
        self.allowed = ({tuple(sorted(p)) for p in allowed_pairs}
                        if allowed_pairs is not None else None)
        self.forbidden = ({tuple(sorted(p)) for p in forbidden_pairs}
                          if forbidden_pairs is not None else set())
        self.name = name

    def _ok(self, i, j):
        if i == j:
            return True
        p = (min(i, j), max(i, j))
        if p in self.forbidden:
            return False
        return True if self.allowed is None else (p in self.allowed)

    def mask_graph(self, triu, ctx):
        out = np.array(triu, dtype=int, copy=True)
        rows, cols = np.triu_indices(ctx.num_modes)
        for k, (i, j) in enumerate(zip(rows, cols)):
            if not self._ok(int(i), int(j)):
                out[k] = NO_COUPLING
        return out


def nearest_neighbour(num_modes: int, ring: bool = False) -> EdgeMask:
    """Open chain 0-1-2-... (ring=True closes it)."""
    pairs = [(i, i + 1) for i in range(num_modes - 1)]
    if ring and num_modes > 2:
        pairs.append((0, num_modes - 1))
    return EdgeMask(allowed_pairs=pairs,
                    name='nearest_neighbour' + ('_ring' if ring else ''))


def lattice_mask(adjacency: np.ndarray, name: str = 'lattice') -> EdgeMask:
    """Any fixed geometry, given as a 0/1 adjacency matrix over the modes."""
    A = np.asarray(adjacency)
    pairs = [(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[1])
             if A[i, j]]
    return EdgeMask(allowed_pairs=pairs, name=name)


def distance_graded(coords, cutoff: float, name: str = 'distance_graded') -> EdgeMask:
    """Couplings only within `cutoff` of each other — the 0/1 form of the
    doc's graded mask. The graded version (|G_ij| <= f(dist)) is a magnitude
    bound, so it belongs to BoundedCoupling with a per-pair ceiling."""
    X = np.asarray(coords, dtype=float)
    pairs = [(i, j) for i in range(len(X)) for j in range(i + 1, len(X))
             if np.linalg.norm(X[i] - X[j]) <= cutoff]
    return EdgeMask(allowed_pairs=pairs, name=name)


def restricted_aux_connectivity(aux_id: int, allowed_targets: Iterable[int],
                                num_modes: int) -> EdgeMask:
    """The drain may touch only `allowed_targets`. Everything not involving
    `aux_id` is left untouched."""
    allowed = set(int(t) for t in allowed_targets)
    forbidden = [(aux_id, j) for j in range(num_modes)
                 if j != aux_id and j not in allowed]
    return EdgeMask(forbidden_pairs=forbidden,
                    name=f'aux{aux_id}_connectivity')


# DegreeBound: coordination-limited hardware. COMBINATORIAL, so it excludes
# the graph outright rather than shrinking a subspace.
#
# NOT MONOTONE in the subgraph order certified_search propagates along — a
# graph can bust the bound while every subgraph respects it — so a verdict
# from this constraint is returned with propagates=False and must not seed the
# search's invalid set. The clean way to use it is to mask the search's
# alphabet instead.
class DegreeBound(SolverConstraint):
    kind = 'graph'
    name = 'degree_bound'

    def __init__(self, max_degree: int, modes: Optional[Iterable[int]] = None):
        self.max_degree = int(max_degree)
        self.modes = None if modes is None else set(int(m) for m in modes)
        self.name = f'degree_bound<={max_degree}'

    def admits_graph(self, triu, ctx):
        deg = [0] * ctx.num_modes
        rows, cols = np.triu_indices(ctx.num_modes)
        for k, (i, j) in enumerate(zip(rows, cols)):
            if i != j and int(np.asarray(triu)[k]) != NO_COUPLING:
                deg[i] += 1
                deg[j] += 1
        who = range(ctx.num_modes) if self.modes is None else self.modes
        return all(deg[m] <= self.max_degree for m in who)


# ───────────────────────────────────────────────────────────────────────────
# Dissipator structure — constraints on Upsilon
# ───────────────────────────────────────────────────────────────────────────
# _annihilation_vectors(modes, num_modes): the isotropic vectors u_m =
# e_{2m} + i e_{2m+1}, one per mode.
#
# These are what makes "vacuum reservoir" LINEAR. A jump vector reads
# L = c_x x + c_p p = alpha a + beta a^dag with beta = (c_x + i c_p)/sqrt2, so
# a plain-loss channel (beta = 0 on every mode) is exactly c_p = i c_x, i.e.
# c^T u_m = 0 for every m. Since Upsilon = sum_mu conj(c_mu) c_mu^T,
#
#       every channel is plain loss   <=>   Upsilon u_m = 0 for all m
#
# (=> is immediate; <= because u_m^dag Upsilon u_m = sum_mu |c_mu^T u_m|^2 is
# a sum of squares, so it vanishing forces each term to vanish — for PSD
# Upsilon, which every physical one is).
def _annihilation_vectors(modes: Sequence[int], num_modes: int) -> np.ndarray:
    U = np.zeros((2 * num_modes, len(modes)), dtype=complex)
    for k, m in enumerate(modes):
        U[2 * m, k] = 1.0
        U[2 * m + 1, k] = 1j
    return U


# VacuumReservoir: pure loss, no anomalous weight anywhere in the dissipator.
#
# LINEAR (see _annihilation_vectors), and it forces the active-H branch: with
# the bath supplying no squeezing, every anomalous resource must come from the
# Hamiltonian. Combined with passive_hamiltonian it is the sharpest question
# on the menu — passive network AND vacuum drains — and the expected answer is
# INVALID for any squeezed target, which is a cross-check on the whole oracle
# rather than a search.
#
# FRAME NOTE: "vacuum" means vacuum in the frame V was completed in. Use with
# complete_covariance(..., aux_squeezing=None).
class VacuumReservoir(SolverConstraint):
    kind = 'linear'
    name = 'vacuum_reservoir'

    def __init__(self, modes: Optional[Iterable[int]] = None,
                 pin_fast_path: bool = True):
        self.modes = None if modes is None else [int(m) for m in modes]
        self.pin_fast_path = pin_fast_path

    def _targets(self, ctx):
        return ctx.aux_ids if self.modes is None else self.modes

    def Y_maps(self, ctx):
        U = _annihilation_vectors(self._targets(ctx), ctx.num_modes)
        return [lambda Y, U=U: Y @ U]

    def fixed_Y(self, ctx):
        if not self.pin_fast_path:
            return None
        return vacuum_dissipation(self._targets(ctx), ctx.num_modes, rate=1.0)

    # Reported for the witness even though the linear constraint already
    # guarantees it: the anomalous weight is what a reader wants to see.
    def violation(self, G, Y, ctx):
        ch = factor_dissipator(Y, ctx.num_modes)
        return max([abs(p['beta']) for c in ch for p in c['modes']], default=0.0)


# SqueezedReservoir: pin the anomalous block instead of killing it — the
# drains sit in a squeezed vacuum of parameter (r, theta), and the solve runs
# inside the corresponding subspace.
#
# The map is the vacuum one conjugated by the drain symplectic. The direction
# of that conjugation is fixed by (**) itself and is worth deriving rather
# than guessing: the equation is invariant under
#       V -> S V S^T,      (G, Upsilon) -> (S^{-T} G S^{-1}, S^{-T} Upsilon S^{-1})
# for symplectic S (use Omega S = S^{-T} Omega on each of the three terms).
# complete_covariance builds the squeezed drain as V_AA = (1/2) S S^T with
# S = squeeze_symplectic(r, theta), so the matching dissipator is
# Upsilon = S^{-T} Upsilon_vac S^{-1} and therefore
#       Upsilon (S u_m) = S^{-T} (Upsilon_vac u_m) = 0,
# which is again linear in Upsilon. (Confirmed against a solved witness: the
# residual on S u is 3.7e-17 against a dissipator of norm 1e-4, while the
# opposite convention S^{-1} u leaves it at full size.) Sweeping r turns "how
# much reservoir squeezing does this graph need?" into a sequence of exact
# solves, each certifying at its own r.
#
# FRAME NOTE: pair it with complete_covariance(..., aux_squeezing=[(r, th)...])
# using the SAME (r, theta). A squeezed bath drives the drain to a squeezed
# steady state; asking for one in a frame whose V says the drain is unsqueezed
# is asking for something inconsistent, and the solve will simply say INVALID.
class SqueezedReservoir(SolverConstraint):
    kind = 'linear'

    def __init__(self, r: float, theta: float = 0.0,
                 modes: Optional[Iterable[int]] = None):
        self.r = float(r)
        self.theta = float(theta)
        self.modes = None if modes is None else [int(m) for m in modes]
        self.name = f'squeezed_reservoir(r={r:.3f},th={theta:.3f})'

    def _targets(self, ctx):
        return ctx.aux_ids if self.modes is None else self.modes

    def _S(self, ctx):
        S = np.eye(2 * ctx.num_modes)
        blk = squeeze_symplectic(self.r, self.theta)
        for m in self._targets(ctx):
            S[2 * m:2 * m + 2, 2 * m:2 * m + 2] = blk
        return S

    def Y_maps(self, ctx):
        U = _annihilation_vectors(self._targets(ctx), ctx.num_modes)
        W = self._S(ctx) @ U
        return [lambda Y, W=W: Y @ W]

    def violation(self, G, Y, ctx):
        # How far the realised channels are from squeezing r, as the doc's
        # "minimum reservoir squeezing" read-out.
        ch = factor_dissipator(Y, ctx.num_modes)
        s = [p['squeezing'] for c in ch for p in c['modes']]
        return max([abs(v - self.r) for v in s], default=0.0)


# DrainSupport: which auxiliary modes may carry dissipation at all, and
# whether a jump operator may span more than one of them.
#
# max_cluster=1 is the "local dissipators" row of the menu: every jump
# operator lives on ONE site, which is also what
# dissipation_basis(coupled_drains=False) gives. Passing modes= is the
# stronger statement "these drains only".
class DrainSupport(SolverConstraint):
    kind = 'linear'
    name = 'drain_support'

    def __init__(self, modes: Optional[Iterable[int]] = None,
                 max_cluster: int = 1):
        self.modes = None if modes is None else set(int(m) for m in modes)
        self.max_cluster = int(max_cluster)
        self.name = ('local_dissipators' if modes is None
                     else f'drain_support{sorted(self.modes)}')

    def keep_Y(self, label, Sb, Tb, ctx):
        supp = sorted({a // 2 for a in np.nonzero(np.abs(Sb) + np.abs(Tb))[0]})
        if self.modes is not None and any(m not in self.modes for m in supp):
            return False
        return len(supp) <= self.max_cluster


# ReciprocalDissipation: kill the off-diagonal imaginary part of Upsilon.
#
# Im Upsilon enters the drift as A = Omega(G + Im Upsilon), so its off-block
# entries are the dissipative coupling between distinct modes; setting them to
# zero removes the interference between coherent and dissipative paths that
# non-reciprocity is built from. This is the doc's LINEAR SUFFICIENT
# condition, not the exact reciprocity boundary (which is bilinear) — a graph
# that is VALID under it is genuinely reciprocal, while INVALID under it means
# only "not reciprocal by this criterion".
#
# With coupled_drains=False the basis has no cross-mode elements to begin
# with, so this constraint only bites in the shared-reservoir case.
class ReciprocalDissipation(SolverConstraint):
    kind = 'linear'
    name = 'reciprocity'

    def keep_Y(self, label, Sb, Tb, ctx):
        if not np.any(Tb):
            return True
        idx = np.nonzero(np.abs(Tb))
        return all(a // 2 == b // 2 for a, b in zip(*idx))


# ChannelCount: rank(Upsilon) = k, i.e. exactly k dissipative channels.
#
# NONCONVEX (rank), so it is an acceptance test over the Hurwitz interior plus
# a continuous surrogate that steers the search: the violation is the
# normalised weight sitting in the eigenvalues BEYOND the k largest, which is
# the same quantity a nuclear-norm relaxation would push down, evaluated
# exactly instead of relaxed. k = 1 is the flat-spectrum branch.
class ChannelCount(SolverConstraint):
    kind = 'nonconvex'

    def __init__(self, k: int, mode: str = 'max', tol: float = 1e-6):
        self.k = int(k)
        self.mode = mode                 # 'max' (<= k) or 'exact'
        self.tol = float(tol)
        self.name = f'channels{"<=" if mode == "max" else "=="}{k}'

    def violation(self, G, Y, ctx):
        w = np.linalg.svd(Y, compute_uv=False)
        tot = float(np.sum(w))
        if tot < 1e-300:
            return 1.0                   # no channels at all: D = 0, never Hurwitz
        # Weight sitting outside the k largest channels, relative to the total.
        # Zero exactly when rank <= k, and it is what a nuclear-norm surrogate
        # would push down — computed exactly here instead of relaxed.
        tail = float(np.sum(w[self.k:])) / tot
        if self.mode != 'exact':
            return tail
        # rank == k also needs the k-th channel to carry real weight, or a
        # rank-1 dissipator would pass as "exactly 2 channels".
        kth = float(w[self.k - 1]) / tot if self.k - 1 < len(w) else 0.0
        return tail + max(0.0, 1e-3 - kth)


# ───────────────────────────────────────────────────────────────────────────
# Symmetry — constraints on G and Upsilon jointly       (all LINEAR)
# ───────────────────────────────────────────────────────────────────────────
# Equivariance: S^T G S = G and S^T Upsilon S = Upsilon for every S given.
#
# The doc's twirl, imposed exactly. Unitary symmetries take S real symplectic
# and orthogonal (mode permutations, phase rotations); ANTIUNITARY ones
# (time reversal, particle-hole, chiral) come with a complex conjugation, and
# the Upsilon condition is then S^T conj(Upsilon) S = Upsilon.
#
# Getting that conjugation right is not cosmetic. Under the quadrature
# reflection R = diag(1,-1) per mode a plain vacuum dissipator obeys
# R^T conj(Upsilon) R = Upsilon but NOT R^T Upsilon R = Upsilon — dissipation
# is not invariant under time reversal alone — so the unitary form of the
# condition would outlaw ordinary loss and report every graph INVALID.
#
# Both conditions are real-linear in the basis coefficients (which are real),
# so this stays one solve; block-diagonalising by irrep would make it cheaper
# still, and is left to the caller's choice of S.
class Equivariance(SolverConstraint):
    kind = 'linear'

    def __init__(self, mats: Iterable[np.ndarray], antiunitary: bool = False,
                 name: str = 'equivariance', on_G: bool = True,
                 on_Y: bool = True):
        self.mats = [np.asarray(S, dtype=float) for S in mats]
        self.antiunitary = bool(antiunitary)
        self.on_G, self.on_Y = bool(on_G), bool(on_Y)
        self.name = name

    def G_maps(self, ctx):
        if not self.on_G:
            return []
        return [lambda G, S=S: S.T @ G @ S - G for S in self.mats]

    def Y_maps(self, ctx):
        if not self.on_Y:
            return []
        if self.antiunitary:
            return [lambda Y, S=S: S.T @ np.conj(Y) @ S - Y for S in self.mats]
        return [lambda Y, S=S: S.T @ Y @ S - Y for S in self.mats]


# mode_permutation(perm, num_modes): the 2n x 2n symplectic that relabels
# modes, i.e. mode i is sent to perm[i]. Both quadratures move together, so it
# is a permutation matrix on quadrature pairs and is orthogonal AND symplectic.
def mode_permutation(perm: Sequence[int], num_modes: int) -> np.ndarray:
    S = np.zeros((2 * num_modes, 2 * num_modes))
    for i, p in enumerate(perm):
        S[2 * p, 2 * i] = 1.0
        S[2 * p + 1, 2 * i + 1] = 1.0
    return S


# phase_rotation(angles, num_modes): local rotation a_i -> e^{i phi_i} a_i, as
# the per-mode 2x2 rotation on (x, p).
def phase_rotation(angles: Sequence[float], num_modes: int) -> np.ndarray:
    S = np.eye(2 * num_modes)
    for m, th in enumerate(angles):
        c, s = np.cos(th), np.sin(th)
        S[2 * m:2 * m + 2, 2 * m:2 * m + 2] = np.array([[c, -s], [s, c]])
    return S


# quadrature_reflection(num_modes): p -> -p on every mode. The symplectic part
# of the antiunitary time-reversal / particle-hole family.
def quadrature_reflection(num_modes: int, modes: Optional[Sequence[int]] = None
                          ) -> np.ndarray:
    S = np.eye(2 * num_modes)
    for m in (range(num_modes) if modes is None else modes):
        S[2 * m + 1, 2 * m + 1] = -1.0
    return S


def translation_invariance(num_modes: int, modes: Optional[Sequence[int]] = None
                           ) -> Equivariance:
    """Circulant G and Upsilon over `modes` (default: all) — the abelian case,
    whose irreps are the quasimomentum sectors."""
    ids = list(range(num_modes)) if modes is None else list(modes)
    perm = list(range(num_modes))
    for k, m in enumerate(ids):
        perm[m] = ids[(k + 1) % len(ids)]
    return Equivariance([mode_permutation(perm, num_modes)],
                        name='translation_invariance')


def permutation_symmetry(perms: Iterable[Sequence[int]], num_modes: int,
                         name: str = 'permutation_symmetry') -> Equivariance:
    """Equivariance under a permutation group, given by its generators."""
    return Equivariance([mode_permutation(p, num_modes) for p in perms],
                        name=name)


def time_reversal(num_modes: int) -> Equivariance:
    """Antiunitary p -> -p invariance. Sharper than real_couplings: it keeps
    tms/bs and kills bs90/tms90/par, a different slice of the palette."""
    return Equivariance([quadrature_reflection(num_modes)], antiunitary=True,
                        name='time_reversal')


# ───────────────────────────────────────────────────────────────────────────
# Resources / performance — magnitude bounds and objectives
# ───────────────────────────────────────────────────────────────────────────
# BoundedCoupling: |G_ij| <= g_max (or ||G|| <= g_max), in units of the total
# drain rate.
#
# CONVEX, and vacuous without the scale gauge — hence _gauge_normalise. This
# is also the "adiabatic elimination stays valid" and "cooperativity cap" rows
# of the menu, which are the same bound read in different units.
#
# mode='entry' bounds the 2x2 BLOCK norm of the pair (i, j), which is the
# gauge-invariant strength of everything on that edge at once — not the same
# number as physical_parameters' |J_ij|, which resolves the block into its
# individual generator coefficients. Bounding the block is the right form for
# a hardware ceiling, since a bench cannot exceed g_max on the BS part while
# hiding extra strength in the TMS part of the same edge.
class BoundedCoupling(SolverConstraint):
    kind = 'convex'

    def __init__(self, g_max: float, mode: str = 'entry', weight: float = 1e2):
        self.g_max = float(g_max)
        self.mode = mode                 # 'entry' (max block norm) or 'norm'
        self.weight = float(weight)
        self.name = f'|g|<={g_max:g}({mode})'

    def violation(self, G, Y, ctx):
        if self.mode == 'norm':
            val = float(np.linalg.norm(G, 2))
        else:
            n = ctx.num_modes
            val = max(float(np.linalg.norm(G[2 * i:2 * i + 2, 2 * j:2 * j + 2]))
                      for i in range(n) for j in range(n))
        return max(0.0, val - self.g_max)


def cooperativity_cap(g_over_kappa_max: float) -> BoundedCoupling:
    """Keep the scheme inside the reduced model's regime: ||G|| / kappa small."""
    c = BoundedCoupling(g_over_kappa_max, mode='norm')
    c.name = f'g/kappa<={g_over_kappa_max:g}'
    return c


# BoundedPower: tr(G^2) <= E, the drive-budget ceiling. Convex, scale-fixed.
class BoundedPower(SolverConstraint):
    kind = 'convex'

    def __init__(self, e_max: float, weight: float = 1e2):
        self.e_max = float(e_max)
        self.weight = float(weight)
        self.name = f'tr(G^2)<={e_max:g}'

    def violation(self, G, Y, ctx):
        return max(0.0, float(np.trace(G @ G)) - self.e_max)


# L1Sparsity: minimise the number of physical couplings, by the l1 surrogate.
#
# A COST, never an acceptance test — it steers which point of the solution set
# the search returns without being able to reject a witness. It is also the
# constraint the doc warns loudest about: minimising l1 over BARE feasibility
# returns the sparse-but-frozen G = 0, so it is applied here on top of the
# Hurwitz score, where the margin term keeps the point attractive. Note also
# that sparsity is a PHYSICAL-frame statement; sparsity in the vacuum-drain
# gauge is not the same thing (see complete_covariance's frame note).
class L1Sparsity(SolverConstraint):
    kind = 'convex'
    name = 'l1_sparsity'

    def __init__(self, weight: float = 1e-2):
        self.weight = float(weight)

    def cost(self, G, Y, ctx):
        n = ctx.num_modes
        s = 0.0
        for i in range(n):
            for j in range(i, n):
                s += float(np.linalg.norm(G[2 * i:2 * i + 2, 2 * j:2 * j + 2]))
        return self.weight * s


# MinimumGap: gamma* >= gamma, the dissipative-gap floor.
#
# NONCONVEX (spectral abscissa) — Move 1 recast as a hard constraint, and the
# one that turns a VALID bit into a Pareto point: run it against
# BoundedCoupling and each verdict becomes a (gap, max coupling) pair rather
# than a yes/no. The gap is the NORMALISED margin (-max Re lambda / ||A||),
# since the raw one is meaningless on a cone.
class MinimumGap(SolverConstraint):
    kind = 'nonconvex'

    def __init__(self, gamma: float, weight: float = 1e2):
        self.gamma = float(gamma)
        self.weight = float(weight)
        self.name = f'gap>={gamma:g}'

    def violation(self, G, Y, ctx):
        return max(0.0, self.gamma - normalised_margin(drift_matrix(G, Y)))


# ───────────────────────────────────────────────────────────────────────────
# presets
# ───────────────────────────────────────────────────────────────────────────
# The doc's four FREE picks (linear, so no cost at all) plus the pairing that
# changes the character of the output.
def highest_value_toggles(num_modes: int, squeezed_r: Optional[float] = None,
                          reciprocal: bool = False,
                          symmetry: Optional[Iterable[Sequence[int]]] = None,
                          passive: bool = True) -> ConstraintSet:
    items: List[SolverConstraint] = []
    if passive:
        items.append(passive_hamiltonian())
    items.append(VacuumReservoir() if squeezed_r in (None, 0.0)
                 else SqueezedReservoir(squeezed_r))
    if reciprocal:
        items.append(ReciprocalDissipation())
    if symmetry:
        items.append(permutation_symmetry(symmetry, num_modes))
    return ConstraintSet(items)


def resource_frontier(g_max: float, gamma: float) -> ConstraintSet:
    """Bounded coupling + gap floor: the pair that upgrades each VALID verdict
    from one bit to a point on a gap-vs-coupling Pareto curve."""
    return ConstraintSet([BoundedCoupling(g_max), MinimumGap(gamma)])


PRESETS: Dict[str, Callable[..., ConstraintSet]] = {
    'passive_vacuum': lambda n=3: ConstraintSet([passive_hamiltonian(),
                                                 VacuumReservoir()]),
    'active_vacuum': lambda n=3: ConstraintSet([VacuumReservoir()]),
    'passive_signal_block': lambda n=3: ConstraintSet([passive_target_block(),
                                                       VacuumReservoir()]),
    'passive_squeezed': lambda n=3, r=0.5: ConstraintSet([passive_hamiltonian(),
                                                          SqueezedReservoir(r)]),
    'reciprocal': lambda n=3: ConstraintSet([ReciprocalDissipation()]),
    'real_passive': lambda n=3: ConstraintSet([passive_hamiltonian(),
                                               real_couplings()]),
    'no_detuning': lambda n=3: ConstraintSet([zero_detunings()]),
    'nn_chain': lambda n=3: ConstraintSet([nearest_neighbour(n)]),
}


# ───────────────────────────────────────────────────────────────────────────
# the constrained solve
# ───────────────────────────────────────────────────────────────────────────
# constrained_bases(triu, ctx, cset, ...) -> (triu_eff, h_basis, d_basis).
#
# Everything linear happens here, before a single solve: mask the graph, build
# the ordinary generator bases on the masked graph, then restrict them. What
# comes out has exactly the interface linear_oracle's solver expects — a list
# of (label, G) and a list of (label, Re, Im) — so every downstream routine
# works unchanged inside the constrained subspace.
def constrained_bases(triu, ctx: ConstraintContext, cset: ConstraintSet,
                      include_detunings: bool = True, allow_phases: bool = True,
                      coupled_drains: bool = False, rtol: float = 1e-10):
    triu_eff = cset.mask_graph(triu, ctx)
    h_basis = hamiltonian_basis(triu_eff, ctx.num_modes,
                                include_detunings=include_detunings,
                                allow_phases=allow_phases)
    d_basis = dissipation_basis(ctx.aux_ids, ctx.num_modes,
                                coupled_drains=coupled_drains)
    return (triu_eff,
            cset.restrict_hamiltonian(h_basis, ctx, rtol),
            cset.restrict_dissipation(d_basis, ctx, rtol))


# constrained_attractivity_filter: linear_oracle's §4C filter with the
# constraint penalty folded into the SEARCH score and the constraint
# satisfaction folded into the ACCEPTANCE test.
#
# The split between the two is the same one linear_oracle makes for PSD, and
# for the same reason: rejecting a violating point outright makes the whole
# infeasible region flat, so a simplex started outside it has no gradient to
# follow and stalls. Penalising continuously lets the search WALK IN from
# outside, while acceptance still demands the genuine article.
def constrained_attractivity_filter(
    sol: Dict, h_basis, d_basis, Y_fixed, ctx: ConstraintContext,
    cset: ConstraintSet, num_samples: int = 32, sample_scale: float = 1.0,
    seed: int = 0, margin_tol: float = MARGIN_TOL_DEFAULT,
    require_psd: bool = False, n_polish: int = 4, maximise: bool = False,
) -> Tuple[bool, Optional[np.ndarray], float]:
    dim = ctx.dim
    z0, N = sol['z0'], sol['nullspace']
    n_null = N.shape[1] if N.size else 0
    psd_weight = 1e3

    def pair(z):
        return _assemble(z, h_basis, d_basis, Y_fixed, dim)

    def margin_of(z):
        G, Y = pair(z)
        if require_psd and not _is_psd(Y):
            return -np.inf
        if not cset.satisfied(G, Y, ctx):
            return -np.inf
        return normalised_margin(drift_matrix(G, Y))

    def score(z):
        G, Y = pair(z)
        m = normalised_margin(drift_matrix(G, Y))
        if require_psd:
            w = np.linalg.eigvalsh((Y + Y.conj().T) / 2)
            scale_w = max(1.0, float(np.max(np.abs(w))))
            m = m - psd_weight * max(0., -float(np.min(w))) / scale_w
        return m - cset.penalty(G, Y, ctx)

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

    def neg_score(alpha):
        s = score(z0 + N @ alpha)
        return 1e9 if not np.isfinite(s) else -s

    starts = [np.linalg.lstsq(N, best_score_z - z0, rcond=None)[0]]
    if maximise:
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


# constrained_dark_subspace(sol, ...) -> dimension of the subspace that is
# dark at EVERY point of the CONSTRAINED solution set.
#
# linear_oracle.solution_set_dark_subspace computes the same object over the
# unconstrained solution set. Recomputing it here is not redundancy: the
# constrained solution set is smaller, so the intersection defining the dark
# subspace is over fewer points and can only be LARGER. That means this
# certificate fires on graphs the unconstrained one leaves undecided — which
# is the entire reason a constrained search can prune harder than the free one.
def constrained_dark_subspace(sol: Dict, h_basis, d_basis, dim: int,
                              rtol: float = 1e-9) -> int:
    N = sol['nullspace']
    if N.size == 0 or N.shape[1] == 0:
        return 0

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
            P = cur @ cur.conj().T
            K = null_c((np.eye(dim) - P) @ M @ cur,
                       ref=float(np.linalg.norm(M, 2)))
            cur = orth_c(cur @ K) if K.shape[1] else np.zeros((dim, 0), dtype=complex)
        if cur.shape[1] == W.shape[1]:
            W = cur
            break
        W = cur
    return int(W.shape[1])


# constrained_routh_certificate(sol, ...): Move 3's universal half, run on the
# constrained solution set.
#
# A(alpha) is linear in the coordinates of the solution set, so each Hurwitz
# minor Delta_k(alpha) is a polynomial of degree <= n*k; sampling n*k+2 points
# on a random line PROVES it vanishes there (fundamental theorem of algebra),
# and repeating on independent lines extends it to the family. If any minor
# vanishes identically, Routh-Hurwitz says no member is Hurwitz — a sound
# INVALID for the constrained problem, and one that does not care whether the
# dark direction rotates from point to point.
def constrained_routh_certificate(sol: Dict, h_basis, d_basis, dim: int,
                                  n_lines: int = 8, rtol: float = 1e-9,
                                  seed: int = 0) -> Optional[Dict]:
    N = sol['nullspace']
    if N.size == 0 or N.shape[1] == 0:
        return None
    As = [drift_matrix(*_assemble(N[:, k], h_basis, d_basis, None, dim))
          for k in range(N.shape[1])]
    m, n = len(As), dim
    rng = np.random.default_rng(seed)

    def minors_at(alpha):
        A = sum(alpha[j] * As[j] for j in range(m))
        nrm = float(np.linalg.norm(A))
        if nrm < 1e-300:
            return None
        return _hurwitz_minors(np.real(np.poly(A / nrm))[1:])

    vanishing = [True] * n
    worst = [0.0] * n
    for _ in range(n_lines):
        alpha0, u = rng.normal(0., 1., m), rng.normal(0., 1., m)
        for k in range(1, n + 1):
            if not vanishing[k - 1]:
                continue
            for t in np.linspace(-1.0, 1.0, n * k + 2):
                mins = minors_at(alpha0 + t * u)
                if mins is None:
                    continue
                val = abs(float(mins[k - 1]))
                worst[k - 1] = max(worst[k - 1], val)
                if val > rtol:
                    vanishing[k - 1] = False
                    break

    for k in range(1, n + 1):
        if vanishing[k - 1]:
            return {'kind': 'constrained_hurwitz_minor_vanishes', 'minor': k,
                    'max_abs_over_samples': worst[k - 1], 'lines': n_lines,
                    'degree_bound': n * k, 'solution_dim': int(m),
                    'note': ('Hurwitz minor Delta_{} vanishes identically on the '
                             'CONSTRAINED solution set (checked on {} random lines '
                             'at {} points each, above the degree bound {}), so by '
                             'Routh-Hurwitz no admissible member is Hurwitz'
                             .format(k, n_lines, n * k + 2, n * k))}
    return None


# constrained_decide(...): linear_oracle.decide, run inside the constrained
# subspaces. Same three-valued contract, same witness verification, plus a
# per-constraint report on whatever it returns.
#
# Pipeline, and where each verdict may come from:
#   graph gate    — a combinatorial constraint excludes the graph outright.
#                   Returned as INVALID with propagates=False: sound for THIS
#                   graph, but not monotone in the subgraph order, so the
#                   search must not propagate it (see the module docstring).
#   structural    — linear_oracle's certificates on the masked graph. Computed
#                   over the UNCONSTRAINED admissible set, hence still sound
#                   (a smaller family cannot contain a Hurwitz member if the
#                   larger one does not) and unchanged in cost.
#   trivial       — the constraints left no free generator at all, so the only
#                   admissible point is (0, 0), whose A = 0 is not Hurwitz.
#                   INVALID, and this is the verdict that makes an
#                   over-constrained question answer itself rather than
#                   silently returning UNDECIDED.
#   fast path     — fixed dissipator (vacuum, or whatever a reservoir
#                   constraint pins) and solve for G alone. Skipped when the
#                   pinned Upsilon violates a Y-constraint.
#   joint path    — free Upsilon inside the constrained S_Upsilon.
#   constrained
#   certificates  — dark subspace and Routh-Hurwitz over the CONSTRAINED
#                   solution set, both strictly sharper than their
#                   unconstrained counterparts.
#   gap max       — Move 1's ascent before conceding UNDECIDED.
#
# VALID is only ever returned with a fully verified witness (_witness checks
# the residual, PSD, Hurwitz and the forward Lyapunov solve) that ALSO
# satisfies every constraint — the check is repeated on the finished witness
# rather than trusted from the filter.
def constrained_decide(
    triu,
    V: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    constraints=None,
    include_detunings: bool = True,
    allow_phases: bool = True,
    coupled_drains: bool = False,
    joint: bool = True,
    num_samples: int = 32,
    seed: Optional[int] = None,
    rtol: float = RESIDUAL_RTOL_DEFAULT,
    margin_tol: float = MARGIN_TOL_DEFAULT,
    include_solution_set: bool = True,
    gap_effort: int = 8,
    use_routh: bool = True,
) -> Dict:
    cset = as_constraint_set(constraints)
    num_modes = len(node_types)
    ctx = ConstraintContext(num_modes, target_mode_ids, node_types, V)
    dim = ctx.dim

    if seed is None:
        seed = int(abs(hash(tuple(int(x) for x in np.asarray(triu)))) % (2 ** 31))

    out = {'triu': np.asarray(triu), 'seed': seed, 'path': None,
           'constraints': [c.name for c in cset], 'propagates': True}

    # ---- combinatorial gate ------------------------------------------------
    ok, offender = cset.admits_graph(triu, ctx)
    if not ok:
        out['verdict'] = INVALID
        out['propagates'] = False
        out['path'] = 'constraint_gate'
        out['certificate'] = {
            'kind': 'excluded_by_constraint', 'constraint': offender.name,
            'note': ('the graph itself is excluded by a combinatorial constraint; '
                     'this verdict is NOT monotone in the subgraph order and must '
                     'not be propagated by the search')}
        return out

    triu_eff, h_basis, d_basis = constrained_bases(
        triu, ctx, cset, include_detunings=include_detunings,
        allow_phases=allow_phases, coupled_drains=coupled_drains)
    out['triu_masked'] = triu_eff
    out['n_hamiltonian_dof'] = len(h_basis)
    out['n_dissipation_dof'] = len(d_basis)

    # ---- structural certificates (unconstrained family; still sound) -------
    cert = structural_certificate(triu_eff, V, target_mode_ids, node_types,
                                  coupled_drains=coupled_drains,
                                  include_solution_set=include_solution_set)
    if cert is not None:
        out['verdict'] = INVALID
        out['certificate'] = cert
        out['path'] = 'structural'
        return out

    # ---- the constraints left nothing to solve with -----------------------
    if not h_basis and not d_basis:
        out['verdict'] = INVALID
        out['path'] = 'constraint_trivial'
        out['certificate'] = {
            'kind': 'empty_admissible_subspace',
            'note': ('the constraints leave no admissible generator, so the only '
                     'point is (G, Upsilon) = (0, 0), whose A = 0 is not Hurwitz')}
        return out

    # ---- fast path: pinned dissipator, solve for G alone -------------------
    Y_fix = cset.fixed_Y(ctx)
    if Y_fix is None:
        Y_fix = vacuum_dissipation(ctx.aux_ids, num_modes, rate=1.0)
    if h_basis and cset.accepts_fixed_Y(Y_fix, ctx):
        sol_fast = solve_stationarity(V, h_basis, None, Y_fix, rtol=rtol)
        out['residual_fixed_upsilon'] = sol_fast['residual']
        if sol_fast['feasible']:
            ok, z, margin = constrained_attractivity_filter(
                sol_fast, h_basis, None, Y_fix, ctx, cset,
                num_samples=num_samples, seed=seed, margin_tol=margin_tol,
                require_psd=False)
            if ok:
                G, Y = _assemble(z, h_basis, None, Y_fix, dim)
                out.update(_witness(G, Y, V, margin, 'constrained_fixed_upsilon'))
                out['constraint_report'] = cset.report(G, Y, ctx)
                return out
    else:
        out['fast_path'] = 'skipped: pinned Upsilon violates a constraint'

    if not joint:
        out['verdict'] = UNDECIDED
        out['reason'] = ('no attractive solution with the pinned dissipator; '
                         'joint=False so the graph itself was not decided')
        return out

    # ---- joint path: free Upsilon inside the constrained subspace ----------
    sol = solve_stationarity(V, h_basis, d_basis, None, rtol=rtol)
    out['residual_joint'] = sol['residual']
    out['solution_dim'] = int(sol['nullspace'].shape[1])
    out['path'] = 'constrained_joint'

    # A homogeneous system whose only solution is zero: same argument as the
    # empty-basis case, and it is the usual way an over-tight constraint set
    # announces itself.
    if sol['nullspace'].shape[1] == 0:
        out['verdict'] = INVALID
        out['path'] = 'constraint_trivial'
        out['certificate'] = {
            'kind': 'trivial_solution_set',
            'note': ('the only constrained solution of (**) is (G, Upsilon) = '
                     '(0, 0), whose A = 0 is not Hurwitz')}
        return out

    ok, z, margin = constrained_attractivity_filter(
        sol, h_basis, d_basis, None, ctx, cset, num_samples=num_samples,
        seed=seed, margin_tol=margin_tol, require_psd=True)
    if ok:
        G, Y = _assemble(z, h_basis, d_basis, None, dim)
        out.update(_witness(G, Y, V, margin, 'constrained_joint'))
        out['constraint_report'] = cset.report(G, Y, ctx)
        return out

    # ---- certificates over the CONSTRAINED solution set --------------------
    if include_solution_set:
        d_dark = constrained_dark_subspace(sol, h_basis, d_basis, dim)
        if d_dark > 0:
            out['verdict'] = INVALID
            out['path'] = 'constrained_dark'
            out['certificate'] = {
                'kind': 'constrained_solution_set_dark', 'dim': d_dark,
                'frame_dependent': True,
                'note': ('every solution of (**) inside the constrained subspace '
                         'shares a {}-dimensional dark subspace, so none of them '
                         'is attractive'.format(d_dark))}
            return out

    # ---- Move 1: gap maximisation before conceding -------------------------
    if gap_effort and gap_effort > 0:
        ok, z, margin = constrained_attractivity_filter(
            sol, h_basis, d_basis, None, ctx, cset,
            num_samples=num_samples * int(gap_effort), seed=seed + 1,
            margin_tol=margin_tol, require_psd=True,
            n_polish=2 + int(gap_effort), maximise=True)
        out['gap_star'] = margin
        if ok:
            G, Y = _assemble(z, h_basis, d_basis, None, dim)
            out.update(_witness(G, Y, V, margin, 'constrained_gap_max'))
            out['constraint_report'] = cset.report(G, Y, ctx)
            return out

    # ---- Move 3's universal half, on the constrained family ----------------
    if use_routh:
        rc = constrained_routh_certificate(sol, h_basis, d_basis, dim, seed=seed)
        if rc is not None:
            out['verdict'] = INVALID
            out['path'] = 'constrained_routh'
            out['certificate'] = rc
            return out

    out['verdict'] = UNDECIDED
    out['best_margin'] = margin
    out['reason'] = ('constrained solutions exist but no strictly Hurwitz, PSD, '
                     'constraint-satisfying member was found, and no certificate '
                     'applies; raise gap_effort/num_samples')
    return out


# ───────────────────────────────────────────────────────────────────────────
# sweeps and search integration
# ───────────────────────────────────────────────────────────────────────────
# scan_constraint(...): re-decide one graph across a family of constraint
# sets, which is how a quantitative statement gets made.
#
# The two sweeps worth running:
#   [SqueezedReservoir(r) for r in grid]      -> minimum reservoir squeezing
#   [resource_frontier(g, gamma) for ...]     -> the gap-vs-coupling Pareto
#                                                curve behind each VALID bit
def scan_constraint(triu, V, target_mode_ids, node_types,
                    constraint_sets: Iterable, labels=None, **kwargs) -> List[Dict]:
    sets = list(constraint_sets)
    labels = list(labels) if labels is not None else [
        ', '.join(c.name for c in as_constraint_set(s)) for s in sets]
    rows = []
    for lbl, cs in zip(labels, sets):
        out = constrained_decide(triu, V, target_mode_ids, node_types,
                                 constraints=cs, **kwargs)
        # The gap is only reported for a VALID verdict: on the other two it is
        # whatever the failed ascent last saw, which is not a figure of merit.
        rows.append({'label': lbl, 'verdict': out['verdict'],
                     'gap': out.get('gap') if out['verdict'] == VALID else None,
                     'reservoir': out.get('reservoir_kind'),
                     'squeezing': out.get('reservoir_squeezing'),
                     'channels': out.get('upsilon_rank'), 'info': out})
    return rows


def format_scan(rows: List[Dict]) -> str:
    out = [f'{"constraints":<40} {"verdict":<10} {"gap":>10} {"chan":>5}  reservoir']
    for r in rows:
        # A non-finite gap is the filter reporting "no admissible point was
        # even evaluable", not a number worth printing.
        g = ('' if r['gap'] is None or not np.isfinite(r['gap'])
             else f'{r["gap"]:.4g}')
        ch = '' if r['channels'] is None else str(r['channels'])
        res = r['reservoir'] or ''
        if r['squeezing']:
            res += f' (s={r["squeezing"]:.3f})'
        out.append(f'{r["label"][:40]:<40} {r["verdict"]:<10} {g:>10} {ch:>5}  {res}')
    return '\n'.join(out)


# ConstrainedSearch: certified_search.CertifiedSearch with the constrained
# oracle substituted in, so a whole sweep runs under the constraints.
#
# The propagation rules survive every LINEAR constraint unchanged: they rest
# on "shrinking the admissible subspace can only shrink the solution set",
# and intersecting with a fixed subspace commutes with adding graph edges. A
# combinatorial constraint (DegreeBound) breaks that, so its verdicts are
# marked propagates=False and are converted to UNDECIDED here rather than
# being allowed to seed the invalid set.
class ConstrainedSearch:
    def __init__(self, *args, constraints=None, **kwargs):
        from reservoir_engineering.certified_search import CertifiedSearch
        self._search = CertifiedSearch(*args, **kwargs)
        self.cset = as_constraint_set(constraints)
        self._ctx = ConstraintContext(self._search.num_modes,
                                      self._search.target_mode_ids,
                                      self._search.node_types, self._search.V)
        self._search.oracle = self._oracle           # bound override

    def _oracle(self, triu) -> Dict:
        s = self._search
        key = tuple(int(x) for x in np.asarray(triu))
        if key not in s.cache:
            kw = {k: v for k, v in s._oracle_kwargs.items() if k != 'use_sdp'}
            out = constrained_decide(np.array(key), s.V, s.target_mode_ids,
                                     s.node_types, constraints=self.cset, **kw)
            if out.get('propagates') is False:
                out = dict(out, verdict=UNDECIDED,
                           reason='excluded by a non-monotone constraint; not '
                                  'propagated (see solver_constraints)')
            s.cache[key] = out
        return s.cache[key]

    def __getattr__(self, item):
        return getattr(self._search, item)


# describe(constraints): the menu as it will actually be applied, tier by
# tier — useful to print before a long sweep so the run records what it asked.
def describe(constraints) -> str:
    cset = as_constraint_set(constraints)
    if not len(cset):
        return 'no constraints (free search)'
    order = {'linear': 0, 'convex': 1, 'nonconvex': 2, 'graph': 3}
    rows = sorted(cset.items, key=lambda c: order.get(c.kind, 9))
    width = max(len(c.name) for c in rows)
    note = {'linear': 'subspace restriction, one sound solve',
            'convex': 'penalty + acceptance on the witness',
            'nonconvex': 'hard filter over the Move-1 interior',
            'graph': 'combinatorial gate, does NOT propagate'}
    return '\n'.join(f'  {c.name:<{width}}  {c.kind:<10} {note.get(c.kind, "")}'
                     for c in rows)


if __name__ == '__main__':
    # Smoke test on the two-mode-squeezed target with one drain: the fully
    # connected graph, decided free and then under each toggle.
    from reservoir_engineering.linear_oracle import complete_covariance
    from reservoir_engineering.targets import two_mode_squeezed

    n, tgt = 3, [0, 1]
    node_types = ['cavity'] * n
    V = complete_covariance(two_mode_squeezed(0.5), tgt, n)
    triu = np.full(n * (n + 1) // 2, 0, dtype=int)
    rows_i, cols_i = np.triu_indices(n)
    for k, (i, j) in enumerate(zip(rows_i, cols_i)):
        if i != j:
            triu[k] = 4                     # BS + TMS on every pair

    sets = [ConstraintSet(),
            ConstraintSet([passive_hamiltonian()]),
            ConstraintSet([VacuumReservoir()]),
            ConstraintSet([passive_hamiltonian(), VacuumReservoir()]),
            ConstraintSet([real_couplings()]),
            ConstraintSet([zero_detunings()]),
            ConstraintSet([ReciprocalDissipation()])]
    labels = ['free', 'passive H', 'vacuum drain', 'passive + vacuum',
              'real couplings', 'no detunings', 'reciprocal']
    print(format_scan(scan_constraint(triu, V, tgt, node_types, sets, labels)))
