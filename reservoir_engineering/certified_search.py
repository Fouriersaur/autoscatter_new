"""
certified_search.py
===================
The discrete search of state_stabilization_algorithm.md §5, wrapped around
the certifying oracle in linear_oracle.py.

MIRRORS covariance_optimizer.CovarianceOptimizer's outer loop, and keeps its
graph encoding (topology_search.py) unchanged. What changes is the inner
loop and, because of it, what the verdicts mean:

  covariance_optimizer          certified_search
  --------------------          ----------------
  Stage 1 unit-cooperativity    structural certificate (exact, provable)
    stability pre-filter
  Stage 2 multi-restart         one global linear solve + attractivity
    L-BFGS-B on log-couplings     filter (linear_oracle.decide)
  success = loss < tol          VALID = verified witness (residual ~1e-15,
                                  Upsilon >= 0, A Hurwitz, forward Lyapunov
                                  solve reproduces the target)
  failure = "gave up"           INVALID = certificate, or UNDECIDED
  2 verdicts                    3 verdicts

The three-valued verdict is the load-bearing difference. The old optimiser
could not distinguish "this graph is impossible" from "30 restarts of
L-BFGS-B did not find it", so it could never prune soundly on a failure —
which is exactly the false-negative risk AUTOSCATTER mitigates by rerunning.
Here INVALID is only ever issued from a proof, so it may be propagated to
the whole down-set; UNDECIDED is issued when the oracle genuinely does not
know, and propagates NOWHERE. The search stays sound: a graph can be lost
only if it is provably impossible.

INPUT IS A COVARIANCE MATRIX ONLY. There is no target_predicate — see
linear_oracle's module docstring for why that is structural (a scalar
functional does not pin V, so the stationarity equation stops being linear
and the whole method's exactness goes with it), not a missing feature.

Cost note. The oracle is one SVD per graph on a matrix of size
(2n)^2 x O(n^2) — milliseconds, with no restarts and no Lyapunov solve in
an inner loop. The 4-mode Cayley cluster search that
tests/cayley_cluster_rediscovery.py had to abandon (33,554,432 graphs, days
of Stage-2 solves, hence its hand-curated topology list) is worth
re-attempting on this oracle.
"""

import numpy as np
from typing import List, Dict, Optional

from reservoir_engineering.topology_search import (
    NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING, PARAMETRIC,
    BEAMSPLITTER_AND_TWO_MODE_SQUEEZING,
    _is_subgraph_slot, check_if_subgraph_triu, calc_number_of_possibilities)
from reservoir_engineering.linear_oracle import (
    VALID, INVALID, UNDECIDED, complete_covariance, decide)

# Per-slot alphabets, as in covariance_optimizer.prepare_all_possible_combinations.
_DIAG_VALUES    = [NO_COUPLING, PARAMETRIC]
_OFFDIAG_VALUES = [NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
                   BEAMSPLITTER_AND_TWO_MODE_SQUEEZING]


# Neighbours of a graph under a single-slot edit, in one direction only.
# 'prune' moves to strict subgraphs (one slot replaced by a value it
# contains), 'grow' to strict supergraphs. Uses topology_search's
# type-aware containment, NOT integer <=: BS(1) is a subgraph of BS+TMS(4)
# but never of TMS(2), since the two are different block structures rather
# than degrees of the same thing.
def _neighbours(triu, num_modes: int, direction: str):
    rows, cols = np.triu_indices(num_modes)
    triu = list(int(x) for x in triu)
    for k, (i, j) in enumerate(zip(rows, cols)):
        alphabet = _DIAG_VALUES if i == j else _OFFDIAG_VALUES
        for v in alphabet:
            if v == triu[k]:
                continue
            ok = (_is_subgraph_slot(v, triu[k]) if direction == 'prune'
                  else _is_subgraph_slot(triu[k], v))
            if ok:
                nb = list(triu)
                nb[k] = v
                yield np.array(nb, dtype=int)


# The root for the prune direction: the MAXIMUM of the subgraph lattice —
# every off-diagonal pair BS+TMS *and* every diagonal slot PARAMETRIC.
#
# Deliberately NOT topology_search.TopologyGraph.fully_connected, which
# leaves the diagonal at NO_COUPLING. That matters here in a way it does not
# for covariance_optimizer: the old search enumerates every graph explicitly
# and only uses the fully-connected graph as a feasibility probe, whereas a
# prune traversal can reach only SUBgraphs of its root. With NO_COUPLING on
# the diagonal, PARAMETRIC is not reachable by pruning (0 is contained in 3,
# not the reverse), so every scheme using a self-squeezing edge would be
# invisible top-down while the grow pass found it — presenting as a
# bidirectional disagreement that looks like a propagation bug but is really
# a root that is not the lattice maximum.
def _fully_connected(num_modes: int) -> np.ndarray:
    rows, cols = np.triu_indices(num_modes)
    return np.array([PARAMETRIC if i == j else BEAMSPLITTER_AND_TWO_MODE_SQUEEZING
                     for i, j in zip(rows, cols)], dtype=int)


def _empty(num_modes: int) -> np.ndarray:
    return np.zeros(num_modes * (num_modes + 1) // 2, dtype=int)


class CertifiedSearch:

    # sigma_target: (2N,2N) covariance for the N target modes. The ONLY
    #   target input — see the module docstring.
    # node_types / target_mode_ids: as in CovarianceOptimizer. Every mode not
    #   in target_mode_ids is auxiliary and carries the dissipation; the
    #   signal modes are assumed lossless (Gamma_signal = 0), which is what
    #   makes the state completion of §2.1 exact.
    # aux_squeezing: drain states, see linear_oracle.complete_covariance.
    #   None (default) fixes the gauge with unsqueezed drains — complete,
    #   but passive Zippilli-Vitali schemes then appear as their gauge image
    #   with BS+TMS on the drain edges rather than BS.
    def __init__(
        self,
        sigma_target: np.ndarray,
        target_mode_ids: List[int],
        node_types: List[str],
        aux_squeezing=None,
        num_samples: int = 32,
        include_detunings: bool = True,
        allow_phases: bool = True,
        coupled_drains: bool = False,
        use_sdp: bool = False,
        verbosity: int = 1,
    ):
        self.node_types      = list(node_types)
        self.num_modes       = len(node_types)
        self.target_mode_ids = list(target_mode_ids)
        self.sigma_target    = np.asarray(sigma_target, dtype=float)
        self.verbosity       = verbosity

        # Raises on a mixed target rather than silently linearising
        # something bilinear — see linear_oracle.complete_covariance.
        self.V = complete_covariance(self.sigma_target, self.target_mode_ids,
                                      self.num_modes, aux_squeezing=aux_squeezing)

        self._oracle_kwargs = dict(
            include_detunings=include_detunings, allow_phases=allow_phases,
            coupled_drains=coupled_drains, use_sdp=use_sdp,
            num_samples=num_samples)

        # Memoized verdicts. §5(ii): this caches the oracle's OWN past
        # computation, never prior knowledge — no scheme is seeded, so
        # rediscovering Kronwald or Zippilli-Vitali stays an output of the
        # search rather than an input to it.
        self.cache: Dict[tuple, Dict] = {}

    # Memoizing oracle call. The per-graph seed inside linear_oracle.decide
    # is derived from the graph itself, so a verdict is a deterministic,
    # traversal-order-independent function of the graph — which is what
    # makes the §5(iii) bidirectional assertion meaningful.
    def oracle(self, triu) -> Dict:
        key = tuple(int(x) for x in np.asarray(triu))
        if key not in self.cache:
            self.cache[key] = decide(np.array(key), self.V, self.target_mode_ids,
                                      self.node_types, **self._oracle_kwargs)
        return self.cache[key]

    # One directional pass (§5's shared, direction-dual BFS body).
    #   prune: descend from the fully connected root through valid territory
    #          until it meets the invalid boundary
    #   grow:  ascend from the empty graph through invalid territory until
    #          it meets the valid boundary
    # Both converge on the same object — the minimal valid graphs — which is
    # why their agreement is a real check and not a tautology of shared code.
    def run(self, direction: str = 'prune') -> Dict:
        if direction not in ('prune', 'grow'):
            raise ValueError("direction must be 'prune' or 'grow'")

        root = _fully_connected(self.num_modes) if direction == 'prune' else _empty(self.num_modes)
        valid, invalid, undecided = [], [], []
        seen = set()
        frontier = [root]

        while frontier:
            triu = frontier.pop()
            key = tuple(int(x) for x in triu)
            if key in seen:
                continue
            seen.add(key)

            # Already settled by propagation from a previously decided
            # graph? Skip the oracle call — but NOT the traversal. A graph
            # settled by propagation must keep the frontier moving exactly
            # as an explicitly-decided one would, or the search silently
            # loses everything beyond it. (Getting this wrong drops valid
            # graphs: in the grow direction an invalid-by-propagation graph
            # that stops instead of ascending strands its whole up-set,
            # which is how the minimal two-drain-edge scheme for a
            # two-mode-squeezed target went missing.)
            if valid and check_if_subgraph_triu([triu], valid):
                # some known-valid graph is a subgraph of this one, so this
                # one is valid too; prune keeps descending, grow stops.
                if direction == 'prune':
                    frontier.extend(_neighbours(triu, self.num_modes, 'prune'))
                continue
            if invalid and check_if_subgraph_triu(invalid, [triu]):
                # this graph is a subgraph of a known-invalid one, so it is
                # invalid too; grow keeps ascending, prune stops.
                if direction == 'grow':
                    frontier.extend(_neighbours(triu, self.num_modes, 'grow'))
                continue

            v = self.oracle(triu)['verdict']

            if v == VALID:
                valid.append(triu)
                if direction == 'prune':
                    frontier.extend(_neighbours(triu, self.num_modes, 'prune'))
                # grow: the whole up-set is valid, stop ascending
            elif v == INVALID:
                invalid.append(triu)
                if direction == 'grow':
                    frontier.extend(_neighbours(triu, self.num_modes, 'grow'))
                # prune: the whole down-set is invalid, stop descending
            else:
                # UNDECIDED propagates NOTHING (§4C). The up-/down-set is
                # unresolved, so keep exploring past it rather than pruning.
                undecided.append(triu)
                frontier.extend(_neighbours(triu, self.num_modes, direction))

        if self.verbosity:
            print(f'  [{direction}] visited {len(seen)}  valid {len(valid)}  '
                  f'invalid {len(invalid)}  undecided {len(undecided)}')
        return {'valid': valid, 'invalid': invalid, 'undecided': undecided,
                'visited': len(seen), 'direction': direction}

    # closure_partition(pass_result): the verdict a pass IMPLIES for every
    # graph, not just the ones it happened to visit — VALID if some
    # explicitly-valid graph is a subgraph of it, INVALID if it is a
    # subgraph of an explicitly-invalid one, UNDECIDED otherwise.
    #
    # This is the object the §5(iii) agreement test must compare, and it is
    # an AMENDMENT to the doc, which compares the raw Valid/Invalid files.
    # Those files are not comparable once the oracle is three-valued: prune
    # descends until it hits INVALID and grow ascends until it hits VALID,
    # so an UNDECIDED band between the two frontiers is traversed from
    # opposite sides and each pass records a different slice of it. The raw
    # lists then differ by COVERAGE while agreeing on every graph both
    # actually decided — which they must, since verdicts are memoized and
    # deterministic. Comparing closures tests what the doc meant: that the
    # two traversals imply the same partition of the whole lattice.
    def closure_partition(self, pass_result: Dict) -> Dict[tuple, str]:
        import itertools
        rows, cols = np.triu_indices(self.num_modes)
        alphabets = [(_DIAG_VALUES if i == j else _OFFDIAG_VALUES)
                     for i, j in zip(rows, cols)]
        valid, invalid = pass_result['valid'], pass_result['invalid']
        part = {}
        for combo in itertools.product(*alphabets):
            t = np.array(combo, dtype=int)
            if valid and check_if_subgraph_triu([t], valid):
                part[combo] = VALID
            elif invalid and check_if_subgraph_triu(invalid, [t]):
                part[combo] = INVALID
            else:
                part[combo] = UNDECIDED
        return part

    # §5(iii): run both directions and compare their implied partitions.
    # Under a sound oracle they MUST coincide, because each graph's verdict
    # is a deterministic function of the graph and cannot depend on
    # traversal order. A disagreement is therefore an implementation bug —
    # a wrong tolerance, a broken neighbour relation, a faulty up-/down-set
    # propagation — and never "one direction lost a scheme". That is what
    # upgrades this from a sanity test to the primary integrity guarantee.
    #
    # compare_closures=False falls back to comparing the raw files, which is
    # cheap but only meaningful when the oracle decided every graph it saw.
    def run_bidirectional(self, compare_closures: bool = True) -> Dict:
        td = self.run('prune')
        bu = self.run('grow')

        def keyset(lst):
            return {tuple(int(x) for x in t) for t in lst}

        if compare_closures:
            ptd, pbu = self.closure_partition(td), self.closure_partition(bu)
            disagree = [g for g in ptd if ptd[g] != pbu[g]]
            agree_valid = all(ptd[g] != VALID and pbu[g] != VALID
                              for g in disagree) or not disagree
            agree_invalid = all(ptd[g] != INVALID and pbu[g] != INVALID
                                for g in disagree) or not disagree
            agree = not disagree
        else:
            ptd = pbu = None
            disagree = []
            agree_valid   = keyset(td['valid'])   == keyset(bu['valid'])
            agree_invalid = keyset(td['invalid']) == keyset(bu['invalid'])
            agree = agree_valid and agree_invalid

        mv_td = keyset(self.minimal_valid(td['valid']))
        mv_bu = keyset(self.minimal_valid(bu['valid']))

        if self.verbosity:
            print(f'  bidirectional: closures agree={agree} '
                  f'({len(disagree)} differing graphs)  '
                  f'minimal-valid sets agree={mv_td == mv_bu}')
        return {'top_down': td, 'bottom_up': bu,
                'agree': agree, 'agree_valid': agree_valid,
                'agree_invalid': agree_invalid, 'disagreeing_graphs': disagree,
                'partition_top_down': ptd, 'partition_bottom_up': pbu,
                'minimal_valid_agree': mv_td == mv_bu,
                'minimal_valid': self.minimal_valid(bu['valid'] + td['valid'])}

    # Irreducible valid graphs: drop any valid graph that has another valid
    # graph as a subgraph (it is superseded by the simpler one).
    def minimal_valid(self, valid=None) -> List[np.ndarray]:
        if valid is None:
            valid = [np.array(k) for k, r in self.cache.items() if r['verdict'] == VALID]
        uniq, seen = [], set()
        for t in valid:
            k = tuple(int(x) for x in t)
            if k not in seen:
                seen.add(k)
                uniq.append(np.array(k))
        kept = []
        for i, t in enumerate(uniq):
            others = uniq[:i] + uniq[i + 1:]
            if others and check_if_subgraph_triu([t], others):
                continue
            kept.append(t)
        return sorted(kept, key=lambda t: int(np.sum(t)))


# Exhaustive sweep — every graph, no pruning. Slower than CertifiedSearch on
# large n but the reference implementation the BFS is checked against: with
# a deterministic oracle the two must produce identical VALID sets, so any
# difference localises a bug in the propagation rather than in the oracle.
def sweep_all(
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    **kwargs,
) -> Dict:
    import itertools
    n = len(node_types)
    rows, cols = np.triu_indices(n)
    alphabets = [(_DIAG_VALUES if i == j else _OFFDIAG_VALUES) for i, j in zip(rows, cols)]

    search = CertifiedSearch(sigma_target, target_mode_ids, node_types,
                             verbosity=0, **kwargs)
    out = {VALID: [], INVALID: [], UNDECIDED: []}
    for combo in itertools.product(*alphabets):
        t = np.array(combo, dtype=int)
        out[search.oracle(t)['verdict']].append(t)
    return {'valid': out[VALID], 'invalid': out[INVALID], 'undecided': out[UNDECIDED],
            'search': search, 'num_graphs': calc_number_of_possibilities(node_types)}


# Human-readable edge list for a triu_array.
def describe(triu, num_modes: int) -> str:
    labels = {BEAMSPLITTER: 'BS', TWO_MODE_SQUEEZING: 'TMS', PARAMETRIC: 'PAR',
              BEAMSPLITTER_AND_TWO_MODE_SQUEEZING: 'BS+TMS'}
    rows, cols = np.triu_indices(num_modes)
    parts = [(f'({i}){labels[int(v)]}' if i == j else f'({i},{j}){labels[int(v)]}')
             for (i, j), v in zip(zip(rows, cols), triu) if int(v) != NO_COUPLING]
    return ', '.join(parts) if parts else '(empty graph)'
