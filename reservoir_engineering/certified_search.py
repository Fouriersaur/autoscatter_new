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
    VALID, INVALID, UNDECIDED, complete_covariance, decide, optimise_aux_state)

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
        auto_reservoir: bool = False,
        gap_effort: int = 8,
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

        # gap_effort: §8 Move 1's budget multiplier for the spectral-abscissa
        # ascent that runs ONLY on graphs about to be filed UNDECIDED. Raising
        # it shrinks the undecided stratum at linear cost in that stratum
        # alone; 0 disables the move and reproduces the plain feasibility
        # oracle.
        self._oracle_kwargs = dict(
            include_detunings=include_detunings, allow_phases=allow_phases,
            coupled_drains=coupled_drains, use_sdp=use_sdp,
            num_samples=num_samples, gap_effort=gap_effort)

        # Memoized verdicts. §5(ii): this caches the oracle's OWN past
        # computation, never prior knowledge — no scheme is seeded, so
        # rediscovering Kronwald or Zippilli-Vitali stays an output of the
        # search rather than an input to it.
        self.cache: Dict[tuple, Dict] = {}

        # Graphs with a PROPAGATING verdict, i.e. one that settles other
        # graphs without an oracle call. Only VALID and INVALID qualify.
        self._valid_seen: List[np.ndarray] = []
        self._invalid_seen: List[np.ndarray] = []

        # auto_reservoir: decide the drain state per graph instead of using
        # the fixed one baked into self.V. Costs ~10x per graph but finds
        # squeezed-reservoir (Zippilli-Vitali) schemes in the same sweep as
        # vacuum ones, with nothing assumed about which frame to look in.
        self.auto_reservoir = auto_reservoir
        self._sigma_target = self.sigma_target

    # Memoizing oracle call. The per-graph seed inside linear_oracle.decide
    # is derived from the graph itself, so a verdict is a deterministic,
    # traversal-order-independent function of the graph — which is what
    # makes the §5(iii) bidirectional assertion meaningful.
    def oracle(self, triu) -> Dict:
        key = tuple(int(x) for x in np.asarray(triu))
        if key not in self.cache:
            if self.auto_reservoir:
                kw = {k: v for k, v in self._oracle_kwargs.items() if k != 'use_sdp'}
                self.cache[key] = optimise_aux_state(
                    np.array(key), self._sigma_target, self.target_mode_ids,
                    self.node_types, **kw)
            else:
                self.cache[key] = decide(np.array(key), self.V, self.target_mode_ids,
                                          self.node_types, **self._oracle_kwargs)
        return self.cache[key]

    # verdict_of(triu): the verdict for a graph, using PROPAGATION first and
    # the oracle only when nothing already implies an answer.
    #
    # This is the mechanism that must never be bypassed: an INVALID verdict
    # holds for every SUBgraph (shrinking the admissible subspace can only
    # shrink the solution set, so if no Hurwitz member existed before, none
    # exists now), and a VALID verdict holds for every SUPERgraph (the
    # witness embeds unchanged). Both directions stay sound with
    # auto_reservoir on, because the drain state is part of the witness and
    # travels with it. UNDECIDED implies nothing and never propagates.
    #
    # Returns (verdict, info, used_oracle) so callers can report how much
    # the propagation actually saved.
    def verdict_of(self, triu):
        key = tuple(int(x) for x in np.asarray(triu))
        if key in self.cache:
            return self.cache[key]['verdict'], self.cache[key], False
        if self._valid_seen and check_if_subgraph_triu([np.asarray(triu)], self._valid_seen):
            return VALID, None, False          # a known-valid graph sits inside this one
        if self._invalid_seen and check_if_subgraph_triu(self._invalid_seen, [np.asarray(triu)]):
            return INVALID, None, False        # this graph sits inside a known-invalid one
        info = self.oracle(triu)
        if info['verdict'] == VALID:
            self._valid_seen.append(np.asarray(triu))
        elif info['verdict'] == INVALID:
            self._invalid_seen.append(np.asarray(triu))
        return info['verdict'], info, True

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
                # §5: also file the REDUCED graph, i.e. the edges the
                # witness actually uses. Filing it makes its up-set valid
                # too, which is both a stronger result and better pruning
                # than filing only the graph that happened to be tested.
                red = reduce_witness(triu, self.oracle(triu), self.num_modes)
                if not np.array_equal(red, triu) and int(np.sum(red)) > 0:
                    if not (valid and check_if_subgraph_triu([red], valid)):
                        valid.append(red)
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

    # §5(iii): run both directions and assert they never CONTRADICT.
    #
    # The invariant is deliberately the weaker one. With a two-valued oracle
    # the two passes would have to produce identical partitions, and the doc
    # originally asserted exactly that. Once the verdict is three-valued that
    # assertion is WRONG, and would fire on correct runs: UNDECIDED
    # propagates neither up nor down (§4C), so prune descends until it meets
    # INVALID while grow ascends until it meets VALID, and each pass leaves a
    # DIFFERENT shadow of the undecided band unresolved. Identical partitions
    # are recovered exactly when UNDECIDED is empty, not before.
    #
    # What must hold unconditionally, and is what this checks:
    #     no graph is VALID in one pass and INVALID in the other.
    # A verdict is a deterministic function of the graph (the per-graph seed
    # inside decide makes it so), and both propagation rules are sound, so a
    # direct contradiction can only be an IMPLEMENTATION bug — a wrong
    # tolerance, a broken neighbour relation, a faulty up-/down-set
    # propagation. It is a regression test ON soundness, not a completeness
    # proof.
    #
    # 'partitions_identical' is still reported, but as a DIAGNOSTIC: it is the
    # signal that the undecided stratum has been emptied, not a pass/fail.
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
            contradictions = [g for g in disagree
                              if {ptd[g], pbu[g]} == {VALID, INVALID}]
            identical = not disagree
        else:
            ptd = pbu = None
            disagree = []
            vtd, vbu = keyset(td['valid']), keyset(bu['valid'])
            itd, ibu = keyset(td['invalid']), keyset(bu['invalid'])
            contradictions = sorted((vtd & ibu) | (itd & vbu))
            identical = (vtd == vbu) and (itd == ibu)

        agree = not contradictions          # THE assertion (§5(iii))

        mv_td = keyset(self.minimal_valid(td['valid']))
        mv_bu = keyset(self.minimal_valid(bu['valid']))

        if self.verbosity:
            print(f'  bidirectional: no VALID/INVALID contradiction={agree} '
                  f'({len(contradictions)} contradictions, '
                  f'{len(disagree)} graphs differing only by UNDECIDED coverage)  '
                  f'minimal-valid sets agree={mv_td == mv_bu}')
        return {'top_down': td, 'bottom_up': bu,
                'agree': agree, 'contradictions': contradictions,
                'partitions_identical': identical,
                'disagreeing_graphs': disagree,
                'partition_top_down': ptd, 'partition_bottom_up': pbu,
                'minimal_valid_agree': mv_td == mv_bu,
                'minimal_valid': self.minimal_valid(bu['valid'] + td['valid'])}

    # Class-minimal valid schemes, with each graph's OWN reservoir resolved.
    #
    # Fixes two distinct losses that both hide physically-different schemes.
    #
    # (1) PROPAGATION INHERITS A WITNESS, NOT A RESERVOIR. When a supergraph
    #     is settled by propagation, the witness that travels is the
    #     SUBGRAPH's — including its drain state. So a graph that would work
    #     with a plain vacuum drain can be filed as "valid with a squeezed
    #     drain" and never evaluated on its own. Verified on the two-mode
    #     squeezed target: the Woolley-Clerk graph (drain coupled to each
    #     mechanical mode by beamsplitter AND two-mode-squeezing) is never
    #     handed to the oracle at all, because the two-beamsplitter graph is
    #     its subgraph; called directly it returns reservoir = VACUUM, which
    #     is the whole point of that scheme.
    #
    # (2) IRREDUCIBILITY IGNORED THE RESERVOIR. Comparing triu arrays alone
    #     drops a supergraph whenever any valid subgraph exists — even when
    #     the subgraph needs a squeezed source and the supergraph does not.
    #     Neither dominates: one trades coupling complexity for not needing
    #     a squeezed reservoir. Minimality is only meaningful WITHIN a
    #     reservoir class.
    #
    # The traversal exploits one asymmetry to stay cheap. If a kept
    # VACUUM-class graph is already a subgraph of the candidate, the
    # candidate is vacuum-valid too (the witness embeds) and cannot be
    # minimal in either class — skip it with no oracle call. If only a
    # SQUEEZED-class graph is a subgraph, the candidate may still turn out
    # vacuum-valid, so it must be evaluated. That is exactly the case that
    # was being lost.
    def minimal_valid_by_reservoir(self, valid_seen=None, verbosity=None) -> Dict[str, list]:
        import itertools
        verb = self.verbosity if verbosity is None else verbosity
        if valid_seen is None:
            valid_seen = [np.array(k) for k, v in self.cache.items()
                          if v['verdict'] == VALID]
        if not valid_seen:
            return {'vacuum': [], 'squeezed': []}

        rows, cols = np.triu_indices(self.num_modes)
        alphabet = [(_DIAG_VALUES if i == j else _OFFDIAG_VALUES)
                    for i, j in zip(rows, cols)]
        # every graph valid by closure, cheapest first
        cands = [np.array(c, dtype=int) for c in itertools.product(*alphabet)
                 if check_if_subgraph_triu([np.array(c, dtype=int)], valid_seen)]
        cands.sort(key=lambda t: int(np.sum(t)))

        kept = {'vacuum': [], 'squeezed': []}
        n_called = 0
        for t in cands:
            if kept['vacuum'] and check_if_subgraph_triu([t], kept['vacuum']):
                continue                      # vacuum-valid already, not minimal
            info = self.oracle(t)             # forces its OWN reservoir (fix 1)
            n_called += 1
            if info['verdict'] != VALID:
                continue
            cls = info.get('reservoir')
            if cls not in ('vacuum', 'squeezed'):
                cls = info.get('reservoir_kind', 'vacuum')
            if cls == 'vacuum':
                kept['vacuum'].append(t)
            elif not (kept['squeezed'] and check_if_subgraph_triu([t], kept['squeezed'])):
                kept['squeezed'].append(t)
        if verb:
            print(f'  reservoir-resolved minimality: {n_called} oracle calls, '
                  f'{len(kept["vacuum"])} vacuum-class, {len(kept["squeezed"])} squeezed-class')
        return kept

    # Irreducible valid graphs: drop any valid graph that has another valid
    # graph as a subgraph (it is superseded by the simpler one).
    #
    # NOTE this pools reservoir classes and so can discard a vacuum-drain
    # scheme in favour of a subgraph that needs a squeezed source. Use
    # minimal_valid_by_reservoir when the reservoir is part of the answer.
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

    # ─── §8 Move 4: the frontier is the only place a verdict is needed ────
    #
    # lattice_verdicts(): the verdict every graph in the lattice CARRIES,
    # obtained by closing the cached verdicts under the propagation rules
    # rather than by calling the oracle again. VALID if some cached-valid
    # graph sits inside it, INVALID if it sits inside a cached-invalid one,
    # else whatever the cache says, else UNDECIDED.
    def lattice_verdicts(self) -> Dict[tuple, str]:
        import itertools
        rows, cols = np.triu_indices(self.num_modes)
        alphabet = [(_DIAG_VALUES if i == j else _OFFDIAG_VALUES)
                    for i, j in zip(rows, cols)]
        valid = [np.array(k) for k, v in self.cache.items() if v['verdict'] == VALID]
        invalid = [np.array(k) for k, v in self.cache.items() if v['verdict'] == INVALID]
        out = {}
        for combo in itertools.product(*alphabet):
            t = np.array(combo, dtype=int)
            if valid and check_if_subgraph_triu([t], valid):
                out[combo] = VALID
            elif invalid and check_if_subgraph_triu(invalid, [t]):
                out[combo] = INVALID
            elif combo in self.cache:
                out[combo] = self.cache[combo]['verdict']
            else:
                out[combo] = UNDECIDED
        return out

    # certify_irreducible(triu): is this valid graph MINIMAL, provably?
    #
    # A graph is irreducible iff it is valid and every one-edge deletion is
    # invalid. That is the whole content of Move 4: irreducibility is a
    # statement about the valid/invalid BOUNDARY, so the deep interior of
    # either region never needs to be decided. Returns one of
    #   'irreducible'  valid, and every one-slot deletion is certified INVALID
    #   'reducible'    some one-slot deletion is itself VALID
    #   'unresolved'   no deletion is valid, but some are UNDECIDED — the
    #                  claim of minimality is NOT established, and the
    #                  offending neighbours are listed so an expensive exact
    #                  method (Move 3) can be pointed at them and nowhere else
    #   'not_valid'    the graph itself is not valid
    def certify_irreducible(self, triu, verdicts: Optional[Dict] = None) -> Dict:
        if verdicts is None:
            verdicts = self.lattice_verdicts()
        key = tuple(int(x) for x in np.asarray(triu))
        if verdicts.get(key) != VALID:
            return {'status': 'not_valid', 'graph': np.array(key),
                    'blocking': [], 'verdict': verdicts.get(key)}
        valid_nb, undec_nb = [], []
        for nb in _neighbours(np.array(key), self.num_modes, 'prune'):
            k = tuple(int(x) for x in nb)
            if verdicts.get(k) == VALID:
                valid_nb.append(nb)
            elif verdicts.get(k) == UNDECIDED:
                undec_nb.append(nb)
        if valid_nb:
            status = 'reducible'
        elif undec_nb:
            status = 'unresolved'
        else:
            status = 'irreducible'
        return {'status': status, 'graph': np.array(key),
                'valid_neighbours': valid_nb, 'blocking': undec_nb}

    # frontier_undecided(): the undecided graphs that actually OBSTRUCT the
    # answer — those sitting one edge below a valid graph, where the search
    # cannot tell whether descent is allowed.
    #
    # This is the number that matters, not the raw UNDECIDED count. An
    # undecided graph in the interior of the invalid region costs nothing:
    # nobody is trying to descend through it. Move 3's exponential backstop
    # is meant to run on THIS set, which is typically a handful, rather than
    # on the whole stratum.
    def frontier_undecided(self, verdicts: Optional[Dict] = None) -> List[np.ndarray]:
        if verdicts is None:
            verdicts = self.lattice_verdicts()
        out, seen = [], set()
        for key, v in verdicts.items():
            if v != VALID:
                continue
            for nb in _neighbours(np.array(key), self.num_modes, 'prune'):
                k = tuple(int(x) for x in nb)
                if verdicts.get(k) == UNDECIDED and k not in seen:
                    seen.add(k)
                    out.append(np.array(k))
        return sorted(out, key=lambda t: int(np.sum(t)))


# Complete sweep — every graph gets a verdict, but the oracle is called only
# where PROPAGATION does not already imply one (see verdict_of).
#
# propagate=True (the default) is what you want in every real run: it gives
# exactly the same partition as calling the oracle on all of them, because
# the propagation rules are sound in both directions, while skipping the
# calls whose answers were already determined.
#
# propagate=False forces an oracle call on every graph. That is the
# verification reference, not a production path: the two must agree
# graph-for-graph, so any difference localises a bug in the propagation
# rather than in the oracle. `check_propagation` below runs exactly that
# comparison.
#
# Graphs are visited in ASCENDING complexity so that valid graphs — which
# tend to be sparse for these targets, the minimal EPR scheme being two
# edges — are decided early and their whole up-set comes for free.
def sweep_all(
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    node_types: List[str],
    propagate: bool = True,
    **kwargs,
) -> Dict:
    import itertools
    n = len(node_types)
    rows, cols = np.triu_indices(n)
    alphabets = [(_DIAG_VALUES if i == j else _OFFDIAG_VALUES) for i, j in zip(rows, cols)]

    search = CertifiedSearch(sigma_target, target_mode_ids, node_types,
                             verbosity=0, **kwargs)
    combos = sorted(itertools.product(*alphabets), key=sum)

    out = {VALID: [], INVALID: [], UNDECIDED: []}
    verdicts, n_oracle = {}, 0
    for combo in combos:
        t = np.array(combo, dtype=int)
        if propagate:
            v, _, used = search.verdict_of(t)
        else:
            v, used = search.oracle(t)['verdict'], True
        n_oracle += int(used)
        verdicts[combo] = v
        out[v].append(t)

    # §8 Move 4: separate the undecided graphs that OBSTRUCT the answer (one
    # edge below a valid graph, so descent cannot be resolved) from those
    # that merely sit in the interior and cost nothing. The frontier set is
    # what an exact backstop should be pointed at.
    frontier = search.frontier_undecided(verdicts)

    return {'valid': out[VALID], 'invalid': out[INVALID], 'undecided': out[UNDECIDED],
            'frontier_undecided': frontier,
            'verdicts': verdicts, 'search': search, 'oracle_calls': n_oracle,
            'num_graphs': len(combos),
            'num_possible': calc_number_of_possibilities(node_types)}


# Verify the propagation against brute force on the same target: the two
# partitions must be identical graph-for-graph. Returns the comparison plus
# the oracle-call saving, so the mechanism can be audited rather than
# trusted.
def check_propagation(sigma_target, target_mode_ids, node_types, **kwargs) -> Dict:
    fast = sweep_all(sigma_target, target_mode_ids, node_types, propagate=True, **kwargs)
    full = sweep_all(sigma_target, target_mode_ids, node_types, propagate=False, **kwargs)
    diff = [g for g in full['verdicts'] if fast['verdicts'][g] != full['verdicts'][g]]
    return {'agree': not diff, 'disagreeing': diff,
            'oracle_calls_propagated': fast['oracle_calls'],
            'oracle_calls_exhaustive': full['oracle_calls'],
            'num_graphs': full['num_graphs'],
            'fast': fast, 'full': full}


# §5 pseudocode, "if some couplings ~ 0: add the reduced graph + its
# extensions": read the witness back and drop edges the solution does not
# actually use.
#
# The oracle is handed a graph as a PERMISSION, not a requirement — nothing
# forces every allowed coupling to be nonzero, and the attractivity filter
# picks an arbitrary member of the solution set, which may well leave some
# edges at zero. Projecting the witness G onto the generator basis and
# thresholding recovers the graph the scheme genuinely occupies, which is
# often strictly smaller than the one tested. Without this the search
# systematically over-reports edge count, exactly as
# covariance_optimizer.check_all_constraints exists to prevent.
#
# The threshold is relative to the largest coupling because (G, Upsilon) ->
# (sG, sUpsilon) is a gauge: only ratios are meaningful, so an absolute cut
# would depend on an arbitrary overall scale.
def reduce_witness(triu, info, num_modes: int, rel_tol: float = 1e-7):
    from reservoir_engineering.linear_oracle import hamiltonian_basis
    if info is None or info.get('G') is None:
        return np.asarray(triu, dtype=int)

    basis = hamiltonian_basis(triu, num_modes, include_detunings=True, allow_phases=True)
    if not basis:
        return np.asarray(triu, dtype=int)
    cols = np.column_stack([G.ravel() for _, G in basis])
    coeffs, *_ = np.linalg.lstsq(cols, info['G'].ravel(), rcond=None)
    scale = max(float(np.max(np.abs(coeffs))), 1e-300)

    # Which edge does each generator belong to, and is it used?
    used = {}
    for (label, _), c in zip(basis, coeffs):
        if label.startswith('delta'):
            continue                       # detunings are not edges
        kind, ij = label.split('_')[0], label.split('_')[-1]
        key = (kind.rstrip('90'), ij)
        used[key] = used.get(key, False) or (abs(c) / scale > rel_tol)

    rows, cols_i = np.triu_indices(num_modes)
    out = np.zeros(len(rows), dtype=int)
    for k, (i, j) in enumerate(zip(rows, cols_i)):
        val = int(np.asarray(triu)[k])
        if val == NO_COUPLING:
            continue
        if i == j:
            if used.get(('par', f'{i}'), False):
                out[k] = PARAMETRIC
            continue
        bs = used.get(('bs', f'{i}{j}'), False)
        tms = used.get(('tms', f'{i}{j}'), False)
        if bs and tms:
            out[k] = BEAMSPLITTER_AND_TWO_MODE_SQUEEZING
        elif bs:
            out[k] = BEAMSPLITTER
        elif tms:
            out[k] = TWO_MODE_SQUEEZING
    return out


# §5 pseudocode, "increment auxiliary modes until the smallest fully
# connected graph is valid". The fully connected graph is the most
# permissive on a given mode count, so if IT cannot stabilise the target no
# subgraph can, and the mode count must go up. Returns the smallest n_aux
# that works, together with the witness that settled it.
#
# The root must be VALID, not merely feasible — the prune pass starts there,
# and a root that is only stationary (the frozen point, feasible on every
# graph) would descend through nothing. That is why §8 Move 1's gap
# maximiser matters most at this call: an UNDECIDED root would wrongly push
# the search to an extra auxiliary mode it does not need.
def find_minimum_auxiliary_modes(
    sigma_target: np.ndarray,
    target_mode_ids: List[int],
    node_types_signal: List[str],
    max_aux: int = 4,
    aux_type: str = 'cavity',
    verbosity: int = 1,
    **kwargs,
):
    from reservoir_engineering.linear_oracle import optimise_aux_state
    for n_aux in range(0, max_aux + 1):
        node_types = list(node_types_signal) + [aux_type] * n_aux
        n = len(node_types)
        rows, cols = np.triu_indices(n)
        root = np.array([PARAMETRIC if i == j else BEAMSPLITTER_AND_TWO_MODE_SQUEEZING
                         for i, j in zip(rows, cols)], dtype=int)
        if n_aux == 0:
            # No drain at all: no dissipation anywhere, so nothing can be
            # attractive. Skip the solve.
            if verbosity:
                print(f'  n_aux=0: no dissipative mode, skipped')
            continue
        info = optimise_aux_state(root, sigma_target, target_mode_ids, node_types, **kwargs)
        if verbosity:
            print(f'  n_aux={n_aux}: fully connected root -> {info["verdict"]}')
        if info['verdict'] == VALID:
            return {'num_aux': n_aux, 'node_types': node_types, 'root': root, 'info': info}
    return None


# Human-readable edge list for a triu_array.
def describe(triu, num_modes: int) -> str:
    labels = {BEAMSPLITTER: 'BS', TWO_MODE_SQUEEZING: 'TMS', PARAMETRIC: 'PAR',
              BEAMSPLITTER_AND_TWO_MODE_SQUEEZING: 'BS+TMS'}
    rows, cols = np.triu_indices(num_modes)
    parts = [(f'({i}){labels[int(v)]}' if i == j else f'({i},{j}){labels[int(v)]}')
             for (i, j), v in zip(zip(rows, cols), triu) if int(v) != NO_COUPLING]
    return ', '.join(parts) if parts else '(empty graph)'
