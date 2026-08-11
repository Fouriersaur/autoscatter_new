"""
cayley_cluster_bidirectional.py
===============================
The Cayley cluster state of cayley_cluster_rediscovery.py, searched with the
ACTUAL certified search — certified_search.CertifiedSearch.run_bidirectional()
driving linear_oracle.decide — rather than the hand-rolled level sweep in
cayley_cluster_certified.py.

What this adds over that sweep
------------------------------
The level sweep enumerates complexity levels bottom-up and its only use of a
verdict is to skip supergraphs of already-valid graphs. It therefore decides
~97% of the graphs it sees individually, and cannot go past complexity ~7 at
5 nodes.

CertifiedSearch propagates in BOTH directions and prunes on each pass's
frontier (certified_search.run):
    prune  descends from the lattice maximum through valid territory; on
           INVALID it stops, and the whole down-set is invalid unvisited.
    grow   ascends from the empty graph through invalid territory; on VALID
           it stops, and the whole up-set is valid unvisited.
UNDECIDED propagates nothing in either direction, so both passes step past it
instead of pruning — which is what keeps a failed attractivity search from
poisoning a whole down-set.

It also files the REDUCED witness (reduce_witness): the edges the solution
actually uses, not the edges the tested graph allowed. That enlarges the valid
up-set and so strengthens the pruning, and it is the reason the answer here is
"schemes that cannot be further reduced" rather than "graphs that happened to
be tested".

run_bidirectional then asserts the two passes never CONTRADICT — no graph
VALID in one and INVALID in the other. That is a soundness check on the
propagation, not a completeness proof; the passes are expected to differ in
UNDECIDED coverage (see certified_search.closure_partition).

Budget guard
------------
CertifiedSearch.run() has no time or visit limit, and at 5 nodes the lattice
is 4^10 * 2^5 = 33,554,432 graphs. The valid up-set is a large fraction of
that, so an unguarded prune pass can run for a very long time. This file wraps
the oracle with a counter and a wall-clock deadline that raises, so a pass
that overruns yields PARTIAL results instead of nothing.

A truncated pass changes what may be claimed, and the report says so: the
minimal-valid set is then "minimal among what was reached", and the certified
lower bound is lost. Only a pass that ran to exhaustion supports the strong
claim.

Usage:
    python3 cayley_cluster_bidirectional.py                  # z=0.6, 1800s/pass
    python3 cayley_cluster_bidirectional.py 0.6 3600         # z, seconds per pass
    python3 cayley_cluster_bidirectional.py 0.6 3600 --auto  # decide the drain
                                                               state per graph
    python3 cayley_cluster_bidirectional.py 0.6 3600 --vacuum # unsqueezed frame
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.certified_search import (CertifiedSearch, describe,
                                                     reduce_witness)
from reservoir_engineering.targets import purity, nullifier_variances

from cayley_cluster_rediscovery import (ADJACENCY_P4, NODE_TYPES, TARGET_MODE_IDS,
                                        N_NODES, cayley_cluster_target, validate_target)


class Budgeted(CertifiedSearch):
    """CertifiedSearch with a wall-clock guard and progress reporting.

    The guard raises out of the traversal rather than returning early, because
    `run` has no partial-return path: it either drains its frontier or it does
    not. The caller catches the exception and keeps whatever the pass filed
    into self._valid_seen / self._invalid_seen up to that point, which is
    sound (each entry came from a real oracle verdict) but INCOMPLETE.
    """

    # Own tallies. CertifiedSearch.run() files verdicts into LOCAL lists and
    # only populates self._valid_seen / self._invalid_seen via verdict_of(),
    # which run() does not call — so those attributes stay empty for the whole
    # traversal. Accumulating here instead is what makes a TRUNCATED pass
    # useful: the valid graphs found before the deadline survive the exception.
    def start_budget(self, seconds, tag):
        self._deadline = time.time() + seconds
        self._calls = 0
        self._tag = tag
        self._t0 = time.time()
        if not hasattr(self, 'found_valid'):
            self.found_valid, self.found_invalid, self.found_undecided = [], [], []

    # Lift the guard once the traversals are done. The deadline must not apply
    # to the REPORTING calls that follow — reduce_witness and the per-scheme
    # readout both go through oracle(), and every one of them is a cache hit,
    # so letting the guard fire there throws away a completed search.
    def clear_budget(self):
        self._deadline = float('inf')

    def oracle(self, triu):
        if time.time() > self._deadline:
            raise TimeoutError(f'{self._tag}: budget exhausted after '
                               f'{self._calls} oracle calls')
        key = tuple(int(x) for x in np.asarray(triu))
        fresh = key not in self.cache
        info = super().oracle(triu)
        if fresh:
            self._calls += 1
            t = np.asarray(triu)
            if info['verdict'] == lo.VALID:
                self.found_valid.append(t)
            elif info['verdict'] == lo.INVALID:
                self.found_invalid.append(t)
            else:
                self.found_undecided.append(t)
            if self._calls % 200 == 0:
                print(f'    [{self._tag}] {self._calls} calls, '
                      f'{len(self.found_valid)}V / {len(self.found_invalid)}I / '
                      f'{len(self.found_undecided)}U, '
                      f'{time.time() - self._t0:.0f}s', flush=True)
        return info


# Persist the graph lists after each pass. These traversals cost ~30 min each
# and the verdicts are the expensive part; dumping them means a failure in the
# REPORTING stage (which is cheap and easy to get wrong) never costs a re-run.
STATE_PATH = os.environ.get('BIDIR_STATE', 'bidirectional_state.npz')


def _save(search, tag):
    try:
        np.savez_compressed(
            STATE_PATH,
            valid=np.array(search.found_valid, dtype=int).reshape(-1, 15),
            invalid=np.array(search.found_invalid, dtype=int).reshape(-1, 15),
            undecided=np.array(search.found_undecided, dtype=int).reshape(-1, 15))
        print(f'    [state saved {tag}: {STATE_PATH}]', flush=True)
    except Exception as e:                      # never let saving kill a run
        print(f'    [state save failed ({tag}): {e}]', flush=True)


# One pass, with the budget guard. Returns (result_or_None, completed).
def guarded_pass(search, direction, seconds):
    search.start_budget(seconds, direction)
    print(f'\n--- {direction} pass (budget {seconds:.0f}s) ---', flush=True)
    try:
        res = search.run(direction)
        print(f'    {direction} COMPLETED: visited {res["visited"]}, '
              f'valid {len(res["valid"])}, invalid {len(res["invalid"])}, '
              f'undecided {len(res["undecided"])}', flush=True)
        return res, True
    except TimeoutError as e:
        print(f'    {direction} TRUNCATED: {e}', flush=True)
        print(f'    partial: {len(search.found_valid)} valid, '
              f'{len(search.found_invalid)} invalid, '
              f'{len(search.found_undecided)} undecided', flush=True)
        return None, False


# Full physical readout of one minimal-valid scheme.
def report_scheme(rank, triu, search, target):
    info = search.oracle(triu)
    print(f'--- #{rank}  complexity {int(np.sum(triu))} ---')
    print(f'  edges: {describe(triu, N_NODES)}')
    if info['verdict'] != lo.VALID:
        print(f'  (verdict on re-check: {info["verdict"]})\n')
        return
    print(f'  stationarity residual {info["stationarity_residual"]:.2e}   '
          f'forward Lyapunov error {info["forward_error"]:.2e}')
    print(f'  normalised stability margin {info["stability_margin"]:.4f}   '
          f'rank(Upsilon) = {info["upsilon_rank"]} channel(s)')
    p = lo.physical_parameters(info, triu, NODE_TYPES, TARGET_MODE_IDS)
    print(lo.format_parameters(p, indent='      '))
    A = lo.drift_matrix(info['G'], info['Upsilon'])
    kap = abs(2.0 * float(np.imag(info['Upsilon'][0, 1]))) or 1.0
    print(f'      dissipative gap = {-max(np.real(np.linalg.eigvals(A))) / kap:.6f} kappa')
    if info.get('V_forward') is not None:
        sig = info['V_forward'][2:10, 2:10]
        print(f'      achieved purity {purity(sig):.10f},  '
              f'||sigma - target|| = {np.linalg.norm(sig - target):.2e}')
        print(f'      nullifiers {np.array2string(nullifier_variances(sig, ADJACENCY_P4), precision=6)}')
    print(f'      passive (beamsplitter-only)? '
          f'{all(int(v) in (0, 1) for v in triu)}')
    print()


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    z = float(args[0]) if len(args) > 0 else 0.6
    per_pass = float(args[1]) if len(args) > 1 else 1800.0
    auto = '--auto' in sys.argv
    vacuum = '--vacuum' in sys.argv

    target = cayley_cluster_target(ADJACENCY_P4, z)
    checks = validate_target(target, ADJACENCY_P4, z)

    aux = None if (vacuum or auto) else [(z, 0.0)]
    frame = ('per-graph (auto_reservoir)' if auto else
             'vacuum drains' if vacuum else f'drain squeezed at r = z = {z}')

    print(f'{"=" * 78}')
    print(f'  Cayley cluster on P4, z = {z} — CertifiedSearch.run_bidirectional')
    print(f'{"=" * 78}')
    print(f'  frame        : {frame}')
    print(f'  target purity: {checks["purity"]:.12f}   '
          f'max|nu-1/2| {checks["symplectic_max_dev"]:.1e}')
    print(f'  lattice      : {4 ** 10 * 2 ** 5:,} graphs (5 nodes)')
    print(f'  budget       : {per_pass:.0f}s per pass', flush=True)

    search = Budgeted(target, TARGET_MODE_IDS, NODE_TYPES, aux_squeezing=aux,
                      num_samples=24, auto_reservoir=auto, verbosity=1)

    # grow first: it ascends through invalid territory and stops at the valid
    # boundary, so it is the pass that reaches the MINIMAL valid graphs. If
    # only one pass fits in the budget, this is the one worth having.
    bu, bu_done = guarded_pass(search, 'grow', per_pass)
    _save(search, 'after grow')
    td, td_done = guarded_pass(search, 'prune', per_pass)
    _save(search, 'after prune')

    # Traversals over: no further oracle call may be refused on time.
    search.clear_budget()

    print(f'\n{"=" * 78}')
    print('RESULTS')
    print(f'{"=" * 78}')
    print(f'  grow  pass complete : {bu_done}')
    print(f'  prune pass complete : {td_done}')

    if bu_done and td_done:
        ptd = search.closure_partition(td)
        pbu = search.closure_partition(bu)
        disagree = [g for g in ptd if ptd[g] != pbu[g]]
        contra = [g for g in disagree if {ptd[g], pbu[g]} == {lo.VALID, lo.INVALID}]
        print(f'  bidirectional agreement (no VALID/INVALID contradiction): '
              f'{not contra}   ({len(contra)} contradictions, '
              f'{len(disagree)} differing only by UNDECIDED coverage)')

    # minimal_valid over everything either pass filed. reduce_witness has
    # already stripped unused edges, so these are irreducible on the evidence
    # the oracle produced, not merely the smallest graphs that were tested.
    # Apply reduce_witness before minimising, as run() does internally: the
    # oracle is handed a graph as a PERMISSION, not a requirement, so a valid
    # graph's witness often leaves some allowed edges at zero. The reduced
    # graph is the scheme actually realised, and it is what "cannot be further
    # reduced" has to be judged on.
    pool = list(search.found_valid)
    for t in search.found_valid:
        red = reduce_witness(t, search.oracle(t), N_NODES)
        if not np.array_equal(red, t) and int(np.sum(red)) > 0:
            pool.append(red)
    minimal = search.minimal_valid(pool) if pool else []
    minimal = sorted(minimal, key=lambda t: int(np.sum(t)))

    print(f'  oracle verdicts cached : {len(search.cache):,}')
    print(f'  VALID     graphs found : {len(search.found_valid):,}')
    print(f'  INVALID   graphs found : {len(search.found_invalid):,}')
    print(f'  UNDECIDED graphs found : {len(search.found_undecided):,}')
    print(f'\n{len(minimal)} VALID scheme(s) that cannot be further reduced:\n')

    if not (bu_done and td_done):
        print('  WARNING: at least one pass was truncated by the budget. These are')
        print('  minimal among the graphs REACHED, and no lower bound is certified —')
        print('  a simpler scheme may exist in the unexplored region.\n')

    for rank, triu in enumerate(minimal, 1):
        report_scheme(rank, triu, search, target)
