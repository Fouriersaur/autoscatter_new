"""
epr_certified.py
================
Full certified search for the TWO-MODE EQUALLY SQUEEZED state (TMSV / EPR),
with the reservoir type decided per graph rather than assumed.

Target: targets.two_mode_squeezed(r) on two mechanical modes, stabilised
through one auxiliary cavity drain. This is the state
vitali_rediscovery.py goes after with the gradient optimiser; here every
one of the 512 three-node graphs is DECIDED, exhaustively, by the linear
oracle.

What is different from every earlier run in this package: each graph is
handed to linear_oracle.optimise_aux_state, not to `decide`. That means the
drain's state is not fixed in advance — the search works out for itself
whether the graph needs a squeezed reservoir or a plain vacuum one, by
minimising (rank drop of the design matrix) + lambda*r^2 from cold starts.
So the passive Zippilli-Vitali schemes and the active Kronwald-like schemes
are found in the SAME sweep, each labelled with the reservoir it actually
requires, with nothing told to the search about which frame to look in.

Reading the output:
  reservoir = vacuum    the graph works with an unsqueezed drain
  reservoir = squeezed  the drain must be squeezed, and r* says by how much
  reservoir = none      no drain state makes this graph work (or the
                        attractivity search declined — see UNDECIDED)

Usage:
    python3 epr_certified.py          # r = 0.5
    python3 epr_certified.py 0.8      # any squeezing
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (two_mode_squeezed, purity,
                                            log_negativity, duan_criterion)
import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.certified_search import describe, sweep_all
from reservoir_engineering.topology_search import check_if_subgraph_triu

NODE_TYPES = ['cavity', 'mechanical', 'mechanical']
TARGET_MODE_IDS = [1, 2]
N_NODES = 3

# Vitali's passive triangle, kept only as a reference point in the report.
VITALI_TRIU = (0, 1, 1, 0, 1, 0)


# Complete sweep of every graph, with SUBGRAPH PROPAGATION always on and
# the reservoir decided per graph.
#
# Propagation is not an optimisation to be switched on for big problems and
# skipped for small ones — it is part of the algorithm, so this goes through
# certified_search.sweep_all rather than reimplementing a bare loop. Every
# graph still receives a verdict; the oracle is simply not consulted where
# an already-decided graph settles the question:
#     INVALID  =>  every SUBgraph is INVALID   (shrinking S_G can only
#                  shrink the solution set, so no Hurwitz member appears)
#     VALID    =>  every SUPERgraph is VALID   (the witness embeds unchanged,
#                  drain state included)
#     UNDECIDED => implies nothing, propagates nowhere
# Both directions stay sound with auto_reservoir on, since the drain state
# travels as part of the witness.
def sweep(target, num_samples: int = 32, propagate: bool = True):
    t0 = time.time()
    out = sweep_all(target, TARGET_MODE_IDS, NODE_TYPES,
                    propagate=propagate, auto_reservoir=True,
                    num_samples=num_samples)
    return out, time.time() - t0


# Verdict plus witness for one graph. Graphs settled by propagation have no
# witness of their own, so the reservoir is inherited from the valid
# subgraph that settled them — sound, because that subgraph's witness is
# literally the one this graph uses.
def info_for(combo, sweep_out):
    cache = sweep_out['search'].cache
    if combo in cache:
        return cache[combo], True
    if sweep_out['verdicts'][combo] == lo.VALID:
        t = np.array(combo, dtype=int)
        for key, inf in cache.items():
            if inf['verdict'] == lo.VALID and check_if_subgraph_triu([t], [np.array(key)]):
                return inf, False
    return None, False


# Irreducible valid graphs within a class: drop any that contains another.
def irreducible(keys):
    arrs = [np.array(k, dtype=int) for k in keys]
    kept = []
    for i, t in enumerate(arrs):
        others = arrs[:i] + arrs[i + 1:]
        if others and check_if_subgraph_triu([t], others):
            continue
        kept.append(t)
    return sorted(kept, key=lambda t: int(np.sum(t)))


# Physical readout: named couplings, drain rate, bath squeezing.
def readout(info, triu):
    basis = lo.hamiltonian_basis(triu, N_NODES, include_detunings=True, allow_phases=True)
    cols = np.column_stack([G.ravel() for _, G in basis])
    coeffs, *_ = np.linalg.lstsq(cols, info['G'].ravel(), rcond=None)
    scale = max(np.max(np.abs(coeffs)), 1e-300)
    return {lbl: float(c / scale) for (lbl, _), c in zip(basis, coeffs)
            if abs(c / scale) > 1e-6}


def report(r, target, sweep_out, elapsed):
    results = {}       # combo -> info, with propagated verdicts filled in
    for combo, v in sweep_out['verdicts'].items():
        info, explicit = info_for(combo, sweep_out)
        if info is None:
            results[combo] = {'verdict': v, 'reservoir': 'n/a', 'explicit': False,
                               'aux_squeezing': [(0.0, 0.0)], 'stability_margin': float('nan')}
        else:
            results[combo] = dict(info, verdict=v, explicit=explicit)
    print(f'\n{"=" * 76}')
    print(f'  Two-mode equally squeezed state (TMSV / EPR), r = {r}')
    print(f'  1 auxiliary cavity drain + 2 mechanical signal modes')
    print(f'{"=" * 76}\n')

    print('Target:')
    print(f'  purity            {purity(target):.12f}   (1 = pure, required)')
    # E_N = 2r/ln2 in this package's convention (see tests/test_epr.py), not
    # r/ln2 — the two_mode_squeezed docstring's "log negativity = r" is in
    # nats, and log_negativity() returns bits.
    print(f'  log-negativity    {log_negativity(target):.6f}   '
          f'(theory 2r/ln2 = {2 * r / np.log(2):.6f})')
    print(f'  Duan inseparable  {duan_criterion(target)}')
    print(f'  covariance:\n{np.array2string(target, precision=4, prefix="    ")}\n')

    by = {}
    for k, v in results.items():
        tag = v['verdict'] if v['verdict'] != lo.VALID else f'VALID/{v["reservoir"]}'
        by.setdefault(tag, []).append(k)

    print(f'All {len(results)} graphs decided in {elapsed:.0f}s:')
    for tag in sorted(by, key=lambda t: -len(by[t])):
        print(f'  {tag:16s} {len(by[tag]):4d}')
    n_call = sweep_out['oracle_calls']
    n_tot = sweep_out['num_graphs']
    print(f'\nSubgraph propagation: {n_call} oracle calls for {n_tot} graphs '
          f'({100 * (1 - n_call / n_tot):.0f}% skipped).')
    print('  INVALID settles every subgraph, VALID settles every supergraph,')
    print('  UNDECIDED settles nothing — which is why the saving is modest here:')
    print(f'  {len(by.get(lo.UNDECIDED, []))} of {n_tot} graphs carry a verdict that cannot propagate.')

    # §8 Move 4: of the undecided graphs, only those sitting one edge below a
    # valid one actually obstruct the answer — they are where the search
    # cannot tell whether a simpler scheme exists. The rest are interior and
    # cost nothing, so this number, not the raw undecided count, is the size
    # of the remaining problem.
    frontier = sweep_out.get('frontier_undecided', [])
    print(f'  of those, {len(frontier)} sit one edge below a VALID graph and so')
    print('  genuinely block a minimality claim (§8 Move 4); the rest are interior.')
    print()

    for cls, title in [('VALID/vacuum', 'VACUUM reservoir'),
                       ('VALID/squeezed', 'SQUEEZED reservoir')]:
        keys = by.get(cls, [])
        if not keys:
            print(f'--- no schemes needing a {title} ---\n')
            continue
        irr = irreducible(keys)
        print(f'{"=" * 76}')
        print(f'  {title}: {len(keys)} valid graphs, {len(irr)} irreducible')
        print(f'{"=" * 76}')
        for n, triu in enumerate(irr):
            info = results[tuple(int(x) for x in triu)]
            print(f'\n--- #{n + 1}  complexity {int(np.sum(triu))} ---')
            print(f'  edges: {describe(triu, N_NODES)}')
            if info['reservoir'] == 'squeezed':
                print(f'  drain: SQUEEZED, r* = {info["aux_squeezing"][0][0]:.6f} '
                      f'(target r = {r})')
            else:
                print('  drain: plain vacuum')
            print(f'  rank(Upsilon) = {info["upsilon_rank"]} channel(s)   '
                  f'normalised margin {info["stability_margin"]:.4f}')
            print(f'  stationarity residual {info["stationarity_residual"]:.2e}   '
                  f'forward Lyapunov error {info["forward_error"]:.2e}')
            passive = all(int(v) in (0, 1) for v in triu)
            print(f'  passive (beamsplitter-only) network? {passive}')
            print('  couplings (relative to the largest):')
            for lbl, val in sorted(readout(info, triu).items(), key=lambda kv: -abs(kv[1])):
                print(f'    {lbl:12s} {val:+.4f}')
            if info['V_forward'] is not None:
                sig = info['V_forward'][2:6, 2:6]
                print(f'  achieved: purity {purity(sig):.10f}, '
                      f'log-negativity {log_negativity(sig):.6f}, '
                      f'Duan {duan_criterion(sig)}')
            if tuple(int(x) for x in triu) == VITALI_TRIU:
                print("  ^ this is Vitali's passive triangle")
        print()

    # Best by stability margin, across both classes — the schemes that
    # actually relax quickly, which minimal edge count alone does not tell.
    valid = [(k, v) for k, v in results.items() if v['verdict'] == lo.VALID]
    valid.sort(key=lambda kv: -kv[1]['stability_margin'])
    print(f'{"-" * 76}')
    print('Ranked by stability margin (how fast the scheme actually relaxes):')
    print(f'  {"edges":42s} {"cx":3s} {"reservoir":10s} {"r*":7s} margin')
    for k, v in valid[:8]:
        passive = all(int(x) in (0, 1) for x in k)
        tag = f'{v["reservoir"]}{" (passive)" if passive else ""}'
        print(f'  {describe(np.array(k), N_NODES):42s} {int(sum(k)):3d} {tag:10s} '
              f'{v["aux_squeezing"][0][0]:<7.4f} {v["stability_margin"]:.4f}')


if __name__ == '__main__':
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    target = two_mode_squeezed(r)
    print(f'Certified sweep of all 512 three-node graphs, TMSV r = {r}')
    print('subgraph propagation ON; reservoir type decided per graph\n')
    out, elapsed = sweep(target)
    report(r, target, out, elapsed)

    # Audit the propagation rather than trusting it: redo the sweep with the
    # oracle forced on every graph and require the two partitions to match
    # graph-for-graph. Skipped unless asked for, since it costs a second
    # full sweep.
    if '--audit' in sys.argv:
        print(f'\n{"-" * 76}')
        print('Auditing propagation against a brute-force sweep...')
        full, _ = sweep(target, propagate=False)
        diff = [g for g in full['verdicts']
                if full['verdicts'][g] != out['verdicts'][g]]
        print(f'  partitions identical graph-for-graph? {not diff}   '
              f'({len(diff)} disagreements)')
        print(f'  oracle calls: {out["oracle_calls"]} propagated '
              f'vs {full["oracle_calls"]} exhaustive')
