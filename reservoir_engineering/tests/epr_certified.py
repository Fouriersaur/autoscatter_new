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
  reservoir = none      no drain state was found that makes this graph
                        work (which is not a proof that none exists)

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
# The VALID direction is sound; the INVALID one is a heuristic, since INVALID
# only means no witness was found. Both survive auto_reservoir, since the
# drain state travels as part of the witness.
def sweep(target, num_samples: int = 32, propagate: bool = True):
    t0 = time.time()
    out = sweep_all(target, TARGET_MODE_IDS, NODE_TYPES,
                    propagate=propagate, auto_reservoir=True,
                    num_samples=num_samples)
    return out, time.time() - t0


# NOTE on how the minimal schemes are extracted (this used to be wrong here).
#
# The reservoir class is a property of a (graph, WITNESS) pair, not of a
# graph: the same graph can be valid with a squeezed drain and, via a
# different Hamiltonian, valid with a plain vacuum one. Propagation carries
# a witness, so it carries that witness's reservoir too — a supergraph of
# the two-edge squeezed-drain star is stamped VALID/squeezed without the
# oracle ever being asked whether it ALSO works at vacuum.
#
# Filtering irreducibility on those inherited labels then deletes real
# answers: the Woolley-Clerk graph (0,1)BS+TMS,(0,2)BS+TMS is VALID with a
# VACUUM drain at complexity 8, but it sits above the complexity-2 star, so
# it was binned squeezed, found to contain a squeezed-class graph, and
# dropped — while strictly worse complexity-10 vacuum schemes printed.
#
# certified_search.minimal_valid_by_reservoir is the fix and says so in its
# own source: it forces each candidate's OWN oracle call and keeps two
# independent minimality lattices. Never re-roll it locally.


# Physical readout: named couplings, drain rate, bath squeezing, all in
# units of the drain rate (the solution set is a cone, so absolute coupling
# strengths are gauge).
def readout(info, triu):
    params = lo.physical_parameters(info, triu, NODE_TYPES, TARGET_MODE_IDS)
    return lo.format_parameters(params, indent='    ')


def report(r, target, sweep_out, elapsed):
    search = sweep_out['search']
    verdicts = sweep_out['verdicts']
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

    counts = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    print(f'All {len(verdicts)} graphs decided in {elapsed:.0f}s:')
    for tag in sorted(counts, key=lambda t: -counts[t]):
        print(f'  {tag:12s} {counts[tag]:4d}')

    n_call = sweep_out['oracle_calls']
    n_tot = sweep_out['num_graphs']
    print(f'\nSubgraph propagation: {n_call} oracle calls for {n_tot} graphs '
          f'({100 * (1 - n_call / n_tot):.0f}% skipped).')
    print('  INVALID settles every subgraph, VALID settles every supergraph.')
    print('  INVALID is not a proof, so the pruning it does is a heuristic: a')
    print('  scheme whose Hurwitz cone the filter missed takes its down-set with it.')

    # The VALID count above is a count of GRAPHS, not of reservoir classes:
    # most of those graphs were settled by propagation, which carries a
    # witness and therefore a reservoir, so they were never themselves asked
    # which reservoir they need. Resolving that is a separate pass.
    print(f'\nResolving the reservoir class per graph — every candidate gets its OWN')
    print('oracle call, so no class is inherited through propagation:')
    by_class = search.minimal_valid_by_reservoir(verbosity=1)

    for cls, title in [('vacuum', 'VACUUM reservoir'),
                       ('squeezed', 'SQUEEZED reservoir')]:
        graphs = sorted(by_class[cls], key=lambda t: int(np.sum(t)))
        if not graphs:
            print(f'\n--- no minimal schemes needing a {title} ---')
            continue
        print(f'\n{"=" * 76}')
        print(f'  {title}: {len(graphs)} minimal scheme(s)')
        print(f'{"=" * 76}')
        for n, triu in enumerate(graphs):
            info = search.cache[tuple(int(x) for x in triu)]
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
            print(readout(info, triu))
            if info['V_forward'] is not None:
                sig = info['V_forward'][2:6, 2:6]
                print(f'  achieved: purity {purity(sig):.10f}, '
                      f'log-negativity {log_negativity(sig):.6f}, '
                      f'Duan {duan_criterion(sig)}, '
                      f'||sigma-target|| {np.linalg.norm(sig - target):.2e}')
            if tuple(int(x) for x in triu) == VITALI_TRIU:
                print("  ^ this is Vitali's passive triangle")
        print()

    # Best by stability margin, over the graphs the oracle actually decided.
    # Propagated graphs are excluded on purpose: they carry someone else's
    # margin, which is not a figure of merit for them.
    valid = [(k, v) for k, v in search.cache.items() if v['verdict'] == lo.VALID]
    valid.sort(key=lambda kv: -kv[1]['stability_margin'])
    print(f'{"-" * 76}')
    print('Ranked by stability margin (explicitly decided graphs only):')
    print(f'  {"edges":42s} {"cx":3s} {"reservoir":10s} {"r*":7s} margin')
    for k, v in valid[:10]:
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
