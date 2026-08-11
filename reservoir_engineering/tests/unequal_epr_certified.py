"""
unequal_epr_certified.py
========================
Full certified (exact-solve) search for an UNEQUALLY SQUEEZED two-mode
squeezed state, with the reservoir type decided per graph rather than
assumed.

This is epr_certified.py's sweep run against the asymmetric target of
vitali_rediscovery_unequal.py: every one of the 512 three-node graphs is
DECIDED by the linear oracle (one global SVD per graph, no restarts), and
the drain state is worked out per graph by optimise_aux_state instead of
being fixed in advance.

Target construction (identical to vitali_rediscovery_unequal.py, so the
gradient-optimiser run and this exact run are answering the same question):
start from targets.two_mode_squeezed(r), then hit each mode with its OWN
local squeezer r1, r2.

    sigma0 = 0.5*[[c*I2, -s*sz], [-s*sz, c*I2]]     c=cosh(2r), s=sinh(2r)
    S      = diag(e^-r1, e^r1, e^-r2, e^r2)
    sigma  = S @ sigma0 @ S.T

Local squeezing is symplectic, so sigma stays PURE — which is what the
linear oracle requires (complete_covariance raises on a mixed target). What
breaks is the mode-swap symmetry: the two marginals now differ, so there is
no reason to expect the symmetric-target answer (Vitali's triangle) to be
the minimal scheme here, and nothing symmetric is imposed anywhere.

Reading the output:
  reservoir = vacuum    the graph works with an unsqueezed drain
  reservoir = squeezed  the drain must be squeezed, and r* says by how much
  reservoir = none      no drain state makes this graph work (or the
                        attractivity search declined — see UNDECIDED)

Usage:
    python3 unequal_epr_certified.py                 # r=0.5, r1=0.2, r2=-0.1
    python3 unequal_epr_certified.py 0.5 0.2 -0.1    # r r1 r2
    python3 unequal_epr_certified.py 0.5 0.2 -0.1 --audit
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (purity, log_negativity,
                                            duan_criterion, symplectic_eigenvalues)
import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.certified_search import describe, sweep_all
from reservoir_engineering.covariance_physics import get_mode_covariance

NODE_TYPES = ['cavity', 'mechanical', 'mechanical']
TARGET_MODE_IDS = [1, 2]
N_NODES = 3

# Vitali's passive triangle for the SYMMETRIC target, kept only as a
# reference point in the report.
VITALI_TRIU = (0, 1, 1, 0, 1, 0)


# Unequally-squeezed two-mode squeezed target: two_mode_squeezed(r) plus
# per-mode local squeezers r1 (mode 1), r2 (mode 2). r1 == r2 == 0
# reproduces targets.two_mode_squeezed(r) exactly.
def unequal_two_mode_squeezed(r: float, r1: float, r2: float) -> np.ndarray:
    c = np.cosh(2 * r)
    s = np.sinh(2 * r)
    sz = np.diag([1., -1.])
    I2 = np.eye(2)
    sigma0 = 0.5 * np.block([[c * I2, -s * sz],
                              [-s * sz, c * I2]])
    S = np.diag([np.exp(-r1), np.exp(r1), np.exp(-r2), np.exp(r2)])
    return S @ sigma0 @ S.T


def sweep(target, num_samples: int = 32, propagate: bool = True):
    t0 = time.time()
    out = sweep_all(target, TARGET_MODE_IDS, NODE_TYPES,
                    propagate=propagate, auto_reservoir=True,
                    num_samples=num_samples)
    return out, time.time() - t0


# NOTE on how the minimal schemes are extracted — see the same note in
# epr_certified.py. The reservoir class belongs to a (graph, WITNESS) pair,
# not to a graph, and propagation carries a witness, so a supergraph of a
# squeezed-drain scheme is stamped VALID/squeezed without ever being asked
# whether it also works at vacuum. Filtering irreducibility on those
# inherited labels deletes real answers.
# certified_search.minimal_valid_by_reservoir forces each candidate's OWN
# oracle call and keeps two independent minimality lattices. Use it.


# Physical readout: named couplings, drain rate, bath squeezing, all in
# units of the drain rate (the solution set is a cone, so absolute coupling
# strengths are gauge).
def readout(info, triu):
    params = lo.physical_parameters(info, triu, NODE_TYPES, TARGET_MODE_IDS)
    return lo.format_parameters(params, indent='    ')


# Is the scheme symmetric under swapping the two signal modes 1 <-> 2?
# For an asymmetric target it should NOT have to be, and this is the line
# that says so graph by graph.
def swap_symmetric(triu) -> bool:
    t = {(i, j): int(v) for (i, j), v in zip(zip(*np.triu_indices(N_NODES)), triu)}
    return (t[(0, 1)] == t[(0, 2)]) and (t[(1, 1)] == t[(2, 2)])


def report(r, r1, r2, target, sweep_out, elapsed):
    search = sweep_out['search']
    verdicts = sweep_out['verdicts']

    print(f'\n{"=" * 76}')
    print(f'  UNEQUALLY squeezed two-mode squeezed state, r = {r}, r1 = {r1}, r2 = {r2}')
    print(f'  1 auxiliary cavity drain + 2 mechanical signal modes')
    print(f'{"=" * 76}\n')

    m1 = get_mode_covariance(target, [0])
    m2 = get_mode_covariance(target, [1])
    print('Target:')
    print(f'  purity            {purity(target):.12f}   (1 = pure, required by the oracle)')
    print(f'  log-negativity    {log_negativity(target):.6f}   '
          f'(unchanged by local squeezing: 2r/ln2 = {2 * r / np.log(2):.6f})')
    print(f'  Duan inseparable  {duan_criterion(target)}')
    print(f'  symplectic evals  {np.array2string(symplectic_eigenvalues(target), precision=6)}')
    print(f'  marginals         mode1 (x,p) = ({m1[0, 0]:.4f}, {m1[1, 1]:.4f})   '
          f'mode2 (x,p) = ({m2[0, 0]:.4f}, {m2[1, 1]:.4f})')
    print(f'  -> flat spectrum? {"yes" if abs(r1 - r2) < 1e-12 else "NO — the two modes are unequally squeezed"}')
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
    frontier = sweep_out.get('frontier_undecided', [])
    print(f'  {counts.get(lo.UNDECIDED, 0)} graphs carry a verdict that cannot propagate;')
    print(f'  of those, {len(frontier)} sit one edge below a VALID graph and so')
    print('  genuinely block a minimality claim (§8 Move 4); the rest are interior.')

    # The VALID count above counts GRAPHS, not reservoir classes: most were
    # settled by propagation and so were never asked which reservoir they
    # need. That is a separate pass.
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
                rs, th = info['aux_squeezing'][0]
                print(f'  drain: SQUEEZED, r* = {rs:.6f}, theta* = {th:+.4f} rad  (target r = {r})')
            else:
                print('  drain: plain vacuum')
            print(f'  rank(Upsilon) = {info["upsilon_rank"]} channel(s)   '
                  f'normalised margin {info["stability_margin"]:.4f}')
            print(f'  stationarity residual {info["stationarity_residual"]:.2e}   '
                  f'forward Lyapunov error {info["forward_error"]:.2e}')
            passive = all(int(v) in (0, 1) for v in triu)
            print(f'  passive (beamsplitter-only) network? {passive}')
            print(f'  symmetric under 1<->2 swap? {swap_symmetric(triu)}  '
                  f'(target is not, so it need not be)')
            print(readout(info, triu))
            if info['V_forward'] is not None:
                sig = info['V_forward'][2:6, 2:6]
                print(f'  achieved: purity {purity(sig):.10f}, '
                      f'log-negativity {log_negativity(sig):.6f}, '
                      f'Duan {duan_criterion(sig)}, '
                      f'||sigma-target|| {np.linalg.norm(sig - target):.2e}')
            if tuple(int(x) for x in triu) == VITALI_TRIU:
                print("  ^ this is Vitali's passive triangle (the SYMMETRIC-target answer)")
        print()

    # Best by stability margin, over the graphs the oracle actually decided.
    # Propagated graphs carry someone else's margin, so they are excluded.
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
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    r  = float(args[0]) if len(args) > 0 else 0.5
    r1 = float(args[1]) if len(args) > 1 else 0.2
    r2 = float(args[2]) if len(args) > 2 else -0.1
    target = unequal_two_mode_squeezed(r, r1, r2)
    print(f'Certified sweep of all 512 three-node graphs — unequally squeezed '
          f'two-mode state, r = {r}, r1 = {r1}, r2 = {r2}')
    print('subgraph propagation ON; reservoir type decided per graph\n')
    out, elapsed = sweep(target)
    report(r, r1, r2, target, out, elapsed)

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
