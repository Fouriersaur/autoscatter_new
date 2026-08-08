"""
discover_two_mode_squeezed.py
=============================
Cold-start discovery of every scheme that stabilises the PURE IDEAL
TWO-MODE SQUEEZED STATE (TMSV / EPR), run as the full §5 pipeline of
state_stabilization_algorithm.md.

Input is one thing only: the target covariance matrix
targets.two_mode_squeezed(r). Nothing about Kronwald, Zippilli-Vitali or
Koga-Yamamoto is supplied, no topology is curated, no reservoir type is
declared, and no starting point is seeded. Everything below is an output.

The pipeline, §5 end to end:

  Phase 0  n_aux auto-increment. Add auxiliary modes until the fully
           connected graph — the most permissive one on that mode count —
           can stabilise the target. If IT cannot, no subgraph can, so the
           mode count must go up.

  Phase 1  Bidirectional certified search on that mode count. Prune from
           the lattice maximum, grow from the empty graph, with verdicts
           propagating: INVALID settles every subgraph, VALID settles every
           supergraph, UNDECIDED settles nothing. §5(iii) then requires the
           two directions to imply the SAME partition — a deterministic
           oracle cannot depend on traversal order, so a disagreement is an
           implementation bug, not a lost scheme.

  Phase 2  Report. Each surviving scheme is reduced to the edges its
           witness actually uses (§5 graph reduction), the reservoir is
           read off the dissipator by factorisation (§4B), and the whole
           thing is re-verified by an INDEPENDENT forward Lyapunov solve.

What the oracle decides per graph, with nothing assumed:
  - whether a stabilising (G, Upsilon) exists at all;
  - if so, whether the drain must be SQUEEZED or plain VACUUM, and by how
    much — from optimise_aux_state, and cross-checked by factoring Upsilon;
  - how many dissipative channels the scheme needs, from rank(Upsilon).

Verdicts are three-valued. VALID carries an explicit witness, INVALID a
certificate, and UNDECIDED means the oracle declined — it is never treated
as invalid and never prunes anything. So "no simpler scheme was found" is
the claim being made, not "none exists"; only the INVALID set is proven.

Usage:
    python3 discover_two_mode_squeezed.py            # r = 0.5
    python3 discover_two_mode_squeezed.py 0.8        # any squeezing
    python3 discover_two_mode_squeezed.py 0.5 --quick   # skip the second
                                                        # (agreement) pass
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (two_mode_squeezed, purity,
                                            log_negativity, duan_criterion,
                                            symplectic_eigenvalues, is_physical)
import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.topology_search import check_if_subgraph_triu
from reservoir_engineering.certified_search import (
    CertifiedSearch, find_minimum_auxiliary_modes, reduce_witness, describe)

SIGNAL_TYPES = ['mechanical', 'mechanical']
TARGET_MODE_IDS = [0, 1]          # signal modes come first; drains appended


# Confirm the target really is the pure ideal TMSV before anything else.
# The oracle requires purity (§2.1) and refuses mixed targets, so a bad
# target should fail here with a readable message rather than deep inside a
# solve.
def validate(target, r):
    nus = symplectic_eigenvalues(target)
    return {
        'physical': is_physical(target),
        'purity': purity(target),
        'symplectic_max_dev': float(np.max(np.abs(nus - 0.5))),
        'log_negativity': log_negativity(target),
        'log_negativity_theory': 2 * r / np.log(2),
        'duan': duan_criterion(target),
    }


# Named couplings from a witness, normalised to the largest. Only ratios
# are physical: (G, Upsilon) -> (sG, sUpsilon) is a gauge, so an absolute
# scale would be meaningless.
def couplings_of(info, triu, num_modes):
    basis = lo.hamiltonian_basis(triu, num_modes, include_detunings=True,
                                  allow_phases=True)
    cols = np.column_stack([G.ravel() for _, G in basis])
    coeffs, *_ = np.linalg.lstsq(cols, info['G'].ravel(), rcond=None)
    scale = max(float(np.max(np.abs(coeffs))), 1e-300)
    return {lbl: float(c / scale) for (lbl, _), c in zip(basis, coeffs)
            if abs(c / scale) > 1e-6}


def describe_scheme(idx, triu, info, target, node_types, r):
    n = len(node_types)
    aux_ids = [i for i in range(n) if i not in TARGET_MODE_IDS]
    print(f'\n--- scheme #{idx}   complexity {int(np.sum(triu))} ---')
    print(f'  edges: {describe(triu, n)}')

    # Reservoir, two independent ways: the search's own answer, and the
    # factorisation of the Upsilon it returned (§4B). They must agree.
    res = lo.reservoir_summary(info['Upsilon'], n)
    scan_kind = info.get('reservoir', '?')
    scan_r = info.get('aux_squeezing', [(0.0, 0.0)])[0][0]
    agree = (scan_kind == res['kind'])
    if res['kind'] == 'squeezed':
        print(f'  reservoir: SQUEEZED, s = {res["max_squeezing"]:.6f}   '
              f'(target r = {r}; drain-state scan said {scan_kind} r*={scan_r:.6f})')
    else:
        print(f'  reservoir: plain VACUUM   (drain-state scan agrees: {scan_kind})')
    if not agree:
        print('  !! scan and Upsilon factorisation DISAGREE — investigate')
    print(f'  dissipative channels: rank(Upsilon) = {info["upsilon_rank"]}')

    passive = all(int(v) in (0, 1) for v in triu)
    Om = lo.symplectic_form(n)
    commutes = float(np.linalg.norm(info['G'] @ Om - Om @ info['G']))
    print(f'  passive network? {passive}   ||[G, Omega]|| = {commutes:.2e} '
          f'(0 => photon-number-conserving H)')
    print(f'  normalised stability margin {info["stability_margin"]:.4f}')

    # The experimentally meaningful numbers, in units of the drain rate.
    params = lo.physical_parameters(info, triu, node_types, TARGET_MODE_IDS)
    print('  PARAMETERS:')
    print(lo.format_parameters(params, indent='    '))

    # Independent verification: never trust the oracle's own algebra.
    print(f'  stationarity residual {info["stationarity_residual"]:.2e}')
    print(f'  forward Lyapunov error {info["forward_error"]:.2e}   '
          f'(scipy solve of the recovered A, D)')
    if info['V_forward'] is not None:
        idx_sig = []
        for m in TARGET_MODE_IDS:
            idx_sig += [2 * m, 2 * m + 1]
        sig = info['V_forward'][np.ix_(idx_sig, idx_sig)]
        print(f'  achieved: purity {purity(sig):.10f}, '
              f'log-negativity {log_negativity(sig):.6f} '
              f'(target {log_negativity(target):.6f}), Duan {duan_criterion(sig)}')


def main(r=0.5, quick=False, num_samples=24):
    target = two_mode_squeezed(r)
    chk = validate(target, r)

    print(f'{"=" * 76}')
    print(f'  Cold-start discovery — pure ideal two-mode squeezed state, r = {r}')
    print(f'{"=" * 76}\n')
    print('Target (the ONLY input):')
    print(np.array2string(target, precision=4, prefix='  '))
    print(f'\n  physical              {chk["physical"]}')
    print(f'  purity                {chk["purity"]:.12f}   (1 = pure, required by §2.1)')
    print(f'  max |nu_k - 1/2|      {chk["symplectic_max_dev"]:.2e}')
    print(f'  log-negativity        {chk["log_negativity"]:.6f}   '
          f'(theory 2r/ln2 = {chk["log_negativity_theory"]:.6f})')
    print(f'  Duan inseparable      {chk["duan"]}')

    # ---- Phase 0: how many auxiliary modes are needed? ------------------
    print(f'\n{"-" * 76}')
    print('Phase 0 — auxiliary-mode count (§5: increment until the fully')
    print('          connected graph can stabilise the target)')
    t0 = time.time()
    found = find_minimum_auxiliary_modes(target, TARGET_MODE_IDS, SIGNAL_TYPES,
                                          max_aux=3, num_samples=num_samples)
    if found is None:
        print('  no auxiliary-mode count up to 3 works; stopping.')
        return
    n_aux, node_types = found['num_aux'], found['node_types']
    n = len(node_types)
    print(f'  -> {n_aux} auxiliary mode(s); node_types = {node_types}   '
          f'({time.time() - t0:.0f}s)')

    # ---- Phase 1: bidirectional certified search ------------------------
    print(f'\n{"-" * 76}')
    print('Phase 1 — bidirectional certified search with verdict propagation')
    search = CertifiedSearch(target, TARGET_MODE_IDS, node_types,
                             auto_reservoir=True, num_samples=num_samples,
                             verbosity=1)
    t0 = time.time()
    if quick:
        res = search.run('grow')
        seen = res['valid']
        agree = None
    else:
        out = search.run_bidirectional()
        seen = out['top_down']['valid'] + out['bottom_up']['valid']
        agree = out['agree']

    # Resolve each surviving scheme's OWN reservoir and take minimality
    # WITHIN each reservoir class. Without this, a scheme needing only a
    # plain vacuum drain is discarded whenever some subgraph works with a
    # squeezed one — which is precisely how the Woolley-Clerk scheme went
    # missing, since the two-beamsplitter graph is its subgraph.
    by_class = search.minimal_valid_by_reservoir(seen)
    minimal = by_class['vacuum'] + by_class['squeezed']
    elapsed = time.time() - t0

    total = 4 ** (n * (n - 1) // 2) * 2 ** n
    print(f'\n  graphs in the lattice      {total}')
    print(f'  oracle calls (memoized)    {len(search.cache)}')
    print(f'  wall time                  {elapsed:.0f}s')
    if agree is not None:
        print(f'  §5(iii) bidirectional agreement: {agree}'
              + ('' if agree else '   <-- IMPLEMENTATION BUG, not a lost scheme'))

    from collections import Counter
    counts = Counter(v['verdict'] for v in search.cache.values())
    print(f'  verdicts among graphs actually decided: {dict(counts)}')
    print('  (UNDECIDED is not "invalid" — the oracle declined, and it prunes nothing)')

    # ---- Phase 2: report ------------------------------------------------
    print(f'\n{"=" * 76}')
    print(f'  {len(minimal)} irreducible scheme(s) discovered')
    print(f'    {len(by_class["vacuum"])} needing only a PLAIN VACUUM drain')
    print(f'    {len(by_class["squeezed"])} needing a SQUEEZED drain')
    print('  (minimality is computed within each reservoir class: a scheme that')
    print('   avoids a squeezed source is not "reducible" to one that needs one)')
    print(f'{"=" * 76}')

    rows = []
    for triu in minimal:
        info = search.oracle(triu)
        if info['verdict'] != lo.VALID:
            continue
        red = reduce_witness(triu, info, n)
        if not np.array_equal(red, triu) and int(np.sum(red)) > 0:
            red_info = search.oracle(red)
            if red_info['verdict'] == lo.VALID:
                triu, info = red, red_info
        rows.append((triu, info))

    # Rank by stability margin, not edge count: a scheme with fewer edges
    # but a near-zero margin relaxes to the target arbitrarily slowly.
    rows.sort(key=lambda ti: -ti[1]['stability_margin'])
    for k, (triu, info) in enumerate(rows):
        describe_scheme(k + 1, triu, info, target, node_types, r)

    if rows:
        def table(title, ordered):
            print(f'\n{"-" * 76}')
            print(title)
            print(f'  {"edges":40s} {"cx":>3s} {"reservoir":10s} {"s":>8s} '
                  f'{"chan":>4s} {"margin":>8s}')
            for triu, info in ordered:
                res = lo.reservoir_summary(info['Upsilon'], n)
                print(f'  {describe(triu, n):40s} {int(np.sum(triu)):3d} '
                      f'{res["kind"]:10s} {res["max_squeezing"]:8.4f} '
                      f'{info["upsilon_rank"]:4d} {info["stability_margin"]:8.4f}')

        # Two orderings, because they disagree and each answers a different
        # question. Margin = how fast the scheme actually relaxes to the
        # target; complexity = how much hardware it costs. On the Cayley
        # cluster these picked different winners (complexity 19 at margin
        # 0.0001 versus complexity 5 at 0.0280), so reporting only one of
        # them hides a real trade-off.
        table('Ranked by STABILITY MARGIN (fastest relaxation first):',
              sorted(rows, key=lambda ti: -ti[1]['stability_margin']))
        table('Ranked by COMPLEXITY (least hardware first):',
              sorted(rows, key=lambda ti: (int(np.sum(ti[0])),
                                            -ti[1]['stability_margin'])))

        # How many graphs are valid in total, versus how many are
        # irreducible? Every valid graph either IS one of these or contains
        # one as a subgraph — the irreducible set is what carries the
        # physics, the rest are the same schemes with spare edges bolted on.
        import itertools
        rows_i, cols_i = np.triu_indices(n)
        alpha = [([0, 3] if i == j else [0, 1, 2, 4]) for i, j in zip(rows_i, cols_i)]
        irr = [t for t, _ in rows]
        total = sum(1 for combo in itertools.product(*alpha)
                    if check_if_subgraph_triu([np.array(combo, dtype=int)], irr))
        print(f'\n{"-" * 76}')
        print(f'{len(rows)} irreducible schemes; {total} of {len(list(itertools.product(*alpha)))} '
              f'graphs in the lattice are valid')
        print('  (every valid graph contains an irreducible one as a subgraph —')
        print('   the extras are the same schemes with redundant edges added)')
        # ---- Phase 3: §8 Move 4 — what is actually still open ------------
        #
        # "N irreducible schemes" is only a complete answer if the lattice
        # is fully decided. Where it is not, the claim must be qualified
        # PRECISELY, and the qualification is much narrower than the raw
        # UNDECIDED count: a graph left undecided in the interior of the
        # invalid region blocks nothing, because nothing is trying to
        # descend through it. What blocks the answer is an undecided graph
        # sitting one edge BELOW a valid one — there the search cannot tell
        # whether a simpler scheme exists. Those, and only those, are where
        # an exact backstop (Move 3) would have to run.
        verdicts = search.lattice_verdicts()
        from collections import Counter as _C
        tally = _C(verdicts.values())
        frontier = search.frontier_undecided(verdicts)
        print(f'\n{"-" * 76}')
        print('Lattice status (§4C three-way verdict, closed under propagation):')
        for k in (lo.VALID, lo.INVALID, lo.UNDECIDED):
            print(f'  {k:10s} {tally.get(k, 0):5d}')
        print(f'\n§8 Move 4 — undecided graphs that actually OBSTRUCT the answer')
        print(f'  (one edge below a valid graph): {len(frontier)} of '
              f'{tally.get(lo.UNDECIDED, 0)} undecided')

        status = _C()
        for triu, _ in rows:
            status[search.certify_irreducible(triu, verdicts)['status']] += 1
        print(f'\nIrreducibility of the {len(rows)} reported schemes:')
        print(f'  certified irreducible (every one-edge deletion proven INVALID) '
              f'{status["irreducible"]}')
        print(f'  minimality UNPROVEN (some deletion is undecided)              '
              f'{status["unresolved"]}')
        # 'reducible' is not a contradiction here: minimality was taken
        # WITHIN a reservoir class, so a vacuum-drain scheme is kept even
        # when a subgraph of it is valid with a squeezed drain. Neither
        # dominates — one trades couplings for not needing a squeezed source.
        print(f'  has a valid subgraph in the OTHER reservoir class            '
              f'{status["reducible"]}')
        if status['unresolved']:
            print('  -> for those, a simpler scheme may exist; the oracle has not')
            print('     ruled it out. Point Move 3 at the frontier graphs above.')
        if not tally.get(lo.UNDECIDED, 0):
            print('\nThe lattice is FULLY DECIDED: every graph is VALID or INVALID')
            print('by certificate, so this irreducible set is complete, not partial.')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(r=float(args[0]) if args else 0.5, quick='--quick' in sys.argv)
