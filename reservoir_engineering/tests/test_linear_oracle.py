"""
test_linear_oracle.py
=====================
Validation for the certifying oracle (linear_oracle.py) and the three-valued
search (certified_search.py) — the implementation of
state_stabilization_algorithm.md.

Every test compares against something independent of the oracle itself:
known schemes from the literature, an exhaustive sweep, or a forward
Lyapunov solve computed by scipy rather than by the oracle's own algebra.

    python3 test_linear_oracle.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import squeezed_vacuum, two_mode_squeezed
import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.certified_search import (
    CertifiedSearch, sweep_all, describe)

_results = []


def check(label, condition, detail=''):
    _results.append(bool(condition))
    print(f'  [{"PASS" if condition else "FAIL"}] {label}' + (f'   {detail}' if detail else ''))
    return bool(condition)


# Test 1 — Kronwald. 1 auxiliary cavity (the drain) + 1 mechanical signal
# mode, target = single-mode squeezed vacuum. The known answer is BS+TMS on
# the single pair (a bichromatic drive). The oracle must find it in ONE
# linear solve, and the witness must survive an independent forward check:
# feed the recovered (A, D) to scipy's Lyapunov solver and recover the
# target.
#
# Note the qualifier on the last check. Everything here uses a VACUUM drain,
# and with a vacuum drain neither sideband alone suffices — a passive
# network cannot squeeze anything and TMS alone cannot damp. That is NOT the
# same as "BS alone is impossible": give the drain a squeezed bath and the
# beamsplitter-only graph works perfectly well (test 6), with a better
# stability margin than Kronwald itself. Whether a sideband is needed is a
# statement about the drain's bath, not about the graph on its own.
def test_kronwald(r=1.0):
    print(f'\nTest 1 — Kronwald, target squeezed_vacuum({r})')
    nt, tmi = ['cavity', 'mechanical'], [1]
    V = lo.complete_covariance(squeezed_vacuum(r), tmi, 2)

    res = {name: lo.decide(t, V, tmi, nt, num_samples=64)
           for name, t in [('empty', [0, 0, 0]), ('BS', [0, 1, 0]),
                            ('TMS', [0, 2, 0]), ('BS+TMS', [0, 4, 0])]}

    check('BS+TMS is VALID (the Kronwald scheme)', res['BS+TMS']['verdict'] == lo.VALID)
    w = res['BS+TMS']
    if w['verdict'] == lo.VALID:
        check('  witness satisfies stationarity to ~1e-12',
              w['stationarity_residual'] < 1e-10, f'residual={w["stationarity_residual"]:.2e}')
        check('  A is strictly Hurwitz', w['stability_margin'] > 0,
              f'normalised margin={w["stability_margin"]:.4f}')
        check('  independent forward Lyapunov solve reproduces the target',
              w['forward_error'] < 1e-10, f'||V_forward - V||={w["forward_error"]:.2e}')
        check('  Upsilon is PSD (physically realisable)', w['upsilon_psd'])

    check('empty graph is INVALID by certificate', res['empty']['verdict'] == lo.INVALID,
          res['empty'].get('certificate', {}).get('kind', ''))
    check('no single sideband is ever reported VALID',
          all(res[k]['verdict'] != lo.VALID for k in ('BS', 'TMS')),
          f"BS={res['BS']['verdict']}, TMS={res['TMS']['verdict']}")


# Test 2 — Zippilli-Vitali. A PASSIVE (beamsplitter-only) triangle plus one
# squeezed drain stabilises a two-mode squeezed state. This is the case the
# default V_aux = (1/2)I gauge fixing hides, so it exercises
# scan_aux_squeezing — and the drain squeezing that works should come out at
# the target's own r, not at some arbitrary grid point.
def test_vitali_passive(r=0.4):
    print(f'\nTest 2 — Zippilli-Vitali passive triangle, target two_mode_squeezed({r})')
    nt, tmi = ['cavity', 'mechanical', 'mechanical'], [1, 2]
    triangle = [0, 1, 1, 0, 1, 0]     # (0,1)BS (0,2)BS (1,2)BS

    out = lo.scan_aux_squeezing(triangle, two_mode_squeezed(r), tmi, nt, num_samples=120)
    ok = check('passive triangle is VALID with a squeezed drain',
               out['verdict'] == lo.VALID, f'aux_squeezing={out.get("aux_squeezing")}')
    if ok:
        r_aux = out['aux_squeezing'][0]
        check('  drain squeezing matches the target squeezing', abs(r_aux - r) < 0.06,
              f'r_aux={r_aux:.3f} vs r={r}')
        check('  forward Lyapunov solve reproduces the target',
              out['forward_error'] < 1e-10, f'{out["forward_error"]:.2e}')
        check('  single dissipative channel (rank-1 Upsilon)',
              out['upsilon_rank'] == 1, f'rank={out["upsilon_rank"]}')

    # And the same scheme in the default gauge: unsqueezed drain, BS+TMS on
    # the drain edges. Squeezing auxiliary mode 0 maps one to the other, so
    # these are the same physics in two frames — see complete_covariance.
    V = lo.complete_covariance(two_mode_squeezed(r), tmi, 3)
    img = lo.decide([0, 4, 4, 0, 1, 0], V, tmi, nt, num_samples=64)
    check('gauge image (BS+TMS drain edges, unsqueezed drain) is also VALID',
          img['verdict'] == lo.VALID)


# Test 3 — search integrity, doc §5(iii). The top-down and bottom-up passes
# must imply the SAME partition of the graph lattice, and both must agree
# with an exhaustive sweep that does no pruning at all. Under a sound oracle
# this is forced: a verdict is a deterministic function of the graph, so it
# cannot depend on traversal order. A failure localises an implementation
# bug in the propagation, not a lost scheme.
def test_search_integrity(r=0.4):
    print(f'\nTest 3 — bidirectional search integrity, target two_mode_squeezed({r})')
    nt, tmi = ['cavity', 'mechanical', 'mechanical'], [1, 2]
    tgt = two_mode_squeezed(r)

    s = CertifiedSearch(tgt, tmi, nt, num_samples=48, verbosity=0)
    out = s.run_bidirectional()
    check('top-down and bottom-up imply identical partitions', out['agree'],
          f'{len(out["disagreeing_graphs"])} differing graphs')
    check('minimal-valid sets agree between directions', out['minimal_valid_agree'])

    sweep = sweep_all(tgt, tmi, nt, num_samples=48)
    sweep_valid = {tuple(int(x) for x in g) for g in sweep['valid']}
    for name in ('partition_top_down', 'partition_bottom_up'):
        bfs_valid = {g for g, v in out[name].items() if v == lo.VALID}
        check(f'{name[10:]:9s} VALID set == exhaustive sweep', bfs_valid == sweep_valid,
              f'{len(bfs_valid)} vs {len(sweep_valid)}')

    mv = out['minimal_valid']
    check('minimal scheme is the two-drain-edge BS+TMS pair',
          len(mv) > 0 and describe(mv[0], 3) == '(0,1)BS+TMS, (0,2)BS+TMS',
          describe(mv[0], 3) if mv else '(none)')
    print(f'    {len(sweep["valid"])} VALID / {len(sweep["invalid"])} INVALID / '
          f'{len(sweep["undecided"])} UNDECIDED out of 512 graphs')


# Test 4 — scope guard. A mixed target has nonzero signal-drain correlation,
# so the full covariance is NOT determined and the stationarity equation is
# bilinear rather than linear. The oracle must refuse rather than silently
# linearise something that is not linear.
def test_mixed_target_refused():
    print('\nTest 4 — mixed target is refused, not silently linearised')
    from reservoir_engineering.targets import thermal
    try:
        lo.complete_covariance(thermal(0.3, 1), [1], 2)
        check('mixed target raises', False, 'no exception raised')
    except ValueError as exc:
        check('mixed target raises ValueError', True)
        check('  message names the scope boundary', 'PURE' in str(exc) or 'pure' in str(exc))


# Test 5 — generator-basis completeness. For the fully connected graph with
# detunings, S_G must be ALL symmetric 2n x 2n matrices: a graph that
# forbids nothing must not restrict the Hamiltonian. dim = n(2n+1).
def test_basis_completeness():
    print('\nTest 5 — generator basis spans the full symmetric space when unconstrained')
    for n in (2, 3, 4, 5):
        rows, cols = np.triu_indices(n)
        triu = np.array([3 if i == j else 4 for i, j in zip(rows, cols)])
        basis = lo.hamiltonian_basis(triu, n, include_detunings=True, allow_phases=True)
        expected = n * (2 * n + 1)
        mats = np.column_stack([G.ravel() for _, G in basis])
        check(f'n={n}: {len(basis)} generators, rank {np.linalg.matrix_rank(mats)}, '
              f'expected {expected}',
              len(basis) == expected and np.linalg.matrix_rank(mats) == expected)


# Test 6 — the oracle decides for ITSELF whether a drain needs squeezing,
# the analogue of covariance_optimizer's free squeeze_ab variables where
# (a_,b_) = (0,0) recovered plain vacuum.
#
# optimise_aux_state tries vacuum first, and only if that fails hunts for a
# drain state that makes the design matrix drop rank. Both outcomes are
# checked here, on graphs whose answer is known independently:
#   BS+TMS with a vacuum drain is Kronwald  -> must report 'vacuum'
#   a PASSIVE network cannot squeeze anything by itself, so it can only work
#   if the drain supplies the squeezing -> must report 'squeezed', at the
#   target's own squeezing
def test_reservoir_choice():
    print('\nTest 6 — vacuum vs squeezed reservoir is decided by the search')
    from reservoir_engineering.targets import purity as _purity

    nt2, tmi2 = ['cavity', 'mechanical'], [1]
    nt3, tmi3 = ['cavity', 'mechanical', 'mechanical'], [1, 2]

    o = lo.optimise_aux_state(np.array([0, 4, 0]), squeezed_vacuum(1.0), tmi2, nt2,
                              num_samples=64)
    check('BS+TMS reports a VACUUM reservoir (Kronwald)',
          o['verdict'] == lo.VALID and o['reservoir'] == 'vacuum')

    # Passive graph, single mode: only reachable with a squeezed drain, and
    # the required squeezing must be the target's own.
    o = lo.optimise_aux_state(np.array([0, 1, 0]), squeezed_vacuum(1.0), tmi2, nt2,
                              num_samples=64)
    ok = check('BS-only reports a SQUEEZED reservoir',
               o['verdict'] == lo.VALID and o['reservoir'] == 'squeezed')
    if ok:
        check('  required squeezing equals the target squeezing',
              abs(o['aux_squeezing'][0][0] - 1.0) < 1e-3,
              f'r* = {o["aux_squeezing"][0][0]:.6f}')
        check('  the Hamiltonian really is passive ([G, Omega] = 0)',
              np.linalg.norm(o['G'] @ lo.symplectic_form(2)
                             - lo.symplectic_form(2) @ o['G']) < 1e-9)
        check('  forward Lyapunov solve reproduces the target',
              o['forward_error'] < 1e-10, f'{o["forward_error"]:.2e}')

    o = lo.optimise_aux_state(np.array([0, 1, 1, 0, 1, 0]), two_mode_squeezed(0.4),
                              tmi3, nt3, num_samples=64)
    ok = check('Vitali passive triangle reports SQUEEZED at the target r',
               o['verdict'] == lo.VALID and o['reservoir'] == 'squeezed'
               and abs(o['aux_squeezing'][0][0] - 0.4) < 1e-3,
               f'r* = {o["aux_squeezing"][0][0]:.6f}')

    o = lo.optimise_aux_state(np.array([0, 4, 4, 0, 1, 0]), two_mode_squeezed(0.4),
                              tmi3, nt3, num_samples=64)
    check('its active gauge image reports VACUUM (no squeezing invented)',
          o['verdict'] == lo.VALID and o['reservoir'] == 'vacuum')


if __name__ == '__main__':
    test_kronwald()
    test_vitali_passive()
    test_search_integrity()
    test_mixed_target_refused()
    test_basis_completeness()
    test_reservoir_choice()
    n_pass, n_tot = sum(_results), len(_results)
    print(f'\n{"="*66}\n  {n_pass}/{n_tot} checks passed\n{"="*66}')
    sys.exit(0 if n_pass == n_tot else 1)
