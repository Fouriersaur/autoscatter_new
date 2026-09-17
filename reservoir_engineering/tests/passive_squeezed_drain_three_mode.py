"""
passive_squeezed_drain_three_mode.py
====================================
Same three-mode pure target as partial_passive_three_mode.py (unflat
Bloch-Messiah spectrum), but now under FULL Hamiltonian passivity —

    H is number-conserving EVERYWHERE: beamsplitters and detunings only,
    no anomalous term anywhere in G, not even on the drain edges —

with the DRAIN SQUEEZING SEARCHED rather than declared. This is the
Zippilli-Vitali branch: a passive linear network plus squeezed reservoirs.
With no anomalous term available in H, every scrap of squeezing in the
target has to be imported through the baths, so the question is entirely
"which drain states, and how many".

What the search actually varies. The drain state enters the solve through
the completed covariance V (complete_covariance's aux_squeezing), NOT as a
free unknown — an unknown V_AA would multiply the unknown G and make (**)
bilinear, destroying the one-shot global solve. So it is an OUTER variable
of 2 parameters per drain around an exactly linear inner problem, exactly as
linear_oracle.optimise_aux_state does it, with two changes:

  - the detector is built on the PASSIVE basis, not the free one, so it
    looks for drain states this constrained problem can use rather than
    ones the unconstrained problem could;
  - the seeds are not blind. For this target V_xp = 0, so
    2V = blockdiag(Q, Q^-1) and the Bloch-Messiah factorisation is
    S = blockdiag(Q^1/2, Q^-1/2) = O Z O^T with O ORTHOGONAL AND SYMPLECTIC
    (it acts identically on x and p, i.e. it is a passive beamsplitter
    network) and Z the squeezings r_k = (1/2) ln lam_k(Q). That is precisely
    a passive network fed by squeezed vacua of parameter r_k, so the
    Bloch-Messiah spectrum is the natural guess for the drain states and is
    seeded as such — then checked by the oracle rather than assumed.

Sign convention on r. complete_covariance's (r, theta) squeezes x and
anti-squeezes p; a Bloch-Messiah mode with the LARGE variance in x needs the
opposite orientation, i.e. -r or equivalently theta = pi/2. Both signs are
seeded.

A grid or an optimiser proves nothing between its points, so a miss here
means "no drain state on the grid worked", not "none exists".
"""

import time
import numpy as np
import scipy.optimize as sciopt

from reservoir_engineering import solver_constraints as sc
from reservoir_engineering.linear_oracle import (
    VALID, complete_covariance, build_linear_system, physical_parameters,
    format_parameters, drift_matrix, factor_dissipator)
from reservoir_engineering.certified_search import reduce_witness
from reservoir_engineering.tests.partial_passive_three_mode import (
    V_GIVEN, purify_xp_block, bloch_messiah_spectrum, describe_target,
    edge_string, full_graph, report_scheme, sparsify)

MAX_AUX = 3
CONSTRAINTS = [sc.passive_hamiltonian()]

# Cheap settings for the scan: a grid point only has to answer "is there a
# witness here?".
SCAN_KW = dict(gap_effort=2)


# rank_drop_detector(...): the cheap signal that a drain state is SPECIAL.
#
# The constrained design matrix M(r, theta) has some generic nullity d0; at a
# drain state the passive network can actually use, extra solutions appear
# and the (d0+1)-th smallest singular value collapses to ~1e-16. One SVD per
# evaluation, no oracle call, so a fine sweep is affordable.
def make_detector(triu, V_target, target_mode_ids, node_types, n_aux):
    n = len(node_types)
    ctx = sc.ConstraintContext(n, target_mode_ids, node_types)
    cset = sc.as_constraint_set(CONSTRAINTS)
    _, h_basis, d_basis = sc.constrained_bases(triu, ctx, cset)

    def svs(params):
        aux = [(float(params[2 * k]), float(params[2 * k + 1]))
               for k in range(n_aux)]
        V = complete_covariance(V_target, target_mode_ids, n, aux_squeezing=aux)
        M, _, _ = build_linear_system(V, h_basis, d_basis, None)
        s = np.linalg.svd(M, compute_uv=False)
        return np.sort(s) / max(float(s[0]) if s.size else 1.0, 1e-300)

    rng = np.random.default_rng(0)
    d0 = min(int(np.sum(svs(rng.uniform(0.1, 1.0, 2 * n_aux)) <= 1e-10))
             for _ in range(5))

    def detect(params):
        s = svs(np.asarray(params, dtype=float))
        return float(s[d0]) if d0 < len(s) else 0.0

    return detect, d0, len(h_basis), len(d_basis)


def seeds_for(n_aux, r_bm):
    """Drain-state seeds: the Bloch-Messiah values in both orientations, the
    shared-squeezing diagonal, and plain vacuum."""
    out = [np.zeros(2 * n_aux)]
    picks = list(r_bm[:n_aux]) if n_aux <= len(r_bm) else \
        list(r_bm) + [r_bm[-1]] * (n_aux - len(r_bm))
    for sign in (+1, -1):
        for th in (0.0, np.pi / 2):
            out.append(np.array([v for r in picks for v in (sign * r, th)]))
    for r in np.linspace(0.1, 1.2, 12):
        out.append(np.array([v for _ in range(n_aux) for v in (r, 0.0)]))
        out.append(np.array([v for _ in range(n_aux) for v in (-r, 0.0)]))
    return out


def refine(detect, seeds, r_max=2.0):
    best = (np.inf, seeds[0])
    for p0 in seeds:
        res = sciopt.minimize(
            lambda p: (1e3 if np.any(np.abs(p[0::2]) > r_max) else
                       detect(p) + 1e-4 * float(np.sum(p[0::2] ** 2))),
            p0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-16})
        if res.fun < best[0]:
            best = (float(res.fun), res.x)
    return best


def aux_str(params):
    n_aux = len(params) // 2
    return ', '.join(f'(r={params[2 * k]:+.5f}, th={params[2 * k + 1]:+.4f})'
                     for k in range(n_aux))


def main():
    n_sig = 3
    target_mode_ids = list(range(n_sig))

    print('=' * 78)
    print('TARGET (purified)')
    print('=' * 78)
    V_t = purify_xp_block(V_GIVEN, n_sig)
    describe_target(V_t)
    r_bm = np.sort(bloch_messiah_spectrum(V_t))[::-1]
    print(f'  -> Bloch-Messiah seeds for the drain states: {np.round(r_bm, 6)}')

    print('\nconstraints imposed inside the solve:')
    print(sc.describe(CONSTRAINTS))

    solved = None
    for n_aux in range(1, MAX_AUX + 1):
        n = n_sig + n_aux
        node_types = ['cavity'] * n
        drains = [i for i in range(n) if i not in target_mode_ids]
        root = full_graph(n)          # complete graph; passivity keeps only BS

        print('\n' + '=' * 78)
        print(f'{n_aux} AUXILIARY MODE(S)   ({n} modes, drains = {drains})')
        print('=' * 78)

        detect, d0, n_h, n_d = make_detector(root, V_t, target_mode_ids,
                                             node_types, n_aux)
        print(f'  passive basis: {n_h} Hamiltonian dof, {n_d} dissipator dof; '
              f'generic nullity d0 = {d0}')

        # 1. shared-squeezing sweep, both orientations
        print('\n  -- shared-squeezing sweep (all drains at the same r) --')
        grid = np.arange(-1.5, 1.5001, 0.05)
        vals = [(detect(np.array([v for _ in range(n_aux) for v in (r, 0.0)])), r)
                for r in grid]
        vals.sort()
        print('     lowest rank-drop scores: ' +
              ',  '.join(f'r={r:+.2f}: {s:.2e}' for s, r in vals[:5]))

        # 2. seeded refine over independent per-drain states
        t0 = time.time()
        score, p_star = refine(detect, seeds_for(n_aux, r_bm))
        raw = detect(p_star)
        print(f'\n  -- seeded refine over independent drain states '
              f'[{time.time() - t0:.1f}s] --')
        print(f'     best drain states : {aux_str(p_star)}')
        print(f'     rank-drop score   : {raw:.3e}  '
              f'({"RANK DROP — a usable drain state" if raw < 1e-9 else "no rank drop"})')

        # 3. decide at the candidates that matter
        candidates = [('vacuum', np.zeros(2 * n_aux)),
                      ('refined', np.asarray(p_star, dtype=float))]
        for sign in (+1, -1):
            for th in (0.0, np.pi / 2):
                picks = list(r_bm[:n_aux]) if n_aux <= len(r_bm) else list(r_bm)
                if len(picks) == n_aux:
                    candidates.append(
                        (f'Bloch-Messiah r{"+" if sign > 0 else "-"} th={th:.2f}',
                         np.array([v for r in picks for v in (sign * r, th)])))
        for _, r in vals[:3]:
            candidates.append((f'shared r={r:+.2f}',
                               np.array([v for _ in range(n_aux) for v in (r, 0.0)])))

        print('\n  -- oracle at the candidate drain states --')
        hit = None
        for label, p in candidates:
            aux = [(float(p[2 * k]), float(p[2 * k + 1])) for k in range(n_aux)]
            V = complete_covariance(V_t, target_mode_ids, n, aux_squeezing=aux)
            out = sc.constrained_decide(root, V, target_mode_ids, node_types,
                                        constraints=CONSTRAINTS, **SCAN_KW)
            g = out.get('gap')
            print(f'     {label:<28} {out["verdict"]:<10} '
                  f'{"" if g is None else f"gap {g:.3e}"}  '
                  f'detector {detect(p):.2e}')
            if out['verdict'] == VALID and hit is None:
                hit = (label, aux, V, out)

        if hit is None:
            print('\n     no drain state found that works at this drain count '
                  '(a scan proves nothing between its points, so this is not '
                  'an impossibility result)')
            continue

        label, aux, V, info = hit
        print(f'\n  >>> VALID at {label}: {aux_str(np.array([v for p in aux for v in p]))}')
        solved = (n_aux, n, node_types, aux, V, info)
        break

    if solved is None:
        print(f'\nno passive scheme found up to {MAX_AUX} drains')
        return

    n_aux, n, node_types, aux, V, info = solved
    root = full_graph(n)
    report_scheme(root, info, n, node_types, target_mode_ids,
                  label=f'complete graph on {n} modes, passive H, '
                        f'squeezed drains')

    print('\n' + '=' * 78)
    print('SPARSIFYING')
    print('=' * 78)
    red = reduce_witness(root, info, n)
    print(f'  witness-reduced: {edge_string(red, n)}')
    t0 = time.time()
    lean = sparsify(red, V, target_mode_ids, node_types, CONSTRAINTS)
    print(f'  greedy:          {edge_string(lean, n)}   [{time.time() - t0:.1f}s]')

    final = sc.constrained_decide(lean, V, target_mode_ids, node_types,
                                  constraints=CONSTRAINTS)
    report_scheme(lean, final, n, node_types, target_mode_ids,
                  label='SPARSE PASSIVE SCHEME with squeezed drains')

    print('\n  per-channel reservoir read-off (factorised from Upsilon):')
    for k, ch in enumerate(factor_dissipator(final['Upsilon'], n)):
        for p in ch['modes']:
            print(f'    channel {k}: mode {p["mode"]}  rate {p["rate"]:+.4f}  '
                  f'squeezing s = {p["squeezing"]:.6f}  '
                  f'phase {p["phase"]:+.4f}')
    print(f'\n  requested drain states : {aux_str(np.array([v for p in aux for v in p]))}')
    print(f'  Bloch-Messiah spectrum : '
          f'{np.round(np.sort(bloch_messiah_spectrum(purify_xp_block(V_GIVEN, 3)))[::-1], 6)}')

    A = drift_matrix(final['G'], final['Upsilon'])
    print(f'\n  A spectrum: {np.round(np.sort(np.real(np.linalg.eigvals(A))), 5)}')


if __name__ == '__main__':
    main()
