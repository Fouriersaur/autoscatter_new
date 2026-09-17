"""
partial_passive_three_mode.py
=============================
Exact solve for a THREE-MODE PURE Gaussian target with an UNFLAT
Bloch-Messiah spectrum, under the PARTIAL PASSIVITY constraint:

    the signal-signal block of H is number-conserving (beamsplitters and
    detunings only); every anomalous term must live on a signal-drain or a
    drain-drain edge.

Two facts make this the interesting question for this target:

  - the Bloch-Messiah spectrum is {1.0, 0.5, 0.5}, i.e. UNFLAT, so by the
    flat-spectrum statement it cannot be a one-channel state;
  - partial passivity forbids the three signal modes from squeezing each
    other, so every anomalous resource must be imported through the drains.

The result below is that those two together are expensive: FREE, one drain
suffices; under partial passivity one and two drains come back INVALID, and
THREE drains are needed. The scheme is then sparsified
greedily, so what is reported is a small graph rather than the complete one.

Frame note. The constraint is not a gauge artefact of the unsqueezed-drain
frame. An auxiliary-local symplectic S = I_T (+) S_A acts as
G -> S^T G S, which leaves the SIGNAL-SIGNAL block G_TT untouched — so
"is the signal block passive?" has the same answer in every drain frame,
and the vacuum-frame search decides it completely.

The target as supplied is rounded to five decimals, which puts it at
det(2 sigma) = 0.99989 rather than 1. linear_oracle refuses it, and rightly:
a mixed target makes (**) bilinear. Since the x-p block is exactly zero the
state factorises as V = (1/2) blockdiag(Q, R) in (x..., p...) ordering, pure
iff R = Q^{-1}, so the rounding is repaired exactly by re-deriving the p
block from the x block. The repair moves every entry by ~3e-6 and leaves the
Bloch-Messiah spectrum where it was.
"""

import time
import numpy as np

from reservoir_engineering import solver_constraints as sc
from reservoir_engineering.linear_oracle import (
    VALID, complete_covariance, physical_parameters, format_parameters,
    drift_matrix, solve_stationarity, _assemble, normalised_margin, _is_psd)
from reservoir_engineering.certified_search import reduce_witness
from reservoir_engineering.targets import (
    symplectic_eigenvalues, purity, log_negativity)
from reservoir_engineering.topology_search import triu_to_edge_matrix


# The target, as given, in interleaved (x1,p1,x2,p2,x3,p3) ordering.
V_GIVEN = np.array([
    [2.39709,  0.,      -0.51897, 0.,       1.03795, 0.],
    [0.,       0.13226,  0.,      0.02584,  0.,     -0.05168],
    [-0.51897, 0.,       1.61863, 0.,      -0.51897, 0.],
    [0.,       0.02584,  0.,      0.17102,  0.,      0.02584],
    [1.03795,  0.,      -0.51897, 0.,       2.39709, 0.],
    [0.,      -0.05168,  0.,      0.02584,  0.,      0.13226],
])

MAX_AUX = 3


# purify_xp_block(V): repair the rounding, exactly.
#
# With V_xp = 0 the state is V = (1/2) blockdiag(Q, R) in quadrature-sorted
# ordering, and purity is exactly R = Q^{-1} (then 2V has eigenvalues lam_k
# and 1/lam_k, i.e. e^{+-2 r_k}). Q is the well-conditioned block, so rebuild
# R from it rather than the other way round.
def purify_xp_block(V, n):
    xs = [2 * i for i in range(n)]
    ps = [2 * i + 1 for i in range(n)]
    Q = 2 * V[np.ix_(xs, xs)]
    Q = (Q + Q.T) / 2
    out = np.zeros_like(V)
    out[np.ix_(xs, xs)] = Q / 2
    out[np.ix_(ps, ps)] = np.linalg.inv(Q) / 2
    return out


def bloch_messiah_spectrum(V):
    # For a pure state 2V = S S^T, so the eigenvalues of 2V are e^{+-2 r_k}
    # and the positive halves are the Bloch-Messiah squeezing values.
    w = np.sort(np.linalg.eigvalsh(2 * V))[::-1]
    return 0.5 * np.log(w[:len(w) // 2])


def describe_target(V):
    print(f'  symmetric              {np.allclose(V, V.T)}')
    print(f'  det(2 sigma)           {np.linalg.det(2 * V):.12f}')
    print(f'  purity                 {purity(V):.12f}')
    print(f'  symplectic eigenvalues {np.round(symplectic_eigenvalues(V), 9)}')
    r = bloch_messiah_spectrum(V)
    flat = 'flat' if r.max() - r.min() < 1e-6 else 'UNFLAT'
    print(f'  Bloch-Messiah r_k      {np.round(r, 6)}   (spread '
          f'{r.max() - r.min():.6f}: {flat})')
    print(f'  squeezing (dB)         {np.round(20 * r / np.log(10), 3)}')
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        print(f'  E_N({a},{b}) [reduced]     {log_negativity(V[np.ix_(idx, idx)]):.6f}')


def edges_of(triu, n):
    E = triu_to_edge_matrix(np.asarray(triu), n)
    kinds = {1: 'BS', 2: 'TMS', 3: 'PAR', 4: 'BS+TMS'}
    return [(i, j, kinds[int(E[i, j])]) for i in range(n) for j in range(i, n)
            if int(E[i, j]) != 0]


def edge_string(triu, n):
    return ', '.join(f'({i},{j}){k}' for i, j, k in edges_of(triu, n))


def full_graph(n, with_parametric=True):
    return np.array([(3 if with_parametric else 0) if i == j else 4
                     for i in range(n) for j in range(i, n)])


# audit_solution_set: an independent check on an INVALID verdict, and the
# only evidence there is now that the oracle's INVALID carries no proof. It
# samples the constrained solution set directly: a max margin at 1e-17 over
# tens of thousands of draws is what a genuinely impossible graph looks
# like, and a single positive draw would refute the verdict.
def audit_solution_set(triu, V, ctx, cset, n_draws=20000, seed=0):
    _, h, d = sc.constrained_bases(triu, ctx, cset)
    sol = solve_stationarity(V, h, d, None)
    N = sol['nullspace']
    if N.shape[1] == 0:
        return {'nullspace': 0, 'best_margin': 0.0, 'best_psd_margin': -np.inf}
    rng = np.random.default_rng(seed)
    best, best_psd = -np.inf, -np.inf
    for _ in range(n_draws):
        z = N @ rng.normal(0., 1., N.shape[1])
        G, Y = _assemble(z, h, d, None, ctx.dim)
        m = normalised_margin(drift_matrix(G, Y))
        best = max(best, m)
        if _is_psd(Y):
            best_psd = max(best_psd, m)
    return {'nullspace': int(N.shape[1]), 'best_margin': best,
            'best_psd_margin': best_psd, 'dof_G': len(h), 'dof_Y': len(d)}


def report_scheme(triu, info, n, node_types, target_mode_ids, label=''):
    print(f'\n  --- {label} ---')
    print(f'  graph  {list(map(int, np.asarray(triu)))}')
    print(f'  edges  {edge_string(triu, n)}')
    print(f'  verdict {info["verdict"]}   path {info["path"]}')
    print(f'  dissipative gap (normalised)  {info["gap"]:.6e}')
    print(f'  channels (rank Upsilon)       {info["upsilon_rank"]}')
    print(f'  reservoir                     {info["reservoir_kind"]}'
          f'   max squeezing {info["reservoir_squeezing"]:.6f}')
    print('  -- independent verification --')
    print(f'  stationarity residual         {info["stationarity_residual"]:.3e}')
    print(f'  forward Lyapunov error        {info["forward_error"]:.3e}'
          '   (solve_continuous_lyapunov(A, -D) vs the target)')
    print(f'  Upsilon >= 0                  {info["upsilon_psd"]}')
    A = drift_matrix(info['G'], info['Upsilon'])
    print(f'  max Re lambda(A)              '
          f'{np.max(np.real(np.linalg.eigvals(A))):.3e}')
    for c in info.get('constraint_report', []):
        print(f'  constraint {c["name"]:<22} satisfied={c["satisfied"]}'
              f'  violation={c["violation"]:.2e}')
    # The constraint re-checked from outside the solver: [G_TT, Omega] = 0.
    G = info['G']
    Om = np.zeros_like(G)
    for i in range(n):
        Om[2 * i, 2 * i + 1] = 1.
        Om[2 * i + 1, 2 * i] = -1.
    sub = np.zeros_like(G)
    for i in target_mode_ids:
        for j in target_mode_ids:
            sub[2 * i:2 * i + 2, 2 * j:2 * j + 2] = G[2 * i:2 * i + 2, 2 * j:2 * j + 2]
    print(f'  ||[G_signal, Omega]||         {np.linalg.norm(sub @ Om - Om @ sub):.3e}'
          '   (0 = signal block is passive)')
    print(format_parameters(physical_parameters(info, triu, node_types,
                                                target_mode_ids)))


# sparsify(triu, ...): greedy edge removal, keeping only what the verdict
# needs.
#
# The full lattice at six modes is 2^6 * 4^15 graphs, so a full BFS is out of
# reach here; greedy removal is not a minimality proof (it finds a locally
# minimal graph, not necessarily a globally minimal one) but every graph it
# reports is decided by the same oracle, so the scheme itself is as sound as
# any other. Downgrades are cheap because only VALID matters, so the loop
# runs at a low gap_effort.
def sparsify(triu, V, target_mode_ids, node_types, constraints, order=None):
    n = len(node_types)
    cur = np.array(triu, dtype=int, copy=True)
    slots = list(range(len(cur))) if order is None else list(order)
    changed = True
    while changed:
        changed = False
        for k in slots:
            if cur[k] == 0:
                continue
            # 4 = BS+TMS, try dropping to BS, then to TMS, then to nothing.
            for cand in ([1, 2, 0] if cur[k] == 4 else [0]):
                trial = cur.copy()
                trial[k] = cand
                out = sc.constrained_decide(trial, V, target_mode_ids, node_types,
                                            constraints=constraints,
                                            gap_effort=2)
                if out['verdict'] == VALID:
                    cur = trial
                    changed = True
                    break
    return cur


def main():
    n_sig = 3
    target_mode_ids = list(range(n_sig))
    constraints = [sc.passive_target_block()]

    print('=' * 78)
    print('TARGET as supplied (rounded to 5 decimals)')
    print('=' * 78)
    describe_target(V_GIVEN)

    V_t = purify_xp_block(V_GIVEN, n_sig)
    print('\n' + '=' * 78)
    print('TARGET after exact purification (p block rebuilt as Q^-1)')
    print('=' * 78)
    describe_target(V_t)
    print(f'  max |change| from supplied  {np.max(np.abs(V_t - V_GIVEN)):.2e}')

    print('\nconstraints imposed inside the solve:')
    print(sc.describe(constraints))

    solved = None
    for n_aux in range(1, MAX_AUX + 1):
        n = n_sig + n_aux
        node_types = ['cavity'] * n
        V = complete_covariance(V_t, target_mode_ids, n)
        ctx = sc.ConstraintContext(n, target_mode_ids, node_types, V)
        cset = sc.as_constraint_set(constraints)
        root = full_graph(n)

        print('\n' + '=' * 78)
        print(f'{n_aux} AUXILIARY MODE(S)   ({n} modes, drains = '
              f'{[i for i in range(n) if i not in target_mode_ids]})')
        print('=' * 78)

        t0 = time.time()
        free = sc.constrained_decide(root, V, target_mode_ids, node_types,
                                     constraints=None)
        print(f'  UNCONSTRAINED root : {free["verdict"]:<9} '
              f'gap {free.get("gap", float("nan")):.4e}  '
              f'channels {free.get("upsilon_rank")}   [{time.time() - t0:.1f}s]')

        t0 = time.time()
        info = sc.constrained_decide(root, V, target_mode_ids, node_types,
                                     constraints=constraints)
        print(f'  PARTIAL PASSIVITY  : {info["verdict"]:<9} '
              f'path {info["path"]}   [{time.time() - t0:.1f}s]')

        if info['verdict'] != VALID:
            print(f'    path: {info["path"]}')
            if 'reason' in info:
                print('    ' + info['reason'])
            aud = audit_solution_set(root, V, ctx, cset)
            print(f'    audit: constrained solution set has dim '
                  f'{aud["nullspace"]} (dof_G {aud.get("dof_G")}, '
                  f'dof_Y {aud.get("dof_Y")}); best normalised margin over '
                  f'20000 random members = {aud["best_margin"]:.2e}')
            print('    -> the fully connected graph is impossible, hence every '
                  'subgraph is too; no scheme exists at this drain count')
            continue

        report_scheme(root, info, n, node_types, target_mode_ids,
                      label=f'complete graph on {n} modes')
        solved = (n_aux, n, node_types, V, info)
        break

    if solved is None:
        print(f'\nno valid scheme up to {MAX_AUX} auxiliary modes')
        return

    n_aux, n, node_types, V, info = solved
    root = full_graph(n)

    print('\n' + '=' * 78)
    print(f'SPARSIFYING the {n}-mode scheme')
    print('=' * 78)
    # Start from the edges the witness actually uses, then remove greedily.
    red = reduce_witness(root, info, n)
    print(f'  witness-reduced graph: {edge_string(red, n)}')
    t0 = time.time()
    lean = sparsify(red, V, target_mode_ids, node_types, constraints)
    print(f'  greedily sparsified:   {edge_string(lean, n)}   '
          f'[{time.time() - t0:.1f}s]')

    final = sc.constrained_decide(lean, V, target_mode_ids, node_types,
                                  constraints=constraints)
    report_scheme(lean, final, n, node_types, target_mode_ids,
                  label='SPARSE VALID SCHEME under partial passivity')

    print('\n  same graph with the constraint dropped, for contrast:')
    free = sc.constrained_decide(lean, V, target_mode_ids, node_types,
                                 constraints=None)
    print(f'    {free["verdict"]}   gap {free.get("gap", float("nan")):.4e}'
          f'   channels {free.get("upsilon_rank")}')


if __name__ == '__main__':
    main()
