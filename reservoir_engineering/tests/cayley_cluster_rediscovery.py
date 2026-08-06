"""
cayley_cluster_rediscovery.py
==============================
Standalone demo (not a test): dissipative engineering of a 4-MODE CAYLEY
CLUSTER STATE on the linear cluster 1-2-3-4 (path graph P4) — the same
exercise as vitali_rediscovery.py / vitali_rediscovery_unequal.py, but for a
4-mode target with an EQUAL (perfectly flat) Bloch-Messiah spectrum.

The target
----------
P4's adjacency eigenvalues are {+-(1+sqrt5)/2, +-(sqrt5-1)/2} — NOT of equal
magnitude, so the CANONICAL finite-squeezing cluster state on P4 (the one
targets.cluster_state builds: CZ gates on p-squeezed modes) does NOT have a
flat Bloch-Messiah spectrum. The Cayley state fixes that.

Construction (in xxpp ordering q1..q4,p1..p4, vacuum = I):
    M   = (A^2 + I)^-1
    K11 = (I - A^2) M          K12 = 2 A M         # K11^2 + K12^2 = I
    K   = [[K11,  K12],
           [K12, -K11]]
    sigma = exp(2 z K) = cosh(2z) I + sinh(2z) K   (exact, since K^2 = I)

K is symmetric, squares to I, and ANTICOMMUTES with the symplectic form, which
is precisely what makes exp(2zK) a pure Gaussian covariance: (Omega sigma)^2 =
-I. K11/K12 are the tangent-half-angle (Cayley) pair of the adjacency matrix,
so every eigenvalue of sigma is exactly e^{+2z} or e^{-2z} — a perfectly FLAT
BM spectrum with squeezing exactly z, on any graph, however irregular its
spectrum. Verified below to machine precision.

Two convention conversions are needed before handing this to the optimizer:
  ordering       xxpp (q1..q4,p1..p4)  ->  xpxp (x0,p0,x1,p1,...)
  normalisation  vacuum = I (symplectic eigenvalues 1)
                   -> vacuum = I/2 (this package: symplectic eigenvalues 1/2)
i.e. sigma_package = P sigma_xxpp P^T / 2.

Why no breadth-first topology search here
-----------------------------------------
The EPR demos search topologies exhaustively (3 nodes -> 512 graphs). This
target needs 4 signal modes + >=1 auxiliary = 5 nodes, i.e.
4^10 * 2^5 = 33,554,432 graphs (topology_search.calc_number_of_possibilities)
— days-to-weeks of Stage-2 solves, so perform_breadth_first_search() is not
usable at this size. Instead this file optimises a CURATED set of physically
motivated topologies (below) and ranks them. That is a weaker claim than the
EPR files make: it shows these schemes DO reach the target, not that no
simpler scheme exists.

The curated set is built around Zippilli & Vitali PRL 126, 020402 (2021),
which targets exactly this problem — mechanical cluster states stabilised by a
PASSIVE (beamsplitter-only) network plus a SINGLE squeezed-bath drain — so the
passive candidates are the ones to beat.

Usage:
    python3 cayley_cluster_rediscovery.py            # z=0.6 (the pictured state)
    python3 cayley_cluster_rediscovery.py 0.4        # any squeezing z
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (purity, symplectic_eigenvalues,
                                            is_physical, nullifier_variances)
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.covariance_physics import check_stability
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import (NO_COUPLING, BEAMSPLITTER,
    TWO_MODE_SQUEEZING, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING, calc_number_of_possibilities)

# Node convention: 0 = auxiliary cavity drain (squeezable bath), 1..4 = the
# four mechanical modes carrying the cluster state. Mechanical mode i holds
# graph vertex i, so the P4 chain 1-2-3-4 maps onto node pairs (1,2),(2,3),(3,4).
NODE_TYPES      = ['cavity'] + ['mechanical'] * 4
TARGET_MODE_IDS = [1, 2, 3, 4]
N_NODES         = len(NODE_TYPES)

# path graph P4 adjacency (the cluster graph)
ADJACENCY_P4 = np.array([[0, 1, 0, 0],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]], dtype=float)

# triu slot index for each node pair, for readable topology construction.
_ROWS, _COLS = np.triu_indices(N_NODES)
_SLOT = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(_ROWS, _COLS))}

DRAIN_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4)]      # drain -> every mechanical mode
CHAIN_EDGES = [(1, 2), (2, 3), (3, 4)]              # the P4 cluster chain itself


# Assemble a triu_array from {pair: edge_code} — everything unlisted is absent.
def make_triu(edge_map: dict) -> np.ndarray:
    triu = np.zeros(len(_ROWS), dtype=int)
    for (i, j), code in edge_map.items():
        triu[_SLOT[(min(i, j), max(i, j))]] = code
    return triu


# The curated candidates. Passive (beamsplitter-only) ones first — those are
# the Zippilli-Vitali class, where ALL the squeezing enters through the drain's
# bath and the coherent network merely distributes it.
def candidate_topologies() -> dict:
    bs_star  = {e: BEAMSPLITTER for e in DRAIN_EDGES}
    bs_chain = {e: BEAMSPLITTER for e in CHAIN_EDGES}
    return {
        'passive: drain->all4 BS + chain BS'     : make_triu({**bs_star, **bs_chain}),
        'passive: drain->all4 BS only'           : make_triu(bs_star),
        'passive: drain->mode1 BS + chain BS'    : make_triu({(0, 1): BEAMSPLITTER, **bs_chain}),
        'active:  drain->all4 BS+TMS + chain BS' : make_triu(
            {**{e: BEAMSPLITTER_AND_TWO_MODE_SQUEEZING for e in DRAIN_EDGES}, **bs_chain}),
        'active:  drain->all4 TMS + chain BS'    : make_triu(
            {**{e: TWO_MODE_SQUEEZING for e in DRAIN_EDGES}, **bs_chain}),
        'active:  drain->all4 BS + chain BS+TMS' : make_triu(
            {**bs_star, **{e: BEAMSPLITTER_AND_TWO_MODE_SQUEEZING for e in CHAIN_EDGES}}),
    }


# ── target construction ────────────────────────────────────────────────────

# Cayley cluster state for adjacency A at squeezing z, in xxpp ordering with
# vacuum = I (the convention the analytic formula is written in).
def cayley_cluster_xxpp(adjacency: np.ndarray, z: float) -> np.ndarray:
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    M = np.linalg.inv(A @ A + I)
    K11 = (I - A @ A) @ M
    K12 = 2 * A @ M
    K = np.block([[K11, K12],
                   [K12, -K11]])
    return np.cosh(2 * z) * np.eye(2 * n) + np.sinh(2 * z) * K


# xxpp (q1..qn,p1..pn) -> xpxp (x0,p0,x1,p1,...), the package's ordering.
def xxpp_to_xpxp(sigma: np.ndarray) -> np.ndarray:
    n = sigma.shape[0] // 2
    perm = np.empty(2 * n, dtype=int)
    perm[0::2] = np.arange(n)          # q_i -> slot 2i
    perm[1::2] = np.arange(n, 2 * n)   # p_i -> slot 2i+1
    return sigma[np.ix_(perm, perm)]


# The target in FULL package convention: xpxp ordering, vacuum = I/2.
def cayley_cluster_target(adjacency: np.ndarray, z: float) -> np.ndarray:
    return xxpp_to_xpxp(cayley_cluster_xxpp(adjacency, z)) / 2.0


# Confirm the constructed target really is the flat-BM pure cluster state:
# purity 1, all symplectic eigenvalues 1/2, and sigma's eigenvalues taking
# exactly the two reciprocal values e^{+-2z}/2 (n-fold each).
def validate_target(sigma: np.ndarray, adjacency: np.ndarray, z: float):
    n = sigma.shape[0] // 2
    eigs = np.sort(np.linalg.eigvalsh(sigma))
    nus = symplectic_eigenvalues(sigma)
    expected = np.array([np.exp(-2 * z) / 2] * n + [np.exp(2 * z) / 2] * n)
    return {
        'physical'          : is_physical(sigma),
        'purity'            : purity(sigma),
        'symplectic_max_dev': float(np.max(np.abs(nus - 0.5))),
        'eig_max_dev'       : float(np.max(np.abs(eigs - expected))),
        'eigenvalues'       : eigs,
        'nullifiers'        : nullifier_variances(sigma, adjacency),
    }


# ── search ─────────────────────────────────────────────────────────────────

# Optimise every curated topology (multi-restart) and keep each one's best
# STABLE solution. Stability is re-checked explicitly: Constraint_stability is
# only a soft penalty, and an unstable A makes the Lyapunov solve return a
# non-state that can still score a low target loss.
def search_topologies(target: np.ndarray, num_tests: int = 30, seed: int = 0,
                       squeezable: bool = True):
    results = []
    for name, triu in candidate_topologies().items():
        np.random.seed(seed)
        optimizer = CovarianceOptimizer(
            sigma_target=target,
            target_mode_ids=TARGET_MODE_IDS,
            node_types=NODE_TYPES,
            num_auxiliary_modes=1,
            gamma=0.0,
            squeezable_aux_ids=[0] if squeezable else [],
            enforced_constraints=[Constraint_stability(penalty_strength=100.0)],
            solver_options=dict(maxiter=3000, ftol=0, gtol=1e-14),
            make_initial_test=False,
        )
        _, infos, _ = optimizer.repeated_optimization(
            num_tests=num_tests, triu_array=triu, max_violation_success=1e-6,
            optimize_detunings=True, interrupt_if_successful=False)

        stable = [i for i in infos if check_stability(i['A'])]
        best = min(stable, key=lambda x: x['final_cost']) if stable else None
        results.append({'name': name, 'triu': triu, 'info': best,
                        'n_stable': len(stable), 'n_runs': len(infos)})
        tag = f'{best["final_cost"]:.3e}' if best else 'no stable solution'
        print(f'  {name:42s} -> {tag}   ({len(stable)}/{len(infos)} stable)')
    return results


# ── reporting ──────────────────────────────────────────────────────────────

def _describe_triu(triu) -> str:
    labels = {1: 'BS', 2: 'TMS', 3: 'PAR', 4: 'BS+TMS'}
    parts = []
    for (i, j), k in _SLOT.items():
        v = int(triu[k])
        if v:
            parts.append(f'({i},{j}){labels[v]}' if i != j else f'({i})PAR')
    return ' '.join(parts) if parts else '(empty)'


def print_report(z: float, target: np.ndarray, checks: dict, results: list):
    print(f'\n{"="*74}')
    print(f'  Cayley cluster state on P4 (1-2-3-4) — 4 modes, flat BM, z = {z}')
    print(f'{"="*74}\n')

    print(f'Adjacency eigenvalues of P4: '
          f'{np.array2string(np.sort(np.linalg.eigvalsh(ADJACENCY_P4)), precision=5)}')
    print('  (unequal magnitudes -> the CANONICAL cluster state on P4 is NOT flat-BM;')
    print('   the Cayley construction is what makes the spectrum flat)\n')

    print('Target validation (package convention: xpxp, vacuum = I/2):')
    print(f'  physical (all nu >= 1/2)      : {checks["physical"]}')
    print(f'  purity                        : {checks["purity"]:.12f}   (1 = pure)')
    print(f'  max |nu_k - 1/2|              : {checks["symplectic_max_dev"]:.3e}')
    print(f'  max |eig - e^(+-2z)/2|        : {checks["eig_max_dev"]:.3e}   -> FLAT BM spectrum')
    print(f'  eigenvalues of sigma          : {np.array2string(checks["eigenvalues"], precision=5)}')
    print(f'  e^(+2z)/2, e^(-2z)/2          : {np.exp(2*z)/2:.5f}, {np.exp(-2*z)/2:.5f}')
    print(f'  nullifier variances           : {np.array2string(checks["nullifiers"], precision=5)}\n')

    n_graphs = calc_number_of_possibilities(NODE_TYPES)
    print(f'Full topology enumeration for {N_NODES} nodes would be {n_graphs:,} graphs '
          f'— BFS not run (see docstring);\ninstead {len(results)} curated topologies were optimised.\n')

    ranked = sorted([r for r in results if r['info']], key=lambda r: r['info']['final_cost'])
    if not ranked:
        print('No topology produced a stable solution.')
        return

    for rank, r in enumerate(ranked):
        info, triu = r['info'], r['triu']
        sa = info['sigma_achieved']
        print(f'--- #{rank+1}  {r["name"]} ---')
        print(f'  edges: {_describe_triu(triu)}')
        print(f'  loss = {info["final_cost"]:.3e}   stable runs {r["n_stable"]}/{r["n_runs"]}')
        print(f'  purity(achieved) = {purity(sa):.8f}    '
              f'||achieved - target|| = {np.linalg.norm(sa - target):.3e}')
        print(f'  nullifier variances = '
              f'{np.array2string(nullifier_variances(sa, ADJACENCY_P4), precision=5)}')
        print('  couplings:')
        for label, gt in info['G_tilde'].items():
            th = info['coherent_phases'].get(label.replace('G~_', 'theta_'))
            th_s = f'  theta={th:+.4f}' if th is not None else '  (real)'
            print(f'    {label.replace("G~_",""):16s} G~ = {gt:12.4f}{th_s}')
        print('  detunings: ' + ', '.join(f'{k}={v:+.4f} kappa0' for k, v in info['detunings'].items()))
        # The drain's decay rate is itself a FREE variable (kappa_aux =
        # kappa0*exp(v_m)), not the `kappa` passed to the constructor — that
        # input is used only for SIGNAL cavities and is overwritten for
        # auxiliary ones. Report the physical rate, in units of kappa0, and the
        # g/kappa ratio that says which regime the solution sits in: g << kappa
        # is the bad-cavity/adiabatic limit where the drain acts as an
        # effective Markovian (squeezed) reservoir — the reservoir-engineering
        # regime of Woolley & Clerk PRA 89, 063805 (2014) sec. V.
        for nid, kap in info['aux_decay_rates'].items():
            g_max = max(info['coupling_strengths']) if len(info['coupling_strengths']) else 0.
            regime = 'bad-cavity/adiabatic' if g_max < kap else 'strong-coupling'
            print(f'  aux mode {nid} decay: kappa_aux = {kap:.4f} kappa0   '
                  f'(g_max/kappa_aux = {g_max/kap:.3f} -> {regime})')
        for nid, bs in info['bath_squeezing'].items():
            print(f'  bath squeeze (mode {nid}): r = {bs["r"]:.6f}  theta = {bs["theta"]:+.4f}')
        print()

    best = ranked[0]
    passive = all(int(v) in (NO_COUPLING, BEAMSPLITTER) for v in best['triu'])
    print(f'{"-"*74}')
    print(f'Best: {best["name"]}  (loss {best["info"]["final_cost"]:.3e})')
    print(f'  passive / beamsplitter-only network? {passive}'
          f'{"  -> Zippilli-Vitali class: all squeezing supplied by the drain bath" if passive else ""}')


if __name__ == '__main__':
    z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6

    target = cayley_cluster_target(ADJACENCY_P4, z)
    checks = validate_target(target, ADJACENCY_P4, z)

    print(f'optimising {len(candidate_topologies())} curated topologies '
          f'({N_NODES} nodes, 4 signal + 1 drain) ...')
    results = search_topologies(target, num_tests=30)
    print_report(z, target, checks, results)
