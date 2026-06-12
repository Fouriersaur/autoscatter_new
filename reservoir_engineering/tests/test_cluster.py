"""
test_cluster.py
===============
End-to-end test: can the BFS find topologies that dissipatively engineer
a 3-node linear chain Gaussian cluster state?

System
------
    node_types       = ['cavity', 'mechanical', 'mechanical', 'mechanical']
    target_mode_ids  = [1, 2, 3]   (3 mechanical modes = cluster nodes)
    Target           : cluster_state(n_modes=3, delta=0.5)  — linear chain graph

Cluster state recap
-------------------
    sigma_cluster = S_CZ @ sigma_product @ S_CZ.T
    sigma_product = block-diag of 3 p-squeezed modes: diag(e^{+2δ}/2, e^{-2δ}/2)
    S_CZ          = CZ gates along chain edges (0,1) and (1,2)
    Nullifiers    : f_j = p_j - sum_k A[j,k] x_k
    Ideal Var(f_j): e^{-2δ}/2  for all j

Physics checks
--------------
  - sigma_target is physical (symplectic eigenvalues ≥ ½)
  - Nullifier variances on target match e^{-2δ}/2 exactly
  - At least 1 topology found by BFS
  - No parametric drives on diagonal triu slots
  - Nullifier variances on achieved state < 2× ideal
  - Cooperativity ratios reported (not absolute)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (
    cluster_state, nullifier_variances, is_physical, symplectic_eigenvalues
)
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import NO_COUPLING


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


def _chain_adjacency(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    return A


def run_cluster_test(delta=0.5, num_tests=20, verbosity=False):
    n_cluster  = 3
    node_types = ['cavity'] + ['mechanical'] * n_cluster

    print(f'\n{"="*60}')
    print(f'  Cluster state engineering test  (delta={delta}, N={n_cluster} chain)')
    print(f'{"="*60}')

    adjacency    = _chain_adjacency(n_cluster)
    sigma_target = cluster_state(n_cluster, delta, adjacency)

    ideal_nullifier_var = 0.5 * np.exp(-2 * delta)

    print(f'\n  Target: {n_cluster}-node linear chain cluster state (delta={delta})')
    print(f'  sigma_target shape: {sigma_target.shape}')
    print(f'  sigma_target is physical: {is_physical(sigma_target)}')
    print(f'  Symplectic eigenvalues: {np.round(symplectic_eigenvalues(sigma_target), 4)}')
    nv_target = nullifier_variances(sigma_target, adjacency)
    print(f'  Nullifier variances on target: {np.round(nv_target, 6)}')
    print(f'  Ideal nullifier variance:      {ideal_nullifier_var:.6f}  (= e^{{-2δ}}/2)')

    print('\nStep 1: target sanity checks')
    all_pass = True
    all_pass &= check('sigma_target is physical', is_physical(sigma_target))
    all_pass &= check(
        'nullifier variances match e^{-2δ}/2',
        np.allclose(nv_target, ideal_nullifier_var, rtol=1e-6)
    )

    print('\nStep 2: build optimizer...')
    optimizer = CovarianceOptimizer(
        sigma_target         = sigma_target,
        target_mode_ids      = list(range(1, n_cluster + 1)),
        node_types           = node_types,
        num_auxiliary_modes  = 1,
        enforced_constraints = [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization  = dict(
            num_tests               = num_tests,
            interrupt_if_successful = True,
            max_violation_success   = 5e-4,
            log_ratio_bound         = 13.0,
        ),
        solver_options       = dict(maxiter=3000, ftol=0, gtol=1e-12),
        make_initial_test    = False,
    )
    print('  optimizer initialised OK')

    print('\nStep 3: breadth-first search...')
    optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        print(f'    triu_array = {list(t)}')

    print('\nStep 4: checks')

    all_pass &= check('at least 1 valid cluster topology found',
                      len(optimizer.valid_combinations) >= 1)

    if optimizer.valid_combinations:
        rows, cols   = np.triu_indices(len(node_types))
        diag_slots   = [k for k, (r, c) in enumerate(zip(rows, cols)) if r == c]
        diag_ok = all(int(t[k]) == NO_COUPLING
                      for t in optimizer.valid_combinations
                      for k in diag_slots)
        all_pass &= check('no parametric drives on any mode (diagonal triu = 0)', diag_ok)

    if optimizer.best_info_list:
        for idx, (triu, info) in enumerate(
                zip(optimizer.valid_combinations, optimizer.best_info_list)):
            sigma_achieved = info['sigma_achieved']
            loss           = info['final_cost']
            max_coop       = max(info.get('cooperativities', {1: 0}).values())

            print(f'\n  ── Topology {idx+1}: {list(triu)}')
            print(f'     loss={loss:.2e}  max_C={max_coop:.2e}')

            t_ok = True
            t_ok &= check('  loss < 5e-4', loss < 5e-4)

            nv_achieved = nullifier_variances(sigma_achieved, adjacency)
            print(f'     Nullifier variances (achieved): {np.round(nv_achieved, 6)}')
            print(f'     Nullifier variances (ideal)   : {ideal_nullifier_var:.6f}')
            t_ok &= check(
                '  all nullifier variances < 2× ideal',
                bool(np.all(nv_achieved < 2 * ideal_nullifier_var))
            )

            coops = info.get('cooperativities', {})
            if coops:
                c_ref_key = max(coops, key=lambda k: coops[k])
                c_ref_val = coops[c_ref_key]
                print(f'     Cooperativity ratios (ref = edge {c_ref_key}, C_ref = {c_ref_val:.2e}):')
                for k, v in coops.items():
                    print(f'       C_{k} / C_ref = {v / c_ref_val:.4f}   (C_{k} = {v:.2e})')

            print('     Detunings:', {k: f'{v:.2f}' for k, v in info.get('detunings', {}).items()})

            if verbosity:
                print('\n     Achieved σ:')
                print(np.round(sigma_achieved, 4))

            all_pass &= t_ok

    print(f'\n{"─"*60}')
    if all_pass:
        print('  ALL CHECKS PASSED — BFS found cluster state engineering topology')
    else:
        print('  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


if __name__ == '__main__':
    run_cluster_test(delta=0.5, num_tests=20)
