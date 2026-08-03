"""
test_direct_parametrization.py
================================
Tests for the §1.5 literal (G-tilde, C-tilde) parametrisation — the only
coupling parametrisation CovarianceOptimizer uses: u_k=log(g_k/kappa0) per
coherent edge, v_m=log(decay_m/kappa0) per AUXILIARY node's dissipation
rate. No lambda, no decay_i*decay_j reconstruction, so this works
identically whether a signal mode's decay rate is exactly 0 or not (an
earlier cooperativity-ratio parametrisation, g_k=sqrt(lambda*C~_k*
decay_i*decay_j/4), was undefined at decay=0 and has been removed).

Test 1 — gamma=0 (the whole point)
------------------------------------
Node types : ['cavity', 'mechanical'], mechanical gamma=0.0 EXACTLY. This
target is unreachable to even express under a cooperativity-based scheme
(C=4g^2/(kappa*gamma) is 0/0 at gamma=0). Expected: BFS still finds
Kronwald [0,4,0], and because gamma=0 truly permits the exact dark state
(no residual mixedness — §1.6), loss should be extremely small (machine
precision, not just "small").

Test 2 — finite gamma
------------------------
Same target, gamma=0.01 (a realistic hardware value). Same topology,
comparably tiny loss — confirms this parametrisation isn't a gamma=0
special case, it's a strict generalisation that works at any decay rate.

Test 3 — aux_node_ids
------------------------
Confirms CovarianceOptimizer automatically treats every mode NOT in
target_mode_ids as auxiliary (free C-tilde rate), and signal modes keep
their fixed (possibly zero) decay rate.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import squeezed_vacuum, purity
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import BEAMSPLITTER_AND_TWO_MODE_SQUEEZING


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


def run_gamma_zero_test(r=1.0, num_tests=8):
    print(f'\n{"="*60}')
    print(f'  gamma=0 EXACTLY  (r={r})')
    print(f'{"="*60}')

    sigma_tgt = squeezed_vacuum(r)

    optimizer = CovarianceOptimizer(
        sigma_target          = sigma_tgt,
        target_mode_ids       = [1],
        node_types            = ['cavity', 'mechanical'],
        num_auxiliary_modes   = 1,
        gamma                 = 0.0,                  # <-- the point
        enforced_constraints  = [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization   = dict(num_tests=num_tests, max_violation_success=1e-8),
        make_initial_test     = True,
    )

    all_pass = True
    all_pass &= check('aux_node_ids == [0]  (cavity is the only non-signal mode)',
                      optimizer.aux_node_ids == [0])

    valid_trius = optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        print(f'    triu_array = {list(t)}')

    all_pass &= check('exactly 1 valid topology found',
                      len(optimizer.valid_combinations) == 1)
    if optimizer.valid_combinations:
        best_triu = optimizer.valid_combinations[0]
        all_pass &= check('triu slot (0,1) = 4  (BS+TMS = Kronwald)',
                          int(best_triu[1]) == BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)

    if optimizer.best_info_list:
        info = optimizer.best_info_list[0]
        loss = info['final_cost']
        s = info['sigma_achieved']
        print(f'\n  loss = {loss:.3e}  (threshold 1e-8; at gamma=0 the exact dark')
        print(f'  state is reachable in principle, so this should be TINY)')
        print(f'  achieved sigma_mech = [[{s[0,0]:.8f}, 0], [0, {s[1,1]:.8f}]]')
        print(f'  target sigma_mech   = [[{sigma_tgt[0,0]:.8f}, 0], [0, {sigma_tgt[1,1]:.8f}]]')
        print(f"  G_tilde:     {info['G_tilde']}")
        print(f"  C_tilde_aux: {info['C_tilde_aux']}")

        all_pass &= check(f'Stage 2 loss < 1e-8  (loss={loss:.2e})', loss < 1e-8)
        all_pass &= check('achieved sigma_xx matches target to 4 decimals',
                          abs(s[0, 0] - sigma_tgt[0, 0]) < 1e-4)
        all_pass &= check('achieved sigma_pp matches target to 4 decimals',
                          abs(s[1, 1] - sigma_tgt[1, 1]) < 1e-4)
        all_pass &= check('achieved purity > 0.9999  (near-exact at gamma=0)',
                          purity(s) > 0.9999)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


def run_finite_gamma_test(r=1.0, num_tests=8, gamma=0.01):
    print(f'\n{"="*60}')
    print(f'  finite gamma={gamma}  (r={r})')
    print(f'{"="*60}')

    sigma_tgt = squeezed_vacuum(r)
    optimizer = CovarianceOptimizer(
        sigma_target          = sigma_tgt,
        target_mode_ids       = [1],
        node_types            = ['cavity', 'mechanical'],
        num_auxiliary_modes   = 1,
        gamma                 = gamma,
        enforced_constraints  = [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization   = dict(num_tests=num_tests, max_violation_success=2e-5),
        make_initial_test     = True,
    )
    valid = optimizer.perform_breadth_first_search()
    print(f'\n  valid: {[list(t) for t in valid]}')

    all_pass = True
    all_pass &= check('exactly 1 valid topology, triu == [0,4,0]',
                      len(optimizer.valid_combinations) == 1 and
                      list(optimizer.valid_combinations[0]) == [0, 4, 0])
    if optimizer.best_info_list:
        loss = optimizer.best_info_list[0]['final_cost']
        print(f'  loss = {loss:.3e}')
        all_pass &= check(f'loss < 2e-5', loss < 2e-5)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


def run_aux_node_ids_test():
    print(f'\n{"="*60}')
    print(f'  aux_node_ids inference test (3-mode: 1 cavity aux + 2 signal)')
    print(f'{"="*60}')

    all_pass = True
    optimizer = CovarianceOptimizer(
        sigma_target          = squeezed_vacuum(0.3),  # placeholder, not driven to BFS here
        target_mode_ids       = [0, 2],   # modes 0 and 2 are signal; mode 1 is auxiliary
        node_types            = ['mechanical', 'cavity', 'mechanical'],
        gamma                 = 0.0,
        make_initial_test     = False,
    )
    all_pass &= check('aux_node_ids == [1]  (only the cavity, not the two signal modes)',
                      optimizer.aux_node_ids == [1])
    all_pass &= check("signal node 0 keeps gamma=0.0 (not overridden)",
                      optimizer.nodes[0]['gamma'] == 0.0)
    all_pass &= check("signal node 2 keeps gamma=0.0 (not overridden)",
                      optimizer.nodes[2]['gamma'] == 0.0)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


if __name__ == '__main__':
    results = {
        'gamma_zero':    run_gamma_zero_test(r=1.0, num_tests=8),
        'finite_gamma':  run_finite_gamma_test(r=1.0, num_tests=8),
        'aux_node_ids':  run_aux_node_ids_test(),
    }

    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for name, passed in results.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')
    if all(results.values()):
        print('\n  ALL TEST GROUPS PASSED')
    else:
        print('\n  SOME TEST GROUPS FAILED')
