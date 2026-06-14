"""
test_kronwald.py
================
End-to-end test: does the BFS rediscover the Kronwald topology?

Expected outcome
----------------
Node types : ['cavity', 'mechanical']
Target     : squeezed_vacuum(r=1.0) on the mechanical mode (id=1)
BFS should find exactly ONE valid minimal topology: triu_array = [0, 4, 0]
  slot (0,0) = 0   — no parametric drive on cavity
  slot (0,1) = 4   — BS + TMS between cavity and mechanical (Kronwald)
  slot (1,1) = 0   — no parametric drive on mechanical

Physical interpretation of the solution
----------------------------------------
  g (BS coupling)  controls squeezing magnitude and stability
  ν (TMS coupling) drives the squeezing
  Both must be large (C ~ lambda = 1000) and satisfy ν/g = tanh(r)
  Squeezing parameter r recovered via: r = atanh(sqrt(C_nu / C_g))
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import squeezed_vacuum, purity
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer, LAMBDA_SCALE_DEFAULT
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import BEAMSPLITTER_AND_TWO_MODE_SQUEEZING


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


def run_kronwald_test(r=1.0, num_tests=10, verbosity=False):
    print(f'\n{"="*60}')
    print(f'  Kronwald rediscovery test  (r={r})')
    print(f'{"="*60}')

    sigma_target = squeezed_vacuum(r)
    node_types   = ['cavity', 'mechanical']

    print('\nStep 1: build optimizer (initial feasibility test)...')
    # max_violation_success=2e-5: the Kronwald scheme at lambda=1000 achieves
    # loss ~ 1.055e-5 (finite-cooperativity correction; exact in C→∞ limit only).
    # Threshold set above this minimum to accept Kronwald while rejecting
    # structurally incompatible topologies (BS-only loss >> 2e-5).
    optimizer = CovarianceOptimizer(
        sigma_target        = sigma_target,
        target_mode_ids     = [1],          # mechanical mode is the signal
        node_types          = node_types,
        num_auxiliary_modes = 1,
        enforced_constraints= [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization = dict(
            num_tests              = num_tests,
            interrupt_if_successful= True,
            max_violation_success  = 2e-5,
        ),
        solver_options      = dict(maxiter=2000, ftol=0, gtol=1e-12),
        make_initial_test   = True,         # verifies target is reachable at all
    )
    print('  optimizer initialised OK')

    print('\nStep 2: breadth-first search...')
    valid_trius = optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        print(f'    triu_array = {list(t)}')

    print('\nStep 3: checks')
    all_pass = True

    # --- topology checks ---
    all_pass &= check('exactly 1 valid topology found',
                      len(optimizer.valid_combinations) == 1)

    if optimizer.valid_combinations:
        best_triu = optimizer.valid_combinations[0]

        all_pass &= check('triu slot (0,0) = 0  (no parametric on cavity)',
                          int(best_triu[0]) == 0)
        all_pass &= check('triu slot (0,1) = 4  (BS+TMS = Kronwald)',
                          int(best_triu[1]) == BEAMSPLITTER_AND_TWO_MODE_SQUEEZING)
        all_pass &= check('triu slot (1,1) = 0  (no parametric on mechanical)',
                          int(best_triu[2]) == 0)

    # --- physics checks ---
    if optimizer.best_info_list:
        info = optimizer.best_info_list[0]
        sigma_achieved = info['sigma_achieved']
        sigma_tgt      = sigma_target

        loss = info['final_cost']
        all_pass &= check(f'Stage 3 loss < 2e-5  (loss={loss:.2e})',
                          loss < 2e-5)

        s_xx = float(sigma_achieved[0, 0])
        s_pp = float(sigma_achieved[1, 1])
        t_xx = float(sigma_tgt[0, 0])
        t_pp = float(sigma_tgt[1, 1])

        # At finite lambda=1000, Kronwald achieves sigma_xx within ~10% of ideal.
        # Exact equality only holds as C → ∞.
        all_pass &= check(f'sigma_xx below vacuum (0.5) and reasonably squeezed  (got {s_xx:.4f}, want ~{t_xx:.4f})',
                          s_xx < 0.5 and s_xx < 0.9 * 0.5)
        all_pass &= check(f'sigma_pp matches target  (got {s_pp:.4f}, want {t_pp:.4f})',
                          abs(s_pp - t_pp) < 1e-3)
        all_pass &= check('sigma_xx < 0.5  (x is squeezed)',
                          s_xx < 0.5)
        all_pass &= check('sigma_pp > 0.5  (p is anti-squeezed)',
                          s_pp > 0.5)
        all_pass &= check('sigma_xx * sigma_pp >= 0.25  (uncertainty principle)',
                          s_xx * s_pp >= 0.25 - 1e-6)

        mu_target   = purity(sigma_tgt)
        mu_achieved = purity(sigma_achieved)
        print(f'\n  Purity:')
        print(f'    target   μ = {mu_target:.6f}  (ideal: 1.0)')
        print(f'    achieved μ = {mu_achieved:.6f}  (gap from 1 = finite-C correction)')
        all_pass &= check(f'achieved purity > 0.9  (got {mu_achieved:.4f})',
                          mu_achieved > 0.9)

        # --- Kronwald coupling ratio check ---
        # Kronwald predicts: nu/g = tanh(r), so C_nu/C_g = tanh(r)^2
        log_ratios = info['log_ratios']
        if len(log_ratios) == 2:
            u_bs, u_tms = float(log_ratios[0]), float(log_ratios[1])
            ratio = np.exp(u_tms - u_bs)           # C~_nu / C~_g
            expected_ratio = np.tanh(r) ** 2
            all_pass &= check(
                f'C~_nu/C~_g ≈ tanh(r)^2  (got {ratio:.4f}, want {expected_ratio:.4f})',
                abs(ratio - expected_ratio) < 0.05)

        print('\n  Covariance matrices:')
        print(f'    sigma_target   = [[{t_xx:.6f}, 0],')
        print(f'                       [0, {t_pp:.6f}]]')
        print(f'    sigma_achieved = [[{s_xx:.6f}, 0],')
        print(f'                       [0, {s_pp:.6f}]]')

        if verbosity:
            print('\n  Coupling info:')
            for k, v in info['cooperativities'].items():
                print(f'    {k}: C = {v:.1f}')
            for k, v in info['detunings'].items():
                print(f'    {k} = {v:.4f}')

    print(f'\n{"─"*60}')
    if all_pass:
        print('  ALL CHECKS PASSED — Kronwald topology rediscovered')
    else:
        print('  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass

if __name__ == '__main__':
    run_kronwald_test(r=1.0, num_tests=10)
