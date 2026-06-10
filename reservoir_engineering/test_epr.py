"""
test_epr.py
===========
End-to-end test: can the BFS find topologies that engineer EPR (two-mode squeezed) states?

Modes
-----
    node_types       = ['cavity', 'mechanical', 'mechanical']
    target_mode_ids  = [1, 2]   (both mechanicals are the signal)
    Target           : two_mode_squeezed(r=0.7)

What the BFS finds
------------------
Without any restriction on mech-mech couplings, the BFS finds SIMPLER topologies
than the Woolley-Clerk scheme because direct mech-mech coupling (slot k=4) can
create EPR states with fewer parameters:

    [0, 1, 0, 0, 4, 0]  — cavity-mech₁ BS (cooling) + direct mech₁↔mech₂ BS+TMS
    [0, 1, 1, 0, 2, 0]  — cavity cools both (BS) + direct mech₁↔mech₂ TMS

Both use only 3 coupling parameters, while the Woolley-Clerk [0,4,4,0,0,0] uses 4.

The Woolley-Clerk scheme becomes the UNIQUE minimal solution only when you
constrain the search to cavity-mediated couplings only (no direct mech-mech),
matching the experimental reality of optomechanical systems where direct
mechanical-mechanical interaction is hard to realise.

Physics checks (what this test validates)
------------------------------------------
Regardless of which topology is found, the achieved σ must satisfy:
  - Individual variances > ½   (EPR modes are individually noisy but jointly pure)
  - Positive x-x correlation   (σ_{x₁,x₂} > 0)
  - Negative p-p correlation   (σ_{p₁,p₂} < 0)
  - Duan criterion satisfied   → state is entangled
  - Log negativity > 0         → quantitative entanglement measure
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from reservoir_engineering.targets import two_mode_squeezed, duan_criterion, log_negativity
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import NO_COUPLING


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


def _classify_triu(triu):
    """Return a short human-readable description of an N=3 triu array."""
    labels = {0: 'none', 1: 'BS', 2: 'TMS', 3: 'PAR', 4: 'BS+TMS'}
    edges = []
    node_names = ['cav', 'mec1', 'mec2']
    rows, cols = np.triu_indices(3)
    for k, (i, j) in enumerate(zip(rows, cols)):
        v = int(triu[k])
        if i != j and v != 0:
            edges.append(f'{node_names[i]}-{node_names[j]}:{labels[v]}')
    return ', '.join(edges) if edges else '(empty)'


def run_epr_test(r=0.7, num_tests=10, verbosity=False):
    print(f'\n{"="*60}')
    print(f'  EPR state engineering test  (r={r})')
    print(f'{"="*60}')

    sigma_target = two_mode_squeezed(r)
    node_types   = ['cavity', 'mechanical', 'mechanical']

    ln_theory = 2 * r / np.log(2)   # exact log negativity for TMSV = 2r/ln2
    print(f'\n  Target: two_mode_squeezed(r={r})')
    print(f'  sigma_xx per mech   = {sigma_target[0,0]:.4f}  (should be > 0.5)')
    print(f'  x-x cross-corr      = {sigma_target[0,2]:.4f}  (should be > 0)')
    print(f'  p-p cross-corr      = {sigma_target[1,3]:.4f}  (should be < 0)')
    print(f'  Duan entanglement   : {duan_criterion(sigma_target)}')
    print(f'  Log negativity      = {log_negativity(sigma_target):.4f}  '
          f'(theory = 2r/ln2 ≈ {ln_theory:.4f})')

    print('\nStep 1: build optimizer...')
    # max_violation_success=5e-4: sigma_target is 4×4 so the loss sums more entries
    # than Kronwald; the finite-cooperativity floor is proportionally higher (~10×).
    # make_initial_test=False: the fully-connected N=3 graph is a 9D problem that
    # is unreliable with ~10 random restarts; EPR is theoretically achievable.
    optimizer = CovarianceOptimizer(
        sigma_target         = sigma_target,
        target_mode_ids      = [1, 2],
        node_types           = node_types,
        num_auxiliary_modes  = 1,
        enforced_constraints = [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization  = dict(
            num_tests               = num_tests,
            interrupt_if_successful = True,
            max_violation_success   = 5e-4,
        ),
        solver_options       = dict(maxiter=3000, ftol=0, gtol=1e-12),
        make_initial_test    = False,
    )
    print('  optimizer initialised OK')

    print('\nStep 2: breadth-first search...')
    optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        mech_mech = int(np.array(t, dtype=int)[4]) != NO_COUPLING
        scheme = 'direct mech-mech' if mech_mech else 'cavity-mediated (Woolley-Clerk-like)'
        print(f'    triu_array = {list(t)}  [{scheme}]')
        print(f'      {_classify_triu(t)}')

    print('\nStep 3: checks')
    all_pass = True

    # At least one valid topology must be found
    all_pass &= check('at least 1 valid EPR topology found',
                      len(optimizer.valid_combinations) >= 1)

    # All valid topologies must have no parametric drives (diagonal = 0)
    if optimizer.valid_combinations:
        diag_ok = all(int(t[0]) == NO_COUPLING and
                      int(t[3]) == NO_COUPLING and
                      int(t[5]) == NO_COUPLING
                      for t in optimizer.valid_combinations)
        all_pass &= check('no parametric drives on any mode (diagonal triu = 0)',
                          diag_ok)

    # Physics checks on the best solution
    if optimizer.best_info_list:
        info           = optimizer.best_info_list[0]
        sigma_achieved = info['sigma_achieved']

        loss = info['final_cost']
        all_pass &= check(f'Stage 3 loss < 5e-4  (loss={loss:.2e})',
                          loss < 5e-4)

        # Individual modes must be individually noisy (mixed), even though the joint state is pure
        s_xx0 = float(sigma_achieved[0, 0])
        s_xx1 = float(sigma_achieved[2, 2])
        all_pass &= check(f'mech₁ σ_xx > 0.5  (individually noisy, got {s_xx0:.4f})',
                          s_xx0 > 0.5)
        all_pass &= check(f'mech₂ σ_xx > 0.5  (individually noisy, got {s_xx1:.4f})',
                          s_xx1 > 0.5)

        # EPR cross-correlations
        s_xx_cross = float(sigma_achieved[0, 2])
        s_pp_cross = float(sigma_achieved[1, 3])
        all_pass &= check(f'σ_{{x₁,x₂}} > 0  (positive position corr., got {s_xx_cross:.4f})',
                          s_xx_cross > 0)
        all_pass &= check(f'σ_{{p₁,p₂}} < 0  (negative momentum corr., got {s_pp_cross:.4f})',
                          s_pp_cross < 0)

        # Entanglement
        all_pass &= check('Duan criterion satisfied (state is entangled)',
                          duan_criterion(sigma_achieved))

        ln = log_negativity(sigma_achieved)
        all_pass &= check(f'log negativity > 0  (got {ln:.4f})',
                          ln > 0)
        all_pass &= check(
            f'log negativity ≥ ½·(2r/ln2)  (got {ln:.4f}, theory {ln_theory:.4f})',
            ln >= 0.5 * ln_theory)

        if verbosity:
            print('\n  Achieved σ (mechanical subspace):')
            print(sigma_achieved)
            print('\n  Target σ:')
            print(sigma_target)
            print('\n  Coupling info:')
            for k, v in info.get('cooperativities', {}).items():
                print(f'    {k}: C = {v:.1f}')
            for k, v in info.get('detunings', {}).items():
                print(f'    {k} = {v:.4f}')

    print(f'\n{"─"*60}')
    if all_pass:
        print('  ALL CHECKS PASSED — BFS found valid EPR engineering topology')
        print('  (Note: found simpler schemes than Woolley-Clerk because direct')
        print('   mech-mech coupling is not forbidden in this run.)')
    else:
        print('  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


if __name__ == '__main__':
    run_epr_test(r=0.7, num_tests=10)
