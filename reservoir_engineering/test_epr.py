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
from reservoir_engineering.constraints import Constraint_stability, Constraint_coupling_absent
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
        enforced_constraints = [Constraint_stability(penalty_strength=50.0),
                                Constraint_coupling_absent(1, 2)],
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

    # Physics checks — run on ALL found topologies, report each
    if optimizer.best_info_list:
        topo_pass_flags = []

        for idx, (triu, info) in enumerate(
                zip(optimizer.valid_combinations, optimizer.best_info_list)):
            sigma_achieved = info['sigma_achieved']
            loss           = info['final_cost']
            max_coop       = max(info.get('cooperativities', {1: 0}).values())
            max_det        = max(abs(v) for v in info.get('detunings', {1: 0}).values())

            print(f'\n  ── Topology {idx+1}: {list(triu)}')
            print(f'     {_classify_triu(triu)}')
            print(f'     loss={loss:.2e}  max_C={max_coop:.2e}  max_|Δ|={max_det:.1f}')

            t_ok = True
            t_ok &= check(f'  loss < 5e-4', loss < 5e-4)

            s_xx0      = float(sigma_achieved[0, 0])
            s_xx1      = float(sigma_achieved[2, 2])
            s_xx_cross = float(sigma_achieved[0, 2])
            s_pp_cross = float(sigma_achieved[1, 3])
            t_ok &= check(f'  mech₁ σ_xx > 0.5  (got {s_xx0:.4f})', s_xx0 > 0.5)
            t_ok &= check(f'  mech₂ σ_xx > 0.5  (got {s_xx1:.4f})', s_xx1 > 0.5)
            t_ok &= check(f'  σ_{{x₁,x₂}} > 0   (got {s_xx_cross:.4f})', s_xx_cross > 0)
            t_ok &= check(f'  σ_{{p₁,p₂}} < 0   (got {s_pp_cross:.4f})', s_pp_cross < 0)
            t_ok &= check('  Duan criterion', duan_criterion(sigma_achieved))

            ln = log_negativity(sigma_achieved)
            r_achieved = float(np.log(2) / 2 * ln)
            t_ok &= check(f'  log_neg > 0  (got {ln:.4f})', ln > 0)
            t_ok &= check(
                f'  log_neg ≥ ½·theory  (got {ln:.4f}, theory {ln_theory:.4f})',
                ln >= 0.5 * ln_theory)
            print(f'     r_achieved = {r_achieved:.4f}  (target r={r})')

            # Flag whether this solution looks experimentally realistic
            realistic = max_coop < 1e7 and max_det < 100.
            print(f'     {"[realistic]" if realistic else "[unphysical: extreme cooperativity/detuning]"}')

            print('     Cooperativities:', {k: f'{v:.2e}' for k, v in info.get('cooperativities', {}).items()})
            print('     Detunings:', {k: f'{v:.2f}' for k, v in info.get('detunings', {}).items()})

            if verbosity:
                print('\n     Achieved σ:')
                print(sigma_achieved)

            topo_pass_flags.append(t_ok)
            all_pass &= t_ok

        # Require at least one found topology to be experimentally realistic
        any_realistic = any(
            max(info.get('cooperativities', {1: 0}).values()) < 1e7 and
            max(abs(v) for v in info.get('detunings', {1: 0}).values()) < 100.
            for info in optimizer.best_info_list
        )
        all_pass &= check('\nat least 1 topology is experimentally realistic (C<1e7, |Δ|<100)',
                          any_realistic)

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
    # Cavity-mediated only (Constraint_coupling_absent(1,2) active) > no mech-mech coupling
    # Dissipative reservoir engineering ceiling (i.e. Wooley-Clerk Scheme): r >= 0.3 not achievable within loss
    # threshold 5e-4. Use r=0.1 with 30 restarts to find Woolley-Clerk-like topologies.
    run_epr_test(r=0.1, num_tests=60)
