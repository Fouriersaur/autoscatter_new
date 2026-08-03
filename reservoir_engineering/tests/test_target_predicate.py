"""
test_target_predicate.py
=========================
End-to-end tests for the §2.1 target predicate: CovarianceOptimizer driven
by scalar TargetTerm quantities instead of (or alongside) a full
sigma_target covariance matrix.

Test 1 — predicate-only rediscovery
------------------------------------
Node types : ['cavity', 'mechanical']
Target     : ONLY a scalar TargetTerm — "mechanical mode's x-quadrature
             variance = squeezed_vacuum(r).sigma_xx" (target_collective_quadrature).
             No sigma_target matrix is ever constructed or passed.
Expected   : BFS still rediscovers the Kronwald topology triu_array=[0,4,0],
             exactly as with the full-matrix sigma_target path — proving the
             predicate alone is enough to drive the discrete+continuous search.

Test 2 — quadratic-form equivalence
------------------------------------
Decomposes squeezed_vacuum(r) into three target_quadratic_form terms (xx,
pp, xp entries) and checks this reproduces the same topology as the direct
sigma_target path — i.e. target_quadratic_form really does subsume
full-matrix matching as the doc's §6 "specific correlations" special case.
The xp term targets q=0, exercising the zero-target (non-relative-error)
branch of CovarianceOptimizer._predicate_residual.

Note on thresholds: this predicate uses PER-ENTRY RELATIVE error
((Q_i/q_i-1)^2), unlike sigma_target's absolute Frobenius error. Since
sigma_xx (~0.068) is ~50x smaller than sigma_pp (~3.69) for r=1.0, absolute
Frobenius error is dominated by sigma_pp and barely constrains sigma_xx (the
direct sigma_target path itself achieves only ~7% relative accuracy on
sigma_xx at lambda=1000, despite loss~1e-5 — verified directly). Relative
error weights both entries equally, which is a STRICTER combined
requirement at finite lambda — so this test uses a looser max_violation_success
than the sigma_target-only tests, calibrated to what finite-cooperativity
Kronwald actually achieves.

Test 3 — mixed target
----------------------
sigma_target AND target_predicate given together; checks the two residuals
combine (both must be satisfied) rather than one silently overriding the
other.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import (
    squeezed_vacuum, purity,
    target_collective_quadrature, target_quadratic_form, target_purity,
)
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.constraints import Constraint_stability
from reservoir_engineering.topology_search import BEAMSPLITTER_AND_TWO_MODE_SQUEEZING


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


def _kronwald_optimizer_kwargs(num_tests=10, max_violation_success=2e-5):
    return dict(
        target_mode_ids      = [1],
        node_types           = ['cavity', 'mechanical'],
        num_auxiliary_modes  = 1,
        enforced_constraints = [Constraint_stability(penalty_strength=50.0)],
        kwargs_optimization  = dict(
            num_tests               = num_tests,
            interrupt_if_successful = True,
            max_violation_success   = max_violation_success,
        ),
        solver_options        = dict(maxiter=2000, ftol=0, gtol=1e-12),
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — predicate-only rediscovery (no sigma_target at all)
# ─────────────────────────────────────────────────────────────────────────

def run_predicate_only_test(r=1.0, num_tests=10):
    print(f'\n{"="*60}')
    print(f'  Predicate-only Kronwald rediscovery test  (r={r})')
    print(f'{"="*60}')

    target_var = float(squeezed_vacuum(r)[0, 0])   # sigma_xx of the pure target
    print(f'\n  target: mechanical x-quadrature variance = {target_var:.6f}')
    print(f'  (no sigma_target matrix constructed — target_predicate only)')

    term = target_collective_quadrature(
        target_var, u=[1.0, 0.0], modes=[1], name='mech_x_variance')

    optimizer = CovarianceOptimizer(
        target_predicate  = [term],
        make_initial_test = True,
        **_kronwald_optimizer_kwargs(num_tests),
    )

    all_pass = True
    all_pass &= check('optimizer.sigma_target is None (genuinely predicate-only)',
                      optimizer.sigma_target is None)

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
        s_xx = float(info['sigma_achieved'][0, 0])
        print(f'\n  Loss: {loss:.2e}  (threshold 2e-5)')
        print(f'  achieved sigma_xx = {s_xx:.6f}  (target {target_var:.6f})')
        all_pass &= check(f'Stage 3 loss < 2e-5  (loss={loss:.2e})', loss < 2e-5)
        all_pass &= check(f'achieved x-variance matches target within 1%',
                          abs(s_xx / target_var - 1.) < 0.01)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — target_quadratic_form decomposition == full sigma_target match
# ─────────────────────────────────────────────────────────────────────────

def run_quadratic_form_equivalence_test(r=1.0, num_tests=8):
    print(f'\n{"="*60}')
    print(f'  target_quadratic_form == sigma_target equivalence test  (r={r})')
    print(f'{"="*60}')

    sigma_tgt = squeezed_vacuum(r)   # diagonal: [[sigma_xx, 0], [0, sigma_pp]]

    terms = [
        target_quadratic_form(sigma_tgt[0, 0], Q=[[1., 0.], [0., 0.]],
                               modes=[1], name='sigma_xx'),
        target_quadratic_form(sigma_tgt[1, 1], Q=[[0., 0.], [0., 1.]],
                               modes=[1], name='sigma_pp'),
        # xp correlation is exactly zero for squeezed_vacuum — exercises the
        # q=0 (non-relative-error) branch of _predicate_residual.
        target_quadratic_form(0.0, Q=[[0., 1.], [1., 0.]],
                               modes=[1], name='sigma_xp', weight=0.5),
    ]

    # Looser threshold than the sigma_target tests — see note above: relative
    # per-entry error is a stricter combined requirement at finite lambda.
    optimizer = CovarianceOptimizer(
        target_predicate  = terms,
        make_initial_test = True,
        **_kronwald_optimizer_kwargs(num_tests, max_violation_success=0.01),
    )

    all_pass = True
    all_pass &= check('optimizer.sigma_target is None (predicate-only decomposition)',
                      optimizer.sigma_target is None)

    valid_trius = optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        print(f'    triu_array = {list(t)}')

    all_pass &= check('exactly 1 valid topology found',
                      len(optimizer.valid_combinations) == 1)
    if optimizer.valid_combinations:
        all_pass &= check('triu_array == [0, 4, 0]  (same topology as direct sigma_target path)',
                          list(optimizer.valid_combinations[0]) == [0, 4, 0])

    if optimizer.best_info_list:
        info = optimizer.best_info_list[0]
        s = info['sigma_achieved']
        print(f'\n  achieved sigma_mech =\n    [[{s[0,0]:.6f}, {s[0,1]:.6f}],')
        print(f'     [{s[1,0]:.6f}, {s[1,1]:.6f}]]')
        print(f'  sigma_target        =\n    [[{sigma_tgt[0,0]:.6f}, 0],')
        print(f'     [0, {sigma_tgt[1,1]:.6f}]]')
        all_pass &= check('achieved sigma_xx within 10% of target',
                          abs(s[0, 0] / sigma_tgt[0, 0] - 1.) < 0.10)
        all_pass &= check('achieved sigma_pp within 10% of target',
                          abs(s[1, 1] / sigma_tgt[1, 1] - 1.) < 0.10)
        all_pass &= check('achieved sigma_xp near zero (matches q=0 term)',
                          abs(s[0, 1]) < 1e-3)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — sigma_target AND target_predicate combined
# ─────────────────────────────────────────────────────────────────────────

def run_mixed_target_test(r=1.0, num_tests=8):
    print(f'\n{"="*60}')
    print(f'  Mixed sigma_target + target_predicate test  (r={r})')
    print(f'{"="*60}')

    sigma_tgt = squeezed_vacuum(r)
    # A purity term on top of the full-matrix match — checks both residuals
    # are actually summed into the loss, not one silently ignored. Target is
    # the ideal q=1.0 (exactly pure), but at finite lambda=1000 Kronwald only
    # reaches purity ~0.97 (residual cavity entanglement) — same finite-C gap
    # documented in test_kronwald.py's own purity check (mu_achieved > 0.9,
    # not ~1). So this term alone contributes ~1e-4 to the loss; threshold
    # loosened accordingly (still 100x tighter than the term's own scale).
    purity_term = target_purity(1.0, modes=[1], weight=0.1)

    optimizer = CovarianceOptimizer(
        sigma_target      = sigma_tgt,
        target_predicate  = [purity_term],
        make_initial_test = True,
        **_kronwald_optimizer_kwargs(num_tests, max_violation_success=5e-4),
    )

    all_pass = True
    all_pass &= check('optimizer.sigma_target is not None (matrix term present)',
                      optimizer.sigma_target is not None)
    all_pass &= check('optimizer.target_predicate has 1 term (predicate term also present)',
                      len(optimizer.target_predicate) == 1)

    valid_trius = optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    all_pass &= check('exactly 1 valid topology found, triu == [0,4,0]',
                      len(optimizer.valid_combinations) == 1 and
                      list(optimizer.valid_combinations[0]) == [0, 4, 0])

    if optimizer.best_info_list:
        loss = optimizer.best_info_list[0]['final_cost']
        print(f'  loss = {loss:.2e}  (threshold 5e-4)')
        all_pass &= check(f'Stage 3 loss < 5e-4  (loss={loss:.2e})', loss < 5e-4)

    print(f'\n{"─"*60}')
    print('  ALL CHECKS PASSED' if all_pass else '  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


if __name__ == '__main__':
    results = {
        'predicate_only':      run_predicate_only_test(r=1.0, num_tests=10),
        'quadratic_form_equiv': run_quadratic_form_equivalence_test(r=1.0, num_tests=8),
        'mixed_target':        run_mixed_target_test(r=1.0, num_tests=8),
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
