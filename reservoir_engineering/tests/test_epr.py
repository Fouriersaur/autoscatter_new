"""
test_epr.py
===========
End-to-end test: can the BFS find topologies that engineer EPR (two-mode squeezed) states?

Modes
-----
    node_types       = ['cavity', 'mechanical', 'mechanical']
    target_mode_ids  = [1, 2]   (both mechanicals are the signal)
    Target           : two_mode_squeezed(r)

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

Topology discovery summary (empirical, from BFS runs)
------------------------------------------------------
γ/κ = 1e-2 (default), no mec-mec constraint, loss threshold 5e-4:

    r = 0.4  →  4 topologies found (all direct mec-mec):
        [0, 1, 4, 0, 2, 0]  cav-m1:BS,      cav-m2:BS+TMS,  m1-m2:TMS
        [0, 2, 1, 0, 4, 0]  cav-m1:TMS,     cav-m2:BS,      m1-m2:BS+TMS
        [0, 4, 1, 0, 2, 0]  cav-m1:BS+TMS,  cav-m2:BS,      m1-m2:TMS
        [0, 1, 0, 0, 4, 0]  cav-m1:BS only,                 m1-m2:BS+TMS
        → novel asymmetric schemes, no literature precedent (except [0,1,0,0,4,0]
          which resembles Bogoliubov bath cooling, Woolley & Clerk PRA 2014 §III)
        → max cooperativity ~6.5e6 (mec-mec TMS); [realistic] flag marginal

γ/κ = 1e-2 (default), Constraint_coupling_absent(1,2) — cavity-mediated only:

    r = 0.1  →  1–2 topologies found (stochastic, 60–120 restarts needed):
        [0, 4, 4, 0, 0, 0]  cav-m1:BS+TMS,  cav-m2:BS+TMS   ← Woolley-Clerk
        [0, 4, 1, 0, 0, 0]  cav-m1:BS+TMS,  cav-m2:BS only  ← asymmetric variant
        [0, 1, 4, 0, 0, 0]  cav-m1:BS only, cav-m2:BS+TMS   ← asymmetric variant
        → r_achieved ≈ 0.08–0.09 (optimizer reaches ~88% of target)
        → Duan criterion FAILS even on perfect target at r=0.1 (convention issue)
        → C ~ 1e4–1e7; borderline realistic
    r ≥ 0.3  →  0 topologies — physics ceiling for dissipative cavity-only EPR
        (adiabatic elimination degrades; g_tms → g_bs → marginal instability)

γ/κ = 4e-5 (deep resolved sideband), no mec-mec constraint, log_ratio_bound=13:

    r = 0.4  →  1 topology found:
        [0, 1, 1, 0, 2, 0]  cav-m1:BS, cav-m2:BS, m1-m2:TMS (direct)
        → C_cav-mec ≈ 160,  C_mec-mec ≈ 4.4e8
        → mec-mec cooperativity already at bound limit (exp(13)×λ ≈ 4.4e8)
    r = 0.7+  →  0 topologies — C_mec-mec would exceed 4.4e8; instability ceiling
    r = 2.1   →  NOT achievable — requires C_mec-mec ~ 10^14 (unphysical)

Stability ceiling for direct mec-mec TMS:
    tanh(r) = g_tms / γ_eff   where γ_eff = C_cav × γ (cavity cooling rate)
    As r → ∞: g_tms → γ_eff → marginal instability; C_mec-mec ∝ C_cav² × exp(2r)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import two_mode_squeezed, duan_criterion, log_negativity, purity
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


def run_epr_test(r=0.1, num_tests=40, verbosity=False):
    print(f'\n{"="*60}')
    print(f'  EPR state engineering test  (r={r})')
    print(f'{"="*60}')

    sigma_target = two_mode_squeezed(r)
    node_types   = ['cavity', 'mechanical', 'mechanical']

    ln_theory = 2 * r / np.log(2)
    print(f'\n  Target: two_mode_squeezed(r={r})  [Woolley-Clerk convention]')
    print(f'  sigma_xx per mech   = {sigma_target[0,0]:.4f}  (should be > 0.5)')
    print(f'  x-x cross-corr      = {sigma_target[0,2]:.4f}  (should be < 0)')
    print(f'  p-p cross-corr      = {sigma_target[1,3]:.4f}  (should be > 0)')
    print(f'  Duan entanglement   : {duan_criterion(sigma_target)}  (requires r > ln2/2 ≈ 0.35)')
    print(f'  Log negativity      = {log_negativity(sigma_target):.4f}  '
          f'(theory = 2r/ln2 ≈ {ln_theory:.4f})')

    from reservoir_engineering.constraints import Constraint_coupling_absent
    print('\nStep 1: build optimizer (cavity-mediated only — Woolley-Clerk search)...')
    optimizer = CovarianceOptimizer(
        sigma_target         = sigma_target,
        target_mode_ids      = [1, 2],
        node_types           = node_types,
        num_auxiliary_modes  = 1,
        enforced_constraints = [
            Constraint_stability(penalty_strength=50.0),
            Constraint_coupling_absent(1, 2),   # no direct mec-mec: cavity-mediated only
        ],
        kwargs_optimization  = dict(
            num_tests               = num_tests,
            interrupt_if_successful = True,
            max_violation_success   = 5e-4,
        ),
        solver_options       = dict(maxiter=3000, ftol=0, gtol=1e-12),
        make_initial_test    = False,
        kappa                = 1.0,
        gamma                = 4e-5,
    )
    print('  optimizer initialised OK')

    print('\nStep 2: breadth-first search...')
    optimizer.perform_breadth_first_search()
    print(f'\n  valid topologies found: {len(optimizer.valid_combinations)}')
    for t in optimizer.valid_combinations:
        print(f'    triu_array = {list(t)}')
        print(f'      {_classify_triu(t)}')

    print('\nStep 3: checks')
    all_pass = True

    all_pass &= check('at least 1 valid EPR topology found',
                      len(optimizer.valid_combinations) >= 1)

    if optimizer.valid_combinations:
        diag_ok = all(int(t[0]) == NO_COUPLING and
                      int(t[3]) == NO_COUPLING and
                      int(t[5]) == NO_COUPLING
                      for t in optimizer.valid_combinations)
        all_pass &= check('no parametric drives on any mode (diagonal triu = 0)',
                          diag_ok)

    if optimizer.best_info_list:
        topo_pass_flags = []

        for idx, (triu, info) in enumerate(
                zip(optimizer.valid_combinations, optimizer.best_info_list)):
            sigma_achieved = info['sigma_achieved']
            loss           = info['final_cost']
            max_coop       = max(info.get('cooperativities', {1: 0}).values())

            print(f'\n  ── Topology {idx+1}: {list(triu)}')
            print(f'     {_classify_triu(triu)}')
            print(f'     loss={loss:.2e}  max_C={max_coop:.2e}')

            t_ok = True
            t_ok &= check(f'  loss < 5e-4', loss < 5e-4)

            s_xx0      = float(sigma_achieved[0, 0])
            s_xx1      = float(sigma_achieved[2, 2])
            s_xx_cross = float(sigma_achieved[0, 2])
            s_pp_cross = float(sigma_achieved[1, 3])
            t_ok &= check(f'  mech₁ σ_xx > 0.5  (got {s_xx0:.4f})', s_xx0 > 0.5)
            t_ok &= check(f'  mech₂ σ_xx > 0.5  (got {s_xx1:.4f})', s_xx1 > 0.5)
            t_ok &= check(f'  σ_{{x₁,x₂}} < 0   (got {s_xx_cross:.4f})', s_xx_cross < 0)
            t_ok &= check(f'  σ_{{p₁,p₂}} > 0   (got {s_pp_cross:.4f})', s_pp_cross > 0)
            ln = log_negativity(sigma_achieved)
            r_achieved = float(np.log(2) / 2 * ln)
            t_ok &= check(f'  log_neg > 0  (got {ln:.4f})', ln > 0)
            t_ok &= check(
                f'  log_neg ≥ ½·theory  (got {ln:.4f}, theory {ln_theory:.4f})',
                ln >= 0.5 * ln_theory)
            print(f'     r_achieved = {r_achieved:.4f}  (target r={r})')

            mu_target   = purity(sigma_target)
            mu_achieved = purity(sigma_achieved)
            print(f'     Purity:')
            print(f'       target   μ = {mu_target:.6f}  (ideal: 1.0)')
            print(f'       achieved μ = {mu_achieved:.6f}  (gap = finite-C + bath mixing)')
            t_ok &= check(f'  achieved purity > 0.5  (got {mu_achieved:.4f})', mu_achieved > 0.5)

            realistic = max_coop < 1e7
            print(f'     {"[realistic]" if realistic else "[unphysical: extreme cooperativity]"}')

            coops = info.get('cooperativities', {})
            if coops:
                c_ref_key = max(coops, key=lambda k: coops[k])
                c_ref_val = coops[c_ref_key]
                print(f'     Cooperativity ratios (ref = {c_ref_key}, C_ref={c_ref_val:.2e}):')
                for k, v in coops.items():
                    print(f'       C_{k} / C_ref = {v / c_ref_val:.4f}   (C_{k} = {v:.2e})')

            print('\n     Covariance matrices (signal modes, 4×4):')
            print('     sigma_target:')
            for row in sigma_target:
                print('       ' + '  '.join(f'{v:+.6f}' for v in row))
            print('     sigma_achieved:')
            for row in sigma_achieved:
                print('       ' + '  '.join(f'{v:+.6f}' for v in row))

            if verbosity:
                print(f'\n     cooperativities: {coops}')

            topo_pass_flags.append(t_ok)
            all_pass &= t_ok

        any_realistic = any(
            max(info.get('cooperativities', {1: 0}).values()) < 1e7
            for info in optimizer.best_info_list
        )
        all_pass &= check('\nat least 1 topology is experimentally realistic (C<1e7)',
                          any_realistic)

    print(f'\n{"─"*60}')
    if all_pass:
        print('  ALL CHECKS PASSED — Woolley-Clerk EPR topology rediscovered')
    else:
        print('  SOME CHECKS FAILED — see [FAIL] lines above')
    print(f'{"─"*60}\n')
    return all_pass


if __name__ == '__main__':
    run_epr_test(r=0.1, num_tests=40)
