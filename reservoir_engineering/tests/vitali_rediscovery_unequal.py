"""
vitali_rediscovery_unequal.py
==============================
Same FULL-algorithm run as vitali_rediscovery.py (discrete topology search
via perform_breadth_first_search, not a hand-picked topology), but the
target is an UNEQUALLY-squeezed EPR state (non-flat Bloch-Messiah spectrum)
instead of the standard symmetric TMSV.

Construction: start from the package's standard two_mode_squeezed(r)
(pure two-mode-squeezed vacuum, Woolley-Clerk convention: x0+x1, p0-p1
squeezed), then hit each mode with its OWN extra local squeezer r1, r2
(r1 != r2). Local single-mode squeezing is symplectic, so the result is
still a pure, physical Gaussian state — just one whose two marginal modes
now have different variances/purity structure (non-flat spectrum), unlike
the r1=r2=0 case which reproduces the flat, symmetric EPR state exactly.

sigma0 = 0.5*[[c*I2, -s*sz], [-s*sz, c*I2]]     c=cosh(2r), s=sinh(2r)
S      = diag(e^-r1, e^r1, e^-r2, e^r2)
sigma  = S @ sigma0 @ S.T

Because the target is no longer symmetric under swapping modes 1 and 2, no
Constraint_coupling_symmetric is imposed anywhere (unlike the old
fixed-topology version of this file) — the discrete search is free to find
asymmetric drain-mode1/drain-mode2 edges, or even a different topology
altogether, whatever the asymmetric target actually needs.

Usage:
    python3 vitali_rediscovery_unequal.py                # r=0.2, r1=0.15, r2=-0.05
    python3 vitali_rediscovery_unequal.py 0.2 0.15 -0.05  # r r1 r2 on the command line
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import purity, log_negativity
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.covariance_physics import get_mode_covariance
from reservoir_engineering.constraints import Constraint_stability

NODE_TYPES = ['cavity', 'mechanical', 'mechanical']
TARGET_MODE_IDS = [1, 2]

# Vitali's known (symmetric-target) answer, kept only as a reference point
# for the report — there's no reason to expect this exact topology to be
# minimal for an ASYMMETRIC target.
VITALI_TRIU = np.array([0, 1, 1, 0, 1, 0])


# Unequally-squeezed EPR target: two_mode_squeezed(r) plus per-mode local
# squeezers r1 (mode 1), r2 (mode 2). r1==r2==0 reproduces
# targets.two_mode_squeezed(r) exactly; r1 != r2 breaks the flat spectrum.
def unequal_two_mode_squeezed(r: float, r1: float, r2: float) -> np.ndarray:
    c = np.cosh(2 * r)
    s = np.sinh(2 * r)
    sz = np.diag([1., -1.])
    I2 = np.eye(2)
    sigma0 = 0.5 * np.block([[c * I2, -s * sz],
                              [-s * sz, c * I2]])
    S = np.diag([np.exp(-r1), np.exp(r1), np.exp(-r2), np.exp(r2)])
    return S @ sigma0 @ S.T


# Run the FULL algorithm on the asymmetric target: same drain-squeezing
# palette as vitali_rediscovery.py, but topology is discovered, not assumed.
def rediscover_unequal_epr(r: float, r1: float, r2: float,
                            num_tests_per_topology: int = 8, seed: int = 0):
    target = unequal_two_mode_squeezed(r, r1, r2)

    np.random.seed(seed)
    optimizer = CovarianceOptimizer(
        sigma_target=target,
        target_mode_ids=TARGET_MODE_IDS,
        node_types=NODE_TYPES,
        num_auxiliary_modes=1,
        gamma=0.0,
        squeezable_aux_ids=[0],
        enforced_constraints=[Constraint_stability(penalty_strength=100.0)],
        kwargs_optimization=dict(
            num_tests=num_tests_per_topology,
            max_violation_success=1e-6,
            interrupt_if_successful=True,
            optimize_detunings=True,
        ),
        solver_options=dict(maxiter=2000, ftol=0, gtol=1e-12),
    )

    optimizer.perform_breadth_first_search()
    return target, optimizer


_EDGE_LABEL = {0: 'absent', 1: 'beamsplitter', 2: 'two-mode-squeezing',
               3: 'parametric', 4: 'beamsplitter+two-mode-squeezing'}


def _describe_triu(triu_array, n_nodes: int) -> str:
    rows, cols = np.triu_indices(n_nodes)
    parts = []
    for (i, j), val in zip(zip(rows, cols), triu_array):
        val = int(val)
        if val == 0:
            continue
        label = _EDGE_LABEL[val]
        parts.append(f'mode{i} self-{label}' if i == j else f'({i},{j}) {label}')
    return ', '.join(parts) if parts else '(empty graph)'


def _print_topology(idx: int, triu_array, info: dict, n_nodes: int):
    complexity = int(np.sum(triu_array))
    print(f'--- Valid topology #{idx+1}  (complexity {complexity}) ---')
    print(f'  Edges: {_describe_triu(triu_array, n_nodes)}')
    print(f'  final loss = {info["final_cost"]:.3e}   success = {info["success"]}')

    print('  Couplings (G~_k = g_k/kappa0):')
    for label, gt in info['G_tilde'].items():
        theta = info['coherent_phases'].get(label.replace('G~_', 'theta_'), None)
        theta_str = f',  theta={theta:+.4f} rad' if theta is not None else '  (real, phase pinned to 0)'
        print(f'    {label}: {gt:.4f}{theta_str}')

    print(f'  Detunings: ' + ', '.join(f'{k}={v:+.4f}' for k, v in info['detunings'].items()))
    print(f'  Auxiliary decay: ' + ', '.join(f'{k}={v:.4f}' for k, v in info['C_tilde_aux'].items()))
    if info['bath_squeezing']:
        for node_id, bs in info['bath_squeezing'].items():
            print(f'  Bath squeeze (mode {node_id}): r={bs["r"]:.6f}  theta={bs["theta"]:+.4f} rad')
    print()


def print_report(r: float, r1: float, r2: float, target: np.ndarray, optimizer: CovarianceOptimizer):
    n_nodes = len(NODE_TYPES)

    print(f'{"="*70}')
    print(f'  Unequally-squeezed EPR — FULL topology search — r={r}, r1={r1}, r2={r2}')
    print(f'{"="*70}\n')

    print('Target covariance matrix (unequal_two_mode_squeezed(r, r1, r2)):')
    print(target, '\n')

    mode1_marginal = get_mode_covariance(target, [0])
    mode2_marginal = get_mode_covariance(target, [1])
    print(f'Target marginal variances: mode1 (x,p)=({mode1_marginal[0,0]:.4f},{mode1_marginal[1,1]:.4f})  '
          f'mode2 (x,p)=({mode2_marginal[0,0]:.4f},{mode2_marginal[1,1]:.4f})')
    print(f'  -> flat spectrum iff these match; here r1={"=" if r1==r2 else "!="}r2, so they '
          f'{"do" if r1==r2 else "do not"}.\n')

    print(f'Graphs enumerated: {optimizer.num_possible_graphs}')
    print(f'Complexity levels tested: {optimizer.tested_complexities}')
    print(f'Graphs tested per level:  {optimizer.num_tested_graphs}')
    print(f'Valid topologies found:   {len(optimizer.valid_combinations)}\n')

    if len(optimizer.valid_combinations) == 0:
        print('No stable/valid topology found by the search.')
        return

    for idx, (triu, info) in enumerate(zip(optimizer.valid_combinations, optimizer.best_info_list)):
        _print_topology(idx, triu, info, n_nodes)
        ln_achieved = log_negativity(info['sigma_achieved'])
        print(f'  Purity (achieved) = {purity(info["sigma_achieved"]):.8f}   '
              f'Log-negativity: achieved={ln_achieved:.4f}  target={log_negativity(target):.4f}')
        matches_vitali = np.array_equal(np.asarray(triu), VITALI_TRIU)
        print(f'  Same edges as Vitali\'s (symmetric-target) triangle? {matches_vitali}')
        print(f'{"─"*70}\n')

    complexities = [int(np.sum(t)) for t in optimizer.valid_combinations]
    minimal_idx = int(np.argmin(complexities))
    print(f'Minimal valid topology: #{minimal_idx+1} at complexity {complexities[minimal_idx]}.')


if __name__ == '__main__':
    r  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    r1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    r2 = float(sys.argv[3]) if len(sys.argv) > 3 else -0.05
    target, optimizer = rediscover_unequal_epr(r, r1, r2)
    print_report(r, r1, r2, target, optimizer)
