"""
vitali_rediscovery.py
======================
Standalone demo (not a test): given an equally-squeezed EPR target
(targets.two_mode_squeezed(r)), runs CovarianceOptimizer's FULL algorithm —
discrete topology search (perform_breadth_first_search: enumerate every
{drain, mode1, mode2} graph, low complexity first, Stage-1 stability filter
+ Stage-2 continuous optimisation per candidate) — NOT a hand-picked
topology fitted with continuous optimisation only. Prints every valid
topology found, its discovered Hamiltonian (coupling magnitudes/phases,
detunings, drain decay + bath squeezing), and whether the minimal one
matches Zippilli & Vitali's known scheme.

Zippilli & Vitali's single-squeezed-reservoir EPR scheme (PRL 126, 020402
(2021), arXiv:2008.02539): a PASSIVE (beamsplitter-only) triangle Hamiltonian
between two mechanical modes and one auxiliary cavity drain, the drain alone
coupled to a squeezed bath — see tests/test_vitali_epr.py for the same
result wrapped in pass/fail checks and the full investigation writeup (why
detunings had to be free, why real-only couplings and topology choice were
red herrings). Here we no longer ASSUME that triangle — we let the discrete
search find it (or something else), which is the actual test of whether the
algorithm "works": does it rediscover Vitali's topology from scratch?

Note: lives in tests/ alongside test_vitali_epr.py for import convenience,
but is a standalone demo script, not a pytest test file.

Usage:
    python3 vitali_rediscovery.py            # r=0.2 by default
    python3 vitali_rediscovery.py 0.35        # any r on the command line
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from reservoir_engineering.targets import two_mode_squeezed, purity, log_negativity
from reservoir_engineering.covariance_optimizer import CovarianceOptimizer
from reservoir_engineering.constraints import Constraint_stability

# Node convention: 0 = auxiliary squeezed drain (cavity), 1,2 = the two
# mechanical target modes. Topology is NOT fixed here — the discrete search
# decides which of the 3 possible edges (0,1)/(0,2)/(1,2) exist and whether
# they're beamsplitter/two-mode-squeezing/both, plus whether each mode
# carries its own parametric (diagonal) drive.
NODE_TYPES = ['cavity', 'mechanical', 'mechanical']
TARGET_MODE_IDS = [1, 2]

# Vitali's known answer, for comparison against whatever the search finds:
# drain-mode1, drain-mode2, mode1-mode2 all beamsplitter, no parametric.
VITALI_TRIU = np.array([0, 1, 1, 0, 1, 0])


# Run the FULL algorithm: build the optimizer (drain allowed a squeezed
# bath, per-mode detunings free) and hand topology discovery to
# perform_breadth_first_search — no TRIU supplied. Returns (target,
# optimizer) so the caller can inspect optimizer.valid_combinations /
# optimizer.best_info_list (there may be more than one valid topology).
def rediscover_vitali_epr(r: float, num_tests_per_topology: int = 8, seed: int = 0):
    target = two_mode_squeezed(r)

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


# Human-readable edge-type labels for a triu code.
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


# Print one discovered topology: which edges, the achieved Hamiltonian
# (whatever couplings/phases/detunings/bath-squeezing this topology
# actually has — no assumption about which edges exist, unlike the old
# fixed-triangle report).
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


def print_report(r: float, target: np.ndarray, optimizer: CovarianceOptimizer):
    n_nodes = len(NODE_TYPES)

    print(f'{"="*70}')
    print(f'  Vitali single-squeezed-drain EPR — FULL topology search, r = {r}')
    print(f'{"="*70}\n')

    print('Target covariance matrix (two_mode_squeezed(r), modes 1,2):')
    print(target, '\n')

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
        print(f'  Matches Vitali\'s known triangle (BS-only)? {matches_vitali}')
        print(f'{"─"*70}\n')

    complexities = [int(np.sum(t)) for t in optimizer.valid_combinations]
    minimal_idx = int(np.argmin(complexities))
    print(f'Minimal valid topology: #{minimal_idx+1} at complexity {complexities[minimal_idx]} — '
          f'{"MATCHES" if np.array_equal(np.asarray(optimizer.valid_combinations[minimal_idx]), VITALI_TRIU) else "DIFFERS FROM"} '
          f'Vitali\'s scheme.')


if __name__ == '__main__':
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    target, optimizer = rediscover_vitali_epr(r)
    print_report(r, target, optimizer)
