"""
benchmarks.py
=============
Known optomechanical and optical schemes for validation and comparison.
No direct AutoScatter analogue (AutoScatter validates via notebooks; here
multiple known literature schemes need checking before novel territory).

Purpose: before using CovarianceOptimizer to discover NEW schemes, validate
the pipeline against KNOWN ones. A scheme "passes" if the optimizer,
starting from scratch, can (1) discover the correct topology, (2) find
coupling strengths reaching the target covariance, (3) recover the known
coupling ratios (e.g. nu/g = tanh(r) for Kronwald).

Scheme progression toward cluster states:
  0: Kronwald (2 modes, squeezed mechanical)
  1: Wang-Clerk (entangled 2-mode state via shared reservoir)
  2: 3-mode entangled state (mechanical + optical EPR)
  3: linear cluster state (chain)
  4: 2D cluster state (universal MBQC resource)

Benchmark dict keys: name, reference, node_types, triu_array, sigma_target,
target_mode_ids, known_couplings, known_cooperativities, description, notes.
"""

import numpy as np


# Kronwald & Marquardt, PRL 111, 133601 (2013): a single optomechanical
# cavity driven by red+blue sideband tones creates beamsplitter (g) and
# two-mode-squeezing (nu) couplings; adiabatic elimination (kappa >> gamma)
# leaves an effective squeezed dissipator on the mechanical mode, with
# nu/g = tanh(r). Two edge TYPES on the same (0,1) pair means two separate
# edge dicts (BS + TMS) — the triu encoding's value 4
# (BEAMSPLITTER_AND_TWO_MODE_SQUEEZING) represents exactly this.
KRONWALD_MARQUARDT_2013 = {
    'name': 'Kronwald-Marquardt dissipative squeezing',
    'reference': 'A. Kronwald & F. Marquardt, PRL 111, 133601 (2013)',
    'description':
        'Two-tone optomechanical driving (red + blue sideband) creates beamsplitter '
        '(g) and two-mode squeezing (ν) interactions between the cavity and mechanical '
        'mode. In steady state, the mechanical mode is squeezed with r = atanh(ν/g). '
        'Requires κ ≫ γ (bad-cavity / resolved sideband limit).',
    'node_types': ['cavity', 'mechanical'],
    'num_signal_modes': 1,
    'target_mode_ids': [1],  # mechanical mode is signal
    'triu_description':
        'Two simultaneous edge types on the (0,1) pair: BS with strength g, '
        'and TMS with strength ν. In edge-dict form: 2 edge dicts for the same pair.',
    'known_coupling_ratio': 'nu/g = tanh(r_target)',
    'known_cooperativities':
        'C_g = 4g²/(κγ), C_ν = 4ν²/(κγ), squeezing r = atanh(sqrt(C_ν/C_g))',
    'expected_scaling_exponents':
        {'BS(0,1)': 1.0, 'TMS(0,1)': 1.0},
    'expected_coupling_ratios':
        'C̃_ν / C̃_g = tanh(r_target)²  [e.g. for r=1.0: C̃_ν/C̃_g ≈ 0.580]',
    'scaling_interpretation':
        'All β_i = 1: take C_g >> 1 and C_ν >> 1 simultaneously. '
        'The squeezing is determined entirely by the ratio C̃_ν/C̃_g = tanh(r)².',
    'validation_tolerance': 0.01,
    'notes':
        'The canonical ground-truth benchmark. If this fails, all others will too. '
        'Already numerically verified in kronwald_optimizer.py. '
        'The automated discoverer must find BS+TMS topology without being told. '
        'Stage 2 MUST return β_g = β_ν = 1 (same-scaling test passes). '
        'Stage 3 MUST return C̃_ν/C̃_g = tanh(r_target)² to within 1% tolerance.',
}


# Wang & Clerk, PRL 108, 153603 / PRL 109, 073601 (2012): two mechanical
# modes each coupled to one common cavity via red+blue sidebands; adiabatic
# elimination of the cavity leaves an effective entangling dissipator
# between the mechanics (no direct mech-mech edge needed). Symmetric case
# g1=g2=g, nu1=nu2=nu gives entanglement r = atanh(nu/g), same formula as
# Kronwald extended to 2 signal modes.
WANG_CLERK_ENTANGLEMENT = {
    'name': 'Wang-Clerk two-mode squeezing via common cavity reservoir',
    'reference':
        'Y.-D. Wang & A. A. Clerk, PRL 108, 153603 (2012) and PRL 109, 073601 (2012)',
    'description':
        'Two mechanical modes are each coupled to a single cavity via red and blue '
        'sideband drives. The cavity, when adiabatically eliminated, creates an '
        'effective entangling dissipator that drives the two mechanics into a '
        'two-mode squeezed (EPR) steady state.',
    'node_types': ['cavity', 'mechanical', 'mechanical'],
    'num_signal_modes': 2,
    'target_mode_ids': [1, 2],  # both mechanical modes are signal
    'triu_description':
        'BS and TMS edges from cavity (mode 0) to each mechanical mode (1 and 2). '
        'No direct mechanical-mechanical edge. Symmetric: g₁=g₂, ν₁=ν₂.',
    'known_coupling_ratio': 'nu/g = tanh(r_target) for each cavity-mech pair',
    'known_cooperativities':
        'C_g = 4g²/(κγ) per mode, entanglement via ν/g = tanh(r)',
    'expected_scaling_exponents':
        {'BS(0,1)': 1.0, 'BS(0,2)': 1.0, 'TMS(0,1)': 1.0, 'TMS(0,2)': 1.0},
    'expected_coupling_ratios':
        'C̃_ν/C̃_g = tanh(r_target)² for each cavity-mech pair (symmetric)',
    'scaling_interpretation':
        'All β_i = 1: single scale knob. '
        'The entanglement is determined by the per-edge ratio C̃_ν/C̃_g.',
    'validation_tolerance': 0.01,
    'notes':
        'Generalisation of Kronwald to two signal modes. '
        'Optimizer must discover cavity-mediated topology without direct mech-mech coupling. '
        'Stage 2 MUST return β=1 for all 4 edges. '
        'Stage 3 MUST return C̃_ν/C̃_g = tanh(r_target)² per pair, '
        'and symmetric solution C̃_g1=C̃_g2 (up to numerical noise). '
        'Compare: AutoScatter discovers auxiliary modes automatically; here the same.',
}


# Woolley & Clerk, PRA 89, 063805 (2014): an alternative single-cavity route
# to the same two-mode-squeezed target as Wang-Clerk, via different drive
# tones. Tests whether the BFS finds BOTH topologies as independent minimal
# solutions for the same target (AutoScatter shows multi-solution discovery
# in notebooks 3-4).
WOOLLEY_CLERK_TMS = {
    'name': 'Woolley-Clerk dissipative two-mode squeezing',
    'reference': 'M. J. Woolley & A. A. Clerk, PRA 89, 063805 (2014)',
    'description':
        'Single-cavity scheme for dissipative two-mode squeezing of two mechanical '
        'modes. Uses a single engineered reservoir to entangle the mechanics. '
        'May require a different topology than Wang-Clerk for the same target.',
    'node_types': ['cavity', 'mechanical', 'mechanical'],
    'num_signal_modes': 2,
    'target_mode_ids': [1, 2],
    'triu_description': 'To be determined by the optimizer — may differ from Wang-Clerk.',
    'expected_scaling_exponents': 'To be discovered — key question: same as Wang-Clerk?',
    'expected_coupling_ratios': 'To be discovered by Stage 3.',
    'scaling_interpretation':
        'If Stage 2 returns β_i = 1 for all edges AND Stage 3 returns same ratios as '
        'Wang-Clerk: the two schemes are equivalent at the level of this algorithm. '
        'If β_i differ: Woolley-Clerk has a genuinely different scaling structure.',
    'notes':
        'If both Woolley-Clerk and Wang-Clerk topologies are valid for the same target, '
        'the BFS should discover both as independent minimal topologies. '
        'This tests whether perform_breadth_first_search finds ALL solutions, not just one. '
        'Stage 2 will reveal whether the two schemes require different scaling exponents '
        '(a structural difference) or only different coupling ratios (cosmetic difference). '
        'AutoScatter demonstrates multi-solution discovery in notebooks 3-4.',
}


# Diehl et al., Nature Physics 4, 878 (2008); Menicucci et al., PRL 97,
# 110501 (2006): three modes in a linear cluster (graph) state, the
# smallest resource useful for single-qubit MBQC. Expects nearest-neighbour
# edges (0,1),(1,2) only (a CHAIN, not fully connected) — tests that the BFS
# prunes unnecessary long-range couplings, plus however many auxiliary
# cavities find_minimum_number_auxiliary_modes decides are needed.
THREE_MODE_CLUSTER = {
    'name': 'Three-mode linear cluster state',
    'reference':
        'S. Diehl et al., Nature Physics 4, 878 (2008); '
        'Menicucci et al., PRL 97, 110501 (2006)',
    'description':
        'Three modes in a linear cluster (graph) state configuration. '
        'This is the smallest cluster state useful for single-qubit MBQC operations. '
        'Requires engineering dissipation that establishes cluster-state correlations '
        'in steady state.',
    'node_types': ['mechanical', 'mechanical', 'mechanical'],  # 3 signal modes
    'num_signal_modes': 3,
    'target_mode_ids': [0, 1, 2],
    'triu_description':
        'Nearest-neighbour edges (0,1) and (1,2) expected. '
        'Likely needs auxiliary cavity modes to engineer the dissipation. '
        'Number of auxiliary cavities: unknown — let find_minimum_number_auxiliary_modes decide.',
    'expected_scaling_exponents':
        'Unknown — to be discovered. Key question: do all edges scale with β=1, '
        'or does the cluster state require a non-trivial scaling structure (β≠1)?',
    'expected_coupling_ratios':
        'Unknown — to be discovered. '
        'If all β=1: the ratios encode the cluster-state geometry directly.',
    'scaling_interpretation':
        'If Stage 2 finds β_i ≠ 1 for some edge: this reveals a structural property '
        'of the cluster state engineering scheme. This would be a new theoretical result. '
        'If Stage 2 finds β_i = 1 for all edges: the scheme is a straightforward '
        'strong-coupling limit, analogous to Kronwald but for graph states.',
    'notes':
        'This is the key stepping stone toward full cluster states. '
        'The optimizer must decide: (a) how many auxiliary cavities are needed, '
        '(b) which topology connects them to the signal modes, '
        '(c) what scaling exponents {β_i} are required (Stage 2), '
        '(d) what coupling ratios {C̃_i} achieve the cluster correlations (Stage 3). '
        'If the optimizer discovers a non-chain topology with non-trivial β_i, '
        'that is a NEW RESULT in dissipative quantum state engineering.',
}


# Programmatic test (not a literature reference): can two_mode_squeezed(r)
# be reached between two mechanical modes (not cavity-mechanical) via one
# or more auxiliary cavities? Expected: yes, with 1 auxiliary cavity — the
# Wang-Clerk topology — linking that benchmark to the cluster-state goal.
TWO_MODE_SQUEEZED_MECHANICS = {
    'name': 'Two-mode squeezed vacuum (mechanical-mechanical)',
    'reference': 'Building block for dissipative cluster state generation',
    'description':
        'Two mechanical modes entangled via a shared cavity reservoir. '
        'The target is a two-mode squeezed vacuum state for the two mechanics. '
        'This is the building block from which larger cluster states can be assembled.',
    'node_types': ['cavity', 'mechanical', 'mechanical'],
    'num_signal_modes': 2,
    'target_mode_ids': [1, 2],
    'triu_description': 'See Wang-Clerk benchmark (same topology expected).',
    'expected_scaling_exponents':
        {'BS(0,1)': 1.0, 'BS(0,2)': 1.0, 'TMS(0,1)': 1.0, 'TMS(0,2)': 1.0},
    'expected_coupling_ratios':
        'C̃_ν/C̃_g = tanh(r_target)² per cavity-mech pair (same as Wang-Clerk)',
    'scaling_interpretation':
        'Identical to Wang-Clerk: all β_i = 1, single scale knob, '
        'physics determined by per-edge ratio C̃_ν/C̃_g.',
    'notes':
        'Run this benchmark to confirm that the Wang-Clerk topology is the '
        'MINIMUM topology for two-mode squeezing of two mechanical modes. '
        'Stage 2 should confirm β_i = 1 for all edges. '
        'Stage 3 should confirm C̃_ν/C̃_g = tanh(r_target)² per pair. '
        'The Stage 3 output directly gives the cooperativity requirements '
        '(C_g, C_ν) needed to entangle the two mechanical modes in experiment.',
}


ALL_BENCHMARKS = {
    'kronwald':          KRONWALD_MARQUARDT_2013,
    'wang_clerk':        WANG_CLERK_ENTANGLEMENT,
    'woolley_clerk':     WOOLLEY_CLERK_TMS,
    'three_mode_cluster': THREE_MODE_CLUSTER,
    'two_mode_mech':     TWO_MODE_SQUEEZED_MECHANICS,
}


# Look up a benchmark dict by name (KeyError if not found). Not yet implemented.
def get_benchmark(name: str) -> dict:
    pass


# Build a CovarianceOptimizer configured for one benchmark: sigma_target
# from bench['target_mode_ids'] via the matching targets.py function
# (squeezed_vacuum for kronwald, two_mode_squeezed for wang_clerk/
# woolley_clerk/two_mode_mech, cluster_state for three_mode_cluster), plus
# bench['node_types']/target_mode_ids and **kwargs. Not yet implemented.
def make_benchmark_optimizer(name: str, r_or_param: float = 1.0, **kwargs):
    pass


# Run the full BFS for one benchmark and check the discovered topology
# against bench['triu_description']. Returns {'benchmark','valid_topologies',
# 'num_valid','passed','cooperativities'}. Not yet implemented.
def run_benchmark(name: str, r_or_param: float = 1.0, **kwargs) -> dict:
    pass


# Run every benchmark in ALL_BENCHMARKS and print a pass/fail summary.
# "Partial pass" = target reached but via a different (possibly new)
# topology than expected. Not yet implemented.
def run_all_benchmarks(r_or_param: float = 1.0, **kwargs) -> dict:
    pass


# Print a formatted table: Name | Expected | Found | Loss | Pass/Fail. Not
# yet implemented.
def print_benchmark_summary(results: dict):
    pass
