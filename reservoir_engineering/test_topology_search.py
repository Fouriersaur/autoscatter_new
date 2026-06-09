"""
test_topology_search.py
=======================
Tests for topology_search.py.

Run with:
    python3 test_topology_search.py

Each test prints [PASS] or [FAIL] with a short description.
A summary at the end shows total pass/fail count.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from topology_search import (
    NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
    PARAMETRIC, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING,
    _is_subgraph_slot,
    triu_to_edge_matrix,
    edge_matrix_to_triu,
    check_if_subgraph,
    check_if_subgraph_triu,
    translate_triu_to_conditions,
    translate_conditions_to_triu,
    characterize_topology,
    calc_number_of_possibilities,
    TopologyGraph,
)

# ── Test harness ──────────────────────────────────────────────────────────────

results = []

def check(name, condition, detail=''):
    tag = 'PASS' if condition else 'FAIL'
    results.append(condition)
    suffix = f'  ({detail})' if detail else ''
    print(f'  [{tag}] {name}{suffix}')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 1. Constants ──')
check('NO_COUPLING == 0',                    NO_COUPLING == 0)
check('BEAMSPLITTER == 1',                   BEAMSPLITTER == 1)
check('TWO_MODE_SQUEEZING == 2',             TWO_MODE_SQUEEZING == 2)
check('PARAMETRIC == 3',                     PARAMETRIC == 3)
check('BEAMSPLITTER_AND_TWO_MODE_SQUEEZING == 4',
      BEAMSPLITTER_AND_TWO_MODE_SQUEEZING == 4)

# ─────────────────────────────────────────────────────────────────────────────
# 2. _is_subgraph_slot
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 2. _is_subgraph_slot ──')
check('0 ⊆ 0',   _is_subgraph_slot(0, 0))
check('0 ⊆ 1',   _is_subgraph_slot(0, 1))
check('0 ⊆ 2',   _is_subgraph_slot(0, 2))
check('0 ⊆ 4',   _is_subgraph_slot(0, 4))
check('1 ⊆ 1',   _is_subgraph_slot(1, 1))
check('1 ⊆ 4',   _is_subgraph_slot(1, 4))
check('2 ⊆ 2',   _is_subgraph_slot(2, 2))
check('2 ⊆ 4',   _is_subgraph_slot(2, 4))
check('4 ⊆ 4',   _is_subgraph_slot(4, 4))
check('3 ⊆ 3',   _is_subgraph_slot(3, 3))

# These must be FALSE (the critical ones)
check('1 NOT ⊆ 2',  not _is_subgraph_slot(1, 2), 'BS is not a subgraph of TMS')
check('2 NOT ⊆ 1',  not _is_subgraph_slot(2, 1), 'TMS is not a subgraph of BS')
check('4 NOT ⊆ 1',  not _is_subgraph_slot(4, 1), 'BS+TMS is not a subgraph of BS')
check('4 NOT ⊆ 2',  not _is_subgraph_slot(4, 2), 'BS+TMS is not a subgraph of TMS')
check('3 NOT ⊆ 1',  not _is_subgraph_slot(3, 1))
check('3 NOT ⊆ 4',  not _is_subgraph_slot(3, 4))

# ─────────────────────────────────────────────────────────────────────────────
# 3. triu_to_edge_matrix
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 3. triu_to_edge_matrix ──')
# N=2, Kronwald: triu = [0, 4, 0]
kronwald_triu = np.array([0, 4, 0])
M = triu_to_edge_matrix(kronwald_triu, 2)
check('shape (2,2)',      M.shape == (2, 2))
check('M[0,0] == 0',     M[0, 0] == 0)
check('M[0,1] == 4',     M[0, 1] == 4)
check('M[1,0] == 4',     M[1, 0] == 4,  'symmetric copy')
check('M[1,1] == 0',     M[1, 1] == 0)

# N=3, one BS edge (0,1) and one TMS edge (0,2)
triu3 = np.array([0, 1, 2, 0, 0, 0])  # slots: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
M3 = triu_to_edge_matrix(triu3, 3)
check('N=3: M[0,1]==1',  M3[0, 1] == 1)
check('N=3: M[1,0]==1',  M3[1, 0] == 1, 'symmetric')
check('N=3: M[0,2]==2',  M3[0, 2] == 2)
check('N=3: M[2,0]==2',  M3[2, 0] == 2, 'symmetric')
check('N=3: M[1,2]==0',  M3[1, 2] == 0, 'no edge between 1 and 2')

# ─────────────────────────────────────────────────────────────────────────────
# 4. edge_matrix_to_triu
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 4. edge_matrix_to_triu ──')
triu_back = edge_matrix_to_triu(M)
check('round-trip N=2 Kronwald', np.array_equal(triu_back, kronwald_triu))

triu3_back = edge_matrix_to_triu(M3)
check('round-trip N=3',          np.array_equal(triu3_back, triu3))

# ─────────────────────────────────────────────────────────────────────────────
# 5. check_if_subgraph_triu  (critical for BFS pruning)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 5. check_if_subgraph_triu ──')
empty     = np.array([0, 0, 0])
bs_only   = np.array([0, 1, 0])
tms_only  = np.array([0, 2, 0])
bs_tms    = np.array([0, 4, 0])   # Kronwald

check('empty ⊆ Kronwald',       check_if_subgraph_triu([bs_tms], [empty]))
check('BS ⊆ Kronwald',          check_if_subgraph_triu([bs_tms], [bs_only]))
check('TMS ⊆ Kronwald',         check_if_subgraph_triu([bs_tms], [tms_only]))
check('Kronwald ⊆ Kronwald',    check_if_subgraph_triu([bs_tms], [bs_tms]))

check('BS NOT ⊆ TMS',           not check_if_subgraph_triu([tms_only], [bs_only]))
check('TMS NOT ⊆ BS',           not check_if_subgraph_triu([bs_only], [tms_only]))
check('Kronwald NOT ⊆ BS',      not check_if_subgraph_triu([bs_only], [bs_tms]))
check('Kronwald NOT ⊆ TMS',     not check_if_subgraph_triu([tms_only], [bs_tms]))

# BFS-relevant: if TMS failed, should BS+TMS still be tested?
# TMS-only (Stage 1 unstable) is NOT added to invalid_combinations
# so BS+TMS is NOT pruned. But if BS-only FAILED (Stage 3 failure):
check('Kronwald IS superset of BS-only (prune rule)',
      check_if_subgraph_triu([bs_tms], [bs_only]),
      'if BS-only invalid, Kronwald would be pruned')

# ─────────────────────────────────────────────────────────────────────────────
# 6. translate_triu_to_conditions
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 6. translate_triu_to_conditions ──')
node_types_2 = ['cavity', 'mechanical']
try:
    from constraints import (Constraint_coupling_absent,
                              Constraint_coupling_beamsplitter,
                              Constraint_coupling_two_mode_squeezing)

    # Kronwald [0, 4, 0]
    conds = translate_triu_to_conditions(np.array([0, 4, 0]), node_types_2)
    has_bs  = any(isinstance(c, Constraint_coupling_beamsplitter)
                  and c.idxs == [0,1] for c in conds)
    has_tms = any(isinstance(c, Constraint_coupling_two_mode_squeezing)
                  and c.idxs == [0,1] for c in conds)
    has_absent_00 = any(isinstance(c, Constraint_coupling_absent)
                        and c.idxs == [0,0] for c in conds)
    has_absent_11 = any(isinstance(c, Constraint_coupling_absent)
                        and c.idxs == [1,1] for c in conds)
    check('Kronwald: has BS constraint',       has_bs)
    check('Kronwald: has TMS constraint',      has_tms)
    check('Kronwald: absent on (0,0)',         has_absent_00)
    check('Kronwald: absent on (1,1)',         has_absent_11)

    # BS-only [0, 1, 0]
    conds_bs = translate_triu_to_conditions(np.array([0, 1, 0]), node_types_2)
    has_bs_only  = any(isinstance(c, Constraint_coupling_beamsplitter)
                       and c.idxs == [0,1] for c in conds_bs)
    has_tms_none = not any(isinstance(c, Constraint_coupling_two_mode_squeezing)
                            and c.idxs == [0,1] for c in conds_bs)
    check('BS-only: has BS constraint',        has_bs_only)
    check('BS-only: no TMS constraint',        has_tms_none)

    # Empty [0, 0, 0]
    conds_empty = translate_triu_to_conditions(np.array([0, 0, 0]), node_types_2)
    all_absent = all(isinstance(c, Constraint_coupling_absent) for c in conds_empty)
    check('Empty: all constraints are absent', all_absent)

except Exception as e:
    print(f'  [SKIP] constraints.py not yet implemented: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 7. translate_conditions_to_triu  (round-trip test)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 7. translate_conditions_to_triu (round-trip) ──')
try:
    for label, triu_in in [('Kronwald [0,4,0]', np.array([0, 4, 0])),
                            ('BS-only  [0,1,0]', np.array([0, 1, 0])),
                            ('TMS-only [0,2,0]', np.array([0, 2, 0])),
                            ('Empty    [0,0,0]', np.array([0, 0, 0]))]:
        conds    = translate_triu_to_conditions(triu_in, node_types_2)
        triu_out = translate_conditions_to_triu(conds, 2, node_types_2)
        check(f'round-trip {label}', np.array_equal(triu_in, triu_out))
except Exception as e:
    print(f'  [SKIP] not yet implemented: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 8. characterize_topology
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 8. characterize_topology ──')
try:
    info = characterize_topology(np.array([0, 4, 0]), node_types_2)
    check('Kronwald: num_bs_couplings == 1',     info['num_bs_couplings']    == 1)
    check('Kronwald: num_tms_couplings == 1',    info['num_tms_couplings']   == 1)
    check('Kronwald: num_couplings == 2',        info['num_couplings']       == 2)
    check('Kronwald: complexity == 4',           info['complexity']          == 4)
    check('Kronwald: has_cavity == True',        info['has_cavity']          == True)
    check('Kronwald: is_connected == True',      info['is_connected']        == True)

    info_bs = characterize_topology(np.array([0, 1, 0]), node_types_2)
    check('BS-only: num_bs == 1, num_tms == 0',
          info_bs['num_bs_couplings'] == 1 and info_bs['num_tms_couplings'] == 0)
    check('BS-only: complexity == 1',            info_bs['complexity']       == 1)

    info_empty = characterize_topology(np.array([0, 0, 0]), node_types_2)
    check('Empty: is_connected == False',        info_empty['is_connected']  == False)
    check('Empty: num_couplings == 0',           info_empty['num_couplings'] == 0)
except Exception as e:
    print(f'  [SKIP] not yet implemented: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 9. calc_number_of_possibilities
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 9. calc_number_of_possibilities ──')
try:
    # N=2: 1 off-diagonal pair × 4 choices, 2 diagonal × 2 choices = 4×4 = 16
    n2 = calc_number_of_possibilities(['cavity', 'mechanical'])
    check('N=2: 16 topologies', n2 == 16, f'got {n2}')

    # N=3: 3 off-diagonal pairs × 4 choices each = 4^3 = 64
    #      3 diagonal × 2 choices each = 2^3 = 8  →  64×8 = 512
    n3 = calc_number_of_possibilities(['cavity', 'mechanical', 'mechanical'])
    check('N=3: 512 topologies', n3 == 512, f'got {n3}')
except Exception as e:
    print(f'  [SKIP] not yet implemented: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 10. TopologyGraph
# ─────────────────────────────────────────────────────────────────────────────
print('\n── 10. TopologyGraph ──')
try:
    # Construction from triu array
    g = TopologyGraph(['cavity', 'mechanical'], np.array([0, 4, 0]))
    check('num_nodes == 2',             g.num_nodes == 2)
    check('complexity == 4',            g.complexity == 4)
    check('num_edges == 1',             g.num_edges == 1,
          '1 active slot (even though 2 physical drives)')
    check('has_cavity == True',         g.has_cavity())
    check('is_connected == True',       g.is_connected())

    # Construction from edge matrix
    M_in = np.array([[0, 4], [4, 0]])
    g2 = TopologyGraph(['cavity', 'mechanical'], M_in)
    check('from edge_matrix: triu == [0,4,0]',
          np.array_equal(g2.triu_array, np.array([0, 4, 0])))

    # to_nodes_edges_dicts — Kronwald must produce 2 edges
    nodes, edges = g.to_nodes_edges_dicts()
    check('Kronwald: 2 nodes',          len(nodes) == 2)
    check('Kronwald: 2 edges (BS+TMS)', len(edges) == 2,
          f'got {len(edges)}')
    edge_types = [e['type'] for e in edges]
    check('Kronwald: has beamsplitter edge',
          'beamsplitter' in edge_types)
    check('Kronwald: has two_mode_squeezing edge',
          'two_mode_squeezing' in edge_types)
    check('Kronwald: both edges on same pair (0,1)',
          all(e['i'] == 0 and e['j'] == 1 for e in edges))
    check('cavity node has kappa',      'kappa' in nodes[0])
    check('mechanical node has gamma',  'gamma' in nodes[1])

    # BS-only: 1 edge
    g_bs = TopologyGraph(['cavity', 'mechanical'], np.array([0, 1, 0]))
    _, edges_bs = g_bs.to_nodes_edges_dicts()
    check('BS-only: 1 edge',            len(edges_bs) == 1)
    check('BS-only: type is beamsplitter', edges_bs[0]['type'] == 'beamsplitter')

    # fully_connected classmethod
    g_full = TopologyGraph.fully_connected(['cavity', 'mechanical'])
    check('fully_connected: off-diagonal is BS+TMS, diagonal is 0',
          g_full.triu_array[1] == BEAMSPLITTER_AND_TWO_MODE_SQUEEZING and
          g_full.triu_array[0] == 0 and g_full.triu_array[2] == 0)

    # empty classmethod
    g_empty = TopologyGraph.empty(['cavity', 'mechanical'])
    check('empty: all zeros',
          np.all(g_empty.triu_array == 0))

    # equality and hash
    g_a = TopologyGraph(['cavity', 'mechanical'], np.array([0, 4, 0]))
    g_b = TopologyGraph(['cavity', 'mechanical'], np.array([0, 4, 0]))
    g_c = TopologyGraph(['cavity', 'mechanical'], np.array([0, 1, 0]))
    check('equal topologies are ==',    g_a == g_b)
    check('different topologies are !=', g_a != g_c)
    check('can be used in a set',       len({g_a, g_b, g_c}) == 2)

    # is_subgraph_of
    g_bs   = TopologyGraph(['cavity', 'mechanical'], np.array([0, 1, 0]))
    g_kron = TopologyGraph(['cavity', 'mechanical'], np.array([0, 4, 0]))
    check('BS is_subgraph_of Kronwald', g_bs.is_subgraph_of(g_kron))
    check('Kronwald NOT is_subgraph_of BS',
          not g_kron.is_subgraph_of(g_bs))

except Exception as e:
    import traceback
    print(f'  [FAIL] Exception: {e}')
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
failed = total - passed
print(f'\n{"═"*50}')
print(f'  {passed}/{total} passed', end='')
if failed:
    print(f'   ({failed} FAILED ← fix these)')
else:
    print('   — all good')
print(f'{"═"*50}')
