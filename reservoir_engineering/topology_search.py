"""
topology_search.py
==================
Graph encoding, topology utilities, and pump-tone counting.
MIRRORS: autoscatter/architecture.py

Data structures and utilities describing a graph topology — no optimiser
here (that's covariance_optimizer.py). Encodes: compact triu array <->
constraint lists, subgraph checks, pump-tone counting, topology
enumeration/characterisation.

Graph encoding: a topology of N modes is a 1D array of length N(N+1)/2, the
upper triangle (incl. diagonal) of an N x N edge matrix, row-major:
    0 = NO_COUPLING, 1 = BEAMSPLITTER, 2 = TWO_MODE_SQUEEZING (both off-diag)
    3 = PARAMETRIC (diagonal only)
"Complexity" = sum of triu entries; the BFS in covariance_optimizer.py walks
low->high complexity, mirroring AutoScatter's complexity_level.
At least one 'cavity' node is required (a graph of only mechanical modes has
no dissipation, so no Lyapunov solution).

Pump-tone counting: BS(i,j) needs a drive at |w_i - w_j|, TMS(i,j) at
w_i + w_j, parametric(i,i) at 2*w_i. Shared frequencies can share a laser;
find_min_number_pump_tones searches mode-frequency assignments to minimise
the count (AutoScatter's find_min_number_of_pumps, for optomechanics).
"""

import numpy as np
import math
from itertools import product, permutations
from typing import List, Optional

# Edge type constants. OFF-DIAGONAL slots: {0,1,2,4}. DIAGONAL slots: {0,3}.
# BEAMSPLITTER_AND_TWO_MODE_SQUEEZING (4) = both a BS and a TMS drive on the
# same mode pair (the Kronwald topology) — creates TWO edges/coupling
# strengths for that one slot, since a single value in {1,2} can't.
NO_COUPLING                       = 0  # no edge between modes i and j
BEAMSPLITTER                      = 1  # BS only (red-sideband drive)
TWO_MODE_SQUEEZING                = 2  # TMS only (blue-sideband drive)
PARAMETRIC                        = 3  # single-mode squeezing (diagonal only)
BEAMSPLITTER_AND_TWO_MODE_SQUEEZING = 4  # both BS and TMS on the same pair

# Subgraph containment: edge type a is a "subgraph of" b iff b in this set.
# NOT plain integer <=: BS(1) is a subgraph of BS(1)/BS+TMS(4) only, never
# TMS(2), even though 1<=2 — the two types are physically different block
# structures, not degrees of the same thing.
_SUBGRAPH_SUPERSETS = {
    NO_COUPLING                       : {NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING,
                                         BEAMSPLITTER_AND_TWO_MODE_SQUEEZING, PARAMETRIC},
    BEAMSPLITTER                      : {BEAMSPLITTER, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING},
    TWO_MODE_SQUEEZING                : {TWO_MODE_SQUEEZING, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING},
    BEAMSPLITTER_AND_TWO_MODE_SQUEEZING: {BEAMSPLITTER_AND_TWO_MODE_SQUEEZING},
    PARAMETRIC                        : {PARAMETRIC},
}

def _is_subgraph_slot(a: int, b: int) -> bool:
    """Return True if edge type a is a subgraph of edge type b at a single slot."""
    return b in _SUBGRAPH_SUPERSETS.get(a, {a})


# 1D triu array -> full symmetric N x N edge matrix.
def triu_to_edge_matrix(triu_array, n_nodes: int) -> np.ndarray:
    M = np.zeros((n_nodes, n_nodes), dtype=int)
    rows, cols = np.triu_indices(n_nodes)
    for k, (i,j) in enumerate(zip(rows, cols)):
        M[i,j] = triu_array[k]
        M[j,i] = triu_array[k]
    return M


# Inverse of triu_to_edge_matrix.
def edge_matrix_to_triu(edge_matrix: np.ndarray) -> np.ndarray:
    n = edge_matrix.shape[0]
    rows, cols = np.triu_indices(n)
    return edge_matrix[rows, cols]


# True if any graph in potential_subgraphs is a subgraph of any in
# edge_matrices, using _is_subgraph_slot per entry (not integer <=).
def check_if_subgraph(edge_matrices, potential_subgraphs) -> bool:
    edge_matrices       = np.atleast_3d(np.array(edge_matrices))
    potential_subgraphs = np.atleast_3d(np.array(potential_subgraphs))
    for sub in potential_subgraphs:
        for mat in edge_matrices:
            if all(_is_subgraph_slot(int(sub[i,j]), int(mat[i,j]))
                   for i in range(sub.shape[0])
                   for j in range(sub.shape[1])):
                return True
    return False


# Same as check_if_subgraph but on 1D triu arrays.
def check_if_subgraph_triu(triu_matrices, potential_subgraphs_triu) -> bool:
    triu_matrices            = np.atleast_2d(triu_matrices)
    potential_subgraphs_triu = np.atleast_2d(potential_subgraphs_triu)
    for sub in potential_subgraphs_triu:
        for mat in triu_matrices:
            if all(_is_subgraph_slot(int(sub[k]), int(mat[k]))
                   for k in range(len(sub))):
                return True
    return False


# §4 step 3 / §5 canonicalise: quotient equivalent triu_arrays by the
# residual gauge symmetry of permuting auxiliary modes that share a node
# type (cavity<->cavity, mechanical<->mechanical) — relabelling two
# interchangeable cavities doesn't change the physics. fixed_mode_ids (e.g.
# target_mode_ids) are never permuted. Overall scale / per-channel rate /
# local-phase gauge freedoms act on the continuous (G,C) parameters, not the
# discrete triu_array, and are already handled elsewhere (reference-edge
# pinning in covariance_optimizer.give_conditions_func_with_conditions).
# max_permutations caps the cost — if the residual group is too big,
# canonicalisation is skipped (dedup by exact equality only); this function
# is a speed/cleanliness aid, never a correctness requirement.
# Returns one canonical (lexicographically smallest image) representative
# per equivalence class.
def canonicalise_by_gauge(
    triu_arrays,
    node_types: List[str],
    fixed_mode_ids: Optional[List[int]] = None,
    max_permutations: int = 2000,
):
    N = len(node_types)
    fixed = set(fixed_mode_ids or [])

    movable_by_type = {}
    for i, t in enumerate(node_types):
        if i in fixed:
            continue
        movable_by_type.setdefault(t, []).append(i)
    groups = [g for g in movable_by_type.values() if len(g) > 1]

    def dedup_exact(arrays):
        seen, out = set(), []
        for t in arrays:
            key = tuple(int(x) for x in np.asarray(t))
            if key not in seen:
                seen.add(key)
                out.append(np.asarray(t))
        return out

    if not groups:
        return dedup_exact(triu_arrays)

    total_perms = 1
    for g in groups:
        total_perms *= math.factorial(len(g))
    if total_perms > max_permutations:
        return dedup_exact(triu_arrays)

    perm_choices_per_group = [list(permutations(g)) for g in groups]

    canonical = {}
    for t in triu_arrays:
        t = np.asarray(t)
        M = triu_to_edge_matrix(t, N)
        best_key = None
        best_arr = None
        for combo in product(*perm_choices_per_group):
            perm = list(range(N))
            for group_orig, group_image in zip(groups, combo):
                for orig_idx, image_idx in zip(group_orig, group_image):
                    perm[orig_idx] = image_idx
            M_perm = M[np.ix_(perm, perm)]
            triu_perm = edge_matrix_to_triu(M_perm)
            key = tuple(int(x) for x in triu_perm)
            if best_key is None or key < best_key:
                best_key, best_arr = key, triu_perm
        canonical.setdefault(best_key, best_arr)

    return list(canonical.values())


# Convert a triu_array to a constraint list — the key bridge from graph
# encoding to the optimizer. val=4 (BS+TMS) appends BOTH constraints,
# creating two edges/coupling strengths for that one slot (Kronwald).
def translate_triu_to_conditions(triu_array, node_types: List[str]) -> list:
    try:
        from reservoir_engineering.constraints import (
            Constraint_coupling_absent,
            Constraint_coupling_beamsplitter,
            Constraint_coupling_two_mode_squeezing)
    except ImportError:
        from constraints import (
            Constraint_coupling_absent,
            Constraint_coupling_beamsplitter,
            Constraint_coupling_two_mode_squeezing)

    n = len(node_types)
    rows, cols = np.triu_indices(n)
    conditions = []

    for k, (i, j) in enumerate(zip(rows, cols)):
        val = int(triu_array[k])
        if i == j:
            if val == NO_COUPLING:
                conditions.append(Constraint_coupling_absent(i, j))
        else:
            if val == NO_COUPLING:
                conditions.append(Constraint_coupling_absent(i, j))
            elif val == BEAMSPLITTER:
                conditions.append(Constraint_coupling_beamsplitter(i, j))
            elif val == TWO_MODE_SQUEEZING:
                conditions.append(Constraint_coupling_two_mode_squeezing(i, j))
            elif val == BEAMSPLITTER_AND_TWO_MODE_SQUEEZING:
                conditions.append(Constraint_coupling_beamsplitter(i, j))
                conditions.append(Constraint_coupling_two_mode_squeezing(i, j))

    return conditions


# Inverse of translate_triu_to_conditions — used to recover a triu_array
# from the constraints "accidentally" satisfied by a solved topology.
def translate_conditions_to_triu(conditions: list, n_nodes: int, node_types: List[str]) -> np.ndarray:
    try:
        from reservoir_engineering.constraints import (
            Constraint_coupling_absent,
            Constraint_coupling_beamsplitter,
            Constraint_coupling_two_mode_squeezing)
    except ImportError:
        from constraints import (
            Constraint_coupling_absent,
            Constraint_coupling_beamsplitter,
            Constraint_coupling_two_mode_squeezing)

    rows, cols = np.triu_indices(n_nodes)
    triu = np.zeros(len(rows), dtype=int)

    for k, (i, j) in enumerate(zip(rows, cols)):
        key = [min(i,j), max(i,j)]
        absent = any(isinstance(c, Constraint_coupling_absent) and c.idxs == key
                     for c in conditions)
        bs     = any(isinstance(c, Constraint_coupling_beamsplitter) and c.idxs == key
                     for c in conditions)
        tms    = any(isinstance(c, Constraint_coupling_two_mode_squeezing) and c.idxs == key
                     for c in conditions)

        if absent:               triu[k] = NO_COUPLING
        elif bs and tms:         triu[k] = BEAMSPLITTER_AND_TWO_MODE_SQUEEZING
        elif bs:                 triu[k] = BEAMSPLITTER
        elif tms:                triu[k] = TWO_MODE_SQUEEZING
        else:                    triu[k] = NO_COUPLING

    return triu


# Count and classify the edges in a topology (num BS/TMS/parametric edges,
# total complexity, connectivity, whether a cavity is present).
def characterize_topology(triu_array, node_types: List[str]) -> dict:
    n = len(node_types)
    rows, cols = np.triu_indices(n)

    num_bs   = sum(1 for v in triu_array if v in (BEAMSPLITTER, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING))
    num_tms  = sum(1 for v in triu_array if v in (TWO_MODE_SQUEEZING, BEAMSPLITTER_AND_TWO_MODE_SQUEEZING))
    num_para = sum(1 for k, v in enumerate(triu_array) if v == PARAMETRIC and rows[k] == cols[k])

    adj = triu_to_edge_matrix(triu_array, n) > 0
    visited, stack = set(), [0]
    while stack:
        nd = stack.pop()
        if nd not in visited:
            visited.add(nd)
            stack.extend(j for j in range(n) if adj[nd, j] and j not in visited)

    return {
        'num_bs_couplings':    num_bs,
        'num_tms_couplings':   num_tms,
        'num_parametric':      num_para,
        'num_couplings':       num_bs + num_tms + num_para,
        'num_active_couplings':num_tms + num_para,
        'complexity':          int(np.sum(triu_array)),
        'has_cavity':          any(t == 'cavity' for t in node_types),
        'is_connected':        len(visited) == n,
    }


# Minimum number of distinct pump-laser tones needed for this topology
# (search over mode-frequency assignments if mode_freqs not given). Not yet
# implemented.
def find_min_number_pump_tones(triu_array, node_types: List[str], mode_freqs=None):
    pass


# Number of distinct topologies for N modes: 3 choices per off-diagonal pair
# (no/BS/TMS) x 2 per diagonal (no/parametric) = 4^(N(N-1)/2) * 2^N. An
# upper bound — physical pruning (cavity required, connected, stable)
# removes most of it in practice.
def calc_number_of_possibilities(node_types: List[str]) -> int:
    N = len(node_types)
    off_diag = N * (N - 1) // 2
    return 4**off_diag * 2**N


# Data container for one graph topology (node types + edge matrix), mirrors
# autoscatter's Architecture class. Used by CovarianceOptimizer to represent
# candidate topologies.
class TopologyGraph:

    # edge_matrix_or_triu: either N x N edge matrix or 1D triu array.
    def __init__(self, node_types: List[str], edge_matrix_or_triu):
        self.node_types = list(node_types)
        self.num_nodes = len(node_types)
        arr = np.array(edge_matrix_or_triu)

        if arr.ndim == 1:
            self.triu_array = arr
            self.edge_matrix = triu_to_edge_matrix(arr, self.num_nodes)
        else:
            self.edge_matrix = arr
            self.triu_array = edge_matrix_to_triu(arr)

    @property
    def num_edges(self) -> int:
        return int(np.sum(self.triu_array > 0))

    # Sum of triu entries; used to sort topologies in the BFS outer loop
    # (one BS edge -> complexity 1, one BS + one TMS -> complexity 3, ...).
    @property
    def complexity(self) -> int:
        return(int(np.sum(self.triu_array)))

    # Convert to the {nodes, edges} dict format covariance_physics.py
    # consumes. val=4 (BS+TMS) creates TWO edge dicts for that slot (two
    # separate coupling strengths — the Kronwald topology). 'strength' is a
    # placeholder; the optimizer supplies real values via coupling_strengths.
    def to_nodes_edges_dicts(self, default_kappa=1.0, default_gamma=0.01, default_n_th=0.0):
        nodes = []
        for i, t in enumerate(self.node_types):
            if t == 'cavity':
                nodes.append({'id': i, 'type': 'cavity',
                               'kappa': default_kappa, 'delta': 0.0})
            else:
                nodes.append({'id': i, 'type': 'mechanical',
                               'gamma': default_gamma, 'n_th': default_n_th, 'delta': 0.0})
        edges = []
        rows, cols = np.triu_indices(self.num_nodes)
        for k, (i, j) in enumerate(zip(rows, cols)):
            val = int(self.triu_array[k])
            if val == BEAMSPLITTER:
                edges.append({'i': i, 'j': j, 'type': 'beamsplitter',        'strength': 1.0})
            elif val == TWO_MODE_SQUEEZING:
                edges.append({'i': i, 'j': j, 'type': 'two_mode_squeezing',  'strength': 1.0})
            elif val == PARAMETRIC:
                edges.append({'i': i, 'j': i, 'type': 'parametric',          'strength': 1.0})
            elif val == BEAMSPLITTER_AND_TWO_MODE_SQUEEZING:
                edges.append({'i': i, 'j': j, 'type': 'beamsplitter',        'strength': 1.0})
                edges.append({'i': i, 'j': j, 'type': 'two_mode_squeezing',  'strength': 1.0})

        return nodes, edges

    def to_conditions(self) -> list:
        return translate_triu_to_conditions(self.triu_array, self.node_types)

    # True if self's edges are all contained in other (same or subgraph type).
    def is_subgraph_of(self, other: 'TopologyGraph') -> bool:
        return check_if_subgraph_triu([other.triu_array], [self.triu_array])

    # No cavity -> no dissipation path -> no stable steady state.
    def has_cavity(self) -> bool:
        return any(t == 'cavity' for t in self.node_types)

    def is_connected(self) -> bool:
        adj = self.edge_matrix > 0
        visited, stack = set(), [0]
        while stack:
            nd = stack.pop()
            if nd not in visited:
                visited.add(nd)
                stack.extend(j for j in range(self.num_nodes)
                              if adj[nd, j] and j not in visited)
        return len(visited) == self.num_nodes

    def __eq__(self, other) -> bool:
        return (self.node_types == other.node_types and
                np.array_equal(self.triu_array, other.triu_array))

    def __hash__(self) -> int:
        return hash((tuple(self.node_types), self.triu_array.tobytes()))

    # Maximally-connected graph: every off-diagonal pair gets BS+TMS (no
    # parametric on the diagonal — destabilises at large lambda). Starting
    # point for the pruning search.
    @classmethod
    def fully_connected(cls, node_types: List[str]) -> 'TopologyGraph':
        n = len(node_types)
        rows, cols = np.triu_indices(n)
        triu = np.array([NO_COUPLING if i == j else BEAMSPLITTER_AND_TWO_MODE_SQUEEZING
                         for i, j in zip(rows, cols)], dtype=int)
        return cls(node_types, triu)

    @classmethod
    def empty(cls, node_types: List[str]) -> 'TopologyGraph':
        n = len(node_types)
        return cls(node_types, np.zeros(n*(n+1)//2, dtype=int))
