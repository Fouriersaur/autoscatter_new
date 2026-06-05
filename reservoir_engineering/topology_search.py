"""
topology_search.py
==================
Graph encoding, topology utilities, and pump-tone counting.

MIRRORS: autoscatter/architecture.py  (1-to-1 conceptual mapping)

═══════════════════════════════════════════════════════════════════════════════
ROLE IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════

This file provides the DATA STRUCTURES and UTILITY FUNCTIONS that describe a
graph topology.  It does NOT run the optimiser — that lives in
covariance_optimizer.py.  It answers questions like:

  • How do I encode a graph as a compact array?                (triu encoding)
  • How do I convert between that encoding and constraint lists? (translate_*)
  • Is graph A a subgraph of graph B?                          (check_if_subgraph)
  • How many pump laser tones does this topology require?       (find_min_number_pump_tones)
  • What are all possible topologies at a given complexity?     (enumerate, characterize)

Exact same role as autoscatter/architecture.py, which handles:
  • Adjacency matrix encoding (triu_matrix ↔ full adjacency)
  • Subgraph checks
  • Constraint list ↔ graph translation
  • find_min_number_of_pumps (counts minimum parametric drives needed)
  • Architecture class (graph data structure)
  • check_if_subgraph_upper_triangle, characterize_architectures

═══════════════════════════════════════════════════════════════════════════════
GRAPH ENCODING  (mirrors architecture.py's triu_matrix convention)
═══════════════════════════════════════════════════════════════════════════════

A topology of N modes is encoded as a 1D integer array of length N(N+1)/2,
the upper triangle (including diagonal) of an N×N edge matrix.

    triu_array[k] = edge type for the k-th upper-triangle entry
    where k indexes (i,j) pairs with j ≥ i, in row-major order.

Edge type values (mirrors NO_COUPLING=0, COUPLING_WITHOUT_PHASE=1,
                  COUPLING_WITH_PHASE=2, DETUNING=1 in architecture.py):
    0 = NO_COUPLING        — no interaction between i and j
    1 = BEAMSPLITTER       — BS coupling (real, energy-conserving)
    2 = TWO_MODE_SQUEEZING — TMS coupling (parametric, active)
    3 = PARAMETRIC         — single-mode squeezing on mode i  (diagonal only)

"Complexity" of a topology = sum of triu_array entries.
Higher complexity = more / stronger edge types.
The breadth-first search in covariance_optimizer.py iterates from
low complexity (sparse graphs) to high complexity (dense graphs).
This mirrors AutoScatter's complexity_level concept exactly.

Physical constraint on which edges are possible:
    For a cavity–cavity pair: BS or TMS are both possible
    For a cavity–mechanical pair: BS or TMS are both possible
    For a mechanical–mechanical pair: BS or TMS are possible but rare in practice
    Diagonal: 0 (no single-mode squeezing) or 3 (OPA on that mode)

    Rule: at least one 'cavity' node must be present in any valid topology.
    A graph of only mechanical modes has no dissipation → no Lyapunov solution.

═══════════════════════════════════════════════════════════════════════════════
PUMP LASER COUNTING  (mirrors find_min_number_of_pumps in architecture.py)
═══════════════════════════════════════════════════════════════════════════════

AutoScatter counts the minimum number of parametric pumps (laser tones) required
to realise a graph.  Each pump drives a specific frequency difference ω_pump.

Here, modes have physical frequencies ω_i.  Each interaction requires:
    Beamsplitter (i,j):   pump at |ω_i − ω_j|  (red-detuned drive, energy-conserving)
    TMS (i,j):            pump at  ω_i + ω_j    (blue-detuned drive, parametric)
    Parametric (i,i):     pump at  2ω_i          (degenerate OPA drive)

If two different edges require the same pump frequency, a single laser tone
can drive both — reducing the total count.

The function find_min_number_pump_tones searches over possible mode frequency
assignments to minimise the total number of distinct pump tones.
This is the direct analogue of find_min_number_of_pumps in architecture.py,
which searches over mode "label" assignments to count pump lasers.

═══════════════════════════════════════════════════════════════════════════════
CONSTANTS  (mirrors architecture.py's NO_COUPLING, DETUNING, etc.)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from itertools import product
from typing import List, Optional

# Edge type integer constants — mirrors:
#   NO_COUPLING = 0, DETUNING = 1, COUPLING_WITHOUT_PHASE = 1, COUPLING_WITH_PHASE = 2
# in autoscatter/architecture.py

NO_COUPLING = 0         # no edge between modes i and j
BEAMSPLITTER = 1        # beam-splitter (energy-conserving, red-sideband drive)
TWO_MODE_SQUEEZING = 2  # two-mode squeezing (parametric, blue-sideband drive)
PARAMETRIC = 3          # single-mode squeezing on mode i (degenerate OPA)


# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: triu_to_edge_matrix(triu_array, n_nodes) → np.ndarray (N, N)
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: triu_to_adjacency_matrix(triu_matrix) in autoscatter/architecture.py
#
# Convert a 1D upper-triangle array to a full symmetric N×N edge matrix.
#
# Parameters:
#   triu_array : 1D int array, length N(N+1)/2 - [00, 01, 02, 11, 12, 22]
#   n_nodes    : int, number of modes N
#
# Returns:
#   edge_matrix : np.ndarray (N, N), symmetric
#                 edge_matrix[i,j] = edge_matrix[j,i] = edge type for (i,j)
#                 Diagonal: edge_matrix[i,i] = parametric type (0 or 3)
# 
# Method:
#   idxs_upper_triangle = np.triu_indices(n_nodes)
#   Initialize edge_matrix = np.zeros([n_nodes, n_nodes], dtype=int)
#   For each (i, j, val) in zip(idxs_upper_triangle.T, triu_array):
#       edge_matrix[i, j] = val
#       edge_matrix[j, i] = val  ← symmetric
#   Return edge_matrix
#
# Note: diagonal entries (i==i) satisfy edge_matrix[i,i] = triu_array[k]
# where k is the diagonal index. They are NOT doubled by the symmetry copy.

def triu_to_edge_matrix(triu_array, n_nodes: int) -> np.ndarray:
    M = np.zeros((n_nodes, n_nodes), dtype=int)
    rows, cols = np.triu_indices(n_nodes)
    for k, (i,j) in enumerate(zip(rows, cols)):
        M[i,j] = triu_array[k]
        M[j,i] = triu_array[k]

    return M 

# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: edge_matrix_to_triu(edge_matrix) → np.ndarray (1D)
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: adjacency_to_triu_matrix(adjacency_matrix) in autoscatter/architecture.py
#
# Convert a symmetric N×N edge matrix to its compact 1D upper-triangle form.
#
# Parameters:
#   edge_matrix : np.ndarray (N, N)
#
# Returns:
#   triu_array : 1D int array, length N(N+1)/2
#
# Method:
#   triu_indices = np.triu_indices(N)
#   return edge_matrix[triu_indices]

def edge_matrix_to_triu(edge_matrix: np.ndarray) -> np.ndarray:
    n = edge_matrix.shape[0]
    rows, cols = np.triu_indices(n)
    return edge_matrix[rows, cols]

# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: check_if_subgraph(edge_matrices, potential_subgraphs) → bool
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: check_if_subgraph(coupling_matrices, potential_subgraphs)
#          in autoscatter/architecture.py  (exact analogue)
#
# Check whether any graph in `potential_subgraphs` is a subgraph of any
# graph in `edge_matrices`.
#
# Graph A is a subgraph of graph B iff A[i,j] ≤ B[i,j] for all (i,j).
# (Lower edge type ≤ higher edge type — absence is a subgraph of anything.)
#
# Parameters:
#   edge_matrices        : np.ndarray shape (M, N, N) OR (N, N)
#   potential_subgraphs  : np.ndarray shape (K, N, N) OR (N, N)
#
# Returns True if any potential subgraph is a subgraph of any edge_matrix.
#
# Implementation: np.any(np.sum((edge_matrices - potential_subgraphs) < 0, (-1,-2)) == 0)
# This mirrors AutoScatter's check_if_subgraph EXACTLY (same logic, different array names).

# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: check_if_subgraph_triu(triu_matrices, potential_subgraphs_triu) → bool
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: check_if_subgraph_upper_triangle in autoscatter/architecture.py
#
# Same as check_if_subgraph but operates on 1D upper-triangle arrays
# instead of full N×N matrices. More memory-efficient for large searches.
#
# Returns True if any potential_subgraph_triu is elementwise ≤ any triu_matrix.
# A is a subgraph of B iff A[k] <= B[k] for all k.

def check_if_subgraph_triu(triu_matrices, potential_subgraphs_triu) -> bool:
    triu_matrices            = np.atleast_2d(triu_matrices)
    potential_subgraphs_triu = np.atleast_2d(potential_subgraphs_triu)
    for sub in potential_subgraphs_triu:
        for mat in triu_matrices:
            if np.all(sub <= mat):
                return True
    return False

# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: translate_triu_to_conditions(triu_array, node_types) → list
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: translate_upper_triangle_coupling_matrix_to_conditions(coupling_matrix_upper_triangle)
#          in autoscatter/architecture.py  (exact analogue)
#
# Convert a triu_array encoding to a list of constraint objects.
# This is the KEY function that connects graph encoding to the optimizer.
#
# For each upper-triangle entry (i, j, val):
#   val == NO_COUPLING (0):
#       → append Constraint_coupling_absent(i, j)
#   val == BEAMSPLITTER (1):
#       → append Constraint_coupling_beamsplitter(i, j)   ← only BS allowed
#   val == TWO_MODE_SQUEEZING (2):
#       → append Constraint_coupling_two_mode_squeezing(i, j)   ← only TMS allowed
#   val == PARAMETRIC (3) and i == j:
#       → no constraint (parametric squeezing allowed on diagonal)
#   val == NO_COUPLING (0) and i == j:
#       → append Constraint_coupling_absent(i, i)   ← no self-squeezing
#
# The returned list tells the optimizer which edges are absent (→ strength=0)
# and which edge types are enforced (→ constrained block structure in A).
#
# AutoScatter analogue:
#   NO_COUPLING → Constraint_coupling_zero(i,j)
#   COUPLING_WITHOUT_PHASE → Constraint_coupling_phase_zero(i,j)
#   COUPLING_WITH_PHASE → no constraint (both BS and TMS allowed in our case too)

def translate_triu_to_conditions(triu_array, node_types: List[str]) -> list:

    from reservoir_engineering.constraints import (
    Constraint_coupling_absent,
    Constraint_coupling_beamsplitter,
    Constraint_coupling_two_mode_squeezing)

    n = len(node_types)

    pass


# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: translate_conditions_to_triu(conditions, n_nodes, node_types) → np.ndarray
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: translate_conditions_to_upper_triangle_coupling_matrix(conditions, num_modes)
#          in autoscatter/architecture.py  (exact analogue)
#
# Reverse of translate_triu_to_conditions.
# Convert a list of constraint objects back to a triu_array.
#
# For each upper-triangle entry (i, j):
#   If Constraint_coupling_absent(i,j) in conditions:
#       triu_array[k] = NO_COUPLING
#   Elif Constraint_coupling_beamsplitter(i,j) in conditions:
#       triu_array[k] = BEAMSPLITTER
#   Elif Constraint_coupling_two_mode_squeezing(i,j) in conditions:
#       triu_array[k] = TWO_MODE_SQUEEZING
#   Else (no constraint = edge present, type unconstrained):
#       triu_array[k] = TWO_MODE_SQUEEZING   ← most general off-diagonal type
#       OR   BEAMSPLITTER if same-frequency pair (a physical convention)
#
# Used by check_all_constraints in CovarianceOptimizer to convert back
# the inferred constraints from a solution to a triu_array for storage.

def translate_conditions_to_triu(conditions: list, n_nodes: int, node_types: List[str]) -> np.ndarray:
    pass


# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: characterize_topology(triu_array, node_types) → dict
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: characterize_architecture(arch) in autoscatter/architecture.py
#          AND characterize_architectures(list_of_architectures)
#
# Count and classify the edges in a topology.
#
# Returns dict:
#   {
#     'num_bs_couplings'   : int  ← number of beamsplitter edges
#     'num_tms_couplings'  : int  ← number of two-mode squeezing edges
#     'num_parametric'     : int  ← number of single-mode squeezing edges (diagonal)
#     'num_couplings'      : int  ← total edges (bs + tms + parametric)
#     'num_active_couplings': int ← num_tms + num_parametric (active = need pump)
#     'complexity'         : int  ← sum of triu_array entries
#     'has_cavity'         : bool ← True if at least one 'cavity' in node_types
#     'is_connected'       : bool ← True if graph is connected
#   }
#
# AutoScatter analogue returns: num_detunings, num_real_couplings,
#   num_complex_couplings_and_squeezings. Our dict is the same spirit.

def characterize_topology(triu_array, node_types: List[str]) -> dict:
    pass


# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: find_min_number_pump_tones(triu_array, node_types, mode_freqs=None) → (int, list)
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: find_min_number_of_pumps(triu_matrix) in autoscatter/architecture.py
#          (conceptually identical; physical interpretation differs)
#
# Find the minimum number of distinct pump laser tones required to implement
# all the interactions in this topology.
#
# Physical background:
#   Each BS edge (i,j) requires a red-detuned drive at frequency ω_pump = |ω_i - ω_j|.
#   Each TMS edge (i,j) requires a blue-detuned drive at frequency ω_pump = ω_i + ω_j.
#   Each parametric edge (i,i) requires a drive at 2ω_i.
#   If multiple edges happen to need the same pump frequency, they can share a laser.
#
# In AutoScatter: modes with the SAME label share the same frequency.
# Here: mode_freqs is a list of mode frequencies. If None, search over all
# possible frequency assignments (integers 0, 1, 2, ...) to minimise pump count.
#
# Method (mirrors find_min_number_of_pumps):
#   If mode_freqs provided: directly compute required pump frequencies and count uniques.
#   If mode_freqs=None:
#     Iterate over all possible frequency assignments (integers per mode).
#     For each assignment, count the number of distinct pump frequencies.
#     Return the minimum count across all assignments.
#
# Returns:
#   (min_num_pumps, best_freq_assignments)
#   best_freq_assignments is a list of frequency assignment arrays that achieve
#   the minimum pump count, analogous to labels_results in AutoScatter.
#
# Equivalent to AutoScatter's find_min_number_of_pumps but for optomechanics.

def find_min_number_pump_tones(triu_array, node_types: List[str], mode_freqs=None):
    pass


# ───────────────────────────────────────────────────────────────────────────
# FUNCTION: calc_number_of_possibilities(node_types) → int
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: calc_number_of_possibilities(mode_types) in autoscatter/architecture.py
#
# Calculate how many distinct topologies exist for N modes of the given types.
#
# For each pair (i,j) with j > i:
#   3 choices: NO_COUPLING, BEAMSPLITTER, TWO_MODE_SQUEEZING
# For each diagonal (i,i):
#   2 choices: NO_COUPLING, PARAMETRIC
#
# Total = 3^(N*(N-1)/2) × 2^N
#
# Note: AutoScatter's version accounts for mode_types (bool list) to determine
# which pairs can have squeezing. Here, all pairs can in principle have both
# BS and TMS (the physical constraint that TMS requires different frequencies
# is noted in the pump-counting function, not in the combinatorics).
#
# This gives an UPPER BOUND on the search space — in practice, physical
# constraints (at least one cavity, connectedness, stability) prune most of it.

def calc_number_of_possibilities(node_types: List[str]) -> int:
    pass


# ───────────────────────────────────────────────────────────────────────────
# CLASS: TopologyGraph
# ───────────────────────────────────────────────────────────────────────────
# MIRRORS: Architecture class in autoscatter/architecture.py
#
# Data container for a single graph topology.
# Stores node types + edge matrix; provides utility methods.
# Used by CovarianceOptimizer to represent candidate topologies.

class TopologyGraph:

    # -----------------------------------------------------------------------
    # __init__(node_types, edge_matrix_or_triu)
    # -----------------------------------------------------------------------
    # MIRRORS: Architecture.__init__(num_modes, detunings, couplings_without_phase,
    #                                couplings_with_phase, coupling_matrix)
    #          — simplified because we accept the full edge_matrix directly.
    #
    # node_types            : list of str, length N  ('cavity'|'mechanical')
    # edge_matrix_or_triu   : either an N×N int array (edge_matrix) OR a 1D triu array.
    #                         If 1D, convert to N×N via triu_to_edge_matrix.
    #
    # Store:
    #   self.node_types    : list of str
    #   self.edge_matrix   : np.ndarray (N, N), symmetric
    #   self.num_nodes     : N
    #   self.triu_array    : 1D compact form (derived via edge_matrix_to_triu)

    def __init__(self, node_types: List[str], edge_matrix_or_triu):
        pass

    # -----------------------------------------------------------------------
    # Property: num_edges → int
    # -----------------------------------------------------------------------
    # Count of non-zero entries in the upper triangle (including diagonal).
    # Zero = no couplings at all. Measures sparsity.

    @property
    def num_edges(self) -> int:
        pass

    # -----------------------------------------------------------------------
    # Property: complexity → int
    # -----------------------------------------------------------------------
    # Sum of triu_array entries.
    # MIRRORS: sum(p_coupl) in Architecture_Optimizer.prepare_all_possible_combinations.
    # A graph with one BS edge has complexity 1.
    # A graph with one BS + one TMS has complexity 1+2=3.
    # Used to sort topologies in the BFS outer loop.

    @property
    def complexity(self) -> int:
        pass

    # -----------------------------------------------------------------------
    # to_nodes_edges_dicts(default_kappa, default_gamma, default_n_th)
    # -----------------------------------------------------------------------
    # Convert to the list-of-dicts format consumed by covariance_physics.py.
    # See covariance_physics.py for the exact dict format.
    #
    # nodes — one dict per node i:
    #   type='cavity' → {'id':i, 'type':'cavity', 'kappa':default_kappa, 'gamma':0., 'n_th':0.}
    #   type='mechanical' → {'id':i, 'type':'mechanical', 'kappa':0., 'gamma':default_gamma, 'n_th':default_n_th}
    #
    # edges — one dict per non-zero triu entry (i, j):
    #   val=1 → {'i':i, 'j':j, 'type':'beamsplitter',      'strength':1.}
    #   val=2 → {'i':i, 'j':j, 'type':'two_mode_squeezing', 'strength':1.}
    #   val=3 → {'i':i, 'j':i, 'type':'parametric',         'strength':1.}
    #
    # 'strength' is a placeholder; CovarianceOptimizer uses coupling_strengths array.

    def to_nodes_edges_dicts(self, default_kappa=1.0, default_gamma=0.01, default_n_th=0.0):
        pass

    # -----------------------------------------------------------------------
    # to_conditions() → list of constraint objects
    # -----------------------------------------------------------------------
    # Convert edge_matrix to a list of Constraint objects.
    # Calls translate_triu_to_conditions(self.triu_array, self.node_types).
    # Returns the condition list used by CovarianceOptimizer.

    def to_conditions(self) -> list:
        pass

    # -----------------------------------------------------------------------
    # is_subgraph_of(other_topology) → bool
    # -----------------------------------------------------------------------
    # MIRRORS: Architecture.is_subgraph_to(arch) in autoscatter/architecture.py
    #
    # Returns True if self is a subgraph of other_topology (every edge in
    # self also exists in other_topology with the same or lower type).
    # Uses check_if_subgraph_triu(self.triu_array, other_topology.triu_array).

    def is_subgraph_of(self, other: 'TopologyGraph') -> bool:
        pass

    # -----------------------------------------------------------------------
    # has_cavity() → bool
    # -----------------------------------------------------------------------
    # Returns True if at least one node has type 'cavity'.
    # A topology without a cavity has no dissipation path — no stable solution.
    # CovarianceOptimizer skips topologies where has_cavity() returns False.

    def has_cavity(self) -> bool:
        pass

    # -----------------------------------------------------------------------
    # is_connected() → bool
    # -----------------------------------------------------------------------
    # Returns True if the graph is connected (ignoring edge types).
    # Disconnected graphs may still achieve local targets on disconnected
    # subgraphs, but are usually not physical for state engineering.
    # BFS/DFS from node 0 — check all N nodes are reachable.

    def is_connected(self) -> bool:
        pass

    # -----------------------------------------------------------------------
    # __eq__(other) → bool  and  __hash__() → int
    # -----------------------------------------------------------------------
    # MIRRORS: Architecture.is_subgraph_to (used for deduplication)
    #
    # Two TopologyGraphs are equal if node_types and triu_array are identical.
    # __hash__ based on (tuple(node_types), triu_array.tobytes()).
    # Required for use in sets and as dict keys (deduplication in BFS).

    def __eq__(self, other) -> bool:
        pass

    def __hash__(self) -> int:
        pass

    # -----------------------------------------------------------------------
    # classmethod: fully_connected(node_types) → TopologyGraph
    # -----------------------------------------------------------------------
    # Build the maximally-connected graph: all off-diagonal pairs get
    # edge type TWO_MODE_SQUEEZING (2), diagonal = 0.
    # Starting point for the pruning searcher.
    # Analogous to the fully-constrained graph in AutoScatter (no coupling zeros).

    @classmethod
    def fully_connected(cls, node_types: List[str]) -> 'TopologyGraph':
        pass

    # -----------------------------------------------------------------------
    # classmethod: empty(node_types) → TopologyGraph
    # -----------------------------------------------------------------------
    # All-zero graph. Starting point for evolutionary search.
    # Analogous to a graph with all couplings set to zero in AutoScatter.

    @classmethod
    def empty(cls, node_types: List[str]) -> 'TopologyGraph':
        pass
