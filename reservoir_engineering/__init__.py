"""
reservoir_engineering
=====================
Automated discovery of dissipative quantum state engineering topologies.

Analogue of AutoScatter, but instead of targeting a scattering matrix
(S-matrix, frequency domain) we target a steady-state covariance matrix
(time domain) obtained from the Lyapunov equation.

Module structure
----------------
covariance_physics.py
    Low-level physics: build A and D matrices from a graph description,
    solve the Lyapunov equation, extract mode covariances.
    All functions are JAX-differentiable.

covariance_optimizer.py
    Parameter optimiser for a FIXED topology.
    Given a graph structure, finds the coupling strengths that minimise
    the Frobenius distance between the achieved and target covariance.

topology_search.py
    Topology search: discovers WHICH graph to use.
    Two strategies:
      PruningSearcher     — differentiable L1 relaxation (soft graph)
      EvolutionarySearcher — discrete mutations + gradient polishing

Typical workflow
----------------
1. Define target covariance (e.g. squeezed vacuum for one mechanical mode).
2. Run PruningSearcher on a fully-connected N-mode candidate graph.
3. Threshold gate weights → discrete topology.
4. Polish with ParameterOptimizer.
5. Compare discovered topology to known schemes (e.g. Kronwald).

Validation
----------
The 2-mode cavity+mechanics system with target squeezing r must
rediscover the Kronwald scheme: two edges (beamsplitter g, two-mode
squeezing ν) with ν/g = tanh(r).  See kronwald_optimizer.py for the
ground-truth reference.
"""

from reservoir_engineering.covariance_physics import (
    build_drift_matrix,
    build_diffusion_matrix,
    solve_lyapunov_kronecker,
    get_mode_covariance,
)
from reservoir_engineering.covariance_optimizer import ParameterOptimizer
from reservoir_engineering.topology_search import (
    TopologyGraph,
    PruningSearcher,
    EvolutionarySearcher,
)

__all__ = [
    "build_drift_matrix",
    "build_diffusion_matrix",
    "solve_lyapunov_kronecker",
    "get_mode_covariance",
    "ParameterOptimizer",
    "TopologyGraph",
    "PruningSearcher",
    "EvolutionarySearcher",
]
