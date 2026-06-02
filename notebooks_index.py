"""
notebooks_index.py
==================
Catalog of all example notebooks in this repository, organised by project.

Run this file to print the catalog:
    python3 notebooks_index.py

-------------------------------------------------------------------
PROJECT 1: AUTOSCATTER  (notebooks 1–6)
-------------------------------------------------------------------
AutoScatter is an algorithm that automatically discovers quantum network
topologies (graphs of coupled bosonic modes) whose scattering matrices
match a user-specified target.

Physics:   input-output theory  →  S-matrix  (frequency domain)
Method:    gradient-based optimisation (JAX autodiff) over coupling strengths
           for a fixed symbolic topology, plus topology search via random
           restarts and graph mutations.
Reference: Landgraaf et al. (2024)

Notebooks  (all located in  notebooks/  subfolder)
---------
notebooks/1_isolator.ipynb
    Minimal AutoScatter example.
    Target: 2-port isolator  S = [[0, 1], [0, 0]]  (unidirectional transmission).
    Demonstrates the basic API: define architecture, run optimizer, inspect result.

notebooks/2_circulator.ipynb
    Minimal AutoScatter example.
    Target: 3-port circulator  S_ij = δ_{i, j+1 mod 3}  (cyclic routing).
    Shows how changing the target S-matrix changes the discovered graph.

notebooks/3_directional_coupler.ipynb
    Automated discovery of directional couplers.
    Target: N-to-1 combiner with zero backscattering.
    Topology search from scratch; AutoScatter discovers the graph structure.

notebooks/4_directional_coupler_generalisation.ipynb
    Generalise discovered solutions.
    Takes the graph found in notebook 3 and analytically extends it to
    arbitrary port count N using the symbolic coupling parametrisation.

notebooks/5_directional_quantum_limited_amplifier.ipynb
    Automated discovery of directional amplifiers.
    Target: phase-preserving amplifier with quantum-limited noise.
    Demonstrates multi-objective constraints (gain + noise figure).

notebooks/6_optomechanical_circulator.ipynb
    Automated discovery of asymptotic solutions.
    Target: optomechanical circulator mediated by far-detuned bus modes.
    Shows how AutoScatter handles near-degenerate (asymptotic) topologies.


-------------------------------------------------------------------
PROJECT 2: RESERVOIR ENGINEERING  (notebooks 7–8+)
-------------------------------------------------------------------
An extension of AutoScatter into the open-quantum-systems domain.
Instead of targeting a scattering matrix, we target a steady-state
covariance matrix — allowing discovery of topologies that engineer
specific quantum states (squeezed, entangled, etc.) via dissipation.

Physics:   Lindblad master equation  →  Lyapunov equation  (time domain)
           A·σ + σ·Aᵀ + D = 0  solved via Kronecker-product trick
Method:    gradient-based parameter optimisation (same JAX machinery)
           + differentiable topology pruning and/or evolutionary search
New code:  reservoir_engineering/  package (see workplan there)

Notebooks  (all located in  notebooks/  subfolder)
---------
notebooks/7_kronwald_squeezing.ipynb
    Reference implementation of the Kronwald dissipative squeezing scheme.
    Hard-coded 2-mode topology (cavity + mechanics); validates the Lyapunov
    solver and squeezing formula r = atanh(ν/g) against numerics.
    This is the "ground truth" that notebook 8 must rediscover automatically.
    Source module: kronwald_optimizer.py

notebooks/8_topology_discovery.ipynb  [TO BE CREATED — see reservoir_engineering/]
    Automated discovery of dissipative squeezing topologies.
    Runs PruningSearcher and EvolutionarySearcher on a 2-mode system;
    the algorithm should rediscover the Kronwald graph (beamsplitter + TMS)
    without being told the answer.
    Source module: reservoir_engineering/


-------------------------------------------------------------------
Quick-start
-------------------------------------------------------------------
AutoScatter examples (notebooks 1-6):
    jupyter notebook notebooks/1_isolator.ipynb

Kronwald reference (notebook 7):
    jupyter notebook notebooks/7_kronwald_squeezing.ipynb
    # or run from terminal:
    python3 kronwald_optimizer.py 1.0

Reservoir engineering (notebook 8, once implemented):
    jupyter notebook notebooks/8_topology_discovery.ipynb
"""

AUTOSCATTER_NOTEBOOKS = [
    {
        "file":    "notebooks/1_isolator.ipynb",
        "title":   "Minimal example — Isolator",
        "target":  "2-port isolator S-matrix",
        "notes":   "Basic API walkthrough",
    },
    {
        "file":    "notebooks/2_circulator.ipynb",
        "title":   "Minimal example — Circulator",
        "target":  "3-port circulator S-matrix",
        "notes":   "Same API, more complex target",
    },
    {
        "file":    "notebooks/3_directional_coupler.ipynb",
        "title":   "Automated discovery — Directional coupler",
        "target":  "N-to-1 coupler with no backscatter",
        "notes":   "Full topology search from scratch",
    },
    {
        "file":    "notebooks/4_directional_coupler_generalisation.ipynb",
        "title":   "Generalise discovered solutions",
        "target":  "Arbitrary-N directional coupler",
        "notes":   "Analytical extension of notebook 3 result",
    },
    {
        "file":    "notebooks/5_directional_quantum_limited_amplifier.ipynb",
        "title":   "Automated discovery — Directional amplifier",
        "target":  "Phase-preserving quantum-limited amplifier",
        "notes":   "Multi-objective: gain + noise",
    },
    {
        "file":    "notebooks/6_optomechanical_circulator.ipynb",
        "title":   "Automated discovery — Asymptotic solutions",
        "target":  "Optomechanical circulator (far-detuned bus)",
        "notes":   "Near-degenerate / asymptotic topologies",
    },
]

RESERVOIR_ENGINEERING_NOTEBOOKS = [
    {
        "file":    "notebooks/7_kronwald_squeezing.ipynb",
        "title":   "Kronwald dissipative squeezing (reference)",
        "target":  "Mechanical covariance σ = ½ diag(e^{-2r}, e^{+2r})",
        "notes":   "Hard-coded topology; validates Lyapunov solver",
        "module":  "kronwald_optimizer.py",
    },
    {
        "file":    "notebooks/8_topology_discovery.ipynb",
        "title":   "Automated topology discovery — Squeezing [TODO]",
        "target":  "Same as notebook 7 but discovered automatically",
        "notes":   "Uses reservoir_engineering/ package; must rediscover Kronwald",
        "module":  "reservoir_engineering/",
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print("AUTOSCATTER — notebooks 1-6  (S-matrix targets)")
    print("=" * 60)
    for nb in AUTOSCATTER_NOTEBOOKS:
        print(f"  {nb['file']}")
        print(f"    {nb['title']}")
        print(f"    Target : {nb['target']}")
        print(f"    Notes  : {nb['notes']}")
        print()

    print("=" * 60)
    print("RESERVOIR ENGINEERING — notebooks 7-8+  (covariance targets)")
    print("=" * 60)
    for nb in RESERVOIR_ENGINEERING_NOTEBOOKS:
        status = "[DONE]" if "TODO" not in nb["notes"] else "[TODO]"
        print(f"  {nb['file']}  {status}")
        print(f"    {nb['title']}")
        print(f"    Target : {nb['target']}")
        print(f"    Module : {nb['module']}")
        print(f"    Notes  : {nb['notes']}")
        print()
