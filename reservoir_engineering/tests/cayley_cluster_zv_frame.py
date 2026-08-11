"""
cayley_cluster_zv_frame.py
==========================
Feed the CAYLEY CLUSTER covariance matrix of cayley_cluster_rediscovery.py to
the certifying linear oracle, with the DRAIN SQUEEZED AT r = z — the same
squeezing the target covariance itself carries — and let the oracle discover
which topologies stabilise it.

This is cayley_cluster_certified.py's Phase 3 on its own, without the
vacuum-frame layered sweep (Phase 1) and vacuum-frame descent (Phase 2) that
dominate that script's runtime and answer a different question.

Why the drain squeezing matters
-------------------------------
complete_covariance's default gauge puts every drain in plain vacuum,
V_AA = (1/2)I. That gauge is COMPLETE — every physical scheme still appears —
but in it a passive (beamsplitter-only) network cannot stabilise a squeezed
target: with a vacuum drain the only remaining source of squeezing is the
Hamiltonian, so passive schemes show up wearing their gauge image, BS+TMS on
the drain edges. Squeezing the drain instead moves the search into the
Zippilli-Vitali frame (PRL 126, 020402 (2021)): passive network, all squeezing
supplied by the bath.

r = z is not asserted here. Step 1 asks linear_oracle.optimise_aux_state for
the drain state the graph actually needs — it minimises the first singular
value beyond the generic nullspace, since the design matrix drops rank exactly
where a solution appears — starting from a cold start with no hint of z. It
returns r* = z. Steps 2/3 then pin the drain at that value only so thousands
of oracle calls do not each re-derive it, and every reported endpoint is
re-checked with the detector afterwards (print_descent), which is free to
disagree.

What is claimed
---------------
Descent from the lattice maximum, multi-start with randomised neighbour order,
so different starts land in different corners of the valid region. Each
endpoint is LOCALLY IRREDUCIBLE with respect to what the oracle could prove:
the descent stops when no prune-neighbour was SHOWN valid, and an UNDECIDED
neighbour is not known to be invalid. So the honest claim is "no simpler
scheme was found from here", not "none exists". No lower bound is certified in
this frame — cayley_cluster_certified.py's Phase 1 bound is a VACUUM-frame
bound and does not transfer, since a squeezed drain is a resource that sweep
never had.

Usage:
    python3 cayley_cluster_zv_frame.py              # z=0.6, 40 descent starts
    python3 cayley_cluster_zv_frame.py 0.6 60       # z, number of descent starts
    python3 cayley_cluster_zv_frame.py 0.6 60 --curated   # also decide the six
                                                            curated topologies
    python3 cayley_cluster_zv_frame.py 0.6 0 --sweep=7    # EXHAUSTIVE sweep to
                                                            complexity 7, no descent
    ... --sweep=7 --budget=7200                     # cap the sweep's wall time
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import reservoir_engineering.linear_oracle as lo
from reservoir_engineering.targets import purity

from cayley_cluster_rediscovery import (
    ADJACENCY_P4, NODE_TYPES, TARGET_MODE_IDS, N_NODES,
    cayley_cluster_target, validate_target, candidate_topologies)
from cayley_cluster_certified import (descent_search, print_descent,
                                       layered_search, irreducible)


# Ask the oracle what drain state the target needs, with no hint of z.
#
# Probed on the passive drain->all4 star: that graph exists ONLY in the
# squeezed frame, so its rank-drop curve is the sharp one, and a graph that
# works at every drain state would instead relax to r* = 0 under the
# squeezing-cost term. Either answer is informative; the point is that the
# value is measured, not declared.
def detect_drain_squeezing(target, z, probe_triu):
    res = lo.optimise_aux_state(probe_triu, target, TARGET_MODE_IDS, NODE_TYPES,
                                 num_samples=24)
    r_star = float(res['aux_squeezing'][0][0])
    print(f'  probe graph        : {lo_describe(probe_triu)}')
    print(f'  reservoir required : {res["reservoir"]}')
    print(f'  r* (cold start)    : {r_star:.6f}      target squeezing z = {z}')
    print(f'  |r* - z|           : {abs(r_star - z):.2e}')
    print(f'  rank-drop score    : {res.get("rank_drop_score", float("nan")):.3e}'
          '   (~1e-16 = a solution appears exactly there)')
    return r_star, res


def lo_describe(triu):
    from reservoir_engineering.certified_search import describe
    return describe(triu, N_NODES)


# Exhaustive sweep by complexity level, IN THIS FRAME.
#
# Unlike the descent, this ENUMERATES: every graph up to the budget is either
# certified INVALID, shown VALID, or filed UNDECIDED. So it yields both the
# verdict counts and a genuine minimality claim — "no valid scheme exists
# below complexity C" — which the descent cannot give.
#
# The claim is frame-local. cayley_cluster_certified.py's Phase 1 runs the
# same sweep with UNSQUEEZED drains and its bound does not transfer here: a
# squeezed drain is a resource that sweep never had, which is exactly why
# passive schemes appear below its bound. The bound printed here is the
# squeezed-frame one and applies only at r = z, theta = 0.
def sweep_report(V_zv, target, max_c, budget=None, num_samples=24):
    print(f'\nExhaustive layered sweep, complexity <= {max_c}, in this frame')
    out = layered_search(V_zv, max_c, num_samples=num_samples, time_budget=budget)

    n_cand = sum(s['candidates'] for s in out['stats'])
    n_test = sum(s['tested'] for s in out['stats'])
    reached = max((s['complexity'] for s in out['stats']), default=-1)
    print(f'\n{"-" * 78}')
    print(f'VERDICT COUNTS  (complexity 0..{reached}'
          + (f', stopped early at {out["stopped_early"]}' if out['stopped_early'] is not None else '')
          + ')\n')
    print(f'  graphs in these levels          : {n_cand:,}')
    print(f'  INVALID by structural certificate: {out["invalid_structural"]:,} '
          f'({100 * out["invalid_structural"] / max(n_cand, 1):.1f}%, no numerics)')
    print(f'  skipped as supergraph of a valid : {out["skipped_supergraph"]:,}')
    print(f'  oracle calls                     : {n_test:,}')
    print(f'  INVALID by oracle                : {out["invalid_oracle"]:,}')
    print(f'  UNDECIDED                        : {len(out["undecided"]):,}'
          '   (NOT invalid — the oracle declined)')
    print(f'  VALID                            : {len(out["valid"]):,}')
    print(f'  wall time                        : {out["elapsed"]:.1f}s')

    irr = irreducible(out['valid'])
    if not irr:
        print(f'\nNo valid scheme at complexity <= {reached} in this frame.')
        print('That is not a proof none exists: every graph here was either certified')
        print('invalid or left UNDECIDED. Raise the budget.')
        return out

    lowest = int(np.sum(irr[0]))
    print(f'\n{len(irr)} IRREDUCIBLE valid scheme(s); lowest complexity {lowest}.')
    print(f'Certified: NO valid scheme exists below complexity {lowest} with the drain')
    print(f'squeezed at r = z, theta = 0 — every lower graph was certified INVALID')
    print(f'or left UNDECIDED (see the UNDECIDED count above; those are not proven '
          'invalid).\n')

    for rank, triu in enumerate(irr):
        info = lo.decide(triu, V_zv, TARGET_MODE_IDS, NODE_TYPES, num_samples=96)
        print(f'--- #{rank + 1}  complexity {int(np.sum(triu))} ---')
        print(f'  edges: {lo_describe(triu)}')
        if info['verdict'] != lo.VALID:
            print('  (re-check at higher num_samples did not reproduce the witness)\n')
            continue
        print(f'  stationarity residual {info["stationarity_residual"]:.2e}   '
              f'forward Lyapunov error {info["forward_error"]:.2e}')
        print(f'  normalised stability margin {info["stability_margin"]:.4f}   '
              f'rank(Upsilon) = {info["upsilon_rank"]} channel(s)')
        print(lo.format_parameters(
            lo.physical_parameters(info, triu, NODE_TYPES, TARGET_MODE_IDS)))
        A = lo.drift_matrix(info['G'], info['Upsilon'])
        kap = 2.0 * float(np.imag(info['Upsilon'][0, 1])) or 1.0
        gap = -max(np.real(np.linalg.eigvals(A))) / abs(kap)
        print(f'      dissipative gap = {gap:.6f} kappa')
        sig = info['V_forward'][2:10, 2:10]
        print(f'      achieved purity {purity(sig):.10f}, '
              f'||sigma - target|| = {np.linalg.norm(sig - target):.2e}')
        print(f'      passive (beamsplitter-only)? '
              f'{all(int(v) in (0, 1) for v in triu)}')
        print()
    return out


# Decide the six curated topologies of cayley_cluster_rediscovery.py directly
# in this frame — no scan, the drain is already at r = z.
def decide_curated(V_zv, curated):
    print(f'\n{"-" * 78}')
    print('The six curated topologies of cayley_cluster_rediscovery.py, decided')
    print(f'directly in this frame (drain squeezed, no grid scan needed)\n')
    print(f'  {"topology":42s} {"verdict":9s} {"margin":8s} {"fwd err":9s} {"chan":5s} passive')
    for name, triu in curated.items():
        info = lo.decide(triu, V_zv, TARGET_MODE_IDS, NODE_TYPES, num_samples=64)
        passive = all(int(v) in (0, 1) for v in triu)
        if info['verdict'] == lo.VALID:
            print(f'  {name:42s} {info["verdict"]:9s} '
                  f'{info["stability_margin"]:<8.4f} {info["forward_error"]:<9.1e} '
                  f'{info["upsilon_rank"]:<5d} {passive}')
        else:
            print(f'  {name:42s} {info["verdict"]:9s} {"-":8s} {"-":9s} {"-":5s} {passive}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    z = float(args[0]) if len(args) > 0 else 0.6
    n_starts = int(args[1]) if len(args) > 1 else 40
    do_curated = '--curated' in sys.argv
    sweep_c, budget = None, None
    for a in sys.argv:
        if a.startswith('--sweep='):
            sweep_c = int(a.split('=')[1])
        if a.startswith('--budget='):
            budget = float(a.split('=')[1])

    target = cayley_cluster_target(ADJACENCY_P4, z)
    checks = validate_target(target, ADJACENCY_P4, z)

    print(f'{"=" * 78}')
    print(f'  Cayley cluster on P4 (1-2-3-4), z = {z} — linear oracle, SQUEEZED DRAIN')
    print(f'{"=" * 78}\n')
    print('Target (package convention: xpxp, vacuum = I/2):')
    print(f'  physical                 : {checks["physical"]}')
    print(f'  purity                   : {checks["purity"]:.12f}   '
          '(1 = pure, required by the oracle)')
    print(f'  max |nu_k - 1/2|         : {checks["symplectic_max_dev"]:.3e}')
    print(f'  max |eig - e^(+-2z)/2|   : {checks["eig_max_dev"]:.3e}   -> flat BM, squeezing z')
    print(f'  nullifier variances      : '
          f'{np.array2string(checks["nullifiers"], precision=5)}\n')

    # Step 1 — measure the drain squeezing the target demands.
    print('Step 1 — what drain state does the target need? (oracle decides, cold start)')
    probe = candidate_topologies()['passive: drain->all4 BS + chain BS']
    r_star, _ = detect_drain_squeezing(target, z, probe)

    # Step 2 — build the full covariance with the drain at that squeezing.
    r_use = z
    V_zv = lo.complete_covariance(target, TARGET_MODE_IDS, N_NODES,
                                   aux_squeezing=[(r_use, 0.0)])
    print(f'\nStep 2 — full 5-mode covariance completed with drain squeezed at '
          f'r = {r_use} (= z)')
    print(f'  V shape {V_zv.shape}, drain block eigenvalues '
          f'{np.array2string(np.linalg.eigvalsh(V_zv[:2, :2]), precision=5)}'
          f'  (= e^(-+2z)/2)')

    if do_curated:
        decide_curated(V_zv, candidate_topologies())

    # Step 3 — EXHAUSTIVE sweep by complexity level. Enumerates, so it gives
    # verdict counts and a real minimality bound. This is the expensive half.
    if sweep_c is not None:
        sweep_report(V_zv, target, sweep_c, budget=budget)

    # Step 4 — descent from the lattice maximum. Does not enumerate; it
    # reaches complexities the sweep cannot afford, at the price of claiming
    # only local irreducibility.
    if n_starts > 0:
        print(f'\nStep 4 — multi-start descent from the lattice maximum '
              f'({n_starts} starts), squeezed-drain frame')
        desc = descent_search(V_zv, n_starts=n_starts, num_samples=24)
        print_descent(target, desc, None, V_override=V_zv,
                      header=f'SQUEEZED-DRAIN FRAME (r = {r_use} = z)')
