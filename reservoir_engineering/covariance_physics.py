"""
covariance_physics.py
=====================
Low-level physics engine for open quantum systems in the real quadrature
basis. Everything here is JAX-differentiable w.r.t. coupling strengths.

Quadrature basis: q = (x_0,p_0,...,x_{N-1},p_{N-1}), mode i at rows/cols
2i,2i+1. Vacuum noise = 1/2 per quadrature ([x,p]=i).

Dynamics: dq/dt = A q + noise, steady state solves the Lyapunov equation
    A sigma + sigma A^T + D = 0
A is the drift matrix (Heisenberg-Langevin EOM), D is the (constant,
diagonal) diffusion matrix from the bath noise. A must be Hurwitz (all
eigenvalues Re<0) for a steady state to exist.

A = K + A_decay, where K is built from the coherent Hamiltonian (couplings +
detunings) and A_decay from the bath coupling (-decay_i/2 per mode). This
mirrors AutoScatter's A = -iH - kappa/2 (Hamiltonian + decay, kept separate
then combined).

Node dict: {'id', 'type': 'cavity'|'mechanical', 'kappa'|'gamma', 'n_th',
'delta'}. 'delta' is a detuning (rotating-frame residual, default 0).
Edge dict: {'i','j','type': 'beamsplitter'|'two_mode_squeezing'|'parametric'}.
  beamsplitter (H=g(a-dag b + a b-dag)): energy-conserving swap.
  two_mode_squeezing (H=nu(a-dag b-dag + a b)): squeezing/entanglement.
  parametric (H=chi(a^2+a-dag^2), i==j only): single-mode squeezing.

Kronwald validation case: 1 cavity + 1 mechanical, edges=[BS(g),TMS(nu)].
Should give mechanical squeezing r=atanh(nu/g) as gamma->0.

Alignment with gaussian_autoscatter_algorithm.md §1.2 (the (G,C) machine):
the doc writes A = Omega(G + Im(C^dag C)), D = 2 Omega Re(C^dag C) Omega^T
for a real Hamiltonian matrix G and complex jump-operator matrix C.
`build_hamiltonian_matrix` returns K = Omega G directly (the drift
generator, like AutoScatter's -iH rather than H itself); `build_jump_matrix`
builds C from each node's bath parameters. `build_drift_matrix` /
`build_diffusion_matrix` are thin wrappers around the general
`build_drift_matrix_from_GC` / `build_diffusion_matrix_from_C`, numerically
identical to the old per-node decay-block code (checked in
tests/test_covariance_physics.py).

Convention note: this package uses vacuum=1/2 (not the doc's vacuum=1), so
the diffusion formula drops the doc's factor of 2:
    A = K + Omega Im(C^dag C)      D = Omega Re(C^dag C) Omega^T
Derived by hand from L = sqrt(kappa) a = sqrt(kappa/2)(x+ip); keeps every
downstream formula (purity, squeezed_vacuum, Kronwald r=atanh(nu/g), ...)
unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict

jax.config.update("jax_enable_x64", True)


# quadrature_slice(mode_id): mode i -> indices 2i, 2i+1.
def quadrature_slice(mode_id: int):
    return slice(2 * mode_id, 2 * mode_id + 2)


# symplectic_form(N): Omega = block-diag([[0,1],[-1,0]]), per-mode ordering.
def symplectic_form(N: int) -> jnp.ndarray:
    Omega = jnp.zeros((2 * N, 2 * N))
    for i in range(N):
        Omega = Omega.at[2 * i, 2 * i + 1].set(1.)
        Omega = Omega.at[2 * i + 1, 2 * i].set(-1.)
    return Omega


# build_jump_matrix(nodes) -> C, the complex jump-operator matrix (§1.2).
# One row per bath channel:
#   cavity i   (vacuum bath, rate kappa):      c = sqrt(kappa/2) [1, i]
#   cavity i   (SQUEEZED bath, optional — node['squeeze_a']/['squeeze_b']
#               set, i.e. a Bogoliubov dissipator L=sqrt(kappa)(cosh(r)a +
#               M a-dag), M=a_+i*b_=sinh(r)e^{i theta} reparametrised as
#               unconstrained (a_,b_) so cosh(r)=sqrt(1+a_^2+b_^2) needs no
#               separate bound/periodicity handling — see
#               build_drift_diffusion_from_GC_tilde's squeezable_aux_ids):
#               c = sqrt(kappa/2) [cosh(r)+a_+i*b_,  i*(cosh(r)-a_)+b_]
#               (a_=b_=0 reduces exactly to the plain-vacuum row above)
#   mechanical i (thermal bath, rate gamma, n_th): two rows,
#       loss: sqrt(gamma(n_th+1)/2) [1, i]   gain: sqrt(gamma n_th/2) [1,-i]
# Normalisations chosen (by hand) so A, D below reproduce exactly the old
# -decay_i/2 and decay*(n_th+1/2) formulas.
#
# JAX-differentiable w.r.t. any node['kappa']/['gamma']/['squeeze_a']/
# ['squeeze_b'] that is itself a jnp value (not just a fixed python float)
# — needed when an auxiliary mode's decay rate or bath squeezing is a free
# search variable (see build_drift_diffusion_from_GC_tilde). Built with
# jnp.zeros().at[].set(), not numpy, for exactly that reason.
def build_jump_matrix(nodes: List[Dict]) -> jnp.ndarray:
    N = len(nodes)
    rows = []
    for i, node in enumerate(nodes):
        if node['type'] == 'cavity':
            kappa = node['kappa']
            a_ = node.get('squeeze_a', 0.0)
            b_ = node.get('squeeze_b', 0.0)
            cosh_r = jnp.sqrt(1. + a_ ** 2 + b_ ** 2)
            coeff_x = jnp.sqrt(kappa / 2.) * ((cosh_r + a_) + 1j * b_)
            coeff_p = jnp.sqrt(kappa / 2.) * (b_ + 1j * (cosh_r - a_))
            row = jnp.zeros(2 * N, dtype=complex)
            row = row.at[2 * i].set(coeff_x)
            row = row.at[2 * i + 1].set(coeff_p)
            rows.append(row)
        else:
            gamma = node['gamma']
            n_th  = node['n_th']

            row_loss = jnp.zeros(2 * N, dtype=complex)
            r_loss = gamma * (n_th + 1.)
            row_loss = row_loss.at[2 * i].set(jnp.sqrt(r_loss / 2.))
            row_loss = row_loss.at[2 * i + 1].set(1j * jnp.sqrt(r_loss / 2.))
            rows.append(row_loss)

            row_gain = jnp.zeros(2 * N, dtype=complex)
            r_gain = gamma * n_th
            row_gain = row_gain.at[2 * i].set(jnp.sqrt(r_gain / 2.))
            row_gain = row_gain.at[2 * i + 1].set(-1j * jnp.sqrt(r_gain / 2.))
            rows.append(row_gain)

    if not rows:
        return jnp.zeros((0, 2 * N), dtype=complex)
    return jnp.stack(rows)


# A = K + Omega Im(C^dag C),  D = Omega Re(C^dag C) Omega^T  (§1.2, 1/2-conv).
# K is the drift generator (build_hamiltonian_matrix's output = Omega*G).
def build_drift_matrix_from_GC(K: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    N = K.shape[0] // 2
    Omega = symplectic_form(N)
    CdC = jnp.conj(C).T @ C
    return K + Omega @ jnp.imag(CdC)


def build_diffusion_matrix_from_C(C: jnp.ndarray) -> jnp.ndarray:
    N = C.shape[1] // 2
    Omega = symplectic_form(N)
    CdC = jnp.conj(C).T @ C
    return Omega @ jnp.real(CdC) @ Omega.T


# build_hamiltonian_matrix(nodes, edges, coupling_strengths) -> K
# The drift generator from the coherent Hamiltonian only (no decay):
#   detuning delta_i:        K[s_i,s_i]   += delta_i * J2
#   beamsplitter J(i,j):     K[s_i,s_j]   += Re(J)*J2 + Im(J)*I2
#                            K[s_j,s_i]   += Re(J)*J2 - Im(J)*I2
#   two_mode_squeezing g(i,j): K[s_i,s_j] += -g*sx,  K[s_j,s_i] += -g*sx
#   parametric chi(i,i):     K[s_i,s_i]   += chi*sz
# where J2=[[0,1],[-1,0]], I2=eye(2), sx=[[0,1],[1,0]], sz=[[1,0],[0,-1]].
#
# Beamsplitter couplings carry a genuine PHASE (coupling_strengths[k] may be
# complex: J=g*exp(i*theta)) — derived from H=J b_j^dag b_k + h.c. via the
# Heisenberg EOM (see covariance_optimizer's module docstring, and Zippilli
# & Vitali PRL 126, 020402 (2021) eq. S.27-S.29: their Lemma shows the
# passivity-preserving phase Θ_jk is generically nonzero, fixed by the
# squeezing phases of whatever's coupled to the edge — a REAL-only edge
# (Θ=0, the special case below) is a restriction the package used to bake
# in unconditionally; it's now the default-off case, opt in per-edge via
# constraints.Constraint_real_coupling). At theta=0 (plain real g), this
# reduces EXACTLY to the old symmetric g*J2 block on both sides — backward
# compatible with any caller still passing real coupling_strengths.
# two_mode_squeezing/parametric stay real-only for now (jnp.real(...) —
# phase generalisation not yet implemented for those edge types).
def build_hamiltonian_matrix(
    nodes: List[Dict],
    edges: List[Dict],
    coupling_strengths: jnp.ndarray,) -> jnp.ndarray:

    N  = len(nodes)
    H  = jnp.zeros((2 * N, 2 * N))
    J2 = jnp.array([[0.,  1.], [-1., 0.]])
    I2 = jnp.eye(2)
    sx = jnp.array([[0.,  1.], [ 1., 0.]])
    sz = jnp.array([[1.,  0.], [ 0., -1.]])

    for i, node in enumerate(nodes):
        s     = quadrature_slice(i)
        delta = node.get('delta', 0.0)
        H     = H.at[s, s].add(delta * J2)

    for k, edge in enumerate(edges):
        si = quadrature_slice(edge['i'])
        sj = quadrature_slice(edge['j'])
        g  = coupling_strengths[k]

        if edge['type'] == 'beamsplitter':
            a_ = jnp.real(g)
            b_ = jnp.imag(g)
            H = H.at[si, sj].add(a_ * J2 + b_ * I2)
            H = H.at[sj, si].add(a_ * J2 - b_ * I2)

        elif edge['type'] == 'two_mode_squeezing':
            g = jnp.real(g)
            H = H.at[si, sj].add(-g * sx)
            H = H.at[sj, si].add(-g * sx)

        elif edge['type'] == 'parametric':
            H = H.at[si, si].add(jnp.real(g) * sz)

    return H


# build_drift_matrix(nodes, edges, coupling_strengths) -> A = K + Omega Im(C^dag C)
def build_drift_matrix(
    nodes: List[Dict],
    edges: List[Dict],
    coupling_strengths: jnp.ndarray,
) -> jnp.ndarray:
    K = build_hamiltonian_matrix(nodes=nodes, edges=edges, coupling_strengths=coupling_strengths)
    C = build_jump_matrix(nodes)
    return build_drift_matrix_from_GC(K, C)


# build_diffusion_matrix(nodes) -> D = Omega Re(C^dag C) Omega^T
def build_diffusion_matrix(nodes: List[Dict]) -> jnp.ndarray:
    C = build_jump_matrix(nodes)
    return build_diffusion_matrix_from_C(C)


# solve_lyapunov_kronecker(A, D): solve A sigma + sigma A^T + D = 0 via
# vec(sigma) = -(I⊗A + A⊗I)^-1 vec(D). M is singular exactly at stability
# boundaries (A has eigenvalues summing to 0) — ensure A is Hurwitz first.
def solve_lyapunov_kronecker(A: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    n = A.shape[0]
    I = jnp.eye(n)
    M = jnp.kron(I, A) + jnp.kron(A, I)
    vec_sigma = jnp.linalg.solve(M, -D.flatten())
    return vec_sigma.reshape(n, n)


# check_stability(A): True iff A is Hurwitz (all eigenvalues Re<0). Cheap
# discrete filter (Stage 1) run before any gradient optimisation — an
# unstable topology at unit cooperativity is structurally broken (wrong
# edge types/graph shape), not fixable by tuning coupling magnitudes.
def check_stability(A, tolerance: float = 1e-10) -> bool:
    eigs = np.linalg.eigvals(np.asarray(A))
    return bool(np.all(np.real(eigs) < tolerance))


# get_mode_covariance(sigma, mode_ids): extract the 2M x 2M block for the
# given modes, e.g. to compare only the signal modes against the target.
def get_mode_covariance(
    sigma: jnp.ndarray,
    mode_ids: List[int],
) -> jnp.ndarray:
    idx = []
    for m in mode_ids:
        idx += [2 * m, 2 * m + 1]
    return sigma[np.ix_(idx, idx)]


# ───────────────────────────────────────────────────────────────────────────
# build_drift_diffusion_from_GC_tilde: (G-tilde, C-tilde) parametrisation
# ───────────────────────────────────────────────────────────────────────────
# §1.5's literal dimensionless reduction: G_tilde = G/kappa0, C_tilde = C/sqrt(kappa0),
# optimised DIRECTLY — no per-edge decay_i*decay_j reconstruction, no shared
# fixed-large lambda. Works identically whether a node's decay rate is 0 or
# not: decay never appears anywhere in the coherent-coupling formula, and
# only appears for auxiliary nodes as the thing being optimised (via
# aux_log_rates), not something coupling strengths are divided/multiplied by.
#
# Parameters:
#   nodes            — list of N node dicts. Auxiliary nodes' 'kappa'/'gamma'
#                       entries are IGNORED (overwritten below); other nodes'
#                       (signal modes) decay rates are used as given — may be
#                       exactly 0 (the idealized gamma->0 limit, §1.6).
#   edges            — list of E edge dicts (types only)
#   coherent_log_ratios — jnp.ndarray (E,), u_k = log(G_tilde_k); one per
#                       coherent edge (BS/TMS/parametric/detuning), no
#                       reference-edge pinning needed (kappa0 already fixes
#                       the overall-scale gauge, so every edge is free —
#                       see module docstring for why the old reference-edge
#                       trick isn't a gauge necessity here).
#   aux_log_rates    — jnp.ndarray (A,), v_m = log(C_tilde_m); one per
#                       auxiliary node in aux_node_ids (rate only — the jump-
#                       operator FORM stays whatever the node's 'type' says,
#                       §2.6(b) domain choice, not searched here).
#   aux_node_ids     — list of int, which node indices are auxiliary (their
#                       decay rate is aux_log_rates-controlled, not fixed).
#   kappa0           — float, the single reference rate (§1.5); default 1.0.
#   squeezable_aux_ids — list of int (subset of aux_node_ids, cavity type
#                       only), the human-declared palette of which drains
#                       MAY have a squeezed (Bogoliubov) bath instead of
#                       plain vacuum — see build_jump_matrix. The optimizer,
#                       not the human, decides HOW squeezed: squeeze_ab
#                       below is a free search variable per node, and
#                       (a_,b_)=(0,0) is reachable, so "plain vacuum" stays
#                       in the search space too (no separate discrete
#                       vacuum-vs-squeezed choice needed).
#   squeeze_ab       — jnp.ndarray (S,2), (a_,b_) per node in
#                       squeezable_aux_ids, in the same order. Required iff
#                       squeezable_aux_ids is non-empty.
#   coherent_phases  — jnp.ndarray (E,), theta_k per coherent edge (only
#                       meaningful for beamsplitter edges — see
#                       build_hamiltonian_matrix's phase generalisation;
#                       ignored for other edge types). Default all-zero =
#                       old real-only behaviour. Entries for edges the user
#                       has restricted via constraints.Constraint_real_coupling
#                       should be pinned to 0 by the caller (this function
#                       itself doesn't know about constraints — see
#                       covariance_optimizer.CovarianceOptimizer, which does).
#
# g_k = kappa0 * exp(u_k) * exp(i*theta_k)   (coherent coupling strengths;
#                                              complex iff theta_k != 0)
# decay_m = kappa0 * exp(v_m)          (auxiliary nodes' kappa, in place of
#                                        the fixed value in nodes[m])
#
# Returns (A, D) — both fully JAX-differentiable w.r.t. coherent_log_ratios,
# coherent_phases, aux_log_rates, AND squeeze_ab (build_jump_matrix above
# was made jnp-native for exactly this: D and the decay/squeezing part of A
# must backprop through these free variables).

def build_drift_diffusion_from_GC_tilde(
    nodes: List[Dict],
    edges: List[Dict],
    coherent_log_ratios: jnp.ndarray,
    aux_log_rates: jnp.ndarray,
    aux_node_ids: List[int],
    kappa0: float = 1.0,
    squeezable_aux_ids: List[int] = None,
    squeeze_ab: jnp.ndarray = None,
    coherent_phases: jnp.ndarray = None,
):
    if coherent_phases is None:
        coupling_strengths = kappa0 * jnp.exp(coherent_log_ratios)
    else:
        coupling_strengths = kappa0 * jnp.exp(coherent_log_ratios) * jnp.exp(1j * coherent_phases)

    aux_decay = kappa0 * jnp.exp(aux_log_rates)
    nodes_eff = list(nodes)
    for slot, node_id in enumerate(aux_node_ids):
        node = dict(nodes_eff[node_id])
        if node['type'] == 'cavity':
            node['kappa'] = aux_decay[slot]
        else:
            node['gamma'] = aux_decay[slot]
        nodes_eff[node_id] = node

    for slot, node_id in enumerate(squeezable_aux_ids or []):
        node = dict(nodes_eff[node_id])
        node['squeeze_a'] = squeeze_ab[slot, 0]
        node['squeeze_b'] = squeeze_ab[slot, 1]
        nodes_eff[node_id] = node

    K = build_hamiltonian_matrix(nodes_eff, edges, coupling_strengths)
    C = build_jump_matrix(nodes_eff)
    A = build_drift_matrix_from_GC(K, C)
    D = build_diffusion_matrix_from_C(C)
    return A, D


# covariance_loss: 1/2 ||sigma_sub - target||_F^2, JAX-differentiable in
# coupling_strengths. D is passed in precomputed (doesn't depend on couplings).
def covariance_loss(
    coupling_strengths: jnp.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    target_cov: jnp.ndarray,
    target_mode_ids: List[int],
) -> jnp.ndarray:
    A = build_drift_matrix(nodes=nodes, edges=edges, coupling_strengths=coupling_strengths)
    D = build_diffusion_matrix(nodes)
    sigma = solve_lyapunov_kronecker(A, D)
    sigma_sub = get_mode_covariance(sigma, target_mode_ids)
    diff = sigma_sub - target_cov
    return jnp.sum(diff ** 2) / 2


# purity_violation(sigma_sub): (det(2 sigma_sub) - 1)^2 — zero for an
# exactly pure state, in this package's 1/2-convention (pure <=> det(2sigma)=1).
# Optional §3 loss term: add lambda_pure * purity_violation(sigma_sub) when
# an exactly pure target matters, not just a high-fidelity one.
def purity_violation(sigma_sub: jnp.ndarray) -> jnp.ndarray:
    return (jnp.linalg.det(2 * sigma_sub) - 1.) ** 2


# ───────────────────────────────────────────────────────────────────────────
# §6 target-quantity reference — JAX-differentiable versions
# ───────────────────────────────────────────────────────────────────────────
# targets.py has numpy versions of these (for post-hoc reporting on a
# concrete achieved sigma). These jnp versions are for use INSIDE a §3 loss
# (targets.py's TargetTerm), where sigma is a JAX tracer under jax.grad —
# np.linalg.eigvals etc. would not differentiate through it.

# jnp version of targets.symplectic_eigenvalues.
def jnp_symplectic_eigenvalues(sigma: jnp.ndarray) -> jnp.ndarray:
    N = sigma.shape[0] // 2
    Omega = symplectic_form(N)
    M = 1j * Omega @ sigma
    eigs = jnp.linalg.eigvals(M)
    nus = jnp.sort(jnp.abs(jnp.real(eigs)))
    return nus[::2]


# jnp version of targets.purity: mu = 1/sqrt(det(2 sigma)).
def jnp_purity(sigma: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / jnp.sqrt(jnp.linalg.det(2 * sigma))


# jnp version of targets.log_negativity, 2-mode (4x4) sigma only.
def jnp_log_negativity(sigma: jnp.ndarray) -> jnp.ndarray:
    T = jnp.diag(jnp.array([1., 1., 1., -1.]))
    sigma_pt = T @ sigma @ T
    nus = jnp_symplectic_eigenvalues(sigma_pt)
    nu_min = jnp.min(nus)
    return jnp.maximum(0., -jnp.log2(2 * nu_min))
