"""
analytical_test.py
==================
Validates a found topology by constructing the BdG Hamiltonian matrix M
directly from the topology's Heisenberg-Langevin equations, solving:

    M · Γ + Γ · M† + D = 0

then converting Γ (BdG basis) → σ (real quadrature basis) and comparing
to the target covariance matrix.

BdG basis ordering for N modes:
    φ = (a₀, a₁, ..., a_{N-1},  a₀†, a₁†, ..., a_{N-1}†)

Real quadrature ordering:
    q = (x₀, p₀, x₁, p₁, ..., x_{N-1}, p_{N-1})

Conversion: aₖ = (xₖ + i·pₖ)/√2  →  σ = Re(T† · Γ · T)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from reservoir_engineering.targets import squeezed_vacuum, two_mode_squeezed, purity


# ─────────────────────────────────────────────────────────────────────────────
# BdG → real quadrature conversion
# ─────────────────────────────────────────────────────────────────────────────

def bdg_to_real_quadrature(Gamma, N):
    """
    Convert BdG covariance Γ (2N×2N complex) to real quadrature σ (2N×2N real).

    Transformation matrix T (2N×2N):
        T[k,   2k]   =  1/√2        aₖ = (xₖ + i·pₖ)/√2
        T[k,   2k+1] =  i/√2
        T[N+k, 2k]   =  1/√2        aₖ† = (xₖ - i·pₖ)/√2
        T[N+k, 2k+1] = -i/√2

    σ = Re(T† · Γ · T)
    """
    T = np.zeros((2*N, 2*N), dtype=complex)
    for k in range(N):
        T[k,   2*k]   =  1.0 / np.sqrt(2)
        T[k,   2*k+1] =  1j  / np.sqrt(2)
        T[N+k, 2*k]   =  1.0 / np.sqrt(2)
        T[N+k, 2*k+1] = -1j  / np.sqrt(2)
    return np.real(T.conj().T @ Gamma @ T)


def check(name, condition):
    status = '[PASS]' if condition else '[FAIL]'
    print(f'  {status} {name}')
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Topology [0, 2, 1, 0, 0, 0]: cav-mec1:TMS (ν), cav-mec2:BS (g)
# ─────────────────────────────────────────────────────────────────────────────
#
# Hamiltonian: H = ν(a†b₁† + ab₁) + g(a†b₂ + ab₂†)
#
# Heisenberg-Langevin equations:
#   ȧ   = -iν·b₁† - ig·b₂  - κ/2·a
#   ḃ₁  = -iν·a†            - γ/2·b₁
#   ḃ₂  = -ig·a             - γ/2·b₂
#   ȧ†  = +iν·b₁  + ig·b₂† - κ/2·a†
#   ḃ₁† = +iν·a             - γ/2·b₁†
#   ḃ₂† =           ig·a†   - γ/2·b₂†
#
# BdG basis: φ = (a, b₁, b₂, a†, b₁†, b₂†)

def build_bdg_matrix_0_2_1(g_bs, g_tms, kappa, gamma):
    """
    Build the BdG drift matrix M for topology [0, 2, 1, 0, 0, 0].

    Parameters
    ----------
    g_bs  : BS coupling strength (cav-mec2)
    g_tms : TMS coupling strength (cav-mec1)
    kappa : cavity decay rate
    gamma : mechanical decay rate (same for both mechanicals)

    Returns
    -------
    M : np.ndarray (6, 6), complex
    """
    g, nu = g_bs, g_tms
    ka, gm = kappa, gamma

    M = np.array([
        [-ka/2,  0,    -1j*g,   0,      -1j*nu,  0    ],   # ȧ   : couples to b₂(2), b₁†(4)
        [ 0,    -gm/2,  0,     -1j*nu,   0,       0    ],   # ḃ₁  : couples to a†(3)
        [-1j*g,  0,    -gm/2,   0,        0,       0    ],   # ḃ₂  : couples to a(0)
        [ 0,    1j*nu,  0,     -ka/2,    0,       1j*g ],   # ȧ†  : couples to b₁(1), b₂†(5)
        [1j*nu,  0,     0,      0,      -gm/2,    0    ],   # ḃ₁† : couples to a(0)
        [ 0,     0,     0,     1j*g,     0,      -gm/2],   # ḃ₂† : couples to a†(3)
    ], dtype=complex)

    return M


def build_bdg_diffusion_0_2_1(kappa, gamma):
    """
    Diffusion matrix D in BdG basis for topology [0, 2, 1, 0, 0, 0].

    BdG ordering: (a, b₁, b₂, a†, b₁†, b₂†)
    Vacuum baths: D[k,k] = decay_k / 2 per mode.
    """
    ka, gm = kappa, gamma
    return np.diag([ka/2, gm/2, gm/2, ka/2, gm/2, gm/2])


# ─────────────────────────────────────────────────────────────────────────────
# Core validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_0_2_1(log_ratios, lambda_scale, r,
                   kappa=1.0, gamma=4e-5,
                   loss_threshold=5e-4):
    """
    Validate topology [0, 2, 1, 0, 0, 0] using the BdG Hamiltonian approach.

    Parameters
    ----------
    log_ratios    : array-like (2,), [u_tms, u_bs] from optimizer info['log_ratios'].
                    Edge ordering: edge 0 = cav-mec1:TMS, edge 1 = cav-mec2:BS.
    lambda_scale  : float, cooperativity scale (from optimizer info['lambda_scale'])
    r             : float, target squeezing parameter
    kappa, gamma  : decay rates
    loss_threshold: pass/fail threshold on Frobenius loss
    """
    print(f"\n{'='*60}")
    print(f"  BdG validation: [0,2,1,0,0,0]  r={r}  λ={lambda_scale}")
    print(f"  cav-mec1:TMS (ν),  cav-mec2:BS (g)")
    print(f"{'='*60}")

    log_ratios = np.array(log_ratios, dtype=float)
    u_tms, u_bs = log_ratios[0], log_ratios[1]

    # Convert log-ratios → coupling strengths
    # g_k = sqrt(λ · exp(u_k) · d_i · d_j / 4)
    C_tms = lambda_scale * np.exp(u_tms)
    C_bs  = lambda_scale * np.exp(u_bs)
    g_tms = float(np.sqrt(C_tms * kappa * gamma / 4.))
    g_bs  = float(np.sqrt(C_bs  * kappa * gamma / 4.))

    print(f'\n  Coupling strengths:')
    print(f'    TMS (cav-mec1): u={u_tms:+.4f}  C={C_tms:.2f}  ν={g_tms:.6f}')
    print(f'    BS  (cav-mec2): u={u_bs:+.4f}  C={C_bs:.2f}  g={g_bs:.6f}')

    # Build BdG matrix and diffusion matrix
    M = build_bdg_matrix_0_2_1(g_bs, g_tms, kappa, gamma)
    D = build_bdg_diffusion_0_2_1(kappa, gamma)

    # Stability check: all eigenvalues of M must have Re < 0
    eigs = np.linalg.eigvals(M)
    stable = bool(np.all(np.real(eigs) < 0))
    if not check('M is stable (all Re(λ) < 0)', stable):
        return False

    # Solve BdG Lyapunov: M·Γ + Γ·M† + D = 0
    Gamma = solve_continuous_lyapunov(M, -D)

    # Convert BdG covariance → real quadrature covariance
    # N=3 modes: (cav, mec1, mec2)
    # Real quadrature ordering: (x_cav, p_cav, x_m1, p_m1, x_m2, p_m2)
    sigma_real = bdg_to_real_quadrature(Gamma, N=3)

    # Extract signal modes: mec1 (rows/cols 2:4) and mec2 (rows/cols 4:6)
    idx = [2, 3, 4, 5]
    sigma_sub = sigma_real[np.ix_(idx, idx)]

    sigma_target = two_mode_squeezed(r)
    diff = sigma_sub - sigma_target
    loss = float(np.sum(diff**2) / 2.)
    mu   = purity(sigma_sub)

    print(f'\n  sigma_target (mec1, mec2):')
    for row in sigma_target:
        print('    ' + '  '.join(f'{v:+.6f}' for v in row))

    print(f'\n  sigma_achieved (BdG → real quadrature → mec submatrix):')
    for row in sigma_sub:
        print('    ' + '  '.join(f'{v:+.6f}' for v in row))

    print(f'\n  Frobenius loss : {loss:.4e}  (threshold {loss_threshold:.0e})')
    print(f'  Purity         : {mu:.6f}  (ideal = 1.0)')

    passed = check(f'loss < {loss_threshold:.0e}', loss < loss_threshold)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Kronwald [0, 4, 0]  cav-mec1: BS + TMS
# ─────────────────────────────────────────────────────────────────────────────
#
# Hamiltonian: H = g(a†b† + ab) + ν(a†b + ab†)   [BS + TMS on same pair]
# Wait - this is the single-mode squeezing topology for Kronwald.
# For Kronwald: 2 modes (cav=a, mec=b), BdG basis (a, b, a†, b†)
#
# H = g_bs(a†b + ab†) + g_tms(a†b† + ab)
# ȧ   = -ig_bs·b - ig_tms·b† - κ/2·a
# ḃ   = -ig_bs·a - ig_tms·a† - γ/2·b
# ȧ†  = +ig_bs·b† + ig_tms·b - κ/2·a†
# ḃ†  = +ig_bs·a† + ig_tms·a - γ/2·b†

def build_bdg_matrix_kronwald(g_bs, g_tms, kappa, gamma):
    """
    BdG drift matrix M for Kronwald topology [0, 4, 0].
    BdG basis: φ = (a, b, a†, b†)
    """
    g, nu = g_bs, g_tms
    ka, gm = kappa, gamma

    M = np.array([
        [-ka/2,  -1j*g,    0,      -1j*nu],   # ȧ
        [-1j*g,  -gm/2,  -1j*nu,    0    ],   # ḃ
        [ 0,     1j*nu,  -ka/2,    1j*g  ],   # ȧ†
        [1j*nu,   0,      1j*g,   -gm/2  ],   # ḃ†
    ], dtype=complex)

    return M


def validate_kronwald(log_ratios, lambda_scale, r,
                      kappa=1.0, gamma=0.01,
                      loss_threshold=2e-3):
    """
    Validate Kronwald topology [0, 4, 0] using the BdG Hamiltonian approach.

    log_ratios: [u_bs, u_tms] from optimizer info['log_ratios'].
    Analytic: u_bs=0 (ref), u_tms = log(tanh²(r)).
    """
    print(f"\n{'='*60}")
    print(f"  BdG validation: Kronwald [0,4,0]  r={r}  λ={lambda_scale}")
    print(f"  cav-mec: BS (g) + TMS (ν)")
    print(f"{'='*60}")

    log_ratios = np.array(log_ratios, dtype=float)
    u_bs, u_tms = log_ratios[0], log_ratios[1]

    C_bs  = lambda_scale * np.exp(u_bs)
    C_tms = lambda_scale * np.exp(u_tms)
    g_bs  = float(np.sqrt(C_bs  * kappa * gamma / 4.))
    g_tms = float(np.sqrt(C_tms * kappa * gamma / 4.))

    print(f'\n  Coupling strengths:')
    print(f'    BS  (cav-mec): u={u_bs:+.4f}  C={C_bs:.2f}  g={g_bs:.6f}')
    print(f'    TMS (cav-mec): u={u_tms:+.4f}  C={C_tms:.2f}  ν={g_tms:.6f}')

    M = build_bdg_matrix_kronwald(g_bs, g_tms, kappa, gamma)
    D = np.diag([kappa/2, gamma/2, kappa/2, gamma/2])

    eigs = np.linalg.eigvals(M)
    stable = bool(np.all(np.real(eigs) < 0))
    if not check('M is stable (all Re(λ) < 0)', stable):
        return False

    Gamma      = solve_continuous_lyapunov(M, -D)
    sigma_real = bdg_to_real_quadrature(Gamma, N=2)

    # Signal mode: mec (mode 1) → real quadrature rows/cols 2:4
    sigma_sub    = sigma_real[2:4, 2:4]
    sigma_target = squeezed_vacuum(r)

    diff = sigma_sub - sigma_target
    loss = float(np.sum(diff**2) / 2.)
    mu   = purity(sigma_sub)

    print(f'\n  sigma_target (mec):')
    for row in sigma_target:
        print('    ' + '  '.join(f'{v:+.6f}' for v in row))

    print(f'\n  sigma_achieved (BdG → real quadrature → mec submatrix):')
    for row in sigma_sub:
        print('    ' + '  '.join(f'{v:+.6f}' for v in row))

    print(f'\n  Frobenius loss : {loss:.4e}  (threshold {loss_threshold:.0e})')
    print(f'  Purity         : {mu:.6f}  (ideal = 1.0)')

    passed = check(f'loss < {loss_threshold:.0e}', loss < loss_threshold)
    return passed


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Kronwald: analytic ratio tanh²(r)
    r = 1.0
    validate_kronwald(
        log_ratios   = [0.0, np.log(np.tanh(r)**2)],
        lambda_scale = 1000.,
        r            = r,
    )

    # EPR [0,2,1,0,0,0]: cav-mec1:TMS, cav-mec2:BS
    # log_ratios from optimizer: edge 0 = TMS (ref, u=0), edge 1 = BS (u=4.66)
    validate_0_2_1(
        log_ratios   = [0.0, 4.663397887287683],
        lambda_scale = 1000.,
        r            = 0.1,
        gamma        = 4e-5,
    )
