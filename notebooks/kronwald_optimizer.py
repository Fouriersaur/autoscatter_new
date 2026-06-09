"""
kronwald_optimizer.py

Steady-state covariance matrix optimiser for the Kronwald dissipative
squeezing scheme:

    Kronwald & Marquardt, "Arbitrarily Large Steady-State Bosonic
    Squeezing via Dissipation", PRL 111, 133601 (2013).

Physics
-------
An optomechanical cavity (decay κ) is driven by TWO tones:
  • red  sideband at ω_c − ω_m  →  beamsplitter coupling g
  • blue sideband at ω_c + ω_m  →  squeezing    coupling ν

In the DOUBLY-ROTATING FRAME (cavity at ω_c AND mechanics at ω_m) the
effective Hamiltonian becomes time-independent:

    H = g(a†b + ab†) + ν(a†b† + ab)

(no ω_m term — it has been rotated away).  This is the BAD-CAVITY limit
(κ ≫ ω_m), where the cavity can be adiabatically eliminated to give an
effective squeezed Lindblad dissipator for the mechanics.

The steady-state mechanical variance predicted by this model is:

    σ_xx = (g − ν) / (2(g + ν))   [x quadrature, squeezed below ½]
    σ_pp = (g + ν) / (2(g − ν))   [p quadrature, anti-squeezed]

giving squeezing parameter  r = atanh(ν/g)  for γ → 0.
(Only the ratio ν/g matters — not the absolute scale of g.)

Implementation
--------------
We work in the REAL QUADRATURE BASIS

    q = (x_cav, p_cav, x_mech, p_mech)ᵀ

All matrices are real; the Lyapunov equation is

    A · σ + σ · Aᵀ + D = 0

with drift matrix A and diffusion matrix D below.
The mechanical covariance is the bottom-right 2×2 block σ[2:, 2:].
The Lyapunov equation is solved via the Kronecker-product trick
  (I⊗A + A⊗I) vec(σ) = −vec(D)
which is a plain real linear system and fully JAX-differentiable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Physical model — real quadrature basis (x_cav, p_cav, x_mech, p_mech)
# ---------------------------------------------------------------------------

def drift_matrix(g, nu, kappa=1.0, gamma=0.01):
    """
    4×4 REAL drift matrix A.

    Frame: doubly rotating (at ω_c for cavity, ω_m for mechanics) so that
    both coupling terms are DC.  This is the bad-cavity (κ ≫ ω_m) limit.

    Coupling structure:
      dx_cav/dt  includes (g − ν) p_mech
      dp_cav/dt  includes −(g + ν) x_mech
      dx_mech/dt includes (g − ν) p_cav
      dp_mech/dt includes −(g + ν) x_cav

    Stability requires  g > ν > 0.
    """
    return jnp.array([
        [-kappa/2,     0.,          0.,          g - nu   ],
        [ 0.,         -kappa/2,   -(g + nu),     0.       ],
        [ 0.,          g - nu,    -gamma/2,       0.       ],
        [-(g + nu),    0.,         0.,           -gamma/2  ],
    ])


def diffusion_matrix(kappa=1.0, gamma=0.01, n_th=0.0):
    """
    4×4 REAL diffusion matrix D.

    Optical bath: zero-temperature vacuum  →  κ/2 per quadrature
    Mechanical bath: thermal phonons n̄_th →  γ(n̄ + ½) per quadrature
    """
    return jnp.diag(jnp.array([
        kappa / 2.,
        kappa / 2.,
        gamma * (n_th + 0.5),
        gamma * (n_th + 0.5),
    ]))


def solve_lyapunov(g, nu, kappa=1.0, gamma=0.01, n_th=0.0):
    """
    Solve  A σ + σ Aᵀ + D = 0  for the 4×4 covariance matrix σ.

    Fully JAX-differentiable via the vectorisation identity
        (I⊗A + A⊗I) vec(σ) = −vec(D)   (16×16 real linear system).
    """
    A = drift_matrix(g, nu, kappa, gamma)
    D = diffusion_matrix(kappa, gamma, n_th)
    n = A.shape[0]
    I = jnp.eye(n)
    M = jnp.kron(I, A) + jnp.kron(A, I)
    vec_sigma = jnp.linalg.solve(M, -D.flatten())
    return vec_sigma.reshape(n, n)


def mechanical_cov(sigma):
    """2×2 mechanical (x, p) covariance — bottom-right block of σ."""
    return sigma[2:, 2:]


# ---------------------------------------------------------------------------
# Target covariance and squeezing helpers
# ---------------------------------------------------------------------------

def squeezed_vacuum_cov(r: float) -> np.ndarray:
    """
    Target covariance for a single-mode squeezed vacuum with parameter r.

        σ_target = ½ · diag(e^{−2r}, e^{+2r})     (x squeezed)
    """
    return 0.5 * np.diag([np.exp(-2. * r), np.exp(+2. * r)])


def squeezing_from_cov(sigma_mech: np.ndarray) -> float:
    """
    Squeezing parameter r from a 2×2 mechanical covariance matrix.

    For a pure squeezed vacuum  σ = ½ diag(e^{−2r}, e^{+2r}):
        r = −½ log(2 σ_xx) = ½ log(1 / (2 σ_xx))
    """
    return float(-0.5 * np.log(2.0 * sigma_mech[0, 0]))


# ---------------------------------------------------------------------------
# Analytical Kronwald prediction
# ---------------------------------------------------------------------------

def kronwald_squeezing(g: float, nu: float) -> float:
    """
    Analytical squeezing r = atanh(ν/g) from the Kronwald formula
    (bad-cavity limit, γ → 0, T → 0).
    """
    return float(np.arctanh(nu / g)) if g > nu > 0 else float('inf')


def kronwald_params_from_r(r: float, g: float = 1.0):
    """
    Given target squeezing r, return the coupling strengths (g, ν) using
    the Kronwald formula  ν = g · tanh(r).
    """
    return g, g * float(np.tanh(r))


# ---------------------------------------------------------------------------
# Loss and gradient
# ---------------------------------------------------------------------------

def make_loss(sigma_target, kappa=1.0, gamma=0.01, n_th=0.0):
    """
    Return a JIT-compiled loss function and its gradient.

        loss(params) = ½ ‖ σ_mech(g, ν) − σ_target ‖²_F

    params is a 2-element JAX array [g, ν].
    """
    st = jnp.array(sigma_target)

    @jax.jit
    def loss(params):
        g, nu = params[0], params[1]
        sigma = solve_lyapunov(g, nu, kappa, gamma, n_th)
        diff  = mechanical_cov(sigma) - st
        return jnp.sum(diff ** 2) / 2.

    return loss, jax.jit(jax.grad(loss))


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def optimize(
    target_r: float,
    kappa: float = 1.0,
    gamma: float = 0.01,
    n_th: float = 0.0,
    num_restarts: int = 20,
    maxiter: int = 2000,
    seed: int = 0,
):
    """
    Find (g, ν) that produce mechanical squeezing r in steady state.

    Works in the bad-cavity limit (κ ≫ ω_m, doubly rotating frame).
    For the ideal case (γ → 0, T = 0), only the ratio ν/g = tanh(r)
    matters; g sets the timescale for approaching steady state.

    Parameters
    ----------
    target_r     : desired squeezing parameter r  (x-variance = ½ e^{−2r})
    kappa        : cavity decay rate  (sets units, default 1)
    gamma        : mechanical decay rate  (should be ≪ κ)
    n_th         : mean thermal phonon number of mechanical bath
    num_restarts : random restarts to escape local minima
    maxiter      : BFGS iteration limit per restart
    seed         : RNG seed for reproducibility

    Returns
    -------
    g_opt, nu_opt : float — optimal coupling strengths in units of κ
    info          : dict  — diagnostics
    """
    rng = np.random.default_rng(seed)
    sigma_target = squeezed_vacuum_cov(target_r)
    loss_fn, _ = make_loss(sigma_target, kappa, gamma, n_th)

    # Reparametrise: q = [log g, log(g − ν)]
    #   g  = exp(q[0])               always > 0
    #   ν  = exp(q[0]) − exp(q[1])   always 0 < ν < g
    @jax.jit
    def loss_q(q):
        g  = jnp.exp(q[0])
        nu = jnp.exp(q[0]) - jnp.exp(q[1])
        return loss_fn(jnp.array([g, nu]))

    grad_q = jax.jit(jax.grad(loss_q))

    def val(q):  return float(loss_q(jnp.array(q, dtype=jnp.float64)))
    def grd(q):  return np.array(grad_q(jnp.array(q, dtype=jnp.float64)))

    # Bound log(g) so the physical g stays in [g_lo, g_hi].
    # The squeezing only depends on ν/g so any g in this range is fine.
    g_lo, g_hi = 0.05 * kappa, 3.0 * kappa
    q_lo = np.array([np.log(g_lo), -np.inf])      # lower bound on q
    q_hi = np.array([np.log(g_hi),  np.inf])       # upper bound on q (unbounded g-ν)

    best = None
    for _ in range(num_restarts):
        g0     = rng.uniform(g_lo, g_hi)
        ratio0 = np.tanh(target_r) * rng.uniform(0.7, 0.99)
        nu0    = g0 * ratio0
        q0     = np.array([np.log(g0), np.log(g0 - nu0)], dtype=np.float64)

        res = sciopt.minimize(val, q0, jac=grd, method='L-BFGS-B',
                              bounds=list(zip(q_lo, q_hi)),
                              options={'maxiter': maxiter, 'ftol': 0., 'gtol': 1e-12})
        if best is None or res.fun < best.fun:
            best = res

    g_opt  = float(np.exp(best.x[0]))
    nu_opt = float(np.exp(best.x[0]) - np.exp(best.x[1]))

    sigma_full = np.array(solve_lyapunov(g_opt, nu_opt, kappa, gamma, n_th))
    sm_ach     = np.array(mechanical_cov(jnp.array(sigma_full)))

    r_ach  = squeezing_from_cov(sm_ach)
    r_kron = kronwald_squeezing(g_opt, nu_opt)

    return g_opt, nu_opt, {
        'g':                   g_opt,
        'nu':                  nu_opt,
        'nu_over_g':           nu_opt / g_opt,
        'tanh_r_target':       float(np.tanh(target_r)),
        'loss':                float(best.fun),
        'sigma_mech_achieved': sm_ach,
        'sigma_mech_target':   sigma_target,
        'r_target':            target_r,
        'r_achieved':          r_ach,
        'r_Kronwald_analytic': r_kron,
        'optimizer_msg':       best.message,
    }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    target_r = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    gamma    = 0.001   # γ/κ ≪ 1 for clean squeezing

    print(f'Kronwald optimiser  |  target r = {target_r}')
    print(f'Parameters: κ=1, γ={gamma}, bad-cavity limit (doubly-rotating frame)')
    print(f'Target x-variance = {0.5*np.exp(-2*target_r):.6f}  '
          f'(vacuum = 0.50000)\n')

    g_opt, nu_opt, info = optimize(
        target_r, gamma=gamma,
        num_restarts=20, maxiter=2000, seed=42,
    )

    print('Optimal parameters (units of κ):')
    print(f'  g  = {g_opt:.5f}')
    print(f'  ν  = {nu_opt:.5f}')
    print(f'  ν/g = {info["nu_over_g"]:.6f}')
    print(f'  tanh(r) = {info["tanh_r_target"]:.6f}  '
          f'← should match ν/g for γ→0\n')

    print(f'Achieved σ_mech (x, p quadratures):')
    print(info['sigma_mech_achieved'])
    print(f'\nTarget σ_mech:')
    print(info['sigma_mech_target'])

    print(f'\nr achieved        = {info["r_achieved"]:.5f}')
    print(f'r target          = {target_r:.5f}')
    print(f'r Kronwald (γ→0) = {info["r_Kronwald_analytic"]:.5f}')
    print(f'Residual loss: {info["loss"]:.2e}')
    print(f'Optimiser: {info["optimizer_msg"]}')

    # Quick scan: show squeezing vs nu/g ratio (fixed g=1)
    print(f'\nSqueezing scan (g=1, γ={gamma}, bad-cavity):')
    print(f'  {"ν/g":>8} {"tanh→r":>8} {"σ_xx":>10} {"r_num":>8} {"r_kron":>8}')
    g_scan = 1.0
    for r_scan in [0.5, 1.0, 1.5, 2.0]:
        nu_scan = g_scan * np.tanh(r_scan)
        s = np.array(mechanical_cov(jnp.array(
            solve_lyapunov(g_scan, nu_scan, kappa=1.0, gamma=gamma, n_th=0.0))))
        r_num = squeezing_from_cov(s)
        r_k   = kronwald_squeezing(g_scan, nu_scan)
        print(f'  {nu_scan/g_scan:>8.5f} {r_scan:>8.3f} {s[0,0]:>10.6f} {r_num:>8.4f} {r_k:>8.4f}')
