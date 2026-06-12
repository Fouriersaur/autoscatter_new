import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))   # adds reservoir_engineering/ to path

import jax.numpy as jnp
import numpy as np

from covariance_physics import (
    build_drift_matrix,
    build_diffusion_matrix,
    solve_lyapunov_kronecker,
    get_mode_covariance,
)

# ── System definition (Kronwald System) ────────────────────────────────────
nodes = [
    {'id': 0, 'type': 'cavity',     'kappa': 1.0,  'delta': 0.0},
    {'id': 1, 'type': 'mechanical', 'gamma': 0.01, 'n_th': 0.0, 'delta': 0.0},
]
edges = [
    {'i': 0, 'j': 1, 'type': 'beamsplitter'},
    {'i': 0, 'j': 1, 'type': 'two_mode_squeezing'},
]

# ── Kronwald coupling values ───────────────────────────────────────────────
r  = 1.0
g  = 0.5
nu = g * float(jnp.tanh(r))   # ≈ 0.381  →  ν/g = tanh(r)

# ── Build A and D ──────────────────────────────────────────────────────────
A = build_drift_matrix(nodes, edges, jnp.array([g, nu]))
D = build_diffusion_matrix(nodes)

print("A")
print(np.array(A))
print()
print("D =")
print(np.array(D))
print()

# ── Solve Lyapunov ─────────────────────────────────────────────────────────
sigma = solve_lyapunov_kronecker(A, D)

print("sigma (full 4x4) =")
print(np.round(np.array(sigma), 4))
print()

# ── Extract mechanical mode covariance ────────────────────────────────────
mech = get_mode_covariance(sigma, [1])

print("sigma_mech (2x2) =")
print(np.round(np.array(mech), 4))
print()

# ── Checks ─────────────────────────────────────────────────────────────────
mech_np   = np.array(mech)
sigma_xx  = mech_np[0, 0]   # x-quadrature variance
sigma_pp  = mech_np[1, 1]   # p-quadrature variance
det       = sigma_xx * sigma_pp - mech_np[0,1]**2

print("── Qualitative checks ──")

# 1. Diagonal entries must be positive (physical covariance matrix)
check1 = sigma_xx > 0 and sigma_pp > 0
print(f"  [{'PASS' if check1 else 'FAIL'}] σ_xx={sigma_xx:.4f} > 0  and  σ_pp={sigma_pp:.4f} > 0")

# 2. x-quadrature must be squeezed below vacuum (σ_xx < 0.5)
check2 = sigma_xx < 0.5
print(f"  [{'PASS' if check2 else 'FAIL'}] σ_xx={sigma_xx:.4f} < 0.5  (squeezed below vacuum)")

# 3. Uncertainty principle: det(σ) >= 0.25
check3 = det >= 0.25 - 1e-6
print(f"  [{'PASS' if check3 else 'FAIL'}] det(σ)={det:.4f} >= 0.25  (uncertainty principle)")

# 4. Nearly pure state: det(σ) should be close to 0.25 (small γ/κ ratio)
check4 = det < 0.5
print(f"  [{'PASS' if check4 else 'FAIL'}] det(σ)={det:.4f} < 0.5  (close to pure state)")

# 5. Squeezing direction: σ_xx < σ_pp (x is squeezed, p is anti-squeezed)
check5 = sigma_xx < sigma_pp
print(f"  [{'PASS' if check5 else 'FAIL'}] σ_xx < σ_pp  (correct squeezing direction)")

print()
r_achieved = 0.5 * np.log(sigma_pp / sigma_xx) / 2
print(f"  Squeezing parameter achieved: r = {r_achieved:.4f}  (target r = {r})")
print(f"  Note: ideal formula ½·exp(-2r) is only exact when g,ν << κ and C >> 1.")
print(f"  With g={g}, κ=1: g/κ = {g} (not small), so finite-coupling deviation is expected.")
print()

if all([check1, check2, check3, check4, check5]):
    print("PASS: physics engine is correct")
else:
    print("FAIL: check your implementation")
