# %%
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from lobes import _H, plot_lobes

# %%
# Random 2-DoF system for sandbox experiments (seed fixed for reproducibility)
def generate_random_spd_matrix_from_rng(rng):
    A = rng.standard_normal((2, 2))
    return np.dot(A, A.T)

rng_parent = np.random.default_rng(seed=52)
rng_children = rng_parent.spawn(3)
system = [generate_random_spd_matrix_from_rng(rng_child) for rng_child in rng_children]
system[1] *= 0.02
# system = [M, C, K]

# sanity check: underdamped system (complex eigenvalues)
augmented_system = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-np.linalg.inv(system[0]) @ system[2], -np.linalg.inv(system[0]) @ system[1]]
])
eigvals = np.linalg.eigvals(augmented_system)
assert np.all(np.imag(eigvals) != 0)

# %%
# Exploration: Im(H) near beta=4..6*pi for several omega values.
# This revealed saddle-node bifurcations: Im(H) develops a local extremum at
# even multiples of pi, eventually creating two new roots that travel apart.
# Blue crosses mark trivial roots (beta = even*pi), where h -> infinity.
f = np.arange(3.9, 6.1, 0.01)*np.pi
d_omega = .0005
for step in np.arange(0.038, 0.042, d_omega):
    omega = step
    plt.plot(
        f, np.imag(_H(f, np.atleast_1d(omega), system)[:, 0]),
        label=f'{omega:.4f}')
plt.axhline(0, color='gray', linestyle='--')
plt.scatter(np.pi*np.arange(4, 7), [0]*3, color='blue', marker='x', zorder=5)
plt.legend()
plt.show()

# %%
# Stability lobe diagram for the random sandbox system
fig, ax, omega_grid, h_lim = plot_lobes(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0,
    n_omega=200, n_lobes=10,
)
plt.show()

# %%
# =============================================================================
# Realistic turning example: boring bar + TMD, Inconel 718
#
# Structural parameters are representative of a carbide boring bar on a lathe;
# see Yang, Munoa & Altintas, Int. J. Mach. Tools Manuf. 50(9):834-842, 2010.
#
# Normal (radial/thrust) cutting coefficient Kn for Inconel 718:
# Pawade, Joshi & Brahmankar, Int. J. Mach. Tools Manuf. 48(1):15-28, 2008.
#   Kn ≈ 1200 N/mm²  (force per unit chip-thickness per unit depth, radial)
#
# TMD optimal tuning: Den Hartog (1956), Mechanical Vibrations, 4th ed., §3.3.
# =============================================================================

# --- Primary structure (boring bar, modal parameters) -----------------------
m1 = 2.0         # kg      modal mass
fn1 = 350.0      # Hz      natural frequency (bare tool, no TMD)
zeta1 = 0.025    # —       structural damping ratio
wn1 = 2 * np.pi * fn1   # rad/s
k1 = m1 * wn1**2         # N/m   ~9.67 MN/m
c1 = 2 * zeta1 * m1 * wn1  # N·s/m

# --- Den Hartog optimal TMD (mass ratio mu = 0.05) --------------------------
mu = 0.05
m2 = mu * m1                                          # kg  = 0.1 kg
f_opt = 1.0 / (1.0 + mu)                              # optimal frequency ratio
zeta2_opt = np.sqrt(3 * mu / (8 * (1.0 + mu)**3))    # optimal damping ratio
wn2 = f_opt * wn1
k2 = m2 * wn2**2
c2 = 2 * zeta2_opt * m2 * wn2

M_t = np.array(
    [[m1,      0.0],
     [0.0,     m2]])
C_t = np.array(
    [[c1 + c2, -c2],
     [-c2,      c2]])
K_t = np.array(
    [[k1 + k2, -k2],
     [-k2,      k2]])
system_turning = [M_t, C_t, K_t]

# --- Normal cutting force coefficient (radial direction, Inconel 718) -------
# In turning the regenerative chip-thickness variation is in the radial direction,
# so the relevant coefficient is Kn, not Kt.
Kn = 1.2e9  # N/m²  (= 1200 N/mm²)

# --- Stability lobe diagram --------------------------------------------------
# Lobe peaks sit near Omega ≈ wn1 / ((2n-1)*pi):
#   n=1 → ~700 rad/s, n=2 → ~233 rad/s, n=3 → ~140 rad/s, n=4 → ~100 rad/s
fig_t, ax_t, omega_grid_t, h_lim_t = plot_lobes(
    system_turning, cut_coeff=Kn,
    omega_min=50.0, omega_max=750.0,
    n_omega=400, n_lobes=5,
)
ax_t.set_xlabel(r'$\Omega$ [rad/s]')
ax_t.set_ylabel('Axial depth of cut [m]')
ax_t.set_title('Turning stability — Inconel 718, boring bar + TMD (Den Hartog)')
plt.show()

# %%
