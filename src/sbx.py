# %%
import sys

import numpy as np
import scipy
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
system[1] *= 0.02 * system[1]
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
# Early root-continuation experiment (superseded by brute-force scan in lobes.py).
# Newton's method from odd-pi seeds; misses branches born at bifurcations.
roots = []
roots.append(np.pi*np.arange(1, 10, step=2))
d_omega = .005
for step in range(1, 3):
    omega = d_omega * step
    fun = lambda f: np.imag(_H(np.atleast_1d(f), np.atleast_1d(omega), system)[:, 0])
    x0 = 2*roots[-1] - roots[-2] if len(roots) > 1 else roots[-1]
    f0 = scipy.optimize.newton(fun, x0=x0)
    roots.append(f0)
# h = 1/(cut_coeff * np.real(_H(np.atleast_1d(f0), np.atleast_1d(omega), system)[:, 0]))

# %%
# Stability lobe diagram for the random sandbox system
fig, ax, omega_grid, h_lim = plot_lobes(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0,
    n_omega=200, n_lobes=10,
)
plt.show()