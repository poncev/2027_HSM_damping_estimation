# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

# %%

# Generate three 2x2 random symmetric, positive definite matrices.
def generate_random_spd_matrix_from_rng(rng):
    """Generate a random SPD matrix from an existing RNG."""
    A = rng.standard_normal((2, 2))
    return np.dot(A, A.T)

# Using spawn method for independent RNG streams
rng_parent = np.random.default_rng(seed=52)
rng_children = rng_parent.spawn(3)
system = [generate_random_spd_matrix_from_rng(rng_child) for rng_child in rng_children]
system[1] *= 0.02 * system[1]

# system[0] is the mass, system[1] is the damping, system[2] is the stiffness

# Augmented system: (0 & I \\ M^-1 K & M^-1 C)
# Check that the eigenvalues of the augmented system have nonzero imaginary parts
augmented_system = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-np.linalg.inv(system[0]) @ system[2], -np.linalg.inv(system[0]) @ system[1]]
])
eigvals = np.linalg.eigvals(augmented_system)
assert np.all(np.imag(eigvals) != 0), "Eigenvalues should have nonzero imaginary parts"

# %%
def g(beta, omega, system):
    """G11 entry of (Omega^2 beta^2 M - i Omega beta C - K)^{-1}.
    beta: (Nb,) or scalar, omega: (No,) or scalar. Returns (Nb, No) or broadcast shape."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    # shape (Nb, No, 1, 1) for beta^2 and beta, (1, 1, 2, 2) for matrices
    b = beta[:, np.newaxis, np.newaxis, np.newaxis]
    w = omega[np.newaxis, :, np.newaxis, np.newaxis]
    M, C, K = system
    ch = -(w**2)*(b**2)*M + 1j*w*b*C + K
    return ch[..., 1, 1] / np.linalg.det(ch)

def H(beta, omega, system):
    """H = G11 * (exp(-i*beta) - 1). Returns (Nb, No) array."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    g11 = g(beta, omega, system)
    return g11 * (np.exp(-1j * beta[:, np.newaxis]) - 1)

# %%
f = np.arange(3.9, 6.1, 0.01)*np.pi
d_omega = .0005
for step in np.arange(0.038, 0.042, d_omega):
    omega = step
    plt.plot(
        f, np.imag(H(f, np.atleast_1d(omega), system)[:, 0]),
        label=f'{omega}')

plt.axhline(0, color='gray', linestyle='--')
# Put a cross at multiples of pi
plt.scatter(np.pi*np.arange(4, 7), [0]*3, color='blue', marker='x', zorder=5)
plt.legend()
plt.show()

# %%
roots = []
roots.append(np.pi*np.arange(1, 10, step=2))
d_omega = .005
for step in range(1, 3):
    omega = d_omega * step
    fun = lambda f: np.imag(H(np.atleast_1d(f), np.atleast_1d(omega), system)[:, 0])
    x0 = 2*roots[-1] - roots[-2] if len(roots) > 1 else roots[-1]
    f0 = scipy.optimize.newton(fun, x0=x0)
    roots.append(f0)

# cut_coeff = 1.
# h = 1/(cut_coeff * np.real(H(np.atleast_1d(f0), np.atleast_1d(omega), system)[:, 0]))

# %%
# Step 1 — parameters
cut_coeff = 1.0
Omega_grid = np.linspace(0.01, 1.0, 200)
N_lobes = 10
# dense scan grid in steps of 0.05*pi, never landing on integer multiples of pi
beta_scan = np.arange(0.05, 2 * N_lobes, 0.05) * np.pi

# %%
# Steps 2-4 — vectorized scan, sign-change detection, brentq refinement
Im_H = np.imag(H(beta_scan, Omega_grid, system))  # (N_beta, N_omega)

sign_changes = np.diff(np.sign(Im_H), axis=0) != 0  # (N_beta-1, N_omega)

# trivial roots at even multiples of pi: flag brackets that straddle k*2*pi
trivial_mask = np.zeros(len(beta_scan) - 1, dtype=bool)
for k in range(1, N_lobes + 1):
    trivial_mask |= (beta_scan[:-1] < 2*k*np.pi) & (beta_scan[1:] > 2*k*np.pi)

sign_changes[trivial_mask, :] = False

all_beta_roots = [[] for _ in range(len(Omega_grid))]
all_h = [[] for _ in range(len(Omega_grid))]

for j, omega in enumerate(Omega_grid):
    col = sign_changes[:, j]
    for i in np.where(col)[0]:
        a, b = beta_scan[i], beta_scan[i + 1]
        fun = lambda beta: np.imag(H(np.array([beta]), np.array([omega]), system)[0, 0])
        try:
            beta_root = scipy.optimize.brentq(fun, a, b, xtol=1e-10)
        except ValueError:
            continue
        h_val = 1.0 / (cut_coeff * np.real(H(np.array([beta_root]), np.array([omega]), system)[0, 0]))
        if h_val > 0:
            all_beta_roots[j].append(beta_root)
            all_h[j].append(h_val)

h_lim = np.array([min(hs) if hs else np.nan for hs in all_h])

# %%
# Step 6 — plot
fig, ax = plt.subplots()
for j, (omega, hs) in enumerate(zip(Omega_grid, all_h)):
    if hs:
        ax.scatter([omega]*len(hs), hs, c='gray', s=4, alpha=0.4)
ax.plot(Omega_grid, h_lim, 'b-', lw=1.5, label='envelope')
ax.set_xlabel(r'$\Omega$')
ax.set_ylabel('h')
ax.set_ylim(0, np.nanpercentile(h_lim, 95) * 3)
ax.legend()
plt.show()

# %%
f = np.linspace(5, 11*np.pi, 1_000)
step = 1
omega = d_omega * step
plt.plot(f, np.imag(H(f, np.atleast_1d(omega), system)[:, 0]))
plt.scatter(roots[step], [0]*len(roots[step]), color='red', zorder=5)
# Draw a horizontal line at y=0
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.show()

# %%