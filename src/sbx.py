# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

# %%

# Generate three 2x2 random symmetric, positive definite matrices.
def generate_random_spd_matrix():
    A = np.random.rand(2, 2)
    return np.dot(A, A.T)

system = [generate_random_spd_matrix() for _ in range(3)]
system[1] *= 0.01 * system[1]

# %%
def g(f, system=None, omega:float=100):
    f = np.expand_dims(f, axis=(1, 2))
    if system is None:
        system = [np.eye(2), np.zeros((2, 2)), np.eye(2)]
    ch = -(omega**2)*(f**2)*system[0][np.newaxis] + 0j
    ch += 1j*omega*f*system[1][np.newaxis]
    ch += system[2][np.newaxis]

    g_ = ch[:, 1, 1] / np.linalg.det(ch)

    return g_

def f_obj(f, system=None, omega:float=100):
    if isinstance(f, (int, float)):
        f = np.array([f], dtype=float)
    f = np.asarray(f, dtype=float)
    assert f.ndim == 1, "f must be a scalar or a 1-dimensional array"

    a = g(f, system=system, omega=omega)
    a = a * (np.exp(-1j*f) - 1)
    return a

# %%
f = np.arange(5, 10, 0.01)*np.pi
d_omega = .001
for step in range(15, 20):
    omega = d_omega * step
    plt.plot(
        f, np.imag(f_obj(f, system=system, omega=omega)),
        label=f'{omega}')

plt.axhline(0, color='gray', linestyle='--')
# Put a cross at multiples of pi
plt.scatter(np.pi*np.arange(5, 10), [0]*5, color='blue', marker='x', zorder=5)
plt.legend()
plt.show()

# %%
roots = []
roots.append(np.pi*np.arange(1, 10, step=2))
d_omega = .005
for step in range(1, 3):
    omega = d_omega * step
    fun = lambda f: np.imag(f_obj(f, system=system, omega=omega))
    x0 = 2*roots[-1] - roots[-2] if len(roots) > 1 else roots[-1]
    f0 = scipy.optimize.newton(fun, x0=x0)
    roots.append(f0)

# cut_coeff = 1.
# h = 1/(cut_coeff * np.real(f_obj(f0, system=system, omega=omega)))

# %%
f = np.linspace(5, 11*np.pi, 1_000)
step = 1
omega = d_omega * step
plt.plot(f, np.imag(f_obj(f, system=system, omega=omega)))
plt.scatter(roots[step], [0]*len(roots[step]), color='red', zorder=5)
# Draw a horizontal line at y=0
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.show()

# %%