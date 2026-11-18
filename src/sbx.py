#!/usr/bin/env python3

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
def g(f, system=None, omega=100):
    if system is None:
        system = [np.eye(2), np.zeros((2, 2)), np.eye(2)]
    ch = -(omega**2)*(f**2)*system[0] + 0j
    ch += 1j*omega*f*system[1]
    ch += system[2]

    g_ = ch[1, 1] / np.linalg.det(ch)

    return g_

def f_obj(f, system=None, omega=100):
    a = g(f, system=system, omega=omega)
    a = a * (np.exp(-1j*f) - 1)
    return a

# %%
omega = 10
fun = lambda f: np.imag(f_obj(f, system=system, omega=omega))
f0 = scipy.optimize.newton(fun, x0=1.2)

cut_coeff = 10
h = 1/(cut_coeff * np.real(f_obj(f0, system=system, omega=omega)))

# %%
f = np.linspace(-1, 15, 200)
y = [np.imag(f_obj(fi, system=system, omega=omega)) for fi in f]
plt.plot(f, y)
plt.scatter(f0, 0, color='red', zorder=5)
# Draw a horizontal line at y=0
plt.axhline(0, color='gray', linestyle='--')
plt.show()

# %%
if __name__ == "__main__":
    pass