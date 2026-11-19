#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def _g11(beta, omega, system):
    """G11 = [A^{-1}]_{11} via cofactor formula for 2x2 A.
    A = Omega^2 beta^2 M - i Omega beta C - K  (angular-normalised characteristic matrix).
    beta: (Nb,), omega: (No,) -> returns (Nb, No) complex array."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    b = beta[:, np.newaxis, np.newaxis, np.newaxis]
    w = omega[np.newaxis, :, np.newaxis, np.newaxis]
    M, C, K = system
    A = -(w**2)*(b**2)*M + 1j*w*b*C + K
    return A[..., 1, 1] / np.linalg.det(A)


def _H(beta, omega, system):
    """H = G11 * (exp(-i*beta) - 1).  Shape (Nb, No)."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return _g11(beta, omega, system) * (np.exp(-1j * beta[:, np.newaxis]) - 1)


def stability_scan(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0, n_omega=200,
    n_lobes=10, beta_step=0.05,
):
    """Brute-force stability boundary scan.

    Returns
    -------
    omega_grid : (n_omega,) array
    h_lim      : (n_omega,) array  — minimum stable chip thickness (NaN where none found)
    all_h      : list of lists    — all positive h values per omega (for branch plots)
    """
    omega_grid = np.linspace(omega_min, omega_max, n_omega)
    # scan points at non-integer multiples of pi to avoid trivial roots
    beta_scan = np.arange(beta_step, 2 * n_lobes, beta_step) * np.pi

    Im_H = np.imag(_H(beta_scan, omega_grid, system))  # (Nb, No)
    sign_changes = np.diff(np.sign(Im_H), axis=0) != 0  # (Nb-1, No)

    # mask brackets straddling trivial roots beta = 2k*pi
    trivial_mask = np.zeros(len(beta_scan) - 1, dtype=bool)
    for k in range(1, n_lobes + 1):
        trivial_mask |= (beta_scan[:-1] < 2*k*np.pi) & (beta_scan[1:] > 2*k*np.pi)
    sign_changes[trivial_mask, :] = False

    all_h = [[] for _ in range(n_omega)]

    for j, omega in enumerate(omega_grid):
        for i in np.where(sign_changes[:, j])[0]:
            a, b = beta_scan[i], beta_scan[i + 1]
            fun = lambda beta, _o=omega: np.imag(
                _H(np.array([beta]), np.array([_o]), system)[0, 0]
            )
            try:
                beta_root = scipy.optimize.brentq(fun, a, b, xtol=1e-10)
            except ValueError:
                continue
            h_val = 1.0 / (
                cut_coeff * np.real(
                    _H(np.array([beta_root]), np.array([omega]), system)[0, 0]
                )
            )
            if h_val > 0:
                all_h[j].append(h_val)

    h_lim = np.array([min(hs) if hs else np.nan for hs in all_h])
    return omega_grid, h_lim, all_h


def is_stable(system, omega, h, cut_coeff=1.0, n_lobes=10, beta_step=0.05):
    """Return True if the operating point (omega, h) is stable."""
    _, h_lim, _ = stability_scan(
        system, cut_coeff=cut_coeff,
        omega_min=omega, omega_max=omega, n_omega=1,
        n_lobes=n_lobes, beta_step=beta_step)
    limit = h_lim[0]
    if np.isnan(limit):
        return True  # no instability found at this speed
    return h < limit


def plot_lobes(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0, n_omega=200,
    n_lobes=10, beta_step=0.05,
    show_branches=True, ax=None,
):
    """Plot stability lobe diagram. Returns (fig, ax, omega_grid, h_lim)."""
    omega_grid, h_lim, all_h = stability_scan(
        system, cut_coeff=cut_coeff,
        omega_min=omega_min, omega_max=omega_max, n_omega=n_omega,
        n_lobes=n_lobes, beta_step=beta_step)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if show_branches:
        for omega, hs in zip(omega_grid, all_h):
            if hs:
                ax.scatter([omega]*len(hs), hs, c='lightgray', s=4, zorder=1)

    ax.plot(omega_grid, h_lim, 'b-', lw=1.5, label='stability boundary')
    ax.set_xlabel(r'$\Omega$')
    ax.set_ylabel('$h$')
    clip = np.nanpercentile(h_lim, 95)
    if not np.isnan(clip):
        ax.set_ylim(0, clip * 3)
    ax.legend()
    return fig, ax, omega_grid, h_lim
