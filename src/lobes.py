import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt


def _build_modal_system(system):
    """Return (modal_system, direction) where modal_system = [I, C_tilde, K_tilde]
    and direction = Phi_n^T e1 (cutting force in modal coords).
    Mode shapes from undamped eigenproblem K phi = w^2 M phi -> guaranteed real."""
    M, C, K = system
    # eigh solves K Phi = M Phi diag(w^2), returns real Phi with Phi^T M Phi = I
    _, Phi_n = scipy.linalg.eigh(K, M)
    C_tilde = Phi_n.T @ C @ Phi_n
    K_tilde = np.diag(Phi_n.T @ K @ Phi_n)  # exactly diagonal by construction
    K_tilde = np.diag(np.real(K_tilde))
    modal_system = [np.eye(2), C_tilde, K_tilde]
    direction = Phi_n.T @ np.array([1.0, 0.0])  # e1 in modal coords
    return modal_system, direction


def _g_quad(beta, omega, system, direction):
    """d^T A^{-1} d via 2x2 adjugate (exact, vectorised).
    direction: (2,) real array.  Returns (Nb, No) complex array."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    d = np.asarray(direction, dtype=float)
    b = beta[:, np.newaxis, np.newaxis, np.newaxis]
    w = omega[np.newaxis, :, np.newaxis, np.newaxis]
    M, C, K = system
    A = -(w**2)*(b**2)*M + 1j*w*b*C + K
    det = np.linalg.det(A)                    # (Nb, No)
    # adjugate of 2x2: adj(A)_ij = cofactor transposed
    adj = np.empty_like(A)
    adj[..., 0, 0] =  A[..., 1, 1]
    adj[..., 1, 1] =  A[..., 0, 0]
    adj[..., 0, 1] = -A[..., 0, 1]
    adj[..., 1, 0] = -A[..., 1, 0]
    # d^T adj d  (scalar per (Nb, No) point)
    quad = (d[0]**2 * adj[..., 0, 0]
            + d[0]*d[1] * (adj[..., 0, 1] + adj[..., 1, 0])
            + d[1]**2 * adj[..., 1, 1])
    return quad / det


def _g11(beta, omega, system):
    """G11 = e1^T A^{-1} e1 = A[...,1,1] / det(A) (cofactor shortcut for e1)."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    b = beta[:, np.newaxis, np.newaxis, np.newaxis]
    w = omega[np.newaxis, :, np.newaxis, np.newaxis]
    M, C, K = system
    A = -(w**2)*(b**2)*M + 1j*w*b*C + K
    return A[..., 1, 1] / np.linalg.det(A)


def _H(beta, omega, system, direction=None):
    """H = G * (exp(-i*beta) - 1), where G = d^T A^{-1} d.  Shape (Nb, No)."""
    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if direction is None:
        g = _g11(beta, omega, system)
    else:
        g = _g_quad(beta, omega, system, direction)
    return g * (np.exp(-1j * beta[:, np.newaxis]) - 1)


def _prepare(system, representation):
    """Return (system, direction) ready for _H.  direction=None uses e1 shortcut."""
    if representation == 'mechanical':
        return system, None
    if representation == 'modal':
        return _build_modal_system(system)
    raise ValueError(f"representation must be 'mechanical' or 'modal', got {representation!r}")


def compute_absolute_stability_limit(
    system, cut_coeff=1.0,
    omega_min=10.0, omega_max=20000.0, n_omega=5000,
    representation='mechanical',
):
    """Compute absolute (unconditional) stability limit h_abs.
    
    From Tlusty & Polacek (1963), Tobias (1965), Altintas (2012):
        h_abs = -1 / (2 * Kn * min_omega Re[G(omega)])
    
    This is the minimum chip thickness below which the system is stable
    for all spindle speeds. Stability lobes touch this limit at frequencies
    near the natural frequencies.
    
    Parameters
    ----------
    system : list [M, C, K]
        Mechanical system matrices
    cut_coeff : float
        Cutting coefficient Kn (N/m²)
    omega_min, omega_max : float
        Frequency search range (rad/s)
    n_omega : int
        Number of frequencies to search
    representation : str
        'mechanical' or 'modal'
    
    Returns
    -------
    h_abs : float
        Absolute stability limit (m)
    omega_at_min : float
        Frequency where Re[G] is minimum (rad/s)
    Re_G_min : float
        Minimum value of Re[G(omega)]
    """
    sys_, direction = _prepare(system, representation)
    
    # Search over frequency range
    omega_search = np.linspace(omega_min, omega_max, n_omega)
    
    # Compute G(omega) at each frequency
    # G = d^T A^{-1} d where A = K + i*omega*C - omega^2*M
    Re_G_values = []
    for omega in omega_search:
        if direction is None:
            # Use e1 shortcut: G = A[1,1] / det(A)
            M, C, K = sys_
            A = K + 1j * omega * C - omega**2 * M
            G = A[1, 1] / np.linalg.det(A)
        else:
            # General directional case
            M, C, K = sys_
            A = K + 1j * omega * C - omega**2 * M
            A_inv = np.linalg.inv(A)
            G = np.dot(direction, np.dot(A_inv, direction))
        Re_G_values.append(np.real(G))
    
    Re_G_values = np.array(Re_G_values)
    
    # Find minimum
    idx_min = np.argmin(Re_G_values)
    omega_at_min = omega_search[idx_min]
    Re_G_min = Re_G_values[idx_min]
    
    # Compute absolute stability limit
    h_abs = -1.0 / (2.0 * cut_coeff * Re_G_min)
    
    return h_abs, omega_at_min, Re_G_min


def stability_scan(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0, n_omega=200,
    n_lobes=10, beta_step=0.05,
    representation='mechanical',
):
    """Brute-force stability boundary scan.

    Returns
    -------
    omega_grid : (n_omega,) array
    h_lim      : (n_omega,) array  — minimum stable chip thickness (NaN where none found)
    all_h      : list of lists    — all positive h values per omega (for branch plots)
    """
    sys_, direction = _prepare(system, representation)
    omega_grid = np.linspace(omega_min, omega_max, n_omega)
    # scan points at non-integer multiples of pi to avoid trivial roots
    beta_scan = np.arange(beta_step, 2 * n_lobes, beta_step) * np.pi

    Im_H = np.imag(_H(beta_scan, omega_grid, sys_, direction))  # (Nb, No)
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
                _H(np.array([beta]), np.array([_o]), sys_, direction)[0, 0]
            )
            try:
                beta_root = scipy.optimize.brentq(fun, a, b, xtol=1e-10)
            except ValueError:
                continue
            h_val = 1.0 / (
                cut_coeff * np.real(
                    _H(np.array([beta_root]), np.array([omega]), sys_, direction)[0, 0]
                )
            )
            if h_val > 0:
                all_h[j].append(h_val)

    h_lim = np.array([min(hs) if hs else np.nan for hs in all_h])
    return omega_grid, h_lim, all_h


def is_stable(
    system, omega, h, cut_coeff=1.0,
    n_lobes=10, beta_step=0.05, representation='mechanical',
):
    """Return True if the operating point (omega, h) is stable."""
    _, h_lim, _ = stability_scan(
        system, cut_coeff=cut_coeff,
        omega_min=omega, omega_max=omega, n_omega=1,
        n_lobes=n_lobes, beta_step=beta_step,
        representation=representation,
    )
    limit = h_lim[0]
    if np.isnan(limit):
        return True  # no instability found at this speed
    return h < limit


def plot_lobes(
    system, cut_coeff=1.0,
    omega_min=0.01, omega_max=1.0, n_omega=200,
    n_lobes=10, beta_step=0.05,
    show_branches=True, ax=None,
    representation='mechanical',
):
    """Plot stability lobe diagram. Returns (fig, ax, omega_grid, h_lim)."""
    omega_grid, h_lim, all_h = stability_scan(
        system, cut_coeff=cut_coeff,
        omega_min=omega_min, omega_max=omega_max, n_omega=n_omega,
        n_lobes=n_lobes, beta_step=beta_step,
        representation=representation,
    )

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
