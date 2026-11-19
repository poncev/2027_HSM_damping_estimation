import numpy as np
import pytest
import sys
sys.path.insert(0, '.')
from lobes import stability_scan


def _make_system(seed=52):
    rng = np.random.default_rng(seed)
    children = rng.spawn(3)
    mats = []
    for child in children:
        A = child.standard_normal((2, 2))
        mats.append(A @ A.T)
    mats[1] *= 0.02 * mats[1]
    return mats


def test_mechanical_vs_modal_hlim():
    """Modal and mechanical representations must yield identical h_lim."""
    system = _make_system()
    kw = dict(cut_coeff=1.0, omega_min=0.1, omega_max=0.5, n_omega=50, n_lobes=5)
    _, h_mech, _ = stability_scan(system, **kw, representation='mechanical')
    _, h_modal, _ = stability_scan(system, **kw, representation='modal')
    np.testing.assert_allclose(h_mech, h_modal, rtol=1e-6, atol=1e-10)


def test_mechanical_vs_modal_branches():
    """Both representations must find the same number of branches at each omega."""
    system = _make_system()
    kw = dict(cut_coeff=1.0, omega_min=0.1, omega_max=0.5, n_omega=30, n_lobes=5)
    _, _, all_h_mech = stability_scan(system, **kw, representation='mechanical')
    _, _, all_h_modal = stability_scan(system, **kw, representation='modal')
    for hs_m, hs_q in zip(all_h_mech, all_h_modal):
        assert len(hs_m) == len(hs_q)
