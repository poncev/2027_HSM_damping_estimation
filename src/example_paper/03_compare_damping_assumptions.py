"""
Compare proportional vs non-proportional damping assumptions using EXACT modal parameters.

This script isolates the effect of the proportional damping assumption by using
exact poles and amplitudes (no identification errors, no noise). The only source
of error is the mathematical constraint that mode shapes must be real.

Steps:
1. Compute exact modal parameters (poles & amplitudes) from true M, C, K
2. Fit mode shapes using RealModes (proportional) and ComplexModes (non-proportional)
3. Reconstruct M, C, K from fitted mode shapes
4. Compare errors
5. Generate and compare IRFs from true and reconstructed systems
6. Compare stability lobes
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bcam.indar import ema
from system_params import SYSTEM_TRUE, FS, DURATION, COORD_LABELS, Kn
from lobes import plot_lobes

# Output directories
DATA_DIR = Path(__file__).parent / 'data'
IMAGES_DIR = Path(__file__).parent.parent.parent / 'images'
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("MODE SHAPE RECOVERY: DAMPING ASSUMPTION COMPARISON")
print("=" * 70)

# --- Load true system ---
M_true, C_true, K_true = SYSTEM_TRUE
dt = 1.0 / FS
ns = int(DURATION * FS)
tt = np.arange(ns) * dt

n_dof = M_true.shape[0]
n_out = n_dof  # measure both coordinates
n_in = 1       # single input on coordinate 1

print(f"\nSystem: {n_dof}-DoF boring bar + TMD")
print(f"True matrices:")
print(f"M =\n{M_true}")
print(f"\nC =\n{C_true}")
print(f"\nK =\n{K_true}")

# --- Compute exact modal parameters (poles & amplitudes ONLY) ---
print("\n" + "-" * 70)
print("STEP 1: Compute Exact Modal Parameters")
print("-" * 70)

# System to modal: returns normalized mode shapes and natural frequencies
mode_shapes_true, nat_freqs_true = ema.system_to_modal(M_true, C_true, K_true)

print(f"\nTrue natural frequencies (continuous-time):")
for i, lam in enumerate(nat_freqs_true):
    fn = np.abs(np.imag(lam)) / (2 * np.pi)
    zeta = -np.real(lam) / np.abs(lam)
    print(f"  Mode {i+1}: f = {fn:.2f} Hz, ζ = {zeta:.6f}")

# Apply reduced normalization (required by indar convention)
mode_shapes_norm = mode_shapes_true / np.sqrt(np.imag(nat_freqs_true))[np.newaxis, :]

print(f"\nTrue mode shapes (reduced normalization):")
print(mode_shapes_norm)

# Check if damping is proportional
# For proportional damping, all components of a mode shape must have the same phase
# This means phase differences between coordinates should be 0° (same sign) or 180° (opposite sign)
mode_phases = np.angle(mode_shapes_true)
phase_diff = np.abs(mode_phases[0, :] - mode_phases[1, :])
phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # wrap to [0, pi]

# Check if phases are close to 0° or 180° (real mode shapes)
is_zero = np.isclose(phase_diff, 0.0, atol=0.01)
is_pi = np.isclose(phase_diff, np.pi, atol=0.01)
is_proportional = np.all(is_zero | is_pi)

print(f"\nProportional damping check:")
print(f"  Phase differences: {np.rad2deg(phase_diff)} degrees")
print(f"  System is {'PROPORTIONAL' if is_proportional else 'NON-PROPORTIONAL'}")

# Compute exact amplitudes (residues) - needed for fitting
amps_true = ema.mode_to_amps(mode_shapes_norm, n_out, n_in)

print(f"\nTrue amplitudes (tensor shape: {amps_true.shape}):")
for i_out in range(n_out):
    for i_in in range(n_in):
        print(f"  Out {i_out}, In {i_in}: {amps_true[i_out, i_in, :]}")

# Convert to discrete-time poles for fitting
poles_discrete = np.exp(nat_freqs_true * dt)

# --- Fit mode shapes using RealModes (proportional assumption) ---
print("\n" + "-" * 70)
print("STEP 2: Fit Mode Shapes with Proportional Damping Assumption")
print("-" * 70)

print("\nFitting RealModes using ONLY poles and amplitudes...")
print("(Mode shapes are what we're trying to recover)")

model_real = ema.RealModes(
    poles=poles_discrete,
    amps=amps_true,
    fs=FS,
    ns=ns,
    response='a'
)

model_real.fit(options_ncg={'disp': 2, 'gtol': 1e-6})

print(f"\nOptimization complete:")
print(f"  Success: {model_real.success_}")

modes_real = model_real.modes_
print(f"\nRecovered mode shapes (real, proportional assumption):")
print(modes_real)

# --- Fit mode shapes using ComplexModes (non-proportional) ---
print("\n" + "-" * 70)
print("STEP 3: Fit Mode Shapes without Proportional Assumption")
print("-" * 70)

print("\nFitting ComplexModes using ONLY poles and amplitudes...")
print("(Mode shapes can be complex)")

model_complex = ema.ComplexModes(
    poles=poles_discrete,
    coords=np.arange(n_out),
    amps=amps_true,
    fs=FS,
    ns=ns,
    response='a'
)

model_complex.fit(
    x0=modes_real,
    maxiter=100,
    options={'verbose': 2, 'gtol': 1e-6, 'xtol': 1e-8}
)

print(f"\nOptimization complete")

modes_complex = model_complex.modes_
print(f"\nRecovered mode shapes (complex, non-proportional):")
print(modes_complex)

# Compare with true mode shapes
print(f"\nTrue mode shapes (for comparison):")
print(mode_shapes_norm)

# --- Reconstruct M, C, K matrices ---
print("\n" + "-" * 70)
print("STEP 4: Reconstruct M, C, K from Recovered Mode Shapes")
print("-" * 70)

# Convert discrete poles back to continuous
roots_continuous = np.log(poles_discrete) / dt

print("\nReconstruction from RealModes (proportional):")
modes_real_scaled = modes_real * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
M_prop, C_prop, K_prop = ema.modal_to_system(modes_real_scaled, roots_continuous)

print(f"M (proportional) =\n{M_prop}")
print(f"\nC (proportional) =\n{C_prop}")
print(f"\nK (proportional) =\n{K_prop}")

print("\nReconstruction from ComplexModes (non-proportional):")
modes_complex_scaled = (
    modes_complex * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
)
M_nonprop, C_nonprop, K_nonprop = ema.modal_to_system(
    modes_complex_scaled, roots_continuous
)

print(f"M (non-proportional) =\n{M_nonprop}")
print(f"\nC (non-proportional) =\n{C_nonprop}")
print(f"\nK (non-proportional) =\n{K_nonprop}")

# Compute relative errors
print("\n" + "-" * 70)
print("STEP 5: Compute Matrix Reconstruction Errors")
print("-" * 70)

M_error_prop = np.linalg.norm(M_prop - M_true) / np.linalg.norm(M_true)
C_error_prop = np.linalg.norm(C_prop - C_true) / np.linalg.norm(C_true)
K_error_prop = np.linalg.norm(K_prop - K_true) / np.linalg.norm(K_true)

M_error_nonprop = np.linalg.norm(M_nonprop - M_true) / np.linalg.norm(M_true)
C_error_nonprop = np.linalg.norm(C_nonprop - C_true) / np.linalg.norm(C_true)
K_error_nonprop = np.linalg.norm(K_nonprop - K_true) / np.linalg.norm(K_true)

print(f"\nProportional damping assumption (RealModes):")
print(f"  M error: {M_error_prop*100:.4f}%")
print(f"  C error: {C_error_prop*100:.4f}%")
print(f"  K error: {K_error_prop*100:.4f}%")

print(f"\nNon-proportional damping (ComplexModes):")
print(f"  M error: {M_error_nonprop*100:.4f}%")
print(f"  C error: {C_error_nonprop*100:.4f}%")
print(f"  K error: {K_error_nonprop*100:.4f}%")

print(f"\nError reduction by using ComplexModes:")
print(f"  M: {(1 - M_error_nonprop/M_error_prop)*100:.2f}%")
print(f"  C: {(1 - C_error_nonprop/C_error_prop)*100:.2f}%")
print(f"  K: {(1 - K_error_nonprop/K_error_prop)*100:.2f}%")

# --- Generate IRFs for comparison ---
print("\n" + "-" * 70)
print("STEP 6: Generate IRFs for Time-Domain Comparison")
print("-" * 70)

print("\nGenerating IRFs from reconstructed systems...")

# True system IRF
kernel_true = ema.Kernel(nat_freqs_true, amps_true)
irf_true = kernel_true(tt)

# Proportional system IRF (from reconstructed M, C, K)
mode_shapes_prop, nat_freqs_prop = ema.system_to_modal(M_prop, C_prop, K_prop)
mode_shapes_prop_norm = (
    mode_shapes_prop / np.sqrt(np.imag(nat_freqs_prop))[np.newaxis, :]
)
amps_prop = ema.mode_to_amps(mode_shapes_prop_norm, n_out, n_in)
kernel_prop = ema.Kernel(nat_freqs_prop, amps_prop)
irf_prop = kernel_prop(tt)

# Non-proportional system IRF (from reconstructed M, C, K)
mode_shapes_nonprop, nat_freqs_nonprop = ema.system_to_modal(
    M_nonprop, C_nonprop, K_nonprop
)
mode_shapes_nonprop_norm = (
    mode_shapes_nonprop / np.sqrt(np.imag(nat_freqs_nonprop))[np.newaxis, :]
)
amps_nonprop = ema.mode_to_amps(mode_shapes_nonprop_norm, n_out, n_in)
kernel_nonprop = ema.Kernel(nat_freqs_nonprop, amps_nonprop)
irf_nonprop = kernel_nonprop(tt)

print(f"IRF shapes: {irf_true.shape}")

# Plot IRF comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot for coordinate 0 (bar tip)
ax = axes[0]
ax.plot(tt*1e3, irf_true[0, 0, :].real, 'k-', linewidth=2, label='True system')
ax.plot(
    tt*1e3, irf_prop[0, 0, :].real, 'r--',
    linewidth=1.5, label='Proportional (RealModes)'
)
ax.plot(
    tt*1e3, irf_nonprop[0, 0, :].real, 'b-.',
    linewidth=1.5, label='Non-proportional (ComplexModes)'
)
ax.set_ylabel(f'{COORD_LABELS[0]} IRF [m/N]', fontsize=11)
ax.set_title(
    'Impulse Response Function Comparison',
    fontsize=12,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim((0, 20))

# Plot for coordinate 1 (TMD)
ax = axes[1]
ax.plot(tt*1e3, irf_true[1, 0, :].real, 'k-', linewidth=2, label='True system')
ax.plot(
    tt*1e3, irf_prop[1, 0, :].real, 'r--',
    linewidth=1.5, label='Proportional (RealModes)'
)
ax.plot(
    tt*1e3, irf_nonprop[1, 0, :].real, 'b-.',
    linewidth=1.5, label='Non-proportional (ComplexModes)'
)
ax.set_xlabel('Time [ms]', fontsize=11)
ax.set_ylabel(f'{COORD_LABELS[1]} IRF [m/N]', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim((0, 20))

plt.tight_layout()

# Save IRF comparison figure
irf_output = IMAGES_DIR / '03_irf_comparison.jpg'
fig.savefig(irf_output, dpi=300, bbox_inches='tight')
print(f"\nIRF comparison saved to: {irf_output}")

# --- Stability Lobe Comparison ---
print("\n" + "-" * 70)
print("STEP 7: Stability Lobe Comparison - Low Speed (Pocket Structure)")
print("-" * 70)

# Fixed omega range for rich lobe visualization
# Lower speeds show characteristic lobe "pockets" where Omega ≈ omega_c / k
omega_min_low = 50.0    # rad/s - low speed regime
omega_max_low = 600.0   # rad/s - focus on packed lobe region

print(f"\nStability lobe parameters (low speed):")
print(f"  Omega range: [{omega_min_low:.1f}, {omega_max_low:.1f}] rad/s")
print(f"  Cutting coefficient Kn: {Kn:.2e} N/m²")

# Create figure for low-speed lobes
fig_low, ax_low = plt.subplots(figsize=(10, 7))

print("\nComputing low-speed stability lobes...")
print("  True system...")
_, _, omega_true_low, h_true_low = plot_lobes(
    [M_true, C_true, K_true],
    cut_coeff=Kn,
    omega_min=omega_min_low,
    omega_max=omega_max_low,
    n_omega=500,
    n_lobes=80,
    beta_step=0.01,
    show_branches=False,
    ax=ax_low,
    representation='mechanical'
)
ax_low.lines[-1].set_label('True system')
ax_low.lines[-1].set_color('#2ca02c')  # Green
ax_low.lines[-1].set_linewidth(3.0)
ax_low.lines[-1].set_linestyle('--')
ax_low.lines[-1].set_zorder(3)
# Convert to mm
ax_low.lines[-1].set_ydata(ax_low.lines[-1].get_ydata() * 1000)

print("  Proportional damping (RealModes)...")
_, _, omega_prop_low, h_prop_low = plot_lobes(
    [M_prop, C_prop, K_prop],
    cut_coeff=Kn,
    omega_min=omega_min_low,
    omega_max=omega_max_low,
    n_omega=500,
    n_lobes=80,
    beta_step=0.01,
    show_branches=False,
    ax=ax_low,
    representation='mechanical'
)
ax_low.lines[-1].set_label('Proportional')
ax_low.lines[-1].set_color('#d62728')  # Red
ax_low.lines[-1].set_linewidth(2.5)
ax_low.lines[-1].set_linestyle('-')
ax_low.lines[-1].set_zorder(2)
# Convert to mm
ax_low.lines[-1].set_ydata(ax_low.lines[-1].get_ydata() * 1000)

print("  Non-proportional damping (ComplexModes)...")
_, _, omega_nonprop_low, h_nonprop_low = plot_lobes(
    [M_nonprop, C_nonprop, K_nonprop],
    cut_coeff=Kn,
    omega_min=omega_min_low,
    omega_max=omega_max_low,
    n_omega=500,
    n_lobes=80,
    beta_step=0.01,
    show_branches=False,
    ax=ax_low,
    representation='mechanical'
)
ax_low.lines[-1].set_label('Non-proportional')
ax_low.lines[-1].set_color('#ff7f0e')  # Orange
ax_low.lines[-1].set_linewidth(2.5)
ax_low.lines[-1].set_linestyle('-')
ax_low.lines[-1].set_zorder(1)
# Convert to mm
ax_low.lines[-1].set_ydata(ax_low.lines[-1].get_ydata() * 1000)

# Format low-speed plot
ax_low.set_xlabel(r'Spindle speed $\Omega$ [rad/s]', fontsize=18)
ax_low.set_ylabel('Chip thickness $h$ [mm]', fontsize=18)
ax_low.legend(loc='upper left', fontsize=16, framealpha=0.9)
ax_low.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_low.set_xlim((omega_min_low, omega_max_low))
ax_low.set_ylim((1, 7))
ax_low.tick_params(axis='both', which='major', labelsize=16)

# Save low-speed figure
lobes_output_low = IMAGES_DIR / '03_stability_lobes_low_speed.jpg'
fig_low.savefig(lobes_output_low, dpi=350, bbox_inches='tight', format='jpeg')
print(f"\nLow-speed stability lobes saved to: {lobes_output_low}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nGenerated figures:")
print(f"  1. Low-speed lobes (pocket structure):  {lobes_output_low}")

plt.show()
