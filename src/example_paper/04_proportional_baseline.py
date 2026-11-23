"""
Baseline validation: Compare RealModes vs ComplexModes on a PROPORTIONAL damping system.

This script validates our methodology by showing that RealModes works correctly
when damping IS proportional. Expected results:
- RealModes should achieve low errors (<5%)
- ComplexModes should also work (it's more general)
- This contrasts with script 03 where the TMD system is non-proportional

Steps:
1. Create a proportionally damped 2-DoF system from modal parameters
2. Verify proportionality by checking phase differences
3. Compute exact modal parameters (poles & amplitudes)
4. Fit mode shapes using RealModes and ComplexModes
5. Reconstruct M, C, K from fitted mode shapes
6. Compare errors
7. Generate and compare IRFs from true and reconstructed systems
8. Compare stability lobes
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bcam.indar import ema
from system_params import Kn
from lobes import plot_lobes

# Output directories
DATA_DIR = Path(__file__).parent / 'data'
IMAGES_DIR = Path(__file__).parent.parent.parent / 'images'
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("MODE SHAPE RECOVERY: PROPORTIONAL DAMPING BASELINE")
print("=" * 70)

# --- Measurement parameters ---
FS = 8192.0        # Hz - sampling frequency
DURATION = 0.5     # s  - measurement duration
dt = 1.0 / FS
ns = int(DURATION * FS)
tt = np.arange(ns) * dt

n_dof = 2
n_out = n_dof  # measure both coordinates
n_in = 1       # single input on coordinate 0

# --- Define modal parameters for proportional system ---
print("\n" + "-" * 70)
print("STEP 1: Create Proportionally Damped System from Modal Parameters")
print("-" * 70)

# Modal parameters (similar frequency range to TMD case)
fn1 = 310.0      # Hz - first natural frequency
fn2 = 345.0      # Hz - second natural frequency
zeta1 = 0.020    # -  modal damping ratio for mode 1
zeta2 = 0.025    # -  modal damping ratio for mode 2

print(f"\nDesired modal parameters:")
print(f"  Mode 1: f = {fn1:.2f} Hz, ζ = {zeta1:.6f}")
print(f"  Mode 2: f = {fn2:.2f} Hz, ζ = {zeta2:.6f}")

# Compute complex poles (continuous time)
wn1 = 2 * np.pi * fn1
wn2 = 2 * np.pi * fn2
lambda1 = -zeta1 * wn1 + 1j * wn1 * np.sqrt(1 - zeta1**2)
lambda2 = -zeta2 * wn2 + 1j * wn2 * np.sqrt(1 - zeta2**2)
nat_freqs_desired = np.array([lambda1, lambda2])

print(f"\nDesired poles (continuous-time):")
for i, lam in enumerate(nat_freqs_desired):
    fn = np.abs(np.imag(lam)) / (2 * np.pi)
    zeta = -np.real(lam) / np.abs(lam)
    print(f"  Mode {i+1}: λ = {lam}, f = {fn:.2f} Hz, ζ = {zeta:.6f}")

# Define REAL mode shapes (this ensures proportional damping)
# Use orthogonal vectors that represent typical structural modes
mode_shapes_desired = np.array([
    [1.0,  0.6],   # Mode 1: both coordinates move in phase
    [0.6, -1.0]    # Mode 2: coordinates move out of phase (orthogonal)
], dtype=float).T  # Shape: (n_dof, n_modes)

# Normalize to unit length
for i in range(mode_shapes_desired.shape[1]):
    mode_shapes_desired[:, i] /= np.linalg.norm(mode_shapes_desired[:, i])

print(f"\nDesired mode shapes (real, normalized):")
print(mode_shapes_desired)

# Apply reduced normalization scaling
mode_shapes_scaled = (
    mode_shapes_desired * np.sqrt(np.imag(nat_freqs_desired))[np.newaxis, :]
)

# Construct M, C, K from modal parameters
print("\nConstructing M, C, K matrices from modal parameters...")
M_true, C_true, K_true = ema.modal_to_system(
    mode_shapes_scaled, nat_freqs_desired
)

print(f"\nTrue M (proportional system) =")
print(M_true)
print(f"\nTrue C (proportional system) =")
print(C_true)
print(f"\nTrue K (proportional system) =")
print(K_true)

# --- Verify proportionality ---
print("\n" + "-" * 70)
print("STEP 2: Verify Proportional Damping")
print("-" * 70)

# Extract modal parameters from constructed system
mode_shapes_check, nat_freqs_check = ema.system_to_modal(M_true, C_true, K_true)

print(f"\nVerification - extracted natural frequencies:")
for i, lam in enumerate(nat_freqs_check):
    fn = np.abs(np.imag(lam)) / (2 * np.pi)
    zeta = -np.real(lam) / np.abs(lam)
    print(f"  Mode {i+1}: f = {fn:.2f} Hz, ζ = {zeta:.6f}")

# Check phase differences (should be 0° or 180° for proportional damping)
# For proportional damping, all components of a mode shape must have the same phase
# This means phase differences between coordinates should be 0° (same sign) or 180° (opposite sign)
mode_phases = np.angle(mode_shapes_check)
phase_diff = np.abs(mode_phases[0, :] - mode_phases[1, :])
phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # wrap to [0, pi]

# Check if phases are close to 0° or 180° (real mode shapes)
is_zero = np.isclose(phase_diff, 0.0, atol=0.01)
is_pi = np.isclose(phase_diff, np.pi, atol=0.01)
is_proportional = np.all(is_zero | is_pi)

print(f"\nProportional damping check:")
print(f"  Phase differences: {np.rad2deg(phase_diff)} degrees")
print(f"  System is {'PROPORTIONAL ✓' if is_proportional else 'NON-PROPORTIONAL ✗'}")

if not is_proportional:
    print("\n⚠ WARNING: System is not proportional! Check construction.")

# --- Compute exact modal parameters (poles & amplitudes ONLY) ---
print("\n" + "-" * 70)
print("STEP 3: Compute Exact Modal Parameters")
print("-" * 70)

# Use the verified modal parameters
nat_freqs_true = nat_freqs_check
mode_shapes_true = mode_shapes_check

# Apply reduced normalization
mode_shapes_norm = mode_shapes_true / np.sqrt(np.imag(nat_freqs_true))[np.newaxis, :]

print(f"\nTrue mode shapes (reduced normalization):")
print(mode_shapes_norm)

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
print("STEP 4: Fit Mode Shapes with Proportional Damping Assumption")
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
print("STEP 5: Fit Mode Shapes without Proportional Assumption")
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
print("STEP 6: Reconstruct M, C, K from Recovered Mode Shapes")
print("-" * 70)

# Convert discrete poles back to continuous
roots_continuous = np.log(poles_discrete) / dt

print("\nReconstruction from RealModes (proportional):")
modes_real_scaled = modes_real * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
M_prop, C_prop, K_prop = ema.modal_to_system(modes_real_scaled, roots_continuous)

print(f"M (proportional) =")
print(M_prop)
print(f"\nC (proportional) =")
print(C_prop)
print(f"\nK (proportional) =")
print(K_prop)

print("\nReconstruction from ComplexModes (non-proportional):")
modes_complex_scaled = (
    modes_complex * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
)
M_nonprop, C_nonprop, K_nonprop = ema.modal_to_system(
    modes_complex_scaled, roots_continuous
)

print(f"M (non-proportional) =")
print(M_nonprop)
print(f"\nC (non-proportional) =")
print(C_nonprop)
print(f"\nK (non-proportional) =")
print(K_nonprop)

# Compute relative errors
print("\n" + "-" * 70)
print("STEP 7: Compute Matrix Reconstruction Errors")
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

print(f"\nExpected result: Both methods should have low errors on proportional system")

# --- Generate IRFs for comparison ---
print("\n" + "-" * 70)
print("STEP 8: Generate IRFs for Time-Domain Comparison")
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

# Plot for coordinate 0
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
ax.set_ylabel('Coordinate 0 IRF [m/N]', fontsize=11)
ax.set_title(
    'Impulse Response Function Comparison',
    fontsize=12,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

# Plot for coordinate 1
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
ax.set_ylabel('Coordinate 1 IRF [m/N]', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

plt.tight_layout()

# Save IRF comparison figure
irf_output = IMAGES_DIR / '04_irf_comparison.jpg'
fig.savefig(irf_output, dpi=300, bbox_inches='tight')
print(f"\nIRF comparison saved to: {irf_output}")

# --- Stability Lobe Comparison ---
print("\n" + "-" * 70)
print("STEP 9: Stability Lobe Comparison")
print("-" * 70)

# Fixed omega range for rich lobe visualization
# Lower speeds show characteristic lobe "pockets" where Omega ≈ omega_c / k
omega_min = 100.0   # rad/s - low speed regime
omega_max = 800.0  # rad/s - up to natural frequency range

print(f"\nStability lobe parameters:")
print(f"  Omega range: [{omega_min:.1f}, {omega_max:.1f}] rad/s")
print(f"  Cutting coefficient Kn: {Kn:.2e} N/m²")

# Create figure with 3 curves
fig, ax = plt.subplots(figsize=(10, 6))

print("\nComputing stability lobes...")
print("  True system...")
_, _, omega_true, h_true = plot_lobes(
    [M_true, C_true, K_true],
    cut_coeff=Kn,
    omega_min=omega_min,
    omega_max=omega_max,
    n_omega=300,
    n_lobes=50,
    beta_step=0.02,
    show_branches=False,
    ax=ax,
    representation='mechanical'
)
ax.lines[-1].set_label('True system')
ax.lines[-1].set_color('black')
ax.lines[-1].set_linewidth(2)

print("  Proportional damping (RealModes)...")
_, _, omega_prop, h_prop = plot_lobes(
    [M_prop, C_prop, K_prop],
    cut_coeff=Kn,
    omega_min=omega_min,
    omega_max=omega_max,
    n_omega=300,
    n_lobes=50,
    beta_step=0.02,
    show_branches=False,
    ax=ax,
    representation='mechanical'
)
ax.lines[-1].set_label('Proportional damping (RealModes)')
ax.lines[-1].set_color('red')
ax.lines[-1].set_linewidth(1.5)
ax.lines[-1].set_linestyle('--')

print("  Non-proportional damping (ComplexModes)...")
_, _, omega_nonprop, h_nonprop = plot_lobes(
    [M_nonprop, C_nonprop, K_nonprop],
    cut_coeff=Kn,
    omega_min=omega_min,
    omega_max=omega_max,
    n_omega=300,
    n_lobes=50,
    beta_step=0.02,
    show_branches=False,
    ax=ax,
    representation='mechanical'
)
ax.lines[-1].set_label('Non-proportional (ComplexModes)')
ax.lines[-1].set_color('blue')
ax.lines[-1].set_linewidth(1.5)
ax.lines[-1].set_linestyle('-.')

# Format plot
ax.set_xlabel(r'Spindle speed $\Omega$ [rad/s]', fontsize=12)
ax.set_ylabel('Chip thickness $h$ [m]', fontsize=12)
ax.set_title(
    'Stability Lobe Diagram: Baseline with Proportional Damping',
    fontsize=13,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Save figure
lobes_output = IMAGES_DIR / '04_stability_lobes.jpg'
fig.savefig(lobes_output, dpi=300, bbox_inches='tight')
print(f"Stability lobes saved to: {lobes_output}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

plt.show()

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nSystem type: {'PROPORTIONAL' if is_proportional else 'NON-PROPORTIONAL'}")
print(f"\nMatrix reconstruction errors:")
print(f"  RealModes (proportional assumption):")
print(f"    M: {M_error_prop*100:.4f}%, C: {C_error_prop*100:.4f}%, "
      f"K: {K_error_prop*100:.4f}%")
print(f"  ComplexModes (non-proportional):")
print(f"    M: {M_error_nonprop*100:.4f}%, C: {C_error_nonprop*100:.4f}%, "
      f"K: {K_error_nonprop*100:.4f}%")

print(f"\n{'✓' if M_error_prop < 0.05 and C_error_prop < 0.05 else '✗'} "
      f"RealModes achieves <5% error on proportional system")
print(f"{'✓' if M_error_nonprop < 0.05 and C_error_nonprop < 0.05 else '✗'} "
      f"ComplexModes also works well (more general)")

print("\n" + "=" * 70)
