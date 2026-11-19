"""
Compare proportional vs non-proportional damping assumptions using EXACT modal parameters.

This script isolates the effect of the proportional damping assumption by using
exact poles and amplitudes (no identification errors, no noise). The only source
of error is the mathematical constraint that mode shapes must be real.

Steps:
1. Compute exact modal parameters from true M, C, K
2. Fit RealModes (proportional assumption) with exact poles/amplitudes
3. Fit ComplexModes (non-proportional) with exact poles/amplitudes
4. Reconstruct M, C, K from both fitted mode shapes
5. Compare errors and stability lobes
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
print("DAMPING ASSUMPTION COMPARISON (EXACT POLES & AMPLITUDES)")
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

# --- Generate exact modal parameters ---
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

# Compute exact amplitudes (residues)
amps_true = ema.mode_to_amps(mode_shapes_norm, n_out, n_in)

print(f"\nTrue amplitudes (tensor shape: {amps_true.shape}):")
for i_out in range(n_out):
    for i_in in range(n_in):
        print(f"  Out {i_out}, In {i_in}: {amps_true[i_out, i_in, :]}")

# Generate noiseless IRF
print("\nGenerating noiseless IRF...")
kernel_true = ema.Kernel(nat_freqs_true, amps_true)
irf_true = kernel_true(tt)
print(f"IRF shape: {irf_true.shape}")

# Convert to discrete-time poles for fitting
poles_discrete = np.exp(nat_freqs_true * dt)
print(f"\nDiscrete-time poles:")
for i, p in enumerate(poles_discrete):
    print(f"  Mode {i+1}: {p}")

# --- Fit RealModes (proportional damping assumption) ---
print("\n" + "-" * 70)
print("STEP 2: Fit RealModes (Proportional Damping Assumption)")
print("-" * 70)

model_real = ema.RealModes(
    poles=poles_discrete,
    amps=amps_true,
    fs=FS,
    ns=ns,
    response='a'
)

print("Starting optimization with exact poles and amplitudes...")
model_real.fit(options_ncg={'disp': 2, 'gtol': 1e-6})

print(f"\nOptimization complete:")
print(f"  Success: {model_real.success_}")

modes_real = model_real.modes_
print(f"\nFitted mode shapes (real, proportional assumption):")
print(modes_real)

# --- Fit ComplexModes (non-proportional damping) ---
print("\n" + "-" * 70)
print("STEP 3: Fit ComplexModes (Non-Proportional Damping)")
print("-" * 70)

model_complex = ema.ComplexModes(
    poles=poles_discrete,
    coords=np.arange(n_out),
    amps=amps_true,
    fs=FS,
    ns=ns,
    response='a'
)

print("Starting optimization with exact poles and amplitudes...")
print("Using RealModes result as initial guess...")
model_complex.fit(
    x0=modes_real,
    maxiter=100,
    options={'verbose': 2, 'gtol': 1e-6, 'xtol': 1e-8}
)

print(f"\nOptimization complete")

modes_complex = model_complex.modes_
print(f"\nFitted mode shapes (complex, non-proportional):")
print(modes_complex)

# Compare with true mode shapes
print(f"\nTrue mode shapes (for comparison):")
print(mode_shapes_norm)

# --- Reconstruct M, C, K matrices ---
print("\n" + "-" * 70)
print("STEP 4: Reconstruct M, C, K Matrices")
print("-" * 70)

# Convert discrete poles back to continuous
roots_continuous = np.log(poles_discrete) / dt

print("\nReconstruction from RealModes (proportional):")
# Apply reduced normalization
modes_real_norm = modes_real * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
M_prop, C_prop, K_prop = ema.modal_to_system(modes_real_norm, roots_continuous)

print(f"M (proportional) =\n{M_prop}")
print(f"\nC (proportional) =\n{C_prop}")
print(f"\nK (proportional) =\n{K_prop}")

print("\nReconstruction from ComplexModes (non-proportional):")
# Apply reduced normalization
modes_complex_norm = modes_complex * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
M_nonprop, C_nonprop, K_nonprop = ema.modal_to_system(
    modes_complex_norm, roots_continuous
)

print(f"M (non-proportional) =\n{M_nonprop}")
print(f"\nC (non-proportional) =\n{C_nonprop}")
print(f"\nK (non-proportional) =\n{K_nonprop}")

# Compute relative errors
print("\n" + "-" * 70)
print("STEP 5: Compute Errors")
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

# --- Stability Lobe Comparison ---
print("\n" + "-" * 70)
print("STEP 6: Stability Lobe Comparison")
print("-" * 70)

# Determine omega range based on natural frequencies
fn_avg = np.mean([np.abs(np.imag(lam)) / (2*np.pi) for lam in nat_freqs_true])
omega_min = 2 * np.pi * fn_avg * 0.5  # 50% of average natural frequency
omega_max = 2 * np.pi * fn_avg * 1.5  # 150% of average natural frequency

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
    n_omega=400,
    n_lobes=20,
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
    n_omega=400,
    n_lobes=20,
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
    n_omega=400,
    n_lobes=20,
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
    'Stability Lobe Diagram: Effect of Proportional Damping Assumption',
    fontsize=13,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Save figure
output_path = IMAGES_DIR / '03_damping_comparison.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved stability lobe comparison to: {output_path}")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
This comparison uses EXACT poles and amplitudes (no identification errors).
The only difference is the mathematical constraint on mode shapes:
  - RealModes: forces mode shapes to be real (proportional damping)
  - ComplexModes: allows complex mode shapes (non-proportional damping)

Results:
  - RealModes C error: {C_error_prop*100:.4f}%
  - ComplexModes C error: {C_error_nonprop*100:.4f}%
  
The proportional damping assumption introduces significant error in the
damping matrix C, which affects the stability lobe predictions.

For this system (non-proportional damping), ComplexModes is essential
for accurate stability analysis.
""")

plt.show()

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
