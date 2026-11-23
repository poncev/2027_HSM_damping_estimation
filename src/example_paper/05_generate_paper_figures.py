"""
Generate FRF comparison figures for the HSM paper.

This script generates high-quality publication-ready figures comparing:
- True system FRF (from known M, C, K)
- ComplexModes reconstruction FRF (non-proportional damping)
- RealModes reconstruction FRF (proportional damping assumption)

The goal is to show the error introduced by the proportional damping assumption,
which should be most visible in the TMD coordinate (coordinate 2).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bcam.indar import ema
from system_params import SYSTEM_TRUE, FS, DURATION, COORD_LABELS

# Output directories
DATA_DIR = Path(__file__).parent / 'data'
IMAGES_DIR = Path(__file__).parent.parent.parent / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("GENERATE FRF COMPARISON FIGURES FOR HSM PAPER")
print("=" * 70)

# --- Load synthetic measurements ---
print("\nLoading data from synthetic_measurements.npz...")
data = np.load(DATA_DIR / 'synthetic_measurements.npz')

M_true = data['M']
C_true = data['C']
K_true = data['K']
nat_freqs_true = data['nat_freqs']
mode_shapes_norm = data['mode_shapes_norm']
amps_true = data['amps']
fs = float(data['fs'])
dt = 1.0 / fs
ns = int(DURATION * fs)
tt = np.arange(ns) * dt

n_dof = M_true.shape[0]
n_out = n_dof
n_in = 1

print(f"System: {n_dof}-DoF")
print(f"Sampling frequency: {fs} Hz")
print(f"Duration: {DURATION} s")
print(f"Number of samples: {ns}")

# --- Fit mode shapes using RealModes (proportional assumption) ---
print("\n" + "-" * 70)
print("FITTING MODE SHAPES")
print("-" * 70)

# Convert to discrete-time poles for fitting
poles_discrete = np.exp(nat_freqs_true * dt)

print("\nFitting RealModes (proportional assumption)...")
model_real = ema.RealModes(
    poles=poles_discrete,
    amps=amps_true,
    fs=fs,
    ns=ns,
    response='a'
)

model_real.fit(options_ncg={'disp': 0, 'gtol': 1e-6})

if not model_real.success_:
    print("Warning: RealModes fitting did not converge!")
else:
    print("RealModes fitting successful")

modes_real = model_real.modes_

# --- Fit mode shapes using ComplexModes (non-proportional) ---
print("\nFitting ComplexModes (non-proportional)...")
model_complex = ema.ComplexModes(
    poles=poles_discrete,
    coords=np.arange(n_out),
    amps=amps_true,
    fs=fs,
    ns=ns,
    response='a'
)

model_complex.fit(
    x0=modes_real,
    maxiter=100,
    options={'verbose': 0, 'gtol': 1e-6, 'xtol': 1e-8}
)

print("ComplexModes fitting complete")
modes_complex = model_complex.modes_

# --- Reconstruct M, C, K matrices ---
print("\n" + "-" * 70)
print("RECONSTRUCTING SYSTEM MATRICES")
print("-" * 70)

# Convert discrete poles back to continuous
roots_continuous = np.log(poles_discrete) / dt

print("\nReconstruction from RealModes (proportional)...")
modes_real_scaled = modes_real * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
M_prop, C_prop, K_prop = ema.modal_to_system(modes_real_scaled, roots_continuous)

print("Reconstruction from ComplexModes (non-proportional)...")
modes_complex_scaled = (
    modes_complex * np.sqrt(np.imag(roots_continuous))[np.newaxis, :]
)
M_nonprop, C_nonprop, K_nonprop = ema.modal_to_system(
    modes_complex_scaled, roots_continuous
)

# --- Compute FRFs ---
print("\n" + "-" * 70)
print("COMPUTING FREQUENCY RESPONSE FUNCTIONS")
print("-" * 70)

# Frequency range for FRF: 200-500 Hz
freq_min = 200.0  # Hz
freq_max = 500.0  # Hz
n_freq = 1000

freq_range = np.linspace(freq_min, freq_max, n_freq)
omega = 2 * np.pi * freq_range

print(f"\nFrequency range: {freq_min} - {freq_max} Hz")
print(f"Number of frequency points: {n_freq}")

def compute_frf(M, C, K, omega):
    """
    Compute frequency response function H(ω) = (K - ω²M + jωC)^(-1)
    
    Parameters
    ----------
    M, C, K : ndarray (n, n)
        System matrices
    omega : ndarray (n_freq,)
        Angular frequencies [rad/s]
    
    Returns
    -------
    H : ndarray (n_dof, n_dof, n_freq)
        FRF tensor [out, in, freq]
    """
    n_dof = M.shape[0]
    n_freq = len(omega)
    H = np.zeros((n_dof, n_dof, n_freq), dtype=complex)
    
    for i, w in enumerate(omega):
        # Dynamic stiffness matrix: D(ω) = K - ω²M + jωC
        D = K - w**2 * M + 1j * w * C
        # FRF: H(ω) = D(ω)^(-1)
        H[:, :, i] = np.linalg.inv(D)
    
    return H

print("\nComputing FRFs...")
print("  True system...")
H_true = compute_frf(M_true, C_true, K_true, omega)

print("  Proportional system...")
H_prop = compute_frf(M_prop, C_prop, K_prop, omega)

print("  Non-proportional system...")
H_nonprop = compute_frf(M_nonprop, C_nonprop, K_nonprop, omega)

# --- Generate Figure: FRF Comparison ---
print("\n" + "-" * 70)
print("GENERATING FRF COMPARISON FIGURE")
print("-" * 70)

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# Input coordinate (always 0, since we have single input on coordinate 0)
i_in = 0

for i_out in range(n_out):
    ax = axes[i_out]
    
    # Extract FRFs for this output coordinate
    H_true_out = H_true[i_out, i_in, :]
    H_prop_out = H_prop[i_out, i_in, :]
    H_nonprop_out = H_nonprop[i_out, i_in, :]
    
    # Convert to magnitude in dB
    mag_true_db = 20 * np.log10(np.abs(H_true_out))
    mag_prop_db = 20 * np.log10(np.abs(H_prop_out))
    mag_nonprop_db = 20 * np.log10(np.abs(H_nonprop_out))
    
    # Plot - solid lines for estimations, dashed for true (on top)
    # Use orange for ComplexModes for better contrast with green True
    ax.plot(
        freq_range, mag_nonprop_db, '-',
        color='#ff7f0e', linewidth=2.5,
        label='ComplexModes (non-proportional)', zorder=1
    )
    ax.plot(
        freq_range, mag_prop_db, '-',
        color='#d62728', linewidth=2.5,
        label='RealModes (proportional)', zorder=2
    )
    ax.plot(
        freq_range, mag_true_db, '--',
        color='#2ca02c', linewidth=3.0,
        label='True system', zorder=3
    )
    
    # Labels and formatting
    ax.set_ylabel(f'{COORD_LABELS[i_out]} FRF [dB re 1 m/N]', fontsize=18)
    ax.legend(loc='best', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Add minor grid
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

# X-axis label only on bottom plot
axes[1].set_xlabel('Frequency [Hz]', fontsize=18)

plt.tight_layout()

# Save figure
output_path = IMAGES_DIR / 'paper_frf_comparison.jpg'
fig.savefig(output_path, dpi=350, bbox_inches='tight', format='jpeg')
print(f"\nFigure saved to: {output_path}")

# --- Compute and report errors ---
print("\n" + "-" * 70)
print("FRF ERROR ANALYSIS")
print("-" * 70)

# Compute RMS errors in dB
for i_out in range(n_out):
    H_true_out = H_true[i_out, i_in, :]
    H_prop_out = H_prop[i_out, i_in, :]
    H_nonprop_out = H_nonprop[i_out, i_in, :]
    
    # Convert to dB
    mag_true_db = 20 * np.log10(np.abs(H_true_out))
    mag_prop_db = 20 * np.log10(np.abs(H_prop_out))
    mag_nonprop_db = 20 * np.log10(np.abs(H_nonprop_out))
    
    # RMS error
    rms_error_prop = np.sqrt(np.mean((mag_prop_db - mag_true_db)**2))
    rms_error_nonprop = np.sqrt(np.mean((mag_nonprop_db - mag_true_db)**2))
    
    # Maximum absolute error
    max_error_prop = np.max(np.abs(mag_prop_db - mag_true_db))
    max_error_nonprop = np.max(np.abs(mag_nonprop_db - mag_true_db))
    
    print(f"\n{COORD_LABELS[i_out]}:")
    print(f"  RMS error (Proportional):     {rms_error_prop:.3f} dB")
    print(f"  RMS error (Non-proportional): {rms_error_nonprop:.3f} dB")
    print(f"  Max error (Proportional):     {max_error_prop:.3f} dB")
    print(f"  Max error (Non-proportional): {max_error_nonprop:.3f} dB")
    print(f"  Error reduction: {(1 - rms_error_nonprop/rms_error_prop)*100:.1f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
