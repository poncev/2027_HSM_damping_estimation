"""
Simulate synthetic hammer test data for boring bar + TMD system.

This script:
1. Converts the known M, C, K matrices to modal representation
2. Generates impulse response functions (IRFs) for acceleration response
3. Adds realistic noise based on SNR specification
4. Saves synthetic measurements in UFF format and as NumPy arrays
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bcam.indar import ema
from system_params import (
    SYSTEM_TRUE, FS, DURATION, N_MEASUREMENTS, SNR_DB, SEED_MASTER, COORD_LABELS
)

# Output directories
DATA_DIR = Path(__file__).parent / 'data'
IMAGES_DIR = Path(__file__).parent.parent.parent / 'images'
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("HAMMER TEST SIMULATION")
print("=" * 70)

# --- Parameters ---
M, C, K = SYSTEM_TRUE
dt = 1.0 / FS
ns = int(DURATION * FS)  # number of samples
tt = np.arange(ns) * dt

n_dof = M.shape[0]
n_out = n_dof  # measure both coordinates
n_in = 1       # single hammer impact on coordinate 1

print(f"\nSystem: {n_dof}-DoF boring bar + TMD")
print(f"Sampling: {FS} Hz, duration: {DURATION} s ({ns} samples)")
print(f"Measurements: {N_MEASUREMENTS} repeated impacts")
print(f"SNR: {SNR_DB} dB")
print(f"Random seed: {SEED_MASTER}")

# --- Convert system to modal representation ---
print("\n" + "-" * 70)
print("Converting to modal representation...")

# System to modal: returns normalized mode shapes and natural frequencies
mode_shapes, nat_freqs = ema.system_to_modal(M, C, K)

print(f"\nNatural frequencies (complex):")
for i, lam in enumerate(nat_freqs):
    fn = np.abs(np.imag(lam)) / (2 * np.pi)
    zeta = -np.real(lam) / np.abs(lam)
    print(f"  Mode {i+1}: f = {fn:.2f} Hz, ζ = {zeta:.4f}")

# Apply reduced normalization (required by indar convention)
mode_shapes_norm = mode_shapes / np.sqrt(np.imag(nat_freqs))[np.newaxis, :]

print(f"\nMode shapes (reduced normalization):")
print(mode_shapes_norm)

# Check if damping is proportional
# For proportional damping, mode shapes should be real (or have same phase)
mode_phases = np.angle(mode_shapes)
phase_diff = np.abs(mode_phases[0, :] - mode_phases[1, :])
phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # wrap to [0, pi]
is_proportional = np.allclose(phase_diff, 0.0, atol=0.01)
print(f"\nProportional damping check:")
print(f"  Phase differences: {np.rad2deg(phase_diff)}")
print(f"  System is {'PROPORTIONAL' if is_proportional else 'NON-PROPORTIONAL'}")

# --- Compute amplitudes (residues) ---
print("\n" + "-" * 70)
print("Computing amplitudes (residues)...")

# For measurement: input at coord 0 (tool tip), output at coords 0, 1
# Amplitudes tensor shape: (n_out, n_in, n_modes)
amps = ema.mode_to_amps(mode_shapes_norm, n_out, n_in)

print(f"Amplitudes shape: {amps.shape}")
print(f"Amplitudes (output, input, mode):")
for i_out in range(n_out):
    for i_in in range(n_in):
        print(f"  Out {i_out}, In {i_in}: {amps[i_out, i_in, :]}")

# --- Generate true IRF (noise-free) ---
print("\n" + "-" * 70)
print("Generating true impulse response...")

kernel = ema.Kernel(nat_freqs, amps)
irf_true = kernel(tt)

print(f"IRF shape: {irf_true.shape} (n_out, n_in, ns)")
print(f"IRF peak values: {np.max(np.abs(irf_true), axis=2)}")

# --- Add noise with specified SNR ---
print("\n" + "-" * 70)
print("Adding measurement noise...")

# Initialize random generator with spawn for independence
rng_master = np.random.default_rng(SEED_MASTER)
rngs = rng_master.spawn(N_MEASUREMENTS)

# Storage for measurements
irf_measurements = np.zeros((N_MEASUREMENTS, n_out, n_in, ns))

for i_meas in range(N_MEASUREMENTS):
    rng = rngs[i_meas]
    
    # Compute noise level from SNR
    # SNR_dB = 20*log10(A_signal / A_noise)
    # A_noise = A_signal / 10^(SNR_dB/20)
    signal_power = np.mean(irf_true**2, axis=2, keepdims=True)
    signal_rms = np.sqrt(signal_power)
    noise_rms = signal_rms / (10 ** (SNR_DB / 20))
    
    # Generate Gaussian noise
    noise = rng.normal(0, 1, irf_true.shape) * noise_rms
    
    # Add noise to signal
    irf_measurements[i_meas] = irf_true + noise

print(f"Measurements shape: {irf_measurements.shape}")
print(f"  (n_measurements, n_out, n_in, ns)")
print(f"\nNoise RMS levels:")
for i_out in range(n_out):
    print(f"  Output {i_out}: {noise_rms[i_out, 0, 0]:.2e}")

# --- Save data ---
print("\n" + "-" * 70)
print("Saving data...")

# Save as NumPy arrays (more flexible for our workflow)
np.savez(
    DATA_DIR / 'synthetic_measurements.npz',
    irf_true=irf_true,
    irf_measurements=irf_measurements,
    tt=tt,
    dt=dt,
    fs=FS,
    nat_freqs=nat_freqs,
    mode_shapes=mode_shapes,
    mode_shapes_norm=mode_shapes_norm,
    amps=amps,
    M=M, C=C, K=K,
    n_measurements=N_MEASUREMENTS,
    snr_db=SNR_DB,
    seed=SEED_MASTER,
    coord_labels=COORD_LABELS,
)

print(f"✓ Saved to: {DATA_DIR / 'synthetic_measurements.npz'}")

# --- Visualization ---
print("\n" + "-" * 70)
print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: True IRF for both outputs
ax = axes[0, 0]
for i_out in range(n_out):
    ax.plot(tt * 1000, irf_true[i_out, 0, :], 
            label=f'{COORD_LABELS[i_out]}', linewidth=1.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Acceleration [m/s²/N]')
ax.set_title('True Impulse Response (noise-free)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Single noisy measurement vs true
ax = axes[0, 1]
for i_out in range(n_out):
    ax.plot(tt * 1000, irf_measurements[0, i_out, 0, :], 
            alpha=0.7, linewidth=1, label=f'{COORD_LABELS[i_out]} (noisy)')
    ax.plot(tt * 1000, irf_true[i_out, 0, :], 
            '--', linewidth=1.5, alpha=0.8, label=f'{COORD_LABELS[i_out]} (true)')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Acceleration [m/s²/N]')
ax.set_title(f'Measurement #1 (SNR = {SNR_DB} dB)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: All measurements for output 0
ax = axes[1, 0]
for i_meas in range(N_MEASUREMENTS):
    ax.plot(tt * 1000, irf_measurements[i_meas, 0, 0, :], 
            alpha=0.5, linewidth=0.8, label=f'Meas. {i_meas+1}')
ax.plot(tt * 1000, irf_true[0, 0, :], 
        'k--', linewidth=2, label='True', zorder=10)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Acceleration [m/s²/N]')
ax.set_title(f'{COORD_LABELS[0]}: All measurements')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Frequency domain (magnitude)
ax = axes[1, 1]
freq = np.fft.rfftfreq(ns, dt)
for i_out in range(n_out):
    frf_true = np.fft.rfft(irf_true[i_out, 0, :])
    ax.semilogy(freq, np.abs(frf_true), 
                linewidth=1.5, label=f'{COORD_LABELS[i_out]}')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|FRF| [m/s²/N]')
ax.set_title('Frequency Response (true)')
ax.set_xlim(0, 1000)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / '01_synthetic_measurements.png', dpi=150)
print(f"✓ Saved figure: {IMAGES_DIR / '01_synthetic_measurements.png'}")

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
