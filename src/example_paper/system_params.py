"""
System parameters for boring bar + TMD example.
Based on Yang, Munoa & Altintas (2010) and Den Hartog optimal tuning.
"""
import numpy as np

# --- Primary structure (boring bar) ---
m1 = 2.0         # kg      modal mass
fn1 = 350.0      # Hz      natural frequency (bare tool, no TMD)
zeta1 = 0.025    # —       structural damping ratio
wn1 = 2 * np.pi * fn1   # rad/s
k1 = m1 * wn1**2         # N/m   ~9.67 MN/m
c1 = 2 * zeta1 * m1 * wn1  # N·s/m

# --- Den Hartog optimal TMD (mass ratio mu = 0.05) ---
mu = 0.05
m2 = mu * m1                                          # kg  = 0.1 kg
f_opt = 1.0 / (1.0 + mu)                              # optimal frequency ratio
zeta2_opt = np.sqrt(3 * mu / (8 * (1.0 + mu)**3))    # optimal damping ratio
wn2 = f_opt * wn1
k2 = m2 * wn2**2
c2 = 2 * zeta2_opt * m2 * wn2

# --- Assembled system matrices ---
M = np.array([
    [m1,  0.0],
    [0.0, m2]
])

C = np.array([
    [c1 + c2, -c2],
    [-c2,      c2]
])

K = np.array([
    [k1 + k2, -k2],
    [-k2,      k2]
])

# System as list [M, C, K]
SYSTEM_TRUE = [M, C, K]

# --- Cutting force coefficient (Inconel 718) ---
Kn = 1.2e9  # N/m²  (= 1200 N/mm²)

# --- Measurement parameters ---
FS = 8192.0        # Hz - sampling frequency (power of 2 for FFT efficiency)
DURATION = 0.5     # s  - measurement duration
N_MEASUREMENTS = 5 # number of repeated hammer impacts

# Signal should decay to ~1% in this time:
# For zeta=0.025, fn=350 Hz: tau ≈ 0.018 s, 4*tau ≈ 0.072 s (1% amplitude)
# 0.5 s gives plenty of room for decay

# --- Noise parameters ---
SNR_DB = 40.0  # dB - Signal-to-Noise Ratio

# --- Random seed for reproducibility ---
SEED_MASTER = 42

# --- Coordinate labels ---
COORD_LABELS = ['Tool tip', 'TMD mass']
