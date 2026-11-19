"""Quick debug script to check StablePoles API"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bcam.indar import ema

# Load data
DATA_DIR = Path(__file__).parent / 'data'
data = np.load(DATA_DIR / 'synthetic_measurements.npz')
irf_estimated = np.mean(data['irf_measurements'], axis=0)
fs = float(data['fs'])

# Try StablePoles
sr_aaa = ema.SuperResolution(fs=fs, rational_fitter={'method': 'AAA'})
sp = ema.StablePoles(sr_aaa, max_order=30)
sp.fit(irf_estimated[:, 0, :].T)

# Inspect attributes
print("StablePoles attributes:")
print([attr for attr in dir(sp) if not attr.startswith('_')])

print("\n\nAccessing poles_set_...")
print(f"Type: {type(sp.poles_set_)}")
print(f"Length: {len(sp.poles_set_)}")
print(f"Keys: {list(sp.poles_set_.keys())}")

# Get the last (highest order) SuperResolution result
sr_final = sp.model
print(f"\nmodel attribute type: {type(sr_final)}")
print(f"model attributes: {[a for a in dir(sr_final) if not a.startswith('_')]}")

# Check clusters (stable poles from stabilization diagram)
print(f"\n\nAccessing clusters_...")
print(f"Type: {type(sp.clusters_)}")
print(f"Clusters: {sp.clusters_}")

if len(sp.clusters_) > 0:
    print(f"\nStable poles from clusters (discrete-time):")
    mech_poles_discrete = []
    for i, cluster_dict in enumerate(sp.clusters_):
        # Get pole from highest model order
        max_order = max(cluster_dict.keys())
        pole_discrete = cluster_dict[max_order]
        mech_poles_discrete.append(pole_discrete)
        
        # Convert to continuous time: λ_continuous = ln(λ_discrete) / dt
        dt = 1.0 / fs
        pole_continuous = np.log(pole_discrete) / dt
        
        f_disc = np.abs(np.imag(pole_discrete)) / (2 * np.pi)
        f_cont = np.abs(np.imag(pole_continuous)) / (2 * np.pi)
        z_cont = -np.real(pole_continuous) / np.abs(pole_continuous)
        
        print(f"  Cluster {i}:")
        print(f"    Discrete: λ_z = {pole_discrete}")
        print(f"    Continuous: f = {f_cont:.2f} Hz, ζ = {z_cont:.4f}")
    
    print(f"\n✓ Found {len(mech_poles_discrete)} stable mechanical poles")
    print(f"  USAGE: Extract from sp.clusters_, then optionally refine with VF")
