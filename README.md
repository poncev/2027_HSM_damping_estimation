# Refined Damping Estimation for Machining Structures

This repository contains the computational tools and analysis scripts supporting research on the impact of damping assumptions in chatter stability predictions for turning operations. The work is being prepared for presentation at the [International Congress on High Speed Machining](https://hsmcongress.com/).

## Overview

Chatter stability in machining is typically predicted using stability lobe diagrams, which require accurate modal parameters of the tool-workpiece system. A common simplifying assumption in experimental modal analysis is that the system has **proportional damping** (i.e., mode shapes are real-valued). However, many practical systems—particularly those with tuned mass dampers (TMDs) or other dynamic absorbers—exhibit **non-proportional damping** where mode shapes have complex components.

This repository investigates:
- **How the proportional damping assumption affects stability predictions** when applied to systems with inherently non-proportional damping
- **Quantitative comparison** of stability lobes computed using different modal identification methods

The analysis focuses on a **2-DOF model** representing a turning operation with a Tuned Mass Damper.

## Repository Structure

```
.
├── src/
│   ├── lobes.py                      # Core stability lobe computation
│   ├── sbx.py                        # Sandbox/experimental functions
│   ├── test_lobes.py                 # Unit tests for stability functions
│   └── example_paper/
│       ├── system_params.py          # 2-DOF TMD system definition
│       ├── 03_compare_damping_assumptions.py    # Non-proportional case analysis
│       ├── 04_proportional_baseline.py          # Proportional validation case
│       ├── 05_generate_paper_figures.py         # Figure generation for publication
│       └── data/
└── README.md                         # This file
```

## Installation

### Requirements

This project uses Python with the following key dependencies:
- `bcam-indar` (v.1.0.2): Modal analysis and system identification; see [documentation](https://indar.readthedocs.io/en/latest/).
- `numpy`.
- `scipy`.
- `matplotlib`.

The recommended approach is to use `mamba` (or `conda`) to create a dedicated environment.

## Usage

### Running the Analysis Scripts

The main analysis scripts are designed to be run sequentially:

```bash
cd src/example_paper/

# 1. Compare damping assumptions on non-proportional system
python 03_compare_damping_assumptions.py

# 2. Validate on proportional baseline system
python 04_proportional_baseline.py

# 3. Generate FRFs for publication
python 05_generate_paper_figures.py
```

### Script Descriptions

**`03_compare_damping_assumptions.py`**
- Uses the 2-DOF TMD system (non-proportional damping)
- Fits both RealModes (assumes proportional) and ComplexModes
- Reconstructs M, C, K matrices from fitted parameters
- Computes stability lobes at low spindle speeds (50-600 rad/s)
- Outputs: IRF comparison and stability lobe diagrams

**`04_proportional_baseline.py`**
- Creates a truly proportional 2-DOF system for validation
- Verifies that RealModes achieves near-zero error when assumption is valid
- Confirms ComplexModes works correctly as a more general method
- Outputs: Validation plots and proportional baseline lobes

**`05_generate_paper_figures.py`**
- Plot FRFs computed assuming or not proportional damping

### Core API

The `lobes.py` module provides the fundamental stability computation functions:

```python
from lobes import plot_lobes

# Compute stability diagram
fig, ax, omega_grid, h_lim = plot_lobes(
    M, C, K,              # System matrices
    cut_coeff, # Specific cutting force coefficient
    omega_min, omega_max, # Plotting range of spindle speed
    n_lobes=10, beta_step=0.05, # upper limit and step size in frequency
)
```

---

**Last Updated:** November 2026  
**Conference:** HSM 2027  
**Research Area:** Machining Dynamics, Chatter Stability, Modal Analysis
