# Project Status: Proportional vs Non-Proportional Damping Analysis

**Last Updated:** 2026-11-19  
**Branch:** develop (merged from ft_tmd)

---

## ✅ Completed Work

### Code Implementation

**Core Scripts:**
- `src/example_paper/system_params.py`: 2-DoF boring bar + TMD system definition
  - Den Hartog optimal tuning (mu=0.05)
  - Non-proportional damping (phase differences: 26° and 148°)
  - Cutting coefficient Kn=1.2e9 N/m² (Inconel 718)

- `src/example_paper/03_compare_damping_assumptions.py`: **Non-proportional case (TMD)**
  - Uses exact poles and amplitudes (no identification errors)
  - Fits RealModes (proportional assumption) and ComplexModes
  - Reconstructs M, C, K matrices from fitted mode shapes
  - Compares IRFs and stability lobes
  - **Key Result:** 90.28% C error with proportional assumption

- `src/example_paper/04_proportional_baseline.py`: **Proportional validation**
  - Creates truly proportional 2-DoF system using `ema.modal_to_system`
  - Validates that RealModes works correctly (0.0000% error) when damping IS proportional
  - ComplexModes also achieves 0.0000% error (more general method)
  - **Confirms methodology is sound**

**Output Files:**
- `images/03_irf_comparison.jpg`: Time-domain IRF comparison (non-proportional case)
- `images/03_stability_lobes.jpg`: Stability lobe diagram (non-proportional case)
- `images/04_irf_comparison.jpg`: Time-domain IRF comparison (proportional baseline)
- `images/04_stability_lobes.jpg`: Stability lobe diagram (proportional baseline)

### Key Findings

| System Type | Phase Diffs | RealModes Errors | ComplexModes Errors |
|-------------|-------------|------------------|---------------------|
| **Proportional** (Script 04) | 0°, 180° | M: 0%, C: 0%, K: 0% | M: 0%, C: 0%, K: 0% |
| **Non-proportional** (Script 03) | 26°, 148° | M: 21.76%, C: **90.28%**, K: 26.13% | M: 0.0004%, C: 0.003%, K: 0.0004% |

**Interpretation:**
- Proportional assumption works when valid (script 04)
- Proportional assumption fails significantly on TMD system (script 03)
- ComplexModes handles both cases accurately
- **C matrix error of 90% seems surprisingly high and warrants investigation**

### Git History
- 10 commits on ft_tmd branch (merged to develop)
- Deleted identification workflow scripts (01, 02, debug) - kept in history
- Clean codebase: only essential scripts remain

---

## 🚧 Current Issues / Questions

### Priority 0: Investigate 90% C Matrix Error

**Question:** Is 90% C error physically realistic or is there a numerical issue?

**Possible causes:**
1. TMD creates strong non-proportionality (phase diffs 26° and 148°)
2. RealModes fundamentally constrained to real mode shapes
3. Frobenius norm may not be the best error metric for damping matrices
4. Potential numerical conditioning issues
5. Scaling or normalization artifacts

**Proposed investigation:**
- Compare element-wise errors vs Frobenius norm
- Check if stability lobes show proportional visual differences
- Verify FRF magnitude differences match C error
- Review matrix conditioning numbers
- Compare relative vs absolute errors

**Status:** Flagged for investigation before publication

---

## 📋 Next Steps

### Immediate: Stability Lobe Analysis

**Goal:** Understand what 90% C error means practically for stability predictions

**Tasks:**
1. Quantify stability lobe differences between True, Proportional, and Non-proportional systems
   - Compute critical depth differences at same spindle speeds
   - Measure frequency shifts in lobe peaks
   - Calculate area differences in stable regions
   
2. Create detailed comparison plots:
   - Overlay plots with error bands
   - Difference plots (Proportional - True, ComplexModes - True)
   - Percentage error maps
   
3. Determine if 90% C error correlates with practical stability prediction errors
   - Is stability prediction error also ~90%?
   - Or does eigenvalue problem dampen the effect?

4. Document findings for paper:
   - If errors are large → strong argument for ComplexModes
   - If errors are small → need to explain why C error doesn't translate

**Note:** Opening new chat session with fresh context for this analysis

### Paper Writing

**Current draft:** `hsm_paper.md` (~6000 chars/page, 10 page limit, targeting <6 pages)

**Remaining sections:**
- Results section with figures from scripts 03 and 04
- Discussion of 90% C error and stability implications
- Conclusions
- Equation formatting strategy (uncomfortable in Word)

**Status:** Code complete, ready for writing phase

---

## 💡 Optional Future Improvements

### Code Enhancements (Low Priority)

**1. More Realistic Simulation:**
- Simulate force-response pairs instead of adding noise to IRF
- Use `LTIKernel` with cross-validation
- Matches real experimental workflows

**2. LTIKernel Cross-Validation:**
- Use `sklearn.model_selection.GridSearchCV`
- Optimize hyperparameters (kernel bandwidth, regularization)

**3. Vector Fitting Refinement:**
- Refine AAA poles with Vector Fitting
- Compare pole accuracy

**4. Multiple SNR Levels:**
- Test robustness at SNR = [20, 30, 40, 50] dB
- Plot error metrics vs SNR

**5. Residual Pole Handling:**
- Demonstrate proper separation of mechanical vs spurious poles
- Handle messier, more realistic data

**Note:** These improvements would make the code more rigorous but are not critical for paper deadline. Current implementation successfully demonstrates the core objective: showing differences between proportional and non-proportional damping assumptions.

---

## 📁 File Structure

```
DataAnalysis/2027_HSM_event/
├── hsm_paper.md                    # Paper draft
├── refs.bib                        # Bibliography
├── STATUS.md                       # This file
├── previous_agent.md               # Background info
├── local.md                        # Local paths
├── .github/copilot-instructions.md # Multi-agent workflow
├── src/
│   ├── lobes.py                   # Stability lobe computation
│   ├── sbx.py                     # (purpose unknown)
│   ├── test_lobes.py              # Tests
│   └── example_paper/
│       ├── system_params.py       # TMD system definition
│       ├── 03_compare_damping_assumptions.py
│       └── 04_proportional_baseline.py
└── images/                        # Generated figures (gitignored)
```

---

## 🔧 Development Environment

- **Python:** 3.14 (mamba env: `resonance`)
- **Key packages:** bcam.indar, numpy, scipy, matplotlib
- **Channels:** conda-forge, defaults
- **Git workflow:** Feature branches → develop (no direct commits to main)

---

## 📝 Notes for Next Session

1. **Main focus:** Stability lobe analysis to understand practical impact of 90% C error
2. **Key question:** Does matrix reconstruction error translate to stability prediction error?
3. **Files to review:** 
   - `images/03_stability_lobes.jpg` (visual inspection)
   - `src/example_paper/03_compare_damping_assumptions.py` (extract lobe data for quantitative comparison)
4. **Context preserved in:** This STATUS.md file + git history
