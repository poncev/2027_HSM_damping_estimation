# Optional Improvements (If Time Permits)

This file tracks improvements that would make the code more rigorous but are not critical for the paper deadline.

## Priority 1: More Realistic Simulation

**Current:** We add noise directly to the IRF (simplified approach).

**Proper approach:**
1. Simulate actual hammer hits with realistic force profiles (half-sine, exponential decay)
2. Simulate force sensor measurements (with noise)
3. Simulate acceleration sensor measurements (with noise)
4. Use `LTIKernel` with cross-validation to estimate IRF from force-response pairs
5. This matches real experimental workflows and validates the LTIKernel method

**Reference:** See examples you provided earlier (force-response simulation)

**Benefit:** More convincing validation, matches experimental procedures

## Priority 2: LTIKernel Cross-Validation

**Current:** Simple averaging of measurements

**Proper approach:**
- Use `sklearn.model_selection.GridSearchCV` with `bcam.indar.ema.LTIKernel`
- Cross-validate hyperparameters (kernel bandwidth, regularization)
- Select optimal kernel based on validation score
- More rigorous IRF estimation from noisy data

**Benefit:** Demonstrates proper statistical methodology

## Priority 3: Vector Fitting Refinement

**Current:** Use AAA poles directly (works well for synthetic data)

**Could add:**
- Vector Fitting to refine AAA initial estimates
- Compare pole accuracy between AAA-only vs AAA+VF
- May not add much for synthetic data but more complete

## Priority 4: Multiple SNR Levels

**Current:** Single SNR (40 dB)

**Could add:**
- Test identification robustness at SNR = [20, 30, 40, 50] dB
- Plot error metrics vs SNR
- Demonstrates method sensitivity to noise

**Benefit:** More comprehensive validation

## Priority 5: Residual Pole Handling

**Current:** `ema.Amplitudes` uses only mechanical poles, ignoring noisy/spurious poles
- Set `res_poles=([], [])` to bypass residual pole handling
- Works for clean synthetic data where stabilization diagram cleanly separates mechanical poles

**Proper approach:**
- Real stabilization diagrams contain mechanical poles AND spurious/noisy poles
- Residual poles in `ema.Amplitudes` should capture spurious contributions
- Demonstrates robust identification in presence of noise artifacts
- Shows proper separation of mechanical vs non-mechanical dynamics

**Benefit:** More realistic demonstration of EMA workflow, handles messier data

---

**Note:** These are nice-to-have improvements. Current implementation is sufficient for paper deadline. The core objective is demonstrating the difference between proportional and non-proportional damping identification, which the current code achieves.
