# Reference Consistency Audit Report
**Paper:** hsm_paper.md  
**Date:** 2026-11-21

---

## 1. In-Text Citations (Found in Paper Body)

Total unique citations found: **17**

| Citation | Location(s) | Status |
|----------|------------|--------|
| [Baldelomar 2026] | Introduction (line ~95) | ✓ In References |
| [Balmes 1997] | Theoretical preliminaries (line ~149) | ✓ In References |
| [Balmes 2019] | Introduction (line ~72) | ✓ In References |
| [Bellos 2025] | Introduction (line ~114) | ✓ In References |
| [Den Hartog 1956] | Table 1, Example section | ✓ In References |
| [Ewins 2000] | Introduction (line ~38) | ✓ In References |
| [Hasselman 1976] | Introduction (line ~87) | ✓ In References |
| [Hintze 2023] | — NOT FOUND IN TEXT — | ✗ UNUSED |
| [Indar 2026] | Introduction (line ~56) | ✓ In References |
| [Muñoa 2015] | Abstract (line ~14), Introduction, Example section | ✓ In References |
| [Muñoa 2016] | Abstract (line ~26) | ✓ In References |
| [Neugebauer 2010] | Introduction (line ~103) | ✓ In References |
| [Pawade 2008] | Table 1 (line ~230) | ✓ In References |
| [Ponce-Vanegas 2026] | Introduction (line ~60), Theoretical preliminaries | ✓ In References |
| [Ponce-Vanegas 2026b] | Theoretical preliminaries (line ~170) | ✓ In References |
| [Reith 2017] | Introduction (line ~108) | ✓ In References |
| [Yang 2010] | Table 1, Example section (multiple locations) | ✓ In References |

---

## 2. References Section Entries

Total references listed: **17**

All references are properly formatted and correspond to citations in the text, **with one exception** (see below).

---

## 3. Missing Citations

**Citations in text but NOT in References:** ✓ NONE  
All 17 citations found in the text are present in the References section.

---

## 4. Unused References

**References in References section but NOT cited in text:**

| Reference | Issue |
|-----------|-------|
| [Hintze 2023] | **NOT CITED** — This reference appears in the References section but is never mentioned anywhere in the main text, table captions, or figure captions. It should either be (1) cited in the text, or (2) removed from References. |

---

## 5. Hyphenation Consistency Analysis: "nonproportional" vs "non-proportional"

**Inconsistency Found:** ⚠️ YES

### Summary of Hyphenation Usage:

| Form | Count | Lines | Status |
|------|-------|-------|--------|
| `nonproportional` (no hyphen) | ~13 | 10, 65, 68, 73, 103, 111, 211, 218, 238, 249, 271, 285, 291 | Dominant form |
| `non-proportional` (with hyphen) | ~4 | 12, 24, + within References | Alternative form used |
| `non proportional` (space, no hyphen) | 1 | 298 | **ERROR** |

### Specific Issues:

1. **Line 298 (Conclusions section):** 
   - **ERROR:** "...properly identify and characterize **non proportional damping**"
   - **Should be:** Either "nonproportional" or "non-proportional" (space missing)
   - Current text is missing the hyphen entirely

2. **Inconsistent hyphenation between sections:**
   - Abstract uses both forms: line 10 "nonproportional" vs line 12 "non-proportional"
   - Introduction uses primarily "nonproportional" but line 24 has "non-proportional"
   - References section uses both forms within citation titles

### Recommendation:
Choose one form and apply consistently throughout the document:
- **Option 1 (Recommended):** Use `nonproportional` (no hyphen) consistently — this is the dominant form in your manuscript
- **Option 2:** Use `non-proportional` (with hyphen) consistently — this matches compound adjectives in some references

---

## Summary of Issues

| Category | Count | Severity |
|----------|-------|----------|
| Missing citations | 0 | — |
| Unused references | 1 | Medium |
| Hyphenation errors | 1 | Low |
| Hyphenation inconsistency | Multiple | Low |

### Action Items:
1. ✓ **REQUIRED:** Remove [Hintze 2023] from References OR add citation to it in the text
2. ⚠️ **REQUIRED:** Fix line 298: Change "non proportional" to either "nonproportional" or "non-proportional"
3. ⚠️ **RECOMMENDED:** Standardize hyphenation throughout document (choose one form)

