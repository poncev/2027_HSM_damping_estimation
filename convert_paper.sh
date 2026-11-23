#!/bin/bash
# Convert hsm_paper.md to Word format using the HSM template
# Equations in LaTeX format will be converted to native Word equations

# Activate the resonance environment to access pandoc
eval "$(conda shell.bash hook)"
conda activate resonance

pandoc hsm_paper.md \
  -o hsm_paper.docx \
  --reference-doc=HSM_Template.docx \
  --number-sections \
  --filter pandoc-crossref 2>/dev/null

echo "✓ Paper converted to hsm_paper.docx"
echo "  - LaTeX equations → Native Word equations (editable!)"
echo "  - Template styling applied"
echo "  - Open in M365 web or desktop - equations will be fully editable"
