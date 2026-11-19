# Objective

This Workspace is dedicated to the code that
will support a conference paper in the field of mechanical engineering,
focusing on cutting operations.

This work is about the difference in the stability lobe diagrams when
the modal parameters of the structure are calculated assuming or not
proportional damping.
I plan to generate stability lobes for turning, and the model has 2-DoF.

For background information, check `previous_agent.md` file.

# Paper

The draft of the paper is in `hsm_paper.md`.
I will paste it later to Word. It is around 6000 characters each page,
and the limit is 10 pages, but I plan to keep it below 6 pages.
Also, it is uncomfortable to write equations in Word, so
we have to do something for that.

# Python Style

- Use PEP 8 hanging indents for multi-line function calls: opening parenthesis at end
  of line, arguments indented one level (4 spaces), closing parenthesis on its own line.
  ```python
  result = func(
      arg1, arg2,
      arg3,
  )
  ```

# Rules

- The path to the Workspace is fixed, and can be found in `local.md`.
- You cannot git commit to the branch `main`.
- Do not install Python packages directly. I use a global env. called `resonance`, which
is defined in a YML file for mamba.
So, whenever you need a package, tell me the line
I have to write in the YML file for mamba.
The channels I use in the YML file are:
  - conda-forge
  - defaults

# Critical Instructions

- You **CANNOT** edit, create, or delete files or directories outside the Workspace.
