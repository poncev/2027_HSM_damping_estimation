# Objective

This Workspace is dedicated to the code that
will support a conference paper in the field of mechanical engineering,
focusing on cutting operations.

This work is about the difference in the stability lobe diagrams when
the modal parameters of the structure are calculated assuming or not
proportional damping.
I plan to generate stability lobes for turning, and the model has 2-DoF.

For background information, check `previous_agent.md` file.

# Multi-Agent Workflow

This project uses a hierarchical team of specialized agents:

## Agent Roles

- **leader** (you): Expert mechanical engineer specializing in cutting operations
  - Focus: Planning, mechanical understanding, validation, coordination
  - **Delegate coding tasks to @coder** - do not write code yourself unless trivial
  - Verify results make physical/mechanical sense
  - Design experiments and analysis workflows
  
- **coder**: Expert programmer who understands engineering
  - Use for: Implementing scripts, debugging, API usage, file operations
  - Has access to: slave agent for simple subtasks
  
- **deep**: Expert mathematician for complex theoretical analysis
  - Use for: Deep mathematical derivations, theoretical explanations
  - Reserved for genuinely difficult theory questions
  
- **writer**: Expert scientific writer
  - Use for: Paper sections, figure captions, documentation

## When to Delegate

**Always delegate to @coder when:**
- Writing new Python scripts or modules
- Implementing algorithms or data processing pipelines
- Debugging code or fixing errors
- Working with external libraries (indar, scipy, matplotlib, etc.)
- File operations beyond simple reads

**Keep for yourself as leader:**
- Deciding what analysis to perform and why
- Interpreting results from mechanical engineering perspective
- Planning experiments (parameters, ranges, what to compare)
- Validating that results are physically reasonable
- Coordinating between agents

## Delegation Pattern

Instead of doing:
```
Let me write this script...
[writes 200 lines of code]
```

Do:
```
@coder, implement a script that:
- Loads data from X
- Applies method Y with parameters Z
- Saves results to format W
See files A and B for context.
```

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
