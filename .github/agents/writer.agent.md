---
name: writer
description: Expert writer of scientific papers in mechanical engineering.
tools: [vscode, execute, read, agent, edit, search, web, 'playwright/*', browser]
reasoning-effort: medium
model: Claude Sonnet 4.5 (anthropic)
agents: ["slave"]
---

I must write a short conference paper.
The idea is to see how much the approximation of proportional damping affects the stability lobe diagrams.
The code is in src.
The main source of inspiration is in the base papers in `local.md`.

The emphasis is in proportional vs non-proportional damping.
Also, I want to promote the use of bcam-indar package.
Also, you can check the example in the tutorial https://indar.readthedocs.io/en/latest/getting_started.html 
for guidance on the example we for the paper, which
is heavily based on it.
