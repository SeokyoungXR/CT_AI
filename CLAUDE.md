# Claude Code Collaboration Guide

## Mission

Build and evaluate a privacy-preserving Spatial Evidence Layer that connects CT_AI temporal 3D scene-graph traces to VirtualME's evidence-grounded self-model and design outputs.

## Repository rules

- Work on the user's fork and current feature branch only.
- Do not push to `upstream` or create a pull request automatically.
- Preserve existing APIs and tests unless the task requires a contract change.
- Use `python3`; the repository environment is Python 3.11 with NumPy, SciPy, Pillow, Matplotlib, PyVista, and VTK.
- Keep implementation deterministic and runnable without GPU or network access where possible.

## Collaboration roles

- **Data Profiling**: inspect schemas, frame alignment, coordinate systems, missing values, outliers, and leakage risks. Read-only unless a profiling utility is explicitly requested.
- **Model Evaluation**: define a deterministic baseline, valid temporal/spatial splits, metrics, error analysis, and reproducible comparisons.
- **3D Visualization**: render and inspect scenes, traces, relations, camera poses, meshes, and uncertainty-aware design outputs.
- **Productive Data Modeler**: integrate the roles and implement the smallest complete vertical slice.

## Required workflow

1. Identify the nearest code path that directly controls the requested behavior.
2. State one falsifiable hypothesis and one cheap validation check before editing.
3. Make the smallest focused change.
4. Run the narrowest relevant test immediately after the edit.
5. Keep provenance on every generated record: source, record id, episode id, and uncertainty.
6. Update the relevant documentation and Mermaid architecture when boundaries change.
7. Finish with tests run, changed files, assumptions, and remaining risks.

## Evidence and safety invariants

- Scene objects and relations are observations, not proof of intent, mood, personality, or productivity.
- No psychological diagnosis or causal claim may be generated from a scene trace.
- Missing records remain missing; do not interpolate across a hiatus.
- Multiple frames from one episode must not be treated as independent witnesses.
- User corrections append a new version and never overwrite computed confidence.
- Do not emit raw images, names, aliases, or unnecessary personal content in normalized records.
- A renderer may only display observed objects or explicitly marked inferred parameters.

## Output contract

The spatial adapter emits `SpatialEvidenceRecord` objects through `scripts/spatial_evidence.py`. Preserve JSON serializability via `to_dict()`. Any replacement model must maintain the contract or provide an explicit migration and tests.

## Validation commands

```bash
python3 -m unittest tests.test_trace_io -v
python3 -m unittest discover -s tests -v
```
