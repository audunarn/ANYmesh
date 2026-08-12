# Damage and Element Deletion Qualification

Date: 2026-08-12

## Implemented contract

ANYsolver owns solver-state activity through the public `ElementActivity` seam. Meshing remains responsible for stable element identity and compact connectivity; solver activity changes do not rewrite geometry truth or renumber the mesh. Inactive/deleted elements remain diagnosable through stable IDs while assembly and contact consume the active state explicitly.

The owned activity slice is recorded in `C:\Github\ANYsolver\ACTIVITY_S4_HANDOFF.md`. It covers the public activity API and its integration into core, boundary, matrix assembly, general assembly, and contact paths.

## Evidence

The focused activity milestone was independently reproduced as `17 passed`. The broader functional solver-mode regression was run under the ecosystem performance lease:

```powershell
$env:PYTHONPATH='C:\Github\ANYsolver\src;C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src;C:\Github\ANYio\src;C:\Github\ANYmaterial\src;C:\Github\ANYbuckling\src;C:\Github\ANYfem\src'
python -m pytest tests\test_fe_solver_contact.py tests\test_fe_solver_nonlinear_static.py tests\test_fe_solver_nonlinear_limit_point.py tests\test_fe_solver_nonlinear_dnv.py tests\test_nonlinear_analysis_diagnostics.py tests\test_nonlinear_state_batches.py tests\test_nonlinear_state_lifecycle.py tests\test_fe_solver_dynamics.py tests\test_fe_solver_mass_modal.py tests\test_fe_solver_buckling.py -q
```

Result: `109 passed in 67.58s`; no failures or skips. Ecosystem review accepted this as functional regression evidence for the listed modules.

## Qualified interpretation

- Static, nonlinear, contact, dynamics, modal, and buckling paths exercised the activity-aware implementation in the declared module scope.
- State batch and lifecycle regressions exercised activity publication and restoration behavior.
- Activity is a solver/backend concern; ANYgeometry records structural truth and ANYmesh records discretization identity.

## Limitations and open gates

- This result is not a damage-model calibration, fracture-mechanics validation, scaling result, or memory/performance measurement.
- The activity hot path is committed at ANYsolver `1fd1c19`, with its SHA ledger
  at `7daa6e8`; reconciliation with the separately owned S4 formulation branch
  remains required.
- Combined activity/S4 regression must follow the recorded merge order before ecosystem closeout.
- Restart compatibility outside the exercised state lifecycle modules and long-running progressive-deletion histories remain open.

## Focused restart qualification

The targeted restart matrix passed 10/10 cases in 16.01 seconds on Windows CPython 3.13.9. It covers activity damage-history JSON ownership, nonlinear packed-state equivalence, plastic prescribed-state and reaction retention, force-control prescribed-state retention, beam scalar-alpha fiber broadcast, constrained nonunit affine state, field-bearing displacement consistency, failed displacement-control checkpoint restoration, and orthotropic fail-closed and staged-history equivalence.

This is scoped restart evidence only. It does not qualify long-running checkpoint cadence, cross-version persistence, distributed restart, or large-model memory behavior.
