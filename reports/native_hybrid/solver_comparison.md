# Solver Integration Comparison

Date: 2026-08-12

## Scope

This report records downstream functional compatibility of native-hybrid mesh and activity contracts. It does not claim broad analytical validation against commercial solvers, experiments, or code-code benchmark suites.

## Accepted regression matrix

The leased functional run completed `109/109` tests across:

- Contact assembly and state handling.
- Nonlinear static behavior and limit-point continuation.
- DNV-oriented nonlinear cases and diagnostics.
- Nonlinear state batches and lifecycle transitions.
- Dynamics and mass/modal behavior.
- Buckling behavior.

The exact command is recorded in `damage_deletion.md`. Ecosystem review accepted the result for the declared module scope only.

Additional focused milestones accepted by independent ecosystem reproduction include:

- Solver activity: `17/17`.
- Solver-focused qualification: `61/61`.
- Incremental runtime publication: `11/11`.

## Architectural comparison

| Concern | Authoritative owner | Downstream behavior |
|---|---|---|
| Exact geometry, topology, tolerance, relationship evidence | ANYgeometry | Consumed through public model APIs; unsupported relationships fail closed. |
| Discretization, stable node/element identity, activity flags | ANYmesh | Publishes compact mesh records without recreating geometry truth. |
| Analysis regions, section assignment, project migration | ANYfem | Resolves explicit persistent intent; no coordinate-inferred coupling. |
| Assembly, constraints, activity/deletion, solve modes | ANYsolver | Consumes mesh identity and keeps backend-specific state out of geometry. |

## Open comparison work

- Numerical reference tables and convergence tolerances for the complete static/modal/buckling/nonlinear/contact/dynamics matrix.
- Nonlinear curved-shell convergence using the qualified native surface charts.
- Restart equivalence across process boundaries and schema migrations.
- Combined activity/S4 formulation regression after the branches are reconciled.
- Leased large-model time, memory, and scaling qualification.

## Focused restart regression

A 10-case ANYsolver restart matrix passed in 16.01 seconds. The exercised paths cover activity history, nonlinear and plastic state, force/displacement control, constrained affine state, initial-field consistency, rollback to the last committed checkpoint, and orthotropic restart integrity. Incomplete orthotropic state remains fail closed rather than using isotropic reconstruction.

The result is limited to the named focused cases and is not a broad restart-performance, cross-version, distributed, or large-model qualification.
