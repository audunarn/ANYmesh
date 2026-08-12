# Native hybrid mesher baseline

Date: 2026-08-12

## Source state

All coordinated repositories use the `native_hybrid_mesher` branch:

- `ANYgeometry`
- `ANYmesh`
- `ANYfem`
- `ANYsolver`

Pre-existing working-tree changes were retained. In particular, the active
ANYfem UI work was kept outside this implementation's initial write sets.

## Test baseline

- ANYgeometry: 272 tests passed in 9.17 seconds.
- ANYmesh initial source-linked run: seven geometry-0.2 compatibility failures
  in mapped face decomposition, caused by direct mutation of immutable metadata.
- ANYmesh mapped decomposition after Wave 0 repair: 28 focused tests passed in
  2.05 seconds.
- Downstream first-pass collection required explicit sibling `src` paths; the
  system Python installation did not expose the coordinated packages.
- ANYsolver full collection additionally required the local ANYio source tree.

The import-path failures are environment findings, not product regressions. All
final qualification commands use the coordinated ecosystem source paths.

## Baseline invariants

- The existing mapped Q4/Q8 algorithm remains available and deterministic.
- ANYgeometry owns topology, model identity, tolerance, spatial candidates,
  structural ownership, mutation policy, and change records.
- Gmsh is optional compatibility only and is not a production dependency.
- Full strict geometry audit is not part of an interactive remesh.
