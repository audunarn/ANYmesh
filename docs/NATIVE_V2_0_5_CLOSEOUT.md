# ANYmesher 0.5.0 native-v2 closeout

This document is the final qualification contract for the opt-in planar and
owner-certified cylindrical Frontal-Delaunay implementation. It supersedes
older remaining-gate prose where that prose treats repeated runtime comparison
as a 0.5.0 release blocker.

## Frozen scope

- Package version: `0.5.0`, Alpha.
- Geometry owner: ANYgeometry commit
  `7edbb8b624d3f7d4548d2ec11b2d00c55f1265d2`.
- Frontal-Delaunay remains opt-in. Legacy native and mapped defaults are
  unchanged.
- Cylindrical support is through owner-certified connected sector faces.
- Field-guided quad fronts remain deferred. Existing collars and local
  triangle recombination are not that architecture.

## Required release evidence

- The complete Windows, Linux and macOS Python 3.11-3.14 source matrix passes.
- The four Windows/Linux Python 3.11-3.12 optional-Gmsh cells pass the focused
  backend, owner-integration and process-global cleanup contracts.
- All twelve native wheels pass installed-wheel cylindrical refinement smoke
  with actual insertions, exact protected topology, positive Jacobians and
  source non-mutation.
- The sdist contains both `tools/release_wheel_smoke.py` and
  `benchmarks/native_v2_cylinder_cases.py`, and passes strict archive and
  metadata validation.
- One fresh work-bearing approximately 500k-element cylindrical run completes
  with actual refinement, strict validity, exact protected topology, periodic
  conformity, owner associations and source non-mutation.

## Explicit deferrals

Elapsed time and peak RSS are recorded for the 500k run, but comparative
runtime thresholds do not block 0.5.0. Repeated timing, matched-resolution
legacy comparisons, workstation performance targets and further optimization
belong to the next release. Infrastructure failure, invalid topology,
non-positive Jacobians, publication of a partial candidate or loss of protected
ownership remain hard failures.

No upload, tag or GitHub release is authorized by this closeout contract.
