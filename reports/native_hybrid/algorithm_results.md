# Native Hybrid Algorithm Results

Date: 2026-08-12

## Current qualified scope

The production path chooses mapped meshing only where its topology and chart preconditions are satisfied. Other supported bounded faces use the native constrained-triangulation and quality-gated recombination path. Unsupported or unqualified charts fail closed with `MeshError`; Gmsh remains an explicit optional backend rather than the production default.

Focused evidence currently covers:

- Shared mapped/native edge-node identity and deterministic owner lineage.
- Authoritative `Plane` handling for triangular, concave, narrow, and holed planar regions.
- Exact circular trim constraints and an empty hole interior.
- Native `Cylinder`, `Cone`, `RuledSurface`, and `CoonsSurface` chart projection, mixed Q4/T3 output, refinement, and Q8 midsides.
- Typed public ANYgeometry intersection/imprint consumption without coordinate-inferred topology.
- Mesh persistence with model/revision identity and fail-closed stale or malformed inputs.

## Focused commands and results

```powershell
$env:PYTHONPATH='C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src'
python -m pytest tests\test_planar_native_qualification.py tests\test_performance_harness_fixtures.py -q
```

Result: `5 passed in 2.21s`.

The curved native qualification was independently reproduced by ecosystem review as `13/13` passing. The native public-path and persistence/intersection slices were independently reproduced as `21/21` and `31/31` passing in their declared scopes.

## Important interpretation

An earlier invalid concave result came from an unqualified bounded fallback chart in the test fixture, not from accepted recombination. Replacing that fixture with an exact, correctly scaled `Plane` eliminated clipping and repeated boundary stations. The production boundary-order/model-identity guard was not weakened.

The focused narrow-region mesh is topologically valid but contains low-shape-quality triangles, as recorded in `quality_results.json`. This remains a quality-improvement target and is not presented as evidence that all production quality targets are met.

## Open qualification

- Large and adversarial planar/curved corpora with acceptance distributions.
- Nonlinear curved convergence and solver-reference comparisons.
- Full platform wheel installation/import matrix.
- Incremental remeshing, cache, cancellation, and deterministic parallel evidence.
- Leased generation, memory, serialization, and scaling measurements at the plan sizes.
- Shell/sheet shared-topology integration after the renewed ANYgeometry public-tree handoff.
