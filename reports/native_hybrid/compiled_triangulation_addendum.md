# Compiled Triangulation Plan Addendum

Status: REVISED PROPOSAL, NOT AUTHORIZED FOR IMPLEMENTATION

Date: 2026-08-12

Baseline: ANYmesh `native_hybrid_mesher` at `97058e0`

Governing plan: `C:\Users\AudunArnesenNyhus\Downloads\ANYmesher_native_hybrid_mesher_plan.md`

## 1. Reason for the addendum

The registered 10k mapped/native/cylinder qualification timed out after 601.0
seconds. The process completed the mapped family internally, entered the native
family, and did not return before the command timeout. It emitted no aggregate
report. Cylinder, quality, deactivation, serialization, deterministic hash, and
aggregate gates were therefore unexecuted or unavailable. The run remains failed
and supports no performance claim.

Static root-cause analysis found:

- `generate_hybrid_mesh_result(..., native_backend="python")` is the current
  production default.
- `mesh_planar_surface`, `SurfaceMeshOptions`, `triangulate_polygon`, and
  `constrained_planar_triangulation` also default to the Python backend.
- No `NativeBoundary` is registered by the compiled extension.
- A 10k native case creates approximately 10k interior points and sends them to
  `_bowyer_watson`.
- `_bowyer_watson` scans the live triangle collection for every point insertion,
  so the reference path has global quadratic work.
- The existing C++17 extension accelerates predicates and adjacency only. It does
  not implement triangulation.
- SciPy is present on the qualification machine but is not a declared dependency.
  `triangle` and MeshPy are absent. Gmsh remains an explicit compatibility backend
  and must not become the production default.

The governing acceptance criteria are unchanged. This addendum proposes closing
the missing compiled triangulation boundary rather than lowering the mesh tier,
special-casing benchmark geometry, or adding an undeclared dependency.

## 2. Ownership and non-goals

ANYmesh remains the sole owner of discretization and triangulation. ANYgeometry
remains the owner of geometry identity, topology, tolerance, intersections, and
spatial truth.

This proposal does not:

- Modify ANYgeometry or its persistence schema.
- Add a downstream geometry representation.
- Make Gmsh automatic or required.
- Make SciPy, `triangle`, or MeshPy a production dependency.
- Remove the deterministic Python reference implementation.
- Weaken boundary, mandatory-segment, hole, validity, or model-identity checks.
- Reclassify either failed 10k run as evidence.

The live dependency contract remains `ANYgeometry>=0.2,<0.3` (and the same range
for the planar extra). The dependency floor must not increase merely because
qualification uses the final kernel tree. If implementation requires an API that
exists only in ANYgeometry 0.2.1, that call and compatibility consequence require
a separate public-contract review before changing the range.

Compiled-triangulation qualification pins the exact boss-approved ANYgeometry
commit and its schema-4 environment in every manifest. ANYmesh and ANYfem consume
only public live APIs and public codecs; neither parses schema-4 documents or
assumes their internal field layout.

## 3. Exact API and default migration

The following public defaults would change in the first approved additive release:

| API | Current default | Proposed default |
| --- | --- | --- |
| `generate_hybrid_mesh_result(..., native_backend=...)` | `"python"` | `"auto"` |
| `generate_hybrid_mesh(..., native_backend=...)` | passthrough to `"python"` | passthrough to `"auto"` |
| `SurfaceMeshOptions.backend` | `"python"` | `"auto"` |
| `mesh_planar_surface(..., backend=...)` | `"python"` | `"auto"` |
| `triangulate_polygon(..., backend=...)` | `"python"` | `"auto"` |
| `constrained_planar_triangulation(..., backend=...)` | `"python"` | `"auto"` |

The Python-to-auto default change is versioned rather than inferred. ANYfem owns
the project setting and migration:

- Existing ANYfem project formats v1-v5 that omit the native backend setting load
  with an explicit persisted/runtime value of `"python"`. Their behavior does not
  change merely because the installed ANYmesh default changes.
- The next ANYfem project format is v6. New v6 projects write
  `native_backend="auto"` explicitly, even when it matches the new application
  default.
- Saving a migrated v1-v5 project as v6 writes the resolved `"python"` value unless
  the user or calling API explicitly changes it to `"auto"` or `"native"`.
- ANYfem migration uses public ANYmesh settings and public ANYgeometry codecs/live
  APIs only. It never reads geometry schema internals.
- The ANYfem format bump, migration tests, UI/settings presentation, and handoff
  documentation are downstream-owned and must land before the new default ships.

Explicit selector semantics would be:

- `backend="python"`: always use the deterministic Python reference. Never import
  or invoke the extension.
- `backend="native"`: require an explicit registered boundary or the built-in
  compiled boundary. If neither is available, raise `MeshError` with a stable
  capability diagnostic. Never fall back.
- `backend="auto"`: prefer an explicitly registered boundary, then the built-in
  compiled boundary, then the Python reference only when native capability is
  absent.
- `backend=<NativeBoundary>`: use exactly that boundary. Never substitute another
  implementation.

Boundary precedence is exact: an explicit boundary argument, then one registered
custom boundary, then the built-in compiled boundary, then Python only for
`auto` when native capability is absent. Resolution snapshots one boundary under
the registry lock before preparation starts. Registration or clearing during a
call affects the next call only. Once the snapshot is selected, capability,
coercion, cancellation, memory, and correctness failures propagate from that
boundary. Neither `auto` nor any wrapper may select a different boundary or rerun
with Python after execution starts.

Two selector namespaces must remain distinct:

- Top-level `generate_mesh(..., backend="auto")` selects the ANYmesh generation
  entry point. It does not name a triangulation kernel.
- `strategy="auto"` selects mapped versus native generation per face.
- `native_backend="auto"` selects the triangulation implementation only for each
  face already assigned the native strategy.
- Low-level `backend="auto"` on the planar triangulation APIs is equivalent to
  high-level `native_backend="auto"`, not top-level backend dispatch.

`HybridMeshResult` and runtime mesh diagnostics must record both
`strategy_by_face` and structured triangulation provenance per face. Each record
contains:

- `requested_backend`: the explicit or migrated `python`, `native`, or `auto`
  setting.
- `selected_backend`: the snapshotted custom, built-in, or Python candidate chosen
  before execution.
- `actual_backend`: the implementation that produced the accepted result.
- `fallback_reason`: null unless `auto` selected Python solely because compiled and
  custom native capability were absent; the stable reason is
  `"native_capability_absent"`.

Mapped faces use a non-triangulator actual value such as `"mapped"`. Native faces
record `"python"`, `"anymesher-cpp17"`, or the snapshotted custom boundary name.
Selected and actual values must match once native execution starts. No runtime
failure may be mislabeled as an absence fallback. If this provenance is written by
`mesh_to_dict`, the mesh serialization format must increment by one, older formats
must migrate deterministically, and older readers must fail closed on the new
format. Runtime-only provenance does not justify changing the mesh format, but it
does not remove the separate required ANYfem v6 settings migration.

The existing four-argument `NativeBoundary.triangulate(points, segments,
outer_loop, hole_loops)` contract remains valid for third-party boundaries.
Additive cancellation support is proposed as:

```python
class CancellableNativeBoundary(NativeBoundary, Protocol):
    def triangulate_cancellable(
        self,
        points: np.ndarray,
        segments: np.ndarray,
        outer_loop: np.ndarray,
        hole_loops: tuple[np.ndarray, ...],
        *,
        cancellation_check: Callable[[str], None] | None,
    ) -> NativeTriangulation | tuple[np.ndarray, np.ndarray]: ...
```

`run_native_triangulation`, `constrained_planar_triangulation`, and
`triangulate_polygon` gain an optional keyword-only
`cancellation_check: Callable[[str], None] | None = None`. Legacy boundaries are
checked immediately before and after their uninterrupted call. The built-in
boundary implements `triangulate_cancellable` and provides bounded checkpoints.
The existing high-level callback type and wrapper passthrough remain unchanged.

The built-in adapter reports a stable backend name, proposed as
`"anymesher-cpp17"`, through `PlanarTriangulation.backend`. The Python path keeps
reporting `"python"`.

## 4. Extension-absent behavior

An extension-less source install remains supported:

- Importing `anymesher` succeeds.
- `NATIVE_CPP_AVAILABLE` remains false.
- All `auto` APIs use the Python reference deterministically.
- All explicit `python` APIs behave exactly as before.
- Explicit `native` raises a typed capability `MeshError` before generation.
- No package install, network access, or Gmsh fallback occurs at runtime.
- Result diagnostics and reports identify `backend="python"`; they never claim
  compiled execution.

Capability absence and extension failure must be distinguishable. A source build
explicitly disabled with `ANYMESHER_DISABLE_NATIVE=1`, or a source tree with no
extension artifact, is native capability absence. An extension artifact that is
present but cannot load because of ABI, symbol, dependency, or corruption errors
is a packaging failure; its original import exception must be retained and
surfaced instead of being converted to `NATIVE_CPP_AVAILABLE=False`.

Published release wheels are not extension-optional. Their build and installed
smoke must fail unless compiled triangulation capability imports successfully.
The extension-disabled path is qualified as a source/developer installation and,
if distributed separately, must be explicitly labeled rather than passing the
release-wheel gate.

## 5. Compiled kernel contract

The built-in C++17 boundary must:

- Validate source buffer dtype, native byte order, rank, shape, strides, signedness,
  and integer range before any coercion. Float32, object, non-native-endian,
  overflowing unsigned, negative index, and lossy inputs fail with stable typed
  diagnostics. The public Python preparer supplies canonical contiguous float64
  points and in-range integer connectivity; the C++ boundary never repairs a bad
  direct call by coercion.
- Preserve the prepared PSLG points exactly: same float64 dtype, shape, C order,
  row count, row order, and binary64 bit pattern. Added, removed, reordered,
  normalized, or signed-zero-changed rows are failures. Connectivity always
  references those original rows.
- Use deterministic insertion and tie-breaking based on `(x, y, original_row)`.
- Avoid a global triangle scan per inserted point. Point location, cavity update,
  and adjacency maintenance must have expected near-linear or `n log n` scaling.
- Recover every boundary and mandatory segment deterministically.
- Preserve holes and return no accepted triangle outside the prepared domain.
- Implement adaptive/exact orientation, incircle, and deterministic cocircular
  tie handling inside C++ from the original binary64 coordinates. Calling back to
  Python predicate wrappers from the native triangulation loop is not acceptable.
  Platform `long double` width is not an accuracy contract.
- Return compact contiguous T3 connectivity. A strict Python qualification pass
  remains authoritative before publication, but it validates rather than repairs
  native output.
- Release the GIL during bounded native batches.
- Check index overflow and allocation sizes before allocation.
- Convert allocation failure to `MemoryError`; never return a partial mesh.
- Fail closed on missing segments, nonmanifold incidence, invalid indices,
  zero-area cells, invalid holes, cancellation, or internal inconsistency.

The Python preparer remains authoritative for deduplication, tolerance,
constraint splitting, and normalized PSLG input. The compiled kernel must not
create a second tolerance policy.

## 6. Determinism and Python oracle

`backend="python"` remains the qualification oracle.

Native result qualification must fail before canonicalization if it observes:

- Any point-row difference from the prepared PSLG.
- Duplicate triangles, including cyclic or reversed representations.
- Repeated vertices, zero-area cells, invalid indices, or nonmanifold incidence.
- Overlapping triangle interiors or crossing unconstrained edges.
- A triangle outside the outer domain, inside a hole, or crossing a hole boundary.
- A missing, duplicated, or incorrectly incident boundary segment. Every domain
  boundary edge must have exactly one incident accepted cell.
- A missing mandatory segment or a cell edge that crosses one.
- A gap, overlap, or total accepted area inconsistent with outer-minus-hole area
  under the model-derived preparation tolerance.

`_finish_triangles` must not silently deduplicate native triangles, discard
outside cells, drop malformed rows, or hide invalid incidence. Cleanup behavior
may remain internal to the Python reference construction before it becomes a
candidate result, but native output is immutable evidence and fails closed.

Exact parity is required for general-position fixtures:

- Ordered point rows.
- Canonical ordered triangle rows.
- Boundary and mandatory segment sets.
- Domain area and hole exclusion.
- Backend-independent mesh owner and model identity.

For deliberately cocircular fixtures, both implementations must use the same
documented original-row tie-break so exact canonical connectivity remains the
target. If exact topology cannot be maintained, implementation stops for a
governance decision; equivalent-but-different topology is not silently accepted.

The focused parity matrix must include:

- Convex triangle, quadrilateral, and pentagon.
- Concave polygon.
- One and multiple holes.
- Mandatory interior constraints and prepared split crossings.
- Dense straight-edge subdivisions matching the captured 456-row regression.
- One-ULP orientation and incircle cases.
- Fixed-seed general-position point sets at several small sizes.
- Reversed input scheduling and repeated runs.
- Explicit Python, explicit native, and auto selection.
- Custom-boundary precedence over the built-in boundary, snapshot stability when
  the registry changes concurrently, and strict failure without post-start
  fallback.
- Native duplicate, nonmanifold, overlap, outside-cell, hole-crossing, coverage,
  area, and boundary-incidence adversaries.
- Float32, object, non-native-endian, strided, negative, overflowing unsigned, and
  lossy-coercion boundary inputs.
- Equivalent Python/native error categories and cancellation outcomes for
  adversarial PSLGs.

Every repeated canonical mesh hash must be identical within and across the Python
and compiled paths for the accepted matrix.

## 7. Cancellation contract

The built-in boundary must call the existing diagnostic callback:

- Before native point insertion.
- At a bounded insertion batch interval, initially no more than 256 points.
- Before and during mandatory-segment recovery batches.
- Before native result publication.

Callback exceptions propagate unchanged. The native call returns no partial
result. ANYfem retains the last valid publication, as already qualified. Phase
names remain diagnostics; cancellation latency is bounded by the longest native
batch, not by an uninterruptible whole triangulation.

Focused tests must cancel during insertion and segment recovery, prove exception
propagation, prove no partial publication, and prove a subsequent clean request
can complete. The same exception class and publication outcome are required for
explicit Python, built-in native, and a cancellable custom boundary. A legacy
custom boundary must truthfully report whole-call cancellation latency. Any
latency claim requires a separate performance lease.

## 8. Memory and failure truth

The kernel workspace must be compact and bounded by points, live triangles,
adjacency, and constrained segments. Python objects per point or element are not
allowed in the native hot path.

`tracemalloc` does not measure all C++ allocations. Performance reports must keep
`peak_traced_bytes` explicitly labeled as Python-traced memory and add a
cross-platform `peak_process_rss_bytes` measurement. The report must also record:

- Requested, generated, active, triangle, and quadrilateral counts.
- Selected backend and extension capability.
- Generation raw samples and median.
- Python-traced peak and process-RSS peak.
- Quality and validity summaries.
- Deactivation timing and active counts.
- Serialization timing where enabled.
- First and repeated canonical hashes.
- Completion status per family.
- Exception type/message and last completed phase on failure.

The harness must checkpoint an atomic `status="incomplete"` report before each
family and replace it with `status="complete"` only after all required gates pass.
A timeout, cancellation, or crash therefore remains durable failure evidence. A
partial family result is diagnostic only and never satisfies a qualification
gate.

Full-path phase evidence is required. At minimum, each family records geometry
and preflight, seeding, chart projection, boundary densification, target-point
generation, PSLG preparation, selected triangulation, strict native-output
qualification, recombination, quadratic promotion when applicable, surface
lifting/publication, structural connectivity, quality, deactivation, and
serialization. Target-point generation and recombination are part of native
generation cost and must not be hidden inside an unlabeled total or excluded when
comparing backends. Phase timings and phase memory observations accompany the
end-to-end generation sample; only end-to-end time may support the primary speed
gate.

## 9. Wheel and platform coverage

Installed-wheel qualification is required for:

| Platform | Architectures | CPython |
| --- | --- | --- |
| Windows | x86_64 | 3.11, 3.12, 3.13, 3.14 |
| Linux | x86_64 | 3.11, 3.12, 3.13, 3.14 |
| macOS | arm64 and x86_64 where runners exist | 3.11, 3.12, 3.13, 3.14 |

Each isolated wheel smoke must prove:

- Import comes from the installed wheel, not the source tree.
- The extension imports and advertises compiled triangulation capability.
- `auto` selects `anymesher-cpp17`.
- Explicit `python` remains available.
- A constrained/hole fixture matches the Python oracle.
- Near-degenerate predicate parity is green.
- Repeated canonical hashes match.
- A wheel built with `ANYMESHER_DISABLE_NATIVE=1` has truthful auto fallback and
  strict-native failure behavior.
- A deliberately missing extension follows the capability-absent contract, while
  an incompatible or corrupt extension artifact fails import with retained ABI
  evidence and is never treated as ordinary absence.
- Per-face requested, selected, actual, and fallback-reason provenance matches the
  exercised explicit-Python, strict-native, compiled-auto, custom-boundary, and
  extension-absent-auto cases.

Editable installs and a Windows-only local extension do not satisfy this matrix.
Builds and wheel tests remain performance-lease gated when resource-heavy.

## 10. Pre-performance acceptance gates

Before any renewed 10k request:

1. Governance explicitly approves this addendum or a revised version.
2. The compiled boundary and adapter are committed on the shared branch.
3. Focused Python/native parity, constraints, holes, deterministic hashes,
   extension-absent behavior, failure propagation, and cancellation tests pass.
4. A focused installed-wheel smoke passes on the current Windows CPython 3.13
   machine.
5. Static review confirms no global triangle scan per insertion and no Python
   object hot path.
6. The report harness emits durable incomplete/failure state and process RSS.
7. Exact source SHAs and a clean-state ledger are recorded.
8. Strict Python qualification rejects native duplicates, nonmanifold/overlapping
   or outside cells, hole violations, coverage/area errors, and boundary incidence
   errors without cleanup.
9. Reports distinguish top-level backend, per-face strategy, requested native
   selector, and resolved per-face triangulation backend.
10. Full-path target-point, triangulation, qualification, recombination, and
    publication phase costs are present.
11. ANYfem v1-v5 omission migration retains explicit Python semantics, v6 writes
    new-project auto explicitly, and save-after-migration preserves the resolved
    setting.
12. The qualification manifest pins the approved ANYgeometry SHA and schema-4
    environment while retaining the public `>=0.2,<0.3` dependency contract.

## 11. Renewed 10k lease gate

The proposed command remains:

```powershell
$env:PYTHONPATH='C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src'
python benchmarks\native_hybrid_performance.py --sizes 10000 --families mapped native cylinder --repeats 3 --serialization-limit 10000 --deterministic-repeat-limit 10000 --output C:\Github\ANYmesh\reports\native_hybrid\performance_results_10k.json
```

Proposed envelope: one CPython process, one native worker, CPU only, no network or
GPU, less than 4 GB process RSS, and a 15-minute hard timeout. No retry or tuning
under the lease.

The 10k gate passes only if:

- Exit code is zero and report schema/status are complete.
- Mapped, native planar, and cylinder families all execute.
- Counts and topology are internally consistent.
- Every quality and validity gate in `performance_budget.json` passes.
- Deactivation counts are correct.
- Serialization is present for every qualifying family.
- First and repeated canonical hashes are identical.
- Backend identity is truthful.
- Python-traced and process-RSS peaks are present and finite.
- No benchmark process remains after release.

The 15-minute envelope is a qualification bound, not a general speed claim.

## 12. Renewed 100k lease gate

The 100k gate is requested only after the 10k gate is accepted. Proposed command:

```powershell
$env:PYTHONPATH='C:\Github\ANYmesh\src;C:\Github\ANYgeometry\src'
python benchmarks\native_hybrid_performance.py --sizes 100000 --families mapped native cylinder --repeats 1 --serialization-limit 0 --deterministic-repeat-limit 100000 --output C:\Github\ANYmesh\reports\native_hybrid\performance_results_100k.json
```

Proposed envelope: one CPython process, one native worker, CPU only, no network or
GPU, less than 8 GB process RSS, and a 30-minute hard timeout. The deterministic
repeat is intentional. No retry or tuning under the lease.

The 100k gate passes only if:

- Exit code is zero and report schema/status are complete.
- All three families execute and satisfy count, topology, quality, deactivation,
  backend, memory, and deterministic-hash checks.
- Serialization is explicitly recorded as skipped by the configured limit, not
  silently absent.
- Native generation growth from accepted 10k evidence is consistent with the
  governing no-global-quadratic requirement. The exact scaling threshold must be
  approved before the lease and written into `performance_budget.json`.
- No benchmark process remains after release.

No 500k run, broad scaling claim, or cross-platform performance claim follows
from the 100k gate without a separate plan entry and lease.

## 13. Evidence and closeout

If approved and implemented, the main reports must record:

- The approved addendum hash.
- API migration and compatibility behavior.
- Compiled and extension-absent test commands/results.
- Wheel artifact identifiers and installed import paths.
- 10k and 100k report paths, schema versions, raw samples, memory, and gates.
- Every failure and timeout without omission.
- Remaining platform, scaling, cancellation-latency, and geometry integration
  limitations.

This addendum does not approve implementation, authorize a performance run, or
change ecosystem closeout status.
