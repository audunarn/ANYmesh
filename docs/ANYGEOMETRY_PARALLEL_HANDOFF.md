# Handoff to ANYgeometry: what parallel meshing needs

Date 2026-10-06. From the ANYmesher parallel-meshing work (branch
`claude/parallel-meshing-research`, commits 66de6e1, 07bcaf3, f27d86b, 0f0b127;
background in `docs/PARALLEL_MESHING_STUDY.md`). **No ANYgeometry source was
read for modification or changed.** This note lists requests for whoever owns
ANYgeometry's intersection/preparation work, ordered by value. Every number is
a single run on a synthetic fixture on a shared 32-core Windows machine
(Python 3.14, ANYgeometry 0.4.5 at c6dc426); treat them as sizing, not
qualification.

## Why this matters

ANYmesher now has an opt-in `generate_hybrid_mesh_result_parallel` that meshes
independent components in separate processes (8x to 23x at 16 to 64
components, parity with the serial route up to numbering). Two limits remain,
and both sit on the ANYgeometry side:

1. The independence decision is made by an ANYmesher stand-in that should be
   an owner API (request 1).
2. On a *connected* model nothing can be split, and about 86% of the time is
   `prepare_structural_closure`, mostly ANYgeometry intersection planning and
   application (request 3). Element generation is under 3%.

## Request 1: independence planner (highest value)

**Need.** Given a model and a separation distance, partition the selected
entities into groups that are provably independent, so each group can be
extracted (`extract_model_closure`), meshed alone and joined without
reconciliation.

**Proposed shape** (names are suggestions):

```python
plan_independent_components(
    model, *, face_ids=None, member_ids=None, edge_ids=(), separation: float,
) -> ComponentPartition   # .components, .merge_reasons, .refusals
```

Each component is a set of entity handles closed under:

- shared vertex or edge (faces, member edges, explicit beam edges);
- sheet membership (all selected faces of a sheet, or the whole sheet refused
  if only partly selected);
- declared attachment (member with its target face or edge owners) and
  declared junction participants, regardless of distance;
- `separation`-grown conservative bounds overlap (`GeometryModel.bounds` is
  already conservative and public).

Proof obligations that the stand-in does **not** discharge and the owner can:

- undeclared crossings and coplanar overlaps between components that the bounds
  test might miss for curved supports (it uses `bounds`, which is conservative
  for the support families it lists, so verify coverage of every
  `Face.surface` family);
- eccentric members: node positions can stand off the entity bounds, so the
  caller adds the offset to `separation`; the owner could expose the right
  inflation per member;
- an explicit refusal path with a reason (`merge_reasons[(a, b)]`: "shared
  vertex", "declared junction", "bounds overlap by x") so callers can report
  why a model did not split.

**Reference implementation and behaviour to match:** `plan_independent_components`
in `src/anymesher/component_parallel.py` (union-find over the rules above,
sort-and-sweep box overlap). Tests that pin it:
`tests/test_component_parallel.py` (distant components separate, a gap inside
the padding merges, a shared vertex merges, structure stays together).

**Evidence it is worth owning:** disjoint components meshed in a spawn pool with
16 workers: serial 3.9 s / 10.1 s / 29.0 s at 16 / 32 / 64 components versus
warm-pool 0.33 s / 0.77 s / 1.27 s. The serial time is itself superlinear in
component count (0.19 s per component at 1, 0.30 s at 16) because the whole
model is prepared at once, so a correct partition pays off even before
parallelism.

## Request 2: model transport contract

- `GeometryModel` does not pickle (`mappingproxy` in `vertices` and others).
  The prototype transports `anygeometry.to_dict(model)` and rebuilds with
  `from_dict`; in the tests, ids survive that round trip (the join maps worker
  results back through `ModelClosure.work_to_source`, and parity holds). Please
  state this as a contract (stable local ids across `to_dict`/`from_dict`), or
  provide a picklable form.
- `extract_model_closure` renumbers entities densely and issues a new model id.
  That is correct and was what the join needed; please keep
  `work_to_source` complete for vertices, edges, faces, sheets, members,
  attachments and junctions (the join relies on all of them).
- ANYmesher's returned `Mesh` also fails to pickle because
  `Mesh.boundary_registry` holds a `GeometryMeshingView` of a live model. That
  is an ANYmesher issue; it is mentioned so nobody looks for it here.

## Request 3: preparation cost on connected models

Profile of `generate_hybrid_mesh_result` on a base plate with 16 x-webs and 16
y-webs crossing (33 faces, 8,976 elements), cProfile on, so wall time is
inflated to 16.9 s (8.4 s unprofiled). Reproduce with
`benchmarks/parallel/exp3_coldstart_heavy_grid.py` (`grid(32)`).

| Item | Calls | Cumulative (profiled) | Note |
|---|---|---|---|
| `prepare_structural_closure` | 1 | 14.58 s (86%) | everything below |
| `plan_intersections` | 1 | 6.85 s | |
| `arrange_material` (inside planning) | 66 | 4.16 s | tiny-array numpy work, about 63 ms per call |
| `apply_intersections` | 1 | 6.78 s | |
| `to_dict` (whole model) | 81 | 2.41 s | each call runs `validate_topology` |
| `validate_topology` | 82 | 1.64 s | 81 of them from `to_dict` |
| `query_trimmed_surface_charts` | 33 | 0.93 s | 2 `to_dict` per call (66 of the 81) |
| `definition_checksum` | 10,255 | 1.84 s | recursion, 552k `_definition_value` calls |

Directions, in the order I would try them (all unverified beyond the profile):

1. **Read-only queries should not serialise and validate the whole model.**
   `query_trimmed_surface_charts` accounts for 66 of the 81 `to_dict` calls and
   about 0.7 s of the 2.4 s, and `to_dict` re-runs `validate_topology` each
   time. A revision-keyed cache of the serialised or validated state, or a query
   path that does not need either, removes most of it without changing results.
   Also check `_apply_intersections` (4 calls), `_finalize_edge_subcurve_preimages`
   (3 calls, 0.54 s) and `_compose_application_preimages` for repeated whole-model
   serialisation within one application.
2. **`definition_checksum`**: memoise per entity at a given model revision (it
   recomputes the same recursive structure 10,255 times).
3. **`arrange_material`**: 66 calls at about 63 ms is small-array overhead
   (an earlier read-only drill-down at 64 crossing members found 1.2M tiny
   `norm` calls). Batching those operations would lower the per-crossing
   constant. If the calls are independent per sheet given the plan (to be
   confirmed; I have not checked shared state), they are also a parallel unit,
   but that needs a picklable plan.
4. **Scaling shape**: crossings grow as n^2, and per-crossing cost grows only mildly
   (31 to 44 ms from 64 to 1,024 crossings in the earlier drill-down), so the goal
   is the per-crossing constant, not the algorithm.

Related, on the ANYmesher side (not for ANYgeometry to fix, listed so the two
changes are not confused): `src/anymesher/preparation.py` around lines 847 and
878 calls `query_trimmed_surface_charts` once per face inside a try/except that
classifies per-face "unsupported" errors. Batching it into one call changes
which face raises, so it needs a design choice (a per-operand result or error
report from the query) rather than a mechanical change. If ANYgeometry offers
such a query, ANYmesher can then batch.

Already done on the ANYmesher side (no action): the per-junction rescan of all
shell nodes in `_connect_junction` (f27d86b): 48-member grid connectivity phase
5.69 s to 0.59 s, identical mesh.

## Request 4 (later): dividing one large face

Single large faces are superlinear on the native route: a plate with a hole takes
10.4 s for 4.5k elements and 193 s for 44k (`benchmarks/parallel/exp4_heavy_face.py`),
almost all in `mesh_planar_surface` quality/recombination. Splitting such a face
into regions that are meshed independently with agreed interface divisions would
win even serially. That needs ANYgeometry to supply cut curves and a partition
with an interface station contract (matching boundary stations on both sides).
Existing authored partition work in ANYgeometry
(`authored_partition_coverage`, `authored_boundary_stations`, the planar-hole
and cylinder partitions) looks related; I have not evaluated whether it can be
reused. This depends on request 1 only for naming and extraction.

## Acceptance suggestions

- Planner: a fixture set where bounds are disjoint but supports interact
  (curved/extruded faces, an undeclared crossing, close-but-not-touching sheets)
  must merge or refuse, never split. Compare against the stand-in on the
  `tests/test_component_parallel.py` fixtures.
- Preparation speed-ups: results must stay identical (the `preparation_hash`
  and the mesh digest in `benchmarks/parallel/exp7_junction.py` are convenient
  checks); report call counts and wall time at 16, 32 and 48 crossing members.

## Running the evidence

- Python 3.14; `PYTHONPATH` must be Windows-style paths (Git Bash `$PWD/...`
  entries are ignored by `py.exe`), with the ANYmesher worktree `src` and
  ANYgeometry `src`. Print `anymesher.__file__` to confirm.
- The compiled `anymesher/_native.cp314-win_amd64.pyd` is gitignored; a fresh
  worktree needs it copied from the main checkout, otherwise meshing falls back
  to pure Python (4 to 5 times slower) and some fixtures raise `finite hull
  triangulation legality did not converge`. That error also occurs on main for
  `test_structured_t_junction_accepts_upstream_imprinted_transverse_edge`,
  independent of this work.
- Timing is noisy while other sessions run; repeat before quoting.
