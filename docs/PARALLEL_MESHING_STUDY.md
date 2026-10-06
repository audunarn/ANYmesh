# Parallel meshing study (living note)

Branch `claude/parallel-meshing-research`, started 2026-10-06 from `main` 7182cb3.
Python 3.14.2 (GIL build), ANYgeometry 0.4.5 dev checkout, 32 logical CPUs,
native extension present. Experiments: `benchmarks/parallel/exp*.py`.

## Question

Can the mesher use several cores by (a) meshing unconnected regions
separately and joining them, or (b) dividing a connected model into regions,
meshing each and reconciling the shared interface? What must ANYgeometry
provide?

## Measured so far

All numbers are single runs on synthetic fixtures; they size the opportunity,
they are not qualification evidence.

1. **Disjoint copies of a small structural model, whole-model run (exp1).**
   Per-component time grows with N although the components are independent:
   0.19 s (N=1..4), 0.23 s (N=8), 0.30 s (N=16). Time is preparation
   (`structural_preparation` + `geometry_and_preflight`, about 90%); element
   generation is about 1%. Splitting helps even serially.
2. **Per-component meshing via `extract_model_closure` (exp2).**
   Same node/element totals as the whole run (1456/1456 nodes, 1424/1424
   elements at N=8). Extraction is 0.01-0.03 s. 8 workers: N=8 1.70 s whole
   -> 0.21 s; N=16 4.86 s -> 0.44 s (process pool, warm). **Threads are slower
   than serial** (2.65 s / 5.38 s): the work is Python-bound, so only
   processes help. Spawn cold start is 0.5 s (4 workers) to 0.85 s (16).
3. **Connected crossing grid (exp3)** - the case component splitting cannot
   help. n=32 members: 12.2 s; preparation 76%, `_connect_junction` about
   13%, element generation under 3%. Inside preparation: `arrange_material`
   (one call per sheet, 66 calls) about 60% of intersection planning; whole-model
   `to_dict` 81 calls / `validate_topology` 82 calls about 13% of the run.
4. **Element-heavy single faces (exp4).** Mapped/pentagon faces mesh 37-43k
   elements in 1.3-1.5 s. A plate with a hole on the native route
   (`anymesher-cpp17`) takes 10.4 s for 4.5k elements and 193 s for 44k:
   superlinear (10x elements, 18x time) and almost all inside
   `mesh_planar_surface` quality/recombination. A single large face is
   therefore a case where dividing the face into regions could win even
   serially (design C below). A profile of this case taken without the compiled
   extension fell back to Python and is not representative; see Environment.

## Transport findings (block any process-based design today)

- `GeometryModel` does not pickle (`mappingproxy`). `anygeometry.to_dict` /
  `from_dict` round-trips it and works as the input transport.
- A returned `Mesh` does not pickle either: `mesh.boundary_registry` holds a
  `GeometryMeshingView` bound to the live model. The experiment drops it; a real
  design needs a transportable registry (or rebuild it after the merge).
  Result size is small (0.23 MB for 8 components).

## Candidate designs

- **A. Component parallelism (exact, simplest).** ANYgeometry partitions the
  model into independent closures; workers mesh closures; ANYmesher merges by
  id offset and handle remap through `ModelClosure.work_to_source`. Disjoint
  components share no nodes, so no reconciliation. The existing
  `build_structural_components` is declared-connectivity only; independence
  must also hold geometrically (padded bounding boxes, undeclared crossings,
  coplanar overlaps), and must fail closed by merging components that could
  interact.
- **B. Phase parallelism inside a connected model.** Per-sheet
  `arrange_material`, pair-chunked intersection evaluation, per-junction
  connectivity. Needs ANYgeometry APIs with transportable inputs; the first
  wins here are serial (drop repeated `to_dict`/`validate_topology`, the
  `_connect_junction` scan), see the many-member scaling note.
- **C. Region division of a connected model or one large face.** After
  preparation and seeding the interface edge divisions are fixed, so regions
  mesh independently and join on shared edge nodes. ANYgeometry supplies the
  cut curves/partition and the interface station contract. Only worthwhile when
  meshing dominates; not yet measured on the native route.

## Prototype: component-parallel route (design A)

`src/anymesher/component_parallel.py`, `tests/test_component_parallel.py`
(13 tests), `benchmarks/parallel/exp5_parity.py`, `exp6_speed.py`.

- `plan_independent_components`: union-find over shared vertices, sheet
  faces, declared attachments and junctions, plus padded conservative AABB
  overlap (`GeometryModel.bounds`). Pad is `target_size` (plus any beam
  offset), so components closer than one element merge. Partial
  `face_ids`/`member_ids` selections are not planned.
- `generate_hybrid_mesh_result_parallel`: closure per component, `to_dict`
  transport, spawn process pool (optionally a caller-supplied warm executor),
  join by node/element id offsets and `work_to_source` handle remap, boundary
  registry rebuilt from the workers' entries. Falls back to the serial route,
  recording the reason in `mesh.hybrid_diagnostics["parallel"]`, for: one
  component, seeding / quad options / change set / audit policy / face or
  member subsets, certification, non-read-only mutation policy, `overrides` or
  `beam_offsets` not owned by a component.
- **Parity** (numbering-independent: node positions, per-face and per-member
  element centroids, per-edge node sets, vertex nodes, strategies, registry
  size, junction edges, couplings with eccentricities, offset-node edges)
  holds at 3-16 components, including eccentric stiffeners that produce
  couplings.
- **Speed** (exp6, 32 cores, machine shared with other sessions, single runs):

  | components | serial | 8 workers cold / warm | 16 workers cold / warm |
  |---|---|---|---|
  | 16 | 3.9 s | 1.5 / 0.49 s | 1.6 / 0.33 s |
  | 32 | 10.1 s | 2.5 / 1.07 s | 2.2 / 0.77 s |
  | 64 | 29.0 s | 2.8 / 1.95 s | 2.6 / 1.27 s |

  Speed-up exceeds the worker count because the serial route is superlinear in
  component count. Plan + extract + serialise + join cost is about 0.2 s at 64.
  Cold start (spawn, import) is 1.2-1.5 s, so a warm executor matters for
  interactive use.
- **Known gaps**: no audit report or structural-preparation report on this
  route (`None`); `preflight`/`connectivity` are per-component records in
  prepared-model ids (the serial route does not map these to source ids either),
  with only node and coupling ids offset; numbering differs from the serial run;
  not wired into any public entry point.

## Step 1 done: junction connectivity scan (f27d86b)

`StructuralMeshingPipeline._connect_junction` rebuilt the set of all shell
nodes per junction, and `_replace_beam_nodes` rescanned every shell and beam.
The shell-node set is now computed once per `apply_connectivity` (lazily, only
when a junction row is connected) and the prune step tests only replaced nodes.
Exp 7 (A/B, main checkout vs worktree): crossing grid n=16/32/48 connectivity
0.12/1.19/5.69 s -> 0.06/0.25/0.59 s; wall 25.8 -> 20.9 s at n=48; digest of
nodes, elements and couplings identical (numbering-exact) and the same action
count. Two new endpoint-junction tests pass on old and new code.
Not covered: the remaining preparation cost, which is ANYgeometry's.

## Step 2 done: opt-in public entry point

`anymesher.generate_hybrid_mesh_result_parallel`, `ParallelOptions`,
`plan_independent_components` are exported from the package root. It is a
separate function, not a keyword on `generate_hybrid_mesh_result`: that keeps
`hybrid.py` untouched while other work is landing there and avoids a circular
import. A `parallel=` keyword can be added later as a thin dispatch. Added:
worker-death -> `MeshError`, worker cap 61 (Windows), serial-equivalent
`hybrid_diagnostics` keys, tests for refinements, quadratic order,
diagnostics, dead pool, exports; CHANGELOG entry.

## ANYgeometry request (independence planner)

Replace the ANYmesher stand-in with an owner API, e.g.
`plan_independent_components(model, *, face_ids=None, member_ids=None,
separation) -> tuple[ComponentSelection, ...]`, where each selection is a set
of entity handles closed under: shared vertex/edge, sheet membership,
declared attachment/junction participants, plus a proof obligation that the
`separation`-grown conservative bounds of different selections are disjoint
(and that no undeclared crossing or coplanar overlap exists between them).
It should also accept partial selections (include or refuse sheets that are
only partly selected) and report why two selections were merged. A picklable
model transport (or documented `to_dict`/`from_dict` contract with stable
ids) would remove the `mappingproxy` workaround. No ANYgeometry source was
changed.

## Environment pitfalls found

- The compiled `anymesher/_native.*.pyd` is gitignored; a fresh worktree has no
  extension and silently meshes on the pure-Python route (about 4-5x slower on
  these fixtures, and it can fail with `finite hull triangulation legality did
  not converge`). Copy the `.pyd` from the main checkout into the worktree
  `src/anymesher/` before measuring.
- Under Git Bash `PYTHONPATH="$PWD/src;..."` is not converted by `py.exe`; the
  import silently resolved to the main checkout (editable install). Use
  Windows-style paths and print `anymesher.__file__`.
- `tests/test_layering.py` has two failures already on main 7182cb3
  (`_authored_project_references.py` imports `anyfem`); unrelated to this work.

## Open

- Real multi-body fixture (not copies), including undeclared close approach.
- Design C: split one large face (hole plate: 193 s for 44k elements) with
  fixed interface divisions and join on shared edge nodes; needs ANYgeometry
  cut-curve/partition support.
- Serial hot spots for connected models in ANYgeometry preparation (repeated
  `to_dict`/`validate_topology`, per-sheet `arrange_material`): handoff only.
- Whether the flows that call the mesher (ANYfem/GUI) can hold a warm pool.
