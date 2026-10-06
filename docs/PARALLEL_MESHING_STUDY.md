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
4. **Element-heavy single faces (exp4).** Mapped rectangle 37k elements 1.3 s
   (mostly the connectivity/validation phase, 1.2 s). Plate with a hole on the
   default route selected the **Python** backend (`native_capability_absent`):
   4.5k elements 14 s, 44k elements 237 s, superlinear. That is the fallback
   quality path, not the production native route; do not read it as native
   scaling. A native-frontal large-face measurement is still missing.

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

## Open

- Real multi-body fixture (not copies), including undeclared close approach.
- Native-frontal large-face time, to decide whether C is worth anything.
- Merge implementation and conformity check on the joined mesh.
