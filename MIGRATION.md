# Migration to ANYmesher

## Geometry authority extracted to ANYgeometry (2026-08-08)

The reusable geometry kernel has moved again, this time to its final neutral
owner: the `ANYgeometry` distribution and `anygeometry` import package.

| Previous ANYmesher source | Current owner |
| --- | --- |
| `geometry/entities.py` | `anygeometry.entities` |
| `geometry/curves.py` | `anygeometry.curves` |
| `geometry/chains.py` | `anygeometry.chains` |
| `geometry/model.py` | `anygeometry.model` |
| general operations in `geometry/operations.py` | `anygeometry.operations` |

`anymesher.geometry` remains a compatibility facade and returns the exact owner
classes. Mapped-only preparation did not move: `check_mappable`,
`triangle_to_quads`, and the legacy butterfly implementation of
`punch_circular_hole` are implemented in `anymesher.decomposition` and re-exported
from their old paths. The historical mapped `split_face_at`,
`split_face_between` and `strip_face` paths also remain there so existing mapped
models retain their exact partitioning. ANYgeometry separately owns the neutral,
general-purpose edit operations under its own import path.

The extraction removes the old geometry/mesher ownership ambiguity. ANYmesher
now imports ANYgeometry; ANYgeometry never imports ANYmesher or an FE package.
Because the dependency is not yet published, CI installs the sibling checkout;
release ANYgeometry 0.1 before publishing ANYmesher 0.1 to a package index.
The historical account below describes the earlier extraction from ANYfem into
ANYmesher and is retained as provenance, not as the current dependency boundary.

ANYmesher is a curated extraction, not a filtered-history import, following the
precedent set by `ANYsolver/MIGRATION.md`. It draws on two sources, because the
family currently has two meshers.

## Provenance

From `audunarn/ANYfem` at `245b82ec68496fde1f8880c6a360f69973208bca`
(branch `main`) — the geometry kernel and the mapped mesher:

| Source | Destination |
| --- | --- |
| `src/anyfem/geometry/curves.py` | `anymesher/geometry/curves.py` |
| `src/anyfem/geometry/entities.py` | `anymesher/geometry/entities.py` |
| `src/anyfem/geometry/model.py` | `anymesher/geometry/model.py` |
| `src/anyfem/geometry/operations.py` | `anymesher/geometry/operations.py` |
| `src/anyfem/mesh/mapped.py` | `anymesher/mesh.py`, `anymesher/mapped.py`, `anymesher/geometry/chains.py` |
| `src/anyfem/mesh/seeding.py` | `anymesher/seeding.py` |
| `src/anyfem/mesh/refinement.py` | `anymesher/refinement.py` |

These seven modules move together and cannot be separated:
`geometry/operations.py` imports `chain_breaks`/`chain_point` from
`mesh/mapped.py`, and all three mesh modules import from `geometry`. Splitting
the group would leave a cycle behind in ANYfem.

From `audunarn/ANYsolver` `src/anysolver/mesh_gen.py` at
`8b4553cc680ff925df850e627165fc336615eaba` (branch `extract_mat_mesh_io`) — the
primitives:

| Source | Destination |
| --- | --- |
| `PanelGeometry`, `MeshConfig`, `StiffenerCrossSection` | `anymesher/primitives.py` |
| `_generate_shell_mesh`, `_generate_beam_mesh` | `anymesher/primitives.py` |
| `_edge_node_sets` | `anymesher/primitives.py` |
| shape functions, structured grid index, `_locate_shell_element_at_xy` | `anymesher/coupling.py` |
| `verify_mesh_quality` | `anymesher/quality.py` |

## Verified equivalence

Checked with all three packages importable at once, exact rather than close.

Against `anyfem` at the commit above, 18 mapped-mesher configurations across a
rectangle, a two-face model with an eccentric stiffener on the shared line, and a
ruled cylindrical patch, in linear and quadratic order at three target sizes:
node IDs and coordinates, quads, beams, offset nodes, the structured grid, every
association map, the seeding divisions, the coupling pairs, and the text of every
error raised. Plus the refinement path, `split_face_at`, `triangle_to_quads` and
`punch_circular_hole` — the three operations that lean hardest on the chain
helpers that moved.

Against `anysolver` 0.1.3 at the commit above, 96 stiffened-panel configurations
over four division-count pairs, one to three stiffeners, one and four beam
divisions, Q4 and Q8 shells, and aligned and uniform transverse grids: node and
element IDs, coordinates, connectivity, every coupling's master nodes, shape
weights and eccentricity vector, all four panel edge node sets compared against
the solver's own private `_edge_node_sets`, and the quality report field by field.
Plus four plates, three beams and all five stiffener profile families.

The permanent parity gates live in ANYsolver and ANYfem, because ANYmesher cannot
import either.

## Deliberate behavioural differences

All were reviewed during the coordinated strip:

- **`Mesh.couplings` holds `Coupling` records rather than `(beam, plate)` pairs.**
  The interpolated case needs several plate nodes with weights, which a pair
  cannot express. `Coupling.node_to_node` builds the single-master case and
  `coupling.plate_node` reads it back, so the ANYfem edit is
  `_add_couplings` in `src/anyfem/solve/build.py` plus two lines in
  `tests/test_structural.py`.
- **`Mesh` gained `tris` and `thickness_of_face`.** Both stay empty for the mapped
  mesher, so nothing changes for it.
- **The chain helpers, `GeometryError` and `MeshError` moved.** At the 0.1
  extraction, `chain_breaks`, `chain_point` and `sample_chain` moved to
  `anymesher.geometry.chains`; they are now owned by `anygeometry.chains` and
  the former path is an exact compatibility re-export. `GeometryError` likewise
  belongs to ANYgeometry, while `MeshError` remains in `anymesher.errors`. The
  reason for the original move was a cycle:
  `geometry/operations.py` imported the chain helpers from `mesh/mapped.py` while
  all three mesh modules imported from `geometry`.
- **`GeometryModel.arc_frame(edge_id)` is public.** New method, delegating to the
  existing private `_arc_frame`.
- **`PanelGeometry` and `MeshConfig` became `StiffenedPanel` and
  `PanelMeshConfig`** and lost their material, support-condition and load fields.
  ANYsolver keeps its own classes with those fields and maps onto these. Renamed
  rather than trimmed in place, so a same-named class with different fields cannot
  be mistaken for the original.

## Included

- The geometry model, its curve shapes and its face operations.
- Seeding, refinement and the transfinite Coons mesher.
- Stiffened-panel, simple-panel and beam primitives.
- Beam-to-plating coupling records: the shell element a beam node lands in, the
  shape weights at that point, and the eccentricity vector.
- Aspect-ratio and warp quality metrics.
- A gmsh backend behind the `[gmsh]` extra, returning the same neutral mesh.
- A tkinter editor and a CLI.

## Excluded

- `InterpolatedBeamShellMPCElement` and `RigidLidMPCElement` from
  `anysolver/mesh_gen.py`. They implement `compute_stiffness_matrix`,
  `compute_mass_matrix` and `get_mpc_constraints`: they are solver elements that
  happen to be constructed during meshing. ANYmesher emits the coupling record;
  ANYsolver builds the constraint from it.
- `generate_stiffened_panel_mesh`, `generate_simple_panel_mesh` and
  `generate_beam_mesh` in their current form. They return `FEModel` objects and
  become adapters in ANYsolver over the neutral generators here.
- `_add_boundary_conditions` and `_add_custom_support`. ANYmesher returns the
  node sets per panel edge; interpreting a support string such as `"Integrated"`
  or `"Girder - long"` is structural convention, not mesh topology, and stays in
  ANYsolver.
- Writing a mesh to `.fem`, `.inp` or any other file. That belongs to ANYfileio,
  which depends on this package.

## Numbering policy

Both sources number nodes and elements differently, and both conventions must
survive verbatim or ANYsolver's deterministic baselines
(`tests/fixtures/fe_baselines/baseline.json`) shift:

- `mesh_gen` primitives: shell nodes from 1, beam nodes from 10000, beam
  elements from 20000, coupling elements from 30000.
- The mapped mesher: vertex nodes first, then `n - 1` interior nodes per edge in
  the edge's own direction, then face interior nodes from the Coons blend. This
  order is what makes neighbouring faces share node objects, so conformity never
  depends on coincident-node merging with a tolerance.

The policies belong to the generators, not to the mesh container, and are
asserted by test rather than left to be rediscovered.

## Historical 0.1 import changes

The first extraction made ANYmesher authoritative for geometry and neutral
meshing. That geometry ownership has since moved to ANYgeometry as described at
the top of this document; only neutral meshing remains authoritative here. The
table records the 0.1 transition for provenance, not current import guidance.

| Import before ANYmesher 0.1 | Replacement at 0.1 |
| --- | --- |
| `anyfem.geometry` | `anymesher.geometry` |
| `anyfem.mesh.mapped` | `anymesher.mapped`, `anymesher.mesh` |
| `anyfem.mesh.seeding` | `anymesher.seeding` |
| `anyfem.mesh.refinement` | `anymesher.refinement` |
| `anysolver.mesh_gen` | `anymesher.primitives` (+ solver-side adapter) |

`anyfem/geometry/__init__.py` and `anyfem/mesh/__init__.py` are re-export
shims, so the fourteen ANYfem modules that import `EntityRef` need no edit.

## Tests

Migrated verbatim from ANYfem, with only import paths rewritten:
`test_geometry.py`, `test_mesh.py`, `test_curves.py`, `test_seeding.py` — 42
tests, all passing unchanged against the moved code.

`test_refinement.py`, `test_operations.py` and `test_decomposition.py` were not
migrated wholesale: they also drive an ANYfem `Project`, a solve, a `BeamSection`
or a `Collision`, none of which belong here. The geometry-only half is covered by
a new `tests/test_operations.py`; the rest stays with ANYfem, where the code it
exercises stays too.
