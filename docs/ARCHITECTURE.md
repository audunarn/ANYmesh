# Architecture

## Position in the family

```text
ANYgeometry ---> ANYmesher
      |              |
      +--------------+---> ANYfem
      +-------------------> ANYstructure
```

ANYgeometry is the single geometry authority. ANYmesher imports it and owns
only meshing concerns: edge seeding, mapped-face preparation, node and element
generation, local refinement, mesh quality, geometry-to-mesh associations and
the optional Gmsh backend. ANYfem and ANYstructure consume the same geometry
objects directly; there is no copied or adapted geometry model.

ANYmesher must not import ANYfem, ANYstructure, ANYsolver, ANYfileio or
ANYmaterial. Any such reverse dependency would either close a cycle or make a
neutral mesh depend on analysis data. `tests/test_layering.py` checks this by
walking the source AST, and also checks that every unconditional third-party
import is declared.

Writing a mesh to `.fem` or `.inp` belongs to ANYfileio. Materials, elements,
assembly and solution likewise stay in their owning packages.

## Geometry extraction and compatibility

The BRep, curve and surface evaluation, topology edits, semantic groups,
history and stable `EntityRef` type belong to ANYgeometry. New code imports
them there.

Temporary modules under `anymesher.geometry` re-export the owner objects. These
are aliases, not wrapper classes, so all of these statements remain true:

```python
from anygeometry import EntityRef, GeometryModel
from anymesher import EntityRef as LegacyRef
from anymesher.geometry import GeometryModel as LegacyModel

assert LegacyRef is EntityRef
assert LegacyModel is GeometryModel
```

General chain sampling and `GeometryError` are also owner exports. `MeshError`
remains local to ANYmesher.

Mapped preparation stays explicitly on the meshing side:

- `check_mappable`, which assesses the four-side mapped parameterization;
- the historical `split_face_at`, `split_face_between` and `strip_face`
  operations, which partition that parameterization while preserving established
  mapped-mesh behavior;
- `triangle_to_quads`, which creates non-degenerate mapped quad patches;
- `punch_circular_hole`, the historical four-patch butterfly/O-grid helper.

They live in `anymesher.decomposition`. The old
`anymesher.geometry.operations` module re-exports them for compatibility and
aliases ANYgeometry's `surface_point`. General-purpose neutral split, strip,
trim, hole, projection and transform operations are imported from ANYgeometry
under that package's own path. A neutral triangular or trimmed face is valid
geometry; the built-in mapped backend diagnoses that it needs partitioning
before it attempts seeding or mesh construction.

## The neutral mesh

`generate_mesh` returns nodes, quadrilaterals, triangles, beams, coupling
records and the association back to geometry: which node came from which
vertex, which nodes lie on each edge and which elements belong to each face.
That association lets loads, supports, mesh controls and results use stable
geometry references across remeshing.

An imported mesh has no live geometry behind it, so it carries element groups
instead of a structured face grid. Consumers still use the same association
lookups.

## Conformity by topology

Node generation order is fixed: one node per used vertex, then edge-interior
nodes in the edge's own direction, then face-interior nodes. Adjacent faces
look up the same shared edge nodes and reverse traversal order where necessary.
They therefore share node IDs by construction, without merging coincident
coordinates after meshing.

This is why shell intersections must be fragmented into real shared
ANYgeometry edges before meshing. A tolerance-based mesh-node merge can make
nearly coincident geometry look connected while leaving invalid topology.

## Coupling records

When an eccentric beam stands off a shell, ANYmesher records the shell element,
shape weights and eccentricity associated with each beam node. This is a mesh
relationship, not an FE constraint. The consuming workflow decides how to turn
it into solver MPC equations.

## Numbering contract

Numbering is deterministic and tested because consumers store results by node
and element ID. The direct primitives retain their established offsets, while
the mapped backend numbers by deterministic registry order. A mathematically
irrelevant renumbering can still invalidate project and verification baselines.

## Backend guarantees

Both backends return `Mesh`, but their guarantees differ:

- The built-in mapped backend fills `grid_of_face`, produces quads and supports
  the qualified curved Coons patches.
- Gmsh leaves `grid_of_face` empty, may retain triangles and currently accepts
  planar faces only. Their straight, circular-arc and Bezier-spline boundaries
  are rebuilt as exact Gmsh curve primitives. Auxiliary circle-centre and
  spline-control points are not imported as disconnected mesh nodes.

The container records those differences instead of inventing a structured grid
for an unstructured mesh.

## Serialization and units

`anymesher.serialize` owns only mesh JSON. Geometry serialization belongs to
ANYgeometry; FEM/project serialization belongs to the relevant application.

Library geometry and mesh coordinates use SI metres. The GUI may accept common
engineering display units and converts them at its boundary.
