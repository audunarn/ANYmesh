# Architecture

## Position in the family

```
ANYmaterial ──┐
              ├──→ ANYfileio ──┐
ANYmesher ────┘                ├──→ ANYsolver ──→ ANYfem
                               │         └──────→ ANYstructure
                               └────────────────→ ANYstructure
```

ANYmesher is a leaf. It imports numpy and the standard library, and nothing else
in the family. ANYsolver and ANYfileio both depend on it, so any import in the
other direction closes a cycle. `tests/test_layering.py` enforces this by
walking the AST of every module.

ANYmaterial is forbidden here too, though it would not close a cycle. A mesh is
geometry and topology; what an element is made of is somebody else's field on
somebody else's model. Keeping the two apart is what allows a mesh to be
generated, saved and re-meshed without ever resolving a steel grade.

Writing a mesh to `.fem` or `.inp` belongs to ANYfileio, which depends on this
package. That is a consequence of the arrow above rather than a preference: an
export function here would need the file layer, and the file layer already needs
the mesh.

## The neutral mesh

`generate_mesh` returns nodes, quadrilaterals, beams, coupling records, and the
**association** back to the geometry: which node came from which vertex, which
nodes lie along which edge, which elements belong to which face. The association
is the point. It is what lets a load or a restraint be named against geometry and
survive a re-mesh, and it is what makes results addressable by the thing the user
drew rather than by node number.

An imported mesh has no geometry behind it, so it carries element groups instead
of a structured grid. Everything downstream goes through the association, which
is why both kinds work through the same code.

## What a coupling record is, and is not

When a stiffener stands off the plating, its beam nodes are not shell nodes. The
mesher records, per beam node: the shell element it projects into, the shape
weights at that point, and the eccentricity vector. That is a statement about
geometry — this point is here, inside that element, offset by that much.

It is not a constraint. The consuming solver decides that the record becomes six
multi-point constraints tying the beam node's translations and rotations to the
shell nodes' with rigid-offset terms. Interpolating through the shell shape
functions is what removes the older requirement that a beam node lie exactly on a
shell node row, so the mesh no longer has to be aligned to the stiffeners for the
coupling to be exact.

## Conformity by construction

Node generation order is fixed, not incidental: one node per used vertex, then
`n - 1` interior nodes per edge stored in the edge's own direction, then face
interior nodes. Faces look their boundary nodes up from the vertex and edge
registries and reverse the list when they traverse an edge backwards, so
neighbouring faces share the very same nodes.

The alternative — meshing faces independently and merging coincident nodes within
a tolerance — fails quietly on nearly-coincident geometry and produces a mesh
that looks connected and is not. Ordering makes the guarantee structural, so
there is no tolerance to tune and nothing to get wrong at small feature sizes.

## Numbering is a contract

Two numbering conventions exist and both are load-bearing. The primitives number
shell nodes from 1, beam nodes from 10000, beam elements from 20000 and couplings
from 30000; the mapped mesher numbers by registry order. ANYsolver's deterministic
baselines record results per node and element ID, so a renumbering that is
mathematically irrelevant still invalidates them.

The numbering therefore belongs to the generator that produces it, is documented
where it is implemented, and is asserted by test. It is not a detail the mesh
container is free to normalize.

## No cycles inside the package either

The same rule that keeps ANYmesher below ANYsolver applies within it. Two edges
had to be cut to make the extraction acyclic:

- The **chain-sampling helpers** live in `geometry/chains.py`. They used to live
  with the mapped mesher, and `geometry/operations.py` imported them from there —
  so the geometry package depended on the mesher while all three mesh modules
  depended on the geometry package. Sampling a chain of edges by arc length is a
  geometry question, so that is where it belongs.
- The **exception types** live in `errors.py`. `GeometryError` used to sit with the
  geometry model and `MeshError` with the mesher, which worked until the chain
  helpers moved and needed to raise `MeshError` without importing the mesher.

Both are still importable from their old module paths, so the move is invisible to
a caller.

## Two backends, two sets of guarantees

`generate_mesh(..., backend=...)` dispatches. Both backends return the same
container, so quality metrics, association lookups and export work either way.
They do not promise the same things, and the container says so rather than
pretending:

- The mapped mesher fills `grid_of_face` with an `(i, j)` index and produces quads
  only. gmsh leaves it empty and may produce triangles.
- gmsh meshes planar faces; it refuses a curved patch and names the mapped mesher,
  which meshes one exactly.

An unstructured mesh with a plausible-looking grid attached would be worse than
one that admits it has none, because the grid would silently be wrong.

## Units

SI throughout, lengths in m. No conversion layer.
