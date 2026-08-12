# Native hybrid mesher decision log

## 2026-08-12: coordinated branches

Created `native_hybrid_mesher` in all four repositories while retaining each
working tree's pre-existing changes.

## 2026-08-12: geometry ownership

The separate `Continue Geometry kernel work` task is the sole writer for
ANYgeometry. This implementation sends requirements to its liaison and consumes
the finalized public contract. It does not vendor geometry audit or overlap
predicates.

## 2026-08-12: package dependencies

ANYmesh base installation requires `ANYgeometry>=0.2,<0.3`. Planar overlap
fragmentation is exposed through the optional `planar` extra. Gmsh remains an
explicit deprecated compatibility extra and is never selected automatically.

ANYfem requires `ANYgeometry[planar]>=0.2,<0.3` and `ANYmesher>=0.2,<0.3`.

## 2026-08-12: mesh storage

The established neutral `Mesh` remains the compatibility facade. Native hot
paths use compact NumPy arrays and stable external IDs. A small optional C++17
extension provides robust predicates and adjacency without becoming a mandatory
runtime dependency.

## 2026-08-12: damage topology

Damage changes dense activity values and masks. It does not renumber or
physically remove persistent element topology. Hard deletion removes numerical
contributions under explicit mass, damping, load, contact, and coupling policy.
