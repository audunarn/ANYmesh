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

## 2026-08-13: final geometry handoff

The authoritative geometry branch tip is
`37234b7bc6b6c3f2e02cf1c53acb875245d9c3aa`, with qualified code parent
`8828019e0f940b0d6f240b98f8be17d6f306155b`. ANYgeometry remains package 0.2.1
within the live API range `>=0.2,<0.3`, reads schemas 1 through 4, and writes
canonical schema 4. Downstream packages use public codecs and model APIs only.

## 2026-08-13: compiled backend selection and migration

Explicit `python` remains available, explicit `native` fails hard when the
compiled capability is absent or corrupt, and `auto` falls back only when the
extension is absent. ANYfem schema 6 writes the project-level selector
explicitly; migrated schema 1 through 5 projects that omitted it retain Python
semantics. Backend request, selection, actual execution, and fallback reason are
recorded without changing in-flight session snapshots.

## 2026-08-13: local installed-wheel path retired

Windows wheel attempts 1 and 2 are preserved as failed transport/dependency
evidence. Attempt 3 timed out without a report. The final bounded Stage A build
attempt timed out without publishing its required atomic bundle. Independent
post-timeout inspection found no remaining compiler/Python process, temporary
tree, or source mutation. No installed-wheel, platform matrix, resolver lock, or
publication-readiness claim follows. Those gates move to dependency-complete CI
or release qualification; no further local build plan is authorized in this
core slice.

## 2026-08-13: consumer resolver boundary

ANYsolver commits `a49fee8` and `925639e` widen its qualified source/runtime
surface to `ANYmesher>=0.1,<0.3` and compare installed requirement metadata
semantically. ANYfileIO still declares `ANYmesher<0.2`; its owner must widen that
range before a combined 0.2.1 resolver lock or publication-readiness claim is
valid.
