# ANYgeometry 0.2 meshing contract

The qualified handoff is ANYgeometry 0.2.1 on branch tip
`37234b7bc6b6c3f2e02cf1c53acb875245d9c3aa`; its qualified code parent is
`8828019e0f940b0d6f240b98f8be17d6f306155b`. The live Python dependency range
remains `ANYgeometry>=0.2,<0.3`.

ANYgeometry 0.2.1 reads schemas 1 through 4 and writes canonical checksummed
schema 4. Loading an older document is a deterministic one-way migration.
ANYgeometry 0.2.0 readers reject schema 4, and legacy relationship evidence
without residual/tolerance data remains `UNVERIFIED` and non-certifying.
ANYmesher consumes public codecs and live model APIs only; it does not parse
geometry document internals.

ANYmesher consumes the kernel as the only geometry authority.

## Identity and state

- Persistent cross-package references use `EntityHandle(model_id, kind, id)`.
- A mesh captures `geometry_model_id` and `geometry_revision`.
- Wrong-model handle queries fail explicitly.
- Geometry records are immutable and changed only through public owner methods.

## Structural ownership

- Shell ownership is `Part -> Sheet -> FaceUse -> Coedge`.
- Beam ownership is `Part -> Member -> MemberEdgeUse`.
- Connectivity comes only from `Attachment` and `Junction` in the normal path.
- Coordinate coincidence does not create structural continuity.

## Computation

- Geometry decisions use the model `TolerancePolicy`.
- Intentional welding uses merge tolerance; computational predicates use
  computational tolerance.
- Geometry broad phase uses kernel spatial candidates and changed AABBs.
- Topology mutations always name an explicit `MutationPolicy`.

## Mesher-enabling public APIs

The frozen public surface is:

- `extract_model_closure`
- `evaluate_edge_many`
- `edge_tangent_many`
- `evaluate_face_many`
- `face_derivatives_many`
- `face_normal_many`
- `project_to_face_many`
- `audit_changed_region`
- `find_coplanar_overlaps`
- `query_intersection`
- `plan_imprint`
- `apply_imprint`

`audit_changed_region` is local and never certifying. Coplanar-overlap selectors
use the maintained geometry index. Qualified face/face `CONNECT` persists a
shared edge and both Sheets' FaceUse/Coedge incidence. Unsupported spline,
mixed-arc, and unqualified non-Plane/Cylinder cases fail closed with typed
results; an empty result is never inferred to mean disjoint.

ANYmesher does not vendor alternate geometry predicates, tolerance policy,
structural ownership, or geometry broad phases.
