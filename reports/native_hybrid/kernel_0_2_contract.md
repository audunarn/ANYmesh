# ANYgeometry 0.2 meshing contract

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

The coordinated kernel task owns these APIs and their final signatures:

- selected structural closure extraction with bidirectional handle maps;
- batched edge evaluation and tangents;
- batched face evaluation, derivatives, normals, and projection;
- `audit_changed_region`, which is local and never full certification;
- indexed `find_coplanar_overlaps` scopes for faces, changed AABBs, or supplied
  candidate pairs.

ANYmesher feature-detects these additions during coordinated development and
does not vendor alternate geometry predicates.
