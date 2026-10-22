# Entity mesh sizes (`Refinement(mode="override")`)

Two different things used to share one `Refinement`:

| mode | meaning | coarser than the global target |
| --- | --- | --- |
| `"refine"` (default) | spatial zone: smaller elements within a radius of an entity, grading back to the target | refused (unchanged) |
| `"override"` | the bound **face or edge is meshed at this size**, finer or coarser than the target | allowed |

An override binds only to a `face` (plate) or an `edge` (beam line). It is never a
sampled distance zone, so it cannot leak into neighbouring entities. A request equal
to the global target is a no-op and keeps every regular fast path and the exact mesh.
Two overrides on one entity resolve to the finer; a spatial refine zone can still
reduce below an override target.

## Validation and the whole-selection collapse

`hybrid._collapse_entity_targets` first validates every request against the
**original** global contract (a missing entity or a refine-only zone coarser than the
original target is an error and can never vanish into a fast path), drops no-op
overrides, and then - if every selected face and beam carries the same override -
makes that size the target. Admitted refine-only zones that cannot refine the new
(finer) target are inert and dropped. An isolated plate or beam therefore takes the
route, and produces the mesh, of the equivalent global target (coarser or finer).

## Mixed plates (all strategies)

`SizeField.face_sizes` / `edge_sizes`: a shell edge takes the finest size of its
adjacent *active* plates, so interfaces conform and neither side is under-resolved.

* **Mapped / native / auto** - each native plate is meshed through
  `SizeField.face_view(face)` (its own target; no other plate's size). Native plates
  keep the tensor-seed path whenever no spatial zone exists.
* **Quad-first and SmartMapped** - one call, one shared `BoundaryStationRegistry`.
  Its counts come from the entity field (`edge_demand`, finest adjacent plate), so
  the canonical stations on a shared edge are one chain. Each plate's seed, Q4 MCF and
  Q5 run at that plate's own target through a per-plate view of the field and of the
  registry (same chains, plate target). SmartMapped keeps its regular-core audits;
  cells next to a coarse/fine interface adapt through the existing adaptive path.
  With no entity size the plate view *is* the original field and registry object.
  A quad-first/legacy mix (`quad_face_ids`) pins the quad/legacy interface edges to the
  same finest count on both sides.

## Beams

A beam target never regrades the shell chain. When the beam's own count differs from
the canonical `nodes_of_edge` chain it is rebuilt on its own stations
(`anymesher._independent_beams`, after native faces / after the quad merge):

* stations that coincide with chain nodes (always the vertex ends) reuse them; an
  eccentric station reuses an existing offset node of a neighbouring beam at the same
  chain node and displacement, so **shared joints keep their node and coupling**;
* other stations are new nodes with an interpolation `Coupling` over the actual chain
  (linear, or quadratic Lagrange for Q8/T6), eccentricity read back from positions;
* declared attachment and junction point stations on the edge stay beam stations;
* a shared node the rebuild keeps or reuses ends with exactly one coupling record
  (the ordinary emitter can leave one identical record per touching beam; exact
  duplicates collapse to the lowest record id, incompatible ones raise `MeshError`;
  untouched nodes keep the generated output);
* only a node no other beam uses (and its couplings) is deleted; no unused offset
  chain or coupling is left; `offset_nodes_of_edge` is not written for such an edge.

Cancellation: the hybrid `cancellation_check` is passed to the post-pass and polled
before the request-wide indexes, during the shell index, per edge and every 64
stations inside an edge; a cancelled call raises out of the (working-copy) candidate
and publishes nothing. A beam asking for more than the seeding per-edge bound
(100000 divisions) is a typed `MeshError` before any station array is allocated.
Declared attachment/junction point stations (owner `Attachment.member_id/member_range`
and `Junction.member_uses[*]`) on members that use the edge are kept as stations; one
that cannot be resolved is a typed `MeshError`, never skipped.

Ownership is carried by the mesh itself: `StructuralMeshingPipeline` decides an edge
is independent from its beam elements against the canonical chain (a strict subset of
it, or an own station that carries a coupling) and reads actual stations from the
elements and couplings. `mesh.hybrid_diagnostics["independent_beam_edges"]` is only a
record. All lookups are request scoped (built once per call); nothing is global.
