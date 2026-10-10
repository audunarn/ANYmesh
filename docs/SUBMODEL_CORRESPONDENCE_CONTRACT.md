# Submodel point correspondence contract

Status: contract proposed in C0 (2026-10-09), implemented for planar linear shells in slice
S3a on branch `claude/submodel-s3a` (`src/anymesher/submodel_correspondence.py`,
`tests/test_submodel_correspondence.py`) and for linear shells on a cylinder in slice S3b on
branch `claude/submodel-s3b` (`tests/test_submodel_correspondence_cylinder.py`). This document states the current rules; section 7
records how they changed. Plan and evidence:
`C:/Github/ANYfem/docs/handoff/submodelling-20261009/`.

ANYmesher owns the discrete question "which source element, and where in it, corresponds to
this submodel node". It does not evaluate displacements; ANYsolver does that from the hit
(`ANYsolver/docs/SUBMODEL_TRANSFER_CONTRACT.md`).

## 1. `TransferHit`

| Field | Meaning |
|---|---|
| `target_node` | Submodel node id |
| `source_element_id`, `source_element_type` | Selected source element, `Q4` or `T3` |
| `source_node_ids` | Its connectivity, in element order |
| `natural`, `weights` | Natural coordinates and weights (Q4 bilinear in `(ξ, η)`, T3 `(1 − r − s, r, s)`) |
| `x_p` | The element's own map `Σ N_i x_i` at `natural`, the point ANYsolver evaluates |
| `distance` | `‖x_f − x_p‖` |
| `inverse_residual` | Planar route: the in-plane residual of the inverse map. Cylindrical route: a physical position uncertainty of `x_p` (the parameter-plane residual times a domain-wide stretch bound) |
| `incident_source_faces` | Every source face of every hit element |
| `conforming` | True for every returned hit (non-conforming hits refuse) |
| `weight_disagreement` | Largest node-indexed weight difference between hits |
| `candidates` | Elements examined after the bounding-box query |
| `alternatives` | The other conforming hits, `(element_id, natural, inverse_residual)` |
| `parametric`, `bound`, `a_priori_bound`, `float_allowance` | Cylindrical route (S3b): located in the parameter plane; the computed bound on `distance`, the element's `R dtheta^2/8`, and the floating-point allowance of the check |

## 2. Query

`locate_for_transfer(source_mesh, points, *, candidate_elements, physical_tolerance,
natural_tolerance, weight_tolerance, bounds_padding, require_faces) -> dict[int, TransferHit]`,
keyed by target node in ascending order; deterministic.

- **Lineage**: `candidate_elements` maps each target node to the source elements it may use
  (ANYfem derives them: submodel face, source faces, the source mesh's elements of those
  faces, filtered by activity). A node is never resolved across unrelated sheets.
- **Tolerances**, all finite and non-negative: `bounds_padding` (≥ `physical_tolerance`) pads
  the bounding-box query; `physical_tolerance` bounds `distance` (planar default `1e-9` times
  the mesh extent); `natural_tolerance` is the slack of the inside test and the zero-weight
  threshold; `weight_tolerance` bounds weight disagreement between hits.
- **Inverse map** (planar slice): explicit outcomes. A Q4 warped by more than `1e-9` of its
  size refuses (`ELEMENT_NOT_PLANAR`); zero-area, folded or degenerate elements refuse
  (`ELEMENT_DEGENERATE`); a non-converged Newton inverse refuses (`NOT_CONVERGED`). A failing
  candidate is never skipped.
- **Several hits** (shared edge or vertex): admitted only when all hits share the same
  interpolating nodes (weights below `natural_tolerance` count as zero) **and** their
  node-indexed weights agree within `weight_tolerance`. Otherwise `NONCONFORMING`, which
  covers coincident or disconnected sheets and non-conforming edges. The other hits are
  returned as `alternatives`, so ANYsolver bounds their value spread.
- **Faces**: with `require_faces` (default) every hit element must belong to a source face
  (`FACE_MISSING`), so the junction rule never runs on incomplete data.
- **Cylinders** (slice S3b, the optional `cylinder=CylinderFrame(origin, axis, radius,
  surface_tolerance)` argument). Points and candidate elements are located in the cylinder's
  unwrapped parameter plane `(s, z) = (R theta, axial)`, over all lineage-allowed candidates with
  a periodic prefilter (no 3D search; `physical_tolerance` and `bounds_padding` are neither used
  nor validated). The point and every node of every candidate element must lie on the cylinder
  within `surface_tolerance` and be finite. An element's angular span must be below `pi/2`. The
  inverse map and the support and weight-agreement rules are those of the planar route, in the
  parameter plane, where a conforming curved mesh tiles the plane. `distance = |x_f - x_p|` to
  the actual target is checked against a bound computed per hit (radial deviations, residual and
  the interpolation error of the cylinder map) plus a floating-point allowance; both are reported
  with the a-priori `R dtheta^2/8`. `inverse_residual` is a physical position uncertainty of
  `x_p`, as ANYsolver expects: the parameter-plane residual times
  `max |J_phys| / min sigma_min(J_param)` over the slack-expanded natural domain (both Jacobians
  are affine in the natural coordinates, so the bound holds over the whole element, not only at
  the hit); a non-positive lower bound refuses as degenerate. The derivation is in
  `ANYfem/docs/SUBMODEL_WORKFLOW_CONTRACT.md`, section 12.

## 3. Refusals (typed, `CorrespondenceError.code`)

`CORRESPONDENCE_NO_LINEAGE`, `_POINT`, `_TOLERANCE`, `_NOT_LOCATED`,
`_ELEMENT_NOT_ADMITTED`, `_ELEMENT_NOT_PLANAR`, `_ELEMENT_DEGENERATE`, `_NOT_CONVERGED`,
`_EMPTY_SUPPORT`, `_NONCONFORMING`, `_FACE_MISSING`; on a cylinder also `_OFF_SURFACE`,
`_FACET_TOO_COARSE`, `_FACET_BOUND` and `_NONFINITE`.

## 4. Evidence

S3b: `tests/test_submodel_correspondence_cylinder.py` (26 collected cases) covers the two review
counterexamples (two triangles sharing an edge; a point outside the parametric element),
random points within the bound and reproducing `(theta, z)` on aligned and rotated cylinders,
arc midpoints across the angle wrap without a physical tolerance, opposite radial deviations,
the natural-coordinate slack at edges, T3 vertices and Q4 corners (prefilter), non-matching
triangulations, roundoff at generator edges, tiny spans and far translations, an injected
inconsistency, one-shot candidate iterables, planar controls ignored, a non-finite candidate,
the position uncertainty on a distorted Q4 with a controlled residual,
and the refusals. S3a's planar tests are unchanged.

Tests cover linear reproduction (Q4, T3), shared edges and vertices, disconnected sheets
(including a permissive weight tolerance), lineage filtering, tolerance separation,
quadratic refusal, determinism, warped and degenerate elements, near-edge disagreement,
face membership, and the element-map `x_p` on a sub-threshold warp. The cross-repository A1
check uses this locator end to end.

## 5. Related owner findings (independent of submodelling)

`handoff/submodelling-20261009/s0/repro_hole_routes.py`: a neutral trimmed hole fails
`prepare_qualified_s3_mesh` admission, and butterfly patches fail ANYfem structural
preparation in multi-face models. A separate task investigates both.

## 6. Open for owner review

- Adopt the tolerances and the planarity limit.
- Whether `locate_for_transfer` should later move onto `MeshElementBVH` internals.

## 7. History

- C0 (2026-10-09): first contract; tuple return, support-set conformity only.
- S3a review round 1 (GPT-6-Astra via Codex): own planar inverse with typed outcomes, weight
  agreement, complete face membership, dict return.
- Round 2: `x_p` is the element map; alternatives returned.
- Round 3: topological support-set check restored alongside weight agreement; this
  consolidation.
- S1 round 5: alternatives carry their own inverse residual.
