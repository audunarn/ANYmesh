# Quality targets and explicit enforcement

`SurfaceMeshOptions` uses 30 degrees as the preferred minimum-angle target.
Its default `enforce_quality=False` lets refinement improve toward the targets
without making every target a condition of publication. The scaled Jacobian,
aspect, angle, warpage and growth targets retain their existing values.

Mesh validity is always required. Invalid topology, non-finite or inverted
elements, invalid owner projection, broken conformity, and changed protected
boundary or midside identities remain errors regardless of target enforcement.

For cylindrical quadratic components, `quality_enforced` reports the internal
component's `SurfaceMeshOptions.enforce_quality` setting. The public final
whole-result validator independently enforces explicit `quality_options`.
`target_met` combines the component's
physical/per-face targets with the promoted mesh's quality report. A valid
default result can be accepted with `target_met=False`; its reported violations
are retained. `accepted` does not mean every advisory target was met.

Explicit `SurfaceMeshOptions(enforce_quality=True)` remains strict. Public
`quality_options` likewise invokes the existing final owner validation of all
five shape fields and adjacency growth; a miss rejects the result. A concrete
structured policy retains its own existing defaults, including its separate
15-degree minimum-angle policy. This does not replace the 30-degree surface
target or change either set of thresholds.

The private eight-sector trial fixture exercises default target behavior. Its
topology, coordinates, protected nodes, midsides, source immutability and exact
compiled-insertion/Python-oracle comparisons remain required. Explicit-policy
refusal and invalid-mesh contracts are tested separately.
