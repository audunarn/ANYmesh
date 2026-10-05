# Private prepared planar network consumer

`_prepared_planar_network` consumes the public whole-document
`PreparedPlanarMemberSheetNetwork` contract from ANYgeometry db5071c (main's
f415d84 has the same tree). It accounts source/current Sheets, Parts, Members,
Member edge uses, attachments, joints and point contacts without coordinate
welding. Public `AuthoredMaterialStations` proofs bind exact requested edges,
correspondences, parameters and all proof fields across callbacks.

The private entry points query/validate accounting, query/validate stations and
generate/validate a provisional connected mesh. They do not change public
default meshing routes. Generation runs the existing quad-first shell stages
with an optional private canonical station override, then creates B2 or B3
bodies from the same shell edge-node chains. Quadratic bodies use start, midside,
end ordering. Beam IDs follow shell IDs in the shared element namespace. Contact
anchors use canonical owner vertices; exact native Member station mappings are
retained separately from authored chart coordinates.

Missing capabilities, stale/wrong-model evidence, malformed proofs, budget hits
and cancellation fail closed. Generation checks full persisted geometry and
identity before and after callback work, rejects changed receipts, and returns
only a complete provisional result. Validating a mesh currently regenerates its
deterministic contents; this is expensive and is not a lightweight cache lookup.
Accounting/station validation also makes fresh owner calls. Record those costs
separately from shell meshing; no speed improvement is claimed.

The optional structural constraint inventory can be unavailable for the exact
known shared-exterior ancestry refusal while whole-document accounting remains
valid. Station generation then refuses: it never invents the missing material
correspondence. Other owner errors, including cancellation, are propagated.
The historical db5071c owner's connected strip remains blocked by this correspondence gap.
Curved/nonplanar Member networks, original Junction remapping and orientation
references remain outside this contract.

The separately frozen 67ef666 installed candidate supplies authenticated
`PreparedEdgeSubcurvePreimages.alias_records`. Its requested-root public material
station queries clear the strip correspondence gap without changing the adapter.
Small linear/quadratic strip integration and altered-alias rejection checks apply
to that exact candidate. The old refusal evidence remains historical; neither
owner version's small checks establish full-scale scientific mesh acceptance.

All mesh, beam-discretization, load-transfer, solver and publication qualification
flags remain false. Small Python-only hub tests establish implementation behavior,
not large-model acceptance, native parity or solver readiness. Existing quality
repair, quadratic promotion and strict mapping paths remain in place; this slice
does not adjudicate their scientific acceptance for new large models.

Evidence lives under
`reports/authored-root/planar-network-consumer-db5071c-01/` and the additive
`planar-network-consumer-67ef666-01/` directory. Old operation and
resource controls remain unchanged. The separate prospective full-scale envelope
has no execution authority. Legacy regression coverage stopped at its bounded
deadline and remains incomplete; no native dispatch, persistent gate, release or
default-route change is included.
