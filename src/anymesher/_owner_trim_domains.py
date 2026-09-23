"""Consume revision-bound trim evidence; never infer a geometry exemption."""


def validated_complementary_trim_domains(
    geometry, first, second, *, cancellation_check=None,
) -> bool:
    """Require the owner's live, ordered-parent complementary-domain proof.

    Imports are deliberately lazy: older dependencies cannot silently substitute
    topology matching for this capability. Import, owner, and cancellation errors
    propagate before preparation can publish or mutate a working closure.
    """
    from anygeometry import (
        TrimBoundaryContact,
        TrimInteriorRelation,
        query_trim_domain_relation,
        validate_trim_domain_binding,
    )

    revision = geometry.revision
    parents = (geometry.handle("face", first), geometry.handle("face", second))
    result = query_trim_domain_relation(
        geometry, *parents, expected_revision=revision,
        cancellation_check=cancellation_check,
    )
    validate_trim_domain_binding(
        geometry, result, *parents, expected_revision=revision,
        cancellation_check=cancellation_check,
    )
    return (
        result.relation is TrimInteriorRelation.DISJOINT_INTERIORS
        and result.complementary is True
        and result.boundary is TrimBoundaryContact.CURVE
    )
