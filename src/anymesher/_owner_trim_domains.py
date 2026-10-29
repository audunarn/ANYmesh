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


def validated_prepared_partition_disjoint_interiors(
    geometry, first, second, *, cancellation_check=None,
) -> bool:
    """Consume only the owner's live common-parent partition/injectivity proof.

    Unknown or unavailable capability leaves the ordinary overlap audit intact.
    A shared support, historical replacement or trim metadata grants no proof.
    """
    import anygeometry as owner
    from .errors import MeshError
    query = getattr(owner, 'query_prepared_partition_disjoint_interiors', None)
    if query is None:
        return False
    result = query(geometry, geometry.handle('face', first),
                   geometry.handle('face', second), cancellation_check=cancellation_check)
    if type(result) is not bool:
        raise MeshError('owner partition query returned a malformed classification')
    return result


def validated_same_support_material_disjoint_interiors(
    geometry, first, second, *, cancellation_check=None,
) -> bool:
    """Consume a fresh owner material-domain proof, independent of lineage.

    Unsupported or unavailable capability retains the actual overlap audit.
    Only an exact bool classification is accepted; owner refusal, mutation and
    caller cancellation exceptions propagate without translation or fallback.
    """
    import anygeometry as owner
    from .errors import MeshError
    query = getattr(owner, 'query_same_support_material_disjoint_interiors', None)
    if query is None:
        return False
    result = query(geometry, geometry.handle('face', first),
                   geometry.handle('face', second), cancellation_check=cancellation_check)
    if type(result) is not bool:
        raise MeshError('owner same-support material query returned a malformed classification')
    return result
