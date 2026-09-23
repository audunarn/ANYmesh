"""Exact topology checks for complementary planar trim domains."""


def _same_closed_ring(first, second) -> bool:
    a = tuple((item.edge, item.forward) for item in first)
    b = tuple((item.edge, item.forward) for item in second)
    if not a or len(a) != len(b):
        return False
    if len({edge for edge, _ in a}) != len(a):
        return False
    if len({edge for edge, _ in b}) != len(b):
        return False
    for candidate in (b, tuple((edge, not forward) for edge, forward in reversed(b))):
        try:
            offset = candidate.index(a[0])
        except ValueError:
            continue
        if a == candidate[offset:] + candidate[:offset]:
            return True
    return False


def _connected_closed_ring(geometry, loop) -> bool:
    ring = tuple(loop)
    if not ring or len({item.edge for item in ring}) != len(ring):
        return False
    starts = tuple(geometry.oriented_start_vertex(item) for item in ring)
    ends = tuple(geometry.oriented_end_vertex(item) for item in ring)
    return all(end == starts[(index + 1) % len(ring)] for index, end in enumerate(ends))


def complementary_trim_domains(geometry, first: int, second: int) -> bool:
    """Recognize a whole topology-owned outer ring excluded by the other face.

    Use only after the owner's coplanar classifier has validated the planar
    polygons. Additionally require exact endpoint connectivity of every trim;
    polygon validity alone cannot certify that its source edges are connected.
    Shared segments, coordinate proximity, and small areas are not exemptions.
    """
    a, b = geometry.faces[first], geometry.faces[second]
    if not all(
        _connected_closed_ring(geometry, loop)
        for face in (a, b)
        for loop in (face.loop, *face.holes)
    ):
        return False
    return any(_same_closed_ring(a.loop, hole) for hole in b.holes) or any(
        _same_closed_ring(b.loop, hole) for hole in a.holes
    )
