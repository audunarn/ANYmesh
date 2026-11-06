"""Prefer physically qualified quads while preserving valid advisory T3 misses."""
import copy
from collections import Counter

from .quality_v2 import assert_valid_mesh, evaluate_quality
from .surface_mesh import _quality_threshold_report


def retain_physical_recombination(mesh, triangulated, settings):
    from .hybrid import _neutral_shell_core

    assert_valid_mesh(_neutral_shell_core(triangulated))
    core = _neutral_shell_core(mesh)
    before = _quality_threshold_report(evaluate_quality(core), settings)
    poor = sorted(int(v) for v in before["poor_element_ids"])
    if settings.enforce_quality and any(element not in mesh.quads for element in poor):
        raise ValueError("physical recombination rejection includes a non-quad")
    rejected = [element for element in poor if element in mesh.quads]
    if not rejected:
        assert_valid_mesh(core)
        return {"rejected_quad_ids": [], "restored_triangle_count": 0,
                "physical_quality_before": before, "physical_quality_after": before}
    owners = {}
    for face, elements in mesh.elements_of_face.items():
        for element in elements:
            if element in owners:
                raise ValueError("ambiguous physical recombination ownership")
            owners[element] = face

    def edge(a, b):
        return tuple(sorted((int(a), int(b))))

    by_face = {}
    for face, elements in triangulated.elements_of_face.items():
        incidence = {}
        for element in elements:
            row = triangulated.tris[element]
            if len(row) != 3:
                raise ValueError("physical recombination source must be T3")
            for i in range(3):
                incidence.setdefault(edge(row[i], row[(i + 1) % 3]), []).append(element)
        by_face[face] = incidence
    plans = []
    used = set()
    for element in rejected:
        row = mesh.quads[element]
        if len(row) != 4 or len(set(row)) != 4 or element not in owners:
            raise ValueError("invalid physical recombination quad")
        face = owners[element]
        links = by_face.get(face, {})
        expected = {edge(row[i], row[(i + 1) % 4]) for i in range(4)}
        pairs = []
        for diagonal in (edge(row[0], row[2]), edge(row[1], row[3])):
            pair = links.get(diagonal, [])
            if len(pair) != 2:
                continue
            cells = [triangulated.tris[i] for i in pair]
            if set(cells[0]) | set(cells[1]) != set(row):
                continue
            counts = Counter(edge(cell[i], cell[(i + 1) % 3]) for cell in cells for i in range(3))
            if {key for key, count in counts.items() if count == 1} == expected:
                pairs.append(tuple(sorted(pair)))
        if len(pairs) != 1 or used.intersection(pairs[0]):
            raise ValueError("cannot uniquely recover the original recombination pair")
        pair = pairs[0]
        used.update(pair)
        plans.append((element, face, pair))

    candidate = copy.deepcopy(mesh)
    next_id = max((*candidate.tris, *candidate.quads, *triangulated.tris), default=0) + 1
    replacements = {}
    for element, face, pair in plans:
        del candidate.quads[element]
        restored = []
        for old_id in pair:
            candidate.tris[next_id] = tuple(triangulated.tris[old_id])
            restored.append(next_id)
            next_id += 1
        replacements[element] = restored
        candidate.elements_of_face[face] = sorted(
            [v for v in candidate.elements_of_face[face] if v != element] + restored
        )
    for name in ("elements_of_sheet", "elements_of_member"):
        groups = getattr(candidate, name)
        for key, values in groups.items():
            groups[key] = [replacement for value in values
                           for replacement in replacements.get(value, [value])]
    final_core = _neutral_shell_core(candidate)
    assert_valid_mesh(final_core)
    after = _quality_threshold_report(evaluate_quality(final_core), settings)
    if settings.enforce_quality and not after["accepted"]:
        raise ValueError("restoring original triangle pairs did not preserve physical quality")
    mesh.tris = candidate.tris
    mesh.quads = candidate.quads
    mesh.elements_of_face = candidate.elements_of_face
    mesh.elements_of_sheet = candidate.elements_of_sheet
    mesh.elements_of_member = candidate.elements_of_member
    return {"rejected_quad_ids": rejected, "restored_triangle_count": 2 * len(plans),
            "physical_quality_before": before, "physical_quality_after": after}
