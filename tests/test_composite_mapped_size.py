"""Composite mapped sizing keeps authoritative stations and final gates."""
import numpy as np
import pytest

from anygeometry import GeometryModel, OrientedEdge, Plane, to_dict
from anymesher import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.refinement import SizeField, refine_at
from anymesher.seeding import _composite_structured_faces, solve_seeding


def panel(*, reverse=False, surface=True, short=.125):
    model = GeometryModel()
    points = [(0., 0., 0.), (short, 0., 0.), (4., 0., 0.),
              (4., 2., 0.), (short, 2., 0.), (0., 2., 0.)]
    if reverse:
        points = [points[0], *reversed(points[1:])]
    vertices = model.add_points(points)
    edges = tuple(model.add_line(vertices[i], vertices[(i+1) % 6]) for i in range(6))
    face = model.add_face_from_loop(
        tuple(OrientedEdge(e, True) for e in edges),
        corners=(0, 1, 3, 4) if reverse else (0, 2, 3, 5),
        surface=Plane(np.array((0., 0., 0.)), np.array((4., 0., 0.)),
                      np.array((0., 2., 0.))) if surface else None,
    )
    return model, face, edges


@pytest.mark.parametrize('reverse', [False, True])
def test_localized_composite_reduces_counts_and_responds_to_size(reverse):
    model, face, _ = panel(reverse=reverse)
    original = to_dict(model)
    ordinary = solve_seeding(model, target_size=1.)
    solutions = [solve_seeding(model, target_size=h,
                              localized_structured_face_ids=(face,)) for h in (1., .5)]
    assert solutions[0].total_elements_estimate(model) < ordinary.total_elements_estimate(model)
    assert solutions[0].total_elements_estimate(model) < solutions[1].total_elements_estimate(model)
    for result in solutions:
        sides = model.faces[face].sides()
        assert result.side_divisions(sides[0]) == result.side_divisions(sides[2])
        assert result.side_divisions(sides[1]) == result.side_divisions(sides[3])
        assert min(result.divisions.values()) >= 1
    assert to_dict(model) == original


def test_explicit_edge_pin_remains_authoritative():
    model, face, edges = panel()
    result = solve_seeding(model, target_size=1., overrides={edges[1]: 9},
                          localized_structured_face_ids=(face,))
    assert result[edges[1]] == 9
    sides = model.faces[face].sides()
    assert result.side_divisions(sides[0]) == result.side_divisions(sides[2])


@pytest.mark.parametrize('unsupported', ['nonuniform', 'coons', 'custom_field', 'custom_plane'])
def test_unsupported_composites_keep_ordinary_counts(unsupported):
    model, face, _ = panel(surface=unsupported != 'coons')
    field = SizeField(model, 1., (refine_at((0., 0., 0.), size=.1, radius=.2),)
                      if unsupported == 'nonuniform' else ())
    if unsupported == 'custom_field':
        class Field(SizeField):
            pass
        field = Field(model, 1.)
    if unsupported == 'custom_plane':
        class Support(Plane):
            pass
        support = model.faces[face].surface
        model.set_face_surface(face, Support(support.origin, support.u_vector, support.v_vector))
    assert not _composite_structured_faces(model, field, [model.faces[face]], frozenset((face,)))
    expected = solve_seeding(model, size_field=field)
    actual = solve_seeding(model, size_field=field, localized_structured_face_ids=(face,))
    assert actual.divisions == expected.divisions


@pytest.mark.parametrize('strategy', ['mapped', 'auto'])
@pytest.mark.parametrize('short', [.125, .4])
def test_public_composite_quadratic_mesh_uses_localized_sizes(strategy, short):
    model, face, _ = panel(short=short)
    before = to_dict(model)
    results = [generate_hybrid_mesh_result(
        model, target_size=h, strategy=strategy, order='quadratic',
        structured_options={}, native_backend='python') for h in (1., .5)]
    if short == .125:
        # The real growth limit still determines a floor at this short joint.
        # Both targets can plateau, but avoid the old face-wide refinement.
        assert len(results[0].mesh.quads) == len(results[1].mesh.quads)
        assert len(results[0].mesh.quads) < solve_seeding(
            model, target_size=1.).total_elements_estimate(model)
    else:
        assert len(results[0].mesh.quads) < len(results[1].mesh.quads)
    for result in results:
        assert result.strategy_by_face[face] == 'mapped'
        assert not result.mesh.tris
        assert result.mesh.hybrid_diagnostics['structured_quality']['accepted']
        assert all(len(body) == 8 for body in result.mesh.quads.values())
        # Every authored station remains represented, including the short chain.
        for vertex in model.vertices:
            np.testing.assert_array_equal(result.mesh.nodes[result.mesh.node_of_vertex[vertex]],
                                          model.vertex_position(vertex))
    assert to_dict(model) == before


def test_composite_growth_never_overrides_explicit_pin():
    model, face, edges = panel()
    before = to_dict(model)
    from anymesher.seeding import SeedingConflict
    with pytest.raises(SeedingConflict, match='pinned edge'):
        solve_seeding(model, target_size=1., overrides={edges[1]: 1},
                      localized_structured_face_ids=(face,), maximum_adjacent_growth=1.5)
    assert to_dict(model) == before


def test_public_final_aspect_gate_still_refuses_without_mutation():
    model, _, _ = panel()
    before = to_dict(model)
    with pytest.raises(MeshError):
        generate_hybrid_mesh_result(
            model, target_size=1., strategy='mapped', order='quadratic',
            structured_options={'quality_policy': {'maximum_aspect_ratio': 1.01}},
            native_backend='python')
    assert to_dict(model) == before


def test_public_cancellation_preserves_identity_and_geometry():
    model, _, _ = panel()
    before = to_dict(model)
    marker = RuntimeError('cancel composite seeds')
    def cancel(phase):
        if 'seed' in phase:
            raise marker
    with pytest.raises(RuntimeError) as caught:
        generate_hybrid_mesh_result(model, target_size=1., strategy='mapped',
            structured_options={}, native_backend='python', cancellation_check=cancel)
    assert caught.value is marker
    assert to_dict(model) == before
