"""ANYgeometry generators and edit history remain addressable after meshing."""

from __future__ import annotations

import numpy as np
import pytest

import anygeometry
from anygeometry.generators import plate

import anymesher


@pytest.mark.parametrize("backend", ("mapped", "gmsh"))
def test_owner_generator_selection_survives_edit_and_remesh(backend: str) -> None:
    if backend == "gmsh":
        pytest.importorskip("gmsh", reason="the Gmsh path needs ANYmesher[gmsh]")

    geometry = plate(2.0, 1.0)
    old_face = geometry.group("shell")[0]
    old_edge = anygeometry.EntityRef("edge", 1)

    assert type(geometry) is anygeometry.GeometryModel
    assert type(old_face) is anygeometry.EntityRef
    assert type(old_edge) is anygeometry.EntityRef
    assert anymesher.GeometryModel is anygeometry.GeometryModel
    assert anymesher.EntityRef is anygeometry.EntityRef

    initial = anymesher.generate_mesh(
        geometry, backend=backend, target_size=0.25
    )
    assert initial.elements_on(old_face)
    assert initial.nodes_on(old_edge)

    anygeometry.split_face_at(geometry, old_face.id, axis=0, fraction=0.5)
    replacement_faces = geometry.resolve_ref(old_face)
    replacement_edges = geometry.resolve_ref(old_edge)

    assert replacement_faces == geometry.replacement_history()[old_face]
    assert replacement_edges == geometry.replacement_history()[old_edge]
    assert all(type(reference) is anygeometry.EntityRef for reference in replacement_faces)
    assert all(type(reference) is anygeometry.EntityRef for reference in replacement_edges)
    assert set(geometry.group("shell")) == set(replacement_faces)

    remeshed = anymesher.generate_mesh(
        geometry, backend=backend, target_size=0.25
    )
    selected_elements = {
        element
        for reference in replacement_faces
        for element in remeshed.elements_on(reference)
    }
    selected_edge_nodes = {
        node
        for reference in replacement_edges
        for node in remeshed.nodes_on(reference)
    }

    assert selected_elements == set(remeshed.shells)
    assert selected_edge_nodes
    assert all(
        np.isclose(remeshed.nodes[node][1], 0.0)
        for node in selected_edge_nodes
    )
