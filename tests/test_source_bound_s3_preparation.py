"""Narrow source-bound S3 authority for a validated authored component."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
from anygeometry.editing import translate_entities
from anygeometry.entities import EntityRef
from anygeometry.serialization import from_dict, to_dict

from anymesher import (
    S3OwnerAuthorityError,
    prepare_source_bound_qualified_s3_mesh,
)
from anymesher.errors import MeshError


def _fixture():
    path = Path(__file__).resolve().parents[1] / "tools" / "authored_pair_fixture.py"
    spec = importlib.util.spec_from_file_location("authored_pair_fixture_source_s3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_fixture("size2")


def test_source_bound_s3_preparation_uses_fresh_owner_and_preserves_cells():
    fixture = _fixture()
    staged = fixture.staged
    source = deepcopy(staged.mesh)
    before_tris = deepcopy(source.tris)

    def qualify(candidate=source, receipt=staged.current_receipt,
                source_geometry=fixture.geometry, cancellation_check=None):
        return prepare_source_bound_qualified_s3_mesh(
            candidate, source_geometry, fixture.geometry, staged.mesh,
            receipt, fixture.component, fixture.source_registry,
            dict(staged.cell_current_faces),
            cancellation_check=cancellation_check,
        )

    made, record = qualify()
    assert made is not source
    assert source.structural_preparation == {}
    assert made.tris == before_tris
    assert record["status"] == "ADMITTED"
    assert record["element_ids"] == sorted(source.tris)
    assert set(record["element_owner_sources"]) == {
        str(cell) for cell in source.shells
    }
    assert record["authority_model"]["source_model_id"] == str(fixture.geometry.model_id)
    assert record["authority_model"]["source_revision"] == fixture.geometry.revision
    assert made.structural_preparation["qualified_s3"] == record

    changed = deepcopy(source)
    first = min(changed.nodes)
    changed.nodes[first] = changed.nodes[first].copy()
    changed.nodes[first][0] += 0.01
    with pytest.raises(S3OwnerAuthorityError, match="physical cells differ"):
        qualify(changed)

    with pytest.raises(MeshError):
        qualify(receipt=replace(staged.current_receipt, mesh_digest="stale"))

    changed = deepcopy(source)
    changed.geometry_revision += 1
    with pytest.raises(S3OwnerAuthorityError, match="stale source geometry"):
        qualify(changed)

    changed = deepcopy(source)
    sheet = min(changed.elements_of_sheet)
    changed.elements_of_sheet[sheet].pop()
    with pytest.raises(S3OwnerAuthorityError, match="sheet ownership"):
        qualify(changed)

    changed = deepcopy(source)
    changed.tris.clear()
    with pytest.raises(S3OwnerAuthorityError, match="requires T3 cells"):
        qualify(changed)
    assert changed.structural_preparation == {}

    separate_source = from_dict(to_dict(fixture.geometry))
    callbacks = []

    def mutate_source(_stage):
        callbacks.append(_stage)
        if len(callbacks) == 1:
            separate_source.add_point(20, 20, 20)
        return False

    with pytest.raises(S3OwnerAuthorityError, match="changed during qualification"):
        qualify(source_geometry=separate_source, cancellation_check=mutate_source)
    assert callbacks
    assert source.structural_preparation == {}

    translated_source = from_dict(to_dict(fixture.geometry))
    translate_entities(
        translated_source,
        tuple(EntityRef("sheet", sheet) for sheet in translated_source.sheets),
        (0, 0, 0.25),
    )
    translated_mesh = deepcopy(source)
    translated_mesh.geometry_revision = translated_source.revision
    with pytest.raises(S3OwnerAuthorityError, match="SOURCE face support"):
        qualify(translated_mesh, source_geometry=translated_source)
