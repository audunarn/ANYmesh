"""Focused controlled example checks; optional consumer, zero provider calls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('anysolid')
spec = importlib.util.spec_from_file_location(
    'controlled_example', Path(__file__).resolve().parents[1]/'examples/solid_volume_controlled.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_controlled_example_p1_p2():
    result = example.run()
    assert result['status'] == 'PASS'
    assert [row['order'] for row in result['orders']] == [1, 2]
    assert result['native_meshing_calls'] == 0


def test_fixture_owns_immutable_storage():
    source, mesh = example.controlled_fixture()
    np.testing.assert_array_equal(mesh.cell_groups[source.region_id], mesh.cell_ids)
    assert len(mesh.face_owners) == 12
    with pytest.raises(ValueError):
        mesh.coordinates.flags.writeable = True


def test_fixture_stale_provenance_is_refused():
    from anymesher import InvalidVolumeMesh, VolumeMesh, validate_volume_mesh
    source, mesh = example.controlled_fixture()
    fields = ('coordinates', 'cells', 'node_ids', 'cell_ids', 'boundary_facets',
              'boundary_ids', 'node_groups', 'cell_groups', 'boundary_groups', 'face_owners')
    stale = VolumeMesh(**{name: getattr(mesh, name) for name in fields},
                       provenance={**mesh.provenance, 'source_revision': source.revision+1})
    with pytest.raises(InvalidVolumeMesh, match='provenance'):
        validate_volume_mesh(stale, source, boundary_tolerance=1e-7)
