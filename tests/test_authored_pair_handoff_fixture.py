"""The standalone handoff hook keeps one live owner model before preparation."""

import importlib.util
from pathlib import Path
import sys

import pytest

import anygeometry


def _module():
    path = Path(__file__).resolve().parents[1] / "tools" / "authored_pair_fixture.py"
    spec = importlib.util.spec_from_file_location("authored_pair_fixture_handoff", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    '.whl' not in str(anygeometry.__file__).lower(),
    reason='needs the exact pinned ANYgeometry wheel imported (tools/authored_pair_fixture.py); this environment imports a source checkout',
)
def test_process_bound_source_factory_precedes_preparation_and_retains_sheet_owner():
    handoff = _module()
    seen = []

    def witnessed_project_extraction():
        working, roots = handoff._default_source_factory()
        sheets = {
            working.face_uses[use].face_id: sheet
            for sheet, record in working.sheets.items()
            for use in record.face_use_ids
        }
        seen.append((working, roots, working.revision, sheets))
        return working, roots

    fixture = handoff.build_fixture("size2", source_factory=witnessed_project_extraction)
    assert len(seen) == 1
    working, roots, before_revision, sheets = seen[0]
    assert fixture.geometry is working
    assert fixture.geometry.revision > before_revision
    assert fixture.component.authored_face_ids == roots
    assert {(root, sheet) for sheet, root, _use, _current in
            fixture.component.occurrence_correspondence} == set(sheets.items())
    assert fixture.staged.current_receipt.geometry_model_id == working.model_id
    assert fixture.staged.current_receipt.geometry_revision == working.revision
    assert fixture.s3_admission.admitted
    assert fixture.staged.publication_qualified is False


@pytest.mark.skipif(
    '.whl' not in str(anygeometry.__file__).lower(),
    reason='needs the exact pinned ANYgeometry wheel imported (tools/authored_pair_fixture.py); this environment imports a source checkout',
)
def test_source_factory_requires_declared_sheets_before_any_preparation():
    handoff = _module()
    working = handoff.owner.GeometryModel()
    first = working.add_plate(working.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    ))
    second = working.add_plate(working.add_points(
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
    ))
    before_revision = working.revision
    with pytest.raises(ValueError, match="separate declared Sheets"):
        handoff.build_fixture("size2", source_factory=lambda: (working, (first, second)))
    assert working.revision == before_revision
