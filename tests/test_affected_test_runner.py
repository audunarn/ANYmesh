"""Selection controls for advisory, change-aware development tests."""

from __future__ import annotations

import pytest

from tools.run_affected_tests import _exports, _imports, _source_index, select_tests


def test_recovery_edit_selects_public_and_direct_consumers():
    selectors, reasons = select_tests(["src/anymesher/recovery.py"])
    assert "tests/test_recovery.py" in reasons
    assert "tests/test_geometry_schema6_consumer.py" in reasons
    assert "tests/test_hybrid.py" in selectors  # Always-on public smoke.
    assert not any(path.startswith("tests/sg1/") for path in selectors)


def test_quad_seed_edit_reaches_planar_and_curved_consumers():
    _, reasons = select_tests(["src/anymesher/quad/seed.py"])
    assert "tests/quad_first_planar/test_pq2_size_seed.py" in reasons
    assert "tests/quad_first_curved/test_ch11_parametric_curved_quadratic.py" in reasons


def test_direct_test_edit_and_explicit_extra_are_included_once():
    selectors, reasons = select_tests(
        ["tests/test_recovery.py"], ["tests/test_geometry_schema6_consumer.py"]
    )
    assert selectors.count("tests/test_recovery.py") == 1
    assert selectors.count("tests/test_geometry_schema6_consumer.py") == 1
    assert reasons["tests/test_recovery.py"] == ["edited test"]


def test_unknown_source_is_rejected_and_qualification_is_not_scheduled(tmp_path, monkeypatch):
    from tools import run_affected_tests as runner
    source = tmp_path / 'src' / 'anymesher'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('', encoding='utf-8')
    gate = tmp_path / 'tests' / 'sg1' / 'test_gate.py'
    gate.parent.mkdir(parents=True)
    gate.write_text('def test_gate(): pass\n', encoding='utf-8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SOURCE', source)
    monkeypatch.setattr(runner, 'TESTS', tmp_path / 'tests')
    with pytest.raises(ValueError, match="No safe affected-test mapping"):
        select_tests(["src/anymesher/not_a_module.py"])
    with pytest.raises(ValueError, match="scientific gate"):
        select_tests([], ["tests/sg1/test_gate.py"])
    with pytest.raises(ValueError, match="scientific gate"):
        select_tests(["tests/sg1/test_gate.py"])


def test_no_changed_paths_runs_smoke_only():
    selectors, reasons = select_tests([])
    assert reasons == {}
    assert "tests/test_hybrid.py" in selectors


def test_package_attribute_import_resolves_to_export_owner(tmp_path):
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import anymesher as mesh\nmesh.generate_automatic_mesh_result\n",
        encoding="utf-8",
    )
    index = _source_index()
    assert "anymesher.recovery" in _imports(consumer, "", index, _exports(index))


def test_import_parser_accepts_utf8_bom(tmp_path):
    consumer = tmp_path / 'consumer.py'
    consumer.write_text('\ufeffimport anymesher.hybrid\n', encoding='utf-8')
    index = {'anymesher.hybrid': tmp_path / 'hybrid.py'}
    assert _imports(consumer, '', index, {}) == {'anymesher.hybrid'}


def test_export_parser_accepts_utf8_bom(tmp_path):
    package = tmp_path / '__init__.py'
    package.write_text('\ufefffrom .producer import entry\n', encoding='utf-8')
    index = {'anymesher': package, 'anymesher.producer': tmp_path / 'producer.py'}
    assert _exports(index)['anymesher']['entry'] == 'anymesher.producer'
