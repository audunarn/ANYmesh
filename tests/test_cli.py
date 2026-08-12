"""The command line interface.

Exercised through ``main(argv)`` rather than a subprocess, so a failure points at
the line that raised.
"""

from __future__ import annotations

import json

import pytest

from anymesher import load_mesh, save_mesh, simple_panel_mesh
from anymesher.__main__ import main


def _run(capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def _json_run(capsys, *argv: str) -> tuple[int, object]:
    code, out = _run(capsys, *argv)
    return code, json.loads(out)


def test_backends_lists_native_first_choices_and_says_what_they_do(capsys) -> None:
    code, payload = _json_run(capsys, "--json", "backends")
    assert code == 0
    assert payload["backends"] == ["auto", "gmsh", "mapped", "native"]

    code, out = _run(capsys, "backends")
    assert code == 0
    assert "selects mapped or native" in out
    assert "trimmed arbitrary faces" in out
    assert "structured four-sided faces" in out
    assert "triangles" in out


def test_panel_reports_counts_quality_and_the_stiffener_section(capsys) -> None:
    code, payload = _json_run(
        capsys, "--json", "panel",
        "--length", "4.0", "--width", "3.0", "--thickness", "0.012",
        "--stiffeners", "1", "--spacing", "1.5",
        "--height", "0.3", "--web-thickness", "0.01",
        "--flange-width", "0.15", "--flange-thickness", "0.015",
        "--divisions-x", "6", "--divisions-y", "5", "--beam-divisions", "3",
    )

    assert code == 0
    assert payload["nodes"] == 46
    assert payload["quads"] == 30
    assert payload["beams"] == 3
    assert payload["couplings"] == 4
    assert payload["order"] == "linear"
    assert payload["quality"]["max_aspect_ratio"] > 0.0
    assert payload["edge_nodes"]["y0"] == 7
    assert payload["stiffener_section"]["area"] == pytest.approx(0.3 * 0.01 + 0.15 * 0.015)


def test_panel_can_write_the_mesh_it_made(capsys, tmp_path) -> None:
    path = tmp_path / "panel.json"
    code, payload = _json_run(
        capsys, "--json", "panel",
        "--length", "2.0", "--width", "1.0", "--thickness", "0.01",
        "--height", "0.2", "--web-thickness", "0.008",
        "--output", str(path),
    )

    assert code == 0
    assert payload["written"] == str(path)
    reloaded = load_mesh(path)
    assert len(reloaded.quads) == payload["quads"]
    assert len(reloaded.couplings) == payload["couplings"]

    # Overwriting needs asking for, so a second run does not silently replace it.
    assert main(["panel", "--length", "2.0", "--width", "1.0", "--thickness", "0.01",
                 "--height", "0.2", "--web-thickness", "0.008", "--output", str(path)]) == 2
    assert main(["panel", "--length", "2.0", "--width", "1.0", "--thickness", "0.01",
                 "--height", "0.2", "--web-thickness", "0.008", "--output", str(path),
                 "--overwrite"]) == 0


def test_quadratic_and_aligned_options_reach_the_mesher(capsys) -> None:
    _code, plain = _json_run(
        capsys, "--json", "panel", "--length", "4", "--width", "3", "--thickness", "0.012",
        "--stiffeners", "2", "--spacing", "1.0", "--height", "0.3", "--web-thickness", "0.01",
        "--divisions-y", "6",
    )
    _code, quadratic = _json_run(
        capsys, "--json", "panel", "--length", "4", "--width", "3", "--thickness", "0.012",
        "--stiffeners", "2", "--spacing", "1.0", "--height", "0.3", "--web-thickness", "0.01",
        "--divisions-y", "6", "--quadratic",
    )
    _code, aligned = _json_run(
        capsys, "--json", "panel", "--length", "4", "--width", "3", "--thickness", "0.012",
        "--stiffeners", "2", "--spacing", "1.0", "--height", "0.3", "--web-thickness", "0.01",
        "--divisions-y", "6", "--align",
    )

    assert quadratic["order"] == "quadratic"
    assert quadratic["nodes"] > plain["nodes"]
    assert quadratic["quads"] == plain["quads"]
    assert aligned["order"] == "linear"


def test_plate_and_beam_are_available_without_stiffener_arguments(capsys) -> None:
    code, plate = _json_run(
        capsys, "--json", "plate", "--length", "2", "--width", "1", "--thickness", "0.01",
        "--divisions-x", "4", "--divisions-y", "2",
    )
    assert code == 0
    assert plate["quads"] == 8
    assert plate["beams"] == 0

    code, beam = _json_run(capsys, "--json", "beam", "--length", "3", "--divisions", "6")
    assert code == 0
    assert beam["beams"] == 6
    assert beam["nodes"] == 7
    assert beam["quads"] == 0


def test_quality_reads_a_saved_mesh(capsys, tmp_path) -> None:
    path = tmp_path / "plate.json"
    # A deliberately stretched plate, so the advisory warning has something to say.
    save_mesh(path, simple_panel_mesh(20.0, 1.0, 0.01, 2, 2))

    code, payload = _json_run(capsys, "--json", "quality", str(path))

    assert code == 0
    assert payload["quads"] == 4
    assert payload["quality"]["max_aspect_ratio"] == pytest.approx(20.0)
    assert payload["quality"]["warnings"]
    # A quality warning is advisory, so it must not fail a build.
    assert code == 0


def test_text_output_is_human_readable(capsys) -> None:
    code, out = _run(capsys, "plate", "--length", "2", "--width", "1", "--thickness", "0.01")

    assert code == 0
    assert "linear mesh" in out
    assert "aspect ratio" in out
    assert "warp" in out


def test_usage_errors_exit_two_and_report_on_stderr(capsys) -> None:
    # A stiffener wider than the panel cannot couple to it.
    assert main(["panel", "--length", "4", "--width", "3", "--thickness", "0.012",
                 "--spacing", "99", "--height", "0.3", "--web-thickness", "0.01"]) == 2
    assert "beam-shell couplings" in capsys.readouterr().err

    assert main(["quality", "does-not-exist.json"]) == 2
    assert "error:" in capsys.readouterr().err


def test_a_missing_subcommand_is_a_parser_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
