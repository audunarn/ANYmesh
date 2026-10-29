"""Default-output tests must not depend on optional quad worker executables.

A locally built Q4/Q5 worker (ignored, ``third_party/quad/worker/out``) changes
meshes when found; the golden/bit-identity tests are pinned worker-less, so the
test session pins both workers off unless a test is marked ``quad_workers``.
"""
import os

from test_qf_nurbs_s0_characterisation import GOLDEN, build, digest


def test_unmarked_tests_run_with_both_quad_workers_pinned_off():
    assert os.environ.get("ANYMESH_Q5_DISABLE_WORKER") == "1"
    assert os.environ.get("ANYMESH_Q4_DISABLE_WORKER") == "1"


def test_s0_cone_digest_ignores_a_present_q5_worker(monkeypatch, tmp_path):
    dummy = tmp_path / "quad_tinyad_optimizer.exe"
    dummy.write_bytes(b"not an executable")
    monkeypatch.setenv("ANYMESH_QUAD_TINYAD_WORKER", str(dummy))
    assert digest(build("cone")) == GOLDEN["cone"]
