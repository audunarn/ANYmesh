"""Explicitly enabled small host-supervisor checks; no ANYmesher imports."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


ENABLED = os.environ.get("ANYMESHER_RUN_BUILD_SUPERVISOR_CHECKS") == "1"


@unittest.skipUnless(ENABLED and os.name == "nt", "explicit Windows host-tool check only")
class SupervisorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "tools" / "run_native_build_supervised.py"
        spec = importlib.util.spec_from_file_location("build_supervisor_check", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.root = Path(os.environ["ANYMESHER_BUILD_SUPERVISOR_CHECK_ROOT"])
        cls.root.mkdir()

    def run_child(self, code, **limits):
        case = self.root / self._testMethodName
        case.mkdir()
        self.evidence = case / "evidence"
        return self.module.supervise(
            [sys.executable, "-B", "-c", code], case, dict(os.environ),
            case / "stage", self.evidence,
            seconds=limits.pop("seconds", 3.), reserve=limits.pop("reserve", 1.),
            poll_seconds=.025, output_period=.05, **limits,
        )

    def test_native_flag_preflight_overrides_inherited_disable(self):
        inherited = {"ANYMESHER_DISABLE_NATIVE": "1", "ANYMESHER_REQUIRE_NATIVE": "0",
                     "DISABLE_NATIVE": "1", "CL": "/MP"}
        before = dict(inherited)
        inventory = {
            "environment_remove": ["ANYMESHER_DISABLE_NATIVE", "DISABLE_NATIVE", "CL"],
            "environment_set": {"ANYMESHER_DISABLE_NATIVE": "0", "ANYMESHER_REQUIRE_NATIVE": "1"},
        }
        environment = self.module.build_environment(inventory, inherited)
        self.assertEqual(environment["ANYMESHER_DISABLE_NATIVE"], "0")
        self.assertEqual(environment["ANYMESHER_REQUIRE_NATIVE"], "1")
        self.assertNotIn("DISABLE_NATIVE", environment)
        self.assertNotIn("CL", environment)
        self.assertEqual(inherited, before)
        legacy = {"environment_remove": [],
                  "environment_set": {"DISABLE_NATIVE": "0", "ANYMESHER_REQUIRE_NATIVE": "1"}}
        with self.assertRaisesRegex(RuntimeError, "fail-hard native build flags"):
            self.module.build_environment(legacy, inherited)

    def test_output_sampling_ignores_only_disappearing_enumerated_entries(self):
        case = self.root / self._testMethodName
        case.mkdir()
        (case / "keep.bin").write_bytes(b"kept")
        (case / "vanished.tmp").write_bytes(b"temporary")
        original = Path.lstat

        def disappeared(path, *args, **kwargs):
            if path.name == "vanished.tmp":
                raise FileNotFoundError(2, "simulated compiler unlink", str(path))
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", disappeared):
            self.assertEqual(self.module.output_bytes(case), 4)
        with patch.object(Path, "lstat", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.module.output_bytes(case)
        reparse = SimpleNamespace(st_mode=0, st_file_attributes=0x400, st_size=1)
        with patch.object(Path, "lstat", return_value=reparse):
            with self.assertRaisesRegex(RuntimeError, "reparse"):
                self.module.output_bytes(case)

        def denied_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("directory denied"))
            return iter(())

        with patch.object(self.module.os, "walk", denied_walk):
            with self.assertRaises(PermissionError):
                self.module.output_bytes(case)
        self.assertEqual((case / "vanished.tmp").read_bytes(), b"temporary")

    def test_root_handle_kill_failure_still_audits_and_writes_uncertainty(self):
        child = Mock()
        child.pid = 424242
        child.poll.return_value = None
        child.kill.side_effect = OSError("simulated root-handle kill failure")
        child.wait.side_effect = self.module.subprocess.TimeoutExpired("fake", 1.)
        with patch.object(self.module.subprocess, "Popen", return_value=child), \
             patch.object(self.module.psutil, "Process",
                          side_effect=self.module.psutil.AccessDenied(child.pid)), \
             patch.object(self.module, "audit", wraps=self.module.audit) as audited:
            result = self.run_child("not executed: mocked Popen")
        child.kill.assert_called_once()
        child.wait.assert_called_once()
        self.assertGreaterEqual(audited.call_count, 2)
        self.assertEqual(result["status"], "failure")
        self.assertFalse(result["safe_to_release"])
        self.assertFalse(result["terminal_proven"])
        self.assertTrue(any("root-handle termination failed" in item
                            for item in result["uncertain"]))
        saved = json.loads((self.evidence / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        self.assertTrue((self.evidence / "stdout.bin").exists())
        self.assertTrue((self.evidence / "stderr.bin").exists())

    def test_normal_exit_retains_raw_streams_and_terminal_audit(self):
        result = self.run_child("import sys,time; print('raw-out'); print('raw-err',file=sys.stderr); time.sleep(.3)")
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["safe_to_release"])
        self.assertTrue(result["processes"])
        self.assertTrue(all(row["state"] in ("ended", "reused") for row in result["processes"]))
        self.assertIn(b"raw-out", (self.evidence / "stdout.bin").read_bytes())
        self.assertIn(b"raw-err", (self.evidence / "stderr.bin").read_bytes())

    def test_nonzero_exit_is_not_success(self):
        result = self.run_child("import sys,time; time.sleep(.3); sys.exit(7)")
        self.assertEqual(result["root_exit"], 7)
        self.assertEqual(result["status"], "failure")
        self.assertTrue(result["safe_to_release"])

    def test_timeout_audits_and_terminates_owned_child_tree(self):
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],creationflags=subprocess.CREATE_NO_WINDOW); time.sleep(20)"
        result = self.run_child(code, seconds=3., reserve=2.)
        self.assertEqual(result["stop_reason"], "wall_timeout")
        self.assertGreaterEqual(len(result["processes"]), 2)
        self.assertTrue(result["safe_to_release"], result)
        self.assertEqual(result["status"], "failure")

    def test_root_exit_does_not_hide_a_live_descendant(self):
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],creationflags=subprocess.CREATE_NO_WINDOW); time.sleep(.4)"
        result = self.run_child(code)
        self.assertEqual(result["root_exit"], 0)
        self.assertEqual(result["stop_reason"], "descendants_after_root_exit")
        self.assertEqual(result["status"], "failure")
        self.assertGreaterEqual(len(result["processes"]), 2)
        self.assertTrue(result["safe_to_release"], result)

    def test_observed_memory_budget_is_not_claimed_as_a_hard_cap(self):
        result = self.run_child("import time; time.sleep(20)", ram_bytes=1)
        self.assertEqual(result["stop_reason"], "ram_budget")
        self.assertGreater(result["ram_budget_overshoot"], 0)
        self.assertEqual(result["budget_mode"], "sampled_not_kernel_caps")
        self.assertTrue(result["safe_to_release"], result)

    def test_output_budget_preserves_overshoot_and_files(self):
        result = self.run_child("import pathlib,time; pathlib.Path('stage/payload.bin').write_bytes(b'x'*1048576); time.sleep(20)", disk_bytes=32768)
        self.assertEqual(result["stop_reason"], "output_budget")
        self.assertGreater(result["output_budget_overshoot"], 0)
        self.assertTrue((self.evidence.parent / "stage" / "payload.bin").is_file())
        self.assertFalse(result["files_deleted"])
        self.assertTrue(result["safe_to_release"], result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
