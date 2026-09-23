"""One approved native build, with best-effort identity-aware tree cleanup.

RSS/output budgets are sampled, not kernel caps. Cleanup can outlast the
nominal envelope; coordinator death is not crash-safe containment. This tool
never releases the resource lock, retries, installs, probes or replaces a DLL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import psutil


REPOSITORY = Path(__file__).resolve().parents[1]
MANAGER = REPOSITORY.parent / ".resource-manager"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity_state(identity):
    """Never treat a reused PID as the retained process."""
    pid, created = identity
    try:
        process = psutil.Process(pid)
        if process.create_time() != created:
            return "reused", None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return "ended", None
        return "live", process
    except psutil.NoSuchProcess:
        return "ended", None
    except psutil.AccessDenied:
        return "unknown", None


def discover(identities):
    uncertain = []
    for identity in tuple(identities):
        state, process = identity_state(identity)
        if state == "unknown":
            uncertain.append(f"cannot inspect retained process {identity}")
        if process is None:
            continue
        try:
            children = process.children(recursive=True)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            uncertain.append(f"cannot enumerate children of {identity}")
            continue
        for child in children:
            try:
                identities.add((child.pid, child.create_time()))
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                uncertain.append(f"cannot identify descendant {child.pid}")
    return uncertain


def audit(identities):
    rows = []
    for identity in sorted(identities):
        state, _ = identity_state(identity)
        rows.append({"pid": identity[0], "created": identity[1], "state": state})
    return rows


def output_bytes(*roots):
    total = 0
    enumerated_directories = set()

    def walk_error(error):
        path = os.path.normcase(os.path.abspath(error.filename)) if error.filename else None
        if isinstance(error, FileNotFoundError) and path in enumerated_directories:
            return
        raise error

    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
            enumerated_directories.update(
                os.path.normcase(os.path.abspath(Path(directory) / name)) for name in dirs
            )
            file_names = set(files)
            for name in (*dirs, *files):
                path = Path(directory) / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue  # Only an entry already enumerated by this sample.
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise RuntimeError(f"reparse output is not permitted: {path}")
                if name in file_names:
                    total += info.st_size
    return total


def supervise(argv, cwd, env, stage, evidence, *, seconds=300., reserve=10.,
              ram_bytes=2 * 1024**3, disk_bytes=500 * 1024**2,
              poll_seconds=.1, output_period=.5, lease_check=lambda: None):
    """Compiler-only invocation; small fake children exercise this function."""
    if not 0 < reserve < seconds or min(poll_seconds, output_period) <= 0:
        raise ValueError("invalid supervision timing")
    stage, evidence = Path(stage), Path(evidence)
    if stage.exists() or evidence.exists():
        raise FileExistsError("staging/evidence must both be absent")
    evidence.mkdir()
    stage.mkdir()
    started = time.monotonic()
    cleanup_deadline = started + seconds
    work_deadline = cleanup_deadline - reserve
    identities, uncertain, errors, actions = set(), [], [], []
    child = None
    root_identity = None
    reason = "launch_error"
    peak_rss = peak_output = 0
    exit_code = None
    last_output_check = -float("inf")
    write_json(evidence / "intent.json", {
        "argv": list(argv), "cwd": str(cwd), "seconds": seconds,
        "compiler_seconds": seconds - reserve, "ram_budget": ram_bytes,
        "output_budget": disk_bytes, "budget_mode": "sampled_not_kernel_caps",
        "no_window": bool(NO_WINDOW), "supervisor_pid": os.getpid(),
    })
    with (evidence / "stdout.bin").open("xb") as out, (evidence / "stderr.bin").open("xb") as err:
        try:
            lease_check()
            child = subprocess.Popen(
                list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, shell=False, creationflags=NO_WINDOW,
            )
            root_identity = (child.pid, psutil.Process(child.pid).create_time())
            identities.add(root_identity)
            write_json(evidence / "process.json", {
                "pid": child.pid, "created": root_identity[1],
                "argv": list(argv), "no_window": bool(NO_WINDOW),
            })
            while True:
                lease_check()
                uncertain.extend(discover(identities))
                rss = psutil.Process(os.getpid()).memory_info().rss
                for identity in tuple(identities):
                    state, process = identity_state(identity)
                    if state == "unknown":
                        uncertain.append(f"memory identity unavailable: {identity}")
                    if process is not None:
                        try:
                            rss += process.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                peak_rss = max(peak_rss, rss)
                now = time.monotonic()
                exit_code = child.poll()
                if now - last_output_check >= output_period or exit_code is not None:
                    peak_output = max(peak_output, output_bytes(stage, evidence))
                    last_output_check = now
                rows = audit(identities)
                if uncertain or any(row["state"] == "unknown" for row in rows):
                    reason = "process_audit_unproven"
                    break
                if rss > ram_bytes:
                    reason = "ram_budget"
                    break
                if peak_output > disk_bytes:
                    reason = "output_budget"
                    break
                if exit_code is not None:
                    reason = ("descendants_after_root_exit" if any(
                        row["state"] == "live" for row in rows
                    ) else "completed")
                    break
                if now >= work_deadline:
                    reason = "wall_timeout"
                    break
                time.sleep(min(poll_seconds, max(0., work_deadline - now)))
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
            reason = "supervisor_exception" if child is not None else "launch_error"
        finally:
            uncertain.extend(discover(identities))
            rows = audit(identities)
            if any(row["state"] == "live" for row in rows) or (child is not None and child.poll() is None):
                if os.name == "nt" and root_identity is not None and identity_state(root_identity)[0] == "live":
                    command = [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                                   "System32" / "taskkill.exe"), "/PID", str(root_identity[0]), "/T", "/F"]
                    try:
                        with (evidence / "termination_stdout.bin").open("xb") as tout, (evidence / "termination_stderr.bin").open("xb") as terr:
                            killed = subprocess.run(
                                command, stdin=subprocess.DEVNULL, stdout=tout, stderr=terr,
                                creationflags=NO_WINDOW, shell=False,
                                timeout=max(.1, cleanup_deadline - time.monotonic()),
                            )
                        actions.append({"action": "taskkill_tree", "root": root_identity,
                                        "exit_code": killed.returncode})
                    except BaseException as error:
                        errors.append(f"tree termination: {type(error).__name__}: {error}")
                # One identity-checked termination per retained survivor; never
                # target a reused PID, and never enumerate unrelated GUI trees.
                uncertain.extend(discover(identities))
                for identity in sorted(identities, reverse=True):
                    state, process = identity_state(identity)
                    if process is not None:
                        try:
                            process.kill()
                            actions.append({"action": "kill_retained", "identity": identity})
                        except psutil.NoSuchProcess:
                            pass
                        except psutil.Error as error:
                            uncertain.append(f"termination {identity}: {error}")
                if child is not None and root_identity is None and child.poll() is None:
                    try:
                        child.kill()  # Popen retains the root handle, not an arbitrary PID.
                    except BaseException as error:
                        uncertain.append(f"root-handle termination failed: {type(error).__name__}: {error}")
                    uncertain.append("root creation identity unavailable; descendant state unproven")
                while time.monotonic() < cleanup_deadline:
                    rows = audit(identities)
                    if not any(row["state"] == "live" for row in rows):
                        break
                    time.sleep(min(poll_seconds, .05))
            if child is not None:
                try:
                    exit_code = child.wait(timeout=max(.1, cleanup_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    uncertain.append("root wait did not prove termination")
            # Normal root exit alone is insufficient: inspect every retained
            # PID/create-time identity on every outcome, after cleanup too.
            uncertain.extend(discover(identities))
            rows = audit(identities)
            out.flush()
            err.flush()
            os.fsync(out.fileno())
            os.fsync(err.fileno())
    elapsed = time.monotonic() - started
    proven = not uncertain and all(row["state"] in ("ended", "reused") for row in rows)
    if child is not None and root_identity is None:
        proven = False
    result = {
        "schema": "anymesher.native_build_supervision/1",
        "status": "success" if reason == "completed" and exit_code == 0 and proven and not errors else "failure",
        "stop_reason": reason, "root_identity": root_identity, "root_exit": exit_code,
        "processes": rows, "actions": actions, "errors": errors,
        "uncertain": sorted(set(uncertain)), "terminal_proven": proven,
        "safe_to_release": proven, "elapsed_seconds": elapsed,
        "nominal_seconds": seconds, "nominal_envelope_exceeded": elapsed > seconds,
        "peak_observed_tree_and_supervisor_rss": peak_rss,
        "peak_observed_output_bytes": peak_output,
        "ram_budget_overshoot": max(0, peak_rss - ram_bytes),
        "output_budget_overshoot": max(0, peak_output - disk_bytes),
        "budget_mode": "sampled_not_kernel_caps", "crash_safe": False,
        "termination_mode": "best_effort_identity_aware_not_kernel_containment",
        "retry_performed": False, "files_deleted": False,
    }
    write_json(evidence / "result.json", result)
    return result


def verified_json(path, expected):
    if digest(path) != expected.upper():
        raise RuntimeError(f"identity mismatch: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_environment(inventory, inherited):
    environment = dict(inherited)
    for name in inventory["environment_remove"]:
        environment.pop(name, None)
    environment.update(inventory["environment_set"])
    if environment.get("ANYMESHER_REQUIRE_NATIVE") != "1" or environment.get("ANYMESHER_DISABLE_NATIVE") != "0":
        raise RuntimeError("fail-hard native build flags are required")
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("this registered build is Windows-only")
    launch = verified_json(args.manifest, args.manifest_sha256)
    if launch["schema"] != "anymesher.cp314_startup_launch/1":
        raise RuntimeError("unknown launch manifest")
    if digest(__file__) != launch["supervisor_sha256"]:
        raise RuntimeError("supervisor source drift")
    runtime = launch["supervisor_runtime"]
    if Path(sys.executable).resolve() != Path(runtime["python"]).resolve() or digest(sys.executable) != runtime["python_sha256"]:
        raise RuntimeError("supervisor interpreter mismatch")
    if psutil.__version__ != runtime["psutil_version"] or Path(psutil.__file__).resolve() != Path(runtime["psutil_path"]).resolve():
        raise RuntimeError("supervisor psutil origin/version mismatch")
    inventory = verified_json(launch["inventory_path"], launch["inventory_sha256"])
    if inventory["schema"] != "anymesher.cp314_startup_repair_request/1" or Path(inventory["repository"]).resolve() != REPOSITORY:
        raise RuntimeError("wrong build inventory")
    for item in inventory["inputs"]:
        if Path(item["path"]).stat().st_size != item["bytes"] or digest(item["path"]) != item["sha256"]:
            raise RuntimeError(f"build input drift: {item['path']}")
    expected_command = subprocess.list2cmdline([
        sys.executable, "-B", str(Path(__file__).resolve()), "--manifest",
        str(Path(args.manifest).resolve()), "--manifest-sha256", args.manifest_sha256,
    ])
    owner = json.loads((MANAGER / "active-lock" / "owner.json").read_text(encoding="utf-8-sig"))
    request_id = owner["request_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", request_id):
        raise RuntimeError("invalid resource request ID")
    request = json.loads((MANAGER / "requests" / f"{request_id}.json").read_text(encoding="utf-8-sig"))
    if request["command"] != expected_command or owner["command"] != expected_command:
        raise RuntimeError("resource request command mismatch")
    if Path(request["repository"]).resolve() != REPOSITORY or request["request_id"] != request_id:
        raise RuntimeError("resource request repository/identity mismatch")
    ledger = (MANAGER / "ledger.md").read_text(encoding="utf-8-sig")
    if not re.search(r"\|\s*" + request_id + r"\s*\|\s*APPROVED\s*\|", ledger):
        raise RuntimeError("resource request is not approved")

    def lease_check():
        current = json.loads((MANAGER / "active-lock" / "owner.json").read_text(encoding="utf-8-sig"))
        if current["request_id"] != request_id or current["command"] != expected_command:
            raise RuntimeError("resource ownership changed")

    environment = build_environment(inventory, os.environ)
    limits = inventory["limits"]
    if limits != {"deadline_seconds": 300, "process_tree_memory_bytes": 2147483648,
                  "output_bytes": 524288000, "serial_builds": 1}:
        raise RuntimeError("unregistered build limits")
    result = supervise(
        inventory["build_argv"], REPOSITORY, environment,
        inventory["stage"], inventory["evidence"], lease_check=lease_check,
    )
    print(json.dumps({"status": result["status"], "safe_to_release": result["safe_to_release"],
                      "evidence": inventory["evidence"]}, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
