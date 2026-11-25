"""Differential harness: native finite-hull seed vs the Python routine (tripod speed option 2).

    python tools/fhs_native_harness.py [--predicates N] [--triangulations N] [--capture fhs_cases.pkl]
                                       [--rounds R] [--seed S] [--out result.json]

1. Predicates: N rows (near-collinear, cocircular, duplicate, lattice, extreme, random); the native
   orient/incircle signs must equal the signs of ``triangulation.orient2d``/``incircle``.
2. Triangulations: N random inputs from the same families; outcome (triangles or exception type and
   message) and the full callback trace must be equal, with a recording callback, with none, and
   (every 10th input) with a callback that changes the input at a random callback.
3. Capture (optional): every captured ``(points, result, seconds)`` call must give the captured result
   natively; with ``--rounds`` > 0 the native and Python routes are timed alternately.

Prints a summary and writes it as JSON; exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np

from anymesher import triangulation
from anymesher.native import _finite_hull_seed_kernel


def _sign(value):
    return (value > 0) - (value < 0)


def _python(function, *args):
    saved = triangulation._NATIVE_FINITE_HULL
    triangulation._NATIVE_FINITE_HULL = False
    try:
        return function(*args)
    finally:
        triangulation._NATIVE_FINITE_HULL = saved


def predicate_rows(rng, count):
    per = max(1, count // 10)
    rows = []
    a = rng.uniform(-10, 10, (per, 2))
    b = rng.uniform(-10, 10, (per, 2))
    t = rng.choice([0.25, 0.5, 0.75, 2.0, -1.0, 1.0 / 3.0, 0.1], per)
    c = a + t[:, None] * (b - a)
    c = np.nextafter(c, c + rng.choice([-1.0, 0.0, 1.0], (per, 2)))
    rows.append(np.c_[a, b, c, rng.uniform(-10, 10, (per, 2))])
    # Collinear boundary stations (the tripod's main exact-fallback source): x along an edge, y equal.
    x = rng.uniform(0, 20, (per, 3))
    y = rng.choice([0.0, 0.1, 1.0 / 3.0, 7.25], (per, 1))
    rows.append(np.c_[x[:, 0], y[:, 0], x[:, 1], y[:, 0], x[:, 2], y[:, 0] + rng.choice([0.0, 0.0, 1e-17], per),
                      rng.uniform(0, 20, (per, 2))])
    rows.append(rng.integers(-20, 20, (per, 8)) * rng.choice([1.0, 0.5, 0.125, 1.0 / 3.0, 0.1], (per, 1)))
    x0, y0 = rng.uniform(-5, 5, (2, per))
    w, h = rng.integers(1, 9, (2, per)) * 0.25
    rect = np.c_[x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
    rows.append(rect + rng.choice([0.0, 0.0, 1e-17, -1e-17, 1e-15], (per, 8)))
    p, q = rng.integers(1, 30, (2, per)).astype(float)
    r = p * p + q * q
    rows.append(np.c_[p * p - q * q, 2 * p * q, -(p * p - q * q), -2 * p * q, 2 * p * q, p * p - q * q, r, 0 * r]
                * rng.choice([1.0, 2.0 ** -7, 0.1], (per, 1)))
    base = rng.uniform(-3, 3, (per, 8))
    k = rng.integers(0, 4, per)
    for row, j in zip(base, k):
        row[2 * ((j + 1) % 4):2 * ((j + 1) % 4) + 2] = row[2 * j:2 * j + 2]
    rows.append(base)
    scale = rng.choice([1e-300, 1e-160, 1e-16, 1.0, 1e150, 1e300, 5e-324, 2.0 ** 401, 2.0 ** 199], (per, 8))
    rows.append(rng.uniform(-1, 1, (per, 8)) * scale)
    mixed = rng.uniform(-1, 1, (per, 8))
    mixed[:, ::3] *= 1e-16
    rows.append(mixed)
    rows.append(rng.choice([0.0, -0.0, 5e-324, -5e-324, 1.0, 2.0 ** -1022, 1.7976931348623157e308], (per, 8)))
    rows.append(rng.uniform(-1, 1, (count - 9 * per, 8)))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def check_predicates(rng, count):
    from anymesher import _native

    mismatches = []
    done = 0
    t_native = t_python = 0.0
    while done < count:
        chunk = min(100_000, count - done)
        rows = predicate_rows(rng, chunk)
        t = time.perf_counter()
        signs = np.frombuffer(_native.fhs_predicate_signs(rows.tobytes()), dtype=np.int8).reshape(-1, 2)
        t_native += time.perf_counter() - t
        t = time.perf_counter()
        orient, incircle = triangulation.orient2d, triangulation.incircle
        for row, (o, c) in zip(rows.tolist(), signs.tolist()):
            f, s, t3, p = row[0:2], row[2:4], row[4:6], row[6:8]
            if o != _sign(orient(f, s, t3)) or c != _sign(incircle(f, s, t3, p)):
                mismatches.append(row)
        t_python += time.perf_counter() - t
        done += len(rows)
    return {"cases": done, "mismatches": len(mismatches), "first": mismatches[:5],
            "native_s": round(t_native, 2), "python_s": round(t_python, 2)}


def random_points(rng):
    n = int(rng.integers(3, 160))
    kind = int(rng.integers(0, 9))
    if kind == 0:
        pts = rng.integers(0, int(rng.integers(2, 12)), (n, 2)) * rng.choice([0.5, 0.1, 1.0 / 3.0])
    elif kind == 1:
        pts = rng.uniform(0, 1, (n, 2))
    elif kind == 2:
        pts = np.r_[rng.integers(0, 6, (n, 2)) * 0.25, rng.uniform(0, 1.5, (n // 3 + 1, 2))]
    elif kind == 3:
        t = rng.uniform(0, 1, n)
        pts = np.c_[t, 3 * t + 1] + rng.choice([0.0, 1e-16, -1e-16, 0.3], (n, 2))
    elif kind == 4:  # quad-seed-like: lattice + boundary stations + interior jitter, maybe rotated/offset
        m = int(rng.integers(3, 14))
        xs, ys = np.meshgrid(np.linspace(0, rng.uniform(0.5, 5), m), np.linspace(0, rng.uniform(0.5, 5), m))
        lattice = np.c_[xs.ravel(), ys.ravel()]
        stations = np.c_[rng.uniform(0, lattice[:, 0].max(), m), np.zeros(m)]
        pts = np.r_[lattice, stations, rng.uniform(0, 1, (m, 2)) * lattice.max(axis=0)]
        if rng.random() < 0.5:
            angle = rng.uniform(0, np.pi)
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            pts = pts @ rot.T + rng.uniform(-1e3, 1e3, 2)
    elif kind == 5:  # cocircular rings
        k = int(rng.integers(4, 40))
        angle = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pts = np.r_[np.c_[np.cos(angle), np.sin(angle)], rng.uniform(-0.5, 0.5, (int(rng.integers(0, 5)), 2))]
    elif kind == 6:  # duplicates
        base = rng.uniform(0, 1, (max(3, n // 2), 2))
        pts = np.r_[base, base[rng.integers(0, len(base), n - len(base) + 1)]]
    elif kind == 7:  # extreme magnitudes
        pts = rng.uniform(0, 1, (n, 2)) * rng.choice([1e-300, 1e-150, 1e150, 1e300, 1.0])
        if rng.random() < 0.3:
            pts[rng.integers(0, n)] = (5e-324, 5e-324)
    else:  # near-collinear clusters around a long edge
        t = np.sort(rng.uniform(0, 1, n))
        pts = np.c_[t * 10, np.nextafter(np.zeros(n), rng.choice([-1.0, 1.0], n)) * rng.integers(0, 3, n)]
        pts = np.r_[pts, [[5.0, 1.0], [5.0, -1.0]]]
    return np.ascontiguousarray(pts, dtype=np.float64)


def outcome(points, callback_kind, rng_state=None):
    stages = []
    trigger = rng_state

    def callback(stage):
        stages.append(stage)
        if trigger is not None and len(stages) == trigger[0]:
            kind = trigger[1]
            if kind == 0:
                points[len(points) // 2] = points[len(points) // 2] * 0.5
            elif kind == 1:
                points[-1] = (np.nan, 0.0)
            elif kind == 2:
                points.resize((3, 2), refcheck=False)
            else:
                raise KeyboardInterrupt("cancelled")

    # The engine's triangles are compared even when the final completeness check refuses them.
    engine = []
    real = triangulation._complete_hull_seed

    def recorded(points, triangles, cancellation_check=None):
        engine.append(list(triangles))
        return real(points, triangles, cancellation_check)

    triangulation._complete_hull_seed = recorded
    try:
        result = ("ok", triangulation._finite_hull_seed(points, callback if callback_kind else None))
    except BaseException as error:
        result = ("error", type(error).__name__, str(error))
    finally:
        triangulation._complete_hull_seed = real
    return result, stages, engine


def check_triangulations(rng, count):
    mismatches = []
    kinds = {}
    t_native = t_python = 0.0
    for index in range(count):
        points = random_points(rng)
        for callback_kind in (True, False):
            t = time.perf_counter()
            native = outcome(points.copy(), callback_kind)
            t_native += time.perf_counter() - t
            t = time.perf_counter()
            python = _python(outcome, points.copy(), callback_kind)
            t_python += time.perf_counter() - t
            if native != python:
                mismatches.append({"index": index, "callback": callback_kind, "points": points.tolist()})
            if callback_kind:
                kinds[native[0][0]] = kinds.get(native[0][0], 0) + 1
                total = len(native[1])
                kinds["engine_compared"] = kinds.get("engine_compared", 0) + bool(native[2])
        if index % 10 == 0 and total > 0:
            trigger = (int(rng.integers(1, total + 1)), int(rng.integers(0, 4)))
            native = outcome(points.copy(), True, trigger)
            python = _python(outcome, points.copy(), True, trigger)
            if native != python:
                mismatches.append({"index": index, "trigger": trigger, "points": points.tolist()})
    return {"triangulations": count, "outcomes": kinds, "mismatches": len(mismatches), "first": mismatches[:3],
            "native_s": round(t_native, 2), "python_s": round(t_python, 2)}


def check_capture(path, rounds):
    cases = pickle.load(open(path, "rb"))
    mismatches = 0
    for points, expected, _ in cases:
        if list(triangulation._finite_hull_seed(points, None)) != expected:
            mismatches += 1
    summary = {"cases": len(cases), "points": int(sum(len(c[0]) for c in cases)),
               "captured_s": round(sum(c[2] for c in cases), 1), "mismatches": mismatches, "rounds": []}
    for r in range(rounds):
        tn = tp = 0.0
        for i, (points, _, _) in enumerate(cases):
            for which in (("n", "p") if (i + r) % 2 == 0 else ("p", "n")):
                t = time.perf_counter()
                if which == "n":
                    triangulation._finite_hull_seed(points, None)
                    tn += time.perf_counter() - t
                else:
                    _python(triangulation._finite_hull_seed, points, None)
                    tp += time.perf_counter() - t
        summary["rounds"].append({"native_s": round(tn, 2), "python_s": round(tp, 2)})
        print("capture round %d: native %.2f s, python %.2f s" % (r, tn, tp), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicates", type=int, default=1_000_000)
    parser.add_argument("--triangulations", type=int, default=10_000)
    parser.add_argument("--capture")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--out")
    args = parser.parse_args()
    if _finite_hull_seed_kernel() is None:
        raise SystemExit("native finite_hull_seed is not built")
    rng = np.random.default_rng(args.seed)
    result = {"seed": args.seed, "numpy": np.__version__, "python": sys.version.split()[0]}
    if args.capture:
        result["capture"] = check_capture(args.capture, args.rounds)
        print("capture", json.dumps(result["capture"]), flush=True)
    if args.predicates:
        result["predicates"] = check_predicates(rng, args.predicates)
        print("predicates", json.dumps(result["predicates"]), flush=True)
    if args.triangulations:
        result["triangulations"] = check_triangulations(rng, args.triangulations)
        print("triangulations", json.dumps({k: v for k, v in result["triangulations"].items() if k != "first"}),
              flush=True)
    bad = sum(part.get("mismatches", 0) for part in result.values() if isinstance(part, dict))
    result["total_mismatches"] = bad
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
    print("total mismatches", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
