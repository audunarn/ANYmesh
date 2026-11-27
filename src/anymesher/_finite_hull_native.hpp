// Native finite-hull seed: decision-identical port of anymesher.triangulation._finite_hull_seed
// (fan seed, exact point insertion, Lawson flips) for exact float64 (n, 2) finite ndarrays.
//
// Decision identity.  The Python routine uses orient2d/incircle only through their signs.
// Both Python predicates return exactly signed values for finite operands (float filter,
// exact fallback).  Here the same float filters run with the same IEEE operations (no FP
// contraction; incircle's filter uses a port of CPython's math.fsum), and every value the
// filter does not certify is decided exactly with integers over one power-of-two
// denominator.  The control flow, row order, list updates, flip order and callback points
// follow the Python code line by line.  After every callback the input is compared with its
// snapshot (shape, strides, address, dtype, bytes); on a change the routine continues on the
// live array through Python helpers that evaluate the original predicate expressions.
// vertex_edges are real Python sets of edge tuples so that their iteration order is Python's.
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace anymesher_fhs {

#if defined(__clang__)
#define ANYMESHER_FHS_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define ANYMESHER_FHS_NO_CONTRACT
#endif

constexpr double kEps = 2.220446049250313e-16;  // np.finfo(np.float64).eps == 2**-52
const double kOrientCap = std::ldexp(1.0, 400);
const double kOrientFloor = std::ldexp(1.0, -800);
const double kIncircleCap = std::ldexp(1.0, 200);
const double kIncircleFloor = std::ldexp(1.0, -500);

// ---------------------------------------------------------------------------------------
// Fixed-capacity signed integers (exact fallback).  Every finite binary64 is m * 2**e with
// |m| < 2**53 and -1126 <= e <= 971, so over the common denominator 2**min(e) a coordinate
// has at most 2150 bits, a difference 2151, and the degree-four incircle determinant at most
// 4 * 2151 + 4 bits: 270 32-bit limbs.
constexpr int kLimbs = 288;

struct Big {
    int sign = 0;  // -1, 0, +1
    int n = 0;     // used limbs of the magnitude (n == 0 <=> zero)
    uint32_t d[kLimbs];
};

inline void big_trim(Big& r) {
    while (r.n > 0 && r.d[r.n - 1] == 0) --r.n;
    if (r.n == 0) r.sign = 0;
}

inline bool big_from_shifted(Big& r, uint64_t magnitude, int negative, int shift) {
    r.sign = 0;
    r.n = 0;
    if (magnitude == 0) return true;
    const int word = shift / 32, bit = shift % 32;
    if (word + 3 > kLimbs) return false;
    for (int i = 0; i < word; ++i) r.d[i] = 0;
    const uint64_t low = magnitude << bit;
    const uint64_t high = bit == 0 ? 0 : (magnitude >> (64 - bit));
    r.d[word] = static_cast<uint32_t>(low);
    r.d[word + 1] = static_cast<uint32_t>(low >> 32);
    r.d[word + 2] = static_cast<uint32_t>(high);
    r.n = word + 3;
    r.sign = negative ? -1 : 1;
    big_trim(r);
    return true;
}

inline int big_cmp_mag(const Big& a, const Big& b) {
    if (a.n != b.n) return a.n < b.n ? -1 : 1;
    for (int i = a.n - 1; i >= 0; --i) {
        if (a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
    }
    return 0;
}

// r = |a| + |b|
inline bool big_add_mag(Big& r, const Big& a, const Big& b) {
    const int n = std::max(a.n, b.n);
    if (n + 1 > kLimbs) return false;
    uint64_t carry = 0;
    for (int i = 0; i < n; ++i) {
        const uint64_t s = static_cast<uint64_t>(i < a.n ? a.d[i] : 0) + (i < b.n ? b.d[i] : 0) + carry;
        r.d[i] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
    r.d[n] = static_cast<uint32_t>(carry);
    r.n = n + 1;
    return true;
}

// r = |a| - |b|, requires |a| >= |b|
inline void big_sub_mag(Big& r, const Big& a, const Big& b) {
    int64_t borrow = 0;
    for (int i = 0; i < a.n; ++i) {
        int64_t s = static_cast<int64_t>(a.d[i]) - (i < b.n ? b.d[i] : 0) - borrow;
        borrow = s < 0 ? 1 : 0;
        if (s < 0) s += (static_cast<int64_t>(1) << 32);
        r.d[i] = static_cast<uint32_t>(s);
    }
    r.n = a.n;
}

// r = a + sb * b (sb = +1 or -1); r must not alias a or b
inline bool big_add(Big& r, const Big& a, const Big& b, int sb) {
    const int bs = b.sign * sb;
    if (b.sign == 0) {
        r.sign = a.sign; r.n = a.n; std::memcpy(r.d, a.d, sizeof(uint32_t) * a.n);
        return true;
    }
    if (a.sign == 0) {
        r.sign = bs; r.n = b.n; std::memcpy(r.d, b.d, sizeof(uint32_t) * b.n);
        return true;
    }
    if (a.sign == bs) {
        if (!big_add_mag(r, a, b)) return false;
        r.sign = a.sign;
    } else {
        const int c = big_cmp_mag(a, b);
        if (c == 0) { r.sign = 0; r.n = 0; return true; }
        if (c > 0) { big_sub_mag(r, a, b); r.sign = a.sign; }
        else { big_sub_mag(r, b, a); r.sign = bs; }
    }
    big_trim(r);
    return true;
}

// r = a * b; r must not alias a or b
inline bool big_mul(Big& r, const Big& a, const Big& b) {
    if (a.sign == 0 || b.sign == 0) { r.sign = 0; r.n = 0; return true; }
    const int n = a.n + b.n;
    if (n > kLimbs) return false;
    for (int i = 0; i < n; ++i) r.d[i] = 0;
    for (int i = 0; i < a.n; ++i) {
        uint64_t carry = 0;
        const uint64_t ai = a.d[i];
        for (int j = 0; j < b.n; ++j) {
            const uint64_t t = ai * b.d[j] + r.d[i + j] + carry;
            r.d[i + j] = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        int k = i + b.n;
        while (carry != 0) {
            const uint64_t t = static_cast<uint64_t>(r.d[k]) + carry;
            r.d[k] = static_cast<uint32_t>(t);
            carry = t >> 32;
            ++k;
        }
    }
    r.n = n;
    r.sign = a.sign * b.sign;
    big_trim(r);
    return true;
}

struct Decomposed {
    uint64_t m = 0;
    int negative = 0;
    int e = 0;
};

inline Decomposed decompose(double v) {
    Decomposed out;
    if (v == 0.0) return out;
    int k = 0;
    const double f = std::frexp(v, &k);
    const double scaled = std::ldexp(std::fabs(f), 53);
    out.m = static_cast<uint64_t>(scaled);
    out.negative = v < 0.0;
    out.e = k - 53;
    return out;
}

// Integers over the common denominator 2**min(e); false only on an (impossible) capacity overflow.
template <size_t N>
inline bool to_common(const std::array<double, N>& values, std::array<Big, N>& out) {
    std::array<Decomposed, N> parts;
    bool any = false;
    int emin = 0;
    for (size_t i = 0; i < N; ++i) {
        parts[i] = decompose(values[i]);
        if (parts[i].m != 0) {
            emin = any ? std::min(emin, parts[i].e) : parts[i].e;
            any = true;
        }
    }
    for (size_t i = 0; i < N; ++i) {
        if (!big_from_shifted(out[i], parts[i].m, parts[i].negative, parts[i].m ? parts[i].e - emin : 0)) {
            return false;
        }
    }
    return true;
}

struct CapacityError {};

inline void need(bool ok) {
    if (!ok) throw CapacityError();
}

// Exact sign of (fx - tx) * (sy - ty) - (fy - ty) * (sx - tx).
inline int exact_orient_sign(double fx, double fy, double sx, double sy, double tx, double ty) {
    std::array<double, 6> values{fx, fy, sx, sy, tx, ty};
    std::array<Big, 6> v;
    need(to_common(values, v));
    Big ax, ay, bx, by, left, right, det;
    need(big_add(ax, v[0], v[4], -1));
    need(big_add(ay, v[1], v[5], -1));
    need(big_add(bx, v[2], v[4], -1));
    need(big_add(by, v[3], v[5], -1));
    need(big_mul(left, ax, by));
    need(big_mul(right, ay, bx));
    need(big_add(det, left, right, -1));
    return det.sign;
}

// The sign anymesher.triangulation.orient2d returns, for finite operands.
inline int orient_sign(double fx, double fy, double sx, double sy, double tx, double ty) {
    ANYMESHER_FHS_NO_CONTRACT
    const double ax = fx - tx;
    const double ay = fy - ty;
    const double bx = sx - tx;
    const double by = sy - ty;
    const double left = ax * by;
    const double right = ay * bx;
    const double determinant = left - right;
    const double magnitude = std::fabs(left) + std::fabs(right);
    const double error = 8.0 * kEps * magnitude;
    double largest = std::fabs(ax);
    if (std::fabs(ay) > largest) largest = std::fabs(ay);
    if (std::fabs(bx) > largest) largest = std::fabs(bx);
    if (std::fabs(by) > largest) largest = std::fabs(by);
    if (std::fabs(determinant) > error && magnitude >= kOrientFloor && largest <= kOrientCap) {
        return determinant > 0.0 ? 1 : (determinant < 0.0 ? -1 : 0);
    }
    return exact_orient_sign(fx, fy, sx, sy, tx, ty);
}

// CPython's math.fsum for finite terms whose partial sums stay finite (the caller's terms are
// bounded by 2**803).  Correctly rounded; the same algorithm and operations as mathmodule.c.
inline double fsum_finite(const double* terms, int count) {
    ANYMESHER_FHS_NO_CONTRACT
    double p[8];
    int n = 0;
    for (int k = 0; k < count; ++k) {
        double x = terms[k];
        int i = 0;
        for (int j = 0; j < n; ++j) {
            double y = p[j];
            if (std::fabs(x) < std::fabs(y)) {
                const double t = x; x = y; y = t;
            }
            const double hi = x + y;
            const double lo = y - (hi - x);
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        n = i;
        if (x != 0.0) p[n++] = x;
    }
    double hi = 0.0;
    if (n > 0) {
        double lo = 0.0;
        hi = p[--n];
        while (n > 0) {
            const double x = hi;
            const double y = p[--n];
            hi = x + y;
            const double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            const double y = lo * 2.0;
            const double x = hi + y;
            const double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

inline int exact_incircle_sign(const std::array<double, 8>& values) {
    std::array<Big, 8> v;
    need(to_common(values, v));
    Big dax, day, dbx, dby, dcx, dcy;
    need(big_add(dax, v[0], v[6], -1));
    need(big_add(day, v[1], v[7], -1));
    need(big_add(dbx, v[2], v[6], -1));
    need(big_add(dby, v[3], v[7], -1));
    need(big_add(dcx, v[4], v[6], -1));
    need(big_add(dcy, v[5], v[7], -1));
    Big s1, s2, da, db, dc, c1, c2, cross_a, cross_b, cross_c, ta, tb, tc, sum1, total;
    need(big_mul(s1, dax, dax)); need(big_mul(s2, day, day)); need(big_add(da, s1, s2, 1));
    need(big_mul(s1, dbx, dbx)); need(big_mul(s2, dby, dby)); need(big_add(db, s1, s2, 1));
    need(big_mul(s1, dcx, dcx)); need(big_mul(s2, dcy, dcy)); need(big_add(dc, s1, s2, 1));
    need(big_mul(c1, dbx, dcy)); need(big_mul(c2, dby, dcx)); need(big_add(cross_a, c1, c2, -1));
    need(big_mul(c1, dax, dcy)); need(big_mul(c2, day, dcx)); need(big_add(cross_b, c1, c2, -1));
    need(big_mul(c1, dax, dby)); need(big_mul(c2, day, dbx)); need(big_add(cross_c, c1, c2, -1));
    need(big_mul(ta, da, cross_a));
    need(big_mul(tb, db, cross_b));
    need(big_mul(tc, dc, cross_c));
    need(big_add(sum1, ta, tb, -1));
    need(big_add(total, sum1, tc, 1));
    return total.sign;
}

// The sign anymesher.triangulation.incircle(first, second, third, point) returns, for finite
// operands (positive for a point inside the circle of a counter-clockwise triangle).
inline int incircle_sign(double fx, double fy, double sx, double sy, double tx, double ty, double px, double py) {
    ANYMESHER_FHS_NO_CONTRACT
    const double ax = fx - px, ay = fy - py;
    const double bx = sx - px, by = sy - py;
    const double cx = tx - px, cy = ty - py;
    double largest = std::fabs(ax);
    for (double value : {ay, bx, by, cx, cy}) {
        if (std::fabs(value) > largest) largest = std::fabs(value);
    }
    if (largest <= kIncircleCap) {
        const double alift = ax * ax + ay * ay;
        const double blift = bx * bx + by * by;
        const double clift = cx * cx + cy * cy;
        const double terms[3] = {
            alift * (bx * cy - by * cx),
            -blift * (ax * cy - ay * cx),
            clift * (ax * by - ay * bx),
        };
        const double determinant = fsum_finite(terms, 3);
        const double scale = alift * (std::fabs(bx * cy) + std::fabs(by * cx))
                             + blift * (std::fabs(ax * cy) + std::fabs(ay * cx))
                             + clift * (std::fabs(ax * by) + std::fabs(ay * bx));
        if (scale >= kIncircleFloor && std::fabs(determinant) > 32.0 * kEps * scale) {
            const int s = determinant > 0.0 ? 1 : (determinant < 0.0 ? -1 : 0);
            return orient_sign(fx, fy, sx, sy, tx, ty) < 0 ? -s : s;
        }
    }
    const int s = exact_incircle_sign({fx, fy, sx, sy, tx, ty, px, py});
    return orient_sign(fx, fy, sx, sy, tx, ty) < 0 ? -s : s;
}

// ---------------------------------------------------------------------------------------
// Engine.

using Tri = std::array<int64_t, 3>;

inline uint64_t edge_key(int64_t a, int64_t b) {
    const uint64_t lo = static_cast<uint64_t>(a < b ? a : b);
    const uint64_t hi = static_cast<uint64_t>(a < b ? b : a);
    return (lo << 32) | hi;
}
inline int64_t key_first(uint64_t key) { return static_cast<int64_t>(key >> 32); }
inline int64_t key_second(uint64_t key) { return static_cast<int64_t>(key & 0xffffffffu); }

struct Rows {
    int n = 0;
    int64_t r[2] = {0, 0};
};

struct PyError {};  // a Python exception is set

struct Ref {
    PyObject* p = nullptr;
    explicit Ref(PyObject* o = nullptr) : p(o) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
};

inline PyObject* checked(PyObject* o) {
    if (o == nullptr) throw PyError();
    return o;
}

struct Snapshot {
    std::vector<Py_ssize_t> shape, strides;
    void* buf = nullptr;
    PyObject* dtype = nullptr;  // owned
    std::vector<unsigned char> bytes;  // C-order contents
};

struct Engine {
    PyObject* points;     // the caller's array (borrowed; the caller holds it)
    PyObject* callback;   // or nullptr
    PyObject* location;   // live helpers
    PyObject* flip_test;
    PyObject* canonical;
    PyObject* stage_insertion;
    PyObject* stage_edge;
    PyObject* stage_legality;

    int64_t count = 0;
    std::vector<double> xs, ys;
    Snapshot snap;
    bool live = false;

    std::vector<Tri> triangles;
    std::unordered_map<uint64_t, Rows> incidence;
    std::unordered_set<uint64_t> believed;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> pending;
    std::vector<PyObject*> vertex_edges;  // owned Python sets (or nullptr)
    Ref dtype_name{nullptr};

    ~Engine() {
        Py_XDECREF(snap.dtype);
        for (PyObject* s : vertex_edges) Py_XDECREF(s);
    }

    struct BufferRelease {
        Py_buffer* view;
        ~BufferRelease() { PyBuffer_Release(view); }
    };

    static bool is_native_float64(const Py_buffer& view) {
        if (view.itemsize != 8 || view.format == nullptr) return false;
        const char* f = view.format;
#if PY_BIG_ENDIAN
        const char order = '>';
#else
        const char order = '<';
#endif
        if (f[0] == 'd' && f[1] == 0) return true;
        return (f[0] == '@' || f[0] == '=' || f[0] == order) && f[1] == 'd' && f[2] == 0;
    }

    // Snapshot and coordinates.  The Python caller guarantees an exact float64 (n, 2) finite
    // ndarray; anything else (a direct call) is refused before any callback.
    void take_snapshot() {
        // Python-visible reads first; while the buffer view is held no Python code runs.
        dtype_name.p = checked(PyUnicode_FromString("dtype"));
        snap.dtype = checked(PyObject_GetAttr(points, dtype_name.p));
        Py_buffer view;
        if (PyObject_GetBuffer(points, &view, PyBUF_RECORDS_RO) != 0) throw PyError();
        BufferRelease release{&view};
        if (view.ndim != 2 || view.shape[1] != 2 || !is_native_float64(view) || view.shape[0] < 3
                || view.shape[0] >= (static_cast<Py_ssize_t>(1) << 28)) {
            PyErr_SetString(PyExc_ValueError,
                            "finite hull seed: points must be a float64 (n, 2) array with 3 <= n < 2**28");
            throw PyError();
        }
        snap.shape.assign(view.shape, view.shape + 2);
        snap.strides.assign(view.strides, view.strides + 2);
        snap.buf = view.buf;
        count = view.shape[0];
        snap.bytes.resize(static_cast<size_t>(count) * 16);
        xs.resize(static_cast<size_t>(count));
        ys.resize(static_cast<size_t>(count));
        const char* base = static_cast<const char*>(view.buf);
        for (int64_t i = 0; i < count; ++i) {
            for (int j = 0; j < 2; ++j) {
                std::memcpy(snap.bytes.data() + i * 16 + j * 8, base + i * view.strides[0] + j * view.strides[1], 8);
            }
            std::memcpy(&xs[static_cast<size_t>(i)], snap.bytes.data() + i * 16, 8);
            std::memcpy(&ys[static_cast<size_t>(i)], snap.bytes.data() + i * 16 + 8, 8);
            if (!std::isfinite(xs[static_cast<size_t>(i)]) || !std::isfinite(ys[static_cast<size_t>(i)])) {
                PyErr_SetString(PyExc_ValueError, "finite hull seed: points must be finite");
                throw PyError();
            }
        }
    }

    // The option-1 checkpoint: call back, then drop to the live array on any change of
    // (shape, strides, dtype, address) or of the contents (the snapshot's bytes).
    void checkpoint(PyObject* stage) {
        Py_DECREF(checked(PyObject_CallOneArg(callback, stage)));  // discarded first, as in Python
        if (live) return;
        {
            // dtype first, without a view held (an exact ndarray: numpy's own getter and comparison).
            Ref dtype(checked(PyObject_GetAttr(points, dtype_name.p)));
            if (dtype.p != snap.dtype) {
                const int equal = PyObject_RichCompareBool(dtype.p, snap.dtype, Py_EQ);
                if (equal < 0) throw PyError();
                if (!equal) {
                    live = true;
                    return;
                }
            }
        }
        Py_buffer view;
        if (PyObject_GetBuffer(points, &view, PyBUF_RECORDS_RO) != 0) {
            PyErr_Clear();  // not exportable: the dtype or the layout changed
            live = true;
            return;
        }
        BufferRelease release{&view};
        // From here to the release no Python code runs; the bytes are read under the export.
        if (view.buf != snap.buf || view.ndim != static_cast<int>(snap.shape.size()) || !is_native_float64(view)
                || !std::equal(snap.shape.begin(), snap.shape.end(), view.shape)
                || !std::equal(snap.strides.begin(), snap.strides.end(), view.strides)) {
            live = true;
            return;
        }
        const Py_ssize_t n = view.shape[0];
        const char* base = static_cast<const char*>(view.buf);
        if (view.strides[0] == 16 && view.strides[1] == 8) {
            live = std::memcmp(base, snap.bytes.data(), static_cast<size_t>(n) * 16) != 0;
            return;
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            for (int j = 0; j < 2; ++j) {
                const char* item = base + i * view.strides[0] + j * view.strides[1];
                if (std::memcmp(snap.bytes.data() + static_cast<size_t>(i) * 16 + static_cast<size_t>(j) * 8,
                                item, 8) != 0) {
                    live = true;
                    return;
                }
            }
        }
    }

    PyObject* tri_tuple(const Tri& t) {
        return checked(Py_BuildValue("(LLL)", static_cast<long long>(t[0]), static_cast<long long>(t[1]),
                                     static_cast<long long>(t[2])));
    }

    static void bad_result(const char* what) {
        PyErr_Format(PyExc_SystemError, "finite hull seed: unexpected %s", what);
        throw PyError();
    }

    // An exact int only: converting it runs no Python code.
    static int64_t as_index(PyObject* o) {
        if (!PyLong_CheckExact(o)) bad_result("non-int index");
        const long long v = PyLong_AsLongLong(o);
        if (v == -1 && PyErr_Occurred()) throw PyError();
        return static_cast<int64_t>(v);
    }

    Tri canonical_triangle(const Tri& raw) {
        if (live) {
            Ref t(tri_tuple(raw));
            Ref result(checked(PyObject_CallFunctionObjArgs(canonical, t.p, points, nullptr)));
            if (!PyTuple_CheckExact(result.p) || PyTuple_GET_SIZE(result.p) != 3) {
                PyErr_SetString(PyExc_SystemError, "finite hull seed: unexpected canonical triangle");
                throw PyError();
            }
            const Tri out{as_index(PyTuple_GET_ITEM(result.p, 0)), as_index(PyTuple_GET_ITEM(result.p, 1)),
                          as_index(PyTuple_GET_ITEM(result.p, 2))};
            // Only a permutation of the input triangle is used (a foreign helper on a direct call
            // could return anything; the library's _canonical_triangle always permutes).
            Tri a = raw, b = out;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a != b) {
                PyErr_SetString(PyExc_SystemError, "finite hull seed: canonical helper did not permute the triangle");
                throw PyError();
            }
            return out;
        }
        Tri v = raw;
        if (orient_sign(xs[v[0]], ys[v[0]], xs[v[1]], ys[v[1]], xs[v[2]], ys[v[2]]) < 0) std::swap(v[1], v[2]);
        int start = 0;
        if (v[1] < v[start]) start = 1;
        if (v[2] < v[start]) start = 2;
        return Tri{v[start], v[(start + 1) % 3], v[(start + 2) % 3]};
    }

    // Location test of one cell: returns "closed-contains", fills zero[i] for zero turns.
    bool locate_in(const Tri& t, int64_t node, bool zero[3]) {
        if (live) {
            Ref tt(tri_tuple(t));
            Ref nn(checked(PyLong_FromLongLong(node)));
            Ref result(checked(PyObject_CallFunctionObjArgs(location, points, tt.p, nn.p, nullptr)));
            if (!PyTuple_CheckExact(result.p) || PyTuple_GET_SIZE(result.p) != 2) bad_result("location result");
            PyObject* flag = PyTuple_GET_ITEM(result.p, 0);
            PyObject* zeros = PyTuple_GET_ITEM(result.p, 1);
            if (!PyBool_Check(flag) || !PyList_CheckExact(zeros)) bad_result("location result");
            const bool contains = flag == Py_True;
            zero[0] = zero[1] = zero[2] = false;
            // The result tuple owns the list; its items are exact ints, so nothing can resize it here.
            for (Py_ssize_t k = 0; k < PyList_GET_SIZE(zeros); ++k) {
                const int64_t position = as_index(PyList_GET_ITEM(zeros, k));
                if (position < 0 || position > 2) {
                    PyErr_SetString(PyExc_SystemError, "finite hull seed: location helper returned a bad edge position");
                    throw PyError();
                }
                zero[position] = true;
            }
            return contains;
        }
        int s[3];
        for (int i = 0; i < 3; ++i) {
            const int64_t p = t[i], q = t[(i + 1) % 3];
            s[i] = orient_sign(xs[p], ys[p], xs[q], ys[q], xs[node], ys[node]);
            zero[i] = s[i] == 0;
        }
        return s[0] >= 0 && s[1] >= 0 && s[2] >= 0;
    }

    static int64_t other_vertex(const Tri& t, int64_t e0, int64_t e1) {
        for (int i = 0; i < 3; ++i) {
            if (t[i] != e0 && t[i] != e1) return t[i];
        }
        PyErr_SetString(PyExc_SystemError, "finite hull seed: degenerate cell");
        throw PyError();
    }

    Rows& rows_of(uint64_t key) {
        auto found = incidence.find(key);
        if (found == incidence.end()) {
            PyErr_SetString(PyExc_SystemError, "finite hull seed: missing edge");
            throw PyError();
        }
        return found->second;
    }

    bool flippable(uint64_t edge, const Rows& rows) {
        if (rows.n != 2) return false;
        const Tri first = triangles[static_cast<size_t>(rows.r[0])];
        const Tri second = triangles[static_cast<size_t>(rows.r[1])];
        const int64_t e0 = key_first(edge), e1 = key_second(edge);
        const int64_t a = other_vertex(first, e0, e1);
        const int64_t b = other_vertex(second, e0, e1);
        if (incidence.count(edge_key(a, b)) != 0) return false;
        if (live) {
            Ref ft(tri_tuple(first));
            Ref result(checked(PyObject_CallFunction(flip_test, "OLLLLO", points, static_cast<long long>(e0),
                                                     static_cast<long long>(e1), static_cast<long long>(a),
                                                     static_cast<long long>(b), ft.p)));
            if (!PyBool_Check(result.p)) bad_result("flip test result");
            return result.p == Py_True;
        }
        const int first_turn = orient_sign(xs[e0], ys[e0], xs[e1], ys[e1], xs[a], ys[a]);
        const int second_turn = orient_sign(xs[e0], ys[e0], xs[e1], ys[e1], xs[b], ys[b]);
        const int third_turn = orient_sign(xs[a], ys[a], xs[b], ys[b], xs[e0], ys[e0]);
        const int fourth_turn = orient_sign(xs[a], ys[a], xs[b], ys[b], xs[e1], ys[e1]);
        const bool proper = ((first_turn > 0 && second_turn < 0) || (first_turn < 0 && second_turn > 0)) &&
                            ((third_turn > 0 && fourth_turn < 0) || (third_turn < 0 && fourth_turn > 0));
        if (!proper) return false;
        return incircle_sign(xs[first[0]], ys[first[0]], xs[first[1]], ys[first[1]], xs[first[2]], ys[first[2]],
                             xs[b], ys[b]) > 0;
    }

    void examine(uint64_t edge) {
        if (callback != nullptr) checkpoint(stage_edge);
        if (believed.count(edge) != 0) return;
        if (flippable(edge, rows_of(edge))) {
            believed.insert(edge);
            pending.push(edge);
        }
    }

    PyObject* edge_tuple(uint64_t key) {
        return checked(Py_BuildValue("(LL)", static_cast<long long>(key_first(key)),
                                     static_cast<long long>(key_second(key))));
    }

    void set_add(int64_t vertex, PyObject* edge) {
        PyObject*& s = vertex_edges[static_cast<size_t>(vertex)];
        if (s == nullptr) {
            s = checked(PySet_New(nullptr));
            PyObject_GC_UnTrack(s);  // private: invisible to gc.get_objects(); holds no cycles
        }
        if (PySet_Add(s, edge) != 0) throw PyError();
    }

    void set_discard(int64_t vertex, PyObject* edge) {
        PyObject* s = vertex_edges[static_cast<size_t>(vertex)];
        if (s == nullptr) {
            PyErr_SetString(PyExc_SystemError, "finite hull seed: missing vertex edges");
            throw PyError();
        }
        if (PySet_Discard(s, edge) < 0) throw PyError();
    }

    static void replace_row(Rows& rows, int64_t old_row, int64_t new_row) {
        int at = -1;
        for (int i = 0; i < rows.n; ++i) {
            if (rows.r[i] == old_row) { at = i; break; }
        }
        if (at < 0) {
            PyErr_SetString(PyExc_SystemError, "finite hull seed: missing row");
            throw PyError();
        }
        rows.r[at] = new_row;
        if (rows.n == 2 && rows.r[1] < rows.r[0]) std::swap(rows.r[0], rows.r[1]);
    }

    // Returns the sorted triangles (new list), or a Python int status: 1 = cannot locate,
    // 2 = legality did not converge.
    PyObject* run(const std::vector<int64_t>& hull) {
        take_snapshot();
        {
            // Distinct, in-range rows (the Python caller routes every other hull to Python).
            std::vector<char> seen(static_cast<size_t>(count), 0);
            bool valid = hull.size() >= 3;
            for (int64_t h : hull) {
                if (!valid) break;
                if (h < 0 || h >= count || seen[static_cast<size_t>(h)]) valid = false;
                else seen[static_cast<size_t>(h)] = 1;
            }
            if (!valid) {
                PyErr_SetString(PyExc_ValueError,
                                "finite hull seed: hull must hold at least three distinct point rows");
                throw PyError();
            }
        }
        for (size_t i = 1; i + 1 < hull.size(); ++i) {
            triangles.push_back(canonical_triangle(Tri{hull[0], hull[i], hull[i + 1]}));
        }
        std::vector<char> installed(static_cast<size_t>(count), 0);
        for (int64_t h : hull) installed[static_cast<size_t>(h)] = 1;
        std::vector<int64_t> order;
        for (int64_t i = 0; i < count; ++i) {
            if (!installed[static_cast<size_t>(i)]) order.push_back(i);
        }
        std::sort(order.begin(), order.end(), [this](int64_t p, int64_t q) {
            const double px = xs[static_cast<size_t>(p)], qx = xs[static_cast<size_t>(q)];
            if (px != qx) return px < qx;
            const double py = ys[static_cast<size_t>(p)], qy = ys[static_cast<size_t>(q)];
            if (py != qy) return py < qy;
            return p < q;
        });
        for (int64_t node : order) {
            if (callback != nullptr) checkpoint(stage_insertion);
            std::vector<int64_t> containing;
            bool has_edge = false;
            int64_t edge0 = 0, edge1 = 0;
            for (size_t row = 0; row < triangles.size(); ++row) {
                const Tri triangle = triangles[row];
                bool zero[3];
                if (locate_in(triangle, node, zero)) {
                    containing.push_back(static_cast<int64_t>(row));
                    for (int i = 0; i < 3; ++i) {
                        if (zero[i]) {
                            const int64_t p = triangle[i], q = triangle[(i + 1) % 3];
                            edge0 = std::min(p, q);
                            edge1 = std::max(p, q);
                            has_edge = true;
                        }
                    }
                    if (!has_edge) break;
                }
            }
            if (containing.empty()) return PyLong_FromLong(1);
            std::vector<Tri> replacements;
            if (has_edge) {
                containing.clear();
                for (size_t row = 0; row < triangles.size(); ++row) {
                    const Tri& t = triangles[row];
                    const bool h0 = t[0] == edge0 || t[1] == edge0 || t[2] == edge0;
                    const bool h1 = t[0] == edge1 || t[1] == edge1 || t[2] == edge1;
                    if (h0 && h1) containing.push_back(static_cast<int64_t>(row));
                }
                for (int64_t row : containing) {
                    const int64_t opposite = other_vertex(triangles[static_cast<size_t>(row)], edge0, edge1);
                    replacements.push_back(Tri{edge0, node, opposite});
                    replacements.push_back(Tri{node, edge1, opposite});
                }
            } else {
                const Tri t = triangles[static_cast<size_t>(containing[0])];
                replacements.push_back(Tri{t[0], t[1], node});
                replacements.push_back(Tri{t[1], t[2], node});
                replacements.push_back(Tri{t[2], t[0], node});
            }
            std::vector<Tri> kept;
            kept.reserve(triangles.size() + replacements.size());
            std::vector<char> removed(triangles.size(), 0);
            for (int64_t row : containing) removed[static_cast<size_t>(row)] = 1;
            for (size_t row = 0; row < triangles.size(); ++row) {
                if (!removed[row]) kept.push_back(triangles[row]);
            }
            triangles.swap(kept);
            for (const Tri& t : replacements) triangles.push_back(canonical_triangle(t));
        }

        // max(64, 32 * cells**2), exact: n < 2**28 gives cells < 2**29 and a limit below 2**63.
        const uint64_t cells = triangles.size();
        uint64_t limit = 32 * cells * cells;
        if (limit < 64) limit = 64;

        std::vector<uint64_t> first_seen;
        for (size_t row = 0; row < triangles.size(); ++row) {
            for (int i = 0; i < 3; ++i) {
                const uint64_t key = edge_key(triangles[row][i], triangles[row][(i + 1) % 3]);
                auto found = incidence.find(key);
                if (found == incidence.end()) {
                    Rows rows;
                    rows.n = 1;
                    rows.r[0] = static_cast<int64_t>(row);
                    incidence.emplace(key, rows);
                    first_seen.push_back(key);
                } else {
                    if (found->second.n >= 2) {
                        PyErr_SetString(PyExc_SystemError, "finite hull seed: nonmanifold edge");
                        throw PyError();
                    }
                    found->second.r[found->second.n++] = static_cast<int64_t>(row);
                }
            }
        }
        vertex_edges.assign(static_cast<size_t>(count), nullptr);
        for (uint64_t key : first_seen) {
            Ref edge(edge_tuple(key));
            set_add(key_first(key), edge.p);
            set_add(key_second(key), edge.p);
        }
        std::vector<uint64_t> sorted_edges = first_seen;
        std::sort(sorted_edges.begin(), sorted_edges.end());
        for (uint64_t key : sorted_edges) examine(key);

        for (uint64_t iteration = 0; iteration < limit; ++iteration) {
            if (callback != nullptr) checkpoint(stage_legality);
            bool found_edge = false;
            uint64_t edge = 0;
            while (!pending.empty()) {
                const uint64_t candidate = pending.top();
                pending.pop();
                if (believed.count(candidate) == 0) continue;
                if (callback != nullptr) checkpoint(stage_edge);
                if (!flippable(candidate, rows_of(candidate))) {
                    believed.erase(candidate);
                    continue;
                }
                edge = candidate;
                found_edge = true;
                break;
            }
            if (!found_edge) {
                std::vector<Tri> result = triangles;
                std::sort(result.begin(), result.end());
                // Every tuple exists before the list does: no allocation (hence no GC pass) can
                // see a partly filled list.
                std::vector<PyObject*> items;
                items.reserve(result.size());
                struct Owned {
                    std::vector<PyObject*>* v;
                    ~Owned() { for (PyObject* o : *v) Py_XDECREF(o); }
                } owned{&items};
                for (const Tri& t : result) items.push_back(tri_tuple(t));
                PyObject* out = checked(PyList_New(static_cast<Py_ssize_t>(items.size())));
                for (size_t i = 0; i < items.size(); ++i) {
                    PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), items[i]);  // steals
                    items[i] = nullptr;
                }
                return out;
            }
            const Rows rows = rows_of(edge);
            const int64_t r0 = rows.r[0], r1 = rows.r[1];
            const Tri first = triangles[static_cast<size_t>(r0)];
            const Tri second = triangles[static_cast<size_t>(r1)];
            const int64_t e0 = key_first(edge), e1 = key_second(edge);
            const int64_t a = other_vertex(first, e0, e1);
            const int64_t b = other_vertex(second, e0, e1);
            triangles[static_cast<size_t>(r0)] = canonical_triangle(Tri{a, b, e0});
            triangles[static_cast<size_t>(r1)] = canonical_triangle(Tri{b, a, e1});
            believed.erase(edge);
            incidence.erase(edge);
            {
                Ref old_edge(edge_tuple(edge));
                set_discard(e0, old_edge.p);
                set_discard(e1, old_edge.p);
            }
            const uint64_t diagonal = edge_key(a, b);
            Rows diagonal_rows;
            diagonal_rows.n = 2;
            diagonal_rows.r[0] = r0;
            diagonal_rows.r[1] = r1;
            incidence[diagonal] = diagonal_rows;
            {
                Ref diagonal_tuple(edge_tuple(diagonal));
                set_add(a, diagonal_tuple.p);
                set_add(b, diagonal_tuple.p);
            }
            replace_row(rows_of(edge_key(a, e1)), r0, r1);
            replace_row(rows_of(edge_key(b, e0)), r1, r0);
            const uint64_t sides[5] = {edge_key(a, e0), edge_key(a, e1), edge_key(b, e0), edge_key(b, e1), diagonal};
            for (uint64_t side : sides) examine(side);
            Ref iterator(checked(PyObject_GetIter(vertex_edges[static_cast<size_t>(e0)])));
            while (true) {
                Ref near_edge(PyIter_Next(iterator.p));
                if (near_edge.p == nullptr) {
                    if (PyErr_Occurred()) throw PyError();
                    break;
                }
                // The sets are private but reachable (gc.get_objects): check every entry.
                if (!PyTuple_CheckExact(near_edge.p) || PyTuple_GET_SIZE(near_edge.p) != 2) bad_result("edge entry");
                const int64_t n0 = as_index(PyTuple_GET_ITEM(near_edge.p, 0));
                const int64_t n1 = as_index(PyTuple_GET_ITEM(near_edge.p, 1));
                if (n0 < 0 || n0 >= count || n1 < 0 || n1 >= count || n0 == n1) bad_result("edge entry");
                const Rows near_rows = rows_of(edge_key(n0, n1));
                for (int k = 0; k < near_rows.n; ++k) {
                    const int64_t row = near_rows.r[k];
                    const Tri triangle = triangles[static_cast<size_t>(row)];
                    const int64_t x = n1 == e0 ? n0 : n1;
                    const int64_t w = other_vertex(triangle, n0, n1);
                    const uint64_t far = edge_key(x, w);
                    const Rows far_rows = rows_of(far);
                    if (far_rows.n == 2) {
                        const int64_t other = far_rows.r[1] == row ? far_rows.r[0] : far_rows.r[1];
                        const Tri& o = triangles[static_cast<size_t>(other)];
                        if (o[0] == e1 || o[1] == e1 || o[2] == e1) examine(far);
                    }
                }
            }
        }
        return PyLong_FromLong(2);
    }
};

// finite_hull_seed(points, hull, cancellation_check, location, flip_test, canonical, stages)
// stages = (insertion, edge, legality): the very str objects the Python route passes.
inline PyObject* py_finite_hull_seed(PyObject*, PyObject* args) {
    PyObject *points, *hull_object, *callback, *location, *flip_test, *canonical, *stages;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &points, &hull_object, &callback, &location, &flip_test, &canonical,
                          &stages)) {
        return nullptr;
    }
    if (!PyTuple_CheckExact(stages) || PyTuple_GET_SIZE(stages) != 3) {
        PyErr_SetString(PyExc_TypeError, "finite hull seed: stages must be a tuple of three str");
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < 3; ++i) {
        if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(stages, i))) {
            PyErr_SetString(PyExc_TypeError, "finite hull seed: stages must be a tuple of three str");
            return nullptr;
        }
    }
    // Strong references for the whole run (the tuple itself is immutable and held by the caller,
    // but the engine never relies on that).
    Ref stage_insertion(Py_NewRef(PyTuple_GET_ITEM(stages, 0)));
    Ref stage_edge(Py_NewRef(PyTuple_GET_ITEM(stages, 1)));
    Ref stage_legality(Py_NewRef(PyTuple_GET_ITEM(stages, 2)));
    {
        // Exact numpy.ndarray only (its type cannot be reassigned): attribute reads and the buffer
        // export are numpy's own and run no Python code.
        Ref numpy(PyImport_ImportModule("numpy"));
        if (numpy.p == nullptr) return nullptr;
        Ref ndarray(PyObject_GetAttrString(numpy.p, "ndarray"));
        if (ndarray.p == nullptr) return nullptr;
        if (reinterpret_cast<PyObject*>(Py_TYPE(points)) != ndarray.p) {
            PyErr_SetString(PyExc_TypeError, "finite hull seed: points must be an exact numpy.ndarray");
            return nullptr;
        }
    }
    Ref held_points(Py_NewRef(points));
    Ref held_helpers[3] = {Ref(Py_NewRef(location)), Ref(Py_NewRef(flip_test)), Ref(Py_NewRef(canonical))};
    Ref held_callback(Py_NewRef(callback));
    try {
        std::vector<int64_t> hull;
        {
            if (!PyList_CheckExact(hull_object) && !PyTuple_CheckExact(hull_object)) {
                PyErr_SetString(PyExc_TypeError, "finite hull seed: hull must be a list or tuple of int");
                return nullptr;
            }
            Ref sequence(PySequence_Fast(hull_object, "hull must be a sequence"));
            if (sequence.p == nullptr) return nullptr;
            const Py_ssize_t m = PySequence_Fast_GET_SIZE(sequence.p);
            for (Py_ssize_t i = 0; i < m; ++i) {
                PyObject* item = PySequence_Fast_GET_ITEM(sequence.p, i);
                if (!PyLong_CheckExact(item)) {
                    PyErr_SetString(PyExc_TypeError, "finite hull seed: hull rows must be int");
                    return nullptr;
                }
                const long long v = PyLong_AsLongLong(item);
                if (v == -1 && PyErr_Occurred()) return nullptr;
                hull.push_back(static_cast<int64_t>(v));
            }
        }
        for (PyObject* helper : {location, flip_test, canonical}) {
            if (!PyCallable_Check(helper)) {
                PyErr_SetString(PyExc_TypeError, "finite hull seed: helpers must be callable");
                return nullptr;
            }
        }
        // No callable check on cancellation_check: calling it raises exactly where Python does.
        Engine engine;
        engine.points = points;
        engine.callback = callback == Py_None ? nullptr : callback;
        engine.location = location;
        engine.flip_test = flip_test;
        engine.canonical = canonical;
        engine.stage_insertion = stage_insertion.p;
        engine.stage_edge = stage_edge.p;
        engine.stage_legality = stage_legality.p;
        return engine.run(hull);
    } catch (const PyError&) {
        return nullptr;
    } catch (const CapacityError&) {
        PyErr_SetString(PyExc_SystemError, "finite hull seed: exact arithmetic capacity exceeded");
        return nullptr;
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_SystemError, "finite hull seed: internal error");
        return nullptr;
    }
}

// Predicate probes for the differential harness: signs only.
inline PyObject* py_fhs_orient_sign(PyObject*, PyObject* args) {
    double v[6];
    if (!PyArg_ParseTuple(args, "dddddd", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5])) return nullptr;
    for (double value : v) {
        if (!std::isfinite(value)) {
            PyErr_SetString(PyExc_ValueError, "finite float64 values expected");
            return nullptr;
        }
    }
    try {
        return PyLong_FromLong(orient_sign(v[0], v[1], v[2], v[3], v[4], v[5]));
    } catch (const CapacityError&) {
        PyErr_SetString(PyExc_SystemError, "capacity");
        return nullptr;
    }
}

inline PyObject* py_fhs_incircle_sign(PyObject*, PyObject* args) {
    double v[8];
    if (!PyArg_ParseTuple(args, "dddddddd", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7])) {
        return nullptr;
    }
    for (double value : v) {
        if (!std::isfinite(value)) {
            PyErr_SetString(PyExc_ValueError, "finite float64 values expected");
            return nullptr;
        }
    }
    try {
        return PyLong_FromLong(incircle_sign(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]));
    } catch (const CapacityError&) {
        PyErr_SetString(PyExc_SystemError, "capacity");
        return nullptr;
    }
}

// Batch probe: rows of 8 doubles (fx fy sx sy tx ty px py) -> (orient(f,s,t), incircle(f,s,t,p)) signs.
inline PyObject* py_fhs_predicate_signs(PyObject*, PyObject* args) {
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) return nullptr;
    if (view.len % static_cast<Py_ssize_t>(8 * sizeof(double)) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "rows of eight float64 values expected");
        return nullptr;
    }
    const Py_ssize_t rows = view.len / static_cast<Py_ssize_t>(8 * sizeof(double));
    PyObject* out = PyBytes_FromStringAndSize(nullptr, rows * 2);
    if (out == nullptr) {
        PyBuffer_Release(&view);
        return nullptr;
    }
    char* signs = PyBytes_AS_STRING(out);
    const char* data = static_cast<const char*>(view.buf);
    try {
        for (Py_ssize_t i = 0; i < rows; ++i) {
            double v[8];
            std::memcpy(v, data + i * 8 * sizeof(double), sizeof(v));  // no alignment assumed
            for (double value : v) {
                if (!std::isfinite(value)) throw CapacityError();
            }
            signs[2 * i] = static_cast<char>(orient_sign(v[0], v[1], v[2], v[3], v[4], v[5]));
            signs[2 * i + 1] = static_cast<char>(incircle_sign(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]));
        }
    } catch (const CapacityError&) {
        PyBuffer_Release(&view);
        Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "finite float64 values expected");
        return nullptr;
    }
    PyBuffer_Release(&view);
    return out;
}

}  // namespace anymesher_fhs

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
