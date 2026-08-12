#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <utility>
#include <vector>

#include "_triangulation_native.hpp"

namespace {

long double orient2d_value(
    long double ax,
    long double ay,
    long double bx,
    long double by,
    long double cx,
    long double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

long double incircle_value(
    long double ax,
    long double ay,
    long double bx,
    long double by,
    long double cx,
    long double cy,
    long double dx,
    long double dy) {
    const long double adx = ax - dx;
    const long double ady = ay - dy;
    const long double bdx = bx - dx;
    const long double bdy = by - dy;
    const long double cdx = cx - dx;
    const long double cdy = cy - dy;
    const long double abdet = adx * bdy - bdx * ady;
    const long double bcdet = bdx * cdy - cdx * bdy;
    const long double cadet = cdx * ady - adx * cdy;
    const long double alift = adx * adx + ady * ady;
    const long double blift = bdx * bdx + bdy * bdy;
    const long double clift = cdx * cdx + cdy * cdy;
    return alift * bcdet + blift * cadet + clift * abdet;
}

bool is_float64(const Py_buffer& view) {
    return view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) &&
           view.format != nullptr &&
           (view.format[0] == 'd' ||
            ((view.format[0] == '=' || view.format[0] == '@') &&
             view.format[1] == 'd'));
}

bool is_integer(const Py_buffer& view) {
    if (view.format == nullptr) {
        return false;
    }
    const char prefix = view.format[0];
    if (prefix == '<' || prefix == '>' || prefix == '!') {
        return false;
    }
    const char code = (prefix == '=' || prefix == '@')
                          ? view.format[1]
                          : prefix;
    const bool integer_code =
        code == 'b' || code == 'B' || code == 'h' || code == 'H' ||
        code == 'i' || code == 'I' || code == 'l' || code == 'L' ||
        code == 'q' || code == 'Q';
    return integer_code &&
           (view.itemsize == 1 || view.itemsize == 2 ||
            view.itemsize == 4 || view.itemsize == 8);
}

long long integer_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(view.buf) +
                          row * view.strides[0] + column * view.strides[1];
    const char prefix = view.format[0];
    const char code = (prefix == '=' || prefix == '@')
                          ? view.format[1]
                          : prefix;
    const bool is_unsigned =
        code == 'B' || code == 'H' || code == 'I' ||
        code == 'L' || code == 'Q';
    if (!is_unsigned) {
        switch (view.itemsize) {
            case 1: return *reinterpret_cast<const std::int8_t*>(address);
            case 2: return *reinterpret_cast<const std::int16_t*>(address);
            case 4: return *reinterpret_cast<const std::int32_t*>(address);
            case 8: return *reinterpret_cast<const std::int64_t*>(address);
            default: return -1;
        }
    }
    unsigned long long value = 0;
    switch (view.itemsize) {
        case 1: value = *reinterpret_cast<const std::uint8_t*>(address); break;
        case 2: value = *reinterpret_cast<const std::uint16_t*>(address); break;
        case 4: value = *reinterpret_cast<const std::uint32_t*>(address); break;
        case 8: value = *reinterpret_cast<const std::uint64_t*>(address); break;
        default: return -1;
    }
    if (value > static_cast<unsigned long long>(
                    std::numeric_limits<long long>::max())) {
        return -1;
    }
    return static_cast<long long>(value);
}

double point_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(view.buf) +
                          row * view.strides[0] + column * view.strides[1];
    return *reinterpret_cast<const double*>(address);
}

bool acquire_matrix(
    PyObject* object,
    Py_buffer& view,
    Py_ssize_t columns,
    bool floating,
    const char* name) {
    if (PyObject_GetBuffer(object, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    const bool valid_shape = view.ndim == 2 && view.shape != nullptr &&
                             view.shape[1] == columns && view.strides != nullptr;
    const bool valid_type = floating ? is_float64(view) : is_integer(view);
    if (!valid_shape || !valid_type) {
        PyBuffer_Release(&view);
        PyErr_Format(
            PyExc_TypeError,
            "%s must be a two-dimensional %s buffer with %zd columns",
            name,
            floating ? "float64" : "integer",
            columns);
        return false;
    }
    return true;
}

PyObject* py_orient2d(PyObject*, PyObject* args) {
    double ax, ay, bx, by, cx, cy;
    if (!PyArg_ParseTuple(args, "dddddd:orient2d", &ax, &ay, &bx, &by, &cx, &cy)) {
        return nullptr;
    }
    return PyFloat_FromDouble(static_cast<double>(orient2d_value(ax, ay, bx, by, cx, cy)));
}

PyObject* py_incircle(PyObject*, PyObject* args) {
    double ax, ay, bx, by, cx, cy, dx, dy;
    if (!PyArg_ParseTuple(
            args,
            "dddddddd:incircle",
            &ax,
            &ay,
            &bx,
            &by,
            &cx,
            &cy,
            &dx,
            &dy)) {
        return nullptr;
    }
    return PyFloat_FromDouble(
        static_cast<double>(incircle_value(ax, ay, bx, by, cx, cy, dx, dy)));
}

PyObject* py_orient2d_many(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* triangles_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OO:orient2d_many",
            &points_object,
            &triangles_object)) {
        return nullptr;
    }
    Py_buffer points{};
    Py_buffer triangles{};
    if (!acquire_matrix(points_object, points, 2, true, "points")) {
        return nullptr;
    }
    if (!acquire_matrix(triangles_object, triangles, 3, false, "triangles")) {
        PyBuffer_Release(&points);
        return nullptr;
    }

    const Py_ssize_t count = triangles.shape[0];
    const Py_ssize_t point_count = points.shape[0];
    std::vector<double> values(static_cast<std::size_t>(count));
    bool invalid_index = false;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row = 0; row < count; ++row) {
        const long long ia = integer_at(triangles, row, 0);
        const long long ib = integer_at(triangles, row, 1);
        const long long ic = integer_at(triangles, row, 2);
        if (ia < 0 || ib < 0 || ic < 0 || ia >= point_count || ib >= point_count ||
            ic >= point_count) {
            invalid_index = true;
            continue;
        }
        values[static_cast<std::size_t>(row)] = static_cast<double>(orient2d_value(
            point_at(points, ia, 0),
            point_at(points, ia, 1),
            point_at(points, ib, 0),
            point_at(points, ib, 1),
            point_at(points, ic, 0),
            point_at(points, ic, 1)));
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&triangles);
    PyBuffer_Release(&points);
    if (invalid_index) {
        PyErr_SetString(PyExc_IndexError, "triangle connectivity references a missing point");
        return nullptr;
    }
    PyObject* result = PyList_New(count);
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* value = PyFloat_FromDouble(values[static_cast<std::size_t>(index)]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, index, value);
    }
    return result;
}

PyObject* py_triangle_adjacency(PyObject*, PyObject* object) {
    Py_buffer triangles{};
    if (!acquire_matrix(object, triangles, 3, false, "triangles")) {
        return nullptr;
    }
    using Edge = std::pair<long long, long long>;
    struct Incidence {
        long long left = -1;
        long long right = -1;
        int count = 0;
    };
    std::map<Edge, Incidence> incidence;
    bool invalid_index = false;
    bool nonmanifold = false;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row = 0; row < triangles.shape[0]; ++row) {
        const std::array<long long, 3> tri = {
            integer_at(triangles, row, 0),
            integer_at(triangles, row, 1),
            integer_at(triangles, row, 2),
        };
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0) {
            invalid_index = true;
            continue;
        }
        for (int local = 0; local < 3; ++local) {
            long long first = tri[local];
            long long second = tri[(local + 1) % 3];
            Edge edge = std::minmax(first, second);
            auto& item = incidence[edge];
            if (item.count == 0) {
                item.left = row;
            } else if (item.count == 1) {
                item.right = row;
            } else {
                nonmanifold = true;
            }
            ++item.count;
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&triangles);
    if (invalid_index) {
        PyErr_SetString(PyExc_ValueError, "triangle connectivity contains a negative index");
        return nullptr;
    }
    if (nonmanifold) {
        PyErr_SetString(PyExc_ValueError, "triangle connectivity contains a nonmanifold edge");
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(incidence.size()));
    if (result == nullptr) {
        return nullptr;
    }
    Py_ssize_t output = 0;
    for (const auto& entry : incidence) {
        PyObject* tuple = Py_BuildValue(
            "(LLLL)",
            entry.first.first,
            entry.first.second,
            entry.second.left,
            entry.second.right);
        if (tuple == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, output++, tuple);
    }
    return result;
}

PyMethodDef methods[] = {
    {"orient2d", py_orient2d, METH_VARARGS, "Robust long-double 2D orientation."},
    {"incircle", py_incircle, METH_VARARGS, "Robust long-double in-circle predicate."},
    {"orient2d_many", py_orient2d_many, METH_VARARGS, "Batch triangle orientations with the GIL released."},
    {"triangle_adjacency", py_triangle_adjacency, METH_O, "Deterministic edge incidence for triangle connectivity."},
    {"constrained_triangulate", anymesher_native::py_constrained_triangulate, METH_VARARGS,
     "Deterministic constrained triangulation over canonical prepared PSLG buffers."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Small deterministic C++17 kernels for ANYmesher.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() {
    return PyModule_Create(&module);
}
