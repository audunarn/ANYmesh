#pragma once

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace anymesher_native_v2 {

using Index = std::int64_t;
using Point = std::array<double, 2>;
using Edge = std::pair<Index, Index>;
using Triangle = std::array<Index, 3>;

constexpr std::size_t kSignalCheckInterval = 4096;
constexpr const char* kGeometryLimitedPrefix = "ANYMESHER_NATIVE_V2_GEOMETRY_LIMITED:";
constexpr const char* kPredicateUncertainPrefix = "ANYMESHER_NATIVE_V2_PREDICATE_UNCERTAIN:";

struct GeometryLimited : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PredicateUncertain : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct SignalInterrupted {};
struct PythonOrientationFailure {};

inline bool call_orientation_oracle(
    PyObject* oracle,
    const Point& a,
    const Point& b,
    const Point& c,
    long double& result) {
    PyObject* first = Py_BuildValue("(dd)", a[0], a[1]);
    PyObject* second = Py_BuildValue("(dd)", b[0], b[1]);
    PyObject* third = Py_BuildValue("(dd)", c[0], c[1]);
    if (first == nullptr || second == nullptr || third == nullptr) {
        Py_XDECREF(first);
        Py_XDECREF(second);
        Py_XDECREF(third);
        return false;
    }
    PyObject* value = PyObject_CallFunctionObjArgs(
        oracle, first, second, third, nullptr);
    Py_DECREF(first);
    Py_DECREF(second);
    Py_DECREF(third);
    if (value == nullptr) {
        return false;
    }
    PyObject* numeric = PyNumber_Float(value);
    Py_DECREF(value);
    if (numeric == nullptr) {
        return false;
    }
    const double converted = PyFloat_AsDouble(numeric);
    Py_DECREF(numeric);
    if (PyErr_Occurred()) {
        return false;
    }
    if (!std::isfinite(converted)) {
        PyErr_SetString(
            PyExc_ValueError,
            "local-edge-flip orientation oracle returned a non-finite value");
        return false;
    }
    result = static_cast<long double>(converted);
    return true;
}

class SignalAwareGilRelease {
public:
    SignalAwareGilRelease() : state_(PyEval_SaveThread()) {}
    ~SignalAwareGilRelease() {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
        }
    }
    bool interrupted() {
        PyEval_RestoreThread(state_);
        state_ = nullptr;
        if (PyErr_CheckSignals() != 0) {
            return true;
        }
        state_ = PyEval_SaveThread();
        return false;
    }
    bool adaptive_orientation(
        PyObject* oracle,
        const Point& a,
        const Point& b,
        const Point& c,
        long double& result) {
        PyEval_RestoreThread(state_);
        state_ = nullptr;
        const bool success = call_orientation_oracle(oracle, a, b, c, result);
        state_ = PyEval_SaveThread();
        return success;
    }

private:
    PyThreadState* state_;
};

struct HeldBuffer {
    Py_buffer value{};
    bool held = false;
    ~HeldBuffer() {
        if (held) {
            PyBuffer_Release(&value);
        }
    }
};

inline bool native_format(const char* format, char code) {
    if (format == nullptr) {
        return false;
    }
    return format[0] == code ||
           ((format[0] == '=' || format[0] == '@') && format[1] == code);
}

inline bool acquire_matrix(
    PyObject* object,
    HeldBuffer& buffer,
    Py_ssize_t columns,
    Py_ssize_t itemsize,
    char format,
    const char* label) {
    if (PyObject_GetBuffer(object, &buffer.value, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
        return false;
    }
    buffer.held = true;
    const bool format_matches = format == 'q'
        ? (native_format(buffer.value.format, 'q') || native_format(buffer.value.format, 'l'))
        : native_format(buffer.value.format, format);
    if (buffer.value.ndim != 2 || buffer.value.shape[1] != columns ||
        buffer.value.itemsize != itemsize || !format_matches ||
        !PyBuffer_IsContiguous(&buffer.value, 'C')) {
        PyErr_Format(PyExc_TypeError, "%s has the wrong native contiguous matrix layout", label);
        return false;
    }
    return true;
}

inline double double_at(const Py_buffer& value, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(value.buf) + row * value.strides[0] + column * value.strides[1];
    double result = 0.0;
    std::memcpy(&result, address, sizeof(result));
    return result;
}

inline Index index_at(const Py_buffer& value, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(value.buf) + row * value.strides[0] + column * value.strides[1];
    Index result = 0;
    std::memcpy(&result, address, sizeof(result));
    return result;
}

inline double double_vector_at(const Py_buffer& value, Py_ssize_t row) {
    const char* address = static_cast<const char*>(value.buf) + row * value.strides[0];
    double result = 0.0;
    std::memcpy(&result, address, sizeof(result));
    return result;
}

inline Edge edge(Index first, Index second) {
    return first < second ? Edge{first, second} : Edge{second, first};
}

inline long double orient(const Point& a, const Point& b, const Point& c) {
    const double ax = a[0] - c[0];
    const double ay = a[1] - c[1];
    const double bx = b[0] - c[0];
    const double by = b[1] - c[1];
    const double determinant = ax * by - ay * bx;
    const double error = 8.0 * std::numeric_limits<double>::epsilon() *
                         (std::abs(ax * by) + std::abs(ay * bx));
    if (std::abs(determinant) <= error) {
        throw PredicateUncertain("orientation requires the adaptive Python predicate");
    }
    return static_cast<long double>(determinant);
}

inline long double in_circle(
    const Point& a, const Point& b, const Point& c, const Point& d) {
    const double ax = a[0] - d[0];
    const double ay = a[1] - d[1];
    const double bx = b[0] - d[0];
    const double by = b[1] - d[1];
    const double cx = c[0] - d[0];
    const double cy = c[1] - d[1];
    const double first = (ax * ax + ay * ay) * (bx * cy - by * cx);
    const double second = (bx * bx + by * by) * (ax * cy - ay * cx);
    const double third = (cx * cx + cy * cy) * (ax * by - ay * bx);
    long double result =
        static_cast<long double>(first) - static_cast<long double>(second) +
        static_cast<long double>(third);
    const long double scale =
        static_cast<long double>(std::abs(first)) +
        static_cast<long double>(std::abs(second)) +
        static_cast<long double>(std::abs(third));
    if (std::abs(result) <=
        32.0L * std::numeric_limits<double>::epsilon() * scale) {
        throw PredicateUncertain("in-circle test requires the adaptive Python predicate");
    }
    if (orient(a, b, c) < 0.0L) {
        result = -result;
    }
    return result;
}

inline Triangle canonical(Triangle value, const std::vector<Point>& points) {
    if (orient(points[static_cast<std::size_t>(value[0])],
               points[static_cast<std::size_t>(value[1])],
               points[static_cast<std::size_t>(value[2])]) < 0.0L) {
        std::swap(value[1], value[2]);
    }
    auto iterator = std::min_element(value.begin(), value.end());
    std::rotate(value.begin(), iterator, value.end());
    return value;
}

inline PyObject* py_metric_lengths(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* edges_object = nullptr;
    PyObject* tensors_object = nullptr;
    if (!PyArg_ParseTuple(args, "OOO:native_v2_metric_lengths", &points_object, &edges_object, &tensors_object)) {
        return nullptr;
    }
    HeldBuffer points_buffer;
    HeldBuffer edges_buffer;
    HeldBuffer tensors_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, sizeof(double), 'd', "points") ||
        !acquire_matrix(edges_object, edges_buffer, 2, sizeof(Index), 'q', "edges") ||
        !acquire_matrix(tensors_object, tensors_buffer, 3, sizeof(double), 'd', "tensors")) {
        return nullptr;
    }
    const Py_ssize_t point_count = points_buffer.value.shape[0];
    if (tensors_buffer.value.shape[0] != point_count) {
        PyErr_SetString(PyExc_ValueError, "one compressed metric tensor is required per point");
        return nullptr;
    }
    std::vector<double> values(static_cast<std::size_t>(edges_buffer.value.shape[0]));
    bool invalid = false;
    bool interrupted = false;
    {
      SignalAwareGilRelease gil;
      for (Py_ssize_t row = 0; row < edges_buffer.value.shape[0]; ++row) {
        if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && gil.interrupted()) {
            interrupted = true;
            break;
        }
        const Index first = index_at(edges_buffer.value, row, 0);
        const Index second = index_at(edges_buffer.value, row, 1);
        if (first < 0 || second < 0 || first >= point_count || second >= point_count) {
            invalid = true;
            break;
        }
        const double dx = double_at(points_buffer.value, second, 0) - double_at(points_buffer.value, first, 0);
        const double dy = double_at(points_buffer.value, second, 1) - double_at(points_buffer.value, first, 1);
        const double m00 = 0.5 * (double_at(tensors_buffer.value, first, 0) + double_at(tensors_buffer.value, second, 0));
        const double m01 = 0.5 * (double_at(tensors_buffer.value, first, 1) + double_at(tensors_buffer.value, second, 1));
        const double m11 = 0.5 * (double_at(tensors_buffer.value, first, 2) + double_at(tensors_buffer.value, second, 2));
        values[static_cast<std::size_t>(row)] = std::sqrt(std::max(0.0, dx * dx * m00 + 2.0 * dx * dy * m01 + dy * dy * m11));
      }
    }
    if (interrupted) {
        return nullptr;
    }
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "metric edge index is out of range");
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(values.size()); ++row) {
        PyObject* value = PyFloat_FromDouble(values[static_cast<std::size_t>(row)]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, row, value);
    }
    return result;
}

inline PyObject* py_gradation_limit(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* edges_object = nullptr;
    PyObject* values_object = nullptr;
    double growth = 0.0;
    int max_iterations = 0;
    if (!PyArg_ParseTuple(args, "OOOdi:native_v2_gradation_limit", &points_object, &edges_object, &values_object, &growth, &max_iterations)) {
        return nullptr;
    }
    HeldBuffer points_buffer;
    HeldBuffer edges_buffer;
    HeldBuffer values_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, sizeof(double), 'd', "points") ||
        !acquire_matrix(edges_object, edges_buffer, 2, sizeof(Index), 'q', "edges")) {
        return nullptr;
    }
    if (PyObject_GetBuffer(values_object, &values_buffer.value, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
        return nullptr;
    }
    values_buffer.held = true;
    if (values_buffer.value.ndim != 1 || values_buffer.value.itemsize != sizeof(double) ||
        !native_format(values_buffer.value.format, 'd') ||
        values_buffer.value.shape[0] != points_buffer.value.shape[0] ||
        !PyBuffer_IsContiguous(&values_buffer.value, 'C') || growth <= 1.0 || max_iterations < 1) {
        PyErr_SetString(PyExc_ValueError, "gradation inputs are invalid");
        return nullptr;
    }
    std::vector<double> values(static_cast<std::size_t>(values_buffer.value.shape[0]));
    for (Py_ssize_t row = 0; row < values_buffer.value.shape[0]; ++row) {
        if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
            return nullptr;
        }
        values[static_cast<std::size_t>(row)] = double_vector_at(values_buffer.value, row);
    }
    int iterations = 0;
    bool invalid = false;
    bool interrupted = false;
    {
      SignalAwareGilRelease gil;
      std::size_t work = 0;
      for (iterations = 1; iterations <= max_iterations; ++iterations) {
        bool changed = false;
        for (Py_ssize_t row = 0; row < edges_buffer.value.shape[0]; ++row, ++work) {
            if (work > 0 && work % kSignalCheckInterval == 0 && gil.interrupted()) {
                interrupted = true;
                break;
            }
            const Index first = index_at(edges_buffer.value, row, 0);
            const Index second = index_at(edges_buffer.value, row, 1);
            if (first < 0 || second < 0 || first >= static_cast<Index>(values.size()) || second >= static_cast<Index>(values.size())) {
                invalid = true;
                break;
            }
            const double dx = double_at(points_buffer.value, second, 0) - double_at(points_buffer.value, first, 0);
            const double dy = double_at(points_buffer.value, second, 1) - double_at(points_buffer.value, first, 1);
            const double maximum_delta = (growth - 1.0) * std::hypot(dx, dy);
            if (values[second] > values[first] + maximum_delta) {
                values[second] = values[first] + maximum_delta;
                changed = true;
            }
            if (values[first] > values[second] + maximum_delta) {
                values[first] = values[second] + maximum_delta;
                changed = true;
            }
        }
        if (invalid || interrupted || !changed) {
            break;
        }
      }
    }
    if (interrupted) {
        return nullptr;
    }
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "gradation edge index is out of range");
        return nullptr;
    }
    PyObject* rows = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (rows == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(values.size()); ++row) {
        if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyObject* value = PyFloat_FromDouble(values[static_cast<std::size_t>(row)]);
        if (value == nullptr) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyList_SET_ITEM(rows, row, value);
    }
    return Py_BuildValue("Ni", rows, std::min(iterations, max_iterations));
}

inline PyObject* py_mutable_t3_insert(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* triangles_object = nullptr;
    PyObject* protected_object = nullptr;
    PyObject* orientation_oracle = nullptr;
    double candidate_x = 0.0;
    double candidate_y = 0.0;
    if (!PyArg_ParseTuple(args, "OOOOdd:native_v2_mutable_t3_insert", &points_object, &triangles_object, &protected_object, &orientation_oracle, &candidate_x, &candidate_y)) {
        return nullptr;
    }
    if (!PyCallable_Check(orientation_oracle)) {
        PyErr_SetString(PyExc_TypeError, "native_v2_mutable_t3_insert requires its internal orientation oracle");
        return nullptr;
    }
    HeldBuffer points_buffer;
    HeldBuffer triangles_buffer;
    HeldBuffer protected_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, sizeof(double), 'd', "points") ||
        !acquire_matrix(triangles_object, triangles_buffer, 3, sizeof(Index), 'q', "triangles") ||
        !acquire_matrix(protected_object, protected_buffer, 2, sizeof(Index), 'q', "protected_edges")) {
        return nullptr;
    }
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    std::set<Edge> protected_edges;
    try {
        points.reserve(static_cast<std::size_t>(points_buffer.value.shape[0] + 1));
        for (Py_ssize_t row = 0; row < points_buffer.value.shape[0]; ++row) {
            if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
                return nullptr;
            }
            points.push_back({double_at(points_buffer.value, row, 0), double_at(points_buffer.value, row, 1)});
        }
        for (Py_ssize_t row = 0; row < triangles_buffer.value.shape[0]; ++row) {
            if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
                return nullptr;
            }
            Triangle value{index_at(triangles_buffer.value, row, 0), index_at(triangles_buffer.value, row, 1), index_at(triangles_buffer.value, row, 2)};
            if (*std::min_element(value.begin(), value.end()) < 0 || *std::max_element(value.begin(), value.end()) >= static_cast<Index>(points.size())) {
                throw std::runtime_error("mutable T3 connectivity is out of range");
            }
            triangles.push_back(canonical(value, points));
        }
        for (Py_ssize_t row = 0; row < protected_buffer.value.shape[0]; ++row) {
            if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
                return nullptr;
            }
            const Index first = index_at(protected_buffer.value, row, 0);
            const Index second = index_at(protected_buffer.value, row, 1);
            if (first < 0 || second < 0 || first >= static_cast<Index>(points.size()) || second >= static_cast<Index>(points.size()) || first == second) {
                throw std::runtime_error("protected edge is invalid");
            }
            protected_edges.insert(edge(first, second));
        }
    } catch (const PredicateUncertain& error) {
        PyErr_Format(PyExc_RuntimeError, "%s%s", kPredicateUncertainPrefix, error.what());
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    const Point candidate{candidate_x, candidate_y};
    std::vector<Triangle> result;
    std::size_t removed_count = 0;
    std::size_t added_count = 0;
    std::string error;
    bool interrupted = false;
    try {
      SignalAwareGilRelease gil;
        double scale = 1.0;
        double minimum_x = points.empty() ? 0.0 : points.front()[0];
        double maximum_x = minimum_x;
        double minimum_y = points.empty() ? 0.0 : points.front()[1];
        double maximum_y = minimum_y;
        std::size_t work = 0;
        for (const Point& point : points) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            minimum_x = std::min(minimum_x, point[0]);
            maximum_x = std::max(maximum_x, point[0]);
            minimum_y = std::min(minimum_y, point[1]);
            maximum_y = std::max(maximum_y, point[1]);
        }
        scale = std::max({maximum_x - minimum_x, maximum_y - minimum_y, 1.0});
        for (const Point& point : points) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const double dx = point[0] - candidate[0];
            const double dy = point[1] - candidate[1];
            if (std::sqrt(dx * dx + dy * dy) <=
                64.0 * std::numeric_limits<double>::epsilon() * scale) {
                throw GeometryLimited("frontal candidate duplicates an existing point");
            }
        }
        const long double tolerance = 64.0L * std::numeric_limits<double>::epsilon() * scale;
        for (const Edge& value : protected_edges) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const Point& first = points[static_cast<std::size_t>(value.first)];
            const Point& second = points[static_cast<std::size_t>(value.second)];
            if (std::abs(orient(first, second, candidate)) <= tolerance * std::max(1.0, std::hypot(second[0] - first[0], second[1] - first[1])) &&
                candidate[0] >= std::min(first[0], second[0]) - tolerance && candidate[0] <= std::max(first[0], second[0]) + tolerance &&
                candidate[1] >= std::min(first[1], second[1]) - tolerance && candidate[1] <= std::max(first[1], second[1]) + tolerance) {
                throw GeometryLimited("frontal candidate encroaches an unsplittable protected edge");
            }
        }
        std::vector<std::size_t> bad_triangles;
        for (std::size_t row = 0; row < triangles.size(); ++row) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const Triangle& triangle = triangles[row];
            if (in_circle(points[triangle[0]], points[triangle[1]], points[triangle[2]], candidate) > tolerance * tolerance) {
                bad_triangles.push_back(row);
            }
        }
        const auto containment_orient = [&](const Point& first, const Point& second, const Point& value) {
            try {
                return orient(first, second, value);
            } catch (const PredicateUncertain&) {
                long double result = 0.0L;
                if (!gil.adaptive_orientation(orientation_oracle, first, second, value, result)) {
                    throw PythonOrientationFailure{};
                }
                return result;
            }
        };
        std::size_t seed = triangles.size();
        for (const std::size_t row : bad_triangles) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const Triangle& triangle = triangles[row];
            if (containment_orient(points[triangle[0]], points[triangle[1]], candidate) >= -tolerance &&
                containment_orient(points[triangle[1]], points[triangle[2]], candidate) >= -tolerance &&
                containment_orient(points[triangle[2]], points[triangle[0]], candidate) >= -tolerance) {
                seed = row;
                break;
            }
        }
        std::vector<std::size_t> cavity;
        if (seed == triangles.size()) {
            for (std::size_t row = 0; row < triangles.size(); ++row) {
                if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                    throw SignalInterrupted{};
                }
                const Triangle& triangle = triangles[row];
                if (containment_orient(points[triangle[0]], points[triangle[1]], candidate) >= -tolerance &&
                    containment_orient(points[triangle[1]], points[triangle[2]], candidate) >= -tolerance &&
                    containment_orient(points[triangle[2]], points[triangle[0]], candidate) >= -tolerance) {
                    cavity.push_back(row);
                    break;
                }
            }
        } else {
            std::map<Edge, std::vector<std::size_t>> edge_rows;
            for (const std::size_t row : bad_triangles) {
                if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                    throw SignalInterrupted{};
                }
                const Triangle& triangle = triangles[row];
                for (int index = 0; index < 3; ++index) {
                    edge_rows[edge(triangle[index], triangle[(index + 1) % 3])].push_back(row);
                }
            }
            std::set<std::size_t> selected{seed};
            std::vector<std::size_t> frontier{seed};
            for (std::size_t cursor = 0; cursor < frontier.size(); ++cursor) {
                if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                    throw SignalInterrupted{};
                }
                const Triangle& triangle = triangles[frontier[cursor]];
                std::set<std::size_t> adjacent;
                for (int index = 0; index < 3; ++index) {
                    const auto& rows = edge_rows[edge(triangle[index], triangle[(index + 1) % 3])];
                    adjacent.insert(rows.begin(), rows.end());
                }
                for (const std::size_t row : adjacent) {
                    if (selected.insert(row).second) frontier.push_back(row);
                }
            }
            cavity.assign(selected.begin(), selected.end());
        }
        if (cavity.empty()) {
                throw GeometryLimited("frontal candidate lies outside the mutable triangulation");
        }
        std::set<std::size_t> removed(cavity.begin(), cavity.end());
        std::map<Edge, int> counts;
        for (std::size_t row : cavity) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const Triangle& triangle = triangles[row];
            for (int index = 0; index < 3; ++index) {
                ++counts[edge(triangle[index], triangle[(index + 1) % 3])];
            }
        }
        for (const auto& item : counts) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            if (item.second == 2 && protected_edges.count(item.first) != 0) {
                throw GeometryLimited("frontal cavity would remove an unsplittable protected edge");
            }
        }
        points.push_back(candidate);
        const Index inserted = static_cast<Index>(points.size() - 1);
        for (std::size_t row = 0; row < triangles.size(); ++row) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            if (removed.count(row) == 0) {
                result.push_back(triangles[row]);
            }
        }
        for (const auto& item : counts) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            if (item.second == 1) {
                Triangle value = canonical({item.first.first, item.first.second, inserted}, points);
                if (orient(points[value[0]], points[value[1]], points[value[2]]) <= 0.0L) {
                    throw GeometryLimited("frontal insertion creates a non-positive triangle");
                }
                result.push_back(value);
                ++added_count;
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        removed_count = cavity.size();
    } catch (const SignalInterrupted&) {
        interrupted = true;
    } catch (const PythonOrientationFailure&) {
        return nullptr;
    } catch (const PredicateUncertain& caught) {
        error = std::string(kPredicateUncertainPrefix) + caught.what();
    } catch (const GeometryLimited& caught) {
        error = std::string(kGeometryLimitedPrefix) + caught.what();
    } catch (const std::exception& caught) {
        error = caught.what();
    }
    if (interrupted) {
        return nullptr;
    }
    if (!error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    PyObject* rows = PyList_New(static_cast<Py_ssize_t>(result.size()));
    if (rows == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(result.size()); ++row) {
        if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
            Py_DECREF(rows);
            return nullptr;
        }
        const Triangle& value = result[static_cast<std::size_t>(row)];
        PyObject* tuple = Py_BuildValue("(LLL)", static_cast<long long>(value[0]), static_cast<long long>(value[1]), static_cast<long long>(value[2]));
        if (tuple == nullptr) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyList_SET_ITEM(rows, row, tuple);
    }
    PyObject* diagnostics = Py_BuildValue(
        "{s:K,s:K,s:O}",
        "removed_triangles", static_cast<unsigned long long>(removed_count),
        "added_triangles", static_cast<unsigned long long>(added_count),
        "native", Py_True);
    if (diagnostics == nullptr) {
        Py_DECREF(rows);
        return nullptr;
    }
    return Py_BuildValue("NN", rows, diagnostics);
}

using Metric2 = std::array<double, 4>;

inline Edge flip_edge(Index first, Index second) {
    return first < second ? Edge{first, second} : Edge{second, first};
}

inline std::array<Edge, 3> flip_triangle_edges(const Triangle& value) {
    return {
        flip_edge(value[0], value[1]),
        flip_edge(value[1], value[2]),
        flip_edge(value[2], value[0]),
    };
}

inline double flip_mean_ratio(
    const std::vector<Point>& points,
    const std::vector<Metric2>& metrics,
    const Triangle& triangle,
    long double doubled_area) {
    const auto& p0 = points[static_cast<std::size_t>(triangle[0])];
    const auto& p1 = points[static_cast<std::size_t>(triangle[1])];
    const auto& p2 = points[static_cast<std::size_t>(triangle[2])];
    double m00 = 0.0;
    double m01 = 0.0;
    double m11 = 0.0;
    for (const Index node : triangle) {
        const auto& metric = metrics[static_cast<std::size_t>(node)];
        m00 += metric[0] / 3.0;
        m01 += (metric[1] + metric[2]) / 6.0;
        m11 += metric[3] / 3.0;
    }
    const double determinant = m00 * m11 - m01 * m01;
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        throw std::runtime_error("local-edge-flip metric is not positive definite");
    }
    const std::array<Point, 3> vectors{{
        Point{p1[0] - p0[0], p1[1] - p0[1]},
        Point{p2[0] - p1[0], p2[1] - p1[1]},
        Point{p0[0] - p2[0], p0[1] - p2[1]},
    }};
    double squared_length_sum = 0.0;
    for (const Point& vector : vectors) {
        squared_length_sum +=
            m00 * vector[0] * vector[0]
            + 2.0 * m01 * vector[0] * vector[1]
            + m11 * vector[1] * vector[1];
    }
    if (!(squared_length_sum > 0.0) || !std::isfinite(squared_length_sum)) {
        return 0.0;
    }
    return static_cast<double>(
        2.0L * std::sqrt(3.0L) * doubled_area
        * std::sqrt(static_cast<long double>(determinant))
        / static_cast<long double>(squared_length_sum));
}

inline PyObject* py_local_edge_flip(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* triangles_object = nullptr;
    PyObject* protected_object = nullptr;
    PyObject* metrics_object = nullptr;
    PyObject* orientation_oracle = nullptr;
    long long flip_limit = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOL:native_v2_local_edge_flip",
            &points_object, &triangles_object, &protected_object,
            &metrics_object, &orientation_oracle, &flip_limit)) {
        return nullptr;
    }
    if (!PyCallable_Check(orientation_oracle)) {
        PyErr_SetString(
            PyExc_TypeError,
            "native_v2_local_edge_flip requires its internal orientation oracle");
        return nullptr;
    }
    if (flip_limit < 0) {
        PyErr_SetString(PyExc_ValueError, "flip_limit must be non-negative");
        return nullptr;
    }
    HeldBuffer points_buffer;
    HeldBuffer triangles_buffer;
    HeldBuffer protected_buffer;
    HeldBuffer metrics_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, sizeof(double), 'd', "points") ||
        !acquire_matrix(triangles_object, triangles_buffer, 3, sizeof(Index), 'q', "triangles") ||
        !acquire_matrix(protected_object, protected_buffer, 2, sizeof(Index), 'q', "protected_edges") ||
        !acquire_matrix(metrics_object, metrics_buffer, 4, sizeof(double), 'd', "metrics")) {
        return nullptr;
    }
    if (metrics_buffer.value.shape[0] != points_buffer.value.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "one 2x2 metric is required per point");
        return nullptr;
    }

    const auto* point_rows = static_cast<const double*>(points_buffer.value.buf);
    const auto* triangle_rows = static_cast<const Index*>(triangles_buffer.value.buf);
    const auto* protected_rows = static_cast<const Index*>(protected_buffer.value.buf);
    const auto* metric_rows = static_cast<const double*>(metrics_buffer.value.buf);
    const std::size_t point_count = static_cast<std::size_t>(points_buffer.value.shape[0]);
    const std::size_t triangle_count = static_cast<std::size_t>(triangles_buffer.value.shape[0]);
    const std::size_t protected_count = static_cast<std::size_t>(protected_buffer.value.shape[0]);
    std::vector<Point> points(point_count);
    std::vector<Triangle> triangles(triangle_count);
    std::vector<Metric2> metrics(point_count);
    std::set<Edge> protected_edges;
    try {
        for (std::size_t row = 0; row < point_count; ++row) {
            if (row > 0 && row % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return nullptr;
            points[row] = Point{point_rows[2 * row], point_rows[2 * row + 1]};
            metrics[row] = Metric2{
                metric_rows[4 * row], metric_rows[4 * row + 1],
                metric_rows[4 * row + 2], metric_rows[4 * row + 3]};
        }
        for (std::size_t row = 0; row < triangle_count; ++row) {
            if (row > 0 && row % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return nullptr;
            Triangle triangle{
                triangle_rows[3 * row], triangle_rows[3 * row + 1], triangle_rows[3 * row + 2]};
            std::set<Index> unique(triangle.begin(), triangle.end());
            for (const Index node : triangle) {
                if (node < 0 || static_cast<std::size_t>(node) >= point_count) {
                    throw std::runtime_error("local-edge-flip triangle index is out of range");
                }
            }
            if (unique.size() != 3) throw std::runtime_error("local-edge-flip triangle repeats a node");
            const auto& first = points[static_cast<std::size_t>(triangle[0])];
            const auto& second = points[static_cast<std::size_t>(triangle[1])];
            const auto& third = points[static_cast<std::size_t>(triangle[2])];
            long double orientation = 0.0L;
            try {
                orientation = orient(first, second, third);
            } catch (const PredicateUncertain&) {
                if (!call_orientation_oracle(
                        orientation_oracle, first, second, third, orientation)) {
                    throw PythonOrientationFailure{};
                }
            }
            if (orientation == 0.0L) throw std::runtime_error("local-edge-flip triangle has zero area");
            if (orientation < 0.0L) std::swap(triangle[1], triangle[2]);
            triangles[row] = triangle;
        }
        for (std::size_t row = 0; row < protected_count; ++row) {
            const Index first = protected_rows[2 * row];
            const Index second = protected_rows[2 * row + 1];
            if (first < 0 || second < 0 || first == second ||
                static_cast<std::size_t>(first) >= point_count ||
                static_cast<std::size_t>(second) >= point_count) {
                throw std::runtime_error("local-edge-flip protected edge is invalid");
            }
            protected_edges.insert(flip_edge(first, second));
        }
    } catch (const PythonOrientationFailure&) {
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }

    std::size_t flip_count = 0;
    std::size_t queue_visits = 0;
    bool converged = true;
    bool interrupted = false;
    try {
        SignalAwareGilRelease gil;
        const auto adaptive_orient = [&](const Point& a, const Point& b, const Point& c) {
            try {
                return orient(a, b, c);
            } catch (const PredicateUncertain&) {
                long double result = 0.0L;
                if (!gil.adaptive_orientation(
                        orientation_oracle, a, b, c, result)) {
                    throw PythonOrientationFailure{};
                }
                return result;
            }
        };
        const auto triangle_quality = [&](const Triangle& triangle) {
            const auto& first = points[static_cast<std::size_t>(triangle[0])];
            const auto& second = points[static_cast<std::size_t>(triangle[1])];
            const auto& third = points[static_cast<std::size_t>(triangle[2])];
            return flip_mean_ratio(
                points,
                metrics,
                triangle,
                std::fabs(adaptive_orient(first, second, third)));
        };
        std::map<Edge, std::set<std::size_t>> incidence;
        for (std::size_t row = 0; row < triangles.size(); ++row) {
            for (const Edge& edge : flip_triangle_edges(triangles[row])) incidence[edge].insert(row);
            if ((row + 1) % kSignalCheckInterval == 0 && gil.interrupted()) {
                interrupted = true;
                break;
            }
        }
        std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> queue;
        std::set<Edge> queued;
        const auto enqueue = [&](const Edge& edge) {
            const auto found = incidence.find(edge);
            if (protected_edges.find(edge) == protected_edges.end() &&
                found != incidence.end() && found->second.size() == 2 &&
                queued.insert(edge).second) {
                queue.push(edge);
            }
        };
        if (!interrupted) for (const auto& item : incidence) enqueue(item.first);
        while (!interrupted && !queue.empty()) {
            if (flip_count >= static_cast<std::size_t>(flip_limit)) {
                converged = false;
                break;
            }
            const Edge edge = queue.top();
            queue.pop();
            queued.erase(edge);
            ++queue_visits;
            if (queue_visits % kSignalCheckInterval == 0 && gil.interrupted()) {
                interrupted = true;
                break;
            }
            const auto found = incidence.find(edge);
            if (protected_edges.find(edge) != protected_edges.end() ||
                found == incidence.end() || found->second.size() != 2) continue;
            auto attached_iterator = found->second.begin();
            const std::size_t first_row = *attached_iterator++;
            const std::size_t second_row = *attached_iterator;
            const Triangle first = triangles[first_row];
            const Triangle second = triangles[second_row];
            Index c = -1;
            Index d = -1;
            for (const Index node : first) if (node != edge.first && node != edge.second) c = node;
            for (const Index node : second) if (node != edge.first && node != edge.second) d = node;
            if (c < 0 || d < 0 || c == d) continue;
            const auto& a_point = points[static_cast<std::size_t>(edge.first)];
            const auto& b_point = points[static_cast<std::size_t>(edge.second)];
            const auto& c_point = points[static_cast<std::size_t>(c)];
            const auto& d_point = points[static_cast<std::size_t>(d)];
            if (adaptive_orient(a_point, b_point, c_point) *
                    adaptive_orient(a_point, b_point, d_point) >= 0.0L ||
                adaptive_orient(c_point, d_point, a_point) *
                    adaptive_orient(c_point, d_point, b_point) >= 0.0L) continue;
            const Edge replacement_edge = flip_edge(c, d);
            const auto replacement_found = incidence.find(replacement_edge);
            if (replacement_found != incidence.end()) {
                bool outside_attached = false;
                for (const std::size_t row : replacement_found->second) {
                    if (row != first_row && row != second_row) outside_attached = true;
                }
                if (outside_attached) continue;
            }
            Triangle first_candidate{c, d, edge.first};
            Triangle second_candidate{d, c, edge.second};
            if (adaptive_orient(points[first_candidate[0]], points[first_candidate[1]], points[first_candidate[2]]) < 0.0L) {
                std::swap(first_candidate[1], first_candidate[2]);
            }
            if (adaptive_orient(points[second_candidate[0]], points[second_candidate[1]], points[second_candidate[2]]) < 0.0L) {
                std::swap(second_candidate[1], second_candidate[2]);
            }
            std::array<Triangle, 2> candidates{first_candidate, second_candidate};
            std::sort(candidates.begin(), candidates.end());
            const double current_quality = std::min(
                triangle_quality(first), triangle_quality(second));
            const double candidate_quality = std::min(
                triangle_quality(candidates[0]), triangle_quality(candidates[1]));
            if (candidate_quality <= current_quality) continue;

            std::set<Edge> old_edges;
            for (const Edge& old_edge : flip_triangle_edges(first)) old_edges.insert(old_edge);
            for (const Edge& old_edge : flip_triangle_edges(second)) old_edges.insert(old_edge);
            for (const Edge& old_edge : old_edges) {
                auto& rows = incidence[old_edge];
                rows.erase(first_row);
                rows.erase(second_row);
                if (rows.empty()) incidence.erase(old_edge);
            }
            triangles[first_row] = candidates[0];
            triangles[second_row] = candidates[1];
            std::set<Edge> new_edges;
            for (const Edge& new_edge : flip_triangle_edges(candidates[0])) new_edges.insert(new_edge);
            for (const Edge& new_edge : flip_triangle_edges(candidates[1])) new_edges.insert(new_edge);
            for (const Edge& new_edge : new_edges) {
                auto& rows = incidence[new_edge];
                const auto first_edges = flip_triangle_edges(candidates[0]);
                const auto second_edges = flip_triangle_edges(candidates[1]);
                if (std::find(first_edges.begin(), first_edges.end(), new_edge) != first_edges.end()) rows.insert(first_row);
                if (std::find(second_edges.begin(), second_edges.end(), new_edge) != second_edges.end()) rows.insert(second_row);
            }
            ++flip_count;
            old_edges.insert(new_edges.begin(), new_edges.end());
            for (const Edge& affected : old_edges) enqueue(affected);
        }
    } catch (const PythonOrientationFailure&) {
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    if (interrupted) return nullptr;

    PyObject* rows = PyList_New(static_cast<Py_ssize_t>(triangles.size()));
    if (rows == nullptr) return nullptr;
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(triangles.size()); ++row) {
        if (row > 0 && static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) {
            Py_DECREF(rows);
            return nullptr;
        }
        const Triangle& triangle = triangles[static_cast<std::size_t>(row)];
        PyObject* tuple = Py_BuildValue("(LLL)", triangle[0], triangle[1], triangle[2]);
        if (tuple == nullptr) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyList_SET_ITEM(rows, row, tuple);
    }
    PyObject* diagnostics = Py_BuildValue(
        "{s:K,s:K,s:O}",
        "flip_count", static_cast<unsigned long long>(flip_count),
        "queue_visits", static_cast<unsigned long long>(queue_visits),
        "converged", converged ? Py_True : Py_False);
    if (diagnostics == nullptr) {
        Py_DECREF(rows);
        return nullptr;
    }
    return Py_BuildValue("NN", rows, diagnostics);
}

struct SmoothCell {
    std::array<Index, 4> nodes{};
    int size = 0;
    int sign = 0;
};

inline PyObject* py_constrained_smoothing(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* cells_object = nullptr;
    PyObject* fixed_object = nullptr;
    PyObject* constraints_object = nullptr;
    PyObject* metrics_object = nullptr;
    PyObject* orientation_oracle = nullptr;
    int preserve_boundary = 0;
    int iteration_limit = 0;
    double relaxation = 0.0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOiid:native_v2_constrained_smoothing",
            &points_object, &cells_object, &fixed_object, &constraints_object,
            &metrics_object, &orientation_oracle, &preserve_boundary,
            &iteration_limit, &relaxation)) {
        return nullptr;
    }
    if (!PyCallable_Check(orientation_oracle)) {
        PyErr_SetString(
            PyExc_TypeError,
            "native_v2_constrained_smoothing requires its internal orientation oracle");
        return nullptr;
    }
    if (iteration_limit < 0 || !std::isfinite(relaxation) ||
        !(relaxation > 0.0 && relaxation <= 1.0)) {
        PyErr_SetString(PyExc_ValueError, "native constrained-smoothing controls are invalid");
        return nullptr;
    }
    HeldBuffer points_buffer;
    HeldBuffer cells_buffer;
    HeldBuffer fixed_buffer;
    HeldBuffer constraints_buffer;
    HeldBuffer metrics_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, sizeof(double), 'd', "points") ||
        !acquire_matrix(cells_object, cells_buffer, 4, sizeof(Index), 'q', "cells") ||
        !acquire_matrix(fixed_object, fixed_buffer, 1, sizeof(Index), 'q', "fixed_nodes") ||
        !acquire_matrix(constraints_object, constraints_buffer, 2, sizeof(Index), 'q', "constrained_edges") ||
        !acquire_matrix(metrics_object, metrics_buffer, 4, sizeof(double), 'd', "metrics")) {
        return nullptr;
    }
    if (metrics_buffer.value.shape[0] != points_buffer.value.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "one 2x2 smoothing metric is required per point");
        return nullptr;
    }

    const auto* point_rows = static_cast<const double*>(points_buffer.value.buf);
    const auto* cell_rows = static_cast<const Index*>(cells_buffer.value.buf);
    const auto* fixed_rows = static_cast<const Index*>(fixed_buffer.value.buf);
    const auto* constraint_rows = static_cast<const Index*>(constraints_buffer.value.buf);
    const auto* metric_rows = static_cast<const double*>(metrics_buffer.value.buf);
    const std::size_t point_count = static_cast<std::size_t>(points_buffer.value.shape[0]);
    const std::size_t cell_count = static_cast<std::size_t>(cells_buffer.value.shape[0]);
    std::vector<Point> points(point_count);
    std::vector<Metric2> metrics(point_count);
    std::vector<SmoothCell> cells(cell_count);
    std::vector<std::set<Index>> neighbors(point_count);
    std::vector<std::set<std::size_t>> incident_cells(point_count);
    std::map<Edge, std::size_t> incidence;
    std::set<Index> fixed;
    try {
        for (std::size_t row = 0; row < point_count; ++row) {
            if (row > 0 && row % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return nullptr;
            points[row] = Point{point_rows[2 * row], point_rows[2 * row + 1]};
            metrics[row] = Metric2{
                metric_rows[4 * row], metric_rows[4 * row + 1],
                metric_rows[4 * row + 2], metric_rows[4 * row + 3]};
            const double m00 = metrics[row][0];
            const double m01 = 0.5 * (metrics[row][1] + metrics[row][2]);
            const double m11 = metrics[row][3];
            if (!std::isfinite(m00) || !std::isfinite(m01) || !std::isfinite(m11) ||
                !(m00 > 0.0) || !(m00 * m11 - m01 * m01 > 0.0)) {
                throw std::runtime_error("smoothing metric is not positive definite");
            }
        }
        const auto held_orient = [&](const Point& a, const Point& b, const Point& c) {
            try {
                return orient(a, b, c);
            } catch (const PredicateUncertain&) {
                long double result = 0.0L;
                if (!call_orientation_oracle(orientation_oracle, a, b, c, result)) {
                    throw PythonOrientationFailure{};
                }
                return result;
            }
        };
        for (std::size_t row = 0; row < cell_count; ++row) {
            if (row > 0 && row % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return nullptr;
            SmoothCell cell;
            cell.size = cell_rows[4 * row + 3] == -1 ? 3 : 4;
            std::set<Index> unique;
            for (int column = 0; column < cell.size; ++column) {
                const Index node = cell_rows[4 * row + static_cast<std::size_t>(column)];
                if (node < 0 || static_cast<std::size_t>(node) >= point_count) {
                    throw std::runtime_error("smoothing cell index is out of range");
                }
                cell.nodes[static_cast<std::size_t>(column)] = node;
                unique.insert(node);
                incident_cells[static_cast<std::size_t>(node)].insert(row);
            }
            if (unique.size() != static_cast<std::size_t>(cell.size)) {
                throw std::runtime_error("smoothing cell repeats a node");
            }
            bool positive = true;
            bool negative = true;
            for (int column = 0; column < cell.size; ++column) {
                const Index previous = cell.nodes[static_cast<std::size_t>((column + cell.size - 1) % cell.size)];
                const Index current = cell.nodes[static_cast<std::size_t>(column)];
                const Index next = cell.nodes[static_cast<std::size_t>((column + 1) % cell.size)];
                const long double turn = held_orient(points[previous], points[current], points[next]);
                positive = positive && turn > 0.0L;
                negative = negative && turn < 0.0L;
                const Edge made_edge = flip_edge(current, next);
                incidence[made_edge] += 1;
                neighbors[static_cast<std::size_t>(current)].insert(next);
                neighbors[static_cast<std::size_t>(next)].insert(current);
            }
            if (!positive && !negative) {
                throw std::runtime_error("smoothing cell is degenerate, concave, or self-intersecting");
            }
            cell.sign = positive ? 1 : -1;
            cells[row] = cell;
        }
        for (Py_ssize_t row = 0; row < fixed_buffer.value.shape[0]; ++row) {
            const Index node = fixed_rows[row];
            if (node < 0 || static_cast<std::size_t>(node) >= point_count) {
                throw std::runtime_error("fixed smoothing node is out of range");
            }
            fixed.insert(node);
        }
        for (Py_ssize_t row = 0; row < constraints_buffer.value.shape[0]; ++row) {
            const Index first = constraint_rows[2 * row];
            const Index second = constraint_rows[2 * row + 1];
            if (first < 0 || second < 0 || first == second ||
                static_cast<std::size_t>(first) >= point_count ||
                static_cast<std::size_t>(second) >= point_count) {
                throw std::runtime_error("constrained smoothing edge is invalid");
            }
            fixed.insert(first);
            fixed.insert(second);
        }
        if (preserve_boundary != 0) {
            for (const auto& item : incidence) {
                if (item.second == 1) {
                    fixed.insert(item.first.first);
                    fixed.insert(item.first.second);
                }
            }
        }
    } catch (const PythonOrientationFailure&) {
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }

    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::set<Index> moved;
    int iterations_run = 0;
    bool converged = cells.empty() || iteration_limit == 0;
    bool interrupted = false;
    try {
        SignalAwareGilRelease gil;
        const auto adaptive_orient = [&](const Point& a, const Point& b, const Point& c) {
            try {
                return orient(a, b, c);
            } catch (const PredicateUncertain&) {
                long double result = 0.0L;
                if (!gil.adaptive_orientation(orientation_oracle, a, b, c, result)) {
                    throw PythonOrientationFailure{};
                }
                return result;
            }
        };
        std::size_t work = 0;
        for (int iteration = 0; iteration < iteration_limit; ++iteration) {
            std::size_t accepted_this_iteration = 0;
            for (std::size_t node = 0; node < point_count; ++node) {
                if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                    throw SignalInterrupted{};
                }
                if (fixed.find(static_cast<Index>(node)) != fixed.end() || neighbors[node].empty()) continue;
                double system00 = 0.0;
                double system01 = 0.0;
                double system10 = 0.0;
                double system11 = 0.0;
                double right0 = 0.0;
                double right1 = 0.0;
                for (const Index other_index : neighbors[node]) {
                    if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                        throw SignalInterrupted{};
                    }
                    const std::size_t other = static_cast<std::size_t>(other_index);
                    const double m00 = 0.5 * (metrics[node][0] + metrics[other][0]);
                    const double m01 = 0.5 * (metrics[node][1] + metrics[other][1]);
                    const double m10 = 0.5 * (metrics[node][2] + metrics[other][2]);
                    const double m11 = 0.5 * (metrics[node][3] + metrics[other][3]);
                    system00 += m00;
                    system01 += m01;
                    system10 += m10;
                    system11 += m11;
                    right0 += m00 * points[other][0] + m01 * points[other][1];
                    right1 += m10 * points[other][0] + m11 * points[other][1];
                }
                const double determinant = system00 * system11 - system01 * system10;
                Point target{};
                if (determinant == 0.0 || !std::isfinite(determinant)) {
                    for (const Index other : neighbors[node]) {
                        target[0] += points[static_cast<std::size_t>(other)][0];
                        target[1] += points[static_cast<std::size_t>(other)][1];
                    }
                    target[0] /= static_cast<double>(neighbors[node].size());
                    target[1] /= static_cast<double>(neighbors[node].size());
                } else {
                    target[0] = (right0 * system11 - system01 * right1) / determinant;
                    target[1] = (system00 * right1 - right0 * system10) / determinant;
                }
                const Point previous = points[node];
                const Point candidate{
                    previous[0] + relaxation * (target[0] - previous[0]),
                    previous[1] + relaxation * (target[1] - previous[1])};
                if (candidate == previous) continue;
                points[node] = candidate;
                bool valid = true;
                for (const std::size_t cell_number : incident_cells[node]) {
                    const SmoothCell& cell = cells[cell_number];
                    for (int column = 0; column < cell.size; ++column) {
                        if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                            throw SignalInterrupted{};
                        }
                        const Index prior = cell.nodes[static_cast<std::size_t>((column + cell.size - 1) % cell.size)];
                        const Index current = cell.nodes[static_cast<std::size_t>(column)];
                        const Index next = cell.nodes[static_cast<std::size_t>((column + 1) % cell.size)];
                        if (static_cast<long double>(cell.sign) *
                                adaptive_orient(points[prior], points[current], points[next]) <= 0.0L) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) break;
                }
                if (!valid) {
                    points[node] = previous;
                    ++rejected;
                    continue;
                }
                ++accepted;
                ++accepted_this_iteration;
                moved.insert(static_cast<Index>(node));
            }
            iterations_run = iteration + 1;
            if (accepted_this_iteration == 0) {
                converged = true;
                break;
            }
        }
    } catch (const SignalInterrupted&) {
        interrupted = true;
    } catch (const PythonOrientationFailure&) {
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    if (interrupted) return nullptr;

    PyObject* point_result = PyList_New(static_cast<Py_ssize_t>(points.size()));
    if (point_result == nullptr) return nullptr;
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(points.size()); ++row) {
        PyObject* value = Py_BuildValue("(dd)", points[static_cast<std::size_t>(row)][0], points[static_cast<std::size_t>(row)][1]);
        if (value == nullptr) {
            Py_DECREF(point_result);
            return nullptr;
        }
        PyList_SET_ITEM(point_result, row, value);
    }
    PyObject* moved_result = PyList_New(static_cast<Py_ssize_t>(moved.size()));
    if (moved_result == nullptr) {
        Py_DECREF(point_result);
        return nullptr;
    }
    Py_ssize_t moved_row = 0;
    for (const Index node : moved) {
        PyObject* value = PyLong_FromLongLong(node);
        if (value == nullptr) {
            Py_DECREF(point_result);
            Py_DECREF(moved_result);
            return nullptr;
        }
        PyList_SET_ITEM(moved_result, moved_row++, value);
    }
    PyObject* diagnostics = Py_BuildValue(
        "{s:i,s:K,s:K,s:O}",
        "iterations", iterations_run,
        "accepted_moves", static_cast<unsigned long long>(accepted),
        "rejected_moves", static_cast<unsigned long long>(rejected),
        "converged", converged ? Py_True : Py_False);
    if (diagnostics == nullptr) {
        Py_DECREF(point_result);
        Py_DECREF(moved_result);
        return nullptr;
    }
    return Py_BuildValue("NNN", point_result, moved_result, diagnostics);
}

}  // namespace anymesher_native_v2
