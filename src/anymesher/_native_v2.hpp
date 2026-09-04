#pragma once

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
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
    double result = first - second + third;
    const double scale = std::abs(first) + std::abs(second) + std::abs(third);
    if (std::abs(result) <= 32.0 * std::numeric_limits<double>::epsilon() * scale) {
        throw PredicateUncertain("in-circle test requires the adaptive Python predicate");
    }
    if (orient(a, b, c) < 0.0L) {
        result = -result;
    }
    return static_cast<long double>(result);
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
    double candidate_x = 0.0;
    double candidate_y = 0.0;
    if (!PyArg_ParseTuple(args, "OOOdd:native_v2_mutable_t3_insert", &points_object, &triangles_object, &protected_object, &candidate_x, &candidate_y)) {
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
        std::vector<std::size_t> cavity;
        for (std::size_t row = 0; row < triangles.size(); ++row) {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                throw SignalInterrupted{};
            }
            const Triangle& triangle = triangles[row];
            if (in_circle(points[triangle[0]], points[triangle[1]], points[triangle[2]], candidate) > tolerance * tolerance) {
                cavity.push_back(row);
            }
        }
        if (cavity.empty()) {
            for (std::size_t row = 0; row < triangles.size(); ++row) {
                if (++work % kSignalCheckInterval == 0 && gil.interrupted()) {
                    throw SignalInterrupted{};
                }
                const Triangle& triangle = triangles[row];
                if (orient(points[triangle[0]], points[triangle[1]], candidate) >= -tolerance &&
                    orient(points[triangle[1]], points[triangle[2]], candidate) >= -tolerance &&
                    orient(points[triangle[2]], points[triangle[0]], candidate) >= -tolerance) {
                    cavity.push_back(row);
                    break;
                }
            }
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

}  // namespace anymesher_native_v2
