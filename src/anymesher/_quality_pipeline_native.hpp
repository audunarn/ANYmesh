#pragma once

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "_triangulation_native.hpp"

namespace anymesher_quality_native {

struct QualitySignalInterrupted {};

class QualitySignalAwareGilRelease {
public:
    QualitySignalAwareGilRelease() : state_(PyEval_SaveThread()) {}
    ~QualitySignalAwareGilRelease() {
        if (state_ != nullptr) PyEval_RestoreThread(state_);
    }
    bool interrupted() {
        PyEval_RestoreThread(state_);
        state_ = nullptr;
        if (PyErr_CheckSignals() != 0) return true;
        state_ = PyEval_SaveThread();
        return false;
    }

private:
    PyThreadState* state_ = nullptr;
};

using anymesher_native::Index;
using anymesher_native::Point;
using anymesher_native::robust_orient;

struct Buffer {
    Py_buffer view{};
    bool held = false;
    ~Buffer() {
        if (held) {
            PyBuffer_Release(&view);
        }
    }
};

inline bool integer_format(const Py_buffer& view) {
    if (view.format == nullptr) {
        return false;
    }
    const char prefix = view.format[0];
    if (prefix == '<' || prefix == '>' || prefix == '!') {
        return false;
    }
    const char code = (prefix == '=' || prefix == '@') ? view.format[1] : prefix;
    return code == 'b' || code == 'B' || code == 'h' || code == 'H' ||
           code == 'i' || code == 'I' || code == 'l' || code == 'L' ||
           code == 'q' || code == 'Q';
}

inline bool acquire_matrix(
    PyObject* object,
    Buffer& buffer,
    Py_ssize_t columns,
    bool floating,
    const char* name) {
    if (PyObject_GetBuffer(
            object,
            &buffer.view,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    buffer.held = true;
    const bool shape = buffer.view.ndim == 2 && buffer.view.shape != nullptr &&
                       buffer.view.strides != nullptr &&
                       buffer.view.shape[1] == columns;
    const bool type = floating
        ? buffer.view.itemsize == static_cast<Py_ssize_t>(sizeof(double)) &&
              buffer.view.format != nullptr &&
              (buffer.view.format[0] == 'd' ||
               ((buffer.view.format[0] == '=' || buffer.view.format[0] == '@') &&
                buffer.view.format[1] == 'd'))
        : integer_format(buffer.view);
    if (!shape || !type) {
        PyErr_Format(PyExc_TypeError, "%s has an invalid native buffer", name);
        return false;
    }
    return true;
}

inline bool acquire_vector(PyObject* object, Buffer& buffer, const char* name) {
    if (PyObject_GetBuffer(
            object,
            &buffer.view,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        return false;
    }
    buffer.held = true;
    if (buffer.view.ndim != 1 || buffer.view.shape == nullptr ||
        buffer.view.strides == nullptr || !integer_format(buffer.view)) {
        PyErr_Format(PyExc_TypeError, "%s has an invalid native buffer", name);
        return false;
    }
    return true;
}

inline Index integer_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column = 0) {
    const char* address = static_cast<const char*>(view.buf) + row * view.strides[0];
    if (view.ndim == 2) {
        address += column * view.strides[1];
    }
    const char prefix = view.format[0];
    const char code = (prefix == '=' || prefix == '@') ? view.format[1] : prefix;
    const bool unsigned_value =
        code == 'B' || code == 'H' || code == 'I' || code == 'L' || code == 'Q';
    if (!unsigned_value) {
        switch (view.itemsize) {
            case 1: return *reinterpret_cast<const std::int8_t*>(address);
            case 2: return *reinterpret_cast<const std::int16_t*>(address);
            case 4: return *reinterpret_cast<const std::int32_t*>(address);
            case 8: return *reinterpret_cast<const std::int64_t*>(address);
            default: return -1;
        }
    }
    std::uint64_t value = 0;
    switch (view.itemsize) {
        case 1: value = *reinterpret_cast<const std::uint8_t*>(address); break;
        case 2: value = *reinterpret_cast<const std::uint16_t*>(address); break;
        case 4: value = *reinterpret_cast<const std::uint32_t*>(address); break;
        case 8: value = *reinterpret_cast<const std::uint64_t*>(address); break;
        default: return -1;
    }
    if (value > static_cast<std::uint64_t>(std::numeric_limits<Index>::max())) {
        return -1;
    }
    return static_cast<Index>(value);
}

inline double floating_at(const Py_buffer& view, Py_ssize_t row, Py_ssize_t column) {
    const char* address = static_cast<const char*>(view.buf) +
                          row * view.strides[0] + column * view.strides[1];
    return *reinterpret_cast<const double*>(address);
}

inline std::vector<Point> load_points(const Py_buffer& view) {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(view.shape[0]));
    for (Py_ssize_t row = 0; row < view.shape[0]; ++row) {
        points.push_back({floating_at(view, row, 0), floating_at(view, row, 1)});
    }
    return points;
}

inline std::vector<Index> load_vector(const Py_buffer& view, Index limit, const char* name) {
    std::vector<Index> values;
    values.reserve(static_cast<std::size_t>(view.shape[0]));
    for (Py_ssize_t row = 0; row < view.shape[0]; ++row) {
        const Index value = integer_at(view, row);
        if (value < 0 || value >= limit) {
            throw std::runtime_error(std::string(name) + " references an invalid row");
        }
        values.push_back(value);
    }
    return values;
}

inline std::vector<std::vector<Index>> load_holes(
    const Py_buffer& indices,
    const Py_buffer& offsets,
    Index point_count) {
    if (offsets.shape[0] < 1 || integer_at(offsets, 0) != 0 ||
        integer_at(offsets, offsets.shape[0] - 1) != indices.shape[0]) {
        throw std::runtime_error("hole offsets are invalid");
    }
    std::vector<std::vector<Index>> holes;
    for (Py_ssize_t number = 0; number + 1 < offsets.shape[0]; ++number) {
        const Index start = integer_at(offsets, number);
        const Index end = integer_at(offsets, number + 1);
        if (start < 0 || end < start || end > indices.shape[0]) {
            throw std::runtime_error("hole offsets are invalid");
        }
        std::vector<Index> hole;
        for (Index row = start; row < end; ++row) {
            const Index value = integer_at(indices, static_cast<Py_ssize_t>(row));
            if (value < 0 || value >= point_count) {
                throw std::runtime_error("hole loop references an invalid row");
            }
            hole.push_back(value);
        }
        holes.push_back(std::move(hole));
    }
    return holes;
}

inline bool point_on_segment(
    const Point& point,
    const Point& first,
    const Point& second,
    double tolerance) {
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    const double length = std::hypot(dx, dy);
    if (std::abs(robust_orient(first, second, point)) >
        tolerance * std::max(1.0, length)) {
        return false;
    }
    return point.x >= std::min(first.x, second.x) - tolerance &&
           point.x <= std::max(first.x, second.x) + tolerance &&
           point.y >= std::min(first.y, second.y) - tolerance &&
           point.y <= std::max(first.y, second.y) + tolerance;
}

inline bool point_in_ring(
    const Point& point,
    const std::vector<Point>& points,
    const std::vector<Index>& ring,
    double tolerance) {
    bool inside = false;
    for (std::size_t index = 0; index < ring.size(); ++index) {
        const Point& first = points[static_cast<std::size_t>(ring[index])];
        const Point& second = points[static_cast<std::size_t>(
            ring[(index + 1) % ring.size()])];
        if (point_on_segment(point, first, second, tolerance)) {
            return true;
        }
        const bool straddles = (first.y > point.y) != (second.y > point.y);
        if (straddles) {
            const double crossing = first.x +
                (point.y - first.y) * (second.x - first.x) /
                (second.y - first.y);
            if (point.x < crossing) {
                inside = !inside;
            }
        }
    }
    return inside;
}

inline bool on_ring(
    const Point& point,
    const std::vector<Point>& points,
    const std::vector<Index>& ring,
    double tolerance) {
    for (std::size_t index = 0; index < ring.size(); ++index) {
        if (point_on_segment(
                point,
                points[static_cast<std::size_t>(ring[index])],
                points[static_cast<std::size_t>(ring[(index + 1) % ring.size()])],
                tolerance)) {
            return true;
        }
    }
    return false;
}

inline bool quality_proper_intersection(
    const Point& a,
    const Point& b,
    const Point& c,
    const Point& d) {
    const double first = robust_orient(a, b, c);
    const double second = robust_orient(a, b, d);
    const double third = robust_orient(c, d, a);
    const double fourth = robust_orient(c, d, b);
    return first * second < 0.0 && third * fourth < 0.0;
}

inline PyObject* py_pslg_segment_memberships(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* segments_object = nullptr;
    double tolerance = 0.0;
    if (!PyArg_ParseTuple(args, "OOd", &points_object, &segments_object, &tolerance)) {
        return nullptr;
    }
    Buffer points_buffer;
    Buffer segments_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, true, "points") ||
        !acquire_matrix(segments_object, segments_buffer, 2, false, "segments")) {
        return nullptr;
    }
    std::vector<std::vector<std::pair<double, Index>>> memberships;
    std::string failure;
    Py_BEGIN_ALLOW_THREADS
    try {
        const auto points = load_points(points_buffer.view);
        memberships.resize(static_cast<std::size_t>(segments_buffer.view.shape[0]));
        for (Py_ssize_t segment_row = 0;
             segment_row < segments_buffer.view.shape[0];
             ++segment_row) {
            const Index first = integer_at(segments_buffer.view, segment_row, 0);
            const Index second = integer_at(segments_buffer.view, segment_row, 1);
            if (first < 0 || second < 0 || first >= static_cast<Index>(points.size()) ||
                second >= static_cast<Index>(points.size()) || first == second) {
                throw std::runtime_error("segment references an invalid point row");
            }
            const Point& start = points[static_cast<std::size_t>(first)];
            const Point& end = points[static_cast<std::size_t>(second)];
            const double dx = end.x - start.x;
            const double dy = end.y - start.y;
            const double denominator = dx * dx + dy * dy;
            auto& rows = memberships[static_cast<std::size_t>(segment_row)];
            for (Index row = 0; row < static_cast<Index>(points.size()); ++row) {
                const Point& point = points[static_cast<std::size_t>(row)];
                if (!point_on_segment(point, start, end, tolerance)) {
                    continue;
                }
                double parameter =
                    ((point.x - start.x) * dx + (point.y - start.y) * dy) /
                    denominator;
                if (parameter >= -tolerance && parameter <= 1.0 + tolerance) {
                    parameter = std::max(0.0, std::min(1.0, parameter));
                    rows.emplace_back(parameter, row);
                }
            }
            std::sort(rows.begin(), rows.end());
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Py_END_ALLOW_THREADS
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(memberships.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t number = 0; number < static_cast<Py_ssize_t>(memberships.size()); ++number) {
        const auto& rows = memberships[static_cast<std::size_t>(number)];
        PyObject* item = PyList_New(static_cast<Py_ssize_t>(rows.size()));
        if (item == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(rows.size()); ++index) {
            PyList_SET_ITEM(
                item,
                index,
                PyLong_FromLongLong(rows[static_cast<std::size_t>(index)].second));
        }
        PyList_SET_ITEM(result, number, item);
    }
    return result;
}

inline PyObject* py_pslg_domain_classification(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* outer_object = nullptr;
    PyObject* hole_indices_object = nullptr;
    PyObject* hole_offsets_object = nullptr;
    double tolerance = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOOd",
            &points_object,
            &outer_object,
            &hole_indices_object,
            &hole_offsets_object,
            &tolerance)) {
        return nullptr;
    }
    Buffer points_buffer;
    Buffer outer_buffer;
    Buffer hole_indices_buffer;
    Buffer hole_offsets_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, true, "points") ||
        !acquire_vector(outer_object, outer_buffer, "outer") ||
        !acquire_vector(hole_indices_object, hole_indices_buffer, "hole_indices") ||
        !acquire_vector(hole_offsets_object, hole_offsets_buffer, "hole_offsets")) {
        return nullptr;
    }
    std::vector<std::array<unsigned char, 3>> classified;
    std::string failure;
    Py_BEGIN_ALLOW_THREADS
    try {
        const auto points = load_points(points_buffer.view);
        const auto outer = load_vector(
            outer_buffer.view, static_cast<Index>(points.size()), "outer");
        const auto holes = load_holes(
            hole_indices_buffer.view,
            hole_offsets_buffer.view,
            static_cast<Index>(points.size()));
        classified.reserve(points.size());
        for (const Point& point : points) {
            const bool boundary = on_ring(point, points, outer, tolerance);
            const bool inside = point_in_ring(point, points, outer, tolerance);
            bool in_hole = false;
            for (const auto& hole : holes) {
                if (point_in_ring(point, points, hole, tolerance) &&
                    !on_ring(point, points, hole, tolerance)) {
                    in_hole = true;
                    break;
                }
            }
            classified.push_back({
                static_cast<unsigned char>(boundary),
                static_cast<unsigned char>(inside),
                static_cast<unsigned char>(in_hole),
            });
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Py_END_ALLOW_THREADS
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(classified.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(classified.size()); ++row) {
        const auto& value = classified[static_cast<std::size_t>(row)];
        PyObject* item = Py_BuildValue("(OOO)",
            value[0] ? Py_True : Py_False,
            value[1] ? Py_True : Py_False,
            value[2] ? Py_True : Py_False);
        if (item == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, row, item);
    }
    return result;
}

struct ValidationEdge {
    Index first = -1;
    Index second = -1;
    ValidationEdge() = default;
    ValidationEdge(Index a, Index b) : first(std::min(a, b)), second(std::max(a, b)) {}
    bool operator<(const ValidationEdge& other) const {
        return std::tie(first, second) < std::tie(other.first, other.second);
    }
};

struct ValidationEdgeBox {
    ValidationEdge edge;
    double minimum_x = 0.0;
    double maximum_x = 0.0;
    double minimum_y = 0.0;
    double maximum_y = 0.0;
};

inline double ring_area(
    const std::vector<Point>& points,
    const std::vector<Index>& ring) {
    double area = 0.0;
    for (std::size_t index = 0; index < ring.size(); ++index) {
        const Point& first = points[static_cast<std::size_t>(ring[index])];
        const Point& second = points[static_cast<std::size_t>(
            ring[(index + 1) % ring.size()])];
        area += first.x * second.y - first.y * second.x;
    }
    return 0.5 * area;
}

inline std::set<ValidationEdge> load_edge_set(
    const Py_buffer& view,
    Index point_count,
    const char* name) {
    std::set<ValidationEdge> result;
    for (Py_ssize_t row = 0; row < view.shape[0]; ++row) {
        const Index first = integer_at(view, row, 0);
        const Index second = integer_at(view, row, 1);
        if (first < 0 || second < 0 || first >= point_count || second >= point_count ||
            first == second) {
            throw std::runtime_error(std::string(name) + " references an invalid row");
        }
        result.emplace(first, second);
    }
    return result;
}

inline PyObject* py_validate_triangulation(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* triangles_object = nullptr;
    PyObject* segments_object = nullptr;
    PyObject* boundary_object = nullptr;
    PyObject* mandatory_object = nullptr;
    PyObject* outer_object = nullptr;
    PyObject* hole_indices_object = nullptr;
    PyObject* hole_offsets_object = nullptr;
    double tolerance = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOd",
            &points_object,
            &triangles_object,
            &segments_object,
            &boundary_object,
            &mandatory_object,
            &outer_object,
            &hole_indices_object,
            &hole_offsets_object,
            &tolerance)) {
        return nullptr;
    }
    Buffer points_buffer;
    Buffer triangles_buffer;
    Buffer segments_buffer;
    Buffer boundary_buffer;
    Buffer mandatory_buffer;
    Buffer outer_buffer;
    Buffer hole_indices_buffer;
    Buffer hole_offsets_buffer;
    if (!acquire_matrix(points_object, points_buffer, 2, true, "points") ||
        !acquire_matrix(triangles_object, triangles_buffer, 3, false, "triangles") ||
        !acquire_matrix(segments_object, segments_buffer, 2, false, "segments") ||
        !acquire_matrix(boundary_object, boundary_buffer, 2, false, "boundary") ||
        !acquire_matrix(mandatory_object, mandatory_buffer, 2, false, "mandatory") ||
        !acquire_vector(outer_object, outer_buffer, "outer") ||
        !acquire_vector(hole_indices_object, hole_indices_buffer, "hole_indices") ||
        !acquire_vector(hole_offsets_object, hole_offsets_buffer, "hole_offsets")) {
        return nullptr;
    }
    std::vector<std::array<Index, 3>> canonical;
    std::string failure;
    bool interrupted = false;
    try {
        QualitySignalAwareGilRelease gil;
        std::size_t work = 0;
        const auto points = load_points(points_buffer.view);
        const Index point_count = static_cast<Index>(points.size());
        const auto outer = load_vector(outer_buffer.view, point_count, "outer");
        const auto holes = load_holes(
            hole_indices_buffer.view, hole_offsets_buffer.view, point_count);
        const auto required = load_edge_set(segments_buffer.view, point_count, "segments");
        const auto boundary = load_edge_set(boundary_buffer.view, point_count, "boundary");
        const auto mandatory = load_edge_set(mandatory_buffer.view, point_count, "mandatory");
        std::set<std::array<Index, 3>> seen;
        std::map<ValidationEdge, std::vector<Index>> incidence;
        double area = 0.0;
        for (Py_ssize_t row = 0; row < triangles_buffer.view.shape[0]; ++row) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            std::array<Index, 3> triangle{
                integer_at(triangles_buffer.view, row, 0),
                integer_at(triangles_buffer.view, row, 1),
                integer_at(triangles_buffer.view, row, 2),
            };
            if (*std::min_element(triangle.begin(), triangle.end()) < 0 ||
                *std::max_element(triangle.begin(), triangle.end()) >= point_count ||
                triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
                triangle[0] == triangle[2]) {
                throw std::runtime_error("native triangulation returned invalid connectivity");
            }
            double determinant = robust_orient(
                points[static_cast<std::size_t>(triangle[0])],
                points[static_cast<std::size_t>(triangle[1])],
                points[static_cast<std::size_t>(triangle[2])]);
            if (determinant < 0.0) {
                std::swap(triangle[1], triangle[2]);
                determinant = -determinant;
            }
            if (determinant <= 0.0) {
                throw std::runtime_error("native triangulation returned a zero-area cell");
            }
            if (!seen.insert(triangle).second) {
                throw std::runtime_error("native triangulation returned duplicate cells");
            }
            Point centroid{
                (points[static_cast<std::size_t>(triangle[0])].x +
                 points[static_cast<std::size_t>(triangle[1])].x +
                 points[static_cast<std::size_t>(triangle[2])].x) / 3.0,
                (points[static_cast<std::size_t>(triangle[0])].y +
                 points[static_cast<std::size_t>(triangle[1])].y +
                 points[static_cast<std::size_t>(triangle[2])].y) / 3.0,
            };
            bool inside = point_in_ring(centroid, points, outer, tolerance);
            for (const auto& hole : holes) {
                if (point_in_ring(centroid, points, hole, tolerance)) {
                    inside = false;
                    break;
                }
            }
            if (!inside) {
                throw std::runtime_error("native triangulation returned a cell outside the domain");
            }
            const Index output_row = static_cast<Index>(canonical.size());
            canonical.push_back(triangle);
            area += 0.5 * determinant;
            for (int local = 0; local < 3; ++local) {
                auto& rows = incidence[ValidationEdge(
                    triangle[local], triangle[(local + 1) % 3])];
                rows.push_back(output_row);
                if (rows.size() > 2) {
                    throw std::runtime_error("native triangulation returned nonmanifold incidence");
                }
            }
        }
        for (const auto& edge : required) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            if (incidence.find(edge) == incidence.end()) {
                throw std::runtime_error("native triangulation omitted mandatory segments");
            }
        }
        for (const auto& edge : boundary) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            const auto found = incidence.find(edge);
            if (found == incidence.end() || found->second.size() != 1) {
                throw std::runtime_error("native triangulation returned invalid boundary incidence");
            }
        }
        for (const auto& item : incidence) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            if (item.second.size() == 1 && boundary.find(item.first) == boundary.end()) {
                throw std::runtime_error("native triangulation left open interior edges");
            }
        }
        for (const auto& edge : mandatory) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            if (incidence.find(edge) == incidence.end()) {
                throw std::runtime_error("native triangulation omitted a mandatory interior constraint");
            }
        }
        std::vector<ValidationEdgeBox> edges;
        edges.reserve(incidence.size());
        for (const auto& item : incidence) {
            if (++work % 4096 == 0 && gil.interrupted()) {
                throw QualitySignalInterrupted{};
            }
            const Point& first = points[static_cast<std::size_t>(item.first.first)];
            const Point& second = points[static_cast<std::size_t>(item.first.second)];
            edges.push_back({
                item.first,
                std::min(first.x, second.x),
                std::max(first.x, second.x),
                std::min(first.y, second.y),
                std::max(first.y, second.y),
            });
        }
        std::sort(
            edges.begin(), edges.end(),
            [](const ValidationEdgeBox& first, const ValidationEdgeBox& second) {
                return std::tie(
                           first.minimum_x, first.maximum_x,
                           first.minimum_y, first.maximum_y,
                           first.edge.first, first.edge.second) <
                       std::tie(
                           second.minimum_x, second.maximum_x,
                           second.minimum_y, second.maximum_y,
                           second.edge.first, second.edge.second);
            });
        for (std::size_t first_row = 0; first_row < edges.size(); ++first_row) {
            const auto& first_box = edges[first_row];
            const auto& first_edge = first_box.edge;
            for (std::size_t second_row = first_row + 1; second_row < edges.size(); ++second_row) {
                if (++work % 4096 == 0 && gil.interrupted()) {
                    throw QualitySignalInterrupted{};
                }
                const auto& second_box = edges[second_row];
                if (second_box.minimum_x > first_box.maximum_x + tolerance) {
                    break;
                }
                if (second_box.minimum_y > first_box.maximum_y + tolerance ||
                    first_box.minimum_y > second_box.maximum_y + tolerance) {
                    continue;
                }
                const auto& second_edge = second_box.edge;
                const bool shared_endpoint =
                    first_edge.first == second_edge.first ||
                    first_edge.first == second_edge.second ||
                    first_edge.second == second_edge.first ||
                    first_edge.second == second_edge.second;
                if (shared_endpoint) {
                    const Point& a = points[static_cast<std::size_t>(first_edge.first)];
                    const Point& b = points[static_cast<std::size_t>(first_edge.second)];
                    const Point& c = points[static_cast<std::size_t>(second_edge.first)];
                    const Point& d = points[static_cast<std::size_t>(second_edge.second)];
                    const bool overlap =
                        (first_edge.first != second_edge.first &&
                         first_edge.first != second_edge.second &&
                         point_on_segment(a, c, d, tolerance)) ||
                        (first_edge.second != second_edge.first &&
                         first_edge.second != second_edge.second &&
                         point_on_segment(b, c, d, tolerance)) ||
                        (second_edge.first != first_edge.first &&
                         second_edge.first != first_edge.second &&
                         point_on_segment(c, a, b, tolerance)) ||
                        (second_edge.second != first_edge.first &&
                         second_edge.second != first_edge.second &&
                         point_on_segment(d, a, b, tolerance));
                    if (overlap) {
                        throw std::runtime_error(
                            "native triangulation returned crossing or overlapping edges");
                    }
                    continue;
                }
                const Point& a = points[static_cast<std::size_t>(first_edge.first)];
                const Point& b = points[static_cast<std::size_t>(first_edge.second)];
                const Point& c = points[static_cast<std::size_t>(second_edge.first)];
                const Point& d = points[static_cast<std::size_t>(second_edge.second)];
                if (quality_proper_intersection(a, b, c, d) ||
                    point_on_segment(a, c, d, tolerance) ||
                    point_on_segment(b, c, d, tolerance) ||
                    point_on_segment(c, a, b, tolerance) ||
                    point_on_segment(d, a, b, tolerance)) {
                    throw std::runtime_error(
                        "native triangulation returned crossing or overlapping edges");
                }
            }
        }
        double expected_area = std::abs(ring_area(points, outer));
        for (const auto& hole : holes) {
            expected_area -= std::abs(ring_area(points, hole));
        }
        const double area_tolerance = std::max(
            tolerance * std::max(1.0, expected_area) *
                static_cast<double>(std::max<std::size_t>(16, boundary.size())),
            128.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, expected_area));
        if (std::abs(area - expected_area) > area_tolerance) {
            throw std::runtime_error(
                "native triangulation coverage area does not match the prepared domain");
        }
        std::sort(canonical.begin(), canonical.end());
    } catch (const QualitySignalInterrupted&) {
        interrupted = true;
    } catch (const std::exception& error) {
        failure = error.what();
    }
    if (interrupted) return nullptr;
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(canonical.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(canonical.size()); ++row) {
        const auto& triangle = canonical[static_cast<std::size_t>(row)];
        PyObject* item = Py_BuildValue("(LLL)",
            static_cast<long long>(triangle[0]),
            static_cast<long long>(triangle[1]),
            static_cast<long long>(triangle[2]));
        if (item == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, row, item);
    }
    return result;
}

struct Point3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Point3 subtract(const Point3& first, const Point3& second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

inline Point3 cross(const Point3& first, const Point3& second) {
    return {
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    };
}

inline double dot(const Point3& first, const Point3& second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

inline double norm(const Point3& value) {
    return std::sqrt(dot(value, value));
}

inline Point3 add(const Point3& first, const Point3& second) {
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

struct RecombineEdge {
    Index first = -1;
    Index second = -1;
    RecombineEdge() = default;
    RecombineEdge(Index a, Index b) : first(std::min(a, b)), second(std::max(a, b)) {}
    bool operator<(const RecombineEdge& other) const {
        return std::tie(first, second) < std::tie(other.first, other.second);
    }
    bool operator==(const RecombineEdge& other) const {
        return first == other.first && second == other.second;
    }
};

struct QuadMetrics {
    double area = 0.0;
    double aspect = 0.0;
    double minimum_angle = 0.0;
    double maximum_angle = 0.0;
    double jacobian = 0.0;
    double warpage = 0.0;
};

inline QuadMetrics quad_metrics(
    const std::vector<Point3>& points,
    const std::array<Index, 4>& corners) {
    std::array<double, 4> lengths{};
    std::array<double, 4> angles{};
    std::array<double, 4> jacobians{};
    Point3 reference{};
    for (int index = 0; index < 4; ++index) {
        reference = add(reference, cross(
            points[static_cast<std::size_t>(corners[index])],
            points[static_cast<std::size_t>(corners[(index + 1) % 4])]));
    }
    const double reference_norm = norm(reference);
    if (reference_norm > 0.0) {
        reference.x /= reference_norm;
        reference.y /= reference_norm;
        reference.z /= reference_norm;
    }
    for (int index = 0; index < 4; ++index) {
        const Point3 outgoing = subtract(
            points[static_cast<std::size_t>(corners[(index + 1) % 4])],
            points[static_cast<std::size_t>(corners[index])]);
        const Point3 incoming = subtract(
            points[static_cast<std::size_t>(corners[(index + 3) % 4])],
            points[static_cast<std::size_t>(corners[index])]);
        lengths[index] = norm(outgoing);
        const double denominator = std::max(
            lengths[index] * norm(incoming), 1.0e-300);
        const double cosine = std::max(-1.0, std::min(1.0, dot(outgoing, incoming) / denominator));
        angles[index] = std::acos(cosine) * 180.0 / std::acos(-1.0);
        jacobians[index] = dot(cross(outgoing, incoming), reference) / denominator;
    }
    const Point3 first_cross = cross(
        subtract(points[static_cast<std::size_t>(corners[1])], points[static_cast<std::size_t>(corners[0])]),
        subtract(points[static_cast<std::size_t>(corners[2])], points[static_cast<std::size_t>(corners[0])]));
    const Point3 second_cross = cross(
        subtract(points[static_cast<std::size_t>(corners[2])], points[static_cast<std::size_t>(corners[0])]),
        subtract(points[static_cast<std::size_t>(corners[3])], points[static_cast<std::size_t>(corners[0])]));
    const double first_norm = norm(first_cross);
    const double second_norm = norm(second_cross);
    double warpage = 1.0;
    if (first_norm > 0.0 && second_norm > 0.0) {
        const double cosine = std::max(-1.0, std::min(
            1.0, dot(first_cross, second_cross) / (first_norm * second_norm)));
        warpage = std::acos(cosine) / std::acos(-1.0);
    }
    return {
        0.5 * (first_norm + second_norm),
        *std::max_element(lengths.begin(), lengths.end()) /
            std::max(*std::min_element(lengths.begin(), lengths.end()), 1.0e-300),
        *std::min_element(angles.begin(), angles.end()),
        *std::max_element(angles.begin(), angles.end()),
        *std::min_element(jacobians.begin(), jacobians.end()),
        warpage,
    };
}

struct RecombineCandidate {
    Index first = -1;
    Index second = -1;
    RecombineEdge shared;
    std::array<Index, 4> corners{};
    double score = 0.0;
    QuadMetrics metrics;
};

inline PyObject* py_recombine_decisions(PyObject*, PyObject* args) {
    if (PyTuple_Size(args) != 14) {
        PyErr_SetString(PyExc_TypeError, "recombine_decisions expects 14 arguments");
        return nullptr;
    }
    PyObject* objects[8]{};
    for (int index = 0; index < 8; ++index) {
        objects[index] = PyTuple_GetItem(args, index);
    }
    const double min_jacobian = PyFloat_AsDouble(PyTuple_GetItem(args, 8));
    const double max_aspect = PyFloat_AsDouble(PyTuple_GetItem(args, 9));
    const double min_angle = PyFloat_AsDouble(PyTuple_GetItem(args, 10));
    const double max_angle = PyFloat_AsDouble(PyTuple_GetItem(args, 11));
    const double max_warpage = PyFloat_AsDouble(PyTuple_GetItem(args, 12));
    const long long max_exchange_work = PyLong_AsLongLong(PyTuple_GetItem(args, 13));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    Buffer points_buffer;
    Buffer triangles_buffer;
    Buffer triangle_ids_buffer;
    Buffer node_ids_buffer;
    Buffer active_rows_buffer;
    Buffer protected_buffer;
    Buffer unused_first;
    Buffer unused_second;
    if (!acquire_matrix(objects[0], points_buffer, 3, true, "points") ||
        !acquire_matrix(objects[1], triangles_buffer, 3, false, "triangles") ||
        !acquire_vector(objects[2], triangle_ids_buffer, "triangle_ids") ||
        !acquire_vector(objects[3], node_ids_buffer, "node_ids") ||
        !acquire_vector(objects[4], active_rows_buffer, "active_rows") ||
        !acquire_matrix(objects[5], protected_buffer, 2, false, "protected_edges")) {
        return nullptr;
    }
    // Arguments 6 and 7 are reserved for deterministic ABI growth.  They are
    // required empty int64 vectors so a future kernel can add ownership data
    // without changing the positional protocol.
    if (!acquire_vector(objects[6], unused_first, "reserved_first") ||
        !acquire_vector(objects[7], unused_second, "reserved_second") ||
        unused_first.view.shape[0] != 0 || unused_second.view.shape[0] != 0) {
        PyErr_SetString(PyExc_TypeError, "reserved recombination vectors must be empty");
        return nullptr;
    }

    std::vector<RecombineCandidate> selected;
    Index candidate_count = 0;
    Index rejected = 0;
    Index exchange_count = 0;
    Index exchange_work = 0;
    bool exchange_truncated = false;
    std::string failure;
    Py_BEGIN_ALLOW_THREADS
    try {
        const Index point_count = points_buffer.view.shape[0];
        const Index triangle_count = triangles_buffer.view.shape[0];
        if (triangle_ids_buffer.view.shape[0] != triangle_count ||
            node_ids_buffer.view.shape[0] != point_count) {
            throw std::runtime_error("recombination identity arrays disagree");
        }
        std::vector<Point3> points;
        points.reserve(static_cast<std::size_t>(point_count));
        for (Index row = 0; row < point_count; ++row) {
            points.push_back({
                floating_at(points_buffer.view, row, 0),
                floating_at(points_buffer.view, row, 1),
                floating_at(points_buffer.view, row, 2),
            });
        }
        std::vector<std::array<Index, 3>> triangles;
        triangles.reserve(static_cast<std::size_t>(triangle_count));
        for (Index row = 0; row < triangle_count; ++row) {
            triangles.push_back({
                integer_at(triangles_buffer.view, row, 0),
                integer_at(triangles_buffer.view, row, 1),
                integer_at(triangles_buffer.view, row, 2),
            });
        }
        std::set<RecombineEdge> protected_edges;
        for (Py_ssize_t row = 0; row < protected_buffer.view.shape[0]; ++row) {
            protected_edges.emplace(
                integer_at(protected_buffer.view, row, 0),
                integer_at(protected_buffer.view, row, 1));
        }
        std::map<RecombineEdge, std::vector<Index>> incidence;
        for (Py_ssize_t active = 0; active < active_rows_buffer.view.shape[0]; ++active) {
            const Index row = integer_at(active_rows_buffer.view, active);
            if (row < 0 || row >= triangle_count) {
                throw std::runtime_error("active triangle row is invalid");
            }
            const auto& triangle = triangles[static_cast<std::size_t>(row)];
            for (int local = 0; local < 3; ++local) {
                incidence[RecombineEdge(triangle[local], triangle[(local + 1) % 3])].push_back(row);
            }
        }
        std::vector<RecombineCandidate> candidates;
        for (const auto& item : incidence) {
            if (item.second.size() != 2 || protected_edges.count(item.first)) {
                continue;
            }
            const Index first_row = item.second[0];
            const Index second_row = item.second[1];
            std::map<RecombineEdge, int> counts;
            for (Index row : {first_row, second_row}) {
                const auto& triangle = triangles[static_cast<std::size_t>(row)];
                for (int local = 0; local < 3; ++local) {
                    ++counts[RecombineEdge(triangle[local], triangle[(local + 1) % 3])];
                }
            }
            std::vector<RecombineEdge> boundary;
            for (const auto& edge : counts) {
                if (edge.second == 1) {
                    boundary.push_back(edge.first);
                }
            }
            if (boundary.size() != 4) {
                ++rejected;
                continue;
            }
            std::map<Index, std::vector<Index>> adjacency;
            for (const auto& edge : boundary) {
                adjacency[edge.first].push_back(edge.second);
                adjacency[edge.second].push_back(edge.first);
            }
            if (adjacency.size() != 4 || std::any_of(
                    adjacency.begin(), adjacency.end(),
                    [](const auto& value) { return value.second.size() != 2; })) {
                ++rejected;
                continue;
            }
            const auto stable_node = [&](Index row) {
                return std::make_pair(integer_at(node_ids_buffer.view, row), row);
            };
            Index start = adjacency.begin()->first;
            for (const auto& value : adjacency) {
                if (stable_node(value.first) < stable_node(start)) {
                    start = value.first;
                }
            }
            auto starts = adjacency[start];
            std::sort(starts.begin(), starts.end(), [&](Index a, Index b) {
                return stable_node(a) < stable_node(b);
            });
            std::vector<std::array<Index, 4>> options;
            for (Index next : starts) {
                std::vector<Index> cycle{start, next};
                while (cycle.size() < 4) {
                    auto choices = adjacency[cycle.back()];
                    choices.erase(std::remove(
                        choices.begin(), choices.end(), cycle[cycle.size() - 2]), choices.end());
                    if (choices.empty()) {
                        break;
                    }
                    cycle.push_back(choices.front());
                }
                if (cycle.size() == 4 && cycle[0] != cycle[1] && cycle[0] != cycle[2] &&
                    cycle[0] != cycle[3] && cycle[1] != cycle[2] && cycle[1] != cycle[3] &&
                    cycle[2] != cycle[3] &&
                    std::find(adjacency[cycle[3]].begin(), adjacency[cycle[3]].end(), start) !=
                        adjacency[cycle[3]].end()) {
                    options.push_back({cycle[0], cycle[1], cycle[2], cycle[3]});
                }
            }
            if (options.empty()) {
                ++rejected;
                continue;
            }
            Point3 reference{};
            for (Index row : {first_row, second_row}) {
                const auto& triangle = triangles[static_cast<std::size_t>(row)];
                reference = add(reference, cross(
                    subtract(points[static_cast<std::size_t>(triangle[1])], points[static_cast<std::size_t>(triangle[0])]),
                    subtract(points[static_cast<std::size_t>(triangle[2])], points[static_cast<std::size_t>(triangle[0])])));
            }
            std::vector<std::array<Index, 4>> aligned;
            for (const auto& cycle : options) {
                Point3 normal{};
                for (int local = 0; local < 4; ++local) {
                    normal = add(normal, cross(
                        points[static_cast<std::size_t>(cycle[local])],
                        points[static_cast<std::size_t>(cycle[(local + 1) % 4])]));
                }
                if (dot(normal, reference) >= 0.0) {
                    aligned.push_back(cycle);
                }
            }
            auto pool = aligned.empty() ? options : aligned;
            const auto cycle_key = [&](const std::array<Index, 4>& cycle) {
                return std::array<Index, 4>{
                    integer_at(node_ids_buffer.view, cycle[0]),
                    integer_at(node_ids_buffer.view, cycle[1]),
                    integer_at(node_ids_buffer.view, cycle[2]),
                    integer_at(node_ids_buffer.view, cycle[3]),
                };
            };
            const auto corners = *std::min_element(pool.begin(), pool.end(), [&](const auto& a, const auto& b) {
                return cycle_key(a) < cycle_key(b);
            });
            const QuadMetrics metrics = quad_metrics(points, corners);
            const bool finite = std::isfinite(metrics.area) && std::isfinite(metrics.aspect) &&
                std::isfinite(metrics.minimum_angle) && std::isfinite(metrics.maximum_angle) &&
                std::isfinite(metrics.jacobian) && std::isfinite(metrics.warpage);
            const bool acceptable = finite && metrics.jacobian >= min_jacobian &&
                metrics.aspect <= max_aspect && metrics.minimum_angle >= min_angle &&
                metrics.maximum_angle <= max_angle && metrics.warpage <= max_warpage;
            if (!acceptable) {
                ++rejected;
                continue;
            }
            const double score = metrics.jacobian -
                0.08 * std::log(std::max(metrics.aspect, 1.0)) -
                0.002 * std::max(0.0, 90.0 - metrics.minimum_angle) -
                0.002 * std::max(0.0, metrics.maximum_angle - 90.0) -
                0.25 * metrics.warpage;
            candidates.push_back({first_row, second_row, item.first, corners, score, metrics});
        }
        candidate_count = static_cast<Index>(candidates.size()) + rejected;
        const auto stable_candidate = [&](const RecombineCandidate& value) {
            const Index first_id = integer_at(triangle_ids_buffer.view, value.first);
            const Index second_id = integer_at(triangle_ids_buffer.view, value.second);
            return std::make_tuple(
                -value.score,
                std::min(first_id, second_id),
                std::max(first_id, second_id),
                std::array<Index, 4>{
                    integer_at(node_ids_buffer.view, value.corners[0]),
                    integer_at(node_ids_buffer.view, value.corners[1]),
                    integer_at(node_ids_buffer.view, value.corners[2]),
                    integer_at(node_ids_buffer.view, value.corners[3]),
                });
        };
        std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
            return stable_candidate(a) < stable_candidate(b);
        });
        std::set<Index> used;
        for (const auto& candidate : candidates) {
            if (used.count(candidate.first) || used.count(candidate.second)) {
                continue;
            }
            used.insert(candidate.first);
            used.insert(candidate.second);
            selected.push_back(candidate);
        }
        std::map<Index, std::vector<const RecombineCandidate*>> by_triangle;
        for (const auto& candidate : candidates) {
            by_triangle[candidate.first].push_back(&candidate);
            by_triangle[candidate.second].push_back(&candidate);
        }
        const auto key = [](const RecombineCandidate& value) {
            return std::make_tuple(value.first, value.second, value.shared.first, value.shared.second);
        };
        std::map<std::tuple<Index, Index, Index, Index>, RecombineCandidate> active;
        for (const auto& candidate : selected) {
            active[key(candidate)] = candidate;
        }
        std::vector<RecombineCandidate> queue = selected;
        for (std::size_t queue_index = 0; queue_index < queue.size(); ++queue_index) {
            const auto current = queue[queue_index];
            if (!active.count(key(current))) {
                continue;
            }
            const auto other = [](const RecombineCandidate& value, Index row) {
                return value.first == row ? value.second : value.first;
            };
            std::vector<const RecombineCandidate*> first_options;
            std::vector<const RecombineCandidate*> second_options;
            for (const auto* candidate : by_triangle[current.first]) {
                if (key(*candidate) != key(current) && !used.count(other(*candidate, current.first))) {
                    first_options.push_back(candidate);
                }
            }
            for (const auto* candidate : by_triangle[current.second]) {
                if (key(*candidate) != key(current) && !used.count(other(*candidate, current.second))) {
                    second_options.push_back(candidate);
                }
            }
            struct Exchange {
                const RecombineCandidate* first;
                const RecombineCandidate* second;
                Index first_other;
                Index second_other;
            };
            std::vector<Exchange> possible;
            bool stop = false;
            for (const auto* first : first_options) {
                const Index first_other = other(*first, current.first);
                for (const auto* second : second_options) {
                    if (exchange_work >= max_exchange_work) {
                        exchange_truncated = true;
                        stop = true;
                        break;
                    }
                    ++exchange_work;
                    const Index second_other = other(*second, current.second);
                    if (first_other != second_other && first != second) {
                        possible.push_back({first, second, first_other, second_other});
                    }
                }
                if (stop) {
                    break;
                }
            }
            if (stop) {
                break;
            }
            if (possible.empty()) {
                continue;
            }
            const auto exchange_key = [&](const Exchange& value) {
                std::array<Index, 4> ids{
                    integer_at(triangle_ids_buffer.view, current.first),
                    integer_at(triangle_ids_buffer.view, current.second),
                    integer_at(triangle_ids_buffer.view, value.first_other),
                    integer_at(triangle_ids_buffer.view, value.second_other),
                };
                std::sort(ids.begin(), ids.end());
                return std::make_pair(-(value.first->score + value.second->score), ids);
            };
            const auto chosen = *std::min_element(possible.begin(), possible.end(), [&](const auto& a, const auto& b) {
                return exchange_key(a) < exchange_key(b);
            });
            active.erase(key(current));
            active[key(*chosen.first)] = *chosen.first;
            active[key(*chosen.second)] = *chosen.second;
            used.insert(chosen.first_other);
            used.insert(chosen.second_other);
            queue.push_back(*chosen.first);
            queue.push_back(*chosen.second);
            ++exchange_count;
        }
        selected.clear();
        for (const auto& item : active) {
            selected.push_back(item.second);
        }
        std::sort(selected.begin(), selected.end(), [&](const auto& a, const auto& b) {
            const auto stable = [&](const RecombineCandidate& value) {
                const Index first_id = integer_at(triangle_ids_buffer.view, value.first);
                const Index second_id = integer_at(triangle_ids_buffer.view, value.second);
                return std::make_tuple(
                    std::min(first_id, second_id),
                    std::max(first_id, second_id),
                    std::array<Index, 4>{
                        integer_at(node_ids_buffer.view, value.corners[0]),
                        integer_at(node_ids_buffer.view, value.corners[1]),
                        integer_at(node_ids_buffer.view, value.corners[2]),
                        integer_at(node_ids_buffer.view, value.corners[3]),
                    });
            };
            return stable(a) < stable(b);
        });
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Py_END_ALLOW_THREADS
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* rows = PyList_New(static_cast<Py_ssize_t>(selected.size()));
    if (rows == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(selected.size()); ++row) {
        const auto& value = selected[static_cast<std::size_t>(row)];
        PyObject* item = Py_BuildValue(
            "(LLLLLLLLddddddd)",
            static_cast<long long>(value.first),
            static_cast<long long>(value.second),
            static_cast<long long>(value.shared.first),
            static_cast<long long>(value.shared.second),
            static_cast<long long>(value.corners[0]),
            static_cast<long long>(value.corners[1]),
            static_cast<long long>(value.corners[2]),
            static_cast<long long>(value.corners[3]),
            value.score,
            value.metrics.area,
            value.metrics.aspect,
            value.metrics.minimum_angle,
            value.metrics.maximum_angle,
            value.metrics.jacobian,
            value.metrics.warpage);
        if (item == nullptr) {
            Py_DECREF(rows);
            return nullptr;
        }
        PyList_SET_ITEM(rows, row, item);
    }
    return Py_BuildValue(
        "{s:N,s:L,s:L,s:L,s:L,s:O}",
        "selected", rows,
        "candidate_count", static_cast<long long>(candidate_count),
        "rejected_candidate_count", static_cast<long long>(rejected),
        "exchange_count", static_cast<long long>(exchange_count),
        "exchange_work", static_cast<long long>(exchange_work),
        "exchange_truncated", exchange_truncated ? Py_True : Py_False);
}

inline QuadMetrics triangle_metrics(
    const std::vector<Point3>& points,
    const std::array<Index, 3>& corners) {
    std::array<double, 3> lengths{};
    std::array<double, 3> angles{};
    std::array<double, 3> jacobians{};
    Point3 reference{};
    for (int index = 0; index < 3; ++index) {
        reference = add(reference, cross(
            points[static_cast<std::size_t>(corners[index])],
            points[static_cast<std::size_t>(corners[(index + 1) % 3])]));
    }
    const double reference_norm = norm(reference);
    if (reference_norm > 0.0) {
        reference.x /= reference_norm;
        reference.y /= reference_norm;
        reference.z /= reference_norm;
    }
    for (int index = 0; index < 3; ++index) {
        const Point3 outgoing = subtract(
            points[static_cast<std::size_t>(corners[(index + 1) % 3])],
            points[static_cast<std::size_t>(corners[index])]);
        const Point3 incoming = subtract(
            points[static_cast<std::size_t>(corners[(index + 2) % 3])],
            points[static_cast<std::size_t>(corners[index])]);
        lengths[index] = norm(outgoing);
        const double denominator = std::max(lengths[index] * norm(incoming), 1.0e-300);
        const double cosine = std::max(-1.0, std::min(1.0, dot(outgoing, incoming) / denominator));
        angles[index] = std::acos(cosine) * 180.0 / std::acos(-1.0);
        jacobians[index] = dot(cross(outgoing, incoming), reference) / denominator;
    }
    const Point3 area_cross = cross(
        subtract(points[static_cast<std::size_t>(corners[1])], points[static_cast<std::size_t>(corners[0])]),
        subtract(points[static_cast<std::size_t>(corners[2])], points[static_cast<std::size_t>(corners[0])]));
    return {
        0.5 * norm(area_cross),
        *std::max_element(lengths.begin(), lengths.end()) /
            std::max(*std::min_element(lengths.begin(), lengths.end()), 1.0e-300),
        *std::min_element(angles.begin(), angles.end()),
        *std::max_element(angles.begin(), angles.end()),
        *std::min_element(jacobians.begin(), jacobians.end()),
        0.0,
    };
}

inline PyObject* py_element_quality(PyObject*, PyObject* args) {
    PyObject* points_object = nullptr;
    PyObject* cells_object = nullptr;
    int corners = 0;
    if (!PyArg_ParseTuple(args, "OOi", &points_object, &cells_object, &corners)) {
        return nullptr;
    }
    if (corners != 3 && corners != 4) {
        PyErr_SetString(PyExc_ValueError, "element quality needs three or four corners");
        return nullptr;
    }
    Buffer points_buffer;
    Buffer cells_buffer;
    if (!acquire_matrix(points_object, points_buffer, 3, true, "points") ||
        !acquire_matrix(cells_object, cells_buffer, corners, false, "cells")) {
        return nullptr;
    }
    std::vector<QuadMetrics> values;
    std::string failure;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<Point3> points;
        points.reserve(static_cast<std::size_t>(points_buffer.view.shape[0]));
        for (Py_ssize_t row = 0; row < points_buffer.view.shape[0]; ++row) {
            points.push_back({
                floating_at(points_buffer.view, row, 0),
                floating_at(points_buffer.view, row, 1),
                floating_at(points_buffer.view, row, 2),
            });
        }
        values.reserve(static_cast<std::size_t>(cells_buffer.view.shape[0]));
        for (Py_ssize_t row = 0; row < cells_buffer.view.shape[0]; ++row) {
            if (corners == 3) {
                std::array<Index, 3> cell{
                    integer_at(cells_buffer.view, row, 0),
                    integer_at(cells_buffer.view, row, 1),
                    integer_at(cells_buffer.view, row, 2),
                };
                if (*std::min_element(cell.begin(), cell.end()) < 0 ||
                    *std::max_element(cell.begin(), cell.end()) >= static_cast<Index>(points.size())) {
                    throw std::runtime_error("triangle quality connectivity is invalid");
                }
                values.push_back(triangle_metrics(points, cell));
            } else {
                std::array<Index, 4> cell{
                    integer_at(cells_buffer.view, row, 0),
                    integer_at(cells_buffer.view, row, 1),
                    integer_at(cells_buffer.view, row, 2),
                    integer_at(cells_buffer.view, row, 3),
                };
                if (*std::min_element(cell.begin(), cell.end()) < 0 ||
                    *std::max_element(cell.begin(), cell.end()) >= static_cast<Index>(points.size())) {
                    throw std::runtime_error("quadrilateral quality connectivity is invalid");
                }
                values.push_back(quad_metrics(points, cell));
            }
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    Py_END_ALLOW_THREADS
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t row = 0; row < static_cast<Py_ssize_t>(values.size()); ++row) {
        const auto& value = values[static_cast<std::size_t>(row)];
        PyObject* item = Py_BuildValue(
            "(dddddd)",
            value.area,
            value.aspect,
            value.minimum_angle,
            value.maximum_angle,
            value.jacobian,
            value.warpage);
        if (item == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, row, item);
    }
    return result;
}

}  // namespace anymesher_quality_native
