#pragma once

// Included inside anymesher_native_v2 after its buffer/predicate helpers.
// Own input connectivity. Share immutable memberships, never NumPy pointers.
constexpr const char* kT3SnapshotName = "anymesher.native_v2.t3_incidence/1";

struct T3IncidenceSnapshot {
    std::vector<Triangle> triangles;
    std::map<Triangle, std::size_t> row_by_cell;
    std::map<Edge, std::shared_ptr<const std::vector<Triangle>>> edge_cells;
};

inline const T3IncidenceSnapshot* t3_snapshot(PyObject* value) {
    return static_cast<const T3IncidenceSnapshot*>(
        PyCapsule_GetPointer(value, kT3SnapshotName));
}

inline void destroy_t3_snapshot(PyObject* capsule) noexcept {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    delete t3_snapshot(capsule);
    PyErr_Clear();
    PyErr_Restore(type, value, traceback);
}

inline bool require_t3_connectivity(const T3IncidenceSnapshot& state, const Py_buffer& buffer) {
    if (static_cast<std::size_t>(buffer.shape[0]) != state.triangles.size()) {
        PyErr_SetString(PyExc_RuntimeError, "mutable T3 incidence snapshot connectivity mismatch");
        return false;
    }
    for (Py_ssize_t row = 0; row < buffer.shape[0]; ++row) {
        if (static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return false;
        const Triangle cell{index_at(buffer, row, 0), index_at(buffer, row, 1), index_at(buffer, row, 2)};
        if (cell != state.triangles[static_cast<std::size_t>(row)]) {
            PyErr_SetString(PyExc_RuntimeError, "mutable T3 incidence snapshot connectivity mismatch");
            return false;
        }
    }
    return true;
}

inline PyObject* py_t3_incidence_check(PyObject*, PyObject* args) {
    PyObject *state_object = nullptr, *rows_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:native_v2_t3_incidence_check", &state_object, &rows_object)) return nullptr;
    const auto* state = t3_snapshot(state_object);
    if (state == nullptr) return nullptr;
    HeldBuffer buffer;
    if (!acquire_matrix(rows_object, buffer, 3, sizeof(Index), 'q', "triangles")) return nullptr;
    if (!require_t3_connectivity(*state, buffer.value)) return nullptr;
    Py_RETURN_TRUE;
}

inline PyObject* py_t3_incidence(PyObject*, PyObject* args) {
    PyObject *rows_object = nullptr, *previous_object = Py_None;
    if (!PyArg_ParseTuple(args, "O|O:native_v2_t3_incidence", &rows_object, &previous_object)) return nullptr;
    const T3IncidenceSnapshot* previous = nullptr;
    if (previous_object != Py_None) {
        previous = t3_snapshot(previous_object);
        if (previous == nullptr) return nullptr;
    }
    HeldBuffer buffer;
    if (!acquire_matrix(rows_object, buffer, 3, sizeof(Index), 'q', "triangles")) return nullptr;
    std::unique_ptr<T3IncidenceSnapshot> state;
    try {
        state = std::make_unique<T3IncidenceSnapshot>();
        state->triangles.reserve(static_cast<std::size_t>(buffer.value.shape[0]));
        // Copy while holding the GIL; all released-GIL work uses owned memory.
        for (Py_ssize_t row = 0; row < buffer.value.shape[0]; ++row) {
            if (static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && PyErr_CheckSignals() != 0) return nullptr;
            state->triangles.push_back({index_at(buffer.value, row, 0), index_at(buffer.value, row, 1), index_at(buffer.value, row, 2)});
        }
        SignalAwareGilRelease gil;
        std::size_t work = 0;
        const auto poll = [&]() {
            if (++work % kSignalCheckInterval == 0 && gil.interrupted()) throw SignalInterrupted{};
        };
        std::set<Triangle> unique_cells;
        for (std::size_t row = 0; row < state->triangles.size(); ++row) {
            poll();
            const Triangle& cell = state->triangles[row];
            Triangle identity = cell;
            std::sort(identity.begin(), identity.end());
            if (identity[0] < 0 || identity[0] == identity[1] || identity[1] == identity[2] || !unique_cells.insert(identity).second) {
                throw std::runtime_error("mutable T3 incidence requires distinct triangle rows");
            }
            state->row_by_cell.emplace(cell, row);
        }
        if (previous != nullptr) {
            for (const auto& item : previous->edge_cells) {
                poll();
                state->edge_cells.emplace(item.first, item.second);
            }
        }
        std::map<Edge, std::set<Triangle>> changes;
        const auto change = [&](const Triangle& cell, bool adding) {
            for (int index = 0; index < 3; ++index) {
                poll();
                const Edge key = edge(cell[index], cell[(index + 1) % 3]);
                auto found = changes.find(key);
                if (found == changes.end()) {
                    std::set<Triangle> members;
                    const auto old = state->edge_cells.find(key);
                    if (old != state->edge_cells.end()) members.insert(old->second->begin(), old->second->end());
                    found = changes.emplace(key, std::move(members)).first;
                }
                if (adding) found->second.insert(cell);
                else found->second.erase(cell);
            }
        };
        if (previous != nullptr) {
            for (const auto& item : previous->row_by_cell) {
                poll();
                if (state->row_by_cell.count(item.first) == 0) change(item.first, false);
            }
        }
        for (const auto& item : state->row_by_cell) {
            poll();
            if (previous == nullptr || previous->row_by_cell.count(item.first) == 0) change(item.first, true);
        }
        for (const auto& item : changes) {
            poll();
            if (item.second.empty()) state->edge_cells.erase(item.first);
            else state->edge_cells[item.first] = std::make_shared<const std::vector<Triangle>>(item.second.begin(), item.second.end());
        }
    } catch (const SignalInterrupted&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    PyObject* capsule = PyCapsule_New(state.get(), kT3SnapshotName, destroy_t3_snapshot);
    if (capsule != nullptr) state.release();
    return capsule;
}
