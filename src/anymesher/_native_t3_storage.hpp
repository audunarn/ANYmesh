#pragma once

// Included inside anymesher_native_v2 after its buffer/predicate helpers.
// Own input connectivity. Share immutable memberships, never NumPy pointers.
// Version the private storage ABI before any capsule pointer is interpreted.
// /2 owns persistent edge memberships and ordered/unoriented cell-rank trees.
constexpr const char* kT3SnapshotName = "anymesher.native_v2.t3_incidence/2";

// Immutable AVL storage keeps snapshot updates local to changed topology keys.
// Membership vectors and untouched subtrees are shared; published roots never
// mutate. Recursion is bounded by AVL height, independent of snapshot history.
template<class Key, class Value>
class PersistentT3Map {
public:
    using Members = Value;

private:
    struct Node;
    using Root = std::shared_ptr<const Node>;
    struct Node {
        std::pair<const Key, Members> entry;
        Root left;
        Root right;
        int height;
        std::size_t count;
        Node(const Key& key, Members value, Root lower, Root upper)
            : entry(key, std::move(value)), left(std::move(lower)),
              right(std::move(upper)),
              height(1 + std::max(left ? left->height : 0,
                                  right ? right->height : 0)),
              count(1 + (left ? left->count : 0) + (right ? right->count : 0)) {}
    };
    Root root_;

    static int height(const Root& root) { return root ? root->height : 0; }
    static Root make(const Key& key, const Members& value,
                     const Root& left, const Root& right) {
        return std::make_shared<const Node>(key, value, left, right);
    }
    static Root rotate_left(const Root& root) {
        const Root& pivot = root->right;
        return make(pivot->entry.first, pivot->entry.second,
                    make(root->entry.first, root->entry.second,
                         root->left, pivot->left), pivot->right);
    }
    static Root rotate_right(const Root& root) {
        const Root& pivot = root->left;
        return make(pivot->entry.first, pivot->entry.second, pivot->left,
                    make(root->entry.first, root->entry.second,
                         pivot->right, root->right));
    }
    static Root balance(const Root& root) {
        const int delta = height(root->left) - height(root->right);
        if (delta > 1) {
            if (height(root->left->left) < height(root->left->right)) {
                return rotate_right(make(root->entry.first, root->entry.second,
                                         rotate_left(root->left), root->right));
            }
            return rotate_right(root);
        }
        if (delta < -1) {
            if (height(root->right->right) < height(root->right->left)) {
                return rotate_left(make(root->entry.first, root->entry.second,
                                        root->left, rotate_right(root->right)));
            }
            return rotate_left(root);
        }
        return root;
    }
    static Root put(const Root& root, const Key& key, const Members& value) {
        if (!root) return make(key, value, {}, {});
        if (key < root->entry.first) {
            return balance(make(root->entry.first, root->entry.second,
                                put(root->left, key, value), root->right));
        }
        if (root->entry.first < key) {
            return balance(make(root->entry.first, root->entry.second,
                                root->left, put(root->right, key, value)));
        }
        return make(key, value, root->left, root->right);
    }
    static Root remove(const Root& root, const Key& key) {
        if (!root) return root;
        if (key < root->entry.first) {
            const Root changed = remove(root->left, key);
            if (changed == root->left) return root;
            return balance(make(root->entry.first, root->entry.second,
                                changed, root->right));
        }
        if (root->entry.first < key) {
            const Root changed = remove(root->right, key);
            if (changed == root->right) return root;
            return balance(make(root->entry.first, root->entry.second,
                                root->left, changed));
        }
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        Root next = root->right;
        while (next->left) next = next->left;
        return balance(make(next->entry.first, next->entry.second, root->left,
                            remove(root->right, next->entry.first)));
    }

    template<class Iterator, class Poll, class Factory>
    static Root sorted_nodes(Iterator& cursor, std::size_t count,
                             Poll& poll, Factory& factory) {
        if (count == 0) return {};
        const std::size_t lower_count = count / 2;
        Root lower = sorted_nodes(cursor, lower_count, poll, factory);
        poll();
        const auto entry = factory(*cursor);
        ++cursor;
        Root upper = sorted_nodes(cursor, count - lower_count - 1, poll, factory);
        return make(entry.first, entry.second, lower, upper);
    }

public:
    template<class Iterator, class Poll, class Factory>
    static PersistentT3Map from_sorted(
            Iterator cursor, std::size_t count, Poll poll, Factory factory) {
        PersistentT3Map staged;
        staged.root_ = sorted_nodes(cursor, count, poll, factory);
        return staged;
    }
    std::size_t size() const { return root_ ? root_->count : 0; }
    std::size_t rank(const Key& key) const {
        const Node* node = root_.get();
        std::size_t row = 0;
        while (node) {
            if (key < node->entry.first) node = node->left.get();
            else if (node->entry.first < key) {
                row += 1 + (node->left ? node->left->count : 0);
                node = node->right.get();
            } else {
                return row + (node->left ? node->left->count : 0);
            }
        }
        throw std::out_of_range("mutable T3 incidence snapshot lost a triangle");
    }
    class ConstIterator {
        const Node* node_;
    public:
        explicit ConstIterator(const Node* value) : node_(value) {}
        const std::pair<const Key, Members>* operator->() const {
            return &node_->entry;
        }
        bool operator==(const ConstIterator& other) const {
            return node_ == other.node_;
        }
        bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }
    };
    ConstIterator find(const Key& key) const {
        const Node* node = root_.get();
        while (node) {
            if (key < node->entry.first) node = node->left.get();
            else if (node->entry.first < key) node = node->right.get();
            else return ConstIterator(node);
        }
        return end();
    }
    ConstIterator end() const { return ConstIterator(nullptr); }
    void assign(const Key& key, const Members& value) {
        // A failed allocation leaves this root and every prior root unchanged.
        Root staged = put(root_, key, value);
        root_ = std::move(staged);
    }
    void erase(const Key& key) {
        Root staged = remove(root_, key);
        root_ = std::move(staged);
    }
};

using PersistentT3Memberships = PersistentT3Map<
    Edge, std::shared_ptr<const std::vector<Triangle>>>;

class T3RowLookup {
    using CellSet = PersistentT3Map<Triangle, bool>;
    CellSet ordered_cells_;
    CellSet unoriented_cells_;
    std::map<Triangle, std::size_t> fallback_rows_;
    bool ordered_ = true;

    static Triangle identity(Triangle cell) {
        std::sort(cell.begin(), cell.end());
        return cell;
    }

    template<class Poll>
    static void sorted_copy(const std::vector<Triangle>& source,
                            std::vector<Triangle>& target, Poll& poll) {
        target.reserve(source.size());
        for (const auto& cell : source) {
            poll();
            target.push_back(cell);
        }
        std::sort(target.begin(), target.end(), [&](const Triangle& a, const Triangle& b) {
            poll();
            return a < b;
        });
    }

    template<class Poll>
    static void differences(const std::vector<Triangle>& cells, bool ordered,
                            const T3RowLookup* previous,
                            const std::vector<Triangle>* previous_cells,
                            std::vector<Triangle>& removed,
                            std::vector<Triangle>& added, Poll& poll) {
        std::vector<Triangle> old_copy, new_copy;
        const std::vector<Triangle>* old = &old_copy;
        const std::vector<Triangle>* current = &cells;
        if (!ordered) {
            sorted_copy(cells, new_copy, poll);
            current = &new_copy;
        }
        if (previous != nullptr) {
            if (previous->ordered_) old = previous_cells;
            else {
                sorted_copy(*previous_cells, old_copy, poll);
                old = &old_copy;
            }
        }
        std::size_t left = 0, right = 0;
        while (left < old->size() || right < current->size()) {
            poll();
            if (right == current->size()
                    || (left < old->size() && (*old)[left] < (*current)[right])) {
                removed.push_back((*old)[left++]);
            } else if (left == old->size() || (*current)[right] < (*old)[left]) {
                added.push_back((*current)[right++]);
            } else {
                ++left;
                ++right;
            }
        }
    }

public:
    std::size_t at(const Triangle& cell) const {
        return ordered_ ? ordered_cells_.rank(cell) : fallback_rows_.at(cell);
    }

    template<class Poll>
    static T3RowLookup build(const std::vector<Triangle>& cells,
                            const T3RowLookup* previous,
                            const std::vector<Triangle>* previous_cells,
                            std::vector<Triangle>& removed,
                            std::vector<Triangle>& added, Poll poll) {
        T3RowLookup staged;
        for (std::size_t row = 0; row < cells.size(); ++row) {
            poll();
            const Triangle key = identity(cells[row]);
            if (key[0] < 0 || key[0] == key[1] || key[1] == key[2]) {
                throw std::runtime_error("mutable T3 incidence requires distinct triangle rows");
            }
            if (row && !(cells[row - 1] < cells[row])) staged.ordered_ = false;
        }
        differences(cells, staged.ordered_, previous, previous_cells,
                    removed, added, poll);
        if (previous != nullptr && previous->ordered_ && staged.ordered_) {
            staged.ordered_cells_ = previous->ordered_cells_;
            staged.unoriented_cells_ = previous->unoriented_cells_;
            for (const Triangle& cell : removed) {
                poll();
                const Triangle key = identity(cell);
                if (staged.ordered_cells_.find(cell) == staged.ordered_cells_.end()
                        || staged.unoriented_cells_.find(key) == staged.unoriented_cells_.end()) {
                    throw std::runtime_error("mutable T3 incidence source cell binding is stale");
                }
                staged.ordered_cells_.erase(cell);
                staged.unoriented_cells_.erase(key);
            }
            for (const Triangle& cell : added) {
                poll();
                const Triangle key = identity(cell);
                if (staged.unoriented_cells_.find(key) != staged.unoriented_cells_.end()) {
                    throw std::runtime_error("mutable T3 incidence requires distinct triangle rows");
                }
                staged.ordered_cells_.assign(cell, true);
                staged.unoriented_cells_.assign(key, true);
            }
        } else {
            std::vector<Triangle> keys;
            keys.reserve(cells.size());
            for (const Triangle& cell : cells) {
                poll();
                keys.push_back(identity(cell));
            }
            std::sort(keys.begin(), keys.end(), [&](const Triangle& a, const Triangle& b) {
                poll();
                return a < b;
            });
            for (std::size_t row = 1; row < keys.size(); ++row) {
                poll();
                if (keys[row - 1] == keys[row]) {
                    throw std::runtime_error("mutable T3 incidence requires distinct triangle rows");
                }
            }
            const auto value = [](const Triangle& cell) { return std::make_pair(cell, true); };
            staged.unoriented_cells_ = CellSet::from_sorted(keys.begin(), keys.size(), poll, value);
            if (staged.ordered_) {
                staged.ordered_cells_ = CellSet::from_sorted(cells.begin(), cells.size(), poll, value);
            } else {
                for (std::size_t row = 0; row < cells.size(); ++row) {
                    poll();
                    staged.fallback_rows_.emplace(cells[row], row);
                }
            }
        }
        if (staged.unoriented_cells_.size() != cells.size()
                || (staged.ordered_ && staged.ordered_cells_.size() != cells.size())) {
            throw std::runtime_error("mutable T3 incidence cell inventory is inconsistent");
        }
        return staged;
    }
};

struct T3IncidenceSnapshot {
    std::vector<Triangle> triangles;
    T3RowLookup row_by_cell;
    PersistentT3Memberships edge_cells;
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

inline bool require_t3_connectivity(const T3IncidenceSnapshot& state, const Py_buffer& buffer, PyObject* cancellation = nullptr) {
    if (static_cast<std::size_t>(buffer.shape[0]) != state.triangles.size()) {
        PyErr_SetString(PyExc_RuntimeError, "mutable T3 incidence snapshot connectivity mismatch");
        return false;
    }
    for (Py_ssize_t row = 0; row < buffer.shape[0]; ++row) {
        if (static_cast<std::size_t>(row) % kSignalCheckInterval == 0 && !insertion_checkpoint(cancellation, "native-v2 compiled insertion binding")) return false;
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
        std::vector<Triangle> removed_cells, added_cells;
        state->row_by_cell = T3RowLookup::build(
            state->triangles, previous ? &previous->row_by_cell : nullptr,
            previous ? &previous->triangles : nullptr,
            removed_cells, added_cells, poll);
        if (previous != nullptr) {
            poll();
            state->edge_cells = previous->edge_cells;
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
        for (const Triangle& cell : removed_cells) {
            poll();
            change(cell, false);
        }
        for (const Triangle& cell : added_cells) {
            poll();
            change(cell, true);
        }
        if (previous == nullptr) {
            // The initial map is already sorted and has no empty memberships.
            // Build one balanced node per edge rather than cloning root paths.
            state->edge_cells = PersistentT3Memberships::from_sorted(
                changes.begin(), changes.size(), poll, [](const auto& item) {
                    if (item.second.empty()) {
                        throw std::runtime_error("initial T3 membership cannot be empty");
                    }
                    return std::make_pair(item.first,
                        std::make_shared<const std::vector<Triangle>>(
                            item.second.begin(), item.second.end()));
                });
        } else {
            for (const auto& item : changes) {
                poll();
                if (item.second.empty()) state->edge_cells.erase(item.first);
                else state->edge_cells.assign(item.first, std::make_shared<const std::vector<Triangle>>(item.second.begin(), item.second.end()));
            }
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
