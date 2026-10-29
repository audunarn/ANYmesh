"""Alias-preserving rollback for caller-owned mesh and registry objects.

Geometry models and opaque geometry receipts remain external owner authority;
this helper neither repairs nor manufactures their verification state.
"""
import numpy as np


class PublicationSnapshot:
    def __init__(self, *roots):
        self.records = []
        seen = set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, np.ndarray):
                self.records.append(('array', value, (value.copy(), value.flags.writeable, value.strides)))
                if isinstance(value.base, np.ndarray):
                    visit(value.base)
                if value.dtype.hasobject:
                    for item in value.flat:
                        visit(item)
            elif type(value) is dict:
                entries = tuple(value.items())
                self.records.append(('dict', value, entries))
                for key, item in entries:
                    visit(key)
                    visit(item)
            elif type(value) is list:
                entries = tuple(value)
                self.records.append(('list', value, entries))
                for item in entries:
                    visit(item)
            elif type(value) is set:
                self.records.append(('set', value, frozenset(value)))
                for item in value:
                    visit(item)
            elif type(value) in (tuple, frozenset):
                for item in value:
                    visit(item)
            elif type(value).__module__.startswith('anymesher.'):
                attributes = getattr(value, '__dict__', None)
                slots = {}
                for cls in type(value).__mro__:
                    names = getattr(cls, '__slots__', ())
                    if isinstance(names, str):
                        names = (names,)
                    for name in names:
                        if name not in ('__dict__', '__weakref__') and hasattr(value, name):
                            slots[name] = getattr(value, name)
                self.records.append(('object', value, (attributes, slots)))
                if attributes is not None:
                    visit(attributes)
                for item in slots.values():
                    visit(item)
        for root in roots:
            visit(root)

    def changed(self):
        for kind, value, saved in self.records:
            if kind == 'array':
                array, writable, strides = saved
                if (value.dtype != array.dtype or value.shape != array.shape
                        or value.strides != strides
                        or value.tobytes() != array.tobytes()
                        or value.flags.writeable != writable):
                    return True
            elif kind == 'dict':
                current = tuple(value.items())
                if len(current) != len(saved) or any(a is not x or b is not y
                        for (a,b),(x,y) in zip(current,saved)):
                    return True
            elif kind == 'list':
                if len(value) != len(saved) or any(a is not b for a,b in zip(value,saved)):
                    return True
            elif kind == 'set':
                if value != saved:
                    return True
            else:
                attributes, slots = saved
                if (getattr(value, '__dict__', None) is not attributes
                        or any(getattr(value, name, None) is not item for name,item in slots.items())):
                    return True
        return False

    def restore(self):
        # DFS traversal order is not array dependency order: another field may
        # visit a shared base before its view. Build that dependency explicitly.
        arrays = {id(value): (value, saved) for kind,value,saved in self.records
                  if kind == 'array'}
        ordered, visited = [], set()
        def order(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value.base, np.ndarray) and id(value.base) in arrays:
                order(value.base)
            ordered.append(arrays[id(value)])
        for value, _saved in arrays.values():
            order(value)
        # Header changes need no writable data buffer. In particular an
        # immutable bytes-backed view can legally change shape/dtype metadata.
        # Restore strides before logical comparison: copying through a mutated
        # zero stride would write several saved values to the same address.
        for value, (array, _writable, strides) in ordered:
            if (value.dtype != array.dtype or value.shape != array.shape
                    or value.strides != strides):
                import warnings
                from math import gcd
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    if value.dtype != array.dtype:
                        old_size, new_size = value.dtype.itemsize, array.dtype.itemsize
                        if old_size != new_size and old_size and new_size:
                            # A destroyed last-axis stride prevents a larger
                            # dtype setter. Normalize a bounds-checked header
                            # over one common element-width, without moving
                            # the data pointer or requesting a writable buffer.
                            width = new_size // gcd(old_size, new_size)
                            value.strides = (0,)*value.ndim
                            value.shape = (value.size//width, width)
                            value.strides = (0, old_size)
                        value.dtype = array.dtype
                    if value.shape != array.shape:
                        if value.size == array.size:
                            value.strides = (0,)*value.ndim
                            value.shape = array.shape
                        else:
                            value.resize(array.shape, refcheck=False)
                    value.strides = strides
        writable_needed = set()
        for value, (array, writable, _strides) in ordered:
            if (value.tobytes() != array.tobytes()
                    or writable and not value.flags.writeable):
                current = value
                while isinstance(current, np.ndarray) and id(current) in arrays:
                    writable_needed.add(id(current))
                    current = current.base
        # Keep every required base writable through the entire value pass.
        for value, _saved in ordered:
            if id(value) in writable_needed:
                value.setflags(write=True)
        for value, (array, _writable, _strides) in ordered:
            if value.tobytes() != array.tobytes():
                np.copyto(value, array)
        # Views first: an originally writable view can legitimately have a
        # base whose flag was later disabled. Recreate that relationship last.
        for value, (_array, writable, _strides) in reversed(ordered):
            if value.flags.writeable != writable:
                value.setflags(write=writable)
        for kind, value, saved in reversed(self.records):
            if kind == 'array':
                continue
            elif kind == 'dict':
                value.clear()
                value.update(saved)
            elif kind == 'list':
                value[:] = saved
            elif kind == 'set':
                value.clear()
                value.update(saved)
            else:
                attributes, slots = saved
                if attributes is not None:
                    object.__setattr__(value, '__dict__', attributes)
                for name, item in slots.items():
                    object.__setattr__(value, name, item)
