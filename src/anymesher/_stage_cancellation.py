"""Mesher stage checks without changing the callback forwarded to owners."""
from contextvars import ContextVar
from functools import wraps
from .errors import MeshError

_errors = ContextVar('mesher_stage_cancellation_errors', default=None)

class _StageCancelled(MeshError):
    pass

def check_stage(callback, phase):
    if callback is None:
        return
    try:
        if callback(phase):
            raise _StageCancelled(f'{phase} cancelled')
    except BaseException as error:
        errors = _errors.get()
        if errors is not None:
            errors.append(error)
        raise

def preserve_stage_cancellation(error):
    if isinstance(error, _StageCancelled) or any(error is item for item in (_errors.get() or ())):
        raise error

def stage_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _errors.get() is not None:
            return function(*args, **kwargs)
        token = _errors.set([])
        try:
            return function(*args, **kwargs)
        finally:
            _errors.reset(token)
    return wrapped
