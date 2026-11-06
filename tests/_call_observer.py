"""Lexical start/successful-return observers for ordinary Python test functions.

No function rebinding, tool slot assumptions, or interpreter-wide purity claim.
The setprofile fallback observes this thread and chains a callable ambient hook.
"""
from contextlib import contextmanager
import dis
import inspect
import sys
from types import CodeType, SimpleNamespace


def _cleanup(operations, primary):
    failures=[]
    for operation in operations:
        try:operation()
        except BaseException as error:failures.append(error)
    if failures:
        if primary is None:raise failures[0]
        BaseException.add_note(primary,'test observer cleanup failed: '+type(failures[0]).__name__)


@contextmanager
def observe_calls(codes, *, on_start=None, on_return=None, name='test call observer', backend='auto'):
    """Pass the actual callee frame to callbacks; preserve original exceptions.

    Codes remain strongly held until cleanup. Generator/coroutine events are
    intentionally outside this ordinary-function test helper's contract.
    """
    if type(codes) is CodeType:codes=(codes,)
    else:codes=tuple(codes)
    if not codes or any(type(code) is not CodeType for code in codes):
        raise TypeError('observer requires actual Python code objects')
    if any(code.co_flags & (inspect.CO_GENERATOR|inspect.CO_COROUTINE|inspect.CO_ASYNC_GENERATOR) for code in codes):
        raise ValueError('observer requires ordinary Python functions')
    if on_start is None and on_return is None:raise ValueError('observer requires a callback')
    if any(callback is not None and not callable(callback) for callback in (on_start,on_return)):
        raise TypeError('observer callbacks must be callable')
    if type(name) is not str or not name:raise TypeError('observer name must be a nonempty string')
    if backend not in ('auto','monitoring','profile'):raise ValueError('unknown observer backend')
    watched={id(code):code for code in codes}
    codes=tuple(watched.values())
    monitoring=getattr(sys,'monitoring',None)
    if backend=='monitoring' and monitoring is None:raise RuntimeError('sys.monitoring unavailable')
    if monitoring is None or backend=='profile':
        previous=sys.getprofile()
        if previous is not None and not callable(previous):
            raise RuntimeError('ambient profile hook cannot be chained')
        returns={dis.opmap[key] for key in ('RETURN_VALUE','RETURN_CONST','INSTRUMENTED_RETURN_VALUE','INSTRUMENTED_RETURN_CONST') if key in dis.opmap}
        def profile(frame,event,value):
            if previous is not None:previous(frame,event,value)
            if watched.get(id(frame.f_code)) is not frame.f_code:return
            if event=='call' and on_start is not None:on_start(frame)
            elif (event=='return' and on_return is not None and frame.f_lasti>=0
                  and frame.f_lasti<len(frame.f_code.co_code)
                  and frame.f_code.co_code[frame.f_lasti] in returns):
                on_return(frame,value)
        try:
            sys.setprofile(profile)
            yield SimpleNamespace(backend='profile',tool_id=None)
        finally:
            primary=sys.exc_info()[1]
            _cleanup((lambda:sys.setprofile(previous),),primary)
        return
    start=monitoring.events.PY_START;returned=monitoring.events.PY_RETURN
    event_kinds=tuple(sorted({value for value in vars(monitoring.events).values()
        if type(value) is int and value>0 and value&(value-1)==0}))
    token=object();label=name+':'+str(id(token));tool=None
    for candidate in range(5,-1,-1):
        if monitoring.get_tool(candidate) is not None:continue
        try:monitoring.use_tool_id(candidate,label)
        except ValueError:
            if monitoring.get_tool(candidate) is not None:continue
            raise
        tool=candidate;break
    if tool is None:raise RuntimeError('no free sys.monitoring tool ID')
    # Pre-3.14 free IDs can retain callbacks for OTHER event kinds too. We
    # now own this free slot, not any other active tool. Clear its complete
    # public callback table without walking unrelated code objects.
    def entered(code,offset):
        frame=sys._getframe(1)
        if watched.get(id(code)) is code:
            if frame.f_code is not code:raise RuntimeError('observer did not receive actual callee frame')
            on_start(frame)
    def finished(code,offset,value):
        frame=sys._getframe(1)
        if watched.get(id(code)) is code:
            if frame.f_code is not code:raise RuntimeError('observer did not receive actual callee frame')
            on_return(frame,value)
    try:
        monitoring.set_events(tool,0)
        for event in event_kinds:monitoring.register_callback(tool,event,None)
        if on_start is not None:monitoring.register_callback(tool,start,entered)
        if on_return is not None:monitoring.register_callback(tool,returned,finished)
        mask=(start if on_start is not None else 0)|(returned if on_return is not None else 0)
        for code in codes:monitoring.set_local_events(tool,code,mask)
        yield SimpleNamespace(backend='monitoring',tool_id=tool)
    finally:
        primary=sys.exc_info()[1]
        if monitoring.get_tool(tool)!=label:
            _cleanup((lambda:(_ for _ in ()).throw(RuntimeError('observer tool ownership changed')),),primary)
        else:
            _cleanup(tuple(lambda code=code:monitoring.set_local_events(tool,code,0) for code in codes)+(
                lambda:monitoring.set_events(tool,0),)+tuple(
                lambda event=event:monitoring.register_callback(tool,event,None) for event in event_kinds)+(
                lambda:monitoring.free_tool_id(tool),),primary)
