"""Actual observer resource/frame contracts, with no geometry or API replacements."""
import importlib.util
from pathlib import Path
import sys
import pytest
from _call_observer import observe_calls


def test_reused_free_slot_cannot_reactivate_other_retired_event_callbacks():
    leaked=[];entered=[]
    def unrelated():
        result=8
        return result
    def target():return 7
    monitoring=getattr(sys,'monitoring',None)
    if monitoring is None:
        with pytest.raises(RuntimeError,match='sys.monitoring unavailable'):
            with observe_calls(target.__code__,on_start=lambda frame:entered.append(frame),backend='monitoring'):
                target()
        assert entered==[]
        return
    tool=next(i for i in range(5,-1,-1) if monitoring.get_tool(i) is None)
    monitoring.use_tool_id(tool,'retired-line-callback-regression')
    monitoring.register_callback(tool,monitoring.events.LINE,lambda *args:leaked.append(True))
    monitoring.set_local_events(tool,unrelated.__code__,monitoring.events.LINE)
    assert unrelated()==8 and leaked  # Prove the retired fixture really observed LINE.
    leaked.clear()
    monitoring.free_tool_id(tool)
    try:
        with observe_calls(target.__code__,on_start=lambda frame:entered.append(frame.f_code)) as observation:
            assert observation.tool_id==tool
            assert unrelated()==8 and target()==7
        assert entered==[target.__code__] and leaked==[]
    finally:
        # The intentionally retired pre-3.14 event is OUR fixture resource.
        # Do not clear a slot that another active owner has acquired.
        assert monitoring.get_tool(tool) is None
        monitoring.use_tool_id(tool,'retired-line-callback-fixture-cleanup')
        try:
            monitoring.set_local_events(tool,unrelated.__code__,0)
            monitoring.register_callback(tool,monitoring.events.LINE,None)
            monitoring.set_events(tool,0)
        finally:monitoring.free_tool_id(tool)


@pytest.mark.parametrize('backend',('auto','profile'))
def test_actual_start_return_frames_multi_code_and_no_extra_executions(backend):
    seen=[];bodies=[];value=object()
    def first(arg):bodies.append('first');return arg
    def second(arg):bodies.append('second');return None
    def started(frame):seen.append(('start',frame.f_code,frame.f_locals['arg']))
    def returned(frame,result):seen.append(('return',frame.f_code,frame.f_locals['arg'],result))
    previous=sys.getprofile()
    with observe_calls((first.__code__,second.__code__),on_start=started,on_return=returned,backend=backend):
        assert first(value) is value
        assert second(value) is None
    assert sys.getprofile() is previous
    assert seen==[('start',first.__code__,value),('return',first.__code__,value,value),
                  ('start',second.__code__,value),('return',second.__code__,value,None)]
    assert bodies==['first','second']
    first(value)
    assert len(seen)==4 and bodies==['first','second','first']


@pytest.mark.parametrize('backend',('auto','profile'))
def test_successful_none_return_is_distinct_from_exception_unwind(backend):
    seen=[];error=RuntimeError('body failure')
    def normal():return None
    def failing():raise error
    with observe_calls((normal.__code__,failing.__code__),on_return=lambda frame,value:seen.append((frame.f_code,value)),backend=backend):
        assert normal() is None
        with pytest.raises(RuntimeError) as caught:failing()
    assert caught.value is error and seen==[(normal.__code__,None)]


@pytest.mark.parametrize('backend',('auto','profile'))
@pytest.mark.parametrize('event',('start','return'))
def test_callback_baseexception_identity_and_ambient_restore(backend,event):
    class Original(BaseException):pass
    error=Original('observer callback');bodies=[];ambient_calls=[]
    def target():bodies.append(1);return 7
    def ambient(frame,kind,value):
        if frame.f_code is target.__code__:ambient_calls.append(kind)
    def fail(*args):raise error
    previous=sys.getprofile();sys.setprofile(ambient)
    try:
        with pytest.raises(Original) as caught:
            with observe_calls(target.__code__,on_start=fail if event=='start' else None,
                               on_return=fail if event=='return' else None,backend=backend):target()
        assert caught.value is error and sys.getprofile() is ambient
        assert bodies==([] if event=='start' else [1])
        assert ambient_calls and ambient_calls[0]=='call'
    finally:sys.setprofile(previous)


def test_nested_profile_scopes_chain_ambient_and_restore_after_failure():
    order=[];bodies=[];error=KeyboardInterrupt('nested callback')
    def target():bodies.append(1)
    def ambient(frame,event,arg):
        if frame.f_code is target.__code__ and event=='call':order.append('ambient')
    def inner(frame):order.append('inner');raise error
    previous=sys.getprofile();sys.setprofile(ambient)
    try:
        with observe_calls(target.__code__,on_start=lambda frame:order.append('outer'),backend='profile'):
            outer=sys.getprofile()
            with pytest.raises(KeyboardInterrupt) as caught:
                with observe_calls(target.__code__,on_start=inner,backend='profile'):target()
            assert caught.value is error and sys.getprofile() is outer
        assert sys.getprofile() is ambient
        assert order==['ambient','outer','inner'] and bodies==[]
    finally:sys.setprofile(previous)


def test_monitoring_scope_does_not_reset_another_active_tool():
    monitoring=getattr(sys,'monitoring',None)
    def target():return 3
    if monitoring is None:
        with pytest.raises(RuntimeError,match='sys.monitoring unavailable'):
            with observe_calls(target.__code__,on_start=lambda frame:None,backend='monitoring'):target()
        return
    free=next((i for i in range(6) if monitoring.get_tool(i) is None),None)
    if free is None:
        with pytest.raises(RuntimeError,match='no free'):
            with observe_calls(target.__code__,on_start=lambda frame:None):target()
        return
    token=object();name='ambient test tool:'+str(id(token));seen=[]
    monitoring.use_tool_id(free,name)
    monitoring.register_callback(free,monitoring.events.PY_START,lambda code,offset:seen.append(code))
    monitoring.set_local_events(free,target.__code__,monitoring.events.PY_START)
    try:
        with observe_calls(target.__code__,on_start=lambda frame:None) as owned:
            assert owned.tool_id!=free
            target()
        assert monitoring.get_tool(free)==name
        assert monitoring.get_local_events(free,target.__code__)==monitoring.events.PY_START
        target();assert seen==[target.__code__,target.__code__]
    finally:
        monitoring.set_local_events(free,target.__code__,0);monitoring.set_events(free,0)
        monitoring.register_callback(free,monitoring.events.PY_START,None);monitoring.free_tool_id(free)


def test_exhaustion_is_fail_closed_without_touching_ambient_tools():
    monitoring=getattr(sys,'monitoring',None);seen=[]
    def target():return 1
    if monitoring is None:
        with observe_calls(target.__code__,on_start=lambda frame:seen.append(frame.f_code)) as owned:target()
        assert owned.backend=='profile' and seen==[target.__code__]
        return
    before={i:(monitoring.get_tool(i),monitoring.get_events(i)) for i in range(6) if monitoring.get_tool(i) is not None}
    claimed=[];token=object()
    try:
        for tool in range(6):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool,'exhaustion test:'+str(id(token)));claimed.append(tool)
        with pytest.raises(RuntimeError,match='no free sys.monitoring tool ID'):
            with observe_calls(target.__code__,on_start=lambda frame:seen.append(1)):target()
        assert seen==[]
        assert {i:(monitoring.get_tool(i),monitoring.get_events(i)) for i in before}==before
    finally:
        for tool in claimed:
            monitoring.set_events(tool,0);monitoring.free_tool_id(tool)


def test_repeated_scope_clears_local_global_events_and_callbacks_before_free():
    monitoring=getattr(sys,'monitoring',None);seen=[]
    def target():return 2
    previous=sys.getprofile()
    for _ in range(3):
        with observe_calls(target.__code__,on_start=lambda frame:seen.append('start'),
                           on_return=lambda frame,value:seen.append('return')) as owned:target()
        if monitoring is not None:
            tool=owned.tool_id;assert monitoring.get_tool(tool) is None
            monitoring.use_tool_id(tool,'verify empty retired observer')
            try:
                assert monitoring.get_events(tool)==0 and monitoring.get_local_events(tool,target.__code__)==0
                assert monitoring.register_callback(tool,monitoring.events.PY_START,None) is None
                assert monitoring.register_callback(tool,monitoring.events.PY_RETURN,None) is None
            finally:monitoring.free_tool_id(tool)
        target()
        assert sys.getprofile() is previous
    assert seen==['start','return']*3


def test_importlib_copies_can_nest_without_shared_slot_state():
    path=Path(__file__).with_name('_call_observer.py')
    spec=importlib.util.spec_from_file_location('_independent_call_observer',path)
    copied=importlib.util.module_from_spec(spec);spec.loader.exec_module(copied)
    seen=[]
    def target():return 5
    with observe_calls(target.__code__,on_start=lambda frame:seen.append('outer')) as outer:
        with copied.observe_calls(target.__code__,on_start=lambda frame:seen.append('inner')) as inner:
            if outer.backend=='monitoring':assert outer.tool_id!=inner.tool_id
            target()
    assert sorted(seen)==['inner','outer']
    target();assert len(seen)==2


def test_cprofile_ambient_is_preserved_or_refused_if_not_chainable():
    import cProfile
    seen=[]
    def target():return None
    previous=sys.getprofile();profiler=cProfile.Profile();profiler.enable()
    try:
        ambient=sys.getprofile()
        if ambient is not None and not callable(ambient):
            with pytest.raises(RuntimeError,match='ambient profile hook cannot be chained'):
                with observe_calls(target.__code__,on_start=lambda frame:seen.append(1),backend='profile'):target()
            assert seen==[]
        else:
            # Modern cProfile owns a monitoring tool instead of this slot.
            with observe_calls(target.__code__,on_start=lambda frame:seen.append(frame.f_code),backend='profile'):target()
            assert seen==[target.__code__]
        assert sys.getprofile() is ambient
        target()
        assert any(entry.code is target.__code__ for entry in profiler.getstats())
    finally:profiler.disable();sys.setprofile(previous)


def test_generator_events_are_not_misreported_as_ordinary_function_returns():
    def generator():yield 1
    previous=sys.getprofile()
    with pytest.raises(ValueError,match='ordinary Python functions'):
        with observe_calls(generator.__code__,on_start=lambda frame:None):list(generator())
    assert sys.getprofile() is previous
