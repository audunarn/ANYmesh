"""The installed smoke's triangle assertion must request triangle generation."""
import ast
from pathlib import Path
import runpy

import anygeometry as ag
import anymesher as am


def test_installed_smoke_native_request_produces_triangles_without_recombination():
    path = Path(__file__).resolve().parents[1] / 'tools/release_wheel_smoke.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    call = next(node.value for node in main.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'native')
    assert isinstance(call, ast.Call)
    assert ast.unparse(call.func) == 'am.generate_hybrid_mesh_result'
    request = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
               if keyword.arg in ('target_size', 'strategy', 'recombine')}
    # Execute the smoke's real geometry/request. This focused source test uses
    # the Python native backend; installed CI retains its C++ backend gate.
    geometry, face = runpy.run_path(str(path))['_quad']()
    before = ag.to_dict(geometry)
    result = am.generate_hybrid_mesh_result(geometry, native_backend='python', **request)
    assert result.strategy_by_face == {face: 'native'}
    assert result.mesh.tris
    assert not result.mesh.quads
    assert request['recombine'] is False
    assert ag.to_dict(geometry) == before
