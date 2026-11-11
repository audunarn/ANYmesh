"""Complete the base-import claim separately from explicit ANYsolid consumption.

No mesher or FE calls. Uses the existing installed-matrix supervisor and budget.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def matrix_tool():
    path = Path(__file__).with_name('workflow2_installed_matrix.py')
    spec = importlib.util.spec_from_file_location('installed_import_matrix',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_base(cell):
    matrix = matrix_tool()
    directory = matrix.ROOT/'cells'/cell
    output = directory/'base-import.json'
    if output.exists():
        raise FileExistsError(output)
    import anygeometry
    import anymesher
    # Inspect before any explicit consumer/backend imports.
    forbidden = ('anysolid','anysolver','skfem','pytetwild','pyvista','vtk','ngsolve')
    loaded = sorted(name for name in sys.modules
                    if any(name == item or name.startswith(item+'.') for item in forbidden))
    assert not loaded, loaded
    origins = {}
    for module in (anygeometry,anymesher):
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(directory/'venv'), origin
        origins[module.__name__] = str(origin)
    result = dict(status='PASS',cell=cell,python=sys.version,executable=sys.executable,
                  origins=origins,forbidden_modules_loaded=loaded,native_calls=0,
                  explicit_consumer_imports=False)
    with output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--cell')
    args = parser.parse_args()
    if args.child:
        check_base(args.cell)
        return
    matrix = matrix_tool()
    manifest = matrix.read(matrix.ROOT/'manifest.json')
    for cell in manifest['cells']:
        assert matrix.read(matrix.ROOT/'cells'/cell/'result.json')['status']=='PASS'
    platform = 'windows' if os.name=='nt' else 'linux'
    for cell in (name for name in manifest['cells'] if name.startswith(platform)):
        directory = matrix.ROOT/'cells'/cell
        name = 'root-'+cell+'-lazy-base-import'
        prior = matrix.ROOT/'processes'/name/'result.json'
        if prior.exists():
            receipt = matrix.read(prior)
            assert receipt['status']=='completed' and receipt.get('exit_code')==0
            assert receipt.get('cleanup',{}).get('verified')
            assert matrix.read(directory/'base-import.json')['status']=='PASS'
            continue
        python = directory/('venv/Scripts/python.exe' if os.name=='nt' else 'venv/bin/python')
        matrix.supervise(name,[python,'-I',Path(__file__).resolve(),'--child','--cell',cell],directory,15)


if __name__=='__main__':
    main()
