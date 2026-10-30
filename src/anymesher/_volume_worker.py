"""Private stdin-gated pytetwild adapter (run by absolute file path).

No alternate import route: the upstream public import must work as packaged.
"""
import sys

# Parent assigns its kill-on-close process-tree container before releasing this
# gate. Do not import optional modules or start threads before the handshake.
if __name__ == '__main__':
    if sys.stdin.readline().strip() != 'start':
        raise SystemExit(2)


def main():
    import json
    from pathlib import Path
    import traceback
    from importlib.metadata import version
    directory = Path(sys.argv[1])
    response = {}
    try:
        installed = version('pytetwild')
        if installed not in ('0.4.2', '0.4.2+anyworkflow2.1'):
            raise ImportError(f'expected pytetwild 0.4.2, found {installed}')
        import pytetwild
    except Exception as exc:
        response.update(status='unavailable', detail=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        (directory / 'response.json').write_text(json.dumps(response), encoding='utf-8')
        return 3
    try:
        import numpy as np
        request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
        options = request['options']
        with np.load(directory / 'input.npz', allow_pickle=False) as data:
            coordinates, facets = data['coordinates'], data['facets']
        result = pytetwild.tetrahedralize(coordinates, facets, **options)
        coordinates, cells = result
        # Write arrays before a terminal response; partial writes never publish.
        np.savez(directory / 'output.npz', coordinates=coordinates, cells=cells)
        response.update(status='success', version=installed, effective_options=options)
    except Exception as exc:
        response.update(status='invalid', detail=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    (directory / 'response.json').write_text(json.dumps(response), encoding='utf-8')
    return 0 if response['status'] == 'success' else 4


if __name__ == '__main__':
    raise SystemExit(main())
