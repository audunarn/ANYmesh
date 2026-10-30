"""Compact local evidence inventory; does not confer scientific acceptance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('workspace',type=Path)
    a = p.parse_args()
    evidence = a.workspace/'evidence'
    records = []
    for receipt in sorted([*evidence.glob('*/result.json'), *evidence.glob('*/process/result.json')]):
        data = json.loads(receipt.read_text(encoding='utf-8'))
        item = dict(evidence=str(receipt.parent.relative_to(evidence)),
                    status=data.get('status'),exit_code=data.get('exit_code'),
                    process_wall_seconds=data.get('process_wall_seconds'),cleanup=data.get('cleanup'))
        folder = receipt.parent.parent if receipt.parent.name=='process' else receipt.parent
        xml = folder/'pytest.xml'
        if xml.exists():
            suites = list(ET.parse(xml).getroot().iter('testsuite'))
            item['pytest'] = {key:sum(int(s.get(key,'0')) for s in suites)
                              for key in ('tests','failures','errors','skipped')}
        records.append(item)
    source = {}
    for name in ('ANYgeometry','ANYmesh'):
        repo = a.workspace/name
        paths = [*repo.joinpath('src').rglob('*.py'),repo/'pyproject.toml']
        paths += list((repo/'tests').glob('*solid*'))+list((repo/'tests').glob('*volume*'))
        for path in paths:
            if path.is_file():source[str(path.relative_to(a.workspace))]=digest(path)
    artifact = {str(path.relative_to(evidence)):digest(path)
                for path in (evidence/'core-artifacts').rglob('*') if path.is_file() and path.name.endswith(('.whl','.tar.gz'))}
    result = dict(status='PARTIAL_IMPLEMENTATION_BLOCKED_PROVIDER_FEASIBILITY',
                  python=sys.version,executable=sys.executable,
                  versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pytest','scikit-fem')},
                  source_sha256=source,artifact_sha256=artifact,checks=records,
                  scope='Local bounded development/installed checks; no provider qualification or release acceptance')
    target = evidence/'workflow2-evidence-index.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(index=str(target),receipts=len(records),source_files=len(source),artifacts=len(artifact))))


if __name__=='__main__':
    main()
