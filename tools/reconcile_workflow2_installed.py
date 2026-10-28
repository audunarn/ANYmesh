"""Reconcile retained installed-matrix evidence without re-executing a case."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]/'evidence/installed-matrix-20261028-a1'


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON '+value)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=reject)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,detail):
    if not value:
        raise ValueError(detail)


def main():
    manifest = read(ROOT/'manifest.json')
    artifacts = read(ROOT/'core-artifacts.json')
    for name,digest in artifacts.items():
        require(sha(ROOT/'wheels'/name)==digest,'core artifact identity '+name)
    for name,digest in manifest['source_hashes'].items():
        require(sha(ROOT/name)==digest,'frozen source identity '+name)
    for backend in manifest['backends'].values():
        require(sha(ROOT.parents[1]/backend['wheel'])==backend['wheel_sha256'],'native wheel identity')
    budget = read(ROOT/'budget.json')
    require(abs(sum(a['process_wall_seconds'] for a in budget['attempts'])-budget['used_seconds'])<1e-8,'budget sum')
    require(budget['used_seconds']<=2400,'cumulative process budget')
    require(all(a.get('cleanup',{}).get('verified') for a in budget['attempts']),'unverified process cleanup')
    cells = []
    for cell in manifest['cells']:
        directory = ROOT/'cells'/cell
        record = read(directory/'result.json')
        base = read(directory/'base-import.json')
        require(record['status']=='PASS' and record['native_calls']==1,'cell outcome '+cell)
        require((directory/'native-attempt.json').is_file(),'native attempt record '+cell)
        require(base['status']=='PASS' and not base['forbidden_modules_loaded'] and base['native_calls']==0,'base import '+cell)
        native = record['native_process']
        require(native['exit_code']==0 and native['active_owned_processes_after_cleanup']==0,'native cleanup '+cell)
        require(native['process_wall_seconds']<=120,'native deadline '+cell)
        suites = list(ET.parse(directory/'pytest.xml').getroot().iter('testsuite'))
        counts = {key:sum(int(s.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
        require(counts==dict(tests=125,failures=0,errors=0,skipped=0),'installed test counts '+cell)
        require([order['order'] for order in record['orders']]==[1,2],'P1/P2 coverage '+cell)
        checks = 0
        for order in record['orders']:
            require(order['status']=='PASS' and all(value is True for value in order['checks'].values()),'physical predicates '+cell)
            require(all(math.isfinite(value) for value in order['metrics'].values()),'physical finite metrics '+cell)
            checks += len(order['checks'])
        require(checks==22,'physical check inventory '+cell)
        elapsed = sum(a['process_wall_seconds'] for a in budget['attempts']
                      if a['name'].startswith(cell+'-') or a['name']=='root-'+cell+'-lazy-base-import')
        require(elapsed<=300,'complete cell process budget '+cell)
        cells.append(dict(cell=cell,python=record['python'],tests=counts,physical_checks=checks,
            native_calls=1,nodes=record['nodes'],cells=record['cells'],geometry=record['geometry'],
            process_wall_seconds=elapsed,native_process_wall_seconds=native['process_wall_seconds'],
            base_import='PASS',result_sha256=sha(directory/'result.json'),
            junit_sha256=sha(directory/'pytest.xml'),base_import_sha256=sha(directory/'base-import.json')))
    require(len(cells)==8,'platform inventory')
    summary = dict(status='EIGHT_INSTALLED_CORE_CELLS_PASSED_NOT_WORKFLOW_COMPLETION',
        manifest_sha256=sha(ROOT/'manifest.json'),core_artifacts=artifacts,cells=cells,
        total_installed_tests=1000,total_physical_checks=176,native_calls_used=8,native_calls_remaining=0,
        process_wall_used_seconds=budget['used_seconds'],process_wall_remaining_seconds=2400-budget['used_seconds'],
        preserved_failed_processes=[a['name'] for a in budget['attempts'] if a['status']!='completed'],
        review='Root non-author reconciliation; independent same-family Read review requested',
        evidence_limit='Initial Windows311 failed installed-selection JUnit was overwritten in the common cell path; original full stdout/receipt retained. No reconstructed XML.',
        omissions=manifest['omissions']+['provider peak-memory comparison','unregistered thin-resolution/quality acceptance'],
        source_mapping_test='Repository-only affected-test mapping excluded from installed subset; covered by source profile.',
        import_probe='Separate eight base-import checks precede explicit consumer/backend imports; original cell lazy flag alone was narrower.',
        root_import_tool_sha256=sha(Path(__file__).with_name('workflow2_installed_imports.py')),
        reconcile_tool_sha256=sha(Path(__file__)))
    output = ROOT/'reconciled-result.json'
    with output.open('x',encoding='utf-8') as handle:
        json.dump(summary,handle,indent=2,allow_nan=False)
    print(json.dumps({key:summary[key] for key in ('status','total_installed_tests','total_physical_checks','native_calls_used','process_wall_used_seconds')}))


if __name__=='__main__':
    main()
