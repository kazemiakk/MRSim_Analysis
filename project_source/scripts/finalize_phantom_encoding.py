"""Finish E049 handoff with checked commands and an immutable local manifest."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from scripts.build_aorta_inlet_picker import sha
from scripts.prepare_phantom_encoding import BASE,ROOT


def finalize():
    target=BASE/'FINAL_MANIFEST.json';site=BASE/'docs_site_v1'
    if target.exists() or site.exists() or (BASE/'final_checks_v1.json').exists():
        raise FileExistsError('Completion outputs already exist')
    v=json.loads((BASE/'validation_v1.json').read_text())
    if v['status']!='PASS_CONTROL_VALIDATION_WITH_CONFIRMED_PRODUCTION32_CLOCK_FAILURE':
        raise ValueError('Wrong validation status')
    suites=list(ET.parse(BASE/'local_tests_v1.xml').getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=377,failures=0,errors=0,skipped=0):raise ValueError(counts)
    commands=[[sys.executable,'paper/verify_manuscript_claims.py'],
              [str(ROOT/'outputs/upstream_migration_e046/docs_env/Scripts/python.exe'),
               '-m','mkdocs','build','--strict','--site-dir',str(site)]]
    results=[]
    for command in commands:
        run=subprocess.run(command,cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True)
        results.append(dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        print(json.dumps(results[-1]),flush=True)
        if run.returncode:raise RuntimeError('Final check failed')
    manuscript=int(re.search(r'PASS: (\d+) manuscript evidence checks',results[0]['stdout']).group(1))
    (BASE/'final_checks_v1.json').write_text(json.dumps(dict(tests=counts,deselected=3,commands=results),indent=2)+'\n')
    with (BASE/'constant_velocity_clock.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(v['constant_velocity_table'][0]))
        writer.writeheader();writer.writerows(v['constant_velocity_table'])
    paths=[p for p in BASE.rglob('*') if p.is_file() and 'docs_site_v1' not in p.parts]
    paths.extend(ROOT/name for name in (
        'scripts/prepare_phantom_encoding.py','scripts/analyze_phantom_encoding.py',
        'scripts/bound_phantom_adc_offset.py','scripts/audit_phantom_adc_times.py',
        'scripts/check_phantom_signal_mixing.py','scripts/validate_phantom_encoding.py',
        'scripts/finalize_phantom_encoding.py','tests/test_phantom_encoding.py',
        'koma_migration_e046/encoding_controls_e049.jl','koma_migration_e046/test_encoding_clock_e049.jl',
        'koma_migration_e046/probe_encoding_timing_e049.jl',
        'koma_migration_e046/Project.toml','koma_migration_e046/Manifest.toml',
        'docs/governance/Project.md','docs/governance/Steps.md',
        'docs/governance/Experiments.md','docs/governance/Decisions.md','paper/EVIDENCE_TRACEABILITY.md',
        'outputs/upstream_migration_e046/depot/packages/KomaMRICore/PMbeb/src/simulation/SimulatorCore.jl',
        'outputs/upstream_migration_e046/depot/packages/KomaMRICore/PMbeb/src/simulation/SimMethods/Bloch/cpu/BlochCPU.jl'))
    report=dict(experiment='E-049',status='COMPLETE_CONTROLS_WITH_DEMONSTRATED_PRODUCTION32_CLOCK_FAILURE',
        created_utc=datetime.now(timezone.utc).isoformat(),independent_checks=v['checks'],
        local_tests_passed=377,local_tests_deselected=3,manuscript_checks_passed=manuscript,
        strict_docs_build_passed=True,production32_late_constant_velocity_gate='FAIL_CONFIRMED_CPU_AND_GPU',
        mitigation='Float64 eliminates clock drift in controls; full phantom mitigation not yet verified',
        full_phantom_reacquired=False,new_1m_run=False,legacy_identities_unchanged=39,
        inputs_and_outputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],bound_files=len(report['inputs_and_outputs']),manuscript_checks=manuscript)))


if __name__=='__main__':finalize()
