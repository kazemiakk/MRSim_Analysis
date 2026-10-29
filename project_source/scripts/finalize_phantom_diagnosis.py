"""Finish E048 documentation/evidence checks and bind the local handoff."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import BASE, ROOT


def finalize():
    target = BASE/'FINAL_MANIFEST_v2.json'
    check_path = BASE/'final_checks_v2.json'
    site = BASE/'docs_site_v2'
    if target.exists() or check_path.exists() or site.exists():
        raise FileExistsError('E048 completion outputs already exist; choose a new version')
    validation = json.loads((BASE/'validation_v1.json').read_text())
    if validation['status'] != 'PASS_INDEPENDENT_ARITHMETIC_AND_PROVENANCE':
        raise ValueError('Independent validation missing')
    suites = list(ET.parse(BASE/'local_tests_v1.xml').getroot().iter('testsuite'))
    test_counts = {key:sum(int(s.attrib.get(key,0)) for s in suites)
                   for key in ('tests','failures','errors','skipped')}
    if test_counts != dict(tests=367,failures=0,errors=0,skipped=0):
        raise ValueError('Unexpected local test result: '+str(test_counts))
    commands = [
        [sys.executable,'paper/verify_manuscript_claims.py'],
        [str(ROOT/'outputs/upstream_migration_e046/docs_env/Scripts/python.exe'),
         '-m','mkdocs','build','--strict','--site-dir',str(site)],
    ]
    results = []
    for command in commands:
        run = subprocess.run(command,cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True)
        results.append(dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        print(json.dumps(results[-1]),flush=True)
        if run.returncode:
            raise RuntimeError('Final check failed; no completion manifest published')
    manuscript_match = re.search(r'PASS: (\d+) manuscript evidence checks',results[0]['stdout'])
    if not manuscript_match:
        raise ValueError('Manuscript check count not reported')
    manuscript_count = int(manuscript_match.group(1))
    check_path.write_text(json.dumps(dict(experiment='E-048',tests=test_counts,
                         deselected=3,independent_checks=validation['checks'],commands=results),indent=2)+'\n',encoding='utf-8')
    paths = [BASE/name for name in ('README.md','PROTOCOL.md','ATTEMPTS.md','EXECUTION.md',
                                   'validation_v1.json','local_tests_v1.xml','final_checks_v2.json',
                                   'FINAL_MANIFEST.json','final_checks_v1.json','finalize_phantom_diagnosis_frozen_v1.py')]
    for folder in ('basic_v1','reference_v1','local_v1','kernel_v1','kernel_v2','figures_v1'):
        paths.extend(p for p in (BASE/folder).iterdir() if p.is_file())
    paths.extend(ROOT/name for name in (
        'scripts/diagnose_phantom_signal.py','scripts/audit_phantom_reference.py',
        'scripts/analyze_phantom_local.py','scripts/diagnose_phantom_encoding_kernel.py',
        'scripts/plot_phantom_diagnosis.py','scripts/validate_phantom_diagnosis.py',
        'scripts/finalize_phantom_diagnosis.py','tests/test_phantom_diagnosis.py',
        'docs/governance/Project.md','docs/governance/Steps.md',
        'docs/governance/Experiments.md','docs/governance/Decisions.md','paper/EVIDENCE_TRACEABILITY.md'))
    result = dict(experiment='E-048',status='COMPLETE_CHECKED_SAVED_DATA_DIAGNOSIS',
                  created_utc=datetime.now(timezone.utc).isoformat(),independent_checks=validation['checks'],
                  local_tests_passed=367,local_tests_deselected=3,manuscript_checks_passed=manuscript_count,
                  strict_docs_build_passed=True,new_acquisition=False,new_1m_run=False,
                  original_scientific_inputs_preserved=True,manuscript_accuracy_promotion=False,
                  visually_reviewed_pngs=[str(p.relative_to(BASE)) for p in sorted((BASE/'figures_v1').glob('*.png'))],
                  PDFs_rendered=False,
                  supersedes='FINAL_MANIFEST.json: initial metadata hardcoded the pre-edit 196 manuscript count; v2 records actual final verifier count dynamically. Scientific fields unchanged.',
                  inputs_and_outputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],bound_files=len(result['inputs_and_outputs']))))


if __name__=='__main__':
    finalize()
