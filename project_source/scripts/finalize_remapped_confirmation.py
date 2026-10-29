"""Bind the completed E047 handoff and verify transitive runtime/helper identities."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from scripts.build_aorta_inlet_picker import sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/remapped_confirmation_e047'


def finalize():
    target=BASE/'FINAL_MANIFEST.json'
    if target.exists():raise FileExistsError(target)
    validation=json.loads((BASE/'validation_v1.json').read_text())
    if validation['status']!='PASS_INDEPENDENT_E047_CHECKS':raise ValueError('Validation missing')
    checks=[]
    for case in ('aorta','phantom'):
        run=tomllib.loads((BASE/f'{case}_mapped_v1/control_report.toml').read_text())
        if run['manifest_sha256']!=sha(ROOT/'koma_migration_e046/Manifest.toml'):
            raise ValueError('Runtime manifest mismatch')
        checks.append(case+':runtime_manifest')
        for helper,identity in run['helper_sha256'].items():
            if sha(ROOT/'koma_migration_e046'/helper)!=identity:raise ValueError('Helper changed')
            checks.append(case+':'+helper)
        for e,identity in run['result_sha256'].items():
            if sha(BASE/f'{case}_mapped_v1/{e.lower()}_result.h5')!=identity:raise ValueError('Encoding changed')
            checks.append(case+':encoding:'+e)
    paths=[BASE/name for name in ('README.md','PROTOCOL.md','ATTEMPTS.md','VALIDATION_COMMANDS.md',
                                'source_registration.json','validation_v1.json','local_tests_v1.xml')]
    for case in ('aorta','phantom'):
        for suffix in ('inputs_v1','map_v1','mapped_v1','analysis_v1'):
            paths.extend(p for p in (BASE/f'{case}_{suffix}').iterdir() if p.is_file())
    paths.extend(p for p in (BASE/'regional_v1').iterdir() if p.is_file())
    paths.extend(ROOT/name for name in ('koma_migration_e046/export_100k_e047.jl',
        'koma_migration_e046/run_100k_e047.jl','scripts/prepare_remapped_confirmation.py',
        'scripts/analyze_remapped_confirmation.py','scripts/validate_remapped_confirmation.py',
        'scripts/summarize_remapped_confirmation.py','scripts/finalize_remapped_confirmation.py',
        'tests/test_remapped_confirmation.py','docs/governance/Project.md','docs/governance/Steps.md',
        'docs/governance/Experiments.md','docs/governance/Decisions.md','paper/EVIDENCE_TRACEABILITY.md'))
    result=dict(experiment='E-047',status='COMPLETE_CHECKED_100K_CONFIRMATIONS',created_utc=datetime.now(timezone.utc).isoformat(),
                independent_checks=validation['checks_passed'],additional_runtime_binding_checks=checks,
                visually_reviewed_pngs=[f'{c}_analysis_v1/{c}_{n}.png' for c in ('aorta','phantom') for n in ('confirmation','velocity_slices')],
                pdfs_rendered=False,new_1m_run=False,physical_gpu_1_released=True,
                inputs_and_outputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],bound_files=len(result['inputs_and_outputs']),additional_binding_checks=len(checks))))


if __name__=='__main__':finalize()
