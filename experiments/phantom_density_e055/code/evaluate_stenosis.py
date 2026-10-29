"""Read-only, source-bound review of the completed E055 stenosis comparison."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

repo=Path('D:/MRSim/MRSIM/MRSimTracks')
run=Path('D:/MRSim/outputs/phantom_density_e055')
analysis=run/'analysis_v2'
out=run/'evaluation_v1'
if out.exists():raise FileExistsError(out)
report_path=analysis/'report.json'
report=json.loads(report_path.read_text())
receipt=json.loads((analysis/'validation.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(report_path)==receipt['report_sha256']
assert receipt['status']=='PASS_INDEPENDENT_300K_MASK_ERROR_AND_FLOW_CHECKS'
assert report['one_million_decision']['new_1m_started'] is False
arms=['baseline100k','candidate300k']
getrow=lambda region,arm,support:next(r for r in report['regions'] if r['threshold']==.05 and r['region']==region and r['arm']==arm and r['support']==support)
getflow=lambda section,arm:next(r for r in report['sections'] if r['section']==section and r['threshold']==.05 and r['arm']==arm and r['support']=='paired')
checks=3
# Recompute the consequential peak profile directly from saved 3D fields,
# rather than treating the serialized profile arrays as independent evidence.
reference_path=repo/'outputs/phantom_diagnosis_e048/reference_v1/reference_fields.h5'
with h5py.File(reference_path) as ref,h5py.File(analysis/'paired_fields.h5') as f:
    fraction=ref['fraction_fft'][:]
    for profile in report['profiles']:
        transverse=profile['orientation']=='transverse'
        idx=getflow('throat',arms[0])['index']
        spatial=(16,slice(None),idx) if transverse else (16,32,slice(None))
        b=profile['bin']-1
        eligible=fraction[spatial]>0
        keep=f['paired_fixed5_mask'][:][spatial]&eligible
        source=ref['fft_adc'][(b,)+spatial+(0,)]
        assert np.isfinite(source[eligible]).all();checks+=1
        assert int((~np.isfinite(source)).sum())==profile['CFD_unavailable_outside_geometry_count'];checks+=1
        assert np.array_equal(keep,np.asarray(profile['paired_retained']));checks+=1
        for arm in arms:
            values=f[arm][(b,)+spatial+(0,)]
            computed=float(np.sqrt(np.mean((values[keep].astype(float)-source[keep].astype(float))**2))) if keep.any() else None
            if computed is None:assert profile[arm+'_paired_RMSE_m_s'] is None
            else:np.testing.assert_allclose(computed,profile[arm+'_paired_RMSE_m_s'],atol=2e-7,rtol=0)
            checks+=1
rows=[]
for label,unit,values in [
    ('Own whole-volume coverage','%', [100*getrow('whole',a,'own')['coverage_fraction'] for a in arms]),
    ('Own throat-volume coverage','%', [100*getrow('throat',a,'own')['coverage_fraction'] for a in arms]),
    ('Paired whole-volume vector RMSE','m/s',[getrow('whole',a,'paired')['metrics']['vector_rmse_m_s'] for a in arms]),
    ('Paired throat vector RMSE','m/s',[getrow('throat',a,'paired')['metrics']['vector_rmse_m_s'] for a in arms]),
    ('Paired upstream flow RMSE','mL/s',[getflow('upstream',a)['metrics']['rmse_ml_s'] for a in arms]),
    ('Paired throat flow RMSE','mL/s',[getflow('throat',a)['metrics']['rmse_ml_s'] for a in arms]),
    ('Paired throat cycle-volume error','%',[100*getflow('throat',a)['metrics']['cycle_volume_error_fraction'] for a in arms]),
]:
    rows.append(dict(metric=label,unit=unit,baseline100k=values[0],candidate300k=values[1],
        relative_change_fraction=values[1]/values[0]-1 if values[0] else None))
evaluation=dict(status='NEEDS_REVISION_FOR_QUANTITATIVE_STENOSIS_ACCURACY',
    created_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(report_path),
    evidence_class='Same-source internal consistency; not independent physical validation',
    rows=rows,additional_profile_checks=checks,
    paired_whole_volume_coverage=getrow('whole',arms[0],'paired')['coverage_fraction'],
    paired_throat_volume_coverage=getrow('throat',arms[0],'paired')['coverage_fraction'],
    paired_throat_section_coverage=getflow('throat',arms[0])['coverage_fraction'],
    paired_upstream_section_coverage=getflow('upstream',arms[0])['coverage_fraction'],
    throat_reference_RMS_normalized_error_300k=getrow('throat',arms[1],'paired')['metrics']['vector_nrmse_reference_rms'],
    peak_transverse_profile_RMSE_300k_m_s=next(p['candidate300k_paired_RMSE_m_s'] for p in report['profiles'] if p['phase']=='peak' and p['orientation']=='transverse'),
    scaleup_decision=report['one_million_decision'],
    interpretation='Coverage improves substantially; paired throat velocity error improves modestly, same-area throat flow remains low and upstream flow RMSE slightly increases. The numerical scale-up gate fails. Do not infer full-lumen or independent physical accuracy.')
out.mkdir()
(out/'evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n',encoding='utf-8')
text='# Stenosis assessment: improved, but not yet quantitatively satisfactory\n\n'
text+='The 300k acquisition and source-bound comparison are complete. The simulation data were not changed or rerun. The first report export failed only because outside-lumen CFD profile gaps were NaN; analysis_v2 records those gaps as null after checking that eligible-lumen values are finite. The failed first attempt remains preserved.\n\n'
text+='| Metric | 100k | 300k |\n|---|---:|---:|\n'
for row in rows:text+=f"| {row['metric']} ({row['unit']}) | {row['baseline100k']:.4f} | {row['candidate300k']:.4f} |\n"
text+='\nCoverage rows use each arm\'s own fixed 5% all-bin/all-encoding mask. Error rows use identical paired support: 14.80% of whole numerical lumen volume and 30.79% of throat volume. Paired flow uses 17.98% of the upstream section and 42.17% of the throat section. These are not full-lumen flow errors.\n\n'
text+='At the CFD-selected peak bin4, the retained transverse throat profile still has 0.869m/s RMSE. All-bin throat vector RMSE is about20.1% of reference RMS; cycle transport is21.5% low on the identical partial area. Low bin15 overlaps the declared synthetic closure tail and is not independent native low-flow validation.\n\n'
text+='Verdict: useful technical progress, not satisfactory for a claim of accurate stenosis hemodynamics. Keep1M on hold: coverage and local velocity improved, but the preregistered no-worsened-upstream-flow gate failed. A 1.9% upstream RMSE rise alone does not establish a significant degradation; seed variability remains unmeasured. More particles have not yet demonstrated resolution of the local bias.\n\n'
text+='Review peak-throat residuals and the small upstream regression using saved data before a new acquisition. No threshold relaxation, fitted waveform, imputation or new1M. This is same-source internal consistency, not independent physical validation.\n'
(out/'EVALUATION.md').write_text(text,encoding='utf-8')
(out/'MANIFEST.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in [Path(__file__),report_path,analysis/'validation.json',analysis/'MANIFEST.json']},outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
print(json.dumps(evaluation),flush=True)
