"""Write a source-bound E056 laboratory assessment after completed analysis."""
from pathlib import Path
import csv
import hashlib
import io
import json
from datetime import datetime,timezone

root=Path('D:/MRSim/outputs/acquisition_physics_e056')
target=root/'assessment_v1';stage=target.with_name(target.name+'.building')
if target.exists() or stage.exists():raise FileExistsError(target)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report_path=root/'analysis_v1/report.json';sensitivity_path=root/'analysis_sensitivity_v1/report.json'
r=json.loads(report_path.read_text(encoding='utf-8'));s=json.loads(sensitivity_path.read_text(encoding='utf-8'))
prefix=json.loads((root/'prefix_v1/independent_gate_v2.json').read_text(encoding='utf-8'))
response_path=root/'sequence_response_v1/report.json'
response=json.loads(response_path.read_text(encoding='utf-8'))
assert len(r['comparisons'])==9 and len(r['numerical_checks'])==36
assert len(s['rows'])==36 and not r['new_1m_started'] and not s['new_1m']
manifest=json.loads((root/'analysis_v1/MANIFEST.json').read_text(encoding='utf-8'))
assert sha(report_path)==manifest['outputs']['report.json']['sha256']
sm=json.loads((root/'analysis_sensitivity_v1/MANIFEST.json').read_text(encoding='utf-8'))
assert sha(sensitivity_path)==sm['report_sha256']
numerical=r['status']=='PASS_INDEPENDENT_REPEATED_RF_COMPLEX_MODEL'
eligible=[x for x in r['mechanism_tests'] if x['model']=='parabolic_osc' and x['predefined_improvement_target_pass']]
fmt=lambda v,scale=1,digits=4:'unavailable' if v is None else f'{v*scale:.{digits}f}'
lines=['# E056 acquisition-physics laboratory assessment','',
       f"Completed UTC: {datetime.now(timezone.utc).isoformat()}",'',
       '## Outcome','',f"Independent numerical gate: **{r['status']}**.",'',
       'This is an analytic, synthetic, full-3D numerical mechanism control. It is not a new CFD-driven phantom or aorta acquisition, independent physical validation, or evidence that the stenosis accuracy problem has been fixed. No 1M run was started.','',
       '## Primary paired comparisons','',
       'Each error uses the same all-arm paired predefined interior for that model. Own retained geometry coverage has a different denominator and is reported separately. The unchanged primary support is 5%, fixed across all bins and encodings. Flow uses the identical retained part of the x=0 section, not the full lumen.','',
       '| Motion | Arm | Own coverage (%) | Paired core (%) | Axial RMSE (m/s) | Vector RMSE (m/s) | Partial-flow bias (%) |',
       '|---|---|---:|---:|---:|---:|---:|']
csv_rows=[]
for row in r['comparisons']:
    e=row['Koma_vs_instantaneous_analytic'];q=row['paired_partial_section_flow']
    bias=None if q is None else q['relative_mean_bias']
    lines.append(f"| {row['model']} | {row['arm']} | {fmt(row['own_geometry_coverage'],100,2)} | {fmt(row['identical_paired_core_coverage'],100,2)} | {fmt(e['axial_rmse_m_s'])} | {fmt(e['vector_rmse_m_s'])} | {fmt(bias,100,2)} |")
    csv_rows.append(dict(model=row['model'],arm=row['arm'],own_geometry_coverage=row['own_geometry_coverage'],
        paired_core_coverage=row['identical_paired_core_coverage'],axial_RMSE_m_s=e['axial_rmse_m_s'],vector_RMSE_m_s=e['vector_rmse_m_s'],
        partial_section_flow_bias_fraction=bias,partial_section_area_mm2=None if q is None else q['retained_area_mm2']))
lines+=['','## Registered improvement tests','',
        'A positive result requires the independent numerical gate, at least 10% paired axial RMSE reduction and no more than 1 percentage-point loss of own coverage. This is an exploratory mechanism threshold, not clinical acceptance.','',
        '| Motion | Candidate vs baseline | Axial RMSE reduction (%) | Own coverage change (percentage points) | Pass |',
        '|---|---|---:|---:|---|']
for row in r['mechanism_tests']:
    lines.append(f"| {row['model']} | {row['arm']} | {fmt(row['paired_axial_RMSE_reduction_fraction'],100,2)} | {fmt(row['own_coverage_change_fraction'],100,2)} | {row['predefined_improvement_target_pass']} |")
lines+=['','## Interpretation and next decision','']
if not numerical:
    lines+=['Full numerical agreement has not cleared its registered gate. Do not interpret the arm differences as demonstrated acquisition physics. Resolve the reported numerical discrepancy on these saved controls before a CFD-driven confirmation. Keep 1M on hold.']
elif eligible:
    candidates=', '.join(x['arm'] for x in eligible)
    lines+=[f'The sheared accelerating control meets the primary improvement target for: {candidates}. This supports a small CFD-driven paired confirmation using the same source, precision, spin set, actual timing reference, and fixed support. It does not authorize or justify an automatic 1M run. Check local throat profiles, identical-area flow and mask sensitivities in that confirmation.']
else:
    lines+=['Neither candidate meets the complete registered improvement target in the sheared accelerating control. A successful forward prediction of a biased image is not a corrected instantaneous velocity field. Do not spend on another 1M run on the basis of these acquisition changes. Use the saved complex fields to distinguish encoding response, finite-resolution mixing and loss of support before the next small CFD-driven test.']
lines+=['','## Numerical evidence and limitations','',
        f"- Maximum raw complex-signal relative L2 across 36 encodings: {fmt(max(x['raw_relative_l2'] for x in r['numerical_checks']),100,4)}%.",
        f"- Maximum complex-image relative L2: {fmt(max(x['image_relative_l2'] for x in r['numerical_checks']),100,4)}%.",
        f"- Maximum independent FFT identity error: {max(x['independent_fft_relative_max'] for x in r['numerical_checks']):.6g}.",
        f"- Registered 5/10/15/20% support sensitivities: {s['status']}, {s['checks']} independent recomputation/provenance checks. These are not optimized masks.",
        '- Fine-RF prefix passes; coarse-RF negative/sensitivity controls fail the unchanged 1% signal limit. No retrospective cause or correction is established for historical E050/E055.',
        '- RF-phase export was corrected from the unchanged Pulseq RF tables, not fitted to MRI output. The original failed reference and reconstruction attempts are preserved.',
        '- Arms: 350 Hz/pixel at TE 2.74 ms; 700 Hz/pixel at TE 2.74 ms; 700 Hz/pixel at TE 2.1 ms. The original 2.0 ms request was rejected before signals by hardware/timing feasibility.',
        '- All 15 samples are stroboscopic control observations at the same periodic-motion phase, not a physiological flow waveform. The shorter-echo arm has its own actual instantaneous reference time.',
        '- Finite-RF/repeated-T1 numerical model shares exported physical sequence inputs but does not use Koma simulation/sort/FFT helpers. Agreement is independent numerical implementation verification, not scanner or patient validation.',
        '- Controls remove inlet, wall-recycling and CFD interpolation uncertainty; they cannot determine how much each contributes to the real stenosis error.',
        '- These noiseless controlled inputs do not establish an SNR benefit from higher receiver bandwidth, turbulence accuracy, wall shear stress or scanner realism.','',
        '## Unfitted acceleration timing sensitivity','',
        'Differential-gradient moments give a constant-acceleration sensitivity equivalent to 2.1785 ms before the echo for baseline and faster-readout/TE 2.74 ms, versus 1.5385 ms for TE 2.1 ms. The faster readout alone leaves this encoding timing sensitivity almost unchanged. This is a single-spin, local constant-acceleration diagnostic, not a universal alignment shift or a correction for the sheared complex image. Its independent moment reintegration passes 63 checks. See `../sequence_response_v1/report.json`.','',
        '## Files','',
        '- `../analysis_v1/report.json`: primary numerical, velocity, raw central-phase, voxel-mean sensitivity and partial-flow results.',
        '- `../analysis_v1/independent_fields.h5`: reconstructed and independent numerical fields.',
        '- `../analysis_sensitivity_v1/report.json`: all registered threshold conditions, paired errors and coverage.',
        '- `primary_comparisons.csv`: primary summary values without rounding.',
        '- `MANIFEST.json`: bound source and assessment hashes.','']
stage.mkdir()
(stage/'ASSESSMENT.md').write_text('\n'.join(lines),encoding='utf-8')
with (stage/'primary_comparisons.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
(stage/'MANIFEST.json').write_text(json.dumps(dict(experiment='E-056',
    inputs_sha256={str(p):sha(p) for p in [report_path,sensitivity_path,response_path,root/'prefix_v1/independent_gate_v2.json',Path(__file__)]},
    outputs_sha256={p.name:sha(p) for p in stage.iterdir() if p.is_file()},new_1m=False),indent=2)+'\n',encoding='utf-8')
stage.rename(target)
print(json.dumps(dict(status='COMPLETE_SOURCE_BOUND_LAB_ASSESSMENT',numerical_pass=numerical,
                      sheared_control_eligible_arms=[x['arm'] for x in eligible],new_1m=False)),flush=True)
