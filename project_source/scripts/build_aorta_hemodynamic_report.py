"""Package reviewed E-027 results for the local shared report runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.report.read_bytes();r=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if r['experiment']!='E-027':raise ValueError('Expected E-027')
    root=Path(__file__).resolve().parents[1]; source=args.report.resolve().relative_to(root).as_posix()
    queries={}
    def add(key,rows,definition,ids):
        queries[key]={'rows':rows,'source':{'label':'E-027 original aortic CFD and preserved Koma comparison',
            'files':[source],'executedAt':r['created_utc'],
            'evidenceFlow':[{'title':'Saved numerical result','detail':source+'; SHA256 '+digest},
                            {'title':'Source and estimator','detail':r['evidence_class']},
                            {'title':'Report extraction','detail':'scripts/build_aorta_hemodynamic_report.py selects saved values and converts named display units.'}],
            'metricDefinitions':[{'label':key,'definition':definition,'componentIds':ids,'sourceLineage':[{'files':[source]}]}]},
            'methods':[{'language':'text','code':definition}]}
    add('flow',r['section_series'],'Signed +MRI-X partial section flux. Separate numerical-support components, common15% mask, voxel mean velocity times transverse numerical-support area. m3/s times1e6 gives mL/s.',
        ['flow-chart','flow-table'])
    summaries=[dict(section=x['section'],x_mm=x['x_m']*1000,paired_pixels=x['paired_pixels'],
        flow_rmse_ml_s=x['flow_error']['rmse']*1e6,correlation=x['flow_error']['pearson_r'],
        CFD_cycle_ml=x['CFD_cycle']['net_m3']*1e6,Koma_cycle_ml=x['Koma_cycle']['net_m3']*1e6,
        cycle_error_pct=100*(x['net_cycle_volume_ratio']-1),peak_error_pct=100*(x['peak_absolute_flow_ratio']-1)) for x in r['section_summary']]
    add('sections',summaries,'Cycle integral is eight nominal-bin values times period/8; signed partial-domain flow, no anatomical inlet/outlet or branch identity inferred.',
        ['section-summary','summary'])
    regions=[dict(x,energy_scaled=x['energy_per_density_m5_s2']*1e6) for x in r['regional_series']]
    add('regions',regions,'KE/rho = half the common numerical-support voxel integral of squared mean velocity. Display multiplies m5/s2 by1e6. Aortic density is unverified; no joule-valued energy inferred.',
        ['energy-chart','speed-chart','region-table','summary'])
    errors=[]
    for name in ('X','Y','Z','speed'):
        x=r['primary_pooled'][name]
        errors.append(dict(quantity=name,bias=x['bias'],mae=x['mae'],rmse=x['rmse'],nrmse_pct=x['nrmse_reference_rms']*100))
    errors.append(dict(quantity='vector',bias=None,mae=None,rmse=r['primary_pooled']['vector_rmse_m_s'],nrmse_pct=r['primary_pooled']['vector_nrmse_reference_rms']*100))
    add('velocity',errors,'Pooled common numerical-volume-weighted velocity errors across8bins. Reference RMS normalization; vector RMSE is Euclidean error, and vector bias/MAE are undefined here.', ['velocity-table','summary'])
    derivative_labels={'vorticity_magnitude_s_inv':'Vorticity magnitude (1/s)','helicity_m_s_inv2':'Helicity density (m/s²)',
        'divergence_s_inv':'Divergence (1/s)','strain_frobenius_s_inv':'Strain norm (1/s)',
        'dissipation_per_viscosity_s_inv2':'Dissipation / viscosity (1/s²)'}
    derivative_modes={'primary':'Common interior','eroded':'Extra-eroded interior','matched_4mm_smoothing':'Matched 4 mm smoothing'}
    derivative_summary=[];derivatives=[]
    for mode,item in r['derivatives'].items():
        for metric,x in item['pooled'].items():
            derivative_summary.append(dict(mode=derivative_modes[mode],metric=derivative_labels[metric],rmse=x['rmse'],nrmse_pct=x['nrmse_reference_rms']*100,stencil_voxels=item['stencil_voxels_per_bin']))
        for x in item['per_bin']:
            derivatives.append(dict(bin=x['bin'],mode=derivative_modes[mode],metric=derivative_labels[x['metric']],rmse=x['rmse'],nrmse_pct=x['nrmse_reference_rms']*100))
    add('derivatives',derivatives,'Identical central differences on center plus six supported neighbors. Pooled/per-bin errors are unweighted over these valid interior stencils. Viscosity is not assumed; dissipation is per dynamic viscosity.', ['derivative-chart'])
    add('derivative_summary',derivative_summary,'Aggregate interior derivative errors, pooled before normalization. Matched4mm normalized smoothing is a method sensitivity. Divergence has a near-zero reference; absolute errors matter.', ['derivative-table'])
    sensitivity_labels={'adc_time_average':'Actual ADC times','nominal_midpoint':'Nominal midpoint','box3':'Nominal 3-time box',
        'quadrature3':'3³ spatial samples','legacy_half_voxel':'Legacy half-voxel grid'}
    sensitivities=[dict(bin=x['bin'],reference=sensitivity_labels[x['reference']],rmse=x['vector_rmse_m_s'],nrmse_pct=x['vector_nrmse_reference_rms']*100) for x in r['sensitivity_series']]
    add('sensitivities',sensitivities,'Same stored estimator, alternative prespecified references. Primary uses actual ADC-time distribution; this is a kinematic mean, not complex MRI signal response. Half-voxel grid is a disclosed coordinate-convention sensitivity.', ['sampling-chart','sampling-table'])
    masks=[dict(mask=x['mask'],voxels=x['voxels'],rmse=x['pooled']['vector_rmse_m_s'],nrmse_pct=x['pooled']['vector_nrmse_reference_rms']*100) for x in r['mask_sensitivity']]
    add('masks',masks,'Fixed all-bin magnitude thresholds5/10/15/20%, stored valid support and nonzero numerical CFD support. Threshold changes the sampled population.15% is not a validated optimum.', ['mask-chart','mask-table','summary'])
    cores=[dict(section=x['section'],bin=x['bin'],threshold=str(x['threshold_fraction']),direction=str(x['direction_sign']),method=x['method'],
        radius_mm=None if x['rms_radius_m'] is None else x['rms_radius_m']*1000,
        angle_deg=x['angle_to_normal_deg'],flux_ml_s=x['forward_flux_m3_s']*1e6,
        centroid_Y_mm=None if x['centroid_m'][1] is None else x['centroid_m'][1]*1000,
        centroid_Z_mm=None if x['centroid_m'][2] is None else x['centroid_m'][2]*1000) for x in r['flow_core_series']]
    add('cores',cores,'Directional flow-core descriptors: common25/50/75% CFD normal-speed p99 threshold; flux weighted centroid, RMS radius and direction. Missing thresholded core stays null. These are not unique physical jet boundaries.', ['core-chart','core-table'])
    add('profiles',r['profile_series'],'Partial-section area-weighted normal velocity profiles along either transverse coordinate; empty rows are null. No interpolated cardiac bins.', ['profile-chart'])
    timing=[dict(encoding=e,bin=x['bin'],profiles=x['profiles'],adc_phase_circular_mean_ms=x['phase_circular_mean_s']*1000,
        nominal_center_ms=x['nominal_midpoint_s']*1000,offset_ms=(x['phase_circular_mean_s']-x['nominal_midpoint_s'])*1000,
        seam_crossing_profiles=x['seam_crossing_profiles']) for e,t in r['timing_audit'].items() for x in t['bin_timing']]
    add('timing',timing,'Saved Pulseq RF/ADC and linked PHS/LIN/PAR labels. Circular mean accounts for bin8 wrapping. RF-to-last-ADC-center crossing audit does not certify simulator runtime reset behavior.', ['timing-table'])
    add('gates',[dict(parameter=k,reason=v) for k,v in r['gated'].items()],'Unavailable endpoints with concrete missing requirements.', ['gates-table'])
    whole=[x for x in regions if x['region']=='whole']
    c=npmean([x['energy_per_density_m5_s2'] for x in whole if x['method']=='CFD'])
    k=npmean([x['energy_per_density_m5_s2'] for x in whole if x['method']=='Koma'])
    output={'surface':'report','title':'Core aortic flow agrees closely; local velocity errors remain',
            'generatedAt':r['created_utc'],'buildStatus':'creating','status':'simulated-internal-comparison','filters':[],
            'queries':queries,'report':{'asOf':r['created_utc'][:10],'experiment':'E-027','sourceReport':source,'sourceReportSha256':digest,
                'vectorRmse':r['primary_pooled']['vector_rmse_m_s'],'vectorNrmsePct':r['primary_pooled']['vector_nrmse_reference_rms']*100,
                'energyRatio':k/c,'coverage':r['coverage'],'timingOffsetMs':timing[0]['offset_ms'],
                'sectionIds':[x['section'] for x in summaries],'regionIds':list(dict.fromkeys(x['region'] for x in regions)),
                'derivativeMetrics':list(derivative_labels.values()),'derivativeModes':list(derivative_modes.values())}}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps({'output':str(args.output),'queries':len(queries),'rows':sum(len(q['rows']) for q in queries.values()),'source_sha256':digest}))


def npmean(values):return sum(values)/len(values)
if __name__=='__main__':main()
