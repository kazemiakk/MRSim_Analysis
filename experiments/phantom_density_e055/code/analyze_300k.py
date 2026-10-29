"""Matched-support 100k/300k Float64 comparison; never promote 1M automatically."""
import argparse
import csv
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
import tomllib
import h5py
import numpy as np

# The retained environment also contains upstream 0.2.1. Local analysis helpers
# require the reviewed project source, not that installed package shadow.
SOURCE_REPO=Path(__file__).resolve().parents[3]/'MRSIM/MRSimTracks'
sys.path.insert(0,str(SOURCE_REPO))
from scripts.analyze_phantom_precision import load_arm,bind_reference,region_rows,coverage,partial_flow
from scripts.analyze_remapped_confirmation import fixed_mask,velocities
from scripts.build_aorta_inlet_picker import sha
from scripts.compare_dp150_hemodynamics import PLANES,REGIONS
from validate_300k import validate


def quantile(value,weight,p):
    order=np.argsort(value);x=value[order];w=weight[order]
    return float(x[min(np.searchsorted(np.cumsum(w),p*w.sum()),len(x)-1)])


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def scaleup_decision(coverage_values,local_values,flow_rows):
    available=lambda values:all(v is not None and np.isfinite(v) for v in values)
    coverage_pass=available(coverage_values) and coverage_values[1]>coverage_values[0]
    local_pass=available(local_values) and local_values[1]<local_values[0]
    numerical=coverage_pass and local_pass and all(r['no_worse_RMSE'] for r in flow_rows)
    return dict(own_coverage_improved=bool(coverage_pass),paired_throat_RMSE_improved=bool(local_pass),paired_flows=flow_rows,
                single_comparison_numeric_gates_pass=bool(numerical),independent_seed_variability_established=False,
                status='HOLD_1M_PENDING_SEED_CONFIRMATION' if numerical else 'HOLD_1M_LOCAL_OR_FLOW_GATE_NOT_MET',new_1m_started=False)


def main(repo,run):
    if repo.resolve()!=SOURCE_REPO.resolve():raise ValueError('Reviewed source location differs')
    final_output=run/'analysis_v1'
    output=run/'analysis_v1.building'
    if output.exists() or final_output.exists():raise FileExistsError(output)
    output.mkdir()
    baseline=repo/'outputs/phantom_precision_e050/mapped_f64_v1';candidate=run/'mapped_f64_v1'
    arms=['baseline100k','candidate300k'];folders=dict(zip(arms,[baseline,candidate]))
    reports={a:tomllib.loads((p/'control_report.toml').read_text()) for a,p in folders.items()}
    assert reports[arms[0]]['status']=='COMPLETE_E050_F64_100K_MAPPED_ACQUISITIONS'
    assert reports[arms[1]]['status']=='COMPLETE_E055_F64_300K_MAPPED_ACQUISITIONS'
    source=json.loads((run/'source_registration.json').read_text());info=tomllib.loads((run/'inputs_v1/case_info.toml').read_text())
    assert sha(run/'inputs_v1/case_info.toml')==source['case_info_sha256']
    for key in ['manifest_sha256','helper_sha256','motion_timing_contract']:
        assert reports[arms[0]][key]==reports[arms[1]][key],key
    pre=tomllib.loads((run/'preflight_v1/report.toml').read_text())
    for key in ['script_sha256','input_sha256','timing_sha256','map_sha256','protocol_sha256','manifest_sha256','precision']:
        assert reports[arms[1]][key]==pre[key],key
    reference_path=bind_reference(repo/'outputs/phantom_diagnosis_e048/reference_v1','reference_fields.h5',
                                  'f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694')
    local_path=bind_reference(repo/'outputs/phantom_diagnosis_e048/local_v1','local_fields.h5',
                              '6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885')
    images={};fields={};audits={}
    for arm in arms:
        print('E055 ANALYZE',arm,flush=True)
        images[arm],audits[arm]=load_arm(folders[arm],reports[arm],candidate=True)
        fields[arm]=velocities(images[arm],info['Venc_m_s'])
    with h5py.File(reference_path) as f:reference=f['fft_adc'][:];fraction=f['fraction_fft'][:]
    with h5py.File(local_path) as f:source_z=f['source_z_fft_m'][:]
    regions=[];masks_by_threshold={};bin_coverage=[]
    for threshold in [.05,.1,.15,.2]:
        masks={a:fixed_mask(images[a],threshold) for a in arms};masks_by_threshold[threshold]=masks
        regions+=region_rows(reference,fields,masks,fraction,source_z,threshold)
        for arm in arms:
            per_bin=np.ones(reference.shape[:-1],bool)
            for im in images[arm].values():
                m=abs(im);per_bin&=(m>=threshold*m.max(axis=(1,2,3),keepdims=True))&(m>0)
            bin_coverage += [dict(arm=arm,threshold=threshold,bin=b+1,coverage_fraction=coverage(fraction,per_bin[b])) for b in range(15)]
    paired=masks_by_threshold[.05][arms[0]]&masks_by_threshold[.05][arms[1]]
    norm_mag={}
    for a,n in zip(arms,[100000,300000]):
        norm_mag[a]=float(np.mean(abs(images[a]['REF'])[:,paired])/n) if paired.any() else None
    del images;gc.collect()
    sections=[];profiles=[]
    with h5py.File(local_path) as f:
        for section,target in PLANES.items():
            idx=int(np.argmin(abs(source_z-target)));area=f['plane_fraction'][:,:,idx]*3.125
            ref=f['plane_reference'][:,:,:,idx,0]
            for threshold,masks in masks_by_threshold.items():
                common=masks[arms[0]][:,:,idx]&masks[arms[1]][:,:,idx]
                for arm in arms:
                    for policy,keep in [('own',masks[arm][:,:,idx]),('paired',common)]:
                        sections.append(dict(section=section,index=idx,source_z_m=float(source_z[idx]),threshold=threshold,arm=arm,support=policy,
                            **partial_flow(ref,fields[arm][:,:,:,idx,0],area,keep,info['period_s'])))
            if section=='throat':
                full_q=ref[:,area>0]@area[area>0]
                for phase,b in [('peak',int(np.argmax(full_q))),('low',int(np.argmin(full_q)))]:
                    for orientation in ['transverse','longitudinal']:
                        if orientation=='transverse':
                            keep=paired[16,:,idx]&(fraction[16,:,idx]>0);r=reference[b,16,:,idx,0]
                            estimate={a:fields[a][b,16,:,idx,0] for a in arms};coordinate=(np.arange(64)-32)*1.25
                        else:
                            keep=paired[16,32,:]&(fraction[16,32,:]>0);r=reference[b,16,32,:,0]
                            estimate={a:fields[a][b,16,32,:,0] for a in arms};coordinate=source_z*1000
                        profiles.append(dict(phase=phase,bin=b+1,orientation=orientation,coordinate_mm=coordinate.tolist(),paired_retained=keep.tolist(),
                            CFD_axial_m_s=r.tolist(),**{a+'_axial_m_s':estimate[a].tolist() for a in arms},
                            **{a+'_paired_RMSE_m_s':float(np.sqrt(np.mean((estimate[a][keep]-r[keep])**2))) if keep.any() else None for a in arms}))
    percentiles=[]
    for region,(low,high) in REGIONS.items():
        eligible=(fraction>0)&((source_z>=low)&(source_z<high))[None,None,:]
        keep=paired&eligible
        weights=np.tile(fraction[keep].astype(float),15)
        for arm in arms:
            speed=np.linalg.norm(fields[arm][:,keep],axis=-1).ravel();ref_speed=np.linalg.norm(reference[:,keep],axis=-1).ravel()
            percentiles.append(dict(region=region,arm=arm,paired_coverage_fraction=coverage(fraction,keep,eligible),
                speed_p95_m_s=quantile(speed,weights,.95) if keep.any() else None,
                speed_p99_m_s=quantile(speed,weights,.99) if keep.any() else None,
                reference_p95_m_s=quantile(ref_speed,weights,.95) if keep.any() else None,
                reference_p99_m_s=quantile(ref_speed,weights,.99) if keep.any() else None))
    primary=[r for r in regions if r['threshold']==.05 and r['region'] in ['whole','throat']]
    find=lambda arm,region,support:next(r for r in primary if r['arm']==arm and r['region']==region and r['support']==support)
    own=[find(a,'whole','own')['coverage_fraction'] for a in arms]
    local=[(find(a,'throat','paired')['metrics'] or {}).get('vector_rmse_m_s') for a in arms]
    flow=[]
    for section in ['upstream','throat']:
        values=[next(r for r in sections if r['section']==section and r['threshold']==.05 and r['arm']==a and r['support']=='paired') for a in arms]
        metrics=[r['metrics'] for r in values]
        flow.append(dict(section=section,paired_area_coverage_fraction=values[0]['coverage_fraction'],
            no_worse_RMSE=all(m is not None for m in metrics) and metrics[1]['rmse_ml_s']<=metrics[0]['rmse_ml_s']))
    decision=scaleup_decision(own,local,flow)
    report=dict(experiment='E-055',status='COMPLETE_300K_MATCHED_COMPARISON_NOT_PRODUCTION_PROMOTION',
        created_utc=datetime.now(timezone.utc).isoformat(),evidence_class='Same-source internal consistency; nested particle-count confirmation',
        primary=primary,regions=regions,sections=sections,profiles=profiles,regional_percentiles=percentiles,bin_coverage=bin_coverage,
        reconstruction_audits=audits,amplitude_mean_REF_divided_by_particle_count=norm_mag,one_million_decision=decision,
        limitations=['Own coverage and identical-paired error populations have different denominators.',
                    'The same nearest-neighbor map algorithm is recomputed at 300k; this is not proof of pure particle-count causality.',
                    'One nested density comparison does not establish random-seed convergence or independent physical accuracy.',
                    'Frozen E048 actual-ADC reference retains its declared timing and synthetic-tail qualifications.',
                    'No fitting, phase filling, threshold relaxation, TKE/wall/pressure validation or automatic 1M.'])
    dump(output/'report.json',report)
    with (output/'regions.csv').open('w',newline='',encoding='utf-8') as f:
        values=[{k:r[k] for k in ['threshold','region','arm','support','coverage_fraction']}|{k:(r['metrics'] or {}).get(k) for k in ['voxels','vector_rmse_m_s','mean_speed_bias_m_s']} for r in regions]
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    with h5py.File(output/'paired_fields.h5','w') as f:
        f['paired_fixed5_mask']=paired
        for a in arms:
            f.create_dataset(a,data=fields[a],compression='gzip',compression_opts=1)
            f[a+'_fixed5_mask']=masks_by_threshold[.05][a]
    text='# E-055 corrected 300k confirmation\n\n**'+decision['status']+'**.\n\n'
    text+='| Arm | Own volume coverage | Paired global RMSE (m/s) | Paired throat RMSE (m/s) |\n|---|---:|---:|---:|\n'
    number=lambda value: 'Unavailable' if value is None else f'{value:.6f}'
    for a in arms:
        cov=find(a,'whole','own')['coverage_fraction']
        cov_text='Unavailable' if cov is None else f'{100*cov:.2f}%'
        text+=f"| {a} | {cov_text} | {number((find(a,'whole','paired')['metrics'] or {}).get('vector_rmse_m_s'))} | {number((find(a,'throat','paired')['metrics'] or {}).get('vector_rmse_m_s'))} |\n"
    text+='\nThe 1M run is not started. See report.json for same-area partial-flow, profiles, percentiles and remaining seed uncertainty.\n'
    (output/'README.md').write_text(text,encoding='utf-8')
    validate(repo,run,output)
    dump(output/'MANIFEST.json',dict(inputs={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('validate_300k.py'),run/'code/PROTOCOL.md',run/'source_registration.json',baseline/'control_report.toml',candidate/'control_report.toml',reference_path,local_path]},
                                   outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    output.rename(final_output)
    print(json.dumps(dict(status=report['status'],primary=primary,one_million_decision=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.run)
