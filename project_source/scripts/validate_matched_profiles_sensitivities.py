"""Independent source/statistics/operator checks for E-040 PNG exports."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

from mrsimtracks.hemodynamics import velocity_gradient,differential_metrics
from scripts.complete_advanced_comparisons import load_inputs
from scripts.compare_aorta_hemodynamics import dump
from scripts.compare_dp150_hemodynamics import matched_smooth
from scripts.extend_hemodynamic_comparisons import ROOT,DP,sha


def validate(run,receipt):
    checks=[]
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)))
        if not passed:raise AssertionError(name)
    def close(name,a,b):check(name,np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True))
    def read(name):
        with (run/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    report=json.loads((run/'report.json').read_text());manifest=json.loads((run/'MANIFEST.json').read_text())
    for name,record in manifest.items():
        check('export hash '+name,sha(run/name)==record['sha256'])
        check('export bytes '+name,(run/name).stat().st_size==record['bytes'])
        if name.endswith('.png'):check('PNG signature '+name,(run/name).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n')
    for item in report['inputs']:check('source '+item['path'],sha(ROOT/item['path'])==item['sha256'])
    check('24 scientific PNGs',len(list(run.glob('*.png')))==24)
    for item in report['inventory']:
        allowed=['25%','50%','75%'] if 'core_threshold_errors' in item['file'] else [] if any(x in item['file'] for x in ['timing_sampling_masks','derivative_sensitivities']) else ['CFD','Koma']
        check('concise legend '+item['file'],item['legend']==allowed)
    _,cases,dp,ao,ext,_,_=load_inputs()
    regional=read('regional_speeds.csv');longitudinal=read('longitudinal_profiles.csv')
    derivatives=read('derivative_sensitivity.csv');derivative_bins=read('derivative_per_bin.csv');raw_cores=read('core_width.csv');core_errors=read('core_errors.csv')
    previous=json.loads((ROOT/'outputs/hemodynamic_extension/e039_profiles_coverage_v2/report.json').read_text())
    old=json.loads((ROOT/'outputs/hemodynamic_extension/e038_v1/report.json').read_text())
    saved_choices=read('analysis_choices.csv')
    for expected,actual in zip(previous['analysis_choices'],saved_choices):
        for key in ['rmse','nrmse_pct','retained_coverage_pct']:
            if expected[key] is None:check('null choice '+key,actual[key]=='')
            else:close('bound E039 choice '+key,float(actual[key]),expected[key])
    check('all E039 choice rows',len(saved_choices)==len(previous['analysis_choices']))
    def stats(row,values,w,axial=None):
        if not len(w):
            check('empty mean unavailable',row['mean']=='');check('empty p95 unavailable',row['p95']=='');check('empty p99 unavailable',row['p99']=='');return
        close('weighted mean',float(row['mean']),np.sum(values*w)/np.sum(w))
        close('weight',float(row['weight']),np.sum(w));check('voxel count',int(row['n'])==len(w))
        for key,q in [('p95',.95),('p99',.99)]:
            value=float(row[key]);check(key+' is source sample',np.any(values==value))
            check(key+' inverse weighted CDF',w[values<value].sum()/w.sum()<q+1e-11 and w[values<=value].sum()/w.sum()>=q-1e-11)
        if axial is not None:close('signed X mean',float(row['axial_mean']),np.sum(axial*w)/w.sum())
    for case in cases:
        name=case['name'];volume=case['volume'];whole=case['regions']['whole']
        nregional=0;nlong=0
        for b in range(len(case['times'])):
            for method,field in [('CFD',case['reference']),('Koma',case['estimate'])]:
                speed=np.linalg.norm(field[b],axis=-1)
                for row in [r for r in regional if r['case']==name and int(r['bin'])==b+1 and r['method']==method]:
                    mask=case['regions'][row['region']];stats(row,speed[mask],volume[mask]);nregional+=1
                    close('regional time',float(row['time_s']),case['times'][b])
                for row in [r for r in longitudinal if r['case']==name and int(r['bin'])==b+1 and r['method']==method]:
                    i=int(row['readout_index']);mask=whole[:,:,i];stats(row,speed[:,:,i][mask],volume[:,:,i][mask],field[b,:,:,i,0][mask]);nlong+=1
                    expected_x=(i+(.5 if name=='phantom_1m' else 0)-whole.shape[2]/2)*case['spacing'][0]*1000
                    close('longitudinal coordinate',float(row['coordinate_mm']),expected_x)
        check('all regional samples '+name,nregional==len(case['times'])*2*len(case['regions']))
        check('all longitudinal samples '+name,nlong==len(case['times'])*2*whole.shape[2])
        # Independently reconstruct core direction and compare all exported radii.
        if name=='phantom_1m':source=dp['jet_values'];flows=[r for r in dp['section_values'] if r['dataset']=='cfd_voxel_common']
        elif name=='aorta_1m':source=ao['flow_core_series'];flows=[r for r in ao['section_series'] if r['method']=='CFD']
        else:source=ext['aorta_pilot_sections']['flow_cores'];flows=[r for r in ext['aorta_pilot_sections']['series'] if r['method']=='CFD']
        expected_cores={}
        for r in source:
            sign=1 if sum(v['net_m3_s'] for v in flows if v['section']==r['section'])>=0 else -1
            if r.get('direction_sign',1)!=sign:continue
            method=r.get('method') or {'cfd':'CFD','koma':'Koma'}[r['dataset']]
            expected_cores[(r['section'],r['bin'],method,r['threshold_fraction'])]=r['rms_radius_m']
        rows=[r for r in raw_cores if r['case']==name];check('core inventory '+name,len(rows)==len(expected_cores))
        for r in rows:
            expected=expected_cores[(r['section'],int(r['bin']),r['method'],float(r['threshold_fraction']))]
            if expected is None or not np.isfinite(expected):check('missing core gap',r['radius_mm']=='')
            else:close('source core radius',float(r['radius_mm']),expected*1000)
        for r in [x for x in core_errors if x['case']==name]:
            expected=next(v for v in old['flow_core_comparison'] if v['case']==name and v['section']==r['section'] and v['bin']==int(r['bin']) and v['direction_sign']==int(r['direction_sign']) and v['threshold_fraction']==float(r['threshold_fraction']))
            for key in ['centroid_error_mm','direction_error_deg']:
                if expected[key] is None:check('unavailable core error',r[key]=='')
                else:close('source paired core error',float(r[key]),expected[key])
        # Reconcile every pooled derivative row from exported per-bin numerators.
        for r in [x for x in derivatives if x['case']==name]:
            rows=[x for x in derivative_bins if x['case']==name and x['choice']==r['choice'] and x['metric']==r['metric']]
            w=sum(float(x['weight_m3']) for x in rows);err=sum(float(x['weighted_error_square']) for x in rows);refsq=sum(float(x['weighted_reference_square']) for x in rows)
            if not w:check('empty derivative null',r['rmse']=='' and r['nrmse_pct']=='' and int(r['voxels'])==0)
            else:close('pooled derivative RMSE',float(r['rmse']),np.sqrt(err/w));close('pooled derivative NRMSE',float(r['nrmse_pct']),100*np.sqrt(err/refsq))
        # Full-grid original operator, one representative bin per choice: an
        # independent implementation check, not falsely an all-bin recomputation.
        b=previous['cases'][name]['regional_selections'][0]['bin']-1
        primary=case['gradient_mask'];masks={'Base':primary,'Smooth':primary}
        if name=='phantom_1m':
            with h5py.File(DP/'matched_reference.h5') as f:masks['Direct']=primary&f['direct_common_mask'][:]
            masks['Erode']=binary_erosion(primary)
        else:masks['Erode']=binary_erosion(primary,structure=np.ones((3,3,3)))
        for choice,mask in masks.items():
            a=case['reference'][b];c=case['estimate'][b]
            if choice=='Smooth':a,c=[matched_smooth(u,mask,case['spacing'],sigma_m=.001 if name=='phantom_1m' else .004) for u in [a,c]]
            ga,ma=velocity_gradient(a,case['spacing'],mask);gc,mc=velocity_gradient(c,case['spacing'],mask)
            check('same original stencil',np.array_equal(ma,mc))
            da=differential_metrics(ga,a);dc=differential_metrics(gc,c)
            da['strain_frobenius_s_inv']=np.linalg.norm(da['strain_rate_s_inv'],axis=(-2,-1));dc['strain_frobenius_s_inv']=np.linalg.norm(dc['strain_rate_s_inv'],axis=(-2,-1))
            for r in [x for x in derivative_bins if x['case']==name and x['choice']==choice and int(x['bin'])==b+1]:
                key=r['metric'];w=volume[ma]
                close('original-operator error numerator',float(r['weighted_error_square']),np.sum((dc[key][ma]-da[key][ma])**2*w))
                close('original-operator reference numerator',float(r['weighted_reference_square']),np.sum(da[key][ma]**2*w))
                close('original-operator weight',float(r['weight_m3']),w.sum())
    dump(receipt,dict(experiment='E-040',status='PASS',passed=len(checks),total=len(checks),checks=checks,
        scope='All regional/longitudinal statistics and quantiles, all core widths and errors, all pooled derivative arithmetic, one CFD-peak-bin original full-grid derivative check per case/choice, input/output hashes, PNG inventory and concise legend metadata; not independent physical validation.'))
    print(f'{len(checks)}/{len(checks)} source/statistics/operator checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();validate(a.run,a.receipt)
