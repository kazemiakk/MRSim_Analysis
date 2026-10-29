"""E-040: concise-legend PNGs from original matched CFD/Koma fields."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import tempfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from mrsimtracks.hemodynamics import differential_metrics
from scripts.complete_advanced_comparisons import load_inputs
from scripts.compare_dp150_hemodynamics import matched_smooth, safe_json
from scripts.compare_aorta_hemodynamics import dump, table
from scripts.extend_hemodynamic_comparisons import ROOT, DP, DP_H5, AO, sha, bind
from scripts.plot_profiles_coverage import read_profiles, ecdf

BASE=ROOT/'outputs/hemodynamic_extension'
PRIOR=BASE/'e039_profiles_coverage_v2'
COLORS={'CFD':'#176489','Koma':'#bf691b'}
LABELS={'phantom_1m':'Phantom | 1M | 15 bins','aorta_1m':'Aorta baseline | 1M | 8 bins','aorta_100k_corrected':'Aorta pilot | 100k | 8 bins'}
REGION_LABELS={'whole':'Whole','upstream':'Upstream','throat':'Throat','near_downstream':'Near DS','mid_downstream':'Mid DS','far_downstream':'Far DS',
    'readout_below_minus45':'X < -45 mm','readout_middle':'-45 to +45 mm','readout_above45':'X > +45 mm'}
METRICS={'vorticity_magnitude_s_inv':'Vorticity','helicity_m_s_inv2':'Helicity','divergence_s_inv':'Divergence',
    'strain_frobenius_s_inv':'Strain','dissipation_per_viscosity_s_inv2':'Dissipation / viscosity'}
SHORT={'Hybrid common':'Hybrid','Raw direct only':'Direct','Regularized only':'Regularized','Full-voxel interior':'Interior','Eroded interior':'Eroded','Exclude synthetic-tail bins':'No tail',
    'core05':'5%','core10':'10%','core15':'15%','core20':'20%','eroded15':'Eroded15%',
    'Midpoint (primary)':'Mid','Three-sample box':'Box','3-cubed reference':'Q3','adc_time_average':'ADC','nominal_midpoint':'Mid','box3':'Box','quadrature3':'Q3','legacy_half_voxel':'Offset'}


def weighted_stats(values,weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if values.shape!=weights.shape:raise ValueError('Values/weights shape mismatch')
    keep=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not keep.any():return dict(mean=None,p95=None,p99=None,weight=0.,n=0)
    x,y=ecdf(values[keep],weights[keep])
    return dict(mean=float(np.average(values[keep],weights=weights[keep])),
        p95=float(x[min(np.searchsorted(y,.95),len(x)-1)]),p99=float(x[min(np.searchsorted(y,.99),len(x)-1)]),
        weight=float(weights[keep].sum()),n=int(keep.sum()))


def compact_derivatives(u,spacing,mask):
    good=mask&np.isfinite(u).all(axis=-1)
    valid=binary_erosion(good,structure=generate_binary_structure(3,1),border_value=0)
    k,j,i=np.where(valid);g=np.empty((len(k),3,3))
    for axis,d in enumerate(spacing):
        before=[k,j,i];after=[k,j,i]
        before[2-axis]=before[2-axis]-1;after[2-axis]=after[2-axis]+1
        g[:,:,axis]=(u[tuple(after)]-u[tuple(before)])/(2*d)
    fields=differential_metrics(g,u[valid]);fields['strain_frobenius_s_inv']=np.linalg.norm(fields['strain_rate_s_inv'],axis=(-2,-1))
    return {key:fields[key] for key in METRICS},valid


def frame(fig,case,title,thresholds=False):
    fig.suptitle(LABELS[case['name']]+'\n'+title,fontsize=14,y=.995)
    if thresholds:
        handles=[Line2D([],[],color=c,ls=ls,label=label) for c,ls,label in [('#176489','-','25%'),('#bf691b','--','50%'),('#756b35',':','75%')]]
    else:
        handles=[Line2D([],[],color=COLORS[m],ls='-' if m=='CFD' else '--',marker='o' if m=='CFD' else 's',ms=3,label=m) for m in COLORS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=len(handles),frameon=False,handlelength=2)


def export(fig,out,name,inventory,legends):
    path=out/(name+'.png')
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.14)
    plt.close(fig)
    inventory.append(dict(file=path.name,legend=legends))


def time_axis(ax,case):
    ax.set_xlim(0,case['period']);ax.set_xlabel('MRI time (s)');ax.grid(alpha=.18)
    if case['name']=='phantom_1m':ax.axvspan(.64/.72*case['period'],case['period'],color='.9',zorder=-2)


def paired(ax,x,values,case=None):
    for method in COLORS:ax.plot(x,values[method],'-o' if method=='CFD' else '--s',color=COLORS[method],ms=3,lw=1.4)
    ax.grid(alpha=.18)
    if case is not None:time_axis(ax,case)


def transverse(case,profiles,selections,out,inventory):
    sections=list(dict.fromkeys(r['section'] for r in selections))
    for phase in ['peak','low']:
        fig,axs=plt.subplots(len(sections),2,figsize=(12,2.1*len(sections)+1.1),squeeze=False)
        fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.89,hspace=.65,wspace=.26)
        for row,section in enumerate(sections):
            selected=next(r for r in selections if r['section']==section and r['phase']==phase)
            for col,axis in enumerate(['Y','Z']):
                ax=axs[row,col]
                values={m:sorted([r for r in profiles if r['section']==section and r['bin']==selected['bin'] and r['axis']==axis and r['method']==m],key=lambda r:r['coordinate_mm']) for m in COLORS}
                coords=[r['coordinate_mm'] for r in values['CFD']]
                if coords!=[r['coordinate_mm'] for r in values['Koma']]:raise ValueError('Unmatched profile coordinates')
                paired(ax,coords,{m:[np.nan if r['velocity'] is None else r['velocity'] for r in values[m]] for m in COLORS})
                tail=' [tail]' if selected['synthetic_tail'] else ''
                ax.set(title=f"{section} | bin {selected['bin']}{tail}",xlabel=f'MRI {axis} (mm)',ylabel='X velocity (m/s)')
                ax.axhline(0,color='.5',lw=.7)
        frame(fig,case,'Matched transverse profiles | '+phase+' flow')
        note='Central-voxel lines.' if case['name']=='phantom_1m' else 'Area-weighted strip means on anonymous partial sections, not wall-to-wall profiles.'
        fig.supxlabel(note+' Same CFD-selected bins and coordinates; gaps are unavailable.\nIndependent panel scales; CFD/Koma share each panel. '+('Low bin15 overlaps synthetic source closure. ' if phase=='low' and case['name']=='phantom_1m' else '')+'Internal consistency only.',fontsize=9,y=.005)
        export(fig,out,case['name']+'_transverse_'+phase,inventory,['CFD','Koma'])


def speed_tables(case):
    regional=[];longitudinal=[];v=case['volume'];mask=case['regions']['whole'];nr=mask.shape[2]
    coordinate=(np.arange(nr)+(.5 if case['name']=='phantom_1m' else 0)-nr/2)*case['spacing'][0]*1000
    for b,t in enumerate(case['times']):
        # One fixed paired support for both methods and all saved bins.
        for method,field in [('CFD',case['reference']),('Koma',case['estimate'])]:
            speed=np.linalg.norm(field[b],axis=-1)
            for region,roi in case['regions'].items():
                stats=weighted_stats(speed[roi],v[roi])
                regional.append(dict(case=case['name'],bin=b+1,time_s=float(t),region=region,method=method,**stats))
            for i,x in enumerate(coordinate):
                roi=mask[:,:,i];w=v[:,:,i][roi];stats=weighted_stats(speed[:,:,i][roi],w)
                axial=weighted_stats(field[b,:,:,i,0][roi],w)
                longitudinal.append(dict(case=case['name'],bin=b+1,time_s=float(t),readout_index=i,coordinate_mm=float(x),method=method,axial_mean=axial['mean'],**stats))
    return regional,longitudinal


def longitudinal_plot(case,rows,selections,out,inventory):
    fig,axs=plt.subplots(3,2,figsize=(13,10));fig.subplots_adjust(left=.09,right=.98,bottom=.105,top=.88,hspace=.55,wspace=.26)
    for col,selection in enumerate(selections):
        for r,(key,title) in enumerate([('axial_mean','Mean X velocity'),('mean','Mean speed'),('p99','Speed p99')]):
            data={m:sorted([x for x in rows if x['bin']==selection['bin'] and x['method']==m],key=lambda x:x['coordinate_mm']) for m in COLORS}
            paired(axs[r,col],[x['coordinate_mm'] for x in data['CFD']],{m:[np.nan if x[key] is None else x[key] for x in data[m]] for m in COLORS})
            tail=' [tail]' if selection['synthetic_tail'] else ''
            axs[r,col].set(title=f"{selection['phase']} | bin {selection['bin']}{tail} | {title}",xlabel='MRI readout X (mm)',ylabel='m/s')
    frame(fig,case,'Matched longitudinal / readout-axis profiles')
    fig.supxlabel('Volume-weighted partial-support readout slices; aortic slices may combine branches, not anatomical centerlines.\nSame CFD-selected anchor bins and positions; empty slices remain gaps. Independent panel scales; CFD/Koma share each panel.\n'+('Low bin15: synthetic-tail overlap. ' if case['name']=='phantom_1m' else '')+'Internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_longitudinal',inventory,['CFD','Koma'])


def regional_plot(case,rows,out,inventory):
    regions=list(case['regions']);fig,axs=plt.subplots(3,len(regions),figsize=(3.2*len(regions),9.5),squeeze=False)
    fig.subplots_adjust(left=.055,right=.99,bottom=.09,top=.88,hspace=.5,wspace=.34)
    for row,(key,title) in enumerate([('mean','Mean speed'),('p95','Speed p95'),('p99','Speed p99')]):
        ymax=max(r[key] or 0 for r in rows)
        for col,region in enumerate(regions):
            data={m:sorted([r for r in rows if r['region']==region and r['method']==m],key=lambda r:r['bin']) for m in COLORS}
            paired(axs[row,col],case['times'],{m:[r[key] for r in data[m]] for m in COLORS},case)
            axs[row,col].set(title=REGION_LABELS[region]+' | '+title,ylabel='m/s',ylim=(0,ymax*1.05))
    frame(fig,case,'Regional speeds and high-percentile speeds')
    fig.supxlabel('Matched volume weights and fixed common support; exact weighted p95/p99; whole overlaps subregions.\nComputational regions, not anatomical aortic labels. '+('Gray: synthetic source tail. ' if case['name']=='phantom_1m' else '')+'Internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_regional_speeds_p95_p99',inventory,['CFD','Koma'])


def choices_plot(case,rows,out,inventory):
    fig,axs=plt.subplots(2,2,figsize=(14,9));fig.subplots_adjust(left=.20,right=.98,bottom=.13,top=.88,hspace=.44,wspace=.20)
    for col,family in enumerate(['reference','support']):
        selected=[r for r in rows if r['case']==case['name'] and r['family']==family];y=np.arange(len(selected));labels=[SHORT.get(r['choice'],r['choice']) for r in selected]
        for row,(key,title,color) in enumerate([('nrmse_pct','Vector NRMSE (%)','#176489'),('retained_coverage_pct','Retained volume (%)','#bf691b')]):
            ax=axs[row,col];values=[r[key] for r in selected]
            ax.barh(y,values,color=color,height=.6);ax.set(yticks=y,yticklabels=labels,xlabel=title,title='Timing / sampling' if col==0 else 'Masks / phase inclusion')
            ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
            for i,value in enumerate(values):ax.text(value,i,f' {value:.1f}',va='center',fontsize=9)
    fig.suptitle(LABELS[case['name']]+'\nReference timing, sampling and masks',fontsize=14,y=.995)
    fig.supxlabel('ADC: actual acquisition-time average; Mid: nominal midpoint; Box: three-time samples; Q3: 3-cubed spatial samples.\nOffset: historical half-voxel sensitivity. Coverage uses original numerical CFD support, not physical lumen.\nFixed primary weights for reference tests; exclusions change the sampled population. Internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_timing_sampling_masks',inventory,[])


def core_plots(case,source,pairs,selections,out,inventory):
    sections=list(dict.fromkeys(s['section'] for s in selections));directions={}
    for section in sections:
        # All phase-selector rows carry the source CFD signed flow. The dominant
        # direction is fixed from its cycle sum, supplied by main below.
        directions[section]=case['core_direction'][section]
    raw=[]
    for r in source:
        if r.get('direction_sign',1)!=directions[r['section']]:continue
        method=r.get('method') or {'cfd':'CFD','koma':'Koma'}[r['dataset']]
        raw.append(dict(case=case['name'],section=r['section'],bin=r['bin'],method=method,threshold_fraction=r['threshold_fraction'],direction_sign=directions[r['section']],radius_mm=None if r['rms_radius_m'] is None or not np.isfinite(r['rms_radius_m']) else r['rms_radius_m']*1000))
    fig,axs=plt.subplots(len(sections),3,figsize=(14,2.05*len(sections)+1.2),squeeze=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.89,hspace=.62,wspace=.27)
    for row,section in enumerate(sections):
        for col,fraction in enumerate([.25,.5,.75]):
            values={m:[next((r['radius_mm'] for r in raw if r['section']==section and r['threshold_fraction']==fraction and r['method']==m and r['bin']==b+1),None) for b in range(len(case['times']))] for m in COLORS}
            paired(axs[row,col],case['times'],values,case)
            axs[row,col].set(title=f'{section} | {int(fraction*100)}%',ylabel='Core RMS radius (mm)')
            if all(x is None for v in values.values() for x in v):axs[row,col].text(.5,.5,'Unavailable',ha='center',transform=axs[row,col].transAxes)
    frame(fig,case,'Flow-core width versus threshold')
    threshold_note='Threshold = fraction of CFD throat p99.' if case['name']=='phantom_1m' else 'Threshold = fraction of section CFD absolute axial p99; direction from CFD cycle net flow.'
    fig.supxlabel(threshold_note+' Missing cores remain gaps, not zero width.\nIndependent panel scales; CFD/Koma share each panel. '+('Gray: synthetic source tail. ' if case['name']=='phantom_1m' else '')+'Partial sections; internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_core_threshold_width',inventory,['CFD','Koma'])
    fig,axs=plt.subplots(2,len(sections),figsize=(3.4*len(sections),7),squeeze=False);fig.subplots_adjust(left=.06,right=.99,bottom=.12,top=.86,hspace=.5,wspace=.33)
    selected_pairs=[r for r in pairs if r['case']==case['name'] and r['direction_sign']==directions[r['section']]]
    for row,(key,label) in enumerate([('centroid_error_mm','Centroid error (mm)'),('direction_error_deg','Direction error (deg)')]):
        for col,section in enumerate(sections):
            for fraction,color,ls in [(.25,'#176489','-'),(.5,'#bf691b','--'),(.75,'#756b35',':')]:
                values=[next((r[key] for r in selected_pairs if r['section']==section and r['bin']==b+1 and r['threshold_fraction']==fraction),None) for b in range(len(case['times']))]
                axs[row,col].plot(case['times'],values,color=color,ls=ls,marker='o',ms=3)
            time_axis(axs[row,col],case);axs[row,col].set(title=section,ylabel=label)
    frame(fig,case,'Paired flow-core errors versus threshold',thresholds=True)
    fig.supxlabel('Same section/bin/threshold/direction pairs; missing paired cores remain gaps. Thresholds follow original case definitions.\nIndependent panel scales; thresholds share each panel. '+('Gray: synthetic source tail. ' if case['name']=='phantom_1m' else '')+'No optimal threshold claimed. Internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_core_threshold_errors',inventory,['25%','50%','75%'])
    return raw,selected_pairs


def derivative_rows(case):
    primary=case['gradient_mask']
    if case['name']=='phantom_1m':
        with h5py.File(DP/'matched_reference.h5') as f:direct=f['direct_common_mask'][:]
        choices=[('Base',primary,None),('Direct',primary&direct,None),('Erode',binary_erosion(primary),None),('Smooth',primary,.001)]
    else:choices=[('Base',primary,None),('Erode',binary_erosion(primary,structure=np.ones((3,3,3))),None),('Smooth',primary,.004)]
    rows=[];per_bin=[]
    for choice,mask,sigma in choices:
        sums={k:np.zeros(3) for k in METRICS};last=None
        for b,(a,c) in enumerate(zip(case['reference'],case['estimate'])):
            if sigma is not None:a,c=[matched_smooth(u,mask,case['spacing'],sigma_m=sigma) for u in (a,c)]
            da,valid=compact_derivatives(a,case['spacing'],mask);dc,vc=compact_derivatives(c,case['spacing'],mask)
            if not np.array_equal(valid,vc) or (last is not None and not np.array_equal(last,valid)):raise ValueError('Different derivative supports')
            last=valid;w=case['volume'][valid];weight=float(w.sum())
            for key in METRICS:
                error=float(np.sum(w*(dc[key]-da[key])**2));reference_square=float(np.sum(w*da[key]**2));sums[key]+=np.array([error,reference_square,weight])
                per_bin.append(dict(case=case['name'],choice=choice,bin=b+1,metric=key,weighted_error_square=error,weighted_reference_square=reference_square,weight_m3=weight))
        for key,(error,reference_square,weight) in sums.items():
            rows.append(dict(case=case['name'],choice=choice,metric=key,rmse=float(np.sqrt(error/weight)) if weight else None,nrmse_pct=float(100*np.sqrt(error/reference_square)) if reference_square else None,
                voxels=int(last.sum()),volume_ml=float(case['volume'][last].sum()*1e6),coverage_pct=float(100*case['volume'][last].sum()/case['volume'].sum()),smoothing_m=sigma))
    return rows,per_bin


def derivative_plot(case,rows,out,inventory):
    fig,axs=plt.subplots(2,3,figsize=(14,9));fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.88,hspace=.50,wspace=.28)
    for ax,(key,label) in zip(axs.flat,METRICS.items()):
        values=[r for r in rows if r['metric']==key];xx=np.arange(len(values))
        for i,r in enumerate(values):
            if r['nrmse_pct'] is None:ax.text(i,.03,'N/A',ha='center',transform=ax.get_xaxis_transform())
            else:ax.bar(i,r['nrmse_pct'],color='#176489',width=.6)
        ax.set(xticks=xx,xticklabels=[r['choice'] for r in values],title=label,ylabel='NRMSE (%)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    values=[r for r in rows if r['metric']==next(iter(METRICS))];ax=axs.flat[-1];xx=np.arange(len(values));ax.bar(xx,[r['coverage_pct'] for r in values],color='#bf691b',width=.6)
    ax.set(xticks=xx,xticklabels=[r['choice'] for r in values],title='Retained derivative support',ylabel='Numerical volume (%)',ylim=(0,max(r['coverage_pct'] for r in values)*1.18))
    for i,r in enumerate(values):ax.text(i,r['coverage_pct'],f"{r['voxels']:,}",ha='center',va='bottom',fontsize=9)
    fig.suptitle(LABELS[case['name']]+'\nAvailable derivative sensitivities',fontsize=14,y=.995)
    fig.supxlabel('Volume-weighted paired errors; independent panel scales; N/A = no valid central stencils, not zero error.\nBase: primary interior; Direct: hybrid field on direct-only stencils; Erode: extra erosion; Smooth: matched '+('1 mm' if case['name']=='phantom_1m' else '4 mm')+' smoothing.\nInterior derivatives, not wall shear or turbulent loss. Smoothing changes bandwidth. Internal consistency only.',fontsize=9,y=.005)
    export(fig,out,case['name']+'_derivative_sensitivities',inventory,[])


def main(target):
    if target.exists():raise FileExistsError(target)
    inputs,cases,dp,ao,ext,_,_=load_inputs()
    delivery=json.loads((BASE/'delivery_validation_e039.json').read_text());inputs.append(bind(PRIOR/'MANIFEST.json',delivery['manifest_sha256']))
    manifest=json.loads((PRIOR/'MANIFEST.json').read_text());inputs.append(bind(PRIOR/'report.json',manifest['report.json']['sha256']))
    previous=json.loads((PRIOR/'report.json').read_text());old_path=BASE/'e038_v1/report.json';old_manifest=json.loads((old_path.parent/'MANIFEST.json').read_text());inputs.append(bind(old_path,old_manifest['report.json']['sha256']));old=json.loads(old_path.read_text())
    inputs+=[bind(Path(__file__)),bind(BASE/'E040_PROTOCOL.md'),bind(DP/'profiles.csv')]
    profiles=read_profiles(dp,ao,ext)
    target.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix=target.name+'.partial-',dir=target.parent))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    inventory=[];regional=[];longitudinal=[];derivatives=[];derivative_bins=[];cores=[];core_errors=[];checks=[]
    for case in cases:
        name=case['name'];print('Processing '+name,flush=True)
        selections=[r for r in previous['phase_selection'] if r['case']==name]
        if name=='phantom_1m':source=dp['jet_values'];flows=[r for r in dp['section_values'] if r['dataset']=='cfd_voxel_common']
        elif name=='aorta_1m':source=ao['flow_core_series'];flows=[r for r in ao['section_series'] if r['method']=='CFD']
        else:source=ext['aorta_pilot_sections']['flow_cores'];flows=[r for r in ext['aorta_pilot_sections']['series'] if r['method']=='CFD']
        case['core_direction']={s:1 if sum(r['net_m3_s'] for r in flows if r['section']==s)>=0 else -1 for s in dict.fromkeys(r['section'] for r in flows)}
        transverse(case,profiles[name],selections,stage,inventory)
        reg,long=speed_tables(case);regional+=reg;longitudinal+=long
        longitudinal_plot(case,long,previous['cases'][name]['regional_selections'],stage,inventory);regional_plot(case,reg,stage,inventory)
        choices_plot(case,previous['analysis_choices'],stage,inventory)
        raw,pairs=core_plots(case,source,old['flow_core_comparison'],selections,stage,inventory);cores+=raw;core_errors+=pairs
        dr,db=derivative_rows(case);derivatives+=dr;derivative_bins+=db;derivative_plot(case,dr,stage,inventory)
        for row in dr:
            if row['choice']=='Base':
                expected=old['cases'][name]['gradients']['weighted_pooled'][row['metric']]['rmse']
                np.testing.assert_allclose(row['rmse'],expected,rtol=5e-7,atol=1e-9)
                checks.append(dict(case=name,metric=row['metric'],check='Primary derivative reproduces E038',passed=True))
    for filename,rows in [('regional_speeds.csv',regional),('longitudinal_profiles.csv',longitudinal),('derivative_sensitivity.csv',derivatives),('derivative_per_bin.csv',derivative_bins),('core_width.csv',cores),('core_errors.csv',core_errors),('analysis_choices.csv',previous['analysis_choices'])]:table(stage/filename,rows)
    report=dict(experiment='E-040',inputs=inputs,inventory=inventory,checks=checks,cases={case['name']:dict(core_direction=case['core_direction'],longitudinal_coordinate='MRI readout X; not aortic centerline',phase_selection=previous['cases'][case['name']]['regional_selections']) for case in cases},
        limitations=['Post hoc internal consistency, not independent validation','Phantom tail-overlap phases flagged','Aortic numerical support and anonymous sections are not physical lumen or branch labels','Phantom central lines and aortic strip profiles have different within-case definitions','Smoothing changes spatial bandwidth; thresholds change population','Missing cores and derivative stencils are not zero errors','No other phantom archive, new simulation or production overwrite'])
    dump(stage/'report.json',safe_json(report))
    lines=['# E-040: concise-legend comparison PNGs','','24 PNGs, eight per acquisition version. Method legends: CFD / Koma. Threshold legends:25% /50% /75%.','','| Case | PNG |','|---|---|']
    for item in inventory:lines.append('| '+item['file'].split('_')[0]+' | ['+item['file']+']('+item['file']+') |')
    lines+=['','Read the adjacent E040_PROTOCOL.md in the parent for selection/weighting definitions. All-bin numeric tables accompany the figures. Longitudinal aortic profiles are readout-axis aggregations and may combine branches; they are not anatomical centerlines. Gray time bands and [tail] mark the phantom synthetic source closure. Missing samples remain gaps; aortic extra-eroded derivatives remain unavailable. No new simulation, TKE, wall-shear, pressure or independent validation claim.']
    (stage/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(stage/'MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in stage.iterdir() if p.is_file()})
    if target.exists():raise FileExistsError(target)
    stage.rename(target);print('Saved '+str(target),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
