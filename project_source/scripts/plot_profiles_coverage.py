"""E-039: immutable-input profiles, weighted distributions and error/coverage."""
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

from scripts.complete_advanced_comparisons import load_inputs, EXT
from scripts.compare_dp150_hemodynamics import matched_smooth, safe_json
from scripts.compare_aorta_hemodynamics import dump, table
from scripts.extend_hemodynamic_comparisons import ROOT, DP, AO, DP_H5, bind, sha

PARENT = ROOT/'outputs/hemodynamic_extension'
LABELS = {'phantom_1m': 'Stenotic phantom | 1M spins | 15 bins',
          'aorta_1m': 'Aorta baseline | 1M spins | 8 bins',
          'aorta_100k_corrected': 'Aorta corrected pilot | 100k spins | 8 bins'}
COLORS = {'CFD': '#176489', 'Koma': '#bf691b'}


def ecdf(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.shape != weights.shape:
        raise ValueError('Values and weights must have equal shapes')
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.array([]), np.array([])
    order = np.argsort(values[keep], kind='stable')
    x, w = values[keep][order], weights[keep][order]
    unique, first = np.unique(x, return_index=True)
    cumulative = np.cumsum(np.add.reduceat(w, first), dtype=float)
    return unique, cumulative/cumulative[-1]


def phase_bins(rows):
    ordered = sorted(rows, key=lambda r: r['bin'])
    if len({r['bin'] for r in ordered}) != len(ordered) or not ordered:
        raise ValueError('Missing or duplicate phase-selector bins')
    q = np.array([r['net_m3_s'] for r in ordered], float)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite section flow')
    return {name: ordered[int(fn(abs(q)))]['bin']
            for name, fn in [('peak', np.argmax), ('low', np.argmin)]}


def retained_coverage(mask, volume, denominator):
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Positive reference volume required')
    return float(np.sum(np.asarray(volume)[mask], dtype=float)/denominator*100)


def vector_error(reference, estimate, mask, volume, bins=None):
    bins = np.arange(len(reference)) if bins is None else np.asarray(bins)
    m = mask & np.isfinite(reference[bins]).all(axis=(0, 4)) & np.isfinite(estimate[bins]).all(axis=(0, 4)) & (volume > 0)
    if not m.any() or len(bins) == 0:
        return None, None, m
    a, c, w = reference[bins][:, m], estimate[bins][:, m], volume[m]
    mse = float(np.sum(np.sum((c-a)**2, axis=-1)*w)/(len(bins)*w.sum()))
    rms = float(np.sqrt(np.sum(np.sum(a*a, axis=-1)*w)/(len(bins)*w.sum())))
    rmse = np.sqrt(mse)
    return float(rmse), float(rmse/rms*100) if rms else None, m


def curl_magnitude(u, spacing, valid):
    """Six-neighbor central curl, computed only at qualified voxel centers."""
    mask = valid & np.isfinite(u).all(axis=-1)
    stencil = binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
    k, j, i = np.where(stencil)
    dx, dy, dz = spacing
    wx = (u[k,j+1,i,2]-u[k,j-1,i,2])/(2*dy) - (u[k+1,j,i,1]-u[k-1,j,i,1])/(2*dz)
    wy = (u[k+1,j,i,0]-u[k-1,j,i,0])/(2*dz) - (u[k,j,i+1,2]-u[k,j,i-1,2])/(2*dx)
    wz = (u[k,j,i+1,1]-u[k,j,i-1,1])/(2*dx) - (u[k,j+1,i,0]-u[k,j-1,i,0])/(2*dy)
    return np.sqrt(wx*wx+wy*wy+wz*wz), stencil


def gradient_error(case, mask, sigma=None):
    numerator = reference_square = weight_sum = 0.
    fixed = None
    for a, c in zip(case['reference'], case['estimate']):
        if sigma is not None:
            a, c = [matched_smooth(u, mask, case['spacing'], sigma_m=sigma) for u in (a, c)]
        va, ma = curl_magnitude(a, case['spacing'], mask)
        vc, mc = curl_magnitude(c, case['spacing'], mask)
        if not np.array_equal(ma, mc):
            raise ValueError('Different CFD/Koma stencil support')
        if fixed is not None and not np.array_equal(fixed, ma):
            raise ValueError('Time-varying derivative support')
        fixed = ma
        w = case['volume'][ma]
        numerator += float(np.sum((vc-va)**2*w))
        reference_square += float(np.sum(va*va*w))
        weight_sum += float(w.sum())
    rmse = float(np.sqrt(numerator/weight_sum)) if weight_sum else None
    nrmse = float(np.sqrt(numerator/reference_square)*100) if reference_square else None
    return rmse, nrmse, fixed


def read_profiles(dp, ao, ext):
    manifest = json.loads((DP/'MANIFEST.json').read_text())
    if isinstance(manifest, list):
        manifest = {r['file']: r for r in manifest}
    path = DP/'profiles.csv'
    bind(path, manifest['profiles.csv']['sha256'])
    with path.open(newline='', encoding='utf-8') as stream:
        dp_rows = list(csv.DictReader(stream))
    result = {'phantom_1m': [dict(section=r['section'], bin=int(r['bin']),
        method={'cfd': 'CFD', 'koma': 'Koma'}[r['dataset']],
        axis=r['profile_axis'][-1], coordinate_mm=float(r['coordinate_mm']),
        velocity=float(r['axial_velocity_m_s']) if r['axial_velocity_m_s'] else None)
        for r in dp_rows]}
    for name, rows in [('aorta_1m', ao['profile_series']),
                       ('aorta_100k_corrected', ext['aorta_pilot_sections']['profiles'])]:
        result[name] = [dict(section=r['section'], bin=r['bin'], method=r['method'],
            axis=(r.get('transverse_axis') or r.get('axis'))[-1],
            coordinate_mm=r['coordinate_mm'], velocity=r['normal_velocity_m_s']) for r in rows]
    return result


def save(fig, path):
    # Include every label/caption even when constrained_layout places a long
    # artist outside the nominal canvas. Do not crop scientific qualifications.
    fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight', pad_inches=.12)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def profile_plot(case, profiles, selections, out):
    sections = list(dict.fromkeys(r['section'] for r in selections))
    fig, axes = plt.subplots(len(sections), 4, figsize=(17, 2.5*len(sections)+1.1), layout='constrained', squeeze=False)
    selected_rows = []
    for row, section in enumerate(sections):
        chosen = {r['phase']: r for r in selections if r['section'] == section}
        section_rows = [r for r in profiles if r['section'] == section and r['bin'] in [v['bin'] for v in chosen.values()]]
        finite = [abs(r['velocity']) for r in section_rows if r['velocity'] is not None and np.isfinite(r['velocity'])]
        bound = max(finite, default=0)*1.1 or .01
        for col, (axis, phase) in enumerate([('Y','peak'),('Y','low'),('Z','peak'),('Z','low')]):
            ax = axes[row, col]; selected = chosen[phase]
            for method in ['CFD', 'Koma']:
                data = sorted([r for r in profiles if r['section'] == section and r['bin'] == selected['bin'] and r['axis'] == axis and r['method'] == method], key=lambda r:r['coordinate_mm'])
                if not data:
                    raise ValueError('Missing requested profile')
                ax.plot([r['coordinate_mm'] for r in data], [np.nan if r['velocity'] is None else r['velocity'] for r in data],
                    '-o' if method == 'CFD' else '--s', color=COLORS[method], ms=2.5, lw=1.3, label=method)
                selected_rows.extend(dict(case=case['name'], phase=phase, **r) for r in data)
            flag = ' [tail overlap]' if selected['synthetic_tail'] else ''
            ax.set(title=f"{section} | {phase} | bin {selected['bin']}{flag}\nt={selected['time_s']:.3f} s; |Q CFD|={abs(selected['CFD_flow_ml_s']):.2f} mL/s",
                   xlabel=f'MRI {axis} (mm)', ylabel='+MRI X velocity (m/s)', ylim=(-bound, bound))
            ax.axhline(0, color='.5', lw=.7); ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle(LABELS[case['name']]+'\nSection profiles at CFD-selected peak and low |net flow|', fontsize=14)
    detail = 'True central-voxel lines; gaps remain unavailable.' if case['name'].startswith('phantom') else 'Area-weighted strip means on partial anonymous sections, not inlet/branch identities; gaps remain unavailable.'
    fig.supxlabel(detail+'\nSame CFD/Koma bin; each section has a common velocity scale across both phases and axes. Internal consistency only.', fontsize=9)
    save(fig, out/(case['name']+'_peak_low_profiles'))
    return selected_rows


def distributions(case, selected, out):
    regions = case['regions']; ncols = 3 if len(regions) > 4 else 2
    nrows = int(np.ceil(len(regions)/ncols))
    speed_fig, axs = plt.subplots(nrows, ncols, figsize=(5.3*ncols, 3.4*nrows+1), layout='constrained', squeeze=False)
    error_fig, eas = plt.subplots(nrows, ncols, figsize=(5.3*ncols, 3.4*nrows+1), layout='constrained', squeeze=False)
    summaries = []; arrays = {}; selection = {r['phase']:r for r in selected}
    for index, (region, mask) in enumerate(regions.items()):
        ax, er = axs.flat[index], eas.flat[index]
        max_speed = max_error = 0.
        for phase, ls in [('peak','-'),('low','--')]:
            row = selection[phase]; b = row['bin']-1
            paired = mask & np.isfinite(case['reference'][b]).all(axis=-1) & np.isfinite(case['estimate'][b]).all(axis=-1) & (case['volume'] > 0)
            w = case['volume'][paired]
            for method, field in [('CFD',case['reference']),('Koma',case['estimate'])]:
                speed = np.linalg.norm(field[b,paired],axis=-1)
                x,y = ecdf(speed,w)
                arrays[f"{region}/{phase}/{method}_speed"] = np.column_stack((x,y))
                if len(x):
                    ax.step(np.r_[0,x],np.r_[0,y],where='post',color=COLORS[method],ls=ls,label=method+' '+phase,rasterized=True)
                    max_speed=max(max_speed,float(x[-1]))
                quantiles = {f'p{int(q*100):02}':float(x[min(np.searchsorted(y,q),len(x)-1)]) if len(x) else None for q in [.05,.25,.5,.75,.95,.99]}
                summaries.append(dict(case=case['name'],region=region,phase=phase,bin=b+1,method=method,
                    quantity='speed_m_s',voxels=int(paired.sum()),volume_ml=float(w.sum()*1e6),**quantiles))
            delta = np.linalg.norm(case['estimate'][b,paired]-case['reference'][b,paired],axis=-1)
            x,y=ecdf(delta,w); arrays[f'{region}/{phase}/vector_error']=np.column_stack((x,y))
            if len(x):
                er.step(np.r_[0,x],np.r_[0,y],where='post',color='#176489' if phase=='peak' else '#bf691b',ls=ls,label=phase,rasterized=True)
                max_error=max(max_error,float(x[-1]))
            quantiles={f'p{int(q*100):02}':float(x[min(np.searchsorted(y,q),len(x)-1)]) if len(x) else None for q in [.05,.25,.5,.75,.95,.99]}
            summaries.append(dict(case=case['name'],region=region,phase=phase,bin=b+1,method='Koma minus CFD',quantity='vector_error_m_s',voxels=int(paired.sum()),volume_ml=float(w.sum()*1e6),**quantiles))
        # Independent region scales are explicit; peak/low and CFD/Koma share
        # exactly the same axes within each regional panel.
        region_label={'readout_below_minus45':'MRI X < -45 mm','readout_middle':'MRI X: -45 to +45 mm','readout_above45':'MRI X > +45 mm'}.get(region,region.replace('_',' '))
        for target, title, label, bound in [(ax, 'Speed', 'Speed (m/s)', max_speed), (er,'Vector error','||Koma - CFD|| (m/s)',max_error)]:
            target.set(title=f"{region_label} | paired {case['volume'][mask].sum()*1e6:.2f} mL",xlabel=label,ylabel='Cumulative volume fraction',ylim=(0,1.02),xlim=(0,max(bound*1.03,1e-8)))
            target.grid(alpha=.18)
    for axes in [axs,eas]:
        for extra in axes.flat[len(regions):]:extra.set_visible(False)
        axes.flat[0].legend(frameon=False,fontsize=8)
    context = '; '.join(f"{phase}: bin {r['bin']} ({r['time_s']:.3f} s)" for phase,r in selection.items())
    for fig,title in [(speed_fig,'Regional speed distributions'),(error_fig,'Regional vector-error distributions')]:
        fig.suptitle(LABELS[case['name']]+'\n'+title+' | '+context,fontsize=13)
        fig.supxlabel('Exact voxel-volume-weighted ECDF; region-specific horizontal scales; shared peak/low scale within each panel.\nWhole overlaps subregions; curves are distributions, not confidence intervals. Internal consistency only.'+('\nLow-flow bin15 overlaps the synthetic CFD-input closure, not an independently simulated native phase.' if any(r['synthetic_tail'] for r in selected) else ''),fontsize=9)
    save(speed_fig,out/(case['name']+'_regional_speed_distributions'))
    save(error_fig,out/(case['name']+'_regional_error_distributions'))
    return summaries,arrays


def sensitivity(case, dp, ao, ext, references, out):
    name=case['name']; ref=case['reference']; est=case['estimate']; v=case['volume']; primary=case['regions']['whole']
    denominator=float(v.sum()); entries=[]
    def record(family, choice, metric, rmse, nrmse, mask, bins):
        entries.append(dict(case=name,family=family,choice=choice,metric=metric,rmse=rmse,nrmse_pct=nrmse,
            voxels=int(mask.sum()),retained_volume_ml=float(v[mask].sum()*1e6),denominator_volume_ml=denominator*1e6,
            retained_coverage_pct=retained_coverage(mask,v,denominator),retained_bins=bins,total_bins=len(ref),
            status='AVAILABLE' if rmse is not None else 'UNAVAILABLE_EMPTY_STENCILS'))
    def velocity(family, choice, a, c, m, bins=None):
        rmse,nrmse,actual=vector_error(a,c,m,v,bins)
        record(family,choice,'velocity_vector_m_s',rmse,nrmse,actual,len(ref) if bins is None else len(bins))
    if name.startswith('phantom'):
        with h5py.File(DP/'matched_reference.h5') as f:
            direct=f['direct_common_mask'][:]
            temporal=f['cfd_voxel_three_sample_mean'][:].astype(float)
            spatial=f['cfd_voxel_3x3x3'][:].astype(float)
        with h5py.File(DP_H5) as f:raw=np.moveaxis(f['velocity_raw_m_per_s'][:],0,-1).astype(float)
        geometric=case['gradient_mask']; eroded=binary_erosion(geometric)
        for choice,m,c in [('Hybrid common',primary,est),('Raw direct only',direct,raw),('Regularized only',primary&~direct,est),('Full-voxel interior',geometric,est),('Eroded interior',eroded,est)]:velocity('support',choice,ref,c,m)
        velocity('support','Exclude synthetic-tail bins',ref,est,primary,np.arange(13))
        for choice,a in [('Midpoint (primary)',ref),('Three-sample box',temporal),('3-cubed reference',spatial)]:velocity('reference',choice,a,est,primary)
        gradient_choices=[('Primary interior',geometric,None),('Direct-only stencil',geometric&direct,None),('Extra erosion',eroded,None),('Matched 1 mm smoothing',geometric,.001)]
    else:
        if name=='aorta_1m':
            with h5py.File(AO/'matched_reference.h5') as f:masks={k:f['masks/'+k][:] for k in f['masks']}
        else:
            from scripts.extend_hemodynamic_comparisons import PILOT
            from scripts.compare_aorta_hemodynamics import masks_for
            with h5py.File(PILOT/'pc_reconstruction.h5') as f:masks=masks_for(f['reference_magnitude'][:],f['mask'][:],v/np.prod(case['spacing']))
            masks['eroded15']=binary_erosion(masks['core15'],structure=np.ones((3,3,3)))
        for key in ['core05','core10','core15','core20','eroded15']:velocity('support',key,ref,est,masks[key])
        for key,a in references.items():velocity('reference',key,a,est,primary)
        gradient_choices=[('Primary core15',primary,None),('Extra erosion',masks['eroded15'],None),('Matched 4 mm smoothing',primary,.004)]
    for choice,m,sigma in gradient_choices:
        rmse,nrmse,actual=gradient_error(case,m,sigma)
        record('derivative',choice,'vorticity_magnitude_s_inv',rmse,nrmse,actual,len(ref))
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    for row,family in enumerate(['support','reference','derivative']):
        data=[r for r in entries if r['family']==family]; pos=np.arange(len(data))
        display={'core05':'5% magnitude core','core10':'10% magnitude core','core15':'15% magnitude core','core20':'20% magnitude core','eroded15':'Eroded 15% core','adc_time_average':'Actual ADC-time average','nominal_midpoint':'Nominal midpoint','box3':'Three-sample box','quadrature3':'3-cubed reference','legacy_half_voxel':'Historical half-voxel reference'}
        labels=[display.get(r['choice'],r['choice']) for r in data]
        left,right=axes[row]
        for i,r in enumerate(data):
            if r['nrmse_pct'] is None:left.text(.02,i,'Unavailable (no stencils)',transform=left.get_yaxis_transform(),va='center',fontsize=9)
            else:left.barh(i,r['nrmse_pct'],color='#176489',height=.58);left.text(r['nrmse_pct'],i,f" {r['nrmse_pct']:.1f}%",va='center',fontsize=9)
            right.barh(i,r['retained_coverage_pct'],color='#bf691b',height=.58)
            right.text(r['retained_coverage_pct'],i,f" {r['retained_coverage_pct']:.1f}% | n={r['voxels']:,}; {r['retained_bins']}/{r['total_bins']} bins",va='center',fontsize=8)
        left.set(yticks=pos,yticklabels=labels,xlabel=('Vorticity' if family=='derivative' else 'Velocity-vector')+' NRMSE (% of CFD RMS)',title={'support':'Support / phase inclusion','reference':'Reference definition (fixed primary weights)','derivative':'Central-stencil / matched-smoothing sensitivity'}[family])
        max_error=max([r['nrmse_pct'] or 0 for r in data]+[1]);left.set_xlim(0,max_error*1.24)
        right.set(yticks=pos,yticklabels=[],xlabel='Retained numerical-support volume (%)',xlim=(0,170),title='Retained coverage + voxel/bin counts')
        right.set_xticks([0,25,50,75,100]);right.axvline(100,color='.6',lw=.7,ls=':')
        for ax in [left,right]:ax.set_ylim(len(data)-.5,-.5);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(LABELS[name]+'\nError and retained coverage versus analysis choice',fontsize=14)
    fig.supxlabel(f"Coverage denominator: original 2-cubed numerical CFD support = {denominator*1e6:.2f} mL; not certified physical lumen.\nLower error after exclusion or smoothing is not proof of better recovery. Null error is not zero.\nReference sensitivities retain original 2-cubed weights. Internal consistency only.",fontsize=9)
    save(fig,out/(name+'_error_and_coverage'))
    return entries


def main(output):
    if output.exists():raise FileExistsError(output)
    inputs,cases,dp,ao,ext,references,_=load_inputs()
    inputs += [bind(Path(__file__)),bind(PARENT/'E039_PROTOCOL.md'),bind(DP/'profiles.csv')]
    old_report=PARENT/'e038_v1/report.json'
    manifest=json.loads((old_report.parent/'MANIFEST.json').read_text())
    inputs.append(bind(old_report,manifest['report.json']['sha256']))
    old=json.loads(old_report.read_text())
    profiles=read_profiles(dp,ao,ext)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+'.partial-',dir=output.parent))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    selections=[]; selected_profiles=[]; distributions_rows=[]; choices=[]; checks=[]
    report={'experiment':'E-039','evidence_class':'Post hoc internal consistency; no new simulation or independent physical validation','inputs':inputs,'cases':{},'coverage_denominator':'Original 2-cubed full numerical CFD support volume, not certified physical lumen'}
    with h5py.File(stage/'distribution_ecdfs.h5','w') as curves:
        for case in cases:
            name=case['name'];print('Processing '+name,flush=True)
            if name.startswith('phantom'):
                flow=[dict(r,method='CFD') for r in dp['section_values'] if r['dataset']=='cfd_voxel_common'];anchor='upstream'
            elif name=='aorta_1m':flow=[r for r in ao['section_series'] if r['method']=='CFD'];anchor='X+000_C1'
            else:flow=[r for r in ext['aorta_pilot_sections']['series'] if r['method']=='CFD'];anchor='X+000_C1'
            local=[]
            for section in dict.fromkeys(r['section'] for r in flow):
                rows=[r for r in flow if r['section']==section]
                if len(rows)!=len(case['times']):raise ValueError('Incomplete section waveform')
                for phase,b in phase_bins(rows).items():
                    row=dict(case=name,section=section,phase=phase,bin=b,time_s=float(case['times'][b-1]),CFD_flow_ml_s=float(next(r['net_m3_s'] for r in rows if r['bin']==b)*1e6),synthetic_tail=name.startswith('phantom') and b>=14)
                    local.append(row)
            selections.extend(local)
            selected_profiles += profile_plot(case,profiles[name],local,stage)
            selected=[r for r in local if r['section']==anchor]
            stats,arrays=distributions(case,selected,stage);distributions_rows+=stats
            for key,array in arrays.items():curves.create_dataset(name+'/'+key,data=array,compression='gzip')
            rows=sensitivity(case,dp,ao,ext,references,stage);choices+=rows
            primary=[r for r in rows if r['family']=='derivative'][0]
            expected=old['cases'][name]['gradients']['weighted_pooled']['vorticity_magnitude_s_inv']['rmse']
            np.testing.assert_allclose(primary['rmse'],expected,rtol=2e-7,atol=1e-9)
            checks.append(dict(case=name,check='Primary weighted curl reproduces E038',passed=True,value=primary['rmse']))
            report['cases'][name]=dict(label=LABELS[name],regional_phase_anchor=anchor,regional_selections=selected,
                profile_definition='True central-voxel line' if name.startswith('phantom') else 'Partial-section area-weighted strip mean',
                numerical_support_volume_ml=float(case['volume'].sum()*1e6),region_names=list(case['regions']))
    for filename,rows in [('phase_selection.csv',selections),('selected_profiles.csv',selected_profiles),('regional_distribution_summary.csv',distributions_rows),('analysis_choices.csv',choices)]:table(stage/filename,rows)
    report.update(phase_selection=selections,analysis_choices=choices,arithmetic_checks=checks,limitations=[
        'No independent validation, wall shear, TKE or pressure result is created.',
        'Aortic sections and regional bounds are computational, not anatomical labels.',
        'Phantom synthetic-tail bins14/15 remain explicitly flagged.',
        'Mask exclusion changes the sampled population; matched smoothing changes the observable.',
        'Region distributions use one CFD-selected anchor phase, not each region independently optimized.',
        'Legacy1M and corrected100k are not a controlled spin-count experiment.'])
    dump(stage/'report.json',safe_json(report))
    (stage/'README.md').write_text('# E-039: profiles, distributions and error–coverage\n\n'
        'Twelve PNG/PDF figure pairs: three cases, each with peak/low profiles, regional speed ECDFs, regional vector-error ECDFs and error–coverage panels.\n\n'
        'Peak/low are the maximum/minimum absolute CFD net flow among saved bins, section by section; Koma uses the same bins. Regional ECDFs use upstream phantom / anonymous aortic X+000_C1 as anchors. Exact selections, times and synthetic-tail flags are in phase_selection.csv.\n\n'
        'Phantom profiles are central-voxel lines. Aortic profiles are partial-section area-weighted strip means, not wall-to-wall physical profiles. Missing profile values remain gaps. ECDFs are volume-weighted distributions, not confidence intervals; exact curves are in distribution_ecdfs.h5.\n\n'
        'Coverage is retained volume divided by the original 2-cubed full numerical CFD support volume, not physical lumen coverage. Voxel/bin counts are separate. Reference-only tests preserve primary support and original weights. Raw-direct phantom velocity and hybrid derivatives restricted to direct stencils are distinct. Zero retained stencils means unavailable error.\n\n'
        'Plots do not select an optimal threshold/smoothing setting. Internal consistency only; source/production outputs are unchanged. See report.json for hashes, definitions and limitations.\n',encoding='utf-8')
    dump(stage/'MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in stage.iterdir() if p.is_file()})
    if output.exists():raise FileExistsError(output)
    stage.rename(output);print('Saved '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output.resolve())
