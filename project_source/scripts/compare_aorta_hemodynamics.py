"""E-027 original aortic point-cloud CFD versus preserved eight-bin Koma."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import tomllib
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree
from mrsimtracks.hemodynamics import (differential_metrics,jet_metrics,
    midpoint_cycle_integrals,region_kinetic_energy,section_flow,velocity_gradient,
    weighted_error_metrics)
from scripts.aorta_hemodynamic_reference import (AortaGrid,build_lookup,
    sequence_timing,temporal_node_values,temporal_weights)
from scripts.audit_aorta_encoding import audit as audit_encoding
from scripts.compare_dp150_hemodynamics import (paired_velocity_metrics,
    finite_summary,matched_smooth,safe_json)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'external_inputs/aorta_pointcloud/processed/dp2'
PIPE=ROOT/'outputs/aorta_dp2_mri_pipeline'
MULTI=PIPE/'koma/multibin_8x4mm'
RUN=MULTI/'simulation_motion_1m_8bin_4mm'
BASE=ROOT/'outputs/aorta_hemodynamic_comparison/e027_v1'
EXPECTED={'velocities.npy':'0dde75ca9b94a1aa6b795912bfc16bb0238f206fdcb0bc9336aaf4a91c5e37f6',
          'point_coordinates.npy':'21af8e15f9542add12320352706ba1d250e84b2c49f4a0035b34647b3e9f0197',
          'geometry.pkl':'dcef0f4debbb0196b6673268863466dee925b328d7dd62c713d00bede6dcae4f',
          'metadata.json':'ccb8055de178e3509574ed5b5a546cc1c7fa11a88d00373042bd817bd079c306'}
RECON_SHA='90b88e5e60b3ea250f462b2ed40254192b703e3a64b48a0fc9b3c1c8ec62cae7'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(safe_json(data),stream,indent=2,allow_nan=False); stream.write('\n')


def table(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        writer.writerows(safe_json(rows))


def section_components(fraction,grid,targets=(-.060,0,.060)):
    """Anonymous connected sections, separate for each limb and signed +X."""
    axes=grid.axes(); rows=[]
    for requested in targets:
        index=int(np.argmin(abs(axes[0]-requested)))
        components,n=label(fraction[:,:,index]>0,structure=np.ones((3,3)))
        found=[]
        for c in range(1,n+1):
            roi=components==c
            if roi.sum()<4:
                continue
            z,y=np.where(roi)
            found.append((float(np.mean(axes[1][y])),float(np.mean(axes[2][z])),roi))
        for rank,(cy,cz,roi) in enumerate(sorted(found,key=lambda p:p[:2]),1):
            rows.append(dict(id=f'X{round(axes[0][index]*1000):+04d}_C{rank}',
                             requested_x_m=requested,x_m=float(axes[0][index]),readout_index=index,
                             centroid_y_m=cy,centroid_z_m=cz,roi=roi))
    return rows


def masks_for(magnitude,stored,fraction):
    maximum=magnitude.reshape(len(magnitude),-1).max(axis=1)
    return {f'core{round(f*100):02d}':np.all(stored & (magnitude>=f*maximum[:,None,None,None]),axis=0)&(fraction>0)
            for f in (.05,.10,.15,.20)}


def derivative_set(ref,est,mask,spacing,sigma=None):
    series=[]; pooled={}; fields={}
    for b in range(len(ref)):
        a=ref[b]; c=est[b]
        if sigma:
            a=matched_smooth(a,mask,spacing,sigma_m=sigma)
            c=matched_smooth(c,mask,spacing,sigma_m=sigma)
        ga,valid=velocity_gradient(a,spacing,mask)
        gc,valid2=velocity_gradient(c,spacing,mask)
        valid &= valid2
        fa=differential_metrics(ga,a); fc=differential_metrics(gc,c)
        fa['strain_frobenius_s_inv']=np.sqrt(np.sum(fa['strain_rate_s_inv']**2,axis=(-2,-1)))
        fc['strain_frobenius_s_inv']=np.sqrt(np.sum(fc['strain_rate_s_inv']**2,axis=(-2,-1)))
        names=('vorticity_magnitude_s_inv','helicity_m_s_inv2','divergence_s_inv',
               'strain_frobenius_s_inv','dissipation_per_viscosity_s_inv2')
        for name in names:
            m=weighted_error_metrics(fa[name],fc[name],valid=valid)
            series.append(dict(bin=b+1,metric=name,**m))
            fields.setdefault(name,[[],[]])
            fields[name][0].append(fa[name][valid]); fields[name][1].append(fc[name][valid])
    for name,(a,c) in fields.items():
        pooled[name]=weighted_error_metrics(np.concatenate(a),np.concatenate(c))
    return dict(stencil_voxels_per_bin=int(valid.sum()),pooled=pooled,per_bin=series)


def figures(out,report,ref,est,mask,fraction,sections,grid,magnitude):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    times=np.array(report['time_s'])
    n=len(sections); fig,axes=plt.subplots(max(1,(n+1)//2),2,figsize=(12,3.1*max(1,(n+1)//2)),squeeze=False)
    for ax,section in zip(axes.ravel(),sections):
        rows=[x for x in report['section_series'] if x['section']==section['id']]
        for kind,color in [('CFD','tab:blue'),('Koma','tab:orange')]:
            ax.plot(times,[r['net_ml_s'] for r in rows if r['method']==kind],'-o',label=kind,color=color,ms=3)
        ax.set(title=section['id']+' | separate numerical-support section',xlabel='MRI cycle time (s)',ylabel='Signed partial flow (mL/s)')
        ax.axhline(0,color='.5',lw=.6); ax.legend(frameon=False); ax.grid(alpha=.2)
    for ax in axes.ravel()[n:]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(out/'section_flow_waveforms.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    whole=[x for x in report['regional_series'] if x['region']=='whole']
    for kind,color in [('CFD','tab:blue'),('Koma','tab:orange')]:
        rows=[x for x in whole if x['method']==kind]
        axes[0].plot(times,[r['mean_speed_m_s'] for r in rows],'-o',label=kind,color=color)
        axes[1].plot(times,[r['p99_speed_m_s'] for r in rows],'-o',label=kind,color=color)
        axes[2].plot(times,[1e6*r['energy_per_density_m5_s2'] for r in rows],'-o',label=kind,color=color)
    for ax,title,unit in zip(axes,['Mean speed','99th-percentile speed','Supported-voxel resolved KE / density'],['m/s','m/s','10^-6 m^5/s^2']):
        ax.set(title=title,xlabel='MRI cycle time (s)',ylabel=unit); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/'speed_energy_waveforms.png',dpi=180); plt.close(fig)
    b=int(np.argmax([r['p99_speed_m_s'] for r in whole if r['method']=='CFD']))
    weight=np.where(mask,fraction,0); denom=weight.sum(axis=0)
    projected=lambda u:np.divide(np.sum(np.where(mask,u,0)*weight,axis=0),denom,out=np.full(denom.shape,np.nan),where=denom>0)
    a=projected(np.linalg.norm(ref[b],axis=-1)); c=projected(np.linalg.norm(est[b],axis=-1))
    extent=[grid.axes()[0][0]*1000-grid.spacing[0]*500,grid.axes()[0][-1]*1000+grid.spacing[0]*500,
            grid.axes()[1][0]*1000-grid.spacing[1]*500,grid.axes()[1][-1]*1000+grid.spacing[1]*500]
    fig,axes=plt.subplots(3,1,figsize=(13,8))
    vmax=np.nanquantile(np.r_[a.ravel(),c.ravel()],.99)
    for ax,img,title in zip(axes,[a,c,c-a],['CFD speed','Koma speed','Koma minus CFD']):
        limit=np.nanmax(abs(c-a)) if title.startswith('Koma minus') else vmax
        h=ax.imshow(img,origin='lower',extent=extent,aspect='equal',cmap='RdBu_r' if title.startswith('Koma minus') else 'viridis',
                    vmin=-limit if title.startswith('Koma minus') else 0,vmax=limit)
        ax.set(title=f'{title} | bin {b+1} | support-weighted projection',xlabel='MRI readout X (mm)',ylabel='MRI phase Y (mm)')
        fig.colorbar(h,ax=ax,label='m/s',fraction=.025)
    fig.tight_layout(); fig.savefig(out/'velocity_comparison.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    transverse_extent=[grid.axes()[1][0]*1000-grid.spacing[1]*500,
                       grid.axes()[1][-1]*1000+grid.spacing[1]*500,
                       grid.axes()[2][0]*1000-grid.spacing[2]*500,
                       grid.axes()[2][-1]*1000+grid.spacing[2]*500]
    for ax,requested in zip(axes,(-.060,0,.060)):
        i=int(np.argmin(abs(grid.axes()[0]-requested)))
        ax.imshow(np.max(magnitude,axis=0)[:,:,i],origin='lower',cmap='gray',extent=transverse_extent)
        for section in sections:
            if section['readout_index']==i:
                ax.contour(section['roi'].astype(float),levels=[.5],colors=['tab:cyan'],origin='lower',extent=transverse_extent)
                ax.text(section['centroid_y_m']*1000,section['centroid_z_m']*1000,section['id'],color='yellow',fontsize=8,ha='center')
        ax.set(title=f'X = {grid.axes()[0][i]*1000:.1f} mm',xlabel='MRI phase Y (mm)',ylabel='MRI slice Z (mm)')
    fig.suptitle('Separate numerical-support components; anatomical branch names are unassigned')
    fig.tight_layout(); fig.savefig(out/'section_locations.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in ['adc_time_average','nominal_midpoint','box3','quadrature3','legacy_half_voxel']:
        rows=[x for x in report['sensitivity_series'] if x['reference']==method and x['mask']=='core15']
        axes[0].plot([r['bin'] for r in rows],[r['vector_rmse_m_s'] for r in rows],'-o',label=method)
    axes[0].set(xlabel='Cardiac bin',ylabel='Vector RMSE (m/s)',title='Reference-definition sensitivity'); axes[0].legend(fontsize=8)
    masks=report['mask_sensitivity']
    axes[1].bar([r['mask'] for r in masks],[100*r['pooled']['vector_nrmse_reference_rms'] for r in masks])
    axes[1].set(ylabel='CFD-RMS normalized vector error (%)',title='Fixed all-bin support sensitivity')
    fig.tight_layout(); fig.savefig(out/'sensitivity_comparison.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results')
    args=parser.parse_args(); target=args.output_dir.resolve()
    if target.exists(): raise FileExistsError('Refusing existing output')
    target.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); inputs=[]
    for name,expected in EXPECTED.items():
        path=DATA/name; actual=sha(path)
        if actual!=expected: raise ValueError('Original identity mismatch: '+name)
        inputs.append(dict(path=os.path.relpath(path,ROOT).replace('\\','/'),bytes=path.stat().st_size,sha256=actual))
    h5=RUN/'pc_reconstruction.h5'
    if sha(h5)!=RECON_SHA: raise ValueError('E-009 identity mismatch')
    with (MULTI/'motion_phantom_report.toml').open('rb') as stream: motion=tomllib.load(stream)
    period=motion['realized_sequence_rr_s']; scale=.72/period
    with h5py.File(h5,'r') as source:
        magnitude=source['reference_magnitude'][...].astype(float)
        stored=source['mask'][...].astype(bool)
        est=np.moveaxis(source['velocity'][...],0,-1).astype(float)
    grid=AortaGrid(center=tuple(motion['source_coordinate_center_m']),order=tuple(np.array(motion['source_axis_order_largest_to_smallest_one_based'])-1))
    coords=np.load(DATA/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    velocities=np.load(DATA/'velocities.npy',mmap_mode='r',allow_pickle=False)
    timing={}; records={}; weights={}
    for encoding in ('REF','X','Y','Z'):
        path=MULTI/f'sequences/{encoding.lower()}.seq'
        timing[encoding],weights[encoding],records[encoding]=sequence_timing(path,period)
        inputs.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    for e in ('X','Y','Z'):
        if not np.allclose(weights['REF'],weights[e],atol=1e-8):
            raise ValueError('Encoding time distributions differ; need component-specific reference')
    nominal=(np.arange(8)+.5)*period/8
    mid=np.stack([temporal_weights([t],period_s=period) for t in nominal])
    box=np.stack([temporal_weights((b+(np.arange(3)+.5)/3)*period/8,period_s=period) for b in range(8)])
    print('Verified inputs and acquisition timing; building numerical source lookup',flush=True)
    tree=cKDTree(np.asarray(coords,dtype=np.float32),copy_data=False)
    lookups={
        'primary':build_lookup(tree,grid,(2,2,2)),
        'quadrature3':build_lookup(tree,grid,(3,3,3)),
        'plane':build_lookup(tree,grid,(1,2,2)),
        'legacy_half_voxel':build_lookup(tree,AortaGrid(center=grid.center,order=grid.order,offset=.5),(2,2,2)),
    }
    allweights=np.concatenate([weights['REF'],mid,box])
    print('Averaging original CFD using actual saved acquisition times',flush=True)
    nodes=temporal_node_values(velocities,allweights,order=grid.order,scale=scale)
    references={'adc_time_average':lookups['primary'].evaluate(nodes[:8]),
                'nominal_midpoint':lookups['primary'].evaluate(nodes[8:16]),
                'box3':lookups['primary'].evaluate(nodes[16:24]),
                'quadrature3':lookups['quadrature3'].evaluate(nodes[:8]),
                'legacy_half_voxel':lookups['legacy_half_voxel'].evaluate(nodes[:8])}
    plane_ref=lookups['plane'].evaluate(nodes[:8]); del nodes,tree
    ref=references['adc_time_average']; fraction=lookups['primary'].fraction
    plane_fraction=lookups['plane'].fraction; masks=masks_for(magnitude,stored,fraction)
    common=masks['core15']; masks['eroded15']=binary_erosion(common,structure=np.ones((3,3,3)))
    finite_all=np.isfinite(ref).all(axis=(0,4))
    if not np.all(finite_all[common]): raise ValueError('Unpaired primary reference support')
    spatial_weight=np.broadcast_to(fraction,ref.shape[:-1])
    common4=np.broadcast_to(common,ref.shape[:-1]); spacing=grid.spacing
    report={'experiment':'E-027','created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'EXECUTED_DESCRIPTIVE_POINT_CLOUD_CFD_KOMA_COMPARISON',
            'evidence_class':'Internal consistency; approximate numerical support; no independent physical validation',
            'inputs':inputs,'grid':asdict(grid),'period_s':period,'source_velocity_time_scale':scale,
            'time_s':nominal,'timing_audit':timing,
            'encoding_moment_audit':audit_encoding(MULTI/'sequences',period),
            'primary_pooled':paired_velocity_metrics(ref,est,spatial_weight,common4),
            'coverage':dict(primary_voxels=int(common.sum()),stored_all_bin_voxels=int(np.all(stored,axis=0).sum()),
                            numerical_support_voxels=int((fraction>0).sum()),
                            numerical_support_volume_ml=float(fraction.sum()*np.prod(spacing)*1e6),
                            common_numerical_volume_ml=float(fraction[common].sum()*np.prod(spacing)*1e6),
                            fraction_definition='Nearest-point numerical support sampled by quadrature; physical lumen coverage unknown'),
            'mask_sensitivity':[],'sensitivity_series':[],'regional_series':[],
            'section_series':[],'section_summary':[],'flow_core_series':[],'profile_series':[]}
    for name,mask in masks.items():
        valid=np.broadcast_to(mask,ref.shape[:-1])
        report['mask_sensitivity'].append(dict(mask=name,voxels=int(mask.sum()),pooled=paired_velocity_metrics(ref,est,spatial_weight,valid)))
    for name,reference in references.items():
        valid=common&np.isfinite(reference).all(axis=(0,4))
        for b in range(8):
            report['sensitivity_series'].append(dict(reference=name,mask='core15',bin=b+1,
                **paired_velocity_metrics(reference[b],est[b],fraction,valid)))
    x=grid.points()[...,0]
    region_masks={'whole':common,'readout_below_minus45':common&(x<-.045),
                  'readout_middle':common&(x>=-.045)&(x<=.045),'readout_above45':common&(x>.045)}
    for region,mask in region_masks.items():
        for b in range(8):
            for method,field in [('CFD',ref),('Koma',est)]:
                summary=finite_summary(np.linalg.norm(field[b],axis=-1),fraction,mask)
                energy=region_kinetic_energy(field[b],fraction*np.prod(spacing),valid=mask)
                report['regional_series'].append(dict(region=region,bin=b+1,method=method,
                    mean_speed_m_s=summary['mean'],p99_speed_m_s=summary['p99'],max_speed_m_s=summary['max'],**energy))
    sections=section_components(plane_fraction,grid)
    point_grid=grid.points()
    for section in sections:
        i=section['readout_index']; roi=section['roi']; valid=roi&common[:,:,i]&(plane_fraction[:,:,i]>0)
        area=plane_fraction[:,:,i]*spacing[1]*spacing[2]
        q={}
        for method,field in [('CFD',ref),('Koma',est)]:
            rows=[]
            for b in range(8):
                flow=section_flow(field[b,:,:,i],area,valid=valid)
                report['section_series'].append(dict(section=section['id'],bin=b+1,method=method,
                    x_m=section['x_m'],net_ml_s=flow['net_m3_s']*1e6,
                    forward_ml_s=flow['forward_m3_s']*1e6,reverse_ml_s=flow['reverse_m3_s']*1e6,
                    sampled_area_mm2=flow['sampled_area_m2']*1e6,
                    numerical_section_area_mm2=float(area[roi].sum()*1e6),
                    **flow))
                rows.append(flow['net_m3_s'])
                for axis in (0,1):
                    for k in range(valid.shape[axis]):
                        line_mask=valid.copy()
                        if axis==0: line_mask[:k]=False; line_mask[k+1:]=False
                        else: line_mask[:,:k]=False; line_mask[:,k+1:]=False
                        w=area*line_mask; denom=w.sum()
                        value=float(np.sum(np.where(line_mask,field[b,:,:,i,0],0)*w)/denom) if denom else np.nan
                        report['profile_series'].append(dict(section=section['id'],bin=b+1,method=method,
                            transverse_axis='slice_Z' if axis==0 else 'phase_Y',
                            coordinate_mm=float(grid.axes()[2 if axis==0 else 1][k]*1000),normal_velocity_m_s=value))
            q[method]=np.array(rows)
        summary=dict(section=section['id'],x_m=section['x_m'],roi_pixels=int(roi.sum()),paired_pixels=int(valid.sum()),
                     normal='positive MRI X',flow_error=weighted_error_metrics(q['CFD'],q['Koma']),
                     CFD_cycle=midpoint_cycle_integrals(q['CFD'],period),Koma_cycle=midpoint_cycle_integrals(q['Koma'],period))
        den=summary['CFD_cycle']['net_m3']; summary['net_cycle_volume_ratio']=summary['Koma_cycle']['net_m3']/den if abs(den)>1e-12 else np.nan
        summary['peak_absolute_flow_ratio']=np.max(abs(q['Koma']))/np.max(abs(q['CFD'])) if np.max(abs(q['CFD']))>0 else np.nan
        report['section_summary'].append(summary)
        for b in range(8):
            absolute=abs(ref[b,:,:,i,0]); p99=finite_summary(absolute,area,valid)['p99']
            for factor in (.25,.50,.75):
                for sign in (-1,1):
                    for method,field in [('CFD',ref),('Koma',est)]:
                        jet=jet_metrics(field[b,:,:,i],point_grid[:,:,i],area,normal=(sign,0,0),valid=valid,minimum_forward_speed_m_s=factor*p99)
                        report['flow_core_series'].append(dict(section=section['id'],bin=b+1,threshold_fraction=factor,
                            threshold_m_s=factor*p99,direction_sign=sign,method=method,**jet))
    print('Computing interior gradients and sensitivity cases',flush=True)
    report['derivatives']={
        'primary':derivative_set(ref,est,common,spacing),
        'eroded':derivative_set(ref,est,masks['eroded15'],spacing),
        'matched_4mm_smoothing':derivative_set(ref,est,common,spacing,sigma=.004)}
    report['gated']={
        'WSS_TAWSS_OSI':'Native physical wall/labels and validated estimator absent',
        'pressure_and_pressure_drop':'Matching pressure fields, operating state and native linkage unavailable',
        'TKE':'Mean vector velocity is insufficient for fluctuation-sensitive MRI TKE',
        'physical_lumen_totals':'Numerical point-cloud support does not certify physical areas/volumes',
        'absolute_KE_and_dissipation':'Aortic density/viscosity unverified; property-normalized values reported',
        'independent_validation':'No independently measured waveform or velocity reference'}
    report['limitations']=[
        'ADC-time mean is a kinematic reference, not the complex MR transfer function.',
        'Point-cloud support and kNN8 interpolation approximate spatial physics.',
        'Section integrals are partial numerical-support estimates, not labeled inlet/outlet or branch splits.',
        'Signed +X sections are kept separate; no anatomy is inferred from component order.',
        'Legacy aortic runtime reset behavior is not certified by a sequence timing check.',
        'Source units are documented downstream; absolute solver phase and physical scanner registration remain unverified.',
        'Coarse approximately4mm sampling limits near-wall and derivative interpretation.',
        'No fitted alignment, amplitude, threshold or composite quality score.']
    report['runtime_s']=time.perf_counter()-start
    for path in [h5,MULTI/'motion_phantom_report.toml',PIPE/'aorta_motion_1m_tracking_report.json',
                 Path(__file__),ROOT/'scripts/aorta_hemodynamic_reference.py',ROOT/'scripts/audit_aorta_encoding.py',
                 ROOT/'scripts/compare_dp150_hemodynamics.py',ROOT/'scripts/dp150_hemodynamic_reference.py',
                 ROOT/'mrsimtracks/hemodynamics.py',BASE/'ANALYSIS_PROTOCOL.md']:
        report['inputs'].append(dict(path=os.path.relpath(path,ROOT).replace('\\','/'),bytes=path.stat().st_size,sha256=sha(path)))
    temporary=Path(tempfile.mkdtemp(prefix=target.name+'.partial-',dir=target.parent))
    with h5py.File(temporary/'matched_reference.h5','x') as f:
        for name,field in references.items(): f.create_dataset(name,data=field,compression='gzip')
        f.create_dataset('Koma',data=est,compression='gzip'); f.create_dataset('fraction',data=fraction,compression='gzip')
        f.create_dataset('plane_fraction',data=plane_fraction,compression='gzip')
        f.create_dataset('plane_point_reference',data=plane_ref,compression='gzip')
        for name,mask in masks.items(): f.create_dataset('masks/'+name,data=mask,compression='gzip')
        for section in sections: f.create_dataset('sections/'+section['id'],data=section['roi'],compression='gzip')
        f.create_dataset('frame_weights',data=allweights)
        f.attrs['dimension_order']='bin,slice,phase,readout,component'; f.attrs['grid_origin_offset_voxels']=0.
        f.attrs['area_volume_definition']='Numerical nearest-point support; not certified physical lumen'
    dump(temporary/'report.json',report)
    for key in ['section_series','regional_series','flow_core_series','profile_series','sensitivity_series']:
        rows=report[key]
        if key=='flow_core_series':
            rows=[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)}|{
                'centroid_X_m':r['centroid_m'][0],'centroid_Y_m':r['centroid_m'][1],'centroid_Z_m':r['centroid_m'][2]} for r in rows]
        if key=='sensitivity_series':
            rows=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
        table(temporary/(key+'.csv'),rows)
    figures(temporary,report,ref,est,common,fraction,sections,grid,magnitude)
    pooled=report['primary_pooled']; cf=np.mean([r['energy_per_density_m5_s2'] for r in report['regional_series'] if r['region']=='whole' and r['method']=='CFD'])
    km=np.mean([r['energy_per_density_m5_s2'] for r in report['regional_series'] if r['region']=='whole' and r['method']=='Koma'])
    lines=['# E-027: aortic CFD versus Koma hemodynamics','',report['evidence_class'],'',
           f"Vector RMSE: {pooled['vector_rmse_m_s']:.6f} m/s; reference-RMS normalized error: {100*pooled['vector_nrmse_reference_rms']:.2f}%.",
           f"Speed RMSE: {pooled['speed']['rmse']:.6f} m/s; bias: {pooled['speed']['bias']:.6f} m/s.",
           f"Mean supported-voxel resolved KE/density ratio Koma/CFD: {km/cf:.6f}.",
           f"Fixed15% support: {common.sum()} voxels. Physical lumen coverage is unknown.",'',
           '## Acquisition timing','',f"Saved ADC windows have a circular bin-center offset of approximately {(timing['REF']['bin_timing'][0]['phase_circular_mean_s']-nominal[0])*1000:.3f} ms from nominal centers.",
           f"RF-to-ADC boundary crossings across REF/X/Y/Z: {sum(t['RF_to_ADC_seam_crossings'] for t in timing.values())}. Runtime reset certification remains separate.",'',
           '## Separate numerical-support sections','',
           '| Section | Paired pixels | Q RMSE (mL/s) | Net cycle-volume ratio | Absolute peak-flow ratio |','|---|---:|---:|---:|---:|']
    for r in report['section_summary']:
        lines.append(f"| {r['section']} | {r['paired_pixels']} | {r['flow_error']['rmse']*1e6:.3f} | {r['net_cycle_volume_ratio']:.4f} | {r['peak_absolute_flow_ratio']:.4f} |")
    lines+=['','These are signed +MRI-X, partial support estimates; component labels do not establish anatomical branch identities.','',
            '## Interior derivatives','','| Quantity | RMSE | Reference-RMS normalized error |','|---|---:|---:|']
    for name,r in report['derivatives']['primary']['pooled'].items(): lines.append(f"| {name} | {r['rmse']:.6g} | {r['nrmse_reference_rms']*100:.2f}% |")
    lines+=['','Kinetic energy is per density; dissipation is per viscosity. Aortic material properties were not assumed.','',
            '## Sensitivities','','Actual ADC-time average, nominal midpoint, nominal3-sample box,2³/3³ quadrature, legacy half-voxel grid,5/10/15/20% masks, erosion and matched4mm smoothing are retained in JSON/HDF5.','',
            '## Unavailable comparisons','']+[f'- {k}: {v}.' for k,v in report['gated'].items()]+['','## Interpretation limits','']+['- '+v for v in report['limitations']]
    (temporary/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir() if p.is_file()}
    dump(temporary/'MANIFEST.json',manifest)
    if target.exists(): raise FileExistsError('Output appeared during run')
    temporary.rename(target)
    print(json.dumps(safe_json(dict(output=str(target),vector_rmse=pooled['vector_rmse_m_s'],
        vector_nrmse=pooled['vector_nrmse_reference_rms'],energy_per_density_ratio=km/cf,sections=len(sections),runtime_s=report['runtime_s'])),indent=2),flush=True)


if __name__=='__main__': main()
