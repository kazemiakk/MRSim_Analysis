"""Native-field wall, gradient, energy and ellipsoidal-interface measurements."""
import argparse,time
from pathlib import Path
import h5py,numpy as np,pyvista as pv
from extended_common import *


def export_context(g,bands,out):
    plot=pv.Plotter(off_screen=True,window_size=(1400,900));plot.set_background('white')
    for name,z in g.zones.items():
        surf=pv.PolyData(g.xyz*1000,np.c_[np.full(len(z['tri']),3),z['tri']].ravel())
        plot.add_mesh(surf,color='#94a3b8' if name=='wall' else '#7c3aed',opacity=.15 if name=='wall' else .7)
        if name=='wall':
            for side,color in [('right','#2563eb'),('left','#d97706')]:
                selected=surf.extract_cells(bands['ICA_wall_3_to_8_mm'][side]);selected.save(out/(side+'-ICA-normalization-reference.vtu'))
                plot.add_mesh(selected,color=color)
    plot.add_text('Reference wall bands: B1 right (blue), B2 left (orange)',font_size=14,color='#172554')
    plot.add_text('3–8 mm geodesic distance from inlet rims; purple = existing sac wall',position='lower_left',font_size=10,color='#172554')
    plot.view_xz();plot.reset_camera();plot.screenshot(out/'parent-wall-reference-overlay.png');plot.close()
    tri=g.faces[g.interface]
    surface=pv.PolyData(g.xyz*1000,np.c_[np.full(len(tri),3),tri].ravel())
    surface.cell_data['OutwardFromROI_sign']=g.roi_flux_sign
    surface.save(out/'ellipsoidal-fluid-interface-NOT-anatomical-neck.vtp')


def analyze(out):
    out.mkdir(exist_ok=False);g=Geometry();bands=g.reference_bands();export_context(g,bands,out)
    w=weights(g.times);assert abs(w.sum()-PERIOD)<1e-12
    rows=[];exchange=[];wall_rows=[];hotspot=[];spatial_rows=[];period_rows=[];normalization=[]
    oldfields=np.load(PREVIOUS/'comparisons/wall-cycle-fields.npz')
    ps_magnitude={};snapshot_index={}
    for k in KINDS:
        cfg=g.cfg[k];ts=np.asarray(cfg['time_s']);total=sum(np.asarray(v['flow_m3_s']) for v in cfg['inlets'].values())
        target=ts[np.argmax(total)];snapshot_index[k]=int(np.argmin(abs(g.times-target)))
    for kind in KINDS:
        tau_stack={z:[] for z in g.zones}
        history=[]
        for j,t in enumerate(g.times):
            v,bc,flux=g.fields(kind,j);grad=g.gradient(v,bc)
            sym=(grad+grad.transpose(0,2,1))/2
            anti=(grad-grad.transpose(0,2,1))/2
            q=.5*(np.sum(anti*anti,axis=(1,2))-np.sum(sym*sym,axis=(1,2)))
            omega=np.column_stack([grad[:,2,1]-grad[:,1,2],grad[:,0,2]-grad[:,2,0],grad[:,1,0]-grad[:,0,1]])
            omega2=np.sum(omega*omega,axis=1);speed2=np.sum(v*v,axis=1)
            dissip=2*cfg['viscosity_pa_s']*np.sum(sym*sym,axis=(1,2))
            hel=np.einsum('ij,ij->i',v,omega)
            normalized=np.full(g.n,np.nan);valid=(speed2>1e-12)&(omega2>1e-12)
            normalized[valid]=hel[valid]/np.sqrt(speed2[valid]*omega2[valid])
            for region,mask in g.masks.items():
                vol=g.volume[mask];r={'case':kind,'time_s':float(t),'region':region,
                    'kinetic_energy_J':float(.5*cfg['density_kg_m3']*np.dot(vol,speed2[mask])),
                    'viscous_dissipation_W':float(np.dot(vol,dissip[mask])),
                    'vorticity_RMS_per_s':float(np.sqrt(np.average(omega2[mask],weights=vol))),
                    'enstrophy_m3_per_s2':float(.5*np.dot(vol,omega2[mask])),
                    'Q_mean_per_s2':float(np.average(q[mask],weights=vol)),
                    'positive_Q_volume_fraction':float(vol[q[mask]>0].sum()/vol.sum()),
                    'helicity_mean_m_per_s2':float(np.average(hel[mask],weights=vol)),
                    'mean_abs_normalized_helicity':float(np.average(np.abs(normalized[mask&valid]),weights=g.volume[mask&valid])) if (mask&valid).any() else None,
                    'gradient_divergence_RMS_per_s':float(np.sqrt(np.average(np.trace(grad[mask],axis1=1,axis2=2)**2,weights=vol)))}
                rows.append(r);history.append(r)
            outward=flux[g.interface]*g.roi_flux_sign/cfg['density_kg_m3']*1e6
            qin=float(-outward[outward<0].sum());qout=float(outward[outward>0].sum())
            vol_ml=float(g.volume[g.masks['sac_ellipsoid']].sum()*1e6)
            # Direct, native mass flux: inward B1/B2 must be negative in c0 orientation.
            for inlet in ['inlet_b1','inlet_b2']:
                a,b=g.face_zones[inlet];assert flux[a:b].sum()<0
            exchange.append({'case':kind,'time_s':float(t),'ROI_inflow_ml_s':qin,'ROI_outflow_ml_s':qout,
                'net_outflow_ml_s':qout-qin,'ROI_volume_ml':vol_ml,
                'instantaneous_volume_over_inflow_s':vol_ml/qin if qin>1e-12 else None,
                'scope':'Stair-step boundary of approved ellipsoidal fluid-cell ROI; NOT anatomical neck flow or particle residence.'})
            tau=g.wall.shear_vectors(g.steps[kind][j]['output_stem']+'.dat.h5',g.zones)
            for z in g.zones:tau_stack[z].append(tau[z])
            if j==snapshot_index[kind]:
                roi=g.grid.extract_cells(g.masks['sac_ellipsoid']);m=g.masks['sac_ellipsoid']
                for name,a in [('Speed_m_s',np.sqrt(speed2)),('Q_per_s2',q),('Vorticity_per_s',np.sqrt(omega2)),('Dissipation_W_m3',dissip),('Normalized_helicity',normalized)]:roi.cell_data[name]=a[m]
                roi.save(out/(kind+'-own-inlet-peak-ROI-fields.vtu'))
            if j%15==0:print('EXTENDED_FIELDS',kind,j,95,flush=True)
        for z,zone in g.zones.items():
            tau=np.asarray(tau_stack[z]);mag=np.linalg.norm(tau,axis=2);ar=zone['area'];cent=zone['centroid']*1000
            meanvec=np.einsum('t,tni->ni',w/PERIOD,tau)
            tawss=np.einsum('t,tn->n',w/PERIOD,mag);meanmag=np.linalg.norm(meanvec,axis=1)
            osi=.5*(1-meanmag/np.maximum(tawss,1e-30))
            assert np.max(abs(tawss-oldfields[kind+'_'+z+'_tawss']))<1e-9
            assert np.max(abs(osi-oldfields[kind+'_'+z+'_osi']))<1e-10
            valid=meanmag>1e-6;direction=np.zeros_like(meanvec);direction[valid]=meanvec[valid]/meanmag[valid,None]
            transverse=np.cross(zone['normal'],direction)
            tw=np.einsum('t,tn->n',w/PERIOD,np.abs(np.einsum('tni,ni->tn',tau,transverse)))
            tw[~valid]=np.nan;rrt=np.full(len(ar),np.nan);rrt[valid]=1/meanmag[valid]
            for band in bands:
                wall_tau=np.asarray(tau_stack['wall']);wm=np.linalg.norm(wall_tau,axis=2)
                means={side:float(np.dot(w/PERIOD,np.average(wm[:,mask],axis=1,weights=g.zones['wall']['area'][mask]))) for side,mask in bands[band].items()}
                reference=means['pooled'];assert reference>0
                normalization.append({'case':kind,'wall_region':z,'reference_band':band,
                    'right_reference_TAWSS_pa':means['right'],'left_reference_TAWSS_pa':means['left'],'pooled_reference_TAWSS_pa':reference,
                    'raw_mean_TAWSS_pa':float(np.average(tawss,weights=ar)),
                    'normalized_mean_TAWSS':float(np.average(tawss/reference,weights=ar)),
                    'low_shear_area_fraction_below_10percent_reference':float(ar[tawss<.1*reference].sum()/ar.sum()),
                    'high_shear_area_fraction_above_2times_reference':float(ar[tawss>2*reference].sum()/ar.sum()),
                    'threshold_scope':'Operational sensitivity thresholds, not clinical risk cutoffs.'})
            wall_rows.append({'case':kind,'region':z,'mean_transverse_WSS_pa':float(np.average(tw[valid],weights=ar[valid])),
                'mean_RRT_per_Pa':float(np.average(rrt[valid],weights=ar[valid])),
                'RRT_p95_per_Pa':g.wall.weighted_quantile(rrt[valid],ar[valid],.95),
                'undefined_direction_or_RRT_area_fraction':float(ar[~valid].sum()/ar.sum()),'mean_vector_floor_Pa':1e-6,
                'RRT_scope':'1/|mean shear vector|, units 1/Pa; not particle residence in seconds.'})
            fields=pv.PolyData(g.xyz*1000,np.c_[np.full(len(zone['tri']),3),zone['tri']].ravel())
            fields.cell_data['TAWSS_Pa']=tawss;fields.cell_data['OSI']=osi;fields.cell_data['Transverse_WSS_Pa']=tw;fields.cell_data['RRT_per_Pa']=rrt
            fields.save(out/(kind+'-'+z+'-extended-wall-fields.vtp'))
            if kind=='patient_specific':ps_magnitude[z]=mag.copy()
            for percentile in [.8,.9,.95]:
                threshold=g.wall.weighted_quantile(oldfields['patient_specific_'+z+'_tawss'],ar,percentile)
                for j,t in enumerate(g.times):
                    active=mag[j]>threshold;ref=ps_magnitude[z][j]>threshold
                    union=float(ar[active|ref].sum());intersection=float(ar[active&ref].sum())
                    ca=np.average(cent[active],axis=0,weights=ar[active]) if active.any() else None
                    cb=np.average(cent[ref],axis=0,weights=ar[ref]) if ref.any() else None
                    hotspot.append({'case':kind,'region':z,'time_s':float(t),'PS_TAWSS_reference_quantile':percentile,
                        'fixed_WSS_threshold_pa':threshold,'high_WSS_area_fraction':float(ar[active].sum()/ar.sum()),
                        'high_WSS_IoU_vs_MRI_percent':100*intersection/union if union else None,
                        'high_WSS_centroid_displacement_vs_MRI_mm':float(np.linalg.norm(ca-cb)) if ca is not None and cb is not None else None,
                        'centroid_x_mm':float(ca[0]) if ca is not None else None,'centroid_y_mm':float(ca[1]) if ca is not None else None,
                        'centroid_z_mm':float(ca[2]) if ca is not None else None,
                        'interpretation':'High-WSS centroid, NOT a validated impingement location.'})
        for region in g.masks:
            records=[r for r in history if r['region']==region]
            record={'case':kind,'region':region,'periodicity_validated':False}
            for name in ['kinetic_energy_J','viscous_dissipation_W','enstrophy_m3_per_s2','Q_mean_per_s2','positive_Q_volume_fraction']:
                record['cycle_mean_'+name]=float(np.dot(w/PERIOD,[r[name] for r in records]))
            period_rows.append(record)
        del tau_stack
    csvsave(out/'vortex-energy-dissipation-all-times.csv',rows)
    csvsave(out/'vortex-energy-dissipation-cycle-summary.csv',period_rows)
    csvsave(out/'ellipsoidal-ROI-exchange-NOT-anatomical-neck.csv',exchange)
    csvsave(out/'directional-WSS-and-RRT-summary.csv',wall_rows)
    csvsave(out/'normalized-WSS-reference-sensitivity.csv',normalization)
    csvsave(out/'moving-high-WSS-regions-and-threshold-sensitivity.csv',hotspot)
    save(out/'method-and-geometry.json',{'source_run':str(RUN),'new_CFD_cycles':0,'periodicity_validated':False,
        'cells':g.n,'ROI_interface_faces':len(g.interface),'gradient_method':'Inverse-distance-squared weighted least squares on four original tetrahedron face neighbors, using native boundary-face velocities.',
        'affine_gradient_test_max_absolute_error':g.gradient_error,
        'gradient_limits':'Exact affine-field reconstruction is an implementation test, not mesh-convergence validation. Derivative-based vortex/dissipation quantities are mesh sensitive.',
        'WSS_reference':'Provisional ICA wall bands at 3–8 mm geodesic distance from inlet rims, with 5–10 mm sensitivity. Not a clinical standard.',
        'exchange_boundary':'Original native internal faces separating approved ellipsoid-centroid fluid ROI from complementary fluid. No anatomical neck is claimed.',
        'cycle_integral':'Actual nonuniform 5–868 ms samples plus assumed 5-ms linear closure; first-cycle startup and seam remain.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.out/'native-analyses')
