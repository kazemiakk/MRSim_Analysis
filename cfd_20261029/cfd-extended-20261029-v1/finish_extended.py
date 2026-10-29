"""Figures, scoped independent checks and an honest completion/blocked inventory."""
import argparse,csv,time,platform
from pathlib import Path
import h5py,numpy as np,pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extended_common import *

COL={'patient_specific':'#2563eb','individualized':'#d97706','generic':'#0d9488'}
STYLE={'patient_specific':'-','individualized':'--','generic':':'}
NAME={'patient_specific':'Patient-specific','individualized':'Individualized','generic':'Population average'}


def read(path):return list(csv.DictReader(Path(path).open()))


def polish(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(True,color='#e2e8f0',lw=.7);ax.set_axisbelow(True)
    ax.set_xlim(5,868);ax.set_xticks([5,200,400,600,868]);ax.set_xlabel('Time (ms)')


def make_figures(out):
    figdir=out/'figures';figdir.mkdir(exist_ok=False);native=out/'native-analyses'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12})
    rows=read(native/'vortex-energy-dissipation-all-times.csv')
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');vol=mapping['volume_m3'];mask=mapping['ellipsoid_mask']
    volume={'sac_ellipsoid':float(vol[mask].sum()),'surrounding_fluid':float(vol[~mask].sum())}
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained',sharey='row')
    for col,region in enumerate(volume):
        for row,(metric,ylabel) in enumerate([('kinetic_energy_J','Kinetic energy density (J/m³)'),('viscous_dissipation_W','Dissipation density (W/m³)')]):
            ax=axes[row,col]
            for k in KINDS:
                r=[r for r in rows if r['case']==k and r['region']==region]
                ax.plot([float(x['time_s'])*1000 for x in r],[float(x[metric])/volume[region] for x in r],STYLE[k],color=COL[k],lw=2,label=NAME[k])
            polish(ax);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0)
            if row==0:ax.set_title('Ellipsoidal sac ROI' if col==0 else 'Complementary vessels');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Energy and dissipation — first-cycle exploratory fields',fontsize=14)
    fig.savefig(figdir/'regional-energy-and-dissipation.png',dpi=170);plt.close(fig)
    ex=read(native/'ellipsoidal-ROI-exchange-NOT-anatomical-neck.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,['ROI_inflow_ml_s','ROI_outflow_ml_s'],['Flow entering the ROI','Flow leaving the ROI']):
        for k in KINDS:
            r=[x for x in ex if x['case']==k];ax.plot([float(x['time_s'])*1000 for x in r],[float(x[key]) for x in r],STYLE[k],color=COL[k],lw=2,label=NAME[k])
        polish(ax);ax.set_title(title);ax.set_ylabel('Flow (mL/s)');ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Exchange across the approved ellipsoidal ROI — not an anatomical neck',fontsize=13)
    fig.savefig(figdir/'ellipsoidal-ROI-exchange.png',dpi=170);plt.close(fig)
    hot=read(native/'moving-high-WSS-regions-and-threshold-sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,key,title,ylabel in zip(axes,['high_WSS_area_fraction','high_WSS_centroid_displacement_vs_MRI_mm'],
            ['Sac-wall area above a fixed threshold','High-WSS centroid shift versus MRI'],['Wall area (%)','Displacement (mm)']):
        for k in KINDS:
            r=[x for x in hot if x['case']==k and x['region']=='aneurysm_sac' and float(x['PS_TAWSS_reference_quantile'])==.9 and x[key]]
            ax.plot([float(x['time_s'])*1000 for x in r],[float(x[key])*(100 if key=='high_WSS_area_fraction' else 1) for x in r],STYLE[k],color=COL[k],lw=2,label=NAME[k])
        polish(ax);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Fixed threshold = MRI sac TAWSS 90th percentile; centroid is not impingement',fontsize=12)
    fig.savefig(figdir/'moving-sac-high-WSS-regions.png',dpi=170);plt.close(fig)
    w=read(out/'weighted-residence/flow-weighted-first-exit-curves.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,scale,xmax,label in zip(axes,[1000,1/PERIOD],[100,10],['ms','cycles']):
        for k in KINDS:
            r=[x for x in w if x['case']==k and x['cohort']=='first_cycle_ROI_entrants' and x['subset']=='all']
            ax.step([float(x['age_s'])*scale for x in r],[float(x['survival'])*100 for x in r],where='post',linestyle=STYLE[k],color=COL[k],lw=2,label=NAME[k])
        ax.set(xlim=(0,xmax),ylim=(0,100),xlabel='Time since first ROI entry ('+label+')',ylabel='Not yet first-exited (%)');ax.spines[['top','right']].set_visible(False);ax.grid(True,color='#e2e8f0');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Post-hoc inlet-volume-weighted retention — original tracks, still exploratory',fontsize=13)
    fig.savefig(figdir/'inlet-volume-weighted-retention.png',dpi=170);plt.close(fig)
    # Each field is at that condition's OWN prescribed inflow peak; no false matched-time claim.
    grids={k:pv.read(native/(k+'-own-inlet-peak-ROI-fields.vtu')) for k in KINDS}
    plot=pv.Plotter(shape=(1,3),off_screen=True,window_size=(1800,700));maximum=max(float(np.percentile(g['Vorticity_per_s'],99)) for g in grids.values())
    for i,k in enumerate(KINDS):
        plot.subplot(0,i);plot.set_background('white');g=grids[k]
        cut=g.slice(normal=[0,1,0],origin=CENTER_MM)
        plot.add_mesh(cut,scalars='Vorticity_per_s',cmap='viridis',clim=[0,maximum],lighting=False,
                      scalar_bar_args={'title':'Vorticity (1/s) — '+NAME[k],'vertical':False,'width':.8,'height':.15,'position_x':.1,'position_y':.03})
        plot.add_mesh(g.extract_surface(algorithm='dataset_surface'),color='#94a3b8',opacity=.08)
        plot.add_text(NAME[k]+'\nown inlet-peak snapshot',font_size=13,color='#172554')
        plot.camera_position=[CENTER_MM+np.array([0,-100,0]),CENTER_MM,[0,0,1]];plot.enable_parallel_projection();plot.camera.parallel_scale=7
    plot.screenshot(figdir/'vorticity-ROI-sections-own-peaks.png');plot.close()


def independent_checks(out):
    native=out/'native-analyses';grid=pv.read(PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu')
    center=grid.points[grid.cells.reshape(-1,5)[:,1:]].mean(1)/1000
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');vol=mapping['volume_m3'];roi=mapping['ellipsoid_mask'];records=[]
    cfg=load(RUN/'patient_specific/input-configuration.json')
    with h5py.File(cfg['source_case'],'r') as f:
        mesh=f['meshes/1'];faces=mesh['faces/nodes/1/nodes'][:].reshape(-1,3).astype(int)-1
        c0=mesh['faces/c0/1'][:].astype(int)-1;c1=mesh['faces/c1/1'][:].astype(int)-1;ni=len(c1)
    fc=(grid.points[faces].mean(1))/1000
    owners=np.r_[c0,c1];faceid=np.r_[np.arange(len(c0)),np.arange(ni)];order=np.argsort(owners,kind='stable');cf=faceid[order].reshape(-1,4)
    extcenter=np.vstack([center,fc[ni:]])
    neighbor=np.r_[np.where(np.arange(len(c0))<ni,np.r_[c1,np.zeros(len(c0)-ni,dtype=int)],len(center)+np.arange(len(c0))-ni),c0[:ni]][order].reshape(-1,4)
    rows=read(native/'vortex-energy-dissipation-all-times.csv')
    for k in KINDS:
        cfg=load(RUN/k/'input-configuration.json');steps=load(RUN/k/'progress.json')['accepted_steps']
        own=np.asarray(cfg['time_s'])[np.argmax(sum(np.asarray(x['flow_m3_s']) for x in cfg['inlets'].values()))]
        j=int(np.argmin(abs(np.array([s['time_s'] for s in steps])-own)));step=steps[j]
        with h5py.File(step['output_stem']+'.dat.h5','r') as f:
            p=f['results/1/phase-1'];v=np.array([p['cells/'+n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']]).T
            bc=np.array([p['faces/'+n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']]).T
            flux=p['faces/SV_FLUX/1'][:]
        expected=.5*cfg['density_kg_m3']*np.sum(vol[roi]*np.sum(v[roi]**2,axis=1))
        row=next(r for r in rows if r['case']==k and r['region']=='sac_ellipsoid' and abs(float(r['time_s'])-step['time_s'])<1e-10)
        assert abs(expected-float(row['kinetic_energy_J']))<1e-12
        snap=pv.read(native/(k+'-own-inlet-peak-ROI-fields.vtu'));ids=snap['FluentCellIndex0'].astype(int);extv=np.vstack([v,bc]);errors=[]
        for local in np.linspace(0,len(ids)-1,40,dtype=int):
            c=int(ids[local]);dx=extcenter[neighbor[c]]-center[c];du=extv[neighbor[c]]-v[c];scale=np.linalg.norm(dx,axis=1)
            grad=np.linalg.lstsq(dx/scale[:,None],du/scale[:,None],rcond=None)[0].T
            sym=(grad+grad.T)/2;expected_diss=2*cfg['viscosity_pa_s']*np.sum(sym*sym)
            error=abs(expected_diss-float(snap['Dissipation_W_m3'][local]))/max(1,abs(expected_diss));errors.append(error)
        assert max(errors)<1e-8
        # Independent whole-cell divergence sum equals oriented interface net flux.
        divergence=np.bincount(c0,weights=flux,minlength=len(vol))-np.bincount(c1,weights=flux[:ni],minlength=len(vol))
        net=float(divergence[roi].sum()/cfg['density_kg_m3']*1e6)
        ex=next(r for r in read(native/'ellipsoidal-ROI-exchange-NOT-anatomical-neck.csv') if r['case']==k and abs(float(r['time_s'])-step['time_s'])<1e-10)
        assert abs(net-float(ex['net_outflow_ml_s']))<1e-10
        records.append({'case':k,'peak_time_s':step['time_s'],'native_ROI_energy_independent_check':'passed',
                        '40_cell_gradient_LS_checks_max_relative_error':max(errors),'ROI_flux_divergence_reconciliation':'passed'})
    prior=load(PREVIOUS/'comparisons/native-data-manifest.json')
    assert all(sha(p['file'])==p['sha256'] for p in prior)
    save(out/'independent-checks.json',{'status':'passed_scoped_checks','records':records,'native_data_hashes_unchanged':len(prior),
        'limitations':'No mesh/time convergence, CFD periodicity or physiological transport validation implied.'})


def document(out):
    dye=[{'folder':p.name,**load(p/'pilot-result.json')} for p in out.glob('virtual-dye-pilot*') if (p/'pilot-result.json').exists()]
    save(out/'MRI-comparison-readiness.json',{'status':'blocked_missing_registered_TB_velocity_volume',
        'available_TB_measurements':['ICA flow/velocity waveform CSV, 25 phases','Right near-aneurysm contour: 4 voxel centers per phase','Left near-aneurysm contour: 5 voxel centers per phase'],
        'not_used':'CMRx4DFlow2026 P001 is an unrelated aortic challenge case, not TB.',
        'required':['TB 4D velocity vector volume and units','Voxel-to-world affine/orientation and spacing','Frame times and ECG phase convention','Approved rigid MRI-to-CFD registration and vector rotation','Magnitude/quality/segmentation masks for partial-volume handling'],
        'additional_timing_issue':'Latest applied ICA waveforms are independently shifted: B1 +503.997ms, B2 +473.997ms. A single MRI clock offset cannot undo their 30-ms relative timing change. Do not optimize phase registration to maximize agreement or call inlet agreement independent validation.',
        'anatomical_neck':'No approved anatomical neck surface found; only the approved exploratory ellipsoid is used.'})
    save(out/'completion-status.json',{'status':'completed_available_offline_analyses_with_blocked_experiments',
        'completed':['Normalized WSS with two ICA reference-band choices','Low/high-shear area fractions with explicit operational thresholds',
            'Transverse WSS and updated latest-case RRT','Vorticity, Q-criterion, helicity, kinetic energy and viscous dissipation',
            'Moving high-WSS centroids, spatial overlap and threshold sensitivity','Native-flux exchange across approved ellipsoidal ROI',
            'Post-hoc inlet-volume-weighted residence, pulse-cohort and seed-subset sensitivity'],
        'incomplete':['Validated anatomical-neck exchange','Full ten-cycle virtual-dye washout and transport convergence',
            'Fresh flux-sampled particle trajectories and particle-count convergence','Registered volume CFD–MRI comparison',
            'CFD periodicity and mesh/time-step independence'],
        'virtual_dye_pilots':dye,'new_Fluent_sessions':0,'new_CFD_cycles':0,
        'source_files_modified':False,'uploaded_data':False})
    with (out/'README.md').open('x',encoding='utf-8') as f:
        f.write('''# Extended TB hemodynamics — 29 October 2026

All results use the latest three revised-outlet, systole-first CFD cases, not the October 2 study.
No Fluent session, source input/result edit, geometry change, material/BC change or upload occurred.

## Open first
- figures/regional-energy-and-dissipation.png
- figures/moving-sac-high-WSS-regions.png
- figures/inlet-volume-weighted-retention.png
- figures/vorticity-ROI-sections-own-peaks.png
- native-analyses/parent-wall-reference-overlay.png
- completion-status.json

## What is computed
The native-analyses CSVs contain all 95 times in each condition and both sac/complement regions.
Wall measurements use existing native aneurysm_sac and wall zones. Fluid measurements use the
previously user-approved ellipsoid (−2,29,−23) mm, radii (4.5,4.5,6.5) mm, by cell centroid.
Those wall and fluid selections are not an identical anatomical neck segmentation.

Normalized WSS uses case-specific area-weighted ICA wall TAWSS: pooled right/left 3–8-mm
geodesic bands from inlet rims, with 5–10-mm sensitivity. These are provisional reference bands,
not a clinical standard. Raw stresses are preserved. Low <10% reference and high >2x reference
are operational analysis choices, not rupture-risk thresholds.

Transverse WSS averages |tau dot (normal cross mean-shear-direction)|. Undefined direction is
masked below 1e-6 Pa mean-vector magnitude and its area is reported. RRT=1/|mean shear vector|,
in 1/Pa, NOT seconds. RRT is not particle residence time or a blood-volume washout measurement.

Velocity gradients use inverse-distance-squared weighted least squares on actual tetrahedral
face neighbors and native boundary-face velocities. The affine-field test checks implementation;
it does not establish derivative accuracy or grid independence. Vorticity is curl(u); Q is the
velocity-gradient vortex criterion (1/s²), not volumetric flow Q. Positive Q volume is a diagnostic,
not proof of turbulence. Helicity is u dot curl(u); normalized helicity masks near-zero factors.
Kinetic energy=0.5 rho integral |u|² dV, J. Viscous dissipation=2 mu integral S:S dV, W.
Rigid incompressible laminar blood remains rho=1060 kg/m³, mu=0.0035 Pa s.

High-WSS thresholds are fixed MRI TAWSS 80th/90th/95th percentiles per wall zone, common to
all cases and times. Centroid displacement and overlap are not validated jet-impingement measures.

ROI exchange integrates original native Fluent face mass flux divided by density, separately
for inward/outward flow over internal faces bounding the ellipsoidal cell set. This is NOT
anatomical neck flow. Volume/inflow is a turnover proxy, not a measured particle residence time.
The comparison VTUs show EACH condition at its OWN inlet peak (nearest native state), not
matched-time snapshots. The CSV time histories retain matched physical timestamps.

## Residence upgrade
Existing uniform-area 20-per-ICA seed releases are assigned post-hoc inlet-volume weights by
exact piecewise-linear flow integration over release-bin intervals. Weights reconcile against
total injected volume. They do not create new particles or establish representative seed sampling.
First-injection-cycle, first-cycle-ROI-entry and all-entry cohorts are separate. Even/odd seed
subsets are sensitivity diagnostics, not independent experiments or proof of count convergence.
The previous saved-polyline ellipsoidal crossings and right censoring remain unchanged.

## What remains blocked or preliminary
Each CFD case is still one first cycle. All period averages assume the existing 5-ms closing
bridge; periodicity, mesh/time independence and individualized production path convergence
remain unvalidated. No clinical risk prediction is justified by these metrics.

The virtual-dye diagnostic uses a conservative one-way pure-advection equation with zero
incoming tracer and initial tracer only inside the ellipsoid. It is a separate new numerical
transport calculation, not a new Navier–Stokes run. Pilot status, solver failures and any bounds
failures are recorded; do not interpret an incomplete/failed pilot as a washout result. Molecular
diffusion was not invented. Upwind numerical diffusion and transport-step sensitivity still need study.

An anatomical neck needs approval. Full TB MRI comparison needs registered velocity volumes,
voxel geometry and frame timing; the located aortic challenge MRI is a different case and is not used.
Latest right/left ICA waveforms were shifted independently by approximately 504/474 ms, changing
relative timing by 30 ms: MRI validation requires a defensible common clock, not an optimized fit.
Additional Fluent continuation has a separately approved less-than-two-hour total budget.
This offline package does not include or certify those new cycles; see the separate continuation run.
See MRI-comparison-readiness.json for unavailable TB MRI volume requirements.

Scripts: D:/FluentAgent/automation/20261029-extended-hemodynamics-v1.
Independent native-energy, local gradient and ROI flux checks are in independent-checks.json.
Sources are read-only; all newly generated files are on D: and originals are preserved.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    make_figures(a.out);independent_checks(a.out);document(a.out)
