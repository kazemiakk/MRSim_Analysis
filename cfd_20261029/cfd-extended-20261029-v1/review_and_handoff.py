"""Non-destructive review fixes, continuation diagnostic and concise evidence index."""
from pathlib import Path
import csv,json
import numpy as np,h5py,pyvista as pv
from extended_common import *
from finish_extended import NAME,read

OUT=PROJECT/'results/20261029-extended-hemodynamics-v1'
CONT=PROJECT/'runs/20261029-174639-139752-patient-specific-cycle2-bounded-v1'


def reviewed_sections():
    grids={k:pv.read(OUT/'native-analyses'/(k+'-own-inlet-peak-ROI-fields.vtu')) for k in KINDS}
    maximum=max(float(np.percentile(g['Vorticity_per_s'],99)) for g in grids.values())
    g=grids[KINDS[0]];mid=(g.bounds[2]+g.bounds[3])/2
    origin=np.asarray(g.center);origin[1]=mid
    cut0=g.slice(normal=[0,1,0],origin=origin);focus=np.asarray(cut0.center)
    p=pv.Plotter(shape=(1,3),off_screen=True,window_size=(1800,700))
    for i,k in enumerate(KINDS):
        p.subplot(0,i);p.set_background('white');g=grids[k]
        cut=g.slice(normal=[0,1,0],origin=origin)
        p.add_mesh(cut,scalars='Vorticity_per_s',clim=[0,maximum],cmap='viridis',lighting=False,
                   scalar_bar_args={'title':'Vorticity (1/s)','vertical':False,'width':.65,'height':.12,
                                    'position_x':.17,'position_y':.04,'n_labels':3,'fmt':'%.0f',
                                    'title_font_size':15,'label_font_size':12})
        p.add_text(NAME[k]+'\nOwn inlet-peak snapshot\nROI mid-Y section: %.2f mm'%mid,
                   font_size=12,color='#172554',position='upper_left')
        p.camera_position=[focus+[0,-100,0],focus,[0,0,1]]
        p.enable_parallel_projection();p.camera.parallel_scale=6
    path=OUT/'figures/vorticity-ROI-sections-own-peaks-reviewed.png'
    assert not path.exists();p.screenshot(path);p.close()
    save(OUT/'figures/reviewed-section-definition.json',{'plane_Y_mm':mid,'normal':[0,1,0],
        'plane_selection':'Midpoint of approved ellipsoidal fluid ROI bounding box; not anatomical neck',
        'each_case_own_peak':True,'common_colour_scale_per_s':[0,maximum],
        'clipped_values':'Above largest case 99th percentile shown at upper colour; no data values modified',
        'source_original_figure_preserved':True})


def continuation_diagnostics():
    progress=load(CONT/'patient_specific/progress.json');old=load(RUN/'patient_specific/progress.json')
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');vol=mapping['volume_m3'];mask=mapping['ellipsoid_mask']
    results=[]
    for prior,current in zip(old['accepted_steps'],progress['accepted_steps']):
        arrays=[]
        for step in [prior,current]:
            with h5py.File(step['output_stem']+'.dat.h5','r') as f:
                c=f['results/1/phase-1/cells']
                v=np.column_stack([c[n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']])
            assert np.isfinite(v).all();arrays.append(v)
        for region,m in [('sac_ellipsoid',mask),('surrounding_fluid',~mask)]:
            a,b=[v[m] for v in arrays];w=vol[m];d=np.sum((b-a)**2,axis=1)
            rmse=np.sqrt(np.dot(w,d)/w.sum());reference=np.sqrt(np.dot(w,np.sum(a*a,axis=1))/w.sum())
            results.append({'phase_s':prior['time_s'],'cycle2_time_s':current['time_s'],'region':region,
                            'vector_velocity_RMSE_m_s':float(rmse),'cycle1_RMS_speed_m_s':float(reference),
                            'relative_vector_L2_difference_percent':float(100*rmse/reference),
                            'scope':'First three phase pairs only; not full-cycle periodicity validation'})
    with (OUT/'patient-specific-cycle2-pilot-regional-velocity.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    report=load(CONT/'study-result.json')
    assert report['owned_session_closed'] and not report['license_issues'] and report['original_files_unchanged']
    save(OUT/'Fluent-continuation-link.json',{'run_directory':str(CONT),'result':report,
        'not_incorporated_into_three_case_metrics':True,
        'explanation':'Three accepted steps in 196 seconds. Conservative complete-cycle estimate exceeded usable 115-minute runtime budget. No full additional cycle completed.'})


def residence_sensitivity():
    rows=[]
    for k in KINDS:
        particles=read(PREVIOUS/'residence'/(k+'-particle-residence.csv'))
        weights={int(r['particle_id']):float(r['weight_ml']) for r in read(OUT/'weighted-residence'/(k+'-particle-inlet-volume-weights.csv'))}
        for cohort in ['all_entrants','first_cycle_ROI_entrants']:
            group=[r for r in particles if r['entered_ROI']=='True' and
                   (cohort=='all_entrants' or float(r['first_entry_s'])<PERIOD+.005)]
            w=np.asarray([weights[int(r['particle_id'])] for r in group]);sw=w.sum()
            def mean(field):
                values=[float(r[field]=='True') if field=='observed_reentry' else float(r[field]) for r in group]
                return float(np.dot(w,values)/sw)
            rows.append({'case':k,'cohort':cohort,'entering_tracers':len(group),
                         'posthoc_weighted_reentry_fraction':mean('observed_reentry'),
                         'weighted_mean_observed_ROI_exposure_s':mean('cumulative_observed_ROI_time_s'),
                         'weighted_mean_observed_ROI_exposure_below_0p01_m_s':mean('ROI_time_below_0p01_m_s'),
                         'weighted_mean_observed_ROI_exposure_below_0p05_m_s':mean('ROI_time_below_0p05_m_s'),
                         'scope':'Saved-track exposure, right-censored; speed cutoffs operational, not biological stasis thresholds'})
    with (OUT/'weighted-residence/recirculation-and-low-speed-exposure.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def handoff():
    with (OUT/'START-HERE.md').open('x',encoding='utf-8') as f:
        f.write('''# TB advanced CFD analyses: evidence and remaining work

## Completed from the latest three original solved cycles

- WSS/OSI and regional RMSE: retained latest-case tables in `../20261029-cycle-cohorts-and-regional-metrics-v1/comparisons/`.
- Normalized WSS, reference-band sensitivity, operational low/high-WSS area, transverse WSS and RRT: `native-analyses/` CSVs and wall VTP files.
- Vorticity, Q criterion, helicity, kinetic energy and dissipation: native-time histories, regional cycle summaries and peak ROI VTUs in `native-analyses/`.
- Ellipsoidal ROI exchange from native internal face flux, NOT an approved anatomical neck: exchange CSV and figure.
- Moving high-WSS area, overlap and centroid displacement: fixed patient-specific CFD TAWSS percentile thresholds, with sensitivity. This is NOT direct MRI voxel comparison or a validated impingement-zone measurement.
- Post-hoc inlet-volume-weighted particle first-exit/retention, re-entry and low-speed exposure: `weighted-residence/`. All still exploratory with original tracking limitations.

## Figures

- `figures/regional-energy-and-dissipation.png`
- `figures/moving-sac-high-WSS-regions.png` (MRI means MRI-driven CFD reference, NOT measured MRI wall shear)
- `figures/inlet-volume-weighted-retention.png`
- `figures/ellipsoidal-ROI-exchange.png`
- `figures/vorticity-ROI-sections-own-peaks-reviewed.png` (filled fluid section; each case at its own peak)
- `native-analyses/parent-wall-reference-overlay.png`

The original and reviewed vorticity figures are both retained; the reviewed section is more central and has unclipped labels.

## Live Fluent continuation under the approved budget

Fluent 26.1.0 was launched through local STDIO PyFluent-MCP, four CPU processes. The patient-specific final checkpoint was read without initialization. Exact waveform repetitions were appended, leaving the original cycle, geometry, material, model, outlet fractions and solver acceptance criteria unchanged.

Three new accepted steps reached 0.873, 0.878 and 0.883 s (90/50/50 inner iterations; approximately 60/38/38 s including checks and saves). Total launch-to-close time was 196 seconds. The owned session was closed, no license issue reported, and original case/data/profile hashes remained unchanged.

At phases 5/10/15 ms, cycle-2 sac mean WSS differs from cycle 1 by about 5.1%, 26.5%, 22.1%. These early phase differences show that identical cycles should NOT be assumed. They do not establish a full-cycle convergence error. See `patient-specific-cycle2-pilot-regional-velocity.csv` for fluid velocity differences and `Fluent-continuation-link.json` for the exact run path.

The pilot forecast about 106 additional minutes for a full second cycle; this did not conservatively fit the 115-minute total implementation cap with an 8-minute cleanup reserve. It stopped at the feasibility check, NOT because 115 minutes elapsed. No partial cycle was substituted into the three-condition comparison. Periodicity remains unvalidated.

## Not completed or not yet validated

1. Full registered TB 4D Flow volume comparison: user confirmed this export is unavailable. ICA and small contour CSVs cannot substitute for the missing volume.
2. Anatomical neck-specific exchange/impingement: requires an approved neck segmentation. The approved ellipsoid is explicitly exploratory.
3. Full ten-cycle numerical dye washout: a bounded conservative transport pilot reached 40 ms, not a completed washout. Pilot failure and improved partial result are both preserved; numerical diffusion and transport-step convergence remain untested.
4. Fresh flux-sampled tracking, particle-count convergence and individualized path refinement: not solved by recoloring or post-hoc weights. Large even/odd-seed residence differences must remain visible as a limitation.
5. Full-cycle periodicity, mesh and time-step independence: not established by these offline tests or the three-step continuation.

RRT has units 1/Pa and is NOT residence in seconds. No clinical risk inference is justified. No patient data were uploaded. All new results are on D: and previous results are preserved.

Detailed definitions: `README.md`. Scoped independent checks: `independent-checks.json` (285 original native data hashes unchanged).
''')


if __name__=='__main__':
    reviewed_sections();continuation_diagnostics();residence_sensitivity();handoff()
