"""Consolidate generated CSVs, independently spot-check native values, and plot files."""
import argparse, csv
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_followup import *

SERIES_COLORS={'patient_specific':'#2563eb','individualized':'#d97706','generic':'#0d9488'}
STYLES={'patient_specific':'-','individualized':'--','generic':':'}
NAMES=dict(zip(KINDS,['Patient-specific','Individualized*','Population average']))


def read(path):
    return list(csv.DictReader(Path(path).open(newline='')))


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(True,color='#e2e8f0',linewidth=.7);ax.set_axisbelow(True)


def consolidate(root):
    folder=root/'comparisons'
    rows=read(folder/'wall-cycle-TAWSS-OSI-summary.csv');result=[]
    arrays=np.load(folder/'wall-cycle-fields.npz')
    # Source triangle areas independently reconstructed for the spatial summaries.
    cfg=load(RUN/'patient_specific/input-configuration.json')
    with h5py.File(cfg['source_case'],'r') as f:
        m=f['meshes/1'];z=m['faces/zoneTopology'];names=z['name'][0].decode().split(';')
        xyz=next(iter(m['nodes/coords'].values()))[:];areas={}
        for region in ['wall','aneurysm_sac']:
            idx=names.index(region);lo=int(z['minId'][idx])-1;hi=int(z['maxId'][idx]);tri=m['faces/nodes/1/nodes'][lo*3:hi*3].reshape(-1,3).astype(int)-1
            areas[region]=np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]]),axis=1)/2
    for region in areas:
        a=next(r for r in rows if r['case']=='patient_specific' and r['wall_region']==region)
        for kind in KINDS[1:]:
            b=next(r for r in rows if r['case']==kind and r['wall_region']==region)
            x=arrays['patient_specific_'+region+'_tawss'];y=arrays[kind+'_'+region+'_tawss'];w=areas[region]/areas[region].sum()
            dx=x-np.dot(w,x);dy=y-np.dot(w,y);denom=np.sqrt(np.dot(w,dx*dx)*np.dot(w,dy*dy))
            ox=arrays['patient_specific_'+region+'_osi'];oy=arrays[kind+'_'+region+'_osi']
            result.append({'candidate':kind,'region':region,'ps_mean_TAWSS_pa':float(a['area_mean_TAWSS_pa']),
                'candidate_mean_TAWSS_pa':float(b['area_mean_TAWSS_pa']),
                'TAWSS_absolute_difference_percent':100*abs(float(b['area_mean_TAWSS_pa'])/float(a['area_mean_TAWSS_pa'])-1),
                'max_instantaneous_WSS_difference_pa':float(b['max_instantaneous_WSS_pa'])-float(a['max_instantaneous_WSS_pa']),
                'TAWSS_spatial_RMSE_pa':float(np.sqrt(np.dot(w,(x-y)**2))),
                'TAWSS_spatial_correlation':float(np.dot(w,dx*dy)/denom) if denom>0 else None,
                'ps_mean_OSI':float(a['area_mean_OSI']),'candidate_mean_OSI':float(b['area_mean_OSI']),
                'OSI_mean_signed_difference':float(b['area_mean_OSI'])-float(a['area_mean_OSI']),
                'OSI_spatial_RMSE':float(np.sqrt(np.dot(w,(ox-oy)**2)))})
    csvsave(folder/'wall-cycle-condition-comparisons.csv',result)


def plots(root):
    plots=root/'figures-final';plots.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11})
    wave=read(root/'comparisons/latest-waveforms-mL-s.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for ax,inlet,title in zip(axs,['inlet_b1','inlet_b2','combined'],['Right ICA / B1','Left ICA / B2','Combined inflow']):
        rows=[r for r in wave if r['inlet']==inlet];t=np.array([float(r['time_s']) for r in rows])*1000
        for k in KINDS:ax.plot(t,[float(r[k]) for r in rows],STYLES[k],color=SERIES_COLORS[k],label=NAMES[k],lw=2)
        ax.set(title=title,xlabel='Time (ms)',ylabel='Flow (mL/s)',xlim=(0,868),ylim=(0,None));ax.set_xticks([0,200,400,600,868]);style(ax)
    axs[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Latest prescribed ICA waveforms — stored simulation timing',fontsize=14)
    fig.savefig(plots/'latest-waveform-comparison.png',dpi=180);plt.close(fig)
    data=read(root/'comparisons/regional-velocity-pressure-all-times.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for ax,region,title in zip(axs,['sac_ellipsoid','surrounding_fluid'],['Approved ellipsoidal sac ROI','Complementary surrounding fluid']):
        for k in KINDS[1:]:
            rows=[r for r in data if r['candidate']==k and r['region']==region]
            ax.plot([float(r['time_s'])*1000 for r in rows],[float(r['vector_velocity_rmse_m_s']) for r in rows],
                    STYLES[k],color=SERIES_COLORS[k],label=NAMES[k]+' vs MRI',lw=2)
        ax.set(title=title,xlabel='Time (ms)',ylabel='Vector velocity RMSE (m/s)',xlim=(5,868),ylim=(0,.5));ax.set_xticks([5,200,400,600,868]);style(ax);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('First-cycle regional differences; periodicity unvalidated',fontsize=14)
    fig.savefig(plots/'latest-regional-velocity-RMSE.png',dpi=180);plt.close(fig)
    data=read(root/'residence/first-exit-retention-curves.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,scale,title in zip(axs,[1000,1/PERIOD],['First 100 ms after ROI entry','Longer follow-up (first-cycle entrants)']):
        for k in KINDS:
            rows=[r for r in data if r['case']==k and r['cohort']=='first_replay_cycle_entrants']
            ax.step([float(r['age_s'])*scale for r in rows],[100*float(r['survival']) for r in rows],where='post',
                    linestyle=STYLES[k],color=SERIES_COLORS[k],label=NAMES[k],lw=2)
        ax.set(title=title,xlabel='Time since first ROI entry ('+('ms' if scale==1000 else 'cycles')+')',ylabel='Not yet first-exited (%)',xlim=(0,100 if scale==1000 else 10),ylim=(0,100));style(ax);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Exploratory tracer retention — not blood-volume washout',fontsize=14)
    fig.savefig(plots/'exploratory-sac-first-exit-retention.png',dpi=180);plt.close(fig)


def verify(root):
    folder=root/'comparisons';reports=[]
    mapping=np.load(folder/'fluid-ROI-mapping.npz');vol=mapping['volume_m3'];mask=mapping['ellipsoid_mask']
    assert len(vol)==1169992 and np.all(vol>0)
    progress={k:load(RUN/k/'progress.json')['accepted_steps'] for k in KINDS}
    data=read(folder/'regional-velocity-pressure-all-times.csv')
    # Independent dot-product implementation, native cell arrays, 3 times x 2 candidates x 2 regions.
    for index in [0,25,94]:
        fields={}
        for k in KINDS:
            with h5py.File(progress[k][index]['output_stem']+'.dat.h5','r') as f:
                cells=f['results/1/phase-1/cells'];fields[k]=np.array([cells[n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']]).T
        for k in KINDS[1:]:
            for region,m in [('sac_ellipsoid',mask),('surrounding_fluid',~mask)]:
                w=vol[m]/vol[m].sum();delta=fields[k][m]-fields['patient_specific'][m]
                expected=np.sqrt(sum(np.dot(w,delta[:,d]**2) for d in range(3)))
                t=progress[k][index]['time_s'];r=next(r for r in data if r['candidate']==k and r['region']==region and abs(float(r['time_s'])-t)<1e-9)
                error=abs(expected-float(r['vector_velocity_rmse_m_s']));assert error<1e-12
                reports.append({'candidate':k,'region':region,'time_s':t,'RMSE_absolute_error_m_s':float(error)})
    # Native wall period averages agree with the already independently audited newer output.
    old=load(PROJECT/'results/20261023-systole-first-wss-comparison-v1/wss-summary.json')
    for r in read(folder/'wall-cycle-TAWSS-OSI-summary.csv'):
        a=old[r['case']][r['wall_region']]
        assert abs(float(r['area_mean_TAWSS_pa'])-a['tawss_mean_pa'])<1e-10
        assert abs(float(r['area_mean_OSI'])-a['osi_mean'])<1e-12
    for kind in KINDS:
        rows=read(root/'residence'/(kind+'-particle-residence.csv'))
        for r in rows:
            assert float(r['cumulative_observed_ROI_time_s'])>=0
            assert float(r['cumulative_observed_ROI_time_s'])<=float(r['last_observed_s'])-float(r['birth_time_s'])+1e-9
            if r['first_exit_observed']=='True':assert float(r['first_exit_s'])>=float(r['first_entry_s'])-1e-10
    # Every source file in the comparison manifest remains byte-identical after analysis.
    manifest=load(folder/'native-data-manifest.json')
    assert all(sha(r['file'])==r['sha256'] for r in manifest)
    save(root/'verification.json',{'status':'passed_scoped_offline_checks','independent_velocity_RMSE_spot_checks':reports,
        'native_wall_summary_agreement':'all three cases, both wall regions; mathematical consistency only',
        'native_source_data_unchanged':True,'source_data_files_checked':len(manifest),
        'residence_duration_bounds':'all reported tracers checked',
        'limits':'Not independent validation of CFD periodicity, mesh/time convergence, anatomical neck boundary, or every particle trajectory.'})


def readme(root, filename='README.md'):
    ref=load(root/'refinement/refinement-result.json') if (root/'refinement/refinement-result.json').exists() else {'status':'pending'}
    text='''# Latest three-case CFD follow-up — 29 October 2026

## Open first

- videos-final/Three-cases-release-cycle-colored.mp4
- comparisons/ellipsoidal-sac-ROI-overlay.png
- figures-final/latest-regional-velocity-RMSE.png
- figures-final/exploratory-sac-first-exit-retention.png

## Scope and sources

Only the revised-outlet, systole-first simulations from
`D:\\FluentAgent\\runs\\20261023-070441-568099-systole-first-checkpoint-recovery-v1`
are used. Each has 95 accepted states, 5–868 ms, from one solved first cycle.
No Fluent session, solver iteration, boundary change, geometry edit, or upload was performed.
Source tracks are those selected by `20261022-systole-first-particles-10cycles-v1`.
Previous scripts, simulations, videos, and reports remain preserved.

## Cycle-colored videos

Ten permanent categories C1–C10 are computed from each particle's saved birth time.
Every particle retains its color. IDs, positions, release schedule and speeds are unchanged.
Real nonuniform saved-time intervals are retained at 10x slower playback, not equal frame durations.
There are 17,240 source tracers per condition, not 250 per cycle. No new releases were invented.
Earlier encoder/layout attempts are preserved under `videos/` and `videos-cycle-colors-v2/`.
Use `videos-final/`, which includes a release-cycle legend in every individual video.
It is the losslessly trimmed version of `videos-cycle-colors-v3/`; only encoder end padding was removed.

## Regional comparisons

The user approved applying the existing sac-selection ellipsoid to fluid cells:
center (−2,29,−23) mm; radii (4.5,4.5,6.5) mm.
Cell-centroid membership defines `sac_ellipsoid`; its complement is `surrounding_fluid`.
The overlay and original Fluent cell-ID mapping are saved. This is an exploratory ROI,
not an approved anatomical neck segmentation. No physical vessel was capped or changed.
Untagged numerical extensions cannot be separately excluded from the complement.
Wall results retain the existing `aneurysm_sac` and complementary `wall` zones.
The native wall patch and ellipsoidal fluid ROI are not identical boundary definitions.

Volume-weighted VECTOR velocity RMSE is sqrt(mean(|u_case−u_MRI|²)); normalized RMSE
uses the MRI RMS vector speed, not a velocity range. Speed-only RMSE is separate.
Pressure values/differences are gauge Pa at the common B4 reference, not systemic pressure.
Waveform Q is mL/s; waveform NRMSE uses the MRI waveform range, sampled uniformly 4000 times.
The exact stored timing, separate ICA shifts, and endpoint ramp are retained; no new alignment is imposed.
The published waveform was previously mapped to an 868-ms period; this work does not undo that mapping.

Peak comparisons use the nearest saved time to the PATIENT-SPECIFIC combined-inflow maximum.
Candidate-own-peak snapshots are separately labeled; they do not replace matched-time comparisons.
Full-period summaries explicitly assume the same 5-ms linear endpoint closure used in the existing
TAWSS/OSI calculation. They are first-cycle, assumed-closure values—not periodic steady-state results.
WSS is integrated tangential force divided by native face area; averages are area weighted.
OSI integrates WSS VECTORS before computing local OSI and area-averaging.
Max instantaneous WSS, max local TAWSS, and peak area-mean WSS are different metrics.

## Exploratory residence and retention

Pointwise membership uses the same approved ellipsoid. Crossings are computed analytically
along straight segments between consecutive SAVED positions. Events/curvature within an output
interval can be missed; a tiny tracking substep does not restore unsaved trajectory detail.
`*-particle-residence.csv` records first entry/exit, cumulative observed time, reentry episodes,
release cycle, and censoring. Unobserved first exits are right-censored at last observation.
First-pass retention curves use descriptive unweighted Kaplan–Meier estimates since first entry.
The main cohort entered the ROI during replay cycle 1. All-entrant curves are supplemental because
later entrants have shorter follow-up. Fifty/90-percent clearance is left blank if not reached.
Occupancy (including reentry) and first-exit survival are not interchangeable; both tables are provided.
These uniform-area VISUAL tracers are not flux/mass-weighted blood-volume fractions.
Mean observed cumulative exposure is not an uncensored mean biological residence time.
Reentry across an ellipsoid is not proof of a closed recirculating vortex.
Time below 0.01 and 0.05 m/s is a configurable operational sensitivity measure, not a clinical
definition of stagnation. RRT derived from WSS is not substituted for particle time in seconds.

## Remaining numerical limitations

Ten repeats of one first-cycle velocity template are not ten independently solved cardiac cycles.
The 0–5-ms wrap is interpolated; startup transients remain. Periodicity is not established.
Mesh and full-cycle timestep independence are not established.
Individualized production tracking at 6.25 microseconds fails its original 0.100-mm criterion
(about 0.498 mm at the tested finer pair). It remains marked exploratory in all new outputs.
The bounded 3.125-versus-1.5625-microsecond test concerns only 40 initial inlet seeds, not the
full production cohort, and cannot retroactively validate the unchanged videos/residence tracks.
'''
    text+='\nBounded additional refinement status: **'+ref['status']+'**.\n'
    text+='\nFurther Fluent cycles require a separately approved bounded runtime. No numerical acceptance criterion was relaxed.\n'
    with (root/filename).open('x',encoding='utf-8') as f:f.write(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    consolidate(a.out);plots(a.out);verify(a.out);readme(a.out)
