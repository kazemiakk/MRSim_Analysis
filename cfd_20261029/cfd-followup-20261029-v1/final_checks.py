"""Read-only independent cohort audit plus final package receipt; no solver calls."""
import argparse,platform,re,subprocess,sys
from pathlib import Path
import h5py,numpy as np,pyvista,matplotlib,scipy
from common_followup import *
FF=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')


def main(root):
    reports=[]
    for kind,folder in selected().items():
        meta=load(folder/'tracking-audit.json');n=meta['released'];assigned=np.zeros(n,int);births=np.full(n,np.nan);total=0
        with h5py.File(folder/'tracks.h5','r') as h:
            for key in h['frames']:
                f=h['frames/'+key];ids=f['id'][:].astype(int);b=f['birth_time_s'][:]
                # Independent integer-bin calculation with the stored source timebase.
                cyc=np.searchsorted(START+PERIOD*np.arange(1,10),b+1e-10,side='right')+1
                known=assigned[ids]!=0
                assert np.array_equal(assigned[ids][known],cyc[known])
                assert np.allclose(births[ids][known],b[known],rtol=0,atol=0)
                assigned[ids]=cyc;births[ids]=b;total+=len(ids)
                assert np.array_equal(cyc,release_cycle(b))
            assert np.all(assigned>0) and np.isfinite(births).all() and set(assigned)==set(range(1,11))
        rows=list(__import__('csv').DictReader((root/'residence'/(kind+'-particle-residence.csv')).open()))
        assert len(rows)==n
        assert all(int(r['release_cycle'])==assigned[int(r['particle_id'])] for r in rows)
        reports.append({'case':kind,'unique_tracers_checked':n,'frame_samples_checked':total,
                        'persistent_release_cycles':'passed','all_ten_release_categories_present':True})
    verification=load(root/'videos-final/video-verification.json');video_checks=[]
    for v in verification['products']:
        path=Path(v['path']);assert sha(path)==v['sha256']
        p=subprocess.run([str(FF),'-hide_banner','-i',str(path)],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        m=re.search(r'Duration: (\d+):(\d+):([\d.]+)',p.stderr);assert m
        seconds=int(m[1])*3600+int(m[2])*60+float(m[3]);assert abs(seconds-86.8)<.034
        video_checks.append({'file':str(path),'duration_s':seconds,'source_manifest_hash_match':True,'full_decode_passed':v['full_decode_passed']})
    movie=Path(verification['products'][0]['path'])
    for t,label in [(7,'early'),(80,'late')]:
        subprocess.run([str(FF),'-v','error','-n','-ss',str(t),'-i',str(movie),'-frames:v','1',str(movie.parent/('decoded-'+label+'.png'))],
                       capture_output=True,check=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    # Check common physical settings and independently test actual B4 pressure reference.
    cfg={k:load(RUN/k/'input-configuration.json') for k in KINDS}
    for k in KINDS:
        assert cfg[k]['density_kg_m3']==1060 and cfg[k]['viscosity_pa_s']==.0035 and cfg[k]['period_s']==PERIOD
        assert {n:a['fraction'] for n,a in cfg[k]['outlets'].items()}=={n:a['fraction'] for n,a in cfg['patient_specific']['outlets'].items()}
    with h5py.File(cfg['patient_specific']['source_case'],'r') as f:
        m=f['meshes/1'];z=m['faces/zoneTopology'];i=z['name'][0].decode().split(';').index('outlet_b4')
        lo=int(z['minId'][i])-1;hi=int(z['maxId'][i]);tri=m['faces/nodes/1/nodes'][lo*3:hi*3].reshape(-1,3).astype(int)-1
        p=next(iter(m['nodes/coords'].values()))[:];area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)/2
    pressure=[]
    for k in KINDS:
        steps=load(RUN/k/'progress.json')['accepted_steps']
        for j in [0,25,94]:
            with h5py.File(steps[j]['output_stem']+'.dat.h5','r') as f:
                d=f['results/1/phase-1/faces/SV_P/1'];offset=int(d.attrs['minId'][0])-1
                mean=float(np.average(d[lo-offset:hi-offset],weights=area))
                assert abs(mean)<1e-6;pressure.append({'case':k,'time_s':steps[j]['time_s'],'B4_gauge_Pa':mean})
    for k,p in selected().items():
        expected=verification['source_tracks'][k]['sha256'];assert sha(p/'tracks.h5')==expected
    save(root/'final-checks.json',{'status':'passed_with_scientific_limitations','cohort_identity_audit':reports,
        'video_checks':video_checks,'B4_pressure_reference_spot_checks':pressure,
        'input_tracks_unchanged':True,'source_native_data_checks':'verification.json: 285 data hashes and 12 independent velocity RMSE checks',
        'source_configs_unchanged_by_this_work':True,'new_Fluent_sessions':0,'new_CFD_cycles':0,
        'dependency_versions':{'python':platform.python_version(),'numpy':np.__version__,'h5py':h5py.__version__,
            'pyvista':pyvista.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__},
        'scripts':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'limitations':['CFD periodicity unvalidated.','Individualized production tracking still fails its original refinement gate.',
            'Additional 1.5625-us QA was incomplete at its bounded runtime; no acceptance criterion was relaxed.',
            'Ellipsoidal ROI, visual-tracer weights and saved-position interpolation limit residence interpretation.']})
    save(root/'package-status.json',{'status':'completed_exploratory_offline_followup',
        'completed':['Three-case and individual cycle-colored videos','Approved ellipsoidal ROI overlay and cell mapping',
            'Exploratory first-entry/exit, exposure, censoring, retention and reentry measurements',
            'Latest-only waveform NRMSE and regional velocity/pressure RMSE',
            'Peak and assumed-closure cycle summaries; existing native sac and surrounding-wall WSS/OSI comparisons'],
        'not_completed':['Validated physiological residence/washout','CFD cycle periodicity validation',
            'Converged individualized production trajectories','Anatomical neck-bound sac segmentation'],
        'primary_video':str(movie),'readme':str(root/'README-reviewed.md'),
        'scope':'No model, boundary condition, source geometry, original result, or license configuration changed; nothing uploaded.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
