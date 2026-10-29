"""One bounded, initial-40-seed refinement attempt; no production replacement."""
import argparse, importlib, subprocess, sys, time
from pathlib import Path
import h5py
import numpy as np
from common_followup import PROJECT, TRACKS, load, save


def worker(out):
    source=PROJECT/'automation/20261022-systole-first-particles-10cycles-v1'
    sys.path.insert(0,str(source))
    track=importlib.import_module('track')
    target=out/'individualized-qa-1p5625us'
    track.folder=lambda kind,step=25.,qa=False: target
    sys.argv=['track.py','--case','individualized','--step-us','1.5625','--qa','--wall-limit-seconds','180']
    track.main()


def main(out):
    out.mkdir(exist_ok=False);start=time.monotonic()
    with (out/'refinement-run.log').open('x',encoding='utf-8') as log:
        try:
            r=subprocess.run([sys.executable,'-u',__file__,'--worker','--out',str(out)],stdout=log,stderr=subprocess.STDOUT,
                timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
            success=r.returncode==0
        except subprocess.TimeoutExpired:
            success=False
    if not success:
        save(out/'refinement-result.json',{'status':'bounded_attempt_incomplete','wall_seconds':time.monotonic()-start,
            'production_tracks_replaced':False,'gate_mm':.1,'scope':'Initial 40 seeds only; inspect refinement-run.log.'})
        return
    coarse=TRACKS/'individualized-qa-3p125us';fine=out/'individualized-qa-1p5625us'
    worst=0.;maximum=0.;rows=[]
    with h5py.File(coarse/'tracks.h5','r') as a,h5py.File(fine/'tracks.h5','r') as b:
        assert list(a['frames'])==list(b['frames'])
        for key in a['frames']:
            x=a['frames/'+key];y=b['frames/'+key]
            assert abs(x.attrs['time_s']-y.attrs['time_s'])<1e-12
            ids,ia,ib=np.intersect1d(x['id'][:],y['id'][:],return_indices=True)
            dist=np.linalg.norm(x['position_m'][:][ia]-y['position_m'][:][ib],axis=1)*1000
            p95=float(np.quantile(dist,.95)) if len(dist) else 0.
            m=float(dist.max()) if len(dist) else 0.;worst=max(worst,p95);maximum=max(maximum,m)
            rows.append({'time_s':float(x.attrs['time_s']),'common_particles':len(ids),'p95_mm':p95,'max_mm':m})
    ma=load(coarse/'tracking-audit.json');mb=load(fine/'tracking-audit.json')
    exits=[{r['id']:r['zone'] for r in m['exit_events']} for m in [ma,mb]]
    mismatch=sum(exits[0].get(i,'inside')!=exits[1].get(i,'inside') for i in range(40))/40
    passed=worst<=.1 and mismatch<=.01
    save(out/'refinement-result.json',{'status':'passed_initial_cohort' if passed else 'needs_refinement',
        'coarse_step_us':3.125,'fine_step_us':1.5625,'maximum_frame_p95_mm':worst,'maximum_position_difference_mm':maximum,
        'destination_disagreement_fraction':mismatch,'gate_mm':.1,'destination_gate':.01,'frames':rows,
        'production_tracks_replaced':False,'source_periodicity_validated':False,
        'wall_seconds':time.monotonic()-start,'scope':'Initial 40 ICA seeds, not the full released production cohort. Existing 6.25-us individualized videos remain unconverged.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker(a.out) if a.worker else main(a.out/'refinement')
