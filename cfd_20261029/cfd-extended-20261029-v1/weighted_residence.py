"""Post-hoc inlet-volume weights for the existing uniform-area visual seed sample."""
import argparse,csv
import numpy as np
from extended_common import *


def weighted_survival(duration,event,weight):
    duration=np.round(duration,9);rows=[{'age_s':0.,'survival':1.,'at_risk_weight_ml':float(weight.sum())}];s=1.
    for t in np.unique(duration):
        risk=float(weight[duration>=t].sum());dead=float(weight[(duration==t)&event].sum())
        if risk>0:s*=1-dead/risk
        rows.append({'age_s':float(t),'survival':float(s),'at_risk_weight_ml':risk})
    return rows


def integrate_bins(knots,flow,edges):
    full=np.unique(np.r_[knots,edges]);v=np.interp(full,knots,flow)
    cum=np.r_[0.,np.cumsum((v[1:]+v[:-1])/2*np.diff(full))]
    integral=np.diff(np.interp(edges,full,cum))
    assert np.all(integral>0)
    return integral


def run(out):
    out.mkdir(exist_ok=False);summaries=[];curves=[];reconciliation=[]
    # Analytic linear-flow quadrature check, independent of patient inputs.
    assert np.allclose(integrate_bins([0,1],[1,3],[0,.25,1]),[.3125,1.6875])
    for kind in KINDS:
        cfg=load(RUN/kind/'input-configuration.json')
        times=np.array([s['time_s'] for s in load(RUN/kind/'progress.json')['accepted_steps']]);times[-1]=PERIOD
        rows=list(csv.DictReader((PREVIOUS/'residence'/(kind+'-particle-residence.csv')).open()))
        ids=np.array([int(r['particle_id']) for r in rows]);birth=np.array([float(r['birth_time_s']) for r in rows]);labels=np.array([int(r['inlet_id']) for r in rows])
        entered=np.array([r['entered_ROI']=='True' for r in rows]);event=np.array([r['first_exit_observed']=='True' for r in rows]);first=np.array([float(r['first_entry_s']) if r['first_entry_s'] else np.nan for r in rows])
        duration=np.array([float(r['first_pass_duration_or_censor_s']) if r['first_pass_duration_or_censor_s'] else 0 for r in rows])
        exposure=np.array([float(r['cumulative_observed_ROI_time_s']) for r in rows]);volume_weight=np.zeros(len(rows))
        for label,inlet in [(1,'inlet_b1'),(2,'inlet_b2')]:
            mask=labels==label;batches=np.unique(birth[mask]);edges=np.r_[START,(batches[:-1]+batches[1:])/2,START+10*PERIOD]
            qnative=np.interp(times,cfg['time_s'],cfg['inlets'][inlet]['flow_m3_s'])*1e6
            knots=np.r_[np.concatenate([times+i*PERIOD for i in range(10)]),START+10*PERIOD]
            q=np.r_[np.tile(qnative,10),qnative[0]];bins=integrate_bins(knots,q,edges)
            for t,vol in zip(batches,bins):
                part=mask&(birth==t);assert part.sum()==20;volume_weight[part]=vol/part.sum()
            exact=float(np.trapezoid(np.interp(np.unique(np.r_[knots,edges]),knots,q),np.unique(np.r_[knots,edges])))
            assert abs(volume_weight[mask].sum()-exact)<1e-9
            reconciliation.append({'case':kind,'inlet':inlet,'represented_injected_volume_ml':float(volume_weight[mask].sum()),'integrated_template_inflow_ml':exact,'relative_error':float(abs(volume_weight[mask].sum()/exact-1))})
        assert np.all(volume_weight>0)
        csvsave(out/(kind+'-particle-inlet-volume-weights.csv'),[{'particle_id':int(i),'inlet_id':int(l),'birth_s':float(t),'weight_ml':float(v)} for i,l,t,v in zip(ids,labels,birth,volume_weight)])
        scopes={'all_entrants':entered,'first_cycle_ROI_entrants':entered&(first<=START+PERIOD+1e-10),'first_injection_cycle_entrants':entered&(release_cycle(birth)==1)}
        for name,mask in scopes.items():
            for subset,selection in [('all',mask),('even_seed_slots',mask&(ids%2==0)),('odd_seed_slots',mask&(ids%2==1))]:
                ix=np.flatnonzero(selection);rr=weighted_survival(duration[ix],event[ix],volume_weight[ix])
                threshold=lambda x:next((r['age_s'] for r in rr if r['survival']<=x),None)
                curves.extend([{'case':kind,'cohort':name,'subset':subset,**r} for r in rr])
                summaries.append({'case':kind,'cohort':name,'subset':subset,'entering_tracers':len(ix),
                    'represented_entering_volume_ml':float(volume_weight[ix].sum()),
                    'weighted_t50_first_exit_s':threshold(.5),'weighted_t90_first_exit_s':threshold(.1),
                    'weighted_mean_observed_cumulative_exposure_s':float(np.average(exposure[ix],weights=volume_weight[ix])) if len(ix) else None,
                    'first_exit_censored_weight_fraction':float(volume_weight[ix][~event[ix]].sum()/volume_weight[ix].sum()) if len(ix) else None})
    csvsave(out/'flow-weighted-residence-summary.csv',summaries);csvsave(out/'flow-weighted-first-exit-curves.csv',curves)
    csvsave(out/'inlet-volume-weight-reconciliation.csv',reconciliation)
    save(out/'method.json',{'status':'completed_exploratory_reweighting','new_trajectories':False,
        'weights':'Integral of each inlet template flow over each release batch Voronoi time interval, divided equally over its 20 uniform-area seeds. Positive uniform-normal inlet profiles; native 5–868-ms flow template with 5-ms closing bridge.',
        'sampling':'Original 20 spatial seed locations per ICA reused; even/odd subsets are sensitivity diagnostics, not independent cohorts or established seed-count convergence.',
        'caveats':['Saved polyline positions and approved ellipsoid define the first-entry/exit events.',
            'This is not a fresh flux-sampled 250-particle-per-cycle experiment.',
            'Individualized trajectories remain unconverged; reweighting cannot fix them.',
            'First-cycle template periodicity is unvalidated; these are not validated blood washout times.',
            'Inlet-volume weights describe the represented injected sample; they are not instantaneous sac blood-volume fractions.']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out/'weighted-residence')
