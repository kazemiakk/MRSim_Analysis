"""Source-preserving corrected inlet pilot waveform and density figures."""
import argparse
import json
from pathlib import Path
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.build_aorta_inlet_picker import sha
from scripts.plot_aorta_inlet_readiness import isolated_style,style,BLUE,ORANGE,GRAY


@isolated_style
def matched_figure(audit,times):
    style();fig,ax=plt.subplots(figsize=(11,6))
    cfd=np.array(audit['CFD_partial_ml_s']);koma=np.array(audit['Koma_partial_ml_s'])
    ax.plot(times,cfd,'-o',color=BLUE,lw=2,ms=6,label='CFD: actual ADC-time average')
    ax.plot(times,koma,'--s',color=ORANGE,lw=2,ms=6,label='Koma: corrected 100k-spin pilot')
    ax.set(xlabel='Nominal MRI bin center (s)',ylabel='Signed inward partial flow (mL/s)',
        title='Confirmed aortic inlet | CFD versus corrected Koma pilot',
        ylim=(min(0,float(cfd.min()),float(koma.min()))*1.15,max(cfd.max(),koma.max())*1.2))
    ax.grid(axis='y',color='#dedede',lw=.7);ax.legend(loc='upper left',fontsize=10)
    fig.text(.1,.12,f"Same fixed partial area: {audit['paired_partial_area_mm2']:.1f} mm² ({100*audit['fixed05_area_fraction']:.1f}% of inferred face). No fitted shift or amplitude scaling.",fontsize=10,color=GRAY)
    fig.text(.1,.075,f"RMSE {audit['partial_flow_rmse_ml_s']:.2f} mL/s | r {audit['partial_flow_correlation']:.4f} | cycle-volume Koma/CFD {audit['cycle_volume_ratio']:.3f}",fontsize=10,color=GRAY)
    fig.text(.1,.035,'Dependent internal consistency; approximate point-cloud/cap geometry. Not full-lumen flow or independent validation.',fontsize=9.5,color=GRAY)
    fig.subplots_adjust(left=.1,right=.97,top=.88,bottom=.25)
    return fig


@isolated_style
def density_figure(old,corrected):
    style();fig,ax=plt.subplots(figsize=(11,5.5))
    t=[r['source_elapsed_s'] for r in old['particle_count_series']]
    y=100*np.array([r['counts'][0] for r in old['particle_count_series']])/old['summary']['first_slab_initial']
    cy=100*np.array(corrected['recorded_slab_counts'])[:,0]/corrected['baseline_counts'][0]
    ct=(np.arange(360)+1)*.002
    ax.plot(t,y,color=GRAY,lw=2,label='Preserved 1M input: volume-pool recycling')
    ax.plot(ct,cy,color=BLUE,lw=1.5,label='Corrected 100k control, recorded cycle after warm-up')
    ax.axhline(100,color='#777777',ls=':',lw=1)
    ax.set(xlabel='Recorded cycle elapsed time (s)',ylabel='First 4 mm inlet-prism count / initial count (%)',
        title='Local inlet particle retention',xlim=(0,.72),ylim=(0,140))
    ax.grid(axis='y',color='#dedede',lw=.7);ax.legend(loc='upper right',fontsize=10)
    fig.text(.1,.065,'Fixed inferred-face prism, not certified lumen density. Different particle counts: this is not a convergence study.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.1,right=.97,top=.87,bottom=.2)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];audit=json.loads(args.audit.read_text())
    if audit['status']!='PASS_MRI_PILOT_VIABILITY':raise ValueError('Do not promote a failed pilot to matched figures')
    tracking_path=root/'outputs/aorta_inlet_replenishment/e033_100k_v1/tracking_report.json'
    old_path=root/'outputs/aorta_inlet_flow/e032_diagnosis_v1/inlet_signal_diagnosis.json'
    times=[r['nominal_time_s'] for r in json.loads((root/'outputs/aorta_inlet_flow/e031_v1/inlet_readiness.json').read_text())['matched_bins']]
    temporary=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    figures=[('cfd_vs_koma_inlet_partial_waveform',matched_figure(audit,times)),
        ('inlet_particle_retention',density_figure(json.loads(old_path.read_text()),json.loads(tracking_path.read_text())))]
    for name,fig in figures:
        with plt.rc_context():
            style()
            for suffix in ('png','pdf'):fig.savefig(temporary/(name+'.'+suffix),dpi=220)
        plt.close(fig)
    manifest=dict(experiment='E-034 presentation',audit_sha256=sha(args.audit),tracking_report_sha256=sha(tracking_path),
       diagnosis_sha256=sha(old_path),plotter_sha256=sha(__file__),
       files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir()})
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    temporary.rename(args.output);print(args.output)


if __name__=='__main__':main()
