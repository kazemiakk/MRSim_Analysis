"""Export requested source-preserving E-031 inlet waveform/audit figures."""
import argparse
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_aorta_inlet_picker import sha

BLUE='#245b82'
ORANGE='#bd671c'
GRAY='#4b5563'


def check_report(report):
    if report.get('experiment')!='E-031':
        raise ValueError('Expected E-031 evidence')
    if len(report['native_cfd_waveform'])!=72 or len(report['matched_bins'])!=8:
        raise ValueError('Unexpected native or cardiac-bin sample count')
    if any(r['koma_inlet_ml_s'] is not None for r in report['matched_bins']):
        raise ValueError('This presentation expects unavailable Koma inlet values')
    for name in ('core05','core15'):
        rows=[r for r in report['coverage_by_offset'] if r['support']==name]
        if [r['inward_offset_mm'] for r in rows]!=[0,4,8,12,16,20]:
            raise ValueError('Unexpected offset coverage sequence')
        for r in rows:
            if not 0<=r['strict_stencil_area_fraction']<=r['any_stencil_neighbor_area_fraction']<=1:
                raise ValueError('Invalid coverage fractions')


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':GRAY,'text.color':'#222222',
        'axes.labelcolor':'#222222','xtick.color':GRAY,'ytick.color':GRAY,
        'legend.frameon':False,'pdf.fonttype':42})


def isolated_style(function):
    """Do not leak non-gray text/axis settings into other scientific renderers."""
    @wraps(function)
    def wrapped(*args,**kwargs):
        with plt.rc_context():
            return function(*args,**kwargs)
    return wrapped


@isolated_style
def waveform_figure(report):
    check_report(report);style()
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    native=report['native_cfd_waveform'];bins=report['matched_bins']
    t=np.array([r['source_time_s'] for r in native]);q=np.array([r['flow_ml_s'] for r in native])
    axes[0].plot(t,q,'-o',color=BLUE,lw=1.8,ms=3,label='CFD: 72 saved source frames')
    axes[0].set(title='Native CFD inlet waveform',xlabel='Source elapsed time (s)',ylabel='Inward flow (mL/s)',
                xlim=(0,report['timing']['native_source_period_s']))
    tb=np.array([r['nominal_time_s'] for r in bins]);qb=np.array([r['cfd_adc_average_ml_s'] for r in bins])
    axes[1].plot(tb,qb,'--s',color=ORANGE,lw=1.8,ms=6,label='CFD: actual ADC-time average for each MRI bin')
    axes[1].set(title='CFD comparator on the MRI acquisition timeline',xlabel='Nominal MRI bin center (s)',
                ylabel='Inward flow (mL/s)',xlim=(0,report['timing']['mri_period_s']))
    axes[1].text(.98,.96,'Koma inlet waveform: unavailable, not zero',transform=axes[1].transAxes,
                 ha='right',va='top',fontsize=10,color=GRAY)
    ymax=max(q.max(),qb.max())*1.18;ymin=min(0,float(q.min()),float(qb.min()))
    for ax in axes:
        ax.set_ylim(ymin,ymax);ax.grid(axis='y',color='#dedede',lw=.7)
        ax.legend(loc='upper left',fontsize=10)
    fig.suptitle('Aortic inlet flow | confirmed original ANSYS origin face',fontsize=15,y=.97)
    fig.text(.08,.055,'Approximate planar face integration; native cap connectivity is unverified. No smoothing or fitted alignment.',fontsize=10,color=GRAY)
    fig.text(.08,.032,'Separate time origins: native source order above; frame-18 start and retiming below. Lines only connect saved values.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.1,right=.97,top=.89,bottom=.15,hspace=.5)
    return fig


@isolated_style
def audit_figure(report):
    check_report(report);style()
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    bins=report['matched_bins'];b=np.array([r['bin'] for r in bins])
    lo=100*np.array([r['magnitude_fraction_of_global_max_min'] for r in bins])
    hi=100*np.array([r['magnitude_fraction_of_global_max_max'] for r in bins])
    axes[0].vlines(b,lo,hi,color=BLUE,lw=2,label='Inlet sampled min–max (not uncertainty)')
    axes[0].plot(b,hi,'o',color=BLUE,ms=5,label='Sampled maximum')
    axes[0].axhline(5,color=GRAY,ls='--',lw=1.3,label='5% magnitude threshold')
    axes[0].axhline(15,color=GRAY,ls=':',lw=1.6,label='15% magnitude threshold')
    axes[0].set(title='Magnitude at the confirmed inlet face',xlabel='MRI cardiac bin',
        ylabel="Reference magnitude / each bin's global maximum (%)",xticks=b,ylim=(0,16.5),xlim=(.5,8.5))
    axes[0].legend(loc='upper left',bbox_to_anchor=(0, .88),fontsize=9)
    for name,label,color,marker in [('core05','5% fixed all-bin mask',BLUE,'o'),('core15','15% fixed all-bin mask',ORANGE,'s')]:
        rows=[r for r in report['coverage_by_offset'] if r['support']==name]
        x=[r['inward_offset_mm'] for r in rows]
        strict=100*np.array([r['strict_stencil_area_fraction'] for r in rows])
        partial=100*np.array([r['any_stencil_neighbor_area_fraction'] for r in rows])
        axes[1].plot(x,strict,color=color,marker=marker,lw=1.8,label=label+': complete stencil')
        axes[1].plot(x,partial,color=color,marker=marker,mfc='white',ls='--',lw=1.4,label=label+': any overlap only')
    axes[1].set(title='MRI support versus inward offset',xlabel='Offset from inlet into vessel (mm)',
        ylabel='Translated face footprint area (%)',xticks=[0,4,8,12,16,20],xlim=(-.5,20.5),ylim=(-2,105))
    axes[1].legend(loc='upper left',fontsize=8.5)
    for ax in axes:ax.grid(axis='y',color='#dedede',lw=.7)
    fig.suptitle('Aortic inlet MRI coverage audit',fontsize=15,y=.97)
    fig.text(.075,.09,'At the inlet (0 mm): no overlap with fixed all-bin masks. Missing inlet flow is NOT a zero-valued measurement.',fontsize=10,color=GRAY)
    fig.text(.075,.055,'Dashed overlap curves do not establish usable interpolation. Offset footprints are diagnostics, not downstream lumen contours.',fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.075,right=.97,top=.85,bottom=.23,wspace=.3)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/aorta_inlet_flow/e031_v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.source/'MANIFEST.json').read_text())
    for name,entry in manifest.items():
        p=args.source/name
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
            raise ValueError('E-031 input artifact mismatch')
    report=json.loads((args.source/'inlet_readiness.json').read_text());check_report(report)
    temporary=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    for name,maker in [('cfd_inlet_waveform',waveform_figure),('inlet_mri_coverage_audit',audit_figure)]:
        with plt.rc_context():
            style()
            fig=maker(report)
            for suffix in ('png','pdf'):fig.savefig(temporary/(name+'.'+suffix),dpi=220)
            plt.close(fig)
    values={k:report[k] for k in ('native_cfd_waveform','matched_bins','coverage_by_offset','timing','geometry','limitations')}
    (temporary/'plotted_source_values.json').write_text(json.dumps(values,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    final=dict(experiment='E-031 presentation derivative',created_utc=datetime.now(timezone.utc).isoformat(),
       source_manifest_sha256=sha(args.source/'MANIFEST.json'),source_report_sha256=sha(args.source/'inlet_readiness.json'),
       script_sha256=sha(__file__),files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir()})
    (temporary/'MANIFEST.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    temporary.rename(args.output)
    print(args.output)


if __name__=='__main__':main()
