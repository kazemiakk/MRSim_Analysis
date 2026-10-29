"""Source-bound E-036 waveform and inferred-face coverage PNG/PDF exports."""
import argparse
import json
from pathlib import Path
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
from scripts.build_aorta_inlet_picker import sha
from scripts.plot_aorta_inlet_readiness import BLUE,ORANGE,GRAY,isolated_style,style


@isolated_style
def native_waveform(report):
    style();fig,ax=plt.subplots(figsize=(11.8,7))
    t=np.asarray(report['native_source_phase_centers_s']);order=np.argsort(t)
    full=np.asarray(report['CFD_full_native_waveform_ml_s'])
    cfd=np.asarray(report['CFD_native_time_ml_s']);koma=np.asarray(report['Koma_native_time_ml_s'])
    ax.plot(np.arange(72)*.01,full,'-o',color=BLUE,lw=1.6,ms=2.5,label='Native CFD: whole inferred face, 72 saved frames')
    ax.plot(t[order],cfd[order],'o',mfc='white',mec=GRAY,mew=1.5,ms=8,
        label='CFD: same-area actual ADC averages, 8 bins')
    ax.plot(t[order],koma[order],'--s',color=ORANGE,lw=1.8,ms=6,
        label='Koma: complex-image integration, 8 bins')
    ax.set(title='Aortic inlet | complex-image integration on native CFD time',
        xlabel='Native CFD cycle time (s)',ylabel='Inward flow (mL/source-s)',
        xlim=(0,.72),ylim=(0,max(full.max(),koma.max(),cfd.max())*1.2))
    ax.grid(axis='y',color='#dedede',lw=.7);ax.legend(loc='upper right',fontsize=10)
    m=report['physical_units_metrics'];scale=report['known_source_time_velocity_scale']
    fig.text(.09,.16,f"Fixed paired area: {report['coverage_fraction']*100:.3f}% of inferred inlet; {report['excluded_area_mm2']:.3f} mm² excluded. Unchanged 5% signal gate.",fontsize=10,color=GRAY)
    fig.text(.09,.115,f"Same-area ADC comparison: r {m['correlation']:.4f} | native-time RMSE {m['rmse_ml_s']/scale:.2f} mL/source-s | cycle volume {100*(m['nominal_bin_cycle_volume_ratio']-1):+.2f}%.",fontsize=10,color=GRAY)
    fig.text(.09,.07,f"Known phase mapping only; paired flows divided by fixed time scale {scale:.7f} to use native CFD units. No fitted gain.",fontsize=9.5,color=GRAY)
    fig.text(.09,.03,'Post hoc 100k-pilot internal consistency; not independent validation, certified cap geometry or a 72-frame MRI recovery.',fontsize=9.5,color=GRAY)
    fig.subplots_adjust(left=.09,right=.97,top=.9,bottom=.28)
    return fig


@isolated_style
def coverage_figure(points,triangles,old,new,report):
    style();fig,axes=plt.subplots(1,2,figsize=(10.5,5.7))
    percentages=(report['original_partial_area_fraction'],report['coverage_fraction'])
    for ax,keep,title,fraction in zip(axes,(old,new),('Original complete-stencil guard','Complex-image point-signal gate'),percentages):
        colors=np.where(keep[:,None],np.array(matplotlib.colors.to_rgb(BLUE))[None,:],np.array([.87,.87,.87])[None,:])
        ax.add_collection(PolyCollection(points[triangles],facecolors=colors,edgecolors='none',antialiaseds=False,rasterized=True))
        ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='Inferred inlet basis u (mm)',ylabel='Inferred inlet basis v (mm)')
        ax.set_title(f'{title}\n{100*fraction:.3f}% retained',fontsize=12,pad=12)
    fig.legend(handles=[Patch(facecolor=BLUE,label='Retained fixed support'),Patch(facecolor='#dedede',label='Excluded / unavailable')],
        loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False)
    fig.text(.08,.035,'Inferred face triangulation, not certified wall/cap geometry. Signal support does not establish velocity accuracy.',fontsize=9.5,color=GRAY)
    fig.subplots_adjust(left=.08,right=.97,top=.85,bottom=.25,wspace=.28)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];report_path=args.run/'complex_inlet_report.json'
    report=json.loads(report_path.read_text());e=np.load(args.run/'complex_inlet_evidence.npz',allow_pickle=False)
    zpath=root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    z=np.load(zpath,allow_pickle=False)
    geom_path=root/'outputs/aorta_inlet_picker/e030_origin_v1/origin_inlet_review.json'
    geometry=json.loads(geom_path.read_text())['candidate']
    points=np.column_stack((z['face_points_source_mm']@geometry['basis_u'],z['face_points_source_mm']@geometry['basis_v']))
    old_path=root/'outputs/aorta_inlet_replenishment/e034_100k_audit_v1/paired_partial_flow_evidence.npz'
    old=np.load(old_path,allow_pickle=False)['common']
    staging=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);staging.mkdir(parents=True)
    figures=[('native_cfd_vs_koma_complex_inlet',native_waveform(report)),
             ('inlet_coverage_comparison',coverage_figure(points,z['inferred_triangles'],old,e['fixed'],report))]
    for name,fig in figures:
        with plt.rc_context():
            style()
            for suffix in ('png','pdf'):fig.savefig(staging/(name+'.'+suffix),dpi=220)
        plt.close(fig)
    manifest=dict(experiment='E-036 figures',report_sha256=sha(report_path),plotter_sha256=sha(__file__),
        quadrature_sha256=sha(zpath),geometry_sha256=sha(geom_path),old_support_sha256=sha(old_path),
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in staging.iterdir()})
    (staging/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    staging.rename(args.output);print(args.output)


if __name__=='__main__':main()
