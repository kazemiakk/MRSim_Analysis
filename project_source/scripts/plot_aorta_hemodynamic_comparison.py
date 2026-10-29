"""Publication figures derived from immutable E-027 saved numerical fields."""
import argparse
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.results/'report.json').read_text())
    with h5py.File(args.results/'matched_reference.h5','r') as f:
        ref=f['adc_time_average'][...];est=f['Koma'][...];mask=f['masks/core15'][...].astype(bool);frac=f['fraction'][...]
    spacing=np.asarray(report['grid']['fov'])/np.asarray(report['grid']['shape'][::-1])
    extent=[-140-spacing[0]*500,140-spacing[0]*500,-64-spacing[1]*500,64-spacing[1]*500]
    w=np.where(mask,frac,0);den=w.sum(axis=0)
    def project(v):return np.divide(np.sum(np.where(mask,v,0)*w,axis=0),den,out=np.full(den.shape,np.nan),where=den>0)
    b=7;speed_ref=project(np.linalg.norm(ref[b],axis=-1));speed_est=project(np.linalg.norm(est[b],axis=-1))
    plt.rcParams.update({'font.size':11})
    fig=plt.figure(figsize=(10,12),layout='constrained');gs=fig.add_gridspec(3,2,width_ratios=[1,.025])
    speed_max=np.nanquantile(np.r_[speed_ref.ravel(),speed_est.ravel()],.995)
    diff=speed_est-speed_ref; diff_max=np.nanmax(abs(diff))
    for row,(values,title,cmap,lo,hi) in enumerate([(speed_ref,'CFD speed','viridis',0,speed_max),
        (speed_est,'Koma speed','viridis',0,speed_max),(diff,'Koma minus CFD','RdBu_r',-diff_max,diff_max)]):
        ax=fig.add_subplot(gs[row,0]);cax=fig.add_subplot(gs[row,1])
        img=ax.imshow(values,origin='lower',extent=extent,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
        ax.set(title=title+' | bin8 | support-weighted XY projection',xlabel='MRI readout X (mm)',ylabel='MRI phase Y (mm)')
        fig.colorbar(img,cax=cax,label='m/s')
    for ext in ('png','pdf'):fig.savefig(args.output/('aorta_speed_comparison_bin08.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for component in range(3):
        maps=[project(u[b,...,component]) for u in (ref,est)]
        vmax=np.nanquantile(abs(np.r_[maps[0].ravel(),maps[1].ravel()]),.995)
        for j,(values,name) in enumerate(zip(maps,('CFD','Koma'))):
            img=axes[component,j].imshow(values,origin='lower',extent=extent,cmap='gray',vmin=-vmax,vmax=vmax,interpolation='nearest')
            axes[component,j].set(title=f'{name} velocity {"XYZ"[component]} | bin8',xlabel='MRI readout X (mm)',ylabel='MRI phase Y (mm)')
            fig.colorbar(img,ax=axes[component,j],label='m/s',shrink=.7)
    fig.suptitle('Signed grayscale component velocity: black negative, gray zero, white positive')
    for ext in ('png','pdf'):fig.savefig(args.output/('aorta_components_bin08.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,section in zip(axes.ravel(),report['section_summary']):
        for method in ('CFD','Koma'):
            rows=[r for r in report['profile_series'] if r['section']==section['section'] and r['bin']==8 and r['method']==method and r['transverse_axis']=='phase_Y']
            ax.plot([r['coordinate_mm'] for r in rows],[r['normal_velocity_m_s'] for r in rows],'-o',ms=3,label=method)
        ax.set(title=section['section']+' | bin8',xlabel='MRI phase Y (mm)',ylabel='Normal velocity (m/s)');ax.legend();ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(args.output/('aorta_velocity_profiles_bin08.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    manifest={'source_report_sha256':digest(args.results/'report.json'),
              'plot_script_sha256':digest(__file__),
              'extent_xy_mm':extent,
              'files':{f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in args.output.iterdir()}}
    (args.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
