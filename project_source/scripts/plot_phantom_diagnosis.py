"""E048 standalone diagnostic PNG/PDF sheets and inspectable numerical CSVs."""
import argparse
import csv
import json
from pathlib import Path
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.diagnose_phantom_signal import BASE,dump
from scripts.diagnose_phantom_encoding_kernel import phase_kernel
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import ROOT


def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def plot(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    basic=json.loads((BASE/'basic_v1/report.json').read_text());reference=json.loads((BASE/'reference_v1/report.json').read_text())
    local=json.loads((BASE/'local_v1/report.json').read_text());kernel=json.loads((BASE/'kernel_v2/report.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(map20k='#235f82',legacy100k='#626973',map100k='#bc721d',legacy1m='#758547')
    label=dict(map20k='Map20k',legacy100k='Legacy100k',map100k='Map100k',legacy1m='Legacy1M')
    def save(fig,name,note):
        fig.text(.075,.017,note,fontsize=9,color='#555c65')
        fig.tight_layout(rect=(.01,.065,.99,.95))
        for ext in ('png','pdf'):fig.savefig(stage/f'{name}.{ext}',dpi=400)
        plt.close(fig)
    arms=list(colors);x=np.arange(4);fig,axs=plt.subplots(1,2,figsize=(13.5,5.3))
    for i,(name,color) in enumerate([('Mean per bin','#aab1ba'),('REF fixed','#235f82'),('All enc. fixed','#bc721d')]):
        values=[]
        for arm in arms:
            r=basic['cases'][arm]
            values.append(100*(np.mean([a['all_encoding_volume_fraction'] for a in r['all_encoding_per_bin']]) if i==0 else
                        r['fixed_encoding_volume_fraction']['REF'] if i==1 else r['fixed_all_bin_all_encoding_volume_fraction']))
        axs[0].bar(x+(i-1)*.23,values,.23,label=name,color=color)
    axs[0].set(xticks=x,xticklabels=['Map\n20k','Legacy\n100k','Map\n100k','Legacy\n1M'],ylabel='Retained numerical volume (%)',ylim=(0,105),title='Coverage | unchanged 5% gate')
    axs[0].legend(frameon=False,fontsize=9,ncol=1,loc='upper left')
    for arm in arms:
        values=[r['all_encoding_volume_fraction']*100 for r in basic['cases'][arm]['all_encoding_per_bin']]
        axs[1].plot(np.arange(1,16),values,'o-' if arm!='legacy100k' else 's--',ms=3,color=colors[arm],label=label[arm])
    axs[1].set(xlabel='Acquired bin',ylabel='Retained numerical volume (%)',ylim=(0,105),title='Each bin | all encodings')
    axs[1].legend(frameon=False,fontsize=9,ncol=2)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Phantom signal coverage | time intersection and density',x=.075,ha='left')
    save(fig,'signal_coverage','Original E047 geometry denominator. Legacy1M is saved pre-remapping data; this is not mapped convergence.')
    # Reference alternatives compare the exact same retained voxel set/weights.
    names=['half_mid','half_adc','fft_mid','fft_adc','fft_adc_f32'];xt=['Old grid\nMidpoint','Old grid\nADC','FFT grid\nMidpoint','FFT grid\nADC','FFT grid\nADC f32']
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.3))
    for ax,region in zip(axs,['whole','throat']):
        values=[next(r['metrics']['vector_rmse_m_s'] for r in reference['results'] if r['region']==region and r['arm']=='map100k' and r['reference']==name) for name in names]
        ax.bar(np.arange(5),values,color=['#626973','#626973','#bc721d','#bc721d','#bc721d'])
        ax.set(xticks=np.arange(5),xticklabels=xt,ylabel='Vector RMSE (m/s)',title='Whole shared support' if region=='whole' else 'Throat shared support',ylim=(0,max(values)*1.22))
        for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.4f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Reference-coordinate and timing sensitivity | Map100k',x=.075,ha='left')
    save(fig,'reference_sensitivity','Identical paired100k support and min geometry weights; no fitted shift or signal changes. ADC f32 is a precision sensitivity.')
    # Historical density response is compared on one fixed shared population.
    names=['legacy100k','map100k','legacy1m'];fig,axs=plt.subplots(1,3,figsize=(14,5.4))
    coverage=[local['cases'][a]['own_coverage_fraction']*100 for a in names]
    axs[0].bar(np.arange(3),coverage,color=[colors[a] for a in names]);axs[0].set(ylabel='Retained numerical volume (%)',ylim=(0,100),title='Own fixed support | FFT geometry')
    for ax,region in zip(axs[1:],['whole','throat']):
        values=[next(r['shared_metrics']['vector_rmse_m_s'] for r in local['regions'] if r['arm']==a and r['region']==region) for a in names]
        ax.bar(np.arange(3),values,color=[colors[a] for a in names]);ax.set(ylabel='Vector RMSE (m/s)',ylim=(0,max(values)*1.22),title=f'{region.capitalize()} | identical shared support')
        for i,v in enumerate(values):ax.text(i,v+max(values)*.03,f'{v:.4f}',ha='center',fontsize=9)
    for ax in axs:
        ax.set(xticks=np.arange(3),xticklabels=['Legacy\n100k','Map\n100k','Legacy\n1M']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Density response | centered-grid, ADC-averaged CFD reference',x=.075,ha='left')
    save(fig,'density_and_local_error','Shared-support errors have matched denominators; old1M improves global error more than throat error. No new1M simulation.')
    # First-moment timing derives from saved gradients, not a fitted time shift.
    a,b,knots,_=phase_kernel(ROOT/'koma_dp150/pc_3d_scanner_matched_venc5')
    t=np.linspace(knots[0],a['echo'],1500);delta=b['gradient'](t)-a['gradient'](t)
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.4))
    axs[0].plot((t-a['echo'])*1000,delta/1e6,color='#235f82')
    axs[0].axvline(kernel['equivalent_acceleration_time_offset_s']*1000,color='#bc721d',ls='--',label='Moment-derived offset')
    axs[0].axvline(0,color='#626973',lw=.8);axs[0].axhline(0,color='#626973',lw=.8)
    axs[0].set(xlabel='Time relative to echo (ms)',ylabel='X − REF gradient (10⁶ Hz/m)',title='Velocity-encoding gradient difference')
    axs[0].legend(frameon=False,fontsize=9)
    regions=['upstream','throat','near_downstream'];pos=np.arange(3)
    for offset,key,title,color in [(-.18,'Koma_vs_instant_particles_axial_rmse_m_s','Instant. particles','#626973'),
                                  (.18,'Koma_vs_gradient_proxy_axial_rmse_m_s','Gradient history','#bc721d')]:
        values=[next(r[key] for r in kernel['regional'] if r['region']==region) for region in regions]
        axs[1].bar(pos+offset,values,.36,label=title,color=color)
    axs[1].set(xticks=pos,xticklabels=['Upstream','Throat','Near downstream'],ylabel='Axial RMSE versus Koma (m/s)',title='Proxy agreement | identical samples')
    axs[1].legend(frameon=False,fontsize=9)
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Trajectory / gradient-history diagnostic',x=.075,ha='left')
    save(fig,'encoding_history','Equal-spin proxy, not full Bloch, fitted alignment, correction, causal fraction or independent velocity validation.')
    fig,axs=plt.subplots(2,2,figsize=(13.5,8));times=(np.arange(15)+.5)*1.0120000839233398/15
    for col,section in enumerate(['upstream','throat']):
        rows=[r for r in local['sections'] if r['section']==section and r['support']=='shared_three_arm']
        coverage=rows[0]['coverage_fraction']
        axs[0,col].plot(times,rows[0]['CFD_ml_s'],'k-',label='CFD',lw=1.7)
        for arm in names:
            row=next(r for r in rows if r['arm']==arm)
            axs[0,col].plot(times,row['Koma_ml_s'],'o--',label=label[arm],ms=3,color=colors[arm])
            axs[1,col].plot(times,np.array(row['Koma_ml_s'])-row['CFD_ml_s'],'o-',ms=3,color=colors[arm])
        axs[0,col].set(title=f'{section.capitalize()} | shared area {coverage:.1%}',ylabel='Partial flow (mL/s)')
        axs[0,col].legend(frameon=False,ncol=2,fontsize=9)
        axs[1,col].axhline(0,color='#626973',lw=.8)
        axs[1,col].set(xlabel='Nominal MRI bin center (s)',ylabel='Koma − CFD (mL/s)')
    for ax in axs.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('True-plane flow | centered grid, same retained area',x=.075,ha='left')
    save(fig,'plane_flow_comparison','ADC-averaged CFD; quadrature-estimated areas, not full-section accuracy. Final bins14/15 overlap the controlled synthetic CFD tail.')
    # Numerical exports are unrounded; missing values stay blank, not zero.
    table(stage/'signal_coverage.csv',[dict(case=a,per_bin_mean=np.mean([x['all_encoding_volume_fraction'] for x in r['all_encoding_per_bin']]),
         REF_fixed=r['fixed_encoding_volume_fraction']['REF'],all_fixed=r['fixed_all_bin_all_encoding_volume_fraction'],
         upstream_area=r['fixed_upstream_area_fraction']) for a,r in basic['cases'].items()])
    table(stage/'reference_errors.csv',[dict(region=r['region'],reference=r['reference'],arm=r['arm'],
         voxels=r['metrics']['voxels'],vector_rmse_m_s=r['metrics']['vector_rmse_m_s'],
         vector_nrmse=r['metrics']['vector_nrmse_reference_rms']) for r in reference['results'] if r['metrics']])
    table(stage/'true_plane_flows.csv',[dict(section=r['section'],arm=r['arm'],support=r['support'],
         retained_area_mm2=r['area_mm2'],geometric_area_mm2=r['geometric_area_mm2'],coverage_fraction=r['coverage_fraction'],
         flow_rmse_ml_s=r['metrics']['rmse_ml_s'],cycle_volume_error_fraction=r['metrics']['cycle_volume_error_fraction']) for r in local['sections']])
    table(stage/'encoding_history.csv',kernel['regional'])
    dump(stage/'MANIFEST.json',dict(inputs={str(p):sha(p) for p in [Path(__file__).resolve(),BASE/'basic_v1/report.json',BASE/'reference_v1/report.json',
         BASE/'local_v1/report.json',BASE/'kernel_v2/report.json']},outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output);print('COMPLETE_E048_FIVE_PNG_PDF_PAIRS_AND_FOUR_CSVS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);plot(p.parse_args().output)
