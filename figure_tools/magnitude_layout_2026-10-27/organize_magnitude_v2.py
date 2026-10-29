"""Re-layout the original scientific panels with concise labels; no simulation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, importlib.util
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT=Path('D:/MRSim/outputs/magnitude_layout_2026-10-27/layout_v2')
AORTA=Path('D:/MRSim/outputs/aorta_presentation_2026-10-25/magnitude_orthoviews_1m_v2')
PHANTOM=Path('D:/MRSim/MRSIM/MRSimTracks/outputs/dp150_mri_pipeline/koma/simulation_static_2m_ref/pc_reconstruction.h5')
HELPER=Path('D:/MRSim/MRSIM/MRSimTracks/scripts/plot_dp150_koma_reconstruction.py')
ORIGINALS=Path('D:/MRSim/manuscripts/virtual_imaging_digital_twin_2026-10-26/draft_v4_user_figures/figures')
KEYS=['projection_xy','projection_xz','projection_yz','slice_xy','slice_xz','slice_yz']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    if OUT.exists() or OUT.with_name(OUT.name+'.building').exists():raise FileExistsError(OUT)
    expected={AORTA/'displayed_arrays.npz':'e5a7169a5ad1e63bdc224c0e34726f948475530b18700e9bcc5f45de3b091d0c',
        PHANTOM:'079712cfc44ef1909cf80eae798f4bdd41cc61a435a376288fd36ae460ec913b',
        ORIGINALS/'figure2a_aorta_original.png':'8784778343d6d2d9277876325792ac936f1877bc9c3ed0613ac67f2f94ef10f8',
        ORIGINALS/'figure2b_phantom_static_original.png':'f70e13a06dee0645cc6b9e33426584b7624b79c72a126bdf061dff23de93698f'}
    for p,digest in expected.items():assert sha(p)==digest,str(p)
    with np.load(AORTA/'displayed_arrays.npz',allow_pickle=False) as saved:
        aorta={key:saved[key].copy() for key in KEYS}
        axes=[saved[key].copy() for key in ('x_mm','y_mm','z_mm')]
    aorta_report=json.loads((AORTA/'REPORT.json').read_text(encoding='utf-8'))
    aorta_limit=aorta_report['display']['vmax']
    bounds=[]
    for axis in axes:
        step=axis[1]-axis[0];bounds.append([float(axis[0]-step/2),float(axis[-1]+step/2)])
    bx,by,bz=bounds
    aorta_extents=[bx+by,bx+bz,by+bz]*2
    # Reuse the exact original static renderer's six panels, extents and window.
    spec=importlib.util.spec_from_file_location('original_static_plot',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with h5py.File(PHANTOM,'r') as source:
        assert source['reference_magnitude'].shape==(2,64,64,256)
        volume=np.asarray(source['reference_magnitude'][1],dtype=np.float32)
    phantom_specs=helper.panel_data(volume)
    phantom={key:panel[1].copy() for key,panel in zip(KEYS,phantom_specs)}
    phantom_extents=[panel[4] for panel in phantom_specs]
    phantom_limit=helper.robust_limit(volume)
    assert np.array_equal(phantom['projection_xy'],volume.max(axis=0))
    assert np.array_equal(phantom['projection_xz'],volume.max(axis=1))
    assert np.array_equal(phantom['projection_yz'],volume.max(axis=2))
    stage=OUT.with_name(OUT.name+'.building');stage.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.titlesize':22,'axes.labelsize':18,
        'xtick.labelsize':16,'ytick.labelsize':16,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(18,15),facecolor='white')
    columns=[(.08,.34),(.455,.34),(.83,.15)]
    checked=[];axes_objects=[]
    for case,top,arrays,extents,limit in [('aorta',.935,aorta,aorta_extents,aorta_limit),('phantom_static',.455,phantom,phantom_extents,phantom_limit)]:
        # The same three column widths and row centers make the static phantom
        # as readable as the aorta, while physical image aspect ratios remain equal.
        for row in range(2):
            y=top-.174-row*.207
            for col,(x,w) in enumerate(columns):
                index=row*3+col;key=KEYS[index]
                ax=fig.add_axes([x,y,w,.174]);axes_objects.append(ax)
                artist=ax.imshow(arrays[key],cmap='gray',origin='lower',extent=extents[index],vmin=0,vmax=limit,
                    interpolation='nearest',aspect='equal')
                if row==0:
                    fig.text(x+w/2,top+.012,('XY','XZ','YZ')[col],ha='center',va='bottom',fontsize=22)
                else:
                    ax.set_xlabel(('X','X','Y')[col],labelpad=4)
                ax.set_ylabel(('Y','Z','Z')[col],labelpad=4)
                ax.tick_params(length=3,width=.7,pad=4)
                ax.grid(False)
                for spine in ax.spines.values():spine.set_linewidth(.7)
                assert np.array_equal(np.asarray(artist.get_array()),arrays[key])
                assert tuple(artist.get_clim())==(0,limit)
                assert list(artist.get_extent())==extents[index]
                assert artist.get_interpolation()=='nearest'
                checked.append({'case':case,'panel':key,'array_shape':list(arrays[key].shape),
                    'array_sha256':hashlib.sha256(arrays[key].tobytes()).hexdigest(),
                    'unchanged_array':True,'display_window':[0,limit],'extent_mm':extents[index]})
        for row,label in enumerate(('Projection','Center slice')):
            fig.text(.018,top-.087-row*.207,label,ha='center',va='center',rotation=90,fontsize=20,fontweight='normal')
    fig.text(.035,.985,'A  Aorta',ha='left',va='top',fontsize=27,fontweight='bold')
    fig.text(.98,.985,'1M spins  |  Bin 2/20',ha='right',va='top',fontsize=21,color='#333333')
    fig.text(.035,.505,'B  Static phantom',ha='left',va='top',fontsize=27,fontweight='bold')
    fig.text(.98,.505,'Bin 2',ha='right',va='top',fontsize=21,color='#333333')
    fig.text(.5,.018,'Grayscale: magnitude   |   Axes: mm   |   Original contrast retained',ha='center',va='center',fontsize=16,color='#333333')
    basename='Aorta_Phantom_Magnitude_Organized'
    # PNG is an export from the scientific plotting layout, not a generative
    # modification of the attached image. PDF/SVG retain editable labels.
    for ext in ('png','pdf','svg'):fig.savefig(stage/(basename+'.'+ext),dpi=300,facecolor='white')
    fig.savefig(stage/'preview.png',dpi=80,facecolor='white')
    plt.close(fig)
    caption=('KomaMRI magnitude images of the aorta (1M spins, bin 2/20) and historical static stenosis phantom (bin 2). '
        'Columns: XY, XZ and YZ. Within each case, the upper row shows maximum-intensity projections and the lower row shows center slices. '
        'Original image arrays and grayscale windows are retained. Coordinates are in millimetres; display scales differ between cases.')
    assert len(caption)<=500
    (stage/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    (stage/'README.md').write_text('# Organized magnitude figure\n\nSame original bin-2 panels, reorganized with matching case widths and concise labels. '
        'Aorta: completed E062, 1M spins, bin 2/20, actual ADC-window center 88.1913 ms. '
        'Phantom: historical STATIC dp150 reference, bin 2; it is not the new dp122 moving acquisition.\n\n'
        'Only layout and text changed. Scientific arrays, per-case display limits, physical extents and nearest-neighbor display are inherited from the original plots. '
        'No masks, added noise, smoothing, altered intensity data, new reconstruction or acquisition. '
        'Plot labels use simulation X/Y/Z axes, not certified anatomical directions. Source mapping X=source Z, Y=source X, Z=source Y.\n\n'
        'Use PNG for the abstract, PDF for print, SVG for label/layout editing. caption.txt is below 500 characters. '
        'Sources and display checks are in REPORT.json.\n',encoding='utf-8')
    with Image.open(stage/(basename+'.png')) as check:assert check.size==(5400,4500);check.verify()
    assert (stage/(basename+'.pdf')).read_bytes().startswith(b'%PDF-')
    for p,digest in expected.items():assert sha(p)==digest,'Source changed during plotting'
    report={'status':'PASS_SAME_PANEL_ARRAYS_AND_CONTRAST','created_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_class':'Presentation/layout derivative only','panels':checked,'input_sha256':{str(p):v for p,v in expected.items()},
        'source_helpers':{str(HELPER):sha(HELPER)},'script_sha256':sha(__file__),'original_panels':12,
        'new_metrics':False,'new_acquisition':False,'pixel_array_changes':False,'layout_and_text_changed':True,
        'caption_characters':len(caption),'visual_review':'Pending preview inspection'}
    (stage/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (stage/'MANIFEST.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in stage.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    stage.rename(OUT)
    print(json.dumps({'status':report['status'],'panels':12,'caption_characters':len(caption),'png':str(OUT/(basename+'.png')),'pdf':str(OUT/(basename+'.pdf'))},indent=2))
if __name__=='__main__':main()
