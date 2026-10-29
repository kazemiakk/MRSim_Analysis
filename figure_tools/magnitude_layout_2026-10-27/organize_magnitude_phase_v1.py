"""Assemble original magnitude panels and previously generated phase displays.

This is scientific figure layout, not new phase reconstruction or acquisition.
Phase RGB panels are extracted from the exact hash-bound phase_views_v3 PNGs.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, importlib.util
import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path('D:/MRSim/outputs/magnitude_layout_2026-10-27')
OUT=ROOT/'magnitude_phase_v1'
SOURCE_HELPER=ROOT/'code/organize_magnitude_v4.py'
PHASE_ROOT=Path('D:/MRSim/outputs/presentation_hemodynamics_2026-10-25/phase_views_v3')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def source_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def panel_from_png(rgb,box):
    """Remove only tick extensions outside the saved black plotting rectangle."""
    x0,y0,x1,y1=box;cut=rgb[y0:y1,x0:x1]
    black=(cut<20).all(axis=2)
    columns=np.flatnonzero(black.sum(axis=0)>.5*cut.shape[0])
    rows=np.flatnonzero(black.sum(axis=1)>.25*cut.shape[1])
    assert len(columns)>0 and len(rows)>0
    a,b,c,d=columns[0],rows[0],columns[-1]+1,rows[-1]+1
    region=[int(x0+a),int(y0+b),int(x0+c),int(y0+d)]
    image=rgb[region[1]:region[3],region[0]:region[2]].copy()
    assert image.shape[0]>200 and image.shape[1]>200
    return image,region

def main():
    if OUT.exists() or OUT.with_name(OUT.name+'.building').exists():raise FileExistsError(OUT)
    original=source_module(SOURCE_HELPER,'original_magnitude_layout')
    expected={original.AORTA/'displayed_arrays.npz':'e5a7169a5ad1e63bdc224c0e34726f948475530b18700e9bcc5f45de3b091d0c',
        original.PHANTOM:'079712cfc44ef1909cf80eae798f4bdd41cc61a435a376288fd36ae460ec913b',
        PHASE_ROOT/'aorta_e062_phase_views.png':'4e88f0a80c7a02fe5333888a8cfe2c167525cf97f70226bbf55a2beb19e5b78f',
        PHASE_ROOT/'phantom_e060_phase_views.png':'3f9c0ad640f8042ea48a77a1f437c6e723be249df0224ae9ed035cb5c2d9ea20'}
    for p,digest in expected.items():assert sha(p)==digest,str(p)
    with np.load(original.AORTA/'displayed_arrays.npz',allow_pickle=False) as saved:
        aorta=[saved[key].copy() for key in original.KEYS[:3]]
        axes=[saved[key].copy() for key in ('x_mm','y_mm','z_mm')]
    edges=[]
    for axis in axes:
        delta=axis[1]-axis[0];edges.append([float(axis[0]-delta/2),float(axis[-1]+delta/2)])
    bx,by,bz=edges
    mag_extents={'aorta':[bx+by,bx+bz,by+bz]}
    limits={'aorta':json.loads((original.AORTA/'REPORT.json').read_text(encoding='utf-8'))['display']['vmax']}
    phantom_helper=source_module(original.HELPER,'original_static_panel_helper')
    with h5py.File(original.PHANTOM,'r') as f:volume=np.asarray(f['reference_magnitude'][1],dtype=np.float32)
    p_specs=phantom_helper.panel_data(volume)
    magnitudes={'aorta':aorta,'phantom':[p[1].copy() for p in p_specs[:3]]}
    mag_extents['phantom']=[p[4] for p in p_specs[:3]];limits['phantom']=phantom_helper.robust_limit(volume)

    # Use only the previously generated display panels requested by the user.
    # Aorta: first row, XY view; phantom: second row, original longitudinal YZ view.
    boxes={'aorta':[(73,240,678,515),(772,240,1377,515),(1471,240,2076,515)],
        'phantom':[(73,805,689,1373),(771,805,1387,1373),(1470,805,2086,1373)]}
    phase_paths={'aorta':PHASE_ROOT/'aorta_e062_phase_views.png','phantom':PHASE_ROOT/'phantom_e060_phase_views.png'}
    phases={};phase_extents={};phase_meta=[]
    for case in ('aorta','phantom'):
        with Image.open(phase_paths[case]) as f:rgb=np.asarray(f.convert('RGB'))
        phases[case]=[];phase_extents[case]=[]
        for component,box in zip(('X','Y','Z'),boxes[case]):
            data,region=panel_from_png(rgb,box);source_crop=data.copy()
            operation='Same saved XY phase panel, no spatial transform'
            if case=='phantom':
                # Original phase figure is P horizontal / S vertical. E060's
                # proper rotation maps (R,P,S) to original CFD (X,Y,Z).
                # Focus P to +/-16 mm; ensure no displayed vessel signal is lost.
                n=data.shape[1];p0,p1=-40.,38.75
                j0=int(np.ceil((-16.-p0)/(p1-p0)*n));j1=int(np.floor((16.-p0)/(p1-p0)*n))
                outside=np.concatenate([data[3:-3,3:j0],data[3:-3,j1:-3]],axis=1)
                assert (outside<=30).all(),'Transverse crop would exclude displayed phase signal'
                window=data[:,j0:j1].copy()
                # Rotate the VIEW to put CFD Z left-to-right and CFD Y upwards.
                data=window[::-1,::-1].transpose(1,0,2).copy()
                assert np.array_equal(data[::-1,::-1].transpose(1,0,2),window)
                low_p=p0+j0/n*(p1-p0);high_p=p0+j1/n*(p1-p0)
                extent=[-37.5,35.,low_p,high_p]
                operation='Retained all nonblack vessel signal; P display window +/-16 mm; transpose and reverse both raster axes to show CFD Z horizontal, Y vertical'
            else:
                extent=[-140.,(72-1-36)*280/72,-64.,60.]
            phases[case].append(data);phase_extents[case].append(extent)
            phase_meta.append({'case':case,'component':component,'source_png':str(phase_paths[case]),
                'source_pixel_box':region,'source_crop_rgb_sha256':hashlib.sha256(source_crop.tobytes()).hexdigest(),
                'display_rgb_sha256':hashlib.sha256(data.tobytes()).hexdigest(),'operation':operation,
                'extent_mm':extent,'phase_values_not_recomputed':True})
    rotation=json.loads(Path('D:/MRSim/outputs/scanner_geometry_e060/phantom_v1/report.json').read_text(encoding='utf-8'))
    assert rotation['transform']['definition']=='(x,y,z)_new = (y, z, x - x_throat)'
    stage=OUT.with_name(OUT.name+'.building');stage.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.labelsize':18,
        'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(18,16),facecolor='white')
    mag_columns=[(.08,.34),(.455,.34),(.83,.15)]
    phase_columns=[(.08,.26),(.38,.26),(.68,.26)]
    magnitude_checks=[]
    settings=[('aorta',.755,.535,.16),('phantom',.252,.06,.14)]
    for case,ymag,yphase,hphase in settings:
        titles=('XY','XZ','YZ') if case=='aorta' else ('XZ','YZ','XY')
        vertical=('Y','Z','Z') if case=='aorta' else ('X','Y','Y')
        for c,(x,w) in enumerate(mag_columns):
            ax=fig.add_axes([x,ymag,w,.16])
            artist=ax.imshow(magnitudes[case][c],origin='lower',extent=mag_extents[case][c],
                cmap='gray',vmin=0,vmax=limits[case],interpolation='nearest',aspect='equal')
            fig.text(x+w/2,ymag+.172,titles[c],ha='center',va='bottom',fontsize=22)
            ax.set_ylabel(vertical[c],labelpad=4);ax.tick_params(length=3,width=.7,pad=4)
            assert np.array_equal(np.asarray(artist.get_array()),magnitudes[case][c])
            assert artist.get_clim()==(0,limits[case])
            magnitude_checks.append({'case':case,'view':titles[c],
                'array_sha256':hashlib.sha256(magnitudes[case][c].tobytes()).hexdigest(),'original_array_and_window':True})
        for c,(x,w) in enumerate(phase_columns):
            ax=fig.add_axes([x,yphase,w,hphase]);data=phases[case][c]
            artist=ax.imshow(data,origin='upper',extent=phase_extents[case][c],interpolation='nearest',aspect='equal')
            fig.text(x+w/2,yphase+hphase+.012,'Phase '+('X','Y','Z')[c],ha='center',va='bottom',fontsize=21)
            ax.set_xlabel('X' if case=='aorta' else 'Z',labelpad=4);ax.set_ylabel('Y',labelpad=4)
            ax.tick_params(length=3,width=.7,pad=4);ax.set_facecolor('black')
            assert np.array_equal(np.asarray(artist.get_array()),data)
        for center,label in [(ymag+.08,'Magnitude'),(yphase+hphase/2,'Phase')]:
            fig.text(.018,center,label,ha='center',va='center',rotation=90,fontsize=20)
        cm=LinearSegmentedColormap.from_list('original_phase_gray',['#3a3a3a','#ffffff'])
        bar=fig.colorbar(ScalarMappable(norm=Normalize(-np.pi,np.pi),cmap=cm),
            cax=fig.add_axes([.958,yphase+.02,.009,hphase-.04]),ticks=[-np.pi,0,np.pi])
        bar.ax.set_yticklabels(['−π','0','π']);bar.ax.tick_params(labelsize=14)
        bar.ax.set_title('rad',fontsize=14,pad=6)
    fig.text(.035,.985,'A  Aorta',ha='left',va='top',fontsize=27,fontweight='bold')
    fig.text(.98,.985,'1M spins  |  Magnitude: bin 2/20  |  Phase: bin 20/20',ha='right',va='top',fontsize=18,color='#333333')
    fig.text(.5,.49,'Phase display: SNR 12, VENC 0.3 m/s (acquired at 2.5 m/s)',ha='center',va='center',fontsize=16,color='#333333')
    fig.text(.035,.472,'B  Phantom',ha='left',va='top',fontsize=27,fontweight='bold')
    fig.text(.98,.472,'Magnitude: static bin 2  |  Phase: E-060 moving bin 8/30, SNR 84',ha='right',va='top',fontsize=17,color='#333333')
    fig.text(.5,.008,'Coordinates: mm   |   Phase: black = excluded signal   |   Display noise only',ha='center',va='center',fontsize=15,color='#333333')
    basename='Aorta_Phantom_Magnitude_Phase'
    for ext in ('png','pdf','svg'):fig.savefig(stage/(basename+'.'+ext),dpi=300,facecolor='white')
    fig.savefig(stage/'preview.png',dpi=80,facecolor='white');plt.close(fig)
    caption=('Magnitude projections (upper rows) and X/Y/Z phase displays (lower rows). '
        'Aorta: magnitude bin 2/20, phase bin 20/20; phase display SNR 12/VENC 0.3 m/s (acquired 2.5). '
        'Phantom: static magnitude bin 2; E060 moving phase bin 8/30, display SNR 84. '
        'Phase views: aorta XY, phantom YZ with Z along the vessel. Saved phase masks/interpolation retained; black is excluded. Rows show different snapshots.')
    assert len(caption)<=500
    (stage/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    (stage/'README.md').write_text('# Magnitude and saved phase displays\n\n'
        'User-requested replacement of the center-slice rows by three previously generated grayscale phase images. '
        'The six magnitude-projection data arrays and their original contrast are unchanged from layout_v4.\n\n'
        'PHASE SOURCES: Exact saved phase_views_v3 PNGs, with hashes checked. Aorta uses their top row (XY depth projection), '
        'E062 bin20/20. Phantom uses their bottom row (longitudinal projection), E060 bin8/30, transformed only in view orientation '
        'to place original CFD Z horizontally and Y vertically. E060 R/P/S encodings map to original CFD X/Y/Z under the verified proper rotation. '
        'Black-only transverse margins are removed to focus Y to approximately +/-16 mm; a check confirms no displayed nonblack vessel signal was removed. '
        'Phase angles were NOT recomputed or converted into new quantitative measurements.\n\n'
        'PRESENTATION LIMITS: Original phase displays already use 4x cubic interpolation of complex real/imaginary data, '
        'a per-view 12%-of-p99.5 amplitude signal mask, and bilinear raster display. These inherited displays are not raw native phase samples. '
        'Aorta phases were constructed as a display-noise realization with SNR12 and display VENC0.3 m/s, whereas the actual acquisition VENC was2.5 m/s. '
        'Phantom phases retain the earlier E060 display-noise realization at scan-estimated SNR~84, VENC1.5/1.5/3.0 m/s. '
        'No private scan file or CFD point-cloud array was read to assemble this figure.\n\n'
        'The top/bottom rows use different bins. Phantom magnitude is a historical STATIC dp150 reference; phase is MOVING E060 dp150, '
        'not the latest dp122 E065 acquisition. This provenance is explicitly labeled and must not be described as same-frame magnitude/phase or E066 results. '
        'Aortic geometry labels remain MRI simulation coordinates; phantom labels use original CFD axes. '
        'PNG/PDF/SVG and preview are presentation derivatives only. No simulation, gate change or new registered metric.\n',encoding='utf-8')
    for p,digest in expected.items():assert sha(p)==digest,'Source changed during assembly'
    with Image.open(stage/(basename+'.png')) as f:assert f.size==(5400,4800);f.verify()
    assert (stage/(basename+'.pdf')).read_bytes().startswith(b'%PDF-')
    report={'status':'PASS_ORIGINAL_MAGNITUDE_AND_SAVED_PHASE_DISPLAY_IDENTITY','created_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'input_sha256':{str(p):v for p,v in expected.items()},
        'magnitude_panels':magnitude_checks,'phase_panels':phase_meta,'phase_source_bins':{'aorta':20,'phantom':8},
        'magnitude_bins':{'aorta':2,'phantom_static':2},'phase_colors_retained':True,'phase_values_recomputed':False,
        'source_rotation_report_sha256':sha('D:/MRSim/outputs/scanner_geometry_e060/phantom_v1/report.json'),
        'new_metrics':False,'new_acquisition':False,'caption_characters':len(caption),'visual_review':'Pending preview inspection'}
    (stage/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (stage/'MANIFEST.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in stage.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    stage.rename(OUT)
    print(json.dumps({'status':report['status'],'panels':12,'caption_characters':len(caption),'png':str(OUT/(basename+'.png'))},indent=2))
if __name__=='__main__':main()
