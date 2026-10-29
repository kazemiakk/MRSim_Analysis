"""Presentation-only six-panel magnitude figure from the completed E062 1M run.

No acquisition, reference recomputation, masks, noise, imputation or new metric.
Preserve this source after execution; revisions require attempt source + v2.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RECON=Path('D:/MRSim/outputs/aorta_20bin_e062/combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5')
AUDIT=Path('D:/MRSim/outputs/aorta_20bin_e062/sequence_audit_v1/report.json')
GRID=Path('D:/MRSim/MRSIM/MRSimTracks/scripts/aorta_hemodynamic_reference.py')
EXPECTED_RECON='7b3072819a723508e5bd8b753dffb76680ea29fee33800123e32f6c5682a44a8'
EXPECTED_AUDIT='04548d001bca5bc54f29c205d645ecf68ba84ea0af9aa47beeef5c43c17b8b29'
PHANTOM='d23241331c8ec57e0e0396ef5468e1902583e7d2095c6ece83e3c71854242720'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    final=args.output.resolve()
    stage=final.with_name(final.name+'.building')
    if final.exists() or stage.exists():raise FileExistsError(final)
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    check(sha(RECON)==EXPECTED_RECON,'E062 reconstruction identity')
    check(sha(AUDIT)==EXPECTED_AUDIT,'E062 actual ADC timing identity')
    with h5py.File(RECON,'r') as f:
        mag=f['reference_magnitude'][:]
        fov=np.asarray(f.attrs['FOV_m'],dtype=float)*1000
        matrix=np.asarray(f.attrs['matrix'],dtype=int)
        phantom=f.attrs['source_phantom_sha256']
        if isinstance(phantom,bytes):phantom=phantom.decode()
    check(str(phantom)==PHANTOM,'Preserved one-million-spin source phantom')
    check(mag.shape==(20,32,32,72),'Twenty-bin saved magnitude dimension order')
    check(tuple(matrix)==(72,32,32),'Native matrix readout-phase-slice')
    check(np.allclose(fov,[280,128,128],atol=1e-8,rtol=0),'Physical field of view')
    check(np.isfinite(mag).all() and (mag>=0).all(),'Magnitude finite and nonnegative')
    # Frozen E027/E062 AortaGrid: centered inverse FFT samples, offset=0.0.
    # This is NOT the old display helper's half-voxel shifted coordinate rule.
    from scripts.aorta_hemodynamic_reference import AortaGrid
    grid=AortaGrid()
    x,y,z=[a*1000 for a in grid.axes()]
    spacing=grid.spacing*1000
    check(tuple(grid.shape)==mag.shape[1:],'Authoritative centered-grid shape')
    check(np.allclose(np.asarray(grid.fov)*1000,fov,atol=1e-8,rtol=0),'Grid and reconstruction FOV agree')
    i=1  # Original cardiac bin2, not relabeled after a phase shift.
    vol=mag[i]
    iz,iy,ix=(n//2 for n in vol.shape)
    check((z[iz],y[iy],x[ix])==(0,0,0),'True central planes are zero-coordinate planes')
    arrays=dict(projection_xy=vol.max(axis=0),projection_xz=vol.max(axis=1),
                projection_yz=vol.max(axis=2),slice_xy=vol[iz],
                slice_xz=vol[:,iy,:],slice_yz=vol[:,:,ix])
    # Independent incremental reductions confirm each displayed MIP pixel.
    for axis,key in [(0,'projection_xy'),(1,'projection_xz'),(2,'projection_yz')]:
        reduced=np.take(vol,0,axis=axis).copy()
        for j in range(1,vol.shape[axis]):
            np.maximum(reduced,np.take(vol,j,axis=axis),out=reduced)
        check(np.array_equal(reduced,arrays[key]),'Independent maximum projection '+key)
    positive=mag[mag>0]
    vmax=float(np.quantile(positive,0.995))
    check(np.isfinite(vmax) and vmax>0,'Shared all-20-bin grayscale display window')
    def bounds(a,d):return [float(a[0]-d/2),float(a[-1]+d/2)]
    bx,by,bz=bounds(x,spacing[0]),bounds(y,spacing[1]),bounds(z,spacing[2])
    specs=[('Projection - XY','projection_xy','X (mm)','Y (mm)',bx+by),
           ('Projection - XZ','projection_xz','X (mm)','Z (mm)',bx+bz),
           ('Projection - YZ','projection_yz','Y (mm)','Z (mm)',by+bz),
           ('Center slice - XY (Z = 0 mm)','slice_xy','X (mm)','Y (mm)',bx+by),
           ('Center slice - XZ (Y = 0 mm)','slice_xz','X (mm)','Z (mm)',bx+bz),
           ('Center slice - YZ (X = 0 mm)','slice_yz','Y (mm)','Z (mm)',by+bz)]
    audit=json.loads(AUDIT.read_text(encoding='utf-8'))
    timing=[r for r in audit['encodings']['REF']['bin_timing'] if r['bin']==2]
    check(len(timing)==1,'Actual timing of original cardiac bin2 exists')
    phase_ms=float(timing[0]['phase_circular_mean_s'])*1000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,
        'axes.titlesize':18,'axes.labelsize':15,'xtick.labelsize':13.5,'ytick.labelsize':13.5,
        'pdf.fonttype':42,'ps.fonttype':42})
    stage.mkdir(parents=True)
    fig,axs=plt.subplots(2,3,figsize=(22,11.8),gridspec_kw={'width_ratios':[1.5,1.5,1.]},
                         layout='constrained')
    fig.get_layout_engine().set(rect=(0,0.07,1,0.88),hspace=0.10,wspace=0.04)
    for ax,(title,key,xlabel,ylabel,extent) in zip(axs.ravel(),specs):
        artist=ax.imshow(arrays[key],cmap='gray',origin='lower',extent=extent,
                         vmin=0,vmax=vmax,interpolation='nearest',aspect='equal')
        ax.set_title(title,pad=12)
        ax.set_xlabel(xlabel,labelpad=7);ax.set_ylabel(ylabel,labelpad=7)
        ax.tick_params(length=4,width=.8)
        ax.grid(False)
        check(np.array_equal(np.asarray(artist.get_array()),arrays[key]),'Unchanged displayed data '+key)
        check(artist.get_clim()==(0,vmax),'Identical grayscale window '+key)
    fig.suptitle('Aorta 1M-spin KomaMRI magnitude | cardiac bin 2 / 20',fontsize=25,y=.995)
    fig.text(.5,.945,f'REF reconstruction | actual ADC-window center {phase_ms:.1f} ms',
             ha='center',va='center',fontsize=17,color='#333333')
    fig.text(.5,.035,'Simulation axes: X = source Z, Y = source X, Z = source Y. '
             'Native voxels: 3.89 x 4 x 4 mm. No added noise, smoothing, or display mask.',
             ha='center',va='center',fontsize=13,color='#333333')
    name='aorta_1m_bin02_magnitude_orthoviews'
    png=stage/(name+'.png');pdf=stage/(name+'.pdf')
    fig.savefig(png,dpi=240,facecolor='white')
    fig.savefig(pdf,facecolor='white')
    plt.close(fig)
    np.savez_compressed(stage/'displayed_arrays.npz',**arrays,x_mm=x,y_mm=y,z_mm=z)
    with Image.open(png) as image:
        check(image.size==(5280,2832),'PNG dimensions')
        image.verify()
    check(pdf.read_bytes().startswith(b'%PDF-'),'PDF file signature')
    check(sha(RECON)==EXPECTED_RECON,'Source reconstruction unchanged after rendering')
    check(sha(AUDIT)==EXPECTED_AUDIT,'Source timing unchanged after rendering')
    report=dict(status='PASS_MAGNITUDE_PRESENTATION_CHECKS',created_utc=datetime.now(timezone.utc).isoformat(),
        evidence_class='Presentation derivative of completed E062; no new scientific metric or acquisition',
        original_bin=2,total_bins=20,actual_adc_center_ms=phase_ms,spins=1000000,
        magnitude_dataset='reference_magnitude',volume_axis_order='bin,sliceZ,phaseY,readoutX',
        coordinate_rule='E027/E062 AortaGrid centered-FFT coordinates, offset0; imshow edges half a voxel beyond centers',
        spacing_mm=spacing.tolist(),slice_indices_zyx=[iz,iy,ix],
        display=dict(cmap='gray',vmin=0,vmax=vmax,
                     window='99.5th percentile of positive reference magnitude across all20 saved bins',
                     interpolation='nearest',mask=False,smoothing=False,added_noise=False),
        checks=checks,new_acquisition=False,new_registered_metric=False,
        limitation='Approximate aortic point-cloud phantom; projections are maximum-intensity depth projections, lower row contains true center planes; no claim of scanner equivalence or independent physical validation.')
    (stage/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (stage/'README.md').write_text(
        '# Aorta 1M-spin magnitude orthoviews\n\n'
        'Source: completed E06220-bin reconstruction; original cardiac bin2.\n'
        f'Actual ADC-window center: {phase_ms:.6f}ms in the native simulator cycle.\n\n'
        'Top: XY/XZ/YZ maximum-intensity projections. Bottom: true central planes.\n'
        'Only reference magnitude is shown, with the same fixed grayscale limits\n'
        'on all panels. No display mask, interpolation/smoothing, phase filling or\n'
        'added noise. Nearest-neighbor rendering preserves the native voxel grid.\n'
        'The upper display limit is a common percentile window, not a data correction.\n\n'
        'Native3.89x4x4-mm resolution is coarser than the supplied phantom example.\n'
        'Coordinate axes are the documented MRI simulation transform, not certified anatomy.\n'
        'No new acquisition or registered metric; frozen inputs and repositories unchanged.\n',encoding='utf-8')
    manifest=dict(input_sha256={str(RECON):EXPECTED_RECON,str(AUDIT):EXPECTED_AUDIT,
                  str(GRID):sha(GRID),str(Path(__file__).resolve()):sha(__file__)},
                  outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
                           for p in stage.iterdir() if p.is_file()})
    (stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    stage.rename(final)
    print(json.dumps(dict(status=report['status'],checks=len(checks),png=str(final/png.name),
                         pdf=str(final/pdf.name),actual_adc_center_ms=phase_ms)))


if __name__=='__main__':main()
