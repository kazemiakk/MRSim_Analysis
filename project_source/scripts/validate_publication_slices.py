"""Independent source-array and export checks for the E-042 figure bundle."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scripts.complete_advanced_comparisons import load_inputs
from scripts.extend_hemodynamic_comparisons import DP,AO,sha
from scripts.compare_aorta_hemodynamics import dump


def validate(run,receipt):
    report=json.loads((run/'report.json').read_text());manifest=json.loads((run/'MANIFEST.json').read_text())
    checks=[]
    def check(label,condition):
        if not condition:raise AssertionError(label)
        checks.append(label)
    for name,meta in manifest.items():
        path=run/name;check('Output hash '+name,sha(path)==meta['sha256']);check('Bytes '+name,path.stat().st_size==meta['bytes'])
        if path.suffix=='.png':check('PNG signature '+name,path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        if path.suffix=='.pdf':check('PDF signature '+name,path.read_bytes().startswith(b'%PDF-'))
    for meta in report['inputs']:
        p=Path(meta['path']);p=p if p.is_absolute() else Path(__file__).resolve().parents[1]/p
        check('Input hash '+str(p),sha(p)==meta['sha256'])
    _,cases,*_=load_inputs();cases={c['name']:c for c in cases}
    check('18 PNG/PDF pairs',len(report['inventory'])==18 and len(list(run.glob('*.png')))==18 and len(list(run.glob('*.pdf')))==18)
    samples=np.load(run/'displayed_samples.npz',allow_pickle=False)
    for r in report['records']:
        case=cases[r['case']];b=r['bin']-1;key=r['key'];mask=case['regions']['whole']
        a=case['reference'][b];c=case['estimate'][b];valid=mask&np.isfinite(a).all(-1)&np.isfinite(c).all(-1)
        fields=[np.sqrt(np.sum(a*a,-1)),np.sqrt(np.sum(c*c,-1)),np.sqrt(np.sum((c-a)**2,-1))]
        coords=[(np.arange(n)-n/2+(.5 if r['case']=='phantom_1m' else 0))*d*1000 for n,d in zip(case['shape'][::-1],case['spacing'])]
        if r.get('kind')=='3D':
            z,y,x=np.where(valid);points=np.stack([coords[0][x],coords[1][y],coords[2][z]],-1)
            check(key+' points',np.array_equal(samples[key+'_points'],points))
            for name,f in zip(['cfd','koma','error'],fields):check(key+' values '+name,np.allclose(samples[key+'_'+name],f[valid],rtol=1e-14,atol=0))
            check(key+' complete paired count',r['points']==int(valid.sum()))
        else:
            axis={'XY':0,'XZ':1,'YZ':2}[r['plane']];sl=[slice(None)]*3;sl[axis]=r['index'];sl=tuple(sl)
            check(key+' fixed mask',np.array_equal(samples[key+'_mask'],valid[sl]))
            check(key+' position',r['position_mm']==coords[2-axis][r['index']])
            for name,f in zip(['cfd','koma','error'],fields):
                expected=np.where(valid[sl],f[sl],np.nan)
                check(key+' source samples '+name,np.allclose(samples[key+'_'+name],expected,equal_nan=True,rtol=1e-14,atol=0))
            check(key+' pixel count',r['valid_pixels']==int(valid[sl].sum()))
            with h5py.File((DP if r['case']=='phantom_1m' else AO)/'matched_reference.h5') as f:
                geometry=(f['volume_fraction_2' if r['case']=='phantom_1m' else 'fraction'][:]>0)[sl]
            check(key+' geometry mask',np.array_equal(samples[key+'_geometry'],geometry))
        check(key+' speed no saturation',max(float(f[valid].max()) for f in fields[:2])<=r['speed_limit'])
        check(key+' error no saturation',float(fields[2][valid].max())<=r['error_limit'])
    for name,case in report['cases'].items():
        if name=='phantom_1m':check('No synthetic-tail selected bin',max(case['bins'].values())<=13)
    dump(receipt,dict(experiment='E-042',status='PASS',passed=len(checks),checks=checks,scope='All saved 2D/3D samples, masks, coordinate centers, counts, no-saturation bounds, source/output hashes and PNG/PDF signatures. No physical/anatomical validation; PDF content not visually rendered.'))
    print(f'PASS: {len(checks)} source-array/export checks')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();validate(a.run,a.receipt)
