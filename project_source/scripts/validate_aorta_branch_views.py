"""Independent E-043 source-value/export checks; no physical-validation claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.complete_advanced_comparisons import load_inputs
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.extend_hemodynamic_comparisons import sha
from scripts.compare_aorta_hemodynamics import dump,DATA


def validate(run):
    report=json.loads((run/'report.json').read_text());manifest=json.loads((run/'MANIFEST.json').read_text())
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    def equal(a,b,name):check(np.allclose(a,b,rtol=0,atol=0,equal_nan=True),name)
    for row in report['inputs']:
        p=Path(row['path'])
        if not p.is_absolute():p=Path(__file__).resolve().parents[1]/p
        check(sha(p)==row['sha256'],'Input hash '+str(p))
    for name,row in manifest.items():
        p=run/name;check(p.stat().st_size==row['bytes'],'Bytes '+name);check(sha(p)==row['sha256'],'Output hash '+name)
        if p.suffix in ['.png','.pdf']:check(p.read_bytes().startswith(b'\x89PNG' if p.suffix=='.png' else b'%PDF'),'Signature '+name)
    _,cases,*_=load_inputs();cases={c['name']:c for c in cases};values=np.load(run/'plotted_values.npz',allow_pickle=False);grid=AortaGrid()
    check(len(report['inventory'])==16,'Sixteen PNG/PDF pairs')
    for r in report['records']:
        c=cases[r['case']];b=r['bin']-1;stem=r['stem'];a=c['reference'][b];k=c['estimate'][b]
        mask=c['regions']['whole']&np.isfinite(a).all(-1)&np.isfinite(k).all(-1)
        fields=[np.sqrt((a*a).sum(-1)),np.sqrt((k*k).sum(-1)),np.sqrt(((k-a)**2).sum(-1))]
        for i,f in enumerate(fields):check(np.max(f[mask])<=report['limits'][int(i==2)],stem+' no clipping '+str(i))
        if r['kind']=='whole_3D':
            zz,yy,xx=np.where(mask);axes=grid.axes()
            mri=np.column_stack([axes[0][xx],axes[1][yy],axes[2][zz]])
            source=np.empty_like(mri);source[:,grid.order]=mri;source+=grid.center
            equal(values[stem+'_source_xyz_mm'],source*1000,stem+' source inverse transform')
            for key,f in zip(['cfd','koma','error'],fields):equal(values[stem+'_'+key],f[mask],stem+' direct '+key)
        elif r['kind']=='same_depth_projection':
            expected_mask=mask.any(0).T;equal(values[stem+'_valid'],expected_mask,stem+' valid')
            d=np.argmax(np.where(mask,fields[0],-np.inf),axis=0);equal(values[stem+'_depth_index'],d.T,stem+' CFD depth selector')
            yy,xx=np.indices(d.shape)
            for key,f in zip(['cfd','koma','error'],fields):
                selected=np.where(mask.any(0),f[d,yy,xx],np.nan).T
                equal(values[stem+'_'+key],selected,stem+' shared-depth '+key)
        elif r['kind']=='true_slice':
            prefix=f"{stem}_{r['row']}";i=r['index']
            equal(values[prefix+'_valid'],mask[i].T,prefix+' exact plane support')
            for key,f in zip(['cfd','koma','error'],fields):equal(values[prefix+'_'+key],np.where(mask,f,np.nan)[i].T,prefix+' exact plane '+key)
            check(abs(grid.axes()[2][i]*1000+grid.center[1]*1000-r['source_y_mm'])<1e-10,prefix+' original Y coordinate')
    wall_path=next(Path(r['path']) for r in report['inputs'] if r['path'].endswith('aorta_wall_points_from_wss.npy'))
    if not wall_path.is_absolute():wall_path=Path(__file__).resolve().parents[1]/wall_path
    wall=np.load(wall_path,allow_pickle=False);native=np.load(DATA/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    distances,_=cKDTree(native).query(wall,workers=-1);check(distances.max()==0,'Every native wall context sample matches source exactly')
    from scipy.cluster.vq import kmeans2
    mm=wall.astype(float)*1000;top=mm[mm[:,2]>90]
    initial=np.array([top[np.argmin(top[:,0]),:2],np.median(top[:,:2],axis=0),top[np.argmax(top[:,0]),:2]])
    centers,groups=kmeans2(top[:,:2],initial,iter=50,minit='matrix',missing='raise')
    order=np.argsort(centers[:,0]);ys=grid.axes()[2]*1000+grid.center[1]*1000
    score=np.array([[np.count_nonzero(abs(top[groups==g,1]-y)<=2)/np.count_nonzero(groups==g) for g in order] for y in ys])
    selected=report['best_all_branch_plane'];i=selected['index']
    equal(np.array(selected['fractions']),score[i],'Selected voxel slab native-tip fractions')
    check(i==int(np.argmax(score.min(axis=1))),'Geometry-only maximum-minimum selection')
    check(score[i].min()>0,'All three geometric tip regions intersect the selected finite voxel slab')
    return dict(experiment='E-043',status='PASS',passed=len(checks),checks=checks,scope='Source-array, transform and export verification, not independent physical validation; PDF signatures/hashes only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=validate(args.run);dump(args.receipt,result);print(f"PASS: {result['passed']} source-array/export checks")
