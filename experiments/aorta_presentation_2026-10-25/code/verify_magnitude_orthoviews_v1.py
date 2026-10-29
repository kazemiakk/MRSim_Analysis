"""Independent readback checks for the presentation-only E062 orthoview export."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import h5py
import numpy as np
from PIL import Image

ROOT=Path('D:/MRSim/outputs/aorta_presentation_2026-10-25/magnitude_orthoviews_1m_v2')
SOURCE=Path('D:/MRSim/outputs/aorta_20bin_e062/combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
checks=[]
def check(v,label):
    if not v:raise AssertionError(label)
    checks.append(label)
destination=ROOT/'INDEPENDENT_VALIDATION.json'
if destination.exists():raise FileExistsError(destination)
man=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
report=json.loads((ROOT/'REPORT.json').read_text(encoding='utf-8'))
for p,digest in man['input_sha256'].items():check(sha(p)==digest,'Input identity '+Path(p).name)
for name,record in man['outputs'].items():
    check((ROOT/name).stat().st_size==record['bytes'],'Export size '+name)
    check(sha(ROOT/name)==record['sha256'],'Export identity '+name)
with h5py.File(SOURCE,'r') as f:
    v=f['reference_magnitude'][1]
    stored=np.asarray(f.attrs['FOV_m'])
check(np.array_equal(stored,np.asarray([.28,.128,.128],dtype=stored.dtype)),
      'Exact nominal FOV at original metadata precision')
with np.load(ROOT/'displayed_arrays.npz',allow_pickle=False) as a:
    expected={'projection_xy':np.max(v,axis=0),'projection_xz':np.max(v,axis=1),
              'projection_yz':np.max(v,axis=2),'slice_xy':v[16,:,:],
              'slice_xz':v[:,16,:],'slice_yz':v[:,:,36]}
    for k,array in expected.items():check(np.array_equal(a[k],array),'Full source/display identity '+k)
    for key,n,fov in [('x_mm',72,280.),('y_mm',32,128.),('z_mm',32,128.)]:
        check(np.array_equal(a[key],(np.arange(n)-n//2)*(fov/1000)/n*1000),
              'Centered FFT coordinates '+key)
check(report['original_bin']==2 and report['total_bins']==20,'Bin labels')
check(len(report['checks'])==32,'Original rendering checks all recorded')
check(report['display']['mask'] is False and report['display']['smoothing'] is False
      and report['display']['added_noise'] is False,'No filtering or invented noise')
with Image.open(ROOT/'aorta_1m_bin02_magnitude_orthoviews.png') as image:
    check(image.size==(5280,2832),'PNG output dimensions')
    image.verify()
check((ROOT/'aorta_1m_bin02_magnitude_orthoviews.pdf').read_bytes().startswith(b'%PDF-'),'PDF signature')
out=dict(status='PASS_INDEPENDENT_ORTHOVIEW_READBACK',utc=datetime.now(timezone.utc).isoformat(),
         checks=len(checks),check_names=checks,source_unchanged=True,
         source_script_sha256=sha(__file__),
         visual_review='PNG inspected separately; PDF signature/hash checked, not independently rasterized',
         new_acquisition=False,new_registered_metric=False)
destination.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':len(checks)}))
