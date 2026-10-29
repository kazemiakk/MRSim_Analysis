from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
ROOT=Path('D:/MRSim/outputs/presentation_hemodynamics_2026-10-25/phase_views_v3')
for path in ROOT.glob('*.png'):
    with Image.open(path) as image: rgb=np.asarray(image.convert('RGB'))
    mask=(rgb<20).all(axis=2)
    labels,count=ndimage.label(mask)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    objects=ndimage.find_objects(labels)
    print(path.name,'shape',rgb.shape)
    for idx in np.argsort(sizes)[-6:]:
        box=objects[idx-1]; y,x=box
        print('black plot component',int(sizes[idx]),[x.start,y.start,x.stop,y.stop])
