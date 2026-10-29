"""Check the actual display mapping and volume renderer before full movie encoding."""
from pathlib import Path
import numpy as np,pyvista as pv
out=Path(r'D:\FluentAgent\results\20261029-virtual-angiography-right-left-v2')
folder=out/'visualization';m=np.load(folder/'voxel-native-cell-map.npz');ids=m['cell'];inside=ids>=0
path=out/'fields/C-t0095000us.npz'
g=pv.ImageData(dimensions=m['dimensions'],spacing=[float(m['spacing'])]*3,origin=m['origin'])
surface=pv.read(folder/'original-fluid-boundary.vtp');c=np.load(path)['right'];values=np.zeros(len(ids),np.float32);values[inside]=c[ids[inside]]
g.point_data['Dye_C']=values
p=pv.Plotter(off_screen=True,window_size=(880,1040));p.set_background('white')
p.add_mesh(surface,color='#64748b',opacity=.055)
opacity=(np.sqrt(np.linspace(0,1,256))*.4*255).astype(np.uint8);opacity[:3]=0
p.add_volume(g,scalars='Dye_C',clim=[0,1],cmap='Oranges',opacity=opacity,shade=False,
    scalar_bar_args={'title':'Virtual dye C','vertical':False,'width':.75,'height':.1,'position_x':.13,'position_y':.02,'n_labels':3,
                     'title_font_size':20,'label_font_size':18})
center=np.asarray(surface.center);p.camera_position=[center+np.array([45,-120,25]),center,[0,0,1]]
p.enable_parallel_projection();p.camera.parallel_scale=32
p.add_text('Right ICA pulse: t = 90 ms',font_size=16,color='#334155')
p.screenshot(folder/'initial-render-preview-corrected.png');p.close()
print('VOLUME_PREVIEW_READY',flush=True)
