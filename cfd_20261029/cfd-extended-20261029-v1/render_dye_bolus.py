"""Source-backed volume rendering and line-integral virtual angiography videos."""
from pathlib import Path
import argparse,csv,json,os,sys,subprocess,time
import numpy as np,pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap,PowerNorm
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import PROJECT,load,save,CENTER_MM,RADII_MM

GRID=PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu'
FFMPEG=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')


def prepare(out):
    folder=out/'visualization';folder.mkdir(exist_ok=False)
    grid=pv.read(GRID);assert np.array_equal(grid['FluentCellIndex0'],np.arange(grid.n_cells))
    b=np.asarray(grid.bounds).reshape(3,2);spacing=.4;origin=b[:,0]-spacing
    dims=np.ceil((b[:,1]-origin)/spacing).astype(int)+2
    image=pv.ImageData(dimensions=dims,spacing=[spacing]*3,origin=origin)
    points=image.points;ids=np.empty(len(points),np.int32)
    for start in range(0,len(points),150000):
        end=min(start+150000,len(points));ids[start:end]=grid.find_containing_cell(points[start:end])
        print('VOXEL_MAP',end,'/',len(points),flush=True)
    np.savez_compressed(folder/'voxel-native-cell-map.npz',cell=ids,dimensions=dims,origin=origin,spacing=spacing)
    surface=grid.extract_surface(algorithm='dataset_surface');surface.save(folder/'original-fluid-boundary.vtp')
    save(folder/'sampling-definition.json',{'method':'Centre-sampled original tetrahedral cell concentration on display grid',
        'spacing_mm':spacing,'dimensions':dims.tolist(),'inside_voxel_samples':int((ids>=0).sum()),
        'projection':'Sum C times 0.4 mm along model-coordinate Y; grayscale line integral in mm, not calibrated x-ray attenuation',
        'data_not_changed':'Visualization sampling only; scalar solve remains on all 1,169,992 original cells',
        'source_grid':str(GRID),'geometry_modified':False})


def render(out):
    folder=out/'visualization';result=load(out/'result.json')
    assert result['status']=='available_first_cycle_transport_complete', 'No full movie from failed/partial transport'
    assert result['original_sources_unchanged'] and result['error'] is None
    frames=load(out/'frames.json');times=np.asarray([x['time_s'] for x in frames]);protocol=load(out/'protocol.json')
    definition_path=folder/'sampling-definition.json'
    definition=load(definition_path)
    original_definition=folder/'sampling-definition-native-reference.json'
    assert not original_definition.exists()
    definition_path.rename(original_definition)
    definition.update(projection='Sum C times display spacing along model-coordinate Y; not registered clinical AP or calibrated x-ray attenuation',
        native_flow_cells=protocol['native_flow_cells'],scalar_transport_cells=protocol['scalar_transport_cells'],
        data_not_changed='Display sampling only; original flow domain preserved. Scalar concentration is reconstructed from connected agglomerates onto native cells.')
    save(definition_path,definition)
    rows=list(csv.DictReader((out/'transport-history.csv').open()));start=times[0]
    mapping=np.load(folder/'voxel-native-cell-map.npz');ids=mapping['cell'];dims=mapping['dimensions'];origin=mapping['origin'];spacing=float(mapping['spacing'])
    mask=ids>=0;g=pv.ImageData(dimensions=dims,spacing=[spacing]*3,origin=origin)
    g.point_data['Dye_C']=np.zeros(len(ids),np.float32)
    surface=pv.read(folder/'original-fluid-boundary.vtp')
    def sample(c):
        v=np.zeros(len(ids),np.float32);v[mask]=c[ids[mask]];return v
    max_projection=0.
    for f in frames:
        with np.load(out/f['file']) as d:
            for side in ['right','left']:
                p=sample(d[side]).reshape(tuple(dims),order='F').sum(axis=1)*spacing
                max_projection=max(max_projection,float(p.max()))
    assert max_projection>0
    save(folder/'shared-display-scales.json',{'concentration':[0,1],'integrated_concentration_mm':[0,max_projection],
        'projection_axis':'Model-coordinate Y, not a registered clinical imaging orientation',
        'projection_display_gamma':.5,'volume_palette':'Oranges truncated to colour interval 0.35 to 1, fixed across all frames and injections',
        'temporal_interpolation':'Linear interpolation of saved scalar fields for evenly spaced movie frames; not additional CFD time points',
        'movie_duration_s':10,'fps':24,'concentration_rendering':'Fixed sequential color and fixed opacity transfer function for both injections',
        'image_physics':'Angiography-style line integral only; no scanner response or physical iodine attenuation'})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    curves={s:[r for r in rows if r['injection']==s] for s in ['right','left']}
    colours={'right':'#c2410c','left':'#1d4ed8'}
    dye_cmap=ListedColormap(plt.get_cmap('Oranges')(np.linspace(.35,1,256)))
    videos=[]
    for side in ['right','left']:
        path=folder/(side+'-ICA-virtual-dye.mp4');assert not path.exists()
        plot=pv.Plotter(off_screen=True,window_size=(880,1040));plot.set_background('white')
        plot.add_mesh(surface,color='#64748b',opacity=.055)
        opacity=(np.sqrt(np.linspace(0,1,256))*.40*255).astype(np.uint8);opacity[0]=0
        plot.add_volume(g,scalars='Dye_C',clim=[0,1],cmap=dye_cmap,opacity=opacity,shade=False,
            scalar_bar_args={'title':'Virtual dye C','vertical':False,'width':.75,'height':.1,'position_x':.13,'position_y':.02,'n_labels':3,
                             'title_font_size':20,'label_font_size':18})
        center=np.asarray(surface.center)
        plot.camera_position=[center+np.array([45,-120,25]),center,[0,0,1]]
        plot.enable_parallel_projection();plot.camera.parallel_scale=32
        fig=plt.figure(figsize=(16,9),dpi=100,layout='constrained')
        spec=fig.add_gridspec(2,2,width_ratios=[1.1,1],height_ratios=[1.45,1])
        ax3=fig.add_subplot(spec[:,0]);ax3.axis('off');ax3.set_title('3D virtual dye',fontsize=16)
        mark3=ax3.imshow(np.full((1040,880,3),255,np.uint8))
        ax2=fig.add_subplot(spec[0,1]);extent=[origin[0],origin[0]+(dims[0]-1)*spacing,origin[2],origin[2]+(dims[2]-1)*spacing]
        dsa=ax2.imshow(np.zeros((dims[2],dims[0])),origin='lower',extent=extent,cmap='Greys',norm=PowerNorm(.5,vmin=0,vmax=max_projection),aspect='equal',interpolation='nearest')
        ax2.set(title='Angiography-style projection',xlabel='X (mm)',ylabel='Z (mm)')
        ax2.add_patch(Ellipse((CENTER_MM[0],CENTER_MM[2]),2*RADII_MM[0],2*RADII_MM[2],
                             fill=False,edgecolor='#2563eb',lw=1.1,ls='--'))
        ax2.text(CENTER_MM[0]+RADII_MM[0],CENTER_MM[2]+RADII_MM[2],'ROI',fontsize=9,color='#1d4ed8')
        fig.colorbar(dsa,ax=ax2,label='Integrated C (mm)',fraction=.04,pad=.02)
        axc=fig.add_subplot(spec[1,1]);maximum=0
        for s in ['right','left']:
            r=curves[s];x=[0]+[(float(q['solver_time_s'])-start)*1000 for q in r];y=[0]+[100*float(q['ellipsoidal_ROI_mean_C']) for q in r]
            axc.plot(x,y,color=colours[s],ls='-' if s=='right' else '--',lw=2,label=s.capitalize()+' ICA injection')
            maximum=max(maximum,max(y))
        axc.set(xlim=(0,(times[-1]-start)*1000),ylim=(0,max(.01,maximum*1.15)),xlabel='Time since injection (ms)',ylabel='Mean C (% of inlet)',title='Ellipsoidal sac ROI')
        axc.grid(True,color='#e2e8f0');axc.spines[['top','right']].set_visible(False);axc.legend(frameon=False,fontsize=10)
        cursor=axc.axvline(0,color='#334155',lw=1);axc.axvspan(0,100,color='#e2e8f0',alpha=.5)
        title=fig.suptitle('',fontsize=18)
        fig.text(.015,.005,'Virtual tracer | %.2f-mm connected scalar groups | Not clinical DSA'%protocol.get('connected_scalar_agglomeration_mm',0),fontsize=9,color='#475569')
        logpath=folder/(side+'-encoding.log')
        command=[str(FFMPEG),'-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24','-video_size','1600x900','-framerate','24','-i','-',
                 '-an','-c:v','libopenh264','-b:v','5M','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        cached={};video_start=time.monotonic()
        with logpath.open('x') as log:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                for index,t in enumerate(np.linspace(times[0],times[-1],240)):
                    j=min(len(times)-2,max(0,int(np.searchsorted(times,t,side='right')-1)))
                    for n in [j,j+1]:
                        if n not in cached:
                            with np.load(out/frames[n]['file']) as d:cached[n]=d[side].copy()
                    alpha=(t-times[j])/(times[j+1]-times[j]);c=(1-alpha)*cached[j]+alpha*cached[j+1]
                    for n in list(cached):
                        if n not in [j,j+1]:del cached[n]
                    v=sample(c);g.point_data['Dye_C'][:]=v;g.GetPointData().GetArray('Dye_C').Modified()
                    plot.render()
                    mark3.set_data(plot.screenshot(return_img=True));projection=v.reshape(tuple(dims),order='F').sum(axis=1)*spacing
                    dsa.set_data(projection.T);elapsed=(t-start)*1000;cursor.set_xdata([elapsed,elapsed])
                    phase='Injection' if elapsed<100 else 'Washout'
                    title.set_text(side.capitalize()+' ICA | '+phase+' | Patient-specific CFD | t = %.0f ms'%elapsed)
                    fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                    proc.stdin.write(rgb.tobytes())
                    if index in [0,28,55,110,239]:
                        fig.savefig(folder/(side+'-preview-%03d.png'%index),dpi=100)
                    if index%30==0:print('DYE_RENDER',side,index,'/240',round(time.monotonic()-video_start,1),flush=True)
                proc.stdin.close();assert proc.wait(timeout=120)==0
            except BaseException:
                proc.stdin.close()
                if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
                raise
        plt.close(fig);plot.close();videos.append({'injection':side,'file':str(path),'bytes':path.stat().st_size,'frames':240,'fps':24,'duration_s':10})
    save(folder/'video-manifest.json',videos)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare(a.out)
    else:render(a.out)
