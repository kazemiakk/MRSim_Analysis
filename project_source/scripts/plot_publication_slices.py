"""E-042: true matched slices and numerical-support 3D CFD/Koma figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from scripts.complete_advanced_comparisons import load_inputs
from scripts.compare_aorta_hemodynamics import DATA, EXPECTED, dump
from scripts.extend_hemodynamic_comparisons import ROOT, DP, AO, bind, sha

BASE=ROOT/'outputs/hemodynamic_extension'
LABELS={'phantom_1m':'Stenotic phantom | 1M','aorta_1m':'Aorta baseline | 1M','aorta_100k_corrected':'Aorta corrected pilot | 100k'}
PLANES={'XY':(0,0,1),'XZ':(1,0,2),'YZ':(2,1,2)}


def coordinates(shape,spacing,half_voxel=False):
    return [(np.arange(n)-n/2+(.5 if half_voxel else 0))*d*1000 for n,d in zip(shape[::-1],spacing)]


def plane(field,name,index):
    return np.take(field,index,axis=PLANES[name][0])


def vector_maps(a,c,mask):
    finite=mask&np.isfinite(a).all(axis=-1)&np.isfinite(c).all(axis=-1)
    values=[np.linalg.norm(a,axis=-1),np.linalg.norm(c,axis=-1),np.linalg.norm(c-a,axis=-1)]
    return [np.where(finite,v,np.nan) for v in values],finite


def rounded_limit(value):
    if not np.isfinite(value) or value<0:raise ValueError('Invalid scale')
    if value==0:return 1e-6
    step=10**np.floor(np.log10(value))/5
    return float(np.ceil(value/step)*step)


def select_bins(rows,section,native_phantom=False):
    r=sorted([x for x in rows if x['section']==section and (not native_phantom or x['bin']<=13)],key=lambda x:x['bin'])
    if not r:raise ValueError('No eligible phase rows')
    q=np.array([abs(x['net_m3_s']) for x in r])
    return dict(peak=r[int(np.argmax(q))]['bin']-1,low=r[int(np.argmin(q))]['bin']-1)


def geometry_planes(geometry):
    return [(name,int(np.argmax(geometry.sum(axis=tuple(a for a in range(3) if a!=axis))))) for name,(axis,_,_) in PLANES.items() if name!='YZ']


def output_check(target):
    if target.exists():raise FileExistsError(target)


def save(fig,out,name,inventory,caption):
    for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)
    inventory.append(dict(stem=name,caption=caption))


def title(case,b,phase):
    return f"{LABELS[case['name']]} | {phase} flow | bin {b+1} | t = {case['times'][b]:.3f} s"


def slice_figure(case,b,phase,selections,geometry,coords,limits,out,inventory,records,arrays,family):
    narrow=case['name']=='phantom_1m' and family=='longitudinal'
    fig,axs=plt.subplots(len(selections),3,figsize=(7.3,3.8 if narrow else 2.3*len(selections)+1.5),squeeze=False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.29 if narrow else .17,top=.79 if narrow else .86,wspace=.32,hspace=1.0 if narrow else .95)
    fields,valid=vector_maps(case['reference'][b],case['estimate'][b],case['regions']['whole'])
    stem=case['name']+'_'+phase+'_'+family
    for row,(name,index,label) in enumerate(selections):
        axis,h,v=PLANES[name];g=plane(geometry,name,index);m=plane(valid,name,index)
        displayed=[plane(f,name,index) for f in fields]
        extent=[coords[h][0]-case['spacing'][h]*500,coords[h][-1]+case['spacing'][h]*500,coords[v][0]-case['spacing'][v]*500,coords[v][-1]+case['spacing'][v]*500]
        yy,xx=np.where(g)
        if not len(xx):raise ValueError('Empty selected slice')
        xlim=[coords[h][max(int(xx.min())-2,0)],coords[h][min(int(xx.max())+2,len(coords[h])-1)]]
        ylim=[coords[v][max(int(yy.min())-2,0)],coords[v][min(int(yy.max())+2,len(coords[v])-1)]]
        if case['name']=='phantom_1m' and name!='YZ':xlim=[case['throat_x']-75,case['throat_x']+75]
        for col,(values,coltitle,cmap,upper) in enumerate(zip(displayed,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
            ax=axs[row,col]
            context=np.ones((*g.shape,4));context[g,:3]=.88
            ax.imshow(context,origin='lower',extent=extent,interpolation='nearest',aspect='equal')
            cm=plt.get_cmap(cmap).copy();cm.set_bad((1,1,1,0))
            ax.imshow(np.ma.masked_invalid(values),origin='lower',extent=extent,interpolation='nearest',cmap=cm,vmin=0,vmax=upper,aspect='equal')
            if g.any() and not g.all():ax.contour(g.astype(float),[.5],extent=extent,origin='lower',colors='#777777',linewidths=.35)
            ax.set(xlim=xlim,ylim=ylim,xlabel='MRI '+ 'XYZ'[h]+' (mm)',ylabel='MRI '+'XYZ'[v]+' (mm)')
            ax.tick_params(labelsize=6.7,length=2,width=.5)
            ax.set_title(f"{chr(97+row*3+col)}) {coltitle}\n{label}",loc='left',fontsize=7.2,pad=7)
        key=f'{stem}_{row}'
        for k,data in [('cfd',displayed[0]),('koma',displayed[1]),('error',displayed[2]),('mask',m),('geometry',g)]:arrays[key+'_'+k]=data
        records.append(dict(key=key,case=case['name'],bin=b+1,plane=name,index=index,position_mm=float(coords[2-axis][index]),label=label,valid_pixels=int(m.sum()),geometry_pixels=int(g.sum()),xlim=xlim,ylim=ylim,extent=extent,speed_limit=limits[0],error_limit=limits[1]))
    cby=.15 if narrow else .09
    cax=fig.add_axes([.13,cby,.46,.013]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,limits[0]),cmap='viridis'),cax=cax,orientation='horizontal');cb.set_label('Speed (m/s) | CFD and Koma',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=6.5,length=2)
    cax=fig.add_axes([.72,cby,.22,.013]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,limits[1]),cmap='inferno'),cax=cax,orientation='horizontal');cb.set_label('Vector error (m/s)',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=6.5,length=2)
    fig.suptitle(title(case,b,phase),fontsize=10,y=.99)
    note='True voxel slices; paired fixed support; gray = excluded; no smoothing. '
    note+= 'Longitudinal stenosis ROI.' if case['name']=='phantom_1m' and family=='longitudinal' else 'MRI planes, not certified anatomical orientation.'
    fig.text(.5,.008,note.replace('gray = excluded; ','gray = excluded;\n'),ha='center',fontsize=6.1)
    caption=title(case,b,phase)+'. '+family+' slices. '+note+' Error = norm(Koma velocity − CFD velocity). Full-value common scales; no clipping. Internal consistency only.'
    save(fig,out,stem,inventory,caption)


def whole_3d(case,b,phase,coords,context,limits,out,inventory,records,arrays):
    fields,valid=vector_maps(case['reference'][b],case['estimate'][b],case['regions']['whole'])
    z,y,x=np.where(valid);points=np.column_stack([coords[0][x],coords[1][y],coords[2][z]])
    # All paired voxels are rendered; only gray geometry context is subsampled.
    geometry=context[::max(1,int(np.ceil(len(context)/12000)))];lo=context.min(axis=0);hi=context.max(axis=0);span=hi-lo
    stem=case['name']+'_'+phase+'_whole3d'
    fig=plt.figure(figsize=(8,6.0));fig.subplots_adjust(left=.0,right=.98,bottom=.22,top=.86,wspace=0,hspace=.12)
    cameras=[(20,-70),(20,30)]
    for row,(elev,azim) in enumerate(cameras):
        for col,(values,label,cmap,upper) in enumerate(zip(fields,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
            ax.scatter(*geometry.T,c='#b9bec4',s=.8,alpha=.10,depthshade=False,edgecolors='none',rasterized=True)
            ax.scatter(*points.T,c=values[valid],s=2.2 if case['name']=='phantom_1m' else 8,cmap=cmap,vmin=0,vmax=upper,depthshade=False,edgecolors='none',rasterized=True)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
            ax.set_box_aspect(np.maximum(span,1));ax.view_init(elev=elev,azim=azim);ax.set_proj_type('ortho')
            ax.set_axis_off();ax.set_title(f'{chr(97+row*3+col)}) {label} | View {row+1}',fontsize=8,loc='left',pad=1)
            center=(lo+hi)/2
            def projected(p):
                xy=np.array(proj3d.proj_transform(*p,ax.get_proj())[:2])
                return ax.transAxes.inverted().transform(ax.transData.transform(xy))
            origin=projected(center)
            length=float(np.linalg.norm(projected(center+np.array([50,0,0]))-origin))
            ax.add_artist(Line2D([.62,.62+length],[.04,.04],transform=ax.transAxes,color='#222222',lw=1))
            ax.text2D(.62+length/2,.06,'50 mm (X)',transform=ax.transAxes,fontsize=5.5,ha='center')
            for j,letter in enumerate('XYZ'):
                direction=projected(center+np.eye(3)[j]*10)-origin
                direction=direction/max(np.linalg.norm(direction),1e-12)*.10
                start=np.array([.13,.12]);end=start+direction
                ax.annotate('',xy=end,xytext=start,xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=.55,color='#555555'))
                ax.text2D(*(end+direction*.15),letter,transform=ax.transAxes,fontsize=5.5,ha='center')
    for left,width,cmap,upper,label in [(.11,.48,'viridis',limits[0],'Speed (m/s) | CFD and Koma'),(.73,.21,'inferno',limits[1],'Vector error (m/s)')]:
        cb=fig.colorbar(ScalarMappable(norm=Normalize(0,upper),cmap=cmap),cax=fig.add_axes([left,.135,width,.017]),orientation='horizontal');cb.set_label(label,fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=7,length=2)
    fig.suptitle(title(case,b,phase)+'\nWhole-shape numerical-support volume | two oblique views',fontsize=10,y=.99)
    fig.text(.5,.015,'Colored: all paired voxel centers. Gray: geometry context, not recovered velocity or a certified wall.\nMRI-axis triads; scale bar = projected 50 mm along MRI X. No interpolation or wall-velocity claim.',ha='center',fontsize=6.5)
    for key,data in [('points',points),('cfd',fields[0][valid]),('koma',fields[1][valid]),('error',fields[2][valid])]:arrays[stem+'_'+key]=data
    records.append(dict(key=stem,case=case['name'],bin=b+1,kind='3D',points=len(points),context_points=len(geometry),camera=cameras,speed_limit=limits[0],error_limit=limits[1]))
    save(fig,out,stem,inventory,title(case,b,phase)+'. Whole-shape numerical-support voxel-center visualization, not an interpolated wall/surface field. All paired points, full maxima scales, identical cameras. Aortic geometry is an approximate point cloud; gray context has no recovered velocity. Internal consistency only.')


def main(target):
    output_check(target)
    inputs,cases,dp,ao,ext,_,_=load_inputs();inputs.extend([bind(Path(__file__)),bind(BASE/'E042_PROTOCOL.md')])
    with h5py.File(DP/'matched_reference.h5') as f:source_z=f['source_z_m'][:]
    path=DATA/'point_coordinates.npy';inputs.append(bind(path,EXPECTED['point_coordinates.npy']))
    native=np.load(path,mmap_mode='r',allow_pickle=False)
    context_aorta=(native[::max(1,int(np.ceil(len(native)/30000)))][:,ao['grid']['order']]-np.array(ao['grid']['center'])[ao['grid']['order']])*1000
    with h5py.File(AO/'matched_reference.h5') as f:geometry_aorta=f['fraction'][:]>0
    for case in cases:
        name=case['name'];case['coords']=coordinates(case['shape'],case['spacing'],name=='phantom_1m')
        if name=='phantom_1m':
            rows=[r for r in dp['section_values'] if r['dataset']=='cfd_voxel_common'];case['bins']=select_bins(rows,'upstream',True);geometry=case['volume']>0
            case['throat_x']=float(case['coords'][0][np.argmin(abs(source_z))])
            long=[('XY',case['shape'][0]//2,'Longitudinal XY | Z = %.2f mm'%case['coords'][2][case['shape'][0]//2]),('XZ',case['shape'][1]//2,'Longitudinal XZ | Y = %.2f mm'%case['coords'][1][case['shape'][1]//2])]
            cross=[('YZ',int(np.argmin(abs(source_z-z))),label) for z,label in [(-.035,'Upstream'),(0,'Throat'),(.030,'Downstream')]]
            zz,yy,xx=np.where(geometry);case['context']=np.column_stack([case['coords'][0][xx],case['coords'][1][yy],case['coords'][2][zz]])
        else:
            rows=[r for r in (ao['section_series'] if name=='aorta_1m' else ext['aorta_pilot_sections']['series']) if r['method']=='CFD'];case['bins']=select_bins(rows,'X+000_C1');geometry=geometry_aorta
            long=[(p,i,f'{p} | '+('Z' if p=='XY' else 'Y')+f' = {case["coords"][2-PLANES[p][0]][i]:.2f} mm') for p,i in geometry_planes(geometry)]
            cross=[('YZ',int(np.argmin(abs(case['coords'][0]-x))),f'X = {x:+.1f} mm') for x in [-58.3333333333,0,58.3333333333]];case['context']=context_aorta
        case['geometry']=geometry;case['long']=long;case['cross']=[(p,i,label+f' | MRI X = {case["coords"][0][i]:+.2f} mm' if name=='phantom_1m' else label) for p,i,label in cross]
    maxima={}
    for group in ['phantom','aorta']:
        vals=[];errors=[]
        for c in cases:
            if c['name'].startswith(group):
                for b in c['bins'].values():
                    f,m=vector_maps(c['reference'][b],c['estimate'][b],c['regions']['whole']);vals.extend([float(np.nanmax(x)) for x in f[:2]]);errors.append(float(np.nanmax(f[2])))
        maxima[group]=(rounded_limit(max(vals)),rounded_limit(max(errors)))
    target.parent.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix=target.name+'.partial-',dir=target.parent))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    inventory=[];records=[];arrays={}
    for c in cases:
        limits=maxima['phantom' if c['name']=='phantom_1m' else 'aorta']
        for phase,b in c['bins'].items():
            print(c['name'],phase,b+1,flush=True)
            for family,selection in [('longitudinal',c['long']),('transverse',c['cross'])]:slice_figure(c,b,phase,selection,c['geometry'],c['coords'],limits,out,inventory,records,arrays,family)
            whole_3d(c,b,phase,c['coords'],c['context'],limits,out,inventory,records,arrays)
    np.savez_compressed(out/'displayed_samples.npz',**arrays)
    report=dict(experiment='E-042',inputs=inputs,limits=maxima,inventory=inventory,records=records,cases={c['name']:dict(bins={k:v+1 for k,v in c['bins'].items()},times_s={k:float(c['times'][v]) for k,v in c['bins'].items()},shape=c['shape'],spacing_xyz_m=c['spacing'].tolist()) for c in cases},limitations=['MRI computational planes, not certified patient sagittal/axial orientation','Point-cloud geometry/support is not a native wall','Fixed primary masks differ across aortic acquisitions','No interpolation, smoothing, inpainting or clipped extrema','Internal consistency, not independent validation'])
    dump(out/'report.json',report)
    lines=['# E-042 publication comparison figures','','18 PNG/PDF pairs at600dpi. See CAPTIONS.md for scientific scope.','','| Figure | PNG | PDF |','|---|---|---|']
    for r in inventory:lines.append(f'| {r["stem"]} | [PNG]({r["stem"]}.png) | [PDF]({r["stem"]}.pdf) |')
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'CAPTIONS.md').write_text('# Suggested figure captions\n\n'+'\n\n'.join('## '+r['stem']+'\n\n'+r['caption'] for r in inventory),encoding='utf-8')
    dump(out/'MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    output_check(target);out.rename(target);print('Saved '+str(target))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output.resolve())
