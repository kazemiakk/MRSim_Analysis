"""E-043: native-orientation aortic 3D, depth projections and true slices."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
from scripts.complete_advanced_comparisons import load_inputs
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.compare_aorta_hemodynamics import DATA, EXPECTED, dump
from scripts.extend_hemodynamic_comparisons import ROOT, bind, sha
from scripts.plot_publication_slices import vector_maps, rounded_limit, select_bins

BASE=ROOT/'outputs/hemodynamic_extension'
WALL=ROOT/'outputs/aorta_dp2_mri_pipeline/visualization_branch_corrected/aorta_wall_points_from_wss.npy'
WALL_SHA='eb195abe0a0a3a604648b9eaf933066af5a178a37c77f00bdb48ff5f9f9983dd'
LABELS={'aorta_1m':'Aorta | baseline 1M','aorta_100k_corrected':'Aorta | corrected 100k pilot'}


def same_depth_projection(fields, mask):
    """ZYX input; source-X/Z output, selected along MRI Z/source Y by CFD speed."""
    a=np.where(mask,fields[0],-np.inf)
    depth=np.argmax(a,axis=0)
    valid=mask.any(axis=0)
    values=[np.where(valid,np.take_along_axis(v,depth[None],axis=0)[0],np.nan).T for v in fields]
    return values,depth.T,valid.T


def branch_clusters(wall_mm):
    top=wall_mm[wall_mm[:,2]>90]
    seeds=np.array([top[np.argmin(top[:,0]),:2],np.median(top[:,:2],axis=0),top[np.argmax(top[:,0]),:2]])
    centers,labels=kmeans2(top[:,:2],seeds,iter=50,minit='matrix',missing='raise')
    return top,labels,np.argsort(centers[:,0])


def branch_planes(wall_mm, source_y_centers):
    """Deterministic geometric neighborhoods, not anatomical labels or flow fitting."""
    top,labels,order=branch_clusters(wall_mm);result=[]
    for n,k in enumerate(order):
        points=top[labels==k]
        center=np.median(points,axis=0)
        i=int(np.argmin(abs(source_y_centers-center[1])))
        result.append(dict(region=f'B{n+1}',index=i,source_center_mm=center.tolist(),source_y_mm=float(source_y_centers[i]),sample_count=len(points)))
    return result


def all_branch_plane(wall_mm,source_y_centers):
    top,labels,order=branch_clusters(wall_mm)
    fractions=np.array([[np.mean(abs(top[labels==k,1]-y)<=2) for k in order] for y in source_y_centers])
    # Maximize the least-represented geometric neighborhood; geometry only.
    scores=fractions.min(axis=1);i=int(np.argmax(scores))
    if scores[i]<=0:raise ValueError('No voxel slab intersects all geometric branch neighborhoods; use projection')
    return dict(index=i,source_y_mm=float(source_y_centers[i]),fractions=fractions[i].tolist(),criterion='maximum minimum native-tip sample fraction within ±2mm; no velocity fitting')


def axes_style(ax,xlim,ylim):
    ax.set(xlim=xlim,ylim=ylim,aspect='equal',xlabel='Source X (mm)',ylabel='Source Z (mm)')
    ax.tick_params(labelsize=7,length=2,width=.5)
    for s in ax.spines.values():s.set_linewidth(.5);s.set_color('#8b8b8b')


def colorbars(fig,limits,y=.12):
    for left,width,cmap,upper,label in [(.10,.49,'viridis',limits[0],'Speed (m/s) | CFD / Koma'),(.73,.23,'inferno',limits[1],'Vector error (m/s)')]:
        cb=fig.colorbar(ScalarMappable(norm=Normalize(0,upper),cmap=cmap),cax=fig.add_axes([left,y,width,.013]),orientation='horizontal')
        cb.set_label(label,fontsize=8,labelpad=3);cb.ax.tick_params(labelsize=7,length=2)


def export(fig,stem,out,inventory,caption):
    for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.08)
    plt.close(fig);inventory.append(dict(stem=stem,caption=caption));print(stem,flush=True)


def title(case,b,phase):
    return f"{LABELS[case['name']]} | {phase} flow | bin {b+1} | t = {case['times'][b]:.3f} s"


def projected_sheet(case,b,phase,fields,valid,wall,grid,limits,out,inventory,arrays,records):
    values,depth,mask=same_depth_projection(fields,valid)
    xyz=grid.axes();xs=xyz[1]*1000+grid.center[0]*1000;zs=xyz[0]*1000+grid.center[2]*1000
    extent=[xs[0]-2,xs[-1]+2,zs[0]-grid.spacing[0]*500,zs[-1]+grid.spacing[0]*500]
    bounds=[wall.min(0)-4,wall.max(0)+4];stem=case['name']+'_'+phase+'_all_branches_projection'
    fig,axs=plt.subplots(2,3,figsize=(8.2,8.4),gridspec_kw={'height_ratios':[3,1]});fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.87,wspace=.34,hspace=.42)
    for row in range(2):
        for col,(v,label,cmap,upper) in enumerate(zip(values,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
            ax=axs[row,col]
            ax.scatter(wall[:,0],wall[:,2],s=.10,c='#c5c8cc',alpha=.6,edgecolors='none',rasterized=True,zorder=0)
            cm=plt.get_cmap(cmap).copy();cm.set_bad((1,1,1,0))
            ax.imshow(np.ma.masked_invalid(v),origin='lower',extent=extent,cmap=cm,vmin=0,vmax=upper,interpolation='nearest',aspect='equal',zorder=1)
            axes_style(ax,[bounds[0][0],bounds[1][0]],[bounds[0][2] if row==0 else 45,bounds[1][2]])
            ax.set_title(f'{chr(97+row*3+col)}) {label} | '+('Whole arch' if row==0 else 'Branch detail'),fontsize=8,loc='left',pad=8)
    fig.suptitle(title(case,b,phase)+'\nAll-branches depth projection (not a planar slice)',fontsize=11,y=.98)
    colorbars(fig,limits)
    fig.text(.5,.022,'Depth chosen by maximum CFD speed; Koma and error use the SAME voxel.\nGray: native geometry context without paired velocity. No interpolation or filling.',ha='center',fontsize=7)
    for k,v in zip(['cfd','koma','error'],values):arrays[stem+'_'+k]=v
    arrays[stem+'_depth_index']=depth;arrays[stem+'_valid']=mask
    records.append(dict(stem=stem,kind='same_depth_projection',bin=b+1,case=case['name'],valid_pixels=int(mask.sum()),depth_axis='source Y / MRI Z',selector='maximum CFD speed on fixed common support'))
    export(fig,stem,out,inventory,title(case,b,phase)+'. Whole arch and branch-detail source-X/Z depth projection, not a single slice. Maximum-CFD-speed depth selector is shared by CFD, Koma and vector-error panels. Gray native wall-associated samples are context only; missing paired velocities are not inferred. Original source orientation, no percentile clipping or smoothing. Internal consistency only.')


def true_slices(case,b,phase,fields,valid,wall,grid,planes,limits,out,inventory,arrays,records):
    xyz=grid.axes();xs=xyz[1]*1000+grid.center[0]*1000;zs=xyz[0]*1000+grid.center[2]*1000
    extent=[xs[0]-2,xs[-1]+2,zs[0]-grid.spacing[0]*500,zs[-1]+grid.spacing[0]*500]
    fig,axs=plt.subplots(3,3,figsize=(8.2,9.4));fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.87,wspace=.34,hspace=.65)
    stem=case['name']+'_'+phase+'_true_branch_slices';bounds=[wall.min(0)-4,wall.max(0)+4]
    for row,p in enumerate(planes):
        i=p['index'];slab=wall[abs(wall[:,1]-p['source_y_mm'])<=2]
        mask=valid[i].T;vals=[v[i].T for v in fields]
        for col,(v,label,cmap,upper) in enumerate(zip(vals,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
            ax=axs[row,col];ax.scatter(slab[:,0],slab[:,2],s=.13,c='#c5c8cc',alpha=.6,edgecolors='none',rasterized=True,zorder=0)
            cm=plt.get_cmap(cmap).copy();cm.set_bad((1,1,1,0))
            ax.imshow(np.ma.masked_invalid(v),origin='lower',extent=extent,cmap=cm,vmin=0,vmax=upper,interpolation='nearest',aspect='equal',zorder=1)
            axes_style(ax,[bounds[0][0],bounds[1][0]],[45,bounds[1][2]])
            ax.set_title(f"{chr(97+row*3+col)}) {label} | {p['region']}\nY = {p['source_y_mm']:.2f} mm",fontsize=8,loc='left',pad=8)
        for k,v in zip(['cfd','koma','error'],vals):arrays[f'{stem}_{row}_{k}']=v
        arrays[f'{stem}_{row}_valid']=mask
        records.append(dict(stem=stem,kind='true_slice',bin=b+1,case=case['name'],row=row,valid_pixels=int(mask.sum()),**p))
    fig.suptitle(title(case,b,phase)+'\nThree geometry-targeted branch-region slices',fontsize=11,y=.985);colorbars(fig,limits,.11)
    fig.text(.5,.019,'True single-voxel planes; no depth projection. B1–B3 are geometric regions, not named arteries.\nGray: native samples in a ±2 mm context slab. One plane cannot be assumed to show all branches.',ha='center',fontsize=7)
    export(fig,stem,out,inventory,title(case,b,phase)+'. Three true longitudinal source-Y planes selected from ordered native geometric branch neighborhoods above Z90mm. Native voxel planes, no projection/smoothing/filling. Gray samples are a ±2mm geometric context slab, not recovered velocity or certified wall. B1–B3 are geometric clusters, not anatomical identities. Branch crop Z>=45mm. Internal consistency only.')


def whole_3d(case,b,phase,fields,valid,wall,grid,limits,out,inventory,arrays,records):
    points=grid.to_source(grid.points()[valid])*1000
    fig=plt.figure(figsize=(8.8,5.5));fig.subplots_adjust(left=0,right=1,bottom=.23,top=.85,wspace=-.08)
    lo=wall.min(0)-3;hi=wall.max(0)+3;span=hi-lo
    stem=case['name']+'_'+phase+'_whole_aorta_3D'
    for col,(v,label,cmap,upper) in enumerate(zip(fields,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
        ax=fig.add_subplot(1,3,col+1,projection='3d')
        ax.scatter(*wall.T,c='#8f969e',s=.35,alpha=.12,edgecolors='none',depthshade=False,rasterized=True)
        ax.scatter(*points.T,c=v[valid],s=6,cmap=cmap,vmin=0,vmax=upper,edgecolors='none',depthshade=False,rasterized=True)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(span,zoom=1.0)
        ax.view_init(elev=8,azim=-100);ax.set_proj_type('ortho');ax.set_axis_off()
        ax.set_title(f'{chr(97+col)}) {label}',loc='center',fontsize=10,pad=8)
        ax.text2D(.45,-.035,'Source X →\nSource Z ↑',transform=ax.transAxes,ha='center',fontsize=7)
        center=(lo+hi)/2
        def projected(p):
            xy=proj3d.proj_transform(*p,ax.get_proj())[:2]
            return ax.transAxes.inverted().transform(ax.transData.transform(xy))
        length=float(np.linalg.norm(projected(center+np.array([0,0,50]))-projected(center)))
        ax.add_artist(Line2D([.12,.12],[.12,.12+length],transform=ax.transAxes,color='#555555',lw=1))
        ax.text2D(.09,.12+length/2,'50 mm',transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=6)
    fig.suptitle(title(case,b,phase)+'\nWhole aorta | branch-revealing 3D view',fontsize=11,y=.98)
    colorbars(fig,limits,.115)
    fig.text(.5,.015,'Colored: paired numerical voxel centers. Gray: native wall-associated point samples.\nIdentical physical aspect; projected50mm source-Z bar. No invented surface or wall velocity.',ha='center',fontsize=7)
    arrays[stem+'_source_xyz_mm']=points
    for k,v in zip(['cfd','koma','error'],fields):arrays[stem+'_'+k]=v[valid]
    records.append(dict(stem=stem,kind='whole_3D',bin=b+1,case=case['name'],points=len(points),wall_context_points=len(wall),camera=dict(elevation=8,azimuth=-100)))
    export(fig,stem,out,inventory,title(case,b,phase)+'. Full native aortic shape and three branch regions in a near-source-XZ oblique orthographic3D view. Color encodes paired voxel-center speeds and vector errors, not continuous wall velocities. Gray native wall-associated point samples are geometry context, not a certified connected wall. Identical cameras and physical aspect. Internal consistency only.')


def best_true_slice(case,b,phase,fields,valid,wall,grid,best,limits,out,inventory,arrays,records):
    i=best['index'];slab=wall[abs(wall[:,1]-best['source_y_mm'])<=2]
    xyz=grid.axes();xs=xyz[1]*1000+grid.center[0]*1000;zs=xyz[0]*1000+grid.center[2]*1000
    extent=[xs[0]-2,xs[-1]+2,zs[0]-grid.spacing[0]*500,zs[-1]+grid.spacing[0]*500]
    bounds=[wall.min(0)-4,wall.max(0)+4];stem=case['name']+'_'+phase+'_best_all_branches_true_slice'
    fig,axs=plt.subplots(1,3,figsize=(8.2,4.2));fig.subplots_adjust(left=.08,right=.985,bottom=.36,top=.73,wspace=.34)
    for col,(v,label,cmap,upper) in enumerate(zip(fields,['CFD','Koma','Vector error'],['viridis','viridis','inferno'],[limits[0],limits[0],limits[1]])):
        ax=axs[col];ax.scatter(slab[:,0],slab[:,2],s=.13,c='#c5c8cc',alpha=.6,edgecolors='none',rasterized=True,zorder=0)
        cm=plt.get_cmap(cmap).copy();cm.set_bad((1,1,1,0))
        ax.imshow(np.ma.masked_invalid(v[i].T),origin='lower',extent=extent,cmap=cm,vmin=0,vmax=upper,interpolation='nearest',aspect='equal',zorder=1)
        axes_style(ax,[bounds[0][0],bounds[1][0]],[45,bounds[1][2]])
        ax.set_title(f'{chr(97+col)}) {label}',fontsize=9,loc='left',pad=8)
    fig.suptitle(title(case,b,phase)+f"\nBranch-revealing true slice | source Y = {best['source_y_mm']:.2f} mm",fontsize=10,y=.98)
    colorbars(fig,limits,.18)
    fig.text(.5,.025,'Single native voxel plane; its4mm context slab intersects all three geometric branch-tip regions.\nGray: native geometry without paired velocity. Coarse resolution limits branch-edge separation.',ha='center',fontsize=7)
    for k,v in zip(['cfd','koma','error'],fields):arrays[stem+'_0_'+k]=v[i].T
    arrays[stem+'_0_valid']=valid[i].T
    records.append(dict(stem=stem,kind='true_slice',row=0,bin=b+1,case=case['name'],valid_pixels=int(valid[i].sum()),**best))
    export(fig,stem,out,inventory,title(case,b,phase)+f". True longitudinal source-Y voxel plane at {best['source_y_mm']:.2f}mm, selected solely to maximize the minimum representation of the three native geometric tip neighborhoods in its ±2mm voxel slab. This is not proof that one infinitesimal plane contains all branch centerlines or that all branches are recovered. Branch crop Z>=45mm. Gray native geometry is context only; no interpolation/filling. Internal consistency only.")


def main(target):
    if target.exists():raise FileExistsError(target)
    inputs,cases,_,ao,ext,_,_=load_inputs();cases=cases[1:]
    inputs.extend([bind(WALL,WALL_SHA),bind(DATA/'point_coordinates.npy',EXPECTED['point_coordinates.npy']),bind(Path(__file__)),bind(BASE/'E043_PROTOCOL.md'),bind(BASE/'E043_PROTOCOL_V2.md'),bind(BASE/'E043_PROTOCOL_V3.md')])
    wall=np.load(WALL,allow_pickle=False).astype(float);native=np.load(DATA/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    if not np.isfinite(wall).all() or wall.shape!=(119097,3):raise ValueError('Wall context contract')
    distances,_=cKDTree(native).query(wall,workers=-1)
    if distances.max()!=0:raise ValueError('Native context does not match source')
    grid=AortaGrid();wall*=1000;planes=branch_planes(wall,grid.axes()[2]*1000+grid.center[1]*1000)
    best=all_branch_plane(wall,grid.axes()[2]*1000+grid.center[1]*1000)
    maxima=[[],[]]
    for case in cases:
        rows=ao['section_series'] if case['name']=='aorta_1m' else ext['aorta_pilot_sections']['series']
        case['bins']=select_bins([r for r in rows if r['method']=='CFD'],'X+000_C1')
        for b in case['bins'].values():
            f,_=vector_maps(case['reference'][b],case['estimate'][b],case['regions']['whole']);maxima[0].extend(float(np.nanmax(x)) for x in f[:2]);maxima[1].append(float(np.nanmax(f[2])))
    limits=[rounded_limit(max(v)) for v in maxima]
    target.parent.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='.partial-e043-',dir=target.parent))
    inventory=[];records=[];arrays={};plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.labelsize':8})
    for case in cases:
        for phase,b in case['bins'].items():
            fields,valid=vector_maps(case['reference'][b],case['estimate'][b],case['regions']['whole'])
            whole_3d(case,b,phase,fields,valid,wall,grid,limits,out,inventory,arrays,records)
            projected_sheet(case,b,phase,fields,valid,wall,grid,limits,out,inventory,arrays,records)
            true_slices(case,b,phase,fields,valid,wall,grid,planes,limits,out,inventory,arrays,records)
            best_true_slice(case,b,phase,fields,valid,wall,grid,best,limits,out,inventory,arrays,records)
    np.savez_compressed(out/'plotted_values.npz',**arrays)
    dump(out/'report.json',dict(experiment='E-043',inputs=inputs,inventory=inventory,records=records,branch_planes=planes,best_all_branch_plane=best,limits=limits,wall_source_distance_max_m=float(distances.max()),grid=dict(shape=grid.shape,center=grid.center,order=grid.order),scope='presentation derivative; internal consistency, not independent physical validation'))
    lines=['# Aorta: clear whole3D and branch views','','16 PNG/PDF figure pairs at600dpi. Baseline1M and corrected100k remain separate.','','| Figure | PNG | PDF |','|---|---|---|']
    lines.extend(f'| {r["stem"]} | [PNG]({r["stem"]}.png) | [PDF]({r["stem"]}.pdf) |' for r in inventory)
    lines+=['','## Read before manuscript use','','The all-branches projection is NOT a true planar slice. CFD chooses a common depth per image pixel; Koma and vector error use that same voxel. Gray native geometry does not imply velocity recovery. The three true-plane sheets show exact source-Y positions; one plane is not assumed to intersect all branches. B1–B3 are geometric neighborhoods, not named arteries. All axes are original source coordinates, not verified patient orientation. Full-value limits, masks, actual ADC timing and acquisition versions remain unchanged. No smoothing or missing-data filling.','', 'These figures are internal-consistency visualizations, not independent validation or certified walls. Low flow looks darker on the common peak-inclusive scale.','', '[Draft captions](CAPTIONS.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'CAPTIONS.md').write_text('# Draft captions\n\n'+'\n\n'.join('## '+r['stem']+'\n\n'+r['caption'] for r in inventory),encoding='utf-8')
    dump(out/'MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    if target.exists():raise FileExistsError(target)
    out.rename(target);print('Saved '+str(target),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output.resolve())
