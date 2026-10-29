"""Recolor original saved tracks by permanent birth-cycle category; no retracking."""
import argparse, json, subprocess, time
from pathlib import Path
import h5py
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from common_followup import *

FF=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')


def run(out):
    out.mkdir(exist_ok=False);dirs=selected();handles=[h5py.File(dirs[k]/'tracks.h5','r') for k in KINDS]
    keys=list(handles[0]['frames']);n=len(keys);assert n==951
    plot=pv.Plotter(shape=(2,3),off_screen=True,window_size=(1920,1280))
    actors=[];texts=[];center=np.asarray(load(PROJECT/'apps/wss-topology-20261007-v1/app/sac-data-v1/index.json')['center_mm'])
    cmap=ListedColormap(COLORS)
    for row in range(2):
        for i,k in enumerate(KINDS):
            plot.subplot(row,i);plot.set_background('#fafcfe');plot.enable_depth_peeling(number_of_peels=8,occlusion_ratio=0)
            bounds=[]
            for z in ['wall','aneurysm_sac']:
                s=pv.read(dirs[k]/(z+'.vtu'));s.points*=1000;bounds.append(np.asarray(s.bounds).reshape(3,2))
                plot.add_mesh(s,color='#73869a',opacity=(.10 if row==0 else .08))
            f=handles[i]['frames/'+keys[0]];cloud=pv.PolyData(f['position_m'][:]*1000)
            cloud['Release cycle']=release_cycle(f['birth_time_s'][:])
            actor=plot.add_mesh(cloud,scalars='Release cycle',cmap=cmap,n_colors=10,clim=[.5,10.5],point_size=5,
                lighting=False,render_points_as_spheres=True,show_scalar_bar=row==1,
                annotations={j:'C'+str(j) for j in range(1,11)},
                scalar_bar_args={'title':'Release cycle — '+['MRI','Individualized','Population'][i],'vertical':False,'position_x':.035,
                    'position_y':.055,'width':.93,'height':.14,'title_font_size':13,'label_font_size':12,
                    'n_labels':0,'color':'#172554'})
            actors.append(actor)
            if row==0:
                b=np.array(bounds);c=np.array([b[:,:,0].min(0),b[:,:,1].max(0)]).mean(0)
                plot.camera_position=[c+np.array([20,-145,24]),c,[0,0,1]];scale=39
                plot.add_text(LABELS[i],position=(12,596),font_size=14,color='#172554')
                texts.append(plot.add_text('',position=(12,565),font_size=11,color='#172554'))
                plot.add_text('First-cycle CFD replay; not 10 solved cycles',position=(12,15),font_size=9,color='#475569')
            else:
                plot.camera_position=[center+np.array([4,-28,5]),center,[0,0,1]];scale=7
                plot.add_text('Sac close-up',position=(12,596),font_size=14,color='#172554')
                if k=='individualized':plot.add_text('* Unconverged paths: exploratory',position=(12,150),font_size=10,color='#9a3412')
            plot.enable_parallel_projection();plot.camera.parallel_scale=scale
    plot.show(auto_close=False,interactive=False)
    movie=out/'Three-cases-release-cycle-colored.mp4';log=(out/'encode.log').open('x',encoding='utf-8')
    # Render each saved state once, hold for its true NONUNIFORM time interval.
    cmd=[str(FF),'-hide_banner','-loglevel','warning','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1280',
         '-r','100','-i','pipe:0','-an','-c:v','libopenh264','-b:v','8M','-pix_fmt','yuv420p',
         '-r','30','-movflags','+faststart',str(movie)]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    start=time.monotonic();manifest=[]
    try:
        for j,key in enumerate(keys):
            for i,h in enumerate(handles):
                f=h['frames/'+key];born=f['birth_time_s'][:];c=release_cycle(born)
                cloud=pv.PolyData(f['position_m'][:]*1000);cloud['Release cycle']=c
                for row in range(2):actors[row*3+i].mapper.dataset=cloud
                texts[i].SetInput('Replay %d/10 | %03d ms' % (int(f.attrs['cycle']),round(float(f.attrs['phase_s'])*1000)))
            plot.render();rgb=plot.screenshot(return_img=True)[:,:,:3]
            next_time=float(handles[0]['frames/'+keys[j+1]].attrs['time_s']) if j+1<len(keys) else float(handles[0]['frames/'+key].attrs['time_s'])
            interval=next_time-float(handles[0]['frames/'+key].attrs['time_s'])
            repeats=round(interval*1000)  # 10x playback slowdown, 100-fps input.
            assert abs(repeats-interval*1000)<1e-5
            packed=np.ascontiguousarray(rgb).tobytes()
            for _ in range(repeats):encoder.stdin.write(packed)
            if j in [60,250,900]:
                from PIL import Image
                Image.fromarray(rgb).save(out/('preview-%04d.png'%j))
                manifest.append({'frame':j,'time_s':float(handles[0]['frames/'+key].attrs['time_s'])})
            if j%100==0:print('COHORT_VIDEO',j,n,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('Render bounded at 30 minutes')
    finally:
        encoder.stdin.close();plot.close()
        for h in handles:h.close()
        code=encoder.wait(timeout=180);log.close()
    assert code==0
    products=[movie]
    for i,name in enumerate(['Patient-specific','Individualized-EXPLORATORY','Population-average']):
        path=out/(name+'-release-cycle-colored.mp4')
        filt='[0:v]split=2[a][b];[a]crop=640:640:%d:0[x];[b]crop=640:640:%d:640[y];[x][y]hstack[v]'%(i*640,i*640)
        args=[str(FF),'-v','error','-n','-i',str(movie),'-filter_complex',filt,'-map','[v]','-an',
              '-c:v','libopenh264','-b:v','5M','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        subprocess.run(args,check=True,capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW);products.append(path)
    checks=[]
    for path in products:
        r=subprocess.run([str(FF),'-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True,
                         timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert r.returncode==0,r.stderr
        checks.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'full_decode_passed':True})
    save(out/'video-verification.json',{'status':'encoded_and_fully_decoded','products':checks,'previews':manifest,
        'release_cycle_colors':dict(zip(range(1,11),COLORS)),'rule':'floor((birth_time-0.005)/0.868)+1, fixed for each particle',
        'source_tracks':{k:{'path':str(dirs[k]/'tracks.h5'),'sha256':sha(dirs[k]/'tracks.h5')} for k in KINDS},
        'new_CFD_cycles':0,'tracks_changed':False,'periodicity_validated':False,'individualized_tracking_converged':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out/'videos-cycle-colors-v3')
