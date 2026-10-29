"""Validate the completed 86.8-s composite and export its three individual views."""
import argparse,re,subprocess
from pathlib import Path
from common_followup import *
FF=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')


def probe(path):
    r=subprocess.run([str(FF),'-hide_banner','-i',str(path)],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    m=re.search(r'Duration: (\d+):(\d+):([\d.]+)',r.stderr);assert m,r.stderr
    duration=int(m[1])*3600+int(m[2])*60+float(m[3]);assert abs(duration-86.8)<.034
    d=subprocess.run([str(FF),'-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True,
        timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    assert d.returncode==0 and not d.stderr.strip(),d.stderr
    return {'path':str(path),'duration_s':duration,'sha256':sha(path),'bytes':path.stat().st_size,'full_decode_passed':True}


def main(out):
    movie=out/'Three-cases-release-cycle-colored.mp4';products=[probe(movie)]
    for i,name in enumerate(['Patient-specific','Individualized-EXPLORATORY','Population-average']):
        path=out/(name+'-release-cycle-colored.mp4')
        filt='[0:v]split=2[a][b];[a]crop=640:640:%d:0[x];[b]crop=640:640:%d:640[y];[x][y]hstack[v]'%(i*640,i*640)
        args=[str(FF),'-v','error','-n','-i',str(movie),'-filter_complex',filt,'-map','[v]','-an',
              '-c:v','libopenh264','-b:v','5M','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        subprocess.run(args,check=True,capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        products.append(probe(path))
    for seconds,name in [(7,'early'),(80,'late')]:
        subprocess.run([str(FF),'-v','error','-n','-ss',str(seconds),'-i',str(movie),'-frames:v','1',str(out/('decoded-'+name+'.png'))],
            check=True,capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    save(out/'video-verification.json',{'status':'completed_and_fully_decoded','products':products,
        'release_cycle_colors':dict(zip(range(1,11),COLORS)),'rule':'floor((birth_time-0.005)/0.868)+1',
        'source_tracks':{k:{'path':str(p/'tracks.h5'),'sha256':sha(p/'tracks.h5')} for k,p in selected().items()},
        'source_frame_holds':'True saved nonuniform intervals at 10x slower playback; 100-fps input, 30-fps MP4.',
        'encoder_note':'Composite output completed its requested 86.8 s and closed the input pipe at the end. Duration and full decode independently checked before exporting individual panels. The reproducible renderer now terminates on input EOF rather than ffmpeg -t.',
        'new_CFD_cycles':0,'tracks_changed':False,'periodicity_validated':False,'individualized_tracking_converged':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out/'videos-cycle-colors-v2')
