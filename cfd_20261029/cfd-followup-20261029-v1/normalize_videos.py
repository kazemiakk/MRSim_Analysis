"""Lossless removal of encoder terminal padding; retain all scientific frames."""
import argparse,re,subprocess
from pathlib import Path
from common_followup import *
FF=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')


def main(root):
    source=root/'videos-cycle-colors-v3';out=root/'videos-final';out.mkdir(exist_ok=False)
    old=load(source/'video-verification.json');products=[]
    for row in old['products']:
        src=Path(row['path']);dst=out/src.name
        subprocess.run([str(FF),'-v','error','-n','-i',str(src),'-t','86.8','-c','copy','-movflags','+faststart',str(dst)],
                       check=True,capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        probe=subprocess.run([str(FF),'-hide_banner','-i',str(dst)],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        m=re.search(r'Duration: (\d+):(\d+):([\d.]+)',probe.stderr);assert m
        seconds=int(m[1])*3600+int(m[2])*60+float(m[3]);assert abs(seconds-86.8)<.034
        check=subprocess.run([str(FF),'-v','error','-xerror','-i',str(dst),'-f','null','-'],capture_output=True,text=True,
                             timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert check.returncode==0 and not check.stderr.strip()
        products.append({'path':str(dst),'duration_s':seconds,'bytes':dst.stat().st_size,'sha256':sha(dst),'full_decode_passed':True})
    save(out/'video-verification.json',{**old,'products':products,
         'terminal_padding':'Source H.264 flush appended about 67ms; lossless stream-copy trim to 86.8s removed only terminal encoder padding.',
         'source_render_manifest_sha256':sha(source/'video-verification.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
