"""Decode the actual MP4s and preserve representative encoded-frame montages."""
from pathlib import Path
import sys,subprocess,re,json,csv
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import PROJECT,load,save,sha

out=PROJECT/'results/20261029-virtual-angiography-connected-250um-v1';folder=out/'visualization'
ffmpeg=Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')
protocol=load(out/'protocol.json')
if not (folder/'sampling-definition-native-reference.json').exists():
    path=folder/'sampling-definition.json';definition=load(path);path.rename(folder/'sampling-definition-native-reference.json')
    definition.update(projection='Sum C times display spacing along model-coordinate Y; not registered clinical AP or calibrated x-ray attenuation',
        native_flow_cells=protocol['native_flow_cells'],scalar_transport_cells=protocol['scalar_transport_cells'],
        data_not_changed='Display sampling only; original flow domain preserved. Scalar concentration is reconstructed from connected agglomerates onto native cells.')
    save(path,definition)
records=[]
for side in ['right','left']:
    video=folder/(side+'-ICA-virtual-dye.mp4')
    command=[str(ffmpeg),'-hide_banner','-i',str(video),'-map','0:v:0','-f','null','-']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    with (folder/(side+'-decode-validation.log')).open('x') as f:f.write(result.stderr)
    frames=[int(v) for v in re.findall(r'frame=\s*(\d+)',result.stderr)]
    assert result.returncode==0 and frames[-1]==240 and '1600x900' in result.stderr and '00:00:10.00' in result.stderr
    montage=folder/(side+'-decoded-video-montage.png');assert not montage.exists()
    filter=r'select=eq(n\,28)+eq(n\,55)+eq(n\,110)+eq(n\,239),scale=800:450,tile=2x2'
    checked=subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-n','-i',str(video),'-vf',filter,
        '-frames:v','1','-update','1',str(montage)],capture_output=True,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    assert checked.returncode==0 and montage.is_file(),checked.stderr
    records.append({'injection':side,'file':str(video),'sha256':sha(video),'frames_decoded':frames[-1],
        'duration_s':10,'resolution':[1600,900],'full_decode_passed':True,'montage_from_encoded_file':str(montage)})
save(folder/'video-review.json',{'status':'decoded_successfully_visual_review_pending','videos':records,
    'display_metadata_corrected':'Native-reference metadata retained; effective metadata now states connected scalar preview and model-coordinate projection',
    'qualification':'Qualitative preview; spatial and temporal transport sensitivity remain material. Not clinical DSA or validated clearance.'})
print('BOTH_MP4S_FULLY_DECODED',flush=True)
