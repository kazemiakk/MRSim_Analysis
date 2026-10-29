// Isolated Chrome verification of E-029. Saves use only the test service.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {spawn} from 'node:child_process';
const out=path.resolve(process.argv[2]||'outputs/aorta_inlet_picker/e029_v1/validation/browser_v1');
await fs.mkdir(out,{recursive:false});
const profile=await fs.mkdtemp(path.join(os.tmpdir(),'mrsim-e029-browser-'));
const child=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
  '--headless=new','--disable-gpu','--disable-background-networking','--no-first-run',
  '--no-default-browser-check','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
  `--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']});
let stderr='',ws,next=1;child.stderr.on('data',v=>stderr+=v.toString());
const pending=new Map(),events=[],checks={};const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function cdp(method,params={}){const id=next++;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('CDP timeout '+method)),15000);pending.set(id,{resolve:v=>{clearTimeout(timer);resolve(v);},reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function screenshot(name){const r=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(out,name+'.png'),Buffer.from(r.data,'base64'),{flag:'wx'});}
async function state(){return evaluate(`({state:window.mrsimPicker.state,selection:window.mrsimPicker.selection,status:document.querySelector('#status').innerText,overflow:document.documentElement.scrollWidth>innerWidth})`);}
async function click(x,y){await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:1});await sleep(300);}
try{
 let port;for(let i=0;i<60;i++){try{port=(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{}await sleep(200);}if(!port)throw Error('Chrome startup failed');
 const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(['Runtime.exceptionThrown','Network.loadingFailed'].includes(m.method))events.push(m);});
 await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Network.enable');
 await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 await cdp('Page.navigate',{url:'http://127.0.0.1:4176/'});
 for(let i=0;i<50;i++){await sleep(200);if(await evaluate(`!!window.mrsimPicker&&window.mrsimPicker.state.drawCount>0`))break;}
 const initial=await state();checks.source_point_count=initial.state.pointCount===119097;
 checks.no_preselected_inlet=initial.state.selected===null;checks.desktop_no_overflow=!initial.overflow;
 await screenshot('desktop_initial');
 const box=await evaluate(`(()=>{const r=document.querySelector('canvas').getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.5}})()`);
 await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:box.x,y:box.y,button:'left',clickCount:1});
 await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:box.x+85,y:box.y+25,button:'left',buttons:1});
 await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:box.x+85,y:box.y+25,button:'left',clickCount:1});await sleep(300);
 const rotated=await state();checks.drag_rotates_without_selecting=rotated.state.yaw!==initial.state.yaw&&rotated.state.selected===null;
 await evaluate(`document.querySelector('[data-view="front"]').click()`);await sleep(250);
 const target=await evaluate(`(()=>{const r=document.querySelector('canvas').getBoundingClientRect(),s=window.mrsimPicker.state;let best=null;for(let i=0;i<s.pointCount;i+=199){const p=window.mrsimPicker.projectIndex(i);if(p[0]>30&&p[0]<s.width-30&&p[1]>70&&p[1]<s.height-60){const d=(p[0]-s.width/2)**2+(p[1]-s.height/2)**2;if(!best||d<best.d)best={x:r.x+p[0],y:r.y+p[1],d};}}return best;})()`);
 if(!target)throw Error('No visible sample to exercise');await click(target.x,target.y);
 const selected=await state();checks.click_selects_source_sample=Number.isInteger(selected.state.selected)&&selected.state.selected>=0;
 checks.coordinates_populate=await evaluate(`!document.querySelector('#copy').disabled&&document.querySelector('#selection-text').value.includes('source XYZ (mm)')`);
 await screenshot('desktop_selected');
 await evaluate(`document.querySelector('#note').value='Automated browser test, not user inlet';document.querySelector('#save').click()`);await sleep(500);
 const saved=await evaluate(`fetch('/selection').then(r=>r.json())`);
 checks.saved_test_not_user=saved.selection?.purpose==='browser_test_not_user_inlet'&&saved.selection.wall_associated_sample_index===selected.state.selected;
 checks.save_uses_trusted_coordinates=saved.selection?.mri_xyz_mm.every((v,i)=>Math.abs(v-selected.selection.mri_xyz_mm[i])<.00002);
 checks.no_fabricated_plane=saved.selection?.plane_normal===null&&saved.selection.inlet_cap_verified===false;
 const rejected=await evaluate(`(async()=>{const statuses=[];for(const body of [{sample_index:-1},{sample_index:true},{sample_index:0,filename:'../../source.npy'},{sample_index:0,source_xyz_m:[0,0,0]}]){statuses.push((await fetch('/save-selection',{method:'POST',headers:{'Content-Type':'application/json','X-MRSim-Token':LOCAL_RUNTIME.csrf},body:JSON.stringify(body)})).status);}statuses.push((await fetch('/save-selection',{method:'POST',headers:{'Content-Type':'application/json','X-MRSim-Token':'wrong'},body:JSON.stringify({sample_index:0})})).status);return statuses;})()`);
 checks.invalid_payloads_rejected=rejected.slice(0,4).every(s=>s===400)&&rejected[4]===403;
 await evaluate(`document.querySelector('#zoom-in').click()`);await sleep(200);checks.zoom_changes_view=(await state()).state.zoom>1;
 await evaluate(`document.querySelector('[data-view="side"]').click()`);await sleep(200);checks.side_view_works=(await state()).state.viewMode==='side';await screenshot('desktop_side');
 await evaluate(`document.querySelector('#clear').click()`);checks.clear_selection=(await state()).state.selected===null;
 await cdp('Page.navigate',{url:'http://127.0.0.1:4175/'});await sleep(650);
 const production=await evaluate(`fetch('/selection').then(r=>r.json())`);checks.production_selection_untouched=production.selection===null;
 await screenshot('production_ready');
 await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});await sleep(400);
 const narrow=await state();checks.narrow_no_overflow=!narrow.overflow;checks.narrow_geometry_rendered=narrow.state.width>250&&narrow.state.height>300;await screenshot('narrow_ready');
 checks.no_runtime_or_network_errors=events.length===0;
 const result={status:Object.values(checks).every(Boolean)?'PASS':'REVIEW_REQUIRED',created_utc:new Date().toISOString(),checks,
  test_url:'http://127.0.0.1:4176/',production_url:'http://127.0.0.1:4175/',initial,selected,saved_selection:saved.selection,
  rejected_statuses:rejected,narrow,events,scope:'Actual rotate/click/save on test service, coordinate identity, invalid-save guards, production no-selection, desktop/narrow screenshots. Clipboard and download were not exercised.'};
 await fs.writeFile(path.join(out,'browser_checks.json'),JSON.stringify(result,null,2),{flag:'wx'});console.log(JSON.stringify({status:result.status,checks},null,2));
}catch(e){await fs.writeFile(path.join(out,'error.json'),JSON.stringify({error:e.message,events},null,2),{flag:'wx'});console.error(e);process.exitCode=1;}
finally{try{if(ws?.readyState===1)await cdp('Browser.close');}catch{}ws?.close();await sleep(200);if(child.exitCode===null)child.kill();await fs.writeFile(path.join(out,'chrome_stderr.log'),stderr,{flag:'wx'});}
