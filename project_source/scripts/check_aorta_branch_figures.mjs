// Inspect native scientific PNGs through a dedicated browser; originals unchanged.
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
const run=process.argv[2]||'e043_aorta_branch_views_v1';
if(!/^e043_aorta_branch_views_v\d+$/.test(run))throw Error('Unexpected run');
const parent=path.resolve('outputs/hemodynamic_extension');
const out=path.join(parent,run+'_visual_review');await mkdir(out,{recursive:true});
const profile=await mkdtemp(path.join(out,'profile-'));
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=9331','--remote-debugging-address=127.0.0.1',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws;
try{
 let tabs;for(let i=0;i<40;i++){try{tabs=await(await fetch('http://127.0.0.1:9331/json/list')).json();break;}catch{await wait(250);}}
 if(!tabs)throw Error('Isolated browser unavailable');
 ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let n=0;const calls=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const c=calls.get(m.id);calls.delete(m.id);m.error?c.reject(Error(JSON.stringify(m.error))):c.resolve(m.result);}};
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;calls.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>(await send('Runtime.evaluate',{expression,returnByValue:true})).result.value;
 await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1250,deviceScaleFactor:1,mobile:false});
 const report=await(await fetch(`http://127.0.0.1:4189/${run}/report.json`)).json();
 const checked=[];
 for(const item of report.inventory){
  const url=`http://127.0.0.1:4189/${run}/${item.stem}.png`;await send('Page.navigate',{url});
  let loaded=false;for(let i=0;i<80;i++){loaded=await evaluate('!!document.querySelector("img")?.complete && document.querySelector("img").naturalWidth>0');if(loaded)break;await wait(100);}
  if(!loaded)throw Error('Image failed '+url);
  const dims=await evaluate('(()=>{document.body.style="margin:0;background:white;display:flex;align-items:center;justify-content:center;width:100vw;height:100vh";const i=document.querySelector("img");i.style="max-width:100vw;max-height:100vh;width:auto;height:auto;display:block";return {width:i.naturalWidth,height:i.naturalHeight}})()');
  await wait(80);const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(path.join(out,item.stem+'.png'),Buffer.from(shot.data,'base64'));checked.push({file:item.stem+'.png',...dims});
 }
 await writeFile(path.join(out,'receipt.json'),JSON.stringify({status:'CAPTURED_FOR_VISUAL_REVIEW',count:checked.length,checked,originalsUnchanged:true},null,2));console.log('Captured '+checked.length+' native-PNG browser views');
}finally{ws?.close();browser.kill();}
