// Bounded local-file viewer checks; no source images changed.
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const repo=process.cwd(),folder=path.resolve('../../MRSim_All_Images_Review_2026-10-22');
const qa=path.resolve('outputs/image_collection_e044');await mkdir(qa,{recursive:true});
const profile=await mkdtemp(path.join(qa,'profile-'));
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=9333','--remote-debugging-address=127.0.0.1',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
const wait=ms=>new Promise(r=>setTimeout(r,ms));let ws;const checks=[],errors=[];
try{
 let tabs;for(let i=0;i<40;i++){try{tabs=await(await fetch('http://127.0.0.1:9333/json/list')).json();break;}catch{await wait(250);}}
 if(!tabs)throw Error('Browser unavailable');
 ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let n=0;const calls=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(m.id){const c=calls.get(m.id);calls.delete(m.id);m.error?c.reject(Error(JSON.stringify(m.error))):c.resolve(m.result);}};
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;calls.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const check=async(expression,name)=>{if(!await evaluate(expression))throw Error(name);checks.push(name);};
 const loaded=async()=>{for(let i=0;i<100;i++){if(await evaluate('document.getElementById("image")?.complete&&document.getElementById("image").naturalWidth>0'))return;await wait(100);}console.log(JSON.stringify({page:await evaluate('({url:location.href,title:document.title,text:document.body.innerText.slice(0,800),src:document.querySelector("#image")?.getAttribute("src")})'),errors}));throw Error('Image not loaded');};
 await send('Page.enable');await send('Runtime.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:pathToFileURL(path.join(folder,'OPEN_IMAGES.html')).href});await loaded();
 await check('document.querySelector("#summary").textContent.includes("1,548")','All1548 images in direct-file viewer');
 await check('document.querySelector("#image").naturalWidth>0','First original PNG loads');
 await check('document.querySelector("#prev").disabled','First previous boundary');
 await evaluate('document.querySelector("#next").click()');await loaded();await check('document.querySelector("#state").textContent.startsWith("2 /")','Next image');
 await send('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowLeft',code:'ArrowLeft'});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowLeft',code:'ArrowLeft'});await loaded();await check('document.querySelector("#state").textContent.startsWith("1 /")','Keyboard previous');
 await evaluate('document.querySelector("#format").value="GIF";document.querySelector("#format").dispatchEvent(new Event("change"))');await loaded();
 await check('document.querySelector("#status").textContent.includes("26 matching")','All26 GIF copies selectable');
 await check('document.querySelector("#image").getAttribute("src").endsWith(".gif")&&document.querySelector("#detail").textContent.includes("animated")','Original animated GIF loaded without conversion');
 await evaluate('document.querySelector("#format").value="";document.querySelector("#format").dispatchEvent(new Event("change"));document.querySelector("#unique").checked=true;document.querySelector("#unique").dispatchEvent(new Event("change"))');await loaded();
 await check('document.querySelector("#status").textContent.includes("1,192")','Optional unique-content filter');
 await evaluate('document.querySelector("#search").value="unlikely-no-match-xyz";document.querySelector("#search").dispatchEvent(new Event("input"))');await check('document.querySelector("#state").textContent==="No matches"&&document.querySelector("#next").disabled','Empty search remains safe');
 await evaluate('document.querySelector("#unique").checked=false;document.querySelector("#search").value="e043_aorta_branch_views_v3/";document.querySelector("#search").dispatchEvent(new Event("input"))');await loaded();await check('document.querySelector("#status").textContent.includes("16 matching")','Latest aortic folder filter');
 await evaluate('document.querySelector("#fit").click()');await check('document.querySelector("#stage").classList.contains("actual")','Actual-pixel toggle');await evaluate('document.querySelector("#fit").click()');
 await writeFile(path.join(qa,'desktop.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
 await evaluate('document.querySelector("#search").value="";document.querySelector("#search").dispatchEvent(new Event("input"));document.querySelector("#jump").value="1548";document.querySelector("#go").click()');await loaded();await check('document.querySelector("#next").disabled&&document.querySelector("#jump").value==="1548"','ID jump and last boundary');
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:1000,deviceScaleFactor:1,mobile:false});await loaded();await check('document.documentElement.scrollWidth<=window.innerWidth','No narrow-screen page overflow');
 await writeFile(path.join(qa,'narrow.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
 if(errors.length)throw Error('Browser runtime errors: '+JSON.stringify(errors));checks.push('No browser runtime exceptions');
 await writeFile(path.join(qa,'browser_receipt.json'),JSON.stringify({status:'PASS',count:checks.length,checks,fileMode:true,sourceImagesChanged:false},null,2));console.log('PASS: '+checks.length+' local-viewer checks');
}finally{ws?.close();browser.kill();}
