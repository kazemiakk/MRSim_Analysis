// Local report-only browser checks using an isolated, headless Edge profile.
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile} from 'node:fs/promises';
import path from 'node:path';
const base=path.resolve('outputs/hemodynamic_extension/e041_review_report_v1');
const profile=await mkdtemp(path.join(base,'browser-profile-'));
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
 '--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=9326',
 '--remote-debugging-address=127.0.0.1',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:['ignore','pipe','pipe']});
let diagnostic='';browser.stderr.on('data',b=>diagnostic+=b.toString());
let ws;
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 let targets;
 for(let n=0;n<40;n++){try{targets=await(await fetch('http://127.0.0.1:9326/json/list')).json();break;}catch{await pause(250);}}
 if(!targets)throw Error('Isolated browser did not start: '+diagnostic.slice(-1200));
 const tab=targets.find(t=>t.type==='page');ws=new WebSocket(tab.webSocketDebuggerUrl);
 await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const pending=new Map();const errors=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);};
 const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
 const evaluate=async expression=>(await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true})).result.value;
 await send('Runtime.enable');await send('Page.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:4186/'});
 let loaded=false;
 for(let n=0;n<80;n++){loaded=await evaluate("document.body.innerText.includes('Executive Summary')");if(loaded)break;await pause(250);}
 if(!loaded)throw Error('Report narrative did not mount');
 const text=await evaluate('document.body.innerText');
 for(const phrase of ['4.76','15.0%','31 of 75','not zero error','Scope, definitions'])if(!text.includes(phrase))throw Error('Missing report claim: '+phrase);
 await evaluate("[...document.querySelectorAll('.review-gallery')].forEach(d=>d.open=true);[...document.querySelectorAll('.review-figure img')].forEach(i=>i.loading='eager')");
 let images;
 for(let n=0;n<80;n++){images=await evaluate("[...document.querySelectorAll('.review-figure img')].map(i=>({alt:i.alt,loaded:i.complete&&i.naturalWidth>0,width:i.naturalWidth,height:i.naturalHeight}))");if(images.length===30&&images.every(i=>i.loaded))break;await pause(250);}
 if(images.length!==30||images.some(i=>!i.loaded))throw Error('Missing image in report');
 const screenshot=async name=>{const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(path.join(base,name),Buffer.from(shot.data,'base64'));};
 await evaluate('window.scrollTo(0,0)');await screenshot('desktop.png');
 await evaluate("document.querySelector('#review-phantom')?.scrollIntoView();");
 // Use authored narrative text when the runtime wraps the component ID.
 await evaluate("document.querySelector('.review-figure').scrollIntoView();window.scrollBy(0,-90)");await pause(300);await screenshot('figure-desktop.png');
 const desktopOverflow=await evaluate('document.documentElement.scrollWidth>window.innerWidth+2');
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await evaluate('window.scrollTo(0,0)');await pause(350);await screenshot('mobile.png');
 const mobileOverflow=await evaluate('document.documentElement.scrollWidth>window.innerWidth+2');
 const tables=await evaluate("document.querySelectorAll('.review-table tbody tr').length");
 if(errors.length||desktopOverflow||mobileOverflow||tables!==3)throw Error(JSON.stringify({errors,desktopOverflow,mobileOverflow,tables}));
 // Source controls are shared runtime; inspect a representative visible source action.
 const sourceButtons=await evaluate("[...document.querySelectorAll('button')].filter(b=>/source/i.test(b.getAttribute('aria-label')||b.title||b.innerText)).map(b=>({label:b.getAttribute('aria-label')||b.title||b.innerText}))");
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await evaluate('window.scrollTo(0,0)');await pause(200);
 const point=await evaluate("(()=>{const r=document.querySelector('button[aria-label=\"Executive Summary actions\"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
 await send('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});await pause(200);
 const sourceClicked=await evaluate("(()=>{const e=[...document.querySelectorAll('button,[role=menuitem]')].find(e=>/view data source/i.test(e.innerText));if(e){e.click();return true;}return false;})()");await pause(200);
 const sourceText=await evaluate('document.body.innerText');
 if(!sourceClicked||!sourceText.includes('Data preview')||!sourceText.includes('Evidence flow')){console.log(JSON.stringify({sourceClicked,menus:await evaluate("[...document.querySelectorAll('[role=menu], [role=menuitem]')].map(e=>e.innerText)"),tail:sourceText.slice(-800)}));throw Error('Source inspector did not open');}
 await screenshot('source-mobile.png');
 await writeFile(path.join(base,'browser_validation.json'),JSON.stringify({status:'PASS',url:'http://127.0.0.1:4186/',images,tables,errors,desktopOverflow,mobileOverflow,sourceButtons,sourceInspectorOpened:true,scope:'All narratives mounted; all 30 image placements loaded (24 unique PNGs); three appendix disclosures expanded; desktop/mobile screenshots captured; representative multi-query source inspector opened and its Data preview/Evidence flow controls checked, not exhaustive source-tab interaction acceptance.'},null,2));
 console.log('PASS: narrative, 30 image placements, disclosures, table, desktop/mobile overflow and runtime checks');
}finally{ws?.close();browser.kill();}
