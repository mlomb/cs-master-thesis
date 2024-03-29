var x=Object.defineProperty;var U=(e,t,r)=>t in e?x(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var h=(e,t,r)=>(U(e,typeof t!="symbol"?t+"":t,r),r);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))s(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const n of o.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&s(n)}).observe(document,{childList:!0,subtree:!0});function r(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function s(i){if(i.ep)return;i.ep=!0;const o=r(i);fetch(i.href,o)}})();class M{constructor(t){h(this,"feature_set","");h(this,"num_features",0);h(this,"layers",[]);const r=new Y(t);switch(this.feature_set=r.readString(),this.feature_set){case"half-compact":this.num_features=192;break;case"half-piece":this.num_features=768;break;case"half-king-piece":this.num_features=40960;break}const s=256,i=32,o=32;this.layers.push(b.read(r,this.num_features,s,16,16)),this.layers.push(b.read(r,2*s,i,8,32)),this.layers.push(b.read(r,i,o,8,32)),this.layers.push(b.read(r,o,1,8,32)),console.assert(r.isEOF(),"Failed to read entire NN file")}get ft(){return this.layers[0]}static async load(t){let s=await(await fetch(t)).arrayBuffer();return new M(s)}}class b{constructor(){h(this,"num_inputs");h(this,"num_outputs");h(this,"weight");h(this,"bias")}static read(t,r,s,i,o){let n=new b;return n.num_inputs=r,n.num_outputs=s,n.weight=t.readIntArray(r*s,i),n.bias=t.readIntArray(s,o),n}getWeightRow(t){let r=new Array(this.num_inputs);for(let s=0;s<this.num_inputs;s++)r[s]=this.weight[s*this.num_outputs+t];return r}getWeightFT(t,r){return this.weight[t*this.num_outputs+r]}}class Y{constructor(t){h(this,"dv");h(this,"offset",0);this.dv=new DataView(t)}readString(){let t="";for(;this.dv.getUint8(this.offset)!==0;)t+=String.fromCharCode(this.dv.getUint8(this.offset++));return this.offset++,t}readInt8Array(t){let r=new Int8Array(t);for(let s=0;s<t;s++)r[s]=this.dv.getInt8(this.offset++);return r}readInt16Array(t){let r=new Int16Array(t);for(let s=0;s<t;s++)r[s]=this.dv.getInt16(this.offset,!0),this.offset+=2;return r}readInt32Array(t){let r=new Int32Array(t);for(let s=0;s<t;s++)r[s]=this.dv.getInt32(this.offset,!0),this.offset+=4;return r}readIntArray(t,r){switch(r){case 8:return this.readInt8Array(t);case 16:return this.readInt16Array(t);case 32:return this.readInt32Array(t)}}isEOF(){return this.offset===this.dv.byteLength}}const k=["w","b"],O=["P","N","B","R","Q","K"],N=new Map,P=new Map;function D(e,t,r,s,i){let o=4,n=15,p=5;for(let g=0;g<s.num_outputs;g++){let d=t+g%o*n,f=r+Math.floor(g/o)*n;if(g==i){e.lineWidth=1;for(let u=0;u<768;u++){if(!N.has(u+""))continue;let[w,E]=N.get(u+""),A=s.getWeightFT(u,i),S="rgba("+(A>0?"0, 255, 0":"255, 0, 0")+", "+Math.abs(A/300)+")";e.strokeStyle=S,e.beginPath(),e.moveTo(w,E),e.lineTo(d,f),e.stroke()}}e.fillStyle="rgba(255, 255, 255, 0.2)",e.strokeStyle="white",e.lineWidth=1,e.beginPath(),e.arc(d,f,p,0,2*Math.PI),e.stroke(),e.fill()}}const l=16;function X(e,t,r,s){console.assert(s.length==64,"Invalid number of cells");let i="#f0d9b5",o="#b58863";for(let n=0;n<64;n++){let p=n%8,g=Math.floor(n/8),d=p*l+t,f=g*l+r;e.fillStyle=(p+g)%2==0?i:o,e.fillRect(d,f,l,l);let{role:u,color:w,opacity:E,text:A,accent:S,coords_key:T}=s[n];if(S&&(e.fillStyle=S,e.fillRect(d,f,l,l)),u&&w){let q=w+u,F=P.get(q);F&&(e.globalAlpha=E===void 0?1:E,e.drawImage(F,d,f,l,l),e.globalAlpha=1)}A&&(e.textAlign="left",e.font="4px Arial",e.fillStyle="white",e.fillText(`${A}`,d+1,f+l-1)),T&&N.set(T,[d+l/2,f+l/2])}}function $(e,t,r){for(let s=0;s<12;s++){let i=k[+(s>=6)],o=O[s%6],n=s*150;X(e,0,n,Array.from({length:64},(g,d)=>{let f=64*s+d,u=t.getWeightFT(f,r),w=Math.abs(u/300);return{text:`${f}`,accent:"rgba("+(u>0?"0, 255, 0":"255, 0, 0")+", "+w+")",coords_key:`${f}`,color:i,role:O[s%6],opacity:w}}));let p=P.get(i+o);p&&e.drawImage(p,-64,n+8*l/2-32/2,32,32)}e.font="20px monospace",e.textAlign="center",e.fillStyle="white",e.fillText("Half-Piece",l*8/2,-30),e.fillText("[768]",l*8/2,-10)}async function B(){for(let e of k)for(let t of O){let r=new Image;r.src=`pieces/${e}${t}.svg`,P.set(e+t,r)}}B();const c=document.querySelector("canvas"),v=c.getContext("2d");let I=null;M.load("best.nn").then(e=>{I=e,m=2,console.log(I)});async function H(){v.imageSmoothingEnabled=!1,I&&(D(v,400,0,I.ft,3),$(v,I.ft,3))}function z(e,t,r){console.log("onClick",e,t,r)}let m=1,_=1,y={x:0,y:0},R=!1;function K(e){(e.x!==0||e.y!==0)&&(R=!0,m=Math.max(m,1),y.x+=e.x,y.y+=e.y)}function C(e,t){_*=t,y.x=e.x-(e.x-y.x)*t,y.y=e.y-(e.y-y.y)*t,m=Math.max(m,1)}async function W(){m>0&&(m--,c.width=window.innerWidth,c.height=window.innerHeight,v.setTransform(1,0,0,1,0,0),v.clearRect(0,0,c.width,c.height),v.setTransform(_,0,0,_,y.x,y.y),await H()),requestAnimationFrame(W)}window.onresize=()=>Math.max(m,1);c.oncontextmenu=e=>e.preventDefault();c.addEventListener("mousemove",L);c.addEventListener("mousedown",L);c.addEventListener("mouseup",L);c.addEventListener("mouseout",L);c.addEventListener("wheel",Q,{passive:!1});const a={x:0,y:0,oldX:0,oldY:0,button:!1};function L(e){if(e.type==="mousedown"&&(a.button=!0,R=!1),e.type==="mouseup"||e.type==="mouseout"){if(a.button&&R==!1){const t=1/_;z(Math.ceil((a.x-y.x)*t),Math.ceil((a.y-y.y)*t),e.button)}a.button=!1}a.oldX=a.x,a.oldY=a.y,a.x=e.offsetX,a.y=e.offsetY,a.button&&K({x:a.x-a.oldX,y:a.y-a.oldY})}function Q(e){var t=e.offsetX,r=e.offsetY;e.deltaY<0?C({x:t,y:r},1.1):C({x:t,y:r},1/1.1),e.preventDefault()}requestAnimationFrame(W);
