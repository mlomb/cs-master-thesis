var Y=Object.defineProperty;var B=(e,t,r)=>t in e?Y(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var c=(e,t,r)=>(B(e,typeof t!="symbol"?t+"":t,r),r);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))i(s);new MutationObserver(s=>{for(const o of s)if(o.type==="childList")for(const n of o.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&i(n)}).observe(document,{childList:!0,subtree:!0});function r(s){const o={};return s.integrity&&(o.integrity=s.integrity),s.referrerPolicy&&(o.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?o.credentials="include":s.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function i(s){if(s.ep)return;s.ep=!0;const o=r(s);fetch(s.href,o)}})();class R{constructor(t){c(this,"feature_set","");c(this,"num_features",0);c(this,"layers",[]);const r=new X(t);switch(this.feature_set=r.readString(),this.feature_set){case"half-compact":this.num_features=192;break;case"half-piece":this.num_features=768;break;case"half-king-piece":this.num_features=40960;break}const i=256,s=32,o=32;this.layers.push(E.read(r,this.num_features,i,16,16)),this.layers.push(E.read(r,2*i,s,8,32)),this.layers.push(E.read(r,s,o,8,32)),this.layers.push(E.read(r,o,1,8,32)),console.assert(r.isEOF(),"Failed to read entire NN file")}get ft(){return this.layers[0]}static async load(t){let i=await(await fetch(t)).arrayBuffer();return new R(i)}}class E{constructor(){c(this,"num_inputs");c(this,"num_outputs");c(this,"weight");c(this,"bias")}static read(t,r,i,s,o){let n=new E;return n.num_inputs=r,n.num_outputs=i,n.weight=t.readIntArray(r*i,s),n.bias=t.readIntArray(i,o),n}getWeightRow(t){let r=new Array(this.num_inputs);for(let i=0;i<this.num_inputs;i++)r[i]=this.weight[i*this.num_outputs+t];return r}getWeightFT(t,r){return this.weight[t*this.num_outputs+r]}}class X{constructor(t){c(this,"dv");c(this,"offset",0);this.dv=new DataView(t)}readString(){let t="";for(;this.dv.getUint8(this.offset)!==0;)t+=String.fromCharCode(this.dv.getUint8(this.offset++));return this.offset++,t}readInt8Array(t){let r=new Int8Array(t);for(let i=0;i<t;i++)r[i]=this.dv.getInt8(this.offset++);return r}readInt16Array(t){let r=new Int16Array(t);for(let i=0;i<t;i++)r[i]=this.dv.getInt16(this.offset,!0),this.offset+=2;return r}readInt32Array(t){let r=new Int32Array(t);for(let i=0;i<t;i++)r[i]=this.dv.getInt32(this.offset,!0),this.offset+=4;return r}readIntArray(t,r){switch(r){case 8:return this.readInt8Array(t);case 16:return this.readInt16Array(t);case 32:return this.readInt32Array(t)}}isEOF(){return this.offset===this.dv.byteLength}}const W=["w","b"],x=["P","N","B","R","Q","K"],k=new Map,F=new Map;function G(e,t,r,i,s){let o=4,n=15,A=5,_=t+s%o*n,m=r+Math.floor(s/o)*n;for(let l=0;l<768;l++){if(!k.has(l+""))continue;let[g,p]=k.get(l+""),S=i.getWeightFT(l,s),v=Math.abs(S/300),L="rgba("+(S>0?"0, 255, 0":"255, 0, 0")+", "+v+")";v<=.03||(e.strokeStyle=L,e.beginPath(),e.moveTo(g,p),e.lineTo(_,m),e.stroke())}for(let l=0;l<i.num_outputs;l++){let g=t+l%o*n,p=r+Math.floor(l/o)*n;l==s?(e.fillStyle="gray",e.strokeStyle="white"):(e.fillStyle="#454545",e.strokeStyle="#6A6A6A"),e.lineWidth=1,e.beginPath(),e.arc(g,p,A,0,2*Math.PI),e.closePath(),e.stroke(),e.fill()}e.lineWidth=1}const f=16;function K(e,t,r,i){console.assert(i.length==64,"Invalid number of cells");let s="#f0d9b5",o="#b58863";for(let n=0;n<64;n++){let A=n%8,_=Math.floor(n/8),m=A*f+t,l=_*f+r;e.fillStyle=(A+_)%2==0?s:o,e.fillRect(m,l,f,f);let{role:g,color:p,opacity:S,text:v,accent:L,coords_key:P}=i[n];if(L&&(e.fillStyle=L,e.fillRect(m,l,f,f)),g&&p){let D=p+g,H=F.get(D);if(H){let C=S===void 0?1:S;C>.03&&(e.globalAlpha=C,e.drawImage(H,m,l,f,f),e.globalAlpha=1)}}v&&(e.textAlign="left",e.font="4px Arial",e.fillStyle="white",e.fillText(`${v}`,m+1,l+f-1)),P&&k.set(P,[m+f/2,l+f/2])}}function z(e,t,r){for(let i=0;i<12;i++){let s=W[+(i>=6)],o=x[i%6],n=i*150;K(e,0,n,Array.from({length:64},(_,m)=>{let l=64*i+m,g=t.getWeightFT(l,r),p=Math.abs(g/300);return{text:`${l}`,accent:"rgba("+(g>0?"0, 255, 0":"255, 0, 0")+", "+p+")",coords_key:`${l}`,color:s,role:x[i%6],opacity:p}}));let A=F.get(s+o);A&&e.drawImage(A,-64,n+8*f/2-32/2,32,32)}e.font="20px monospace",e.textAlign="center",e.fillStyle="white",e.fillText("Half-Piece",f*8/2,-30),e.fillText("[768]",f*8/2,-10)}async function Q(){for(let e of W)for(let t of x){let r=new Image;r.src=`pieces/${e}${t}.svg`,F.set(e+t,r)}}Q();const h=document.querySelector("canvas"),u=h.getContext("2d");let b=null,I=0,q=0,w=0,U=0;R.load("best.nn").then(e=>{console.log(b),b=e,y=2,I=0,q=0,w=0,U=b.ft.num_outputs-1});function V(){u.textAlign="left",u.textBaseline="top",u.font="30px monospace",u.fillStyle="white",u.fillText(`↑↓ Step: ${I}`,10,10),u.fillText(`←→ L1 neuron: ${w}`,10,45),u.fillText("(shift x10)",10,80)}async function Z(){u.imageSmoothingEnabled=!1,b&&(G(u,400,1800/2-960/2,b.ft,w),z(u,b.ft,w))}function j(e,t,r){console.log("onClick",e,t,r)}window.addEventListener("keydown",e=>{if(!b)return;let t=1*(e.shiftKey?10:1);e.key==="ArrowUp"&&(I=Math.min(I+t,q)),e.key==="ArrowDown"&&(I=Math.max(I-t,0)),e.key==="ArrowLeft"&&(w=Math.max(w-t,0)),e.key==="ArrowRight"&&(w=Math.min(w+t,U)),y=Math.max(y,1)});let y=1,T=1,d={x:0,y:0},O=!1;function J(e){(e.x!==0||e.y!==0)&&(O=!0,y=Math.max(y,1),d.x+=e.x,d.y+=e.y)}function N(e,t){T*=t,d.x=e.x-(e.x-d.x)*t,d.y=e.y-(e.y-d.y)*t,y=Math.max(y,1)}async function $(){y>0&&(y--,h.width=window.innerWidth,h.height=window.innerHeight,u.clearRect(0,0,h.width,h.height),u.setTransform(T,0,0,T,d.x,d.y),await Z(),u.setTransform(1,0,0,1,0,0),V()),requestAnimationFrame($)}window.onresize=()=>Math.max(y,1);h.oncontextmenu=e=>e.preventDefault();h.addEventListener("mousemove",M);h.addEventListener("mousedown",M);h.addEventListener("mouseup",M);h.addEventListener("mouseout",M);h.addEventListener("wheel",ee,{passive:!1});const a={x:0,y:0,oldX:0,oldY:0,button:!1};function M(e){if(e.type==="mousedown"&&(a.button=!0,O=!1),e.type==="mouseup"||e.type==="mouseout"){if(a.button&&O==!1){const t=1/T;j(Math.ceil((a.x-d.x)*t),Math.ceil((a.y-d.y)*t),e.button)}a.button=!1}a.oldX=a.x,a.oldY=a.y,a.x=e.offsetX,a.y=e.offsetY,a.button&&J({x:a.x-a.oldX,y:a.y-a.oldY})}function ee(e){var t=e.offsetX,r=e.offsetY;e.deltaY<0?N({x:t,y:r},1.1):N({x:t,y:r},1/1.1),e.preventDefault()}requestAnimationFrame($);
