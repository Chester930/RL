"""
Post-processes course_slides_preview.html to inject:
  - JS training-curve animations for all 7 RL algorithms
  - Laser pointer tool  (L key or toolbar button)
  - Drawing/annotation  (D key or toolbar button)
  - Eraser              (E key or toolbar button)
  - Cross-window sync via BroadcastChannel + localStorage fallback
    (animations triggered from presenter view play on the main window)

Usage:
    python plots/inject_animations.py
"""
import json, os, re

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, 'course_slides_preview.html')

# ── animation data ────────────────────────────────────────────────────────────
ANIM = {
    6: {
        "title": "BC vs SAC：分布偏移測試",
        "type": "grouped_bar",
        "labels": ["0°","30°","60°","90°","120°","150°","180°"],
        "datasets": [
            {"label":"SAC Expert","values":[0,-132.2,-127.1,-123.4,-122.2,-231.1,-353.5],"color":"#2ecc71"},
            {"label":"BC Policy", "values":[0,-132.7,-127.3,-122.2,-245.0,-231.3,-436.5],"color":"#e74c3c"},
        ],
        "xLabel":"初始角度（0°=直立，180°=朝下）","yLabel":"平均回報",
        "yMin":-490,"yMax":40,"shadeFrom":4,"note":"120°+ 開始崩潰",
    },
    10: {
        "title": "Q-table 維度詛咒",
        "type": "bar_log",
        "labels":["1關節","2關節","3關節","4關節","5關節","6關節","7關節","8關節"],
        "values":[30,900,27000,810000,24300000,729000000,21870000000,656100000000],
        "color":"#3498db",
        "xLabel":"關節數量","yLabel":"Q-table 大小（log scale）",
        "annotations":{"5":"機器手臂\n(6 DOF)","7":"❌ 完全不可行"},
    },
    13: {
        "title": "DQN 學習曲線（CartPole-v1）",
        "type": "line",
        "xs":[10000,30000,50000,100000,150000],
        "datasets":[{"label":"DQN eval","ys":[30,115,28.8,106.7,500.0],"color":"#3498db"}],
        "xLabel":"訓練步數","yLabel":"平均回報","yMin":0,"yMax":560,
        "hlines":[{"y":500,"label":"滿分 500","dash":True}],
        "milestone":{"xi":4,"text":"150K → 滿分 500！🎉","color":"#27ae60"},
    },
    17: {
        "title": "DDPG 訓練曲線（Pendulum-v1）",
        "type": "line_band",
        "xs":[10000,20000,30000,50000,80000,100000,130000,170000,200000],
        "datasets":[{
            "label":"DDPG eval","color":"#9b59b6",
            "mean":[-164.9,-212.7,-113.6,-136.4,-195.1,-101.6,-120.9,-150.4,-121.1],
            "std": [126.7,  93.1, 103.6,  64.3,  76.5,  72.0,  49.8,  42.4,  72.8],
        }],
        "xLabel":"訓練步數","yLabel":"平均回報","yMin":-380,"yMax":80,
        "hlines":[{"y":-100,"label":"近似最優（-100）","dash":True}],
        "milestone":{"xi":5,"text":"峰值 -101.6\n但方差極高","color":"#9b59b6"},
    },
    20: {
        "title": "TD3 vs DDPG（Pendulum-v1）",
        "type": "multi_line",
        "datasets":[
            {"label":"DDPG","color":"#9b59b6",
             "xs":[10000,20000,30000,50000,80000,100000,130000,200000],
             "ys":[-164.9,-212.7,-113.6,-136.4,-195.1,-101.6,-120.9,-121.1]},
            {"label":"TD3","color":"#1abc9c",
             "xs":[10000,20000,40000,70000,80000,100000,150000,200000],
             "ys":[-566.6,-136.9,-122.5,-119.8,-123.2,-120.8,-155.1,-122.8]},
        ],
        "xLabel":"訓練步數","yLabel":"平均回報","yMin":-620,"yMax":60,
        "hlines":[{"y":-100,"label":"近似最優（-100）","dash":True}],
        "milestone":{"dsi":1,"xi":1,"text":"TD3 從 20K 快速收斂","color":"#1abc9c"},
    },
    23: {
        "title": "SAC：LunarLander + 自動溫度調節",
        "type": "dual_line",
        "panels":[
            {
                "label":"SAC LunarLander eval",
                "xs":[25000,50000,75000,100000,125000,150000],
                "ys":[-117.2,160.5,182.9,262.4,198.9,215.7],
                "color":"#f39c12",
                "xLabel":"訓練步數","yLabel":"平均回報","yMin":-150,"yMax":300,
                "hline":{"y":200,"label":"過關標準 200"},
                "milestone":{"xi":1,"text":"50K 突破 200！\n(PPO 需 163K)"},
            },
            {
                "label":"Alpha（溫度）",
                "xs":[10000,20000,30000,40000,100000],
                "ys":[0.3211,0.1255,0.0471,0.0261,0.0201],
                "color":"#e74c3c",
                "xLabel":"訓練步數（Pendulum）","yLabel":"Alpha 值","yMin":0,"yMax":0.36,
                "milestone":{"xi":4,"text":"自動降至 0.02\n無需手動調"},
            },
        ],
    },
    27: {
        "title": "REINFORCE：高方差 → 無法收斂",
        "type": "line",
        "xs":[500,1000,1500,2000,2500,3000,3500,4000,4500,5000],
        "datasets":[{"label":"REINFORCE eval","ys":[9.6,9.7,9.1,9.1,9.0,9.6,9.2,9.2,27.3,9.5],"color":"#e74c3c"}],
        "xLabel":"訓練集數 (Episodes)","yLabel":"平均回報","yMin":0,"yMax":550,
        "hlines":[
            {"y":195,"label":"官方標準 (195)","dash":True},
            {"y":500,"label":"滿分 (500)","dash":True},
        ],
        "milestone":{"xi":9,"text":"5000集後仍 ~9\n完全沒有學習！","color":"#e74c3c"},
    },
    30: {
        "title": "REINFORCE vs PPO（CartPole-v1）",
        "type": "multi_line",
        "datasets":[
            {"label":"REINFORCE","color":"#e74c3c",
             "xs":[5000,10000,15000,20000,25000,30000,35000,40000,45000,50000],
             "ys":[9.6,9.7,9.1,9.1,9.0,9.6,9.2,9.2,27.3,9.5]},
            {"label":"PPO","color":"#2ecc71",
             "xs":[20480,40960,61440,81920,102400,122880,143360],
             "ys":[500,500,500,500,500,500,500]},
        ],
        "xLabel":"訓練步數","yLabel":"平均回報","yMin":-20,"yMax":560,
        "hlines":[{"y":195,"label":"官方標準 (195)","dash":True}],
        "milestone":{"dsi":1,"xi":0,"text":"20K → 滿分 500！\nClip 機制生效","color":"#2ecc71"},
    },
}

# ── shared style pieces ────────────────────────────────────────────────────────
_B = ("background:transparent;color:#bbb;border:1px solid transparent;"
      "padding:5px 10px;border-radius:16px;font-size:13px;cursor:pointer;"
      "transition:background .15s,color .15s,border-color .15s;"
      "white-space:nowrap;font-family:'Microsoft JhengHei',sans-serif;")
_S = '<div style="width:1px;height:22px;background:rgba(255,255,255,0.15);margin:0 2px;"></div>'

# ── overlay HTML ──────────────────────────────────────────────────────────────
OVERLAY_HTML = f"""<!-- RL Animations injected by inject_animations.py -->
<div id="rl-overlay" style="display:none;position:fixed;inset:0;background:rgba(10,15,35,0.96);z-index:9984;flex-direction:column;align-items:center;justify-content:flex-start;padding:16px 20px;">
  <div style="width:100%;max-width:1200px;display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <span id="rl-title" style="color:#f0f0f0;font-size:19px;font-family:'Microsoft JhengHei',sans-serif;font-weight:bold;"></span>
    <button onclick="rlClose()" style="background:#e74c3c;color:white;border:none;width:34px;height:34px;border-radius:50%;font-size:16px;cursor:pointer;line-height:1;">✕</button>
  </div>
  <canvas id="rl-canvas" width="1160" height="530" style="background:#f8f9fa;border-radius:8px;max-width:100%;"></canvas>
  <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;justify-content:center;">
    <button id="rl-playpause" onclick="rlTogglePlay()" style="background:#27ae60;color:white;border:none;padding:9px 22px;border-radius:22px;font-size:15px;cursor:pointer;font-family:'Microsoft JhengHei',sans-serif;min-width:90px;">▶ 播放</button>
    <button onclick="rlRestart()" style="background:#2980b9;color:white;border:none;padding:9px 22px;border-radius:22px;font-size:15px;cursor:pointer;font-family:'Microsoft JhengHei',sans-serif;">↺ 重播</button>
    <label style="color:#ccc;font-family:'Microsoft JhengHei',sans-serif;font-size:14px;display:flex;align-items:center;gap:8px;">速度 <input type="range" id="rl-speed-input" min="0.5" max="4" step="0.5" value="1" oninput="rlSetSpeed(this.value)" style="width:80px;"><span id="rl-speed-lbl" style="min-width:26px;color:#aaa;">1×</span></label>
  </div>
</div>
<canvas id="rl-laser-cv" style="display:none;position:fixed;top:0;left:0;pointer-events:none;z-index:9982;"></canvas>
<canvas id="rl-draw-cv"  style="display:none;position:fixed;top:0;left:0;pointer-events:none;z-index:9980;cursor:crosshair;"></canvas>
<div id="rl-toolbar" style="position:fixed;bottom:70px;left:20px;z-index:9988;display:none;gap:5px;align-items:center;background:rgba(18,22,38,0.88);border:1px solid rgba(255,255,255,0.1);padding:6px 12px;border-radius:28px;backdrop-filter:blur(14px);box-shadow:0 4px 24px rgba(0,0,0,0.45);">
  <button id="rl-tb-anim"   onclick="rlTbAnim()"          title="動態播放"   style="{_B}opacity:0.4;">📊 動態</button>
  {_S}
  <button id="rl-tb-laser"  onclick="rlSetMode('laser')"  title="雷射筆 (L)" style="{_B}">🔴 雷射</button>
  <button id="rl-tb-draw"   onclick="rlSetMode('draw')"   title="畫筆 (D)"   style="{_B}">✏️ 畫筆</button>
  <button id="rl-tb-eraser" onclick="rlSetMode('eraser')" title="橡皮擦 (E)" style="{_B}">🧹 橡皮</button>
  {_S}
  <input type="color" id="rl-color" value="#ff3333" title="顏色" style="width:26px;height:26px;border-radius:50%;border:2px solid rgba(255,255,255,0.25);cursor:pointer;padding:0;vertical-align:middle;">
  <input type="range" id="rl-size"  min="2" max="24" value="5" title="筆觸大小" style="width:60px;vertical-align:middle;">
  {_S}
  <button id="rl-tb-clear" onclick="rlClearDraw()" title="清除畫布 (C)" style="{_B}">🗑️ 清除</button>
</div>
<div id="rl-toast" style="display:none;position:fixed;bottom:70px;left:50%;transform:translateX(-50%);background:rgba(39,174,96,0.92);color:white;padding:10px 22px;border-radius:24px;font-size:15px;font-family:'Microsoft JhengHei',sans-serif;z-index:9986;transition:opacity .4s;pointer-events:none;white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,0.3);"></div>
"""

# ── JavaScript ────────────────────────────────────────────────────────────────
JS_TEMPLATE = r"""
<script>
(function(){
'use strict';
var ANIM=__DATA__;
var FONT="14px 'Microsoft JhengHei','Microsoft YaHei',sans-serif";
var FONTS="12px 'Microsoft JhengHei','Microsoft YaHei',sans-serif";
var FONTB="bold 15px 'Microsoft JhengHei','Microsoft YaHei',sans-serif";

/* ── cross-window sync ── */
var isPresenter=/[?&]view=presenter/i.test(location.search)||/view%3Dpresenter/i.test(location.search);
var bc=null;
try{bc=new BroadcastChannel('rl-pres-v2');}catch(e){}
function bcSend(msg){
  try{if(bc)bc.postMessage(msg);}catch(e){}
  /* localStorage fallback for file:// in Safari */
  try{localStorage.setItem('rl-sync-msg',JSON.stringify(Object.assign({},msg,{_ts:Date.now()})));}catch(e){}
}

/* ── animation state ── */
var curPage=null,progress=0,playing=false,lastTs=null,speed=1,rafId=null;
var DURATION=4500;
var ANIM_PAGES=new Set(Object.keys(ANIM).map(Number));

/* ── tool state ── */
var toolMode=null; /* 'laser'|'draw'|'eraser'|null */
var drawColor='#ff3333',drawSize=5;
var localDrawing=false,drawPts=[];
var laserCv=null,laserCtx=null,drawCv=null,drawCtx=null;

/* ── helpers ── */
function $id(id){return document.getElementById(id);}
function mapX(v,mn,mx,l,r){return l+(v-mn)/(mx-mn)*(r-l);}
function mapY(v,mn,mx,t,b){return b-(v-mn)/(mx-mn)*(b-t);}
function kfmt(v){return Math.abs(v)>=1e9?(v/1e9).toFixed(1)+'B':Math.abs(v)>=1e6?(v/1e6).toFixed(0)+'M':Math.abs(v)>=1000?(v/1000).toFixed(0)+'K':String(Math.round(v*10)/10);}
function hexRgba(hex,a){hex=hex.replace('#','');var r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16);return 'rgba('+r+','+g+','+b+','+a+')';}
function updateBtn(){var b=$id('rl-playpause');if(b)b.textContent=playing?'⏸ 暫停':'▶ 播放';}

/* ── layout ── */
function lay(w,h,pL,pR,pT,pB){pL=pL||80;pR=pR||30;pT=pT||52;pB=pB||68;return{l:pL,r:w-pR,t:pT,b:h-pB,w:w-pL-pR,h:h-pT-pB};}

/* ── polyfill roundRect ── */
if(!CanvasRenderingContext2D.prototype.roundRect){
  CanvasRenderingContext2D.prototype.roundRect=function(x,y,w,h,r){
    r=Math.min(r,w/2,h/2);this.beginPath();
    this.moveTo(x+r,y);this.lineTo(x+w-r,y);this.arcTo(x+w,y,x+w,y+r,r);
    this.lineTo(x+w,y+h-r);this.arcTo(x+w,y+h,x+w-r,y+h,r);
    this.lineTo(x+r,y+h);this.arcTo(x,y+h,x,y+h-r,r);
    this.lineTo(x,y+r);this.arcTo(x,y,x+r,y,r);this.closePath();
  };
}

/* ════════ CHART PRIMITIVES ════════ */
function axes(g,L,xMin,xMax,yMin,yMax,xLabel,yLabel,xTks){
  g.strokeStyle='#555';g.lineWidth=1.5;g.setLineDash([]);
  g.beginPath();g.moveTo(L.l,L.t);g.lineTo(L.l,L.b);g.lineTo(L.r,L.b);g.stroke();
  var range=yMax-yMin,rawStep=range/5;
  var mag=Math.pow(10,Math.floor(Math.log10(Math.abs(rawStep)||1)));
  var niceList=[1,2,2.5,5,10],niceStep=mag;
  for(var ni=0;ni<niceList.length;ni++){if(niceList[ni]*mag>=rawStep){niceStep=niceList[ni]*mag;break;}}
  g.font=FONTS;g.fillStyle='#555';
  for(var yv=Math.ceil(yMin/niceStep)*niceStep;yv<=yMax+niceStep*0.05;yv+=niceStep){
    var py=mapY(yv,yMin,yMax,L.t,L.b);
    g.strokeStyle='#e0e0e0';g.lineWidth=0.5;g.setLineDash([]);
    g.beginPath();g.moveTo(L.l,py);g.lineTo(L.r,py);g.stroke();
    g.strokeStyle='#777';g.lineWidth=1;g.beginPath();g.moveTo(L.l-4,py);g.lineTo(L.l,py);g.stroke();
    g.fillStyle='#444';g.textAlign='right';g.textBaseline='middle';g.fillText(kfmt(Math.round(yv*100)/100),L.l-8,py);
  }
  if(xTks&&xTks.length){
    var step=Math.ceil(xTks.length/7);
    for(var xi=0;xi<xTks.length;xi+=step){
      var px=mapX(xTks[xi].v,xMin,xMax,L.l,L.r);
      g.strokeStyle='#777';g.lineWidth=1;g.setLineDash([]);g.beginPath();g.moveTo(px,L.b);g.lineTo(px,L.b+4);g.stroke();
      g.fillStyle='#444';g.font=FONTS;g.textAlign='center';g.textBaseline='top';g.fillText(xTks[xi].lbl,px,L.b+7);
    }
    var last=xTks[xTks.length-1],plast=mapX(last.v,xMin,xMax,L.l,L.r);
    g.strokeStyle='#777';g.lineWidth=1;g.beginPath();g.moveTo(plast,L.b);g.lineTo(plast,L.b+4);g.stroke();
    g.fillStyle='#444';g.font=FONTS;g.textAlign='center';g.textBaseline='top';g.fillText(last.lbl,plast,L.b+7);
  }
  g.font=FONT;g.fillStyle='#333';g.setLineDash([]);
  g.textAlign='center';g.textBaseline='bottom';g.fillText(xLabel,L.l+L.w/2,L.b+58);
  g.save();g.translate(L.l-58,L.t+L.h/2);g.rotate(-Math.PI/2);g.textAlign='center';g.fillText(yLabel,0,0);g.restore();
}
function mkTks(xs){return xs.map(function(v){return{v:v,lbl:kfmt(v)};});}
function hlines(g,L,list,yMin,yMax){
  list.forEach(function(hl){
    var py=mapY(hl.y,yMin,yMax,L.t,L.b);if(py<L.t||py>L.b)return;
    g.strokeStyle='#888';g.lineWidth=1.4;g.setLineDash(hl.dash?[6,4]:[]);
    g.beginPath();g.moveTo(L.l,py);g.lineTo(L.r,py);g.stroke();g.setLineDash([]);
    g.fillStyle='#888';g.font=FONTS;g.textAlign='right';g.textBaseline='bottom';g.fillText(hl.label,L.r-4,py-3);
  });
}
function lineSeries(g,L,xs,ys,color,p,xMin,xMax,yMin,yMax){
  var n=xs.length,tp=p*(n-1),full=Math.floor(tp),frac=tp-full,pts=[];
  for(var i=0;i<=Math.min(full,n-1);i++)pts.push([mapX(xs[i],xMin,xMax,L.l,L.r),mapY(ys[i],yMin,yMax,L.t,L.b)]);
  if(full<n-1&&frac>0){
    var xi2=xs[full]+frac*(xs[full+1]-xs[full]),yi2=ys[full]+frac*(ys[full+1]-ys[full]);
    pts.push([mapX(xi2,xMin,xMax,L.l,L.r),mapY(yi2,yMin,yMax,L.t,L.b)]);
  }
  if(!pts.length)return;
  g.strokeStyle=color;g.lineWidth=2.5;g.setLineDash([]);
  g.beginPath();g.moveTo(pts[0][0],pts[0][1]);for(var j=1;j<pts.length;j++)g.lineTo(pts[j][0],pts[j][1]);g.stroke();
  g.fillStyle=color;
  for(var ii=0;ii<=Math.min(full,n-1);ii++){g.beginPath();g.arc(mapX(xs[ii],xMin,xMax,L.l,L.r),mapY(ys[ii],yMin,yMax,L.t,L.b),4,0,2*Math.PI);g.fill();}
}
function bandSeries(g,L,xs,mean,std,color,p,xMin,xMax,yMin,yMax){
  var n=xs.length,tp=p*(n-1),pts=Math.min(Math.floor(tp)+1,n-1);if(pts<1)return;
  g.save();g.globalAlpha=0.22;g.fillStyle=color;g.beginPath();
  g.moveTo(mapX(xs[0],xMin,xMax,L.l,L.r),mapY(mean[0]+std[0],yMin,yMax,L.t,L.b));
  for(var i=1;i<=pts;i++)g.lineTo(mapX(xs[i],xMin,xMax,L.l,L.r),mapY(mean[i]+std[i],yMin,yMax,L.t,L.b));
  for(var i=pts;i>=0;i--)g.lineTo(mapX(xs[i],xMin,xMax,L.l,L.r),mapY(mean[i]-std[i],yMin,yMax,L.t,L.b));
  g.closePath();g.fill();g.restore();
}
function legend(g,L,items){
  var x0=L.l+10,y0=L.t+12;
  items.forEach(function(it){
    g.strokeStyle=it.color;g.lineWidth=2.5;g.setLineDash([]);g.beginPath();g.moveTo(x0,y0+6);g.lineTo(x0+22,y0+6);g.stroke();
    g.fillStyle=it.color;g.beginPath();g.arc(x0+11,y0+6,4,0,2*Math.PI);g.fill();
    g.fillStyle='#333';g.font=FONTS;g.textAlign='left';g.textBaseline='middle';g.fillText(it.label,x0+28,y0+6);
    y0+=22;
  });
}
function milestone(g,L,xs,ys,xi,text,color,xMin,xMax,yMin,yMax,p){
  if(xi>=xs.length)return;
  var threshold=(xi+0.7)/xs.length;if(p<threshold)return;
  var alpha=Math.min(1,(p-threshold)*6);
  var px=mapX(xs[xi],xMin,xMax,L.l,L.r),py=mapY(ys[xi],yMin,yMax,L.t,L.b);
  var lines=text.split('\n'),maxLen=0;for(var i=0;i<lines.length;i++)maxLen=Math.max(maxLen,lines[i].length);
  var bw=maxLen*7.5+20,bh=lines.length*18+14;
  var bx=Math.min(px+14,L.r-bw-4),by=Math.max(py-bh-12,L.t+4);
  g.save();g.globalAlpha=alpha;
  g.fillStyle='white';g.strokeStyle=color;g.lineWidth=1.5;g.roundRect(bx,by,bw,bh,5);g.fill();g.stroke();
  g.fillStyle=color;g.font="bold 12px 'Microsoft JhengHei',sans-serif";g.textAlign='left';g.textBaseline='top';
  lines.forEach(function(line,li){g.fillText(line,bx+10,by+8+li*18);});
  g.strokeStyle=color;g.lineWidth=1.2;g.setLineDash([3,3]);g.beginPath();g.moveTo(px,py);g.lineTo(bx,by+bh/2);g.stroke();
  g.setLineDash([]);g.fillStyle=color;g.beginPath();g.arc(px,py,5,0,2*Math.PI);g.fill();
  g.restore();
}
function chartTitle(g,w,title){g.font=FONTB;g.fillStyle='#2c3e50';g.textAlign='center';g.textBaseline='top';g.fillText(title,w/2,14);}
function progressBar(g,w,h,p){g.fillStyle='#e0e0e0';g.fillRect(0,h-6,w,6);g.fillStyle='#27ae60';g.fillRect(0,h-6,w*p,6);}

/* ════════ CHART TYPES ════════ */
function drawLine(g,w,h,cfg,p){
  var L=lay(w,h),ds=cfg.datasets[0],xs=cfg.xs,xMin=xs[0],xMax=xs[xs.length-1],yMin=cfg.yMin,yMax=cfg.yMax;
  axes(g,L,xMin,xMax,yMin,yMax,cfg.xLabel,cfg.yLabel,mkTks(xs));
  if(cfg.hlines)hlines(g,L,cfg.hlines,yMin,yMax);
  lineSeries(g,L,xs,ds.ys,ds.color,p,xMin,xMax,yMin,yMax);
  if(cfg.milestone){var m=cfg.milestone;milestone(g,L,xs,ds.ys,m.xi,m.text,m.color||ds.color,xMin,xMax,yMin,yMax,p);}
  legend(g,L,[{label:ds.label,color:ds.color}]);chartTitle(g,w,cfg.title);
}
function drawLineBand(g,w,h,cfg,p){
  var L=lay(w,h),ds=cfg.datasets[0],xs=cfg.xs,xMin=xs[0],xMax=xs[xs.length-1],yMin=cfg.yMin,yMax=cfg.yMax;
  axes(g,L,xMin,xMax,yMin,yMax,cfg.xLabel,cfg.yLabel,mkTks(xs));
  if(cfg.hlines)hlines(g,L,cfg.hlines,yMin,yMax);
  bandSeries(g,L,xs,ds.mean,ds.std,ds.color,p,xMin,xMax,yMin,yMax);
  lineSeries(g,L,xs,ds.mean,ds.color,p,xMin,xMax,yMin,yMax);
  if(cfg.milestone){var m=cfg.milestone;milestone(g,L,xs,ds.mean,m.xi,m.text,m.color||ds.color,xMin,xMax,yMin,yMax,p);}
  legend(g,L,[{label:ds.label+' (±1 std)',color:ds.color}]);chartTitle(g,w,cfg.title);
}
function drawMultiLine(g,w,h,cfg,p){
  var L=lay(w,h),yMin=cfg.yMin,yMax=cfg.yMax,xMin=Infinity,xMax=-Infinity;
  cfg.datasets.forEach(function(ds){xMin=Math.min(xMin,ds.xs[0]);xMax=Math.max(xMax,ds.xs[ds.xs.length-1]);});
  axes(g,L,xMin,xMax,yMin,yMax,cfg.xLabel,cfg.yLabel,[]);
  var allX=[];cfg.datasets.forEach(function(ds){ds.xs.forEach(function(x){if(allX.indexOf(x)<0)allX.push(x);});});
  allX.sort(function(a,b){return a-b;});
  var step=Math.ceil(allX.length/7);
  allX.forEach(function(x,i){
    if(i%step!==0&&i!==allX.length-1)return;
    var px=mapX(x,xMin,xMax,L.l,L.r);
    g.strokeStyle='#777';g.lineWidth=1;g.setLineDash([]);g.beginPath();g.moveTo(px,L.b);g.lineTo(px,L.b+4);g.stroke();
    g.fillStyle='#444';g.font=FONTS;g.textAlign='center';g.textBaseline='top';g.fillText(kfmt(x),px,L.b+7);
  });
  if(cfg.hlines)hlines(g,L,cfg.hlines,yMin,yMax);
  cfg.datasets.forEach(function(ds){lineSeries(g,L,ds.xs,ds.ys,ds.color,p,xMin,xMax,yMin,yMax);});
  if(cfg.milestone){var m=cfg.milestone,mds=cfg.datasets[m.dsi];milestone(g,L,mds.xs,mds.ys,m.xi,m.text,m.color||mds.color,xMin,xMax,yMin,yMax,p);}
  legend(g,L,cfg.datasets.map(function(ds){return{label:ds.label,color:ds.color};}));chartTitle(g,w,cfg.title);
}
function drawDualLine(g,w,h,cfg,p){
  var panW=(w-24)/2;
  cfg.panels.forEach(function(panel,pi){
    g.save();g.translate(pi*(panW+24),0);
    var L=lay(panW,h,72,22,52,68),xs=panel.xs,xMin=xs[0],xMax=xs[xs.length-1],yMin=panel.yMin,yMax=panel.yMax;
    axes(g,L,xMin,xMax,yMin,yMax,panel.xLabel,panel.yLabel,mkTks(xs));
    if(panel.hline)hlines(g,L,[Object.assign({dash:true},panel.hline)],yMin,yMax);
    lineSeries(g,L,xs,panel.ys,panel.color,p,xMin,xMax,yMin,yMax);
    if(panel.milestone){var m=panel.milestone;milestone(g,L,xs,panel.ys,m.xi,m.text,panel.color,xMin,xMax,yMin,yMax,p);}
    legend(g,L,[{label:panel.label,color:panel.color}]);g.restore();
  });
  chartTitle(g,w,cfg.title);
}
function drawBarLog(g,w,h,cfg,p){
  var L=lay(w,h,70,25,52,70),vals=cfg.values,n=vals.length;
  var logVals=vals.map(function(v){return Math.log10(v+1);}),yMax=Math.max.apply(null,logVals)*1.15;
  var barsShown=Math.round(p*n),gap=L.w/n,bw=gap*0.72;
  g.strokeStyle='#555';g.lineWidth=1.5;g.setLineDash([]);g.beginPath();g.moveTo(L.l,L.t);g.lineTo(L.l,L.b);g.lineTo(L.r,L.b);g.stroke();
  for(var i=0;i<barsShown;i++){
    var barP=(i<barsShown-1)?1:(p*n-Math.floor(p*n)||1);
    var lv=logVals[i]*barP,bx=L.l+gap*i+(gap-bw)/2,bh2=lv/yMax*L.h,by=L.b-bh2;
    g.fillStyle=cfg.color;g.globalAlpha=0.82;g.fillRect(bx,by,bw,bh2);g.globalAlpha=1;
    g.font=FONTS;g.fillStyle='#444';g.textAlign='center';g.textBaseline='top';g.fillText(cfg.labels[i],bx+bw/2,L.b+7);
    if(bh2>18){g.font="bold 10px 'Microsoft JhengHei',sans-serif";g.fillStyle='#2c3e50';g.textBaseline='bottom';g.fillText(kfmt(vals[i]),bx+bw/2,by-3);}
    if(cfg.annotations&&cfg.annotations[i+1]){
      var lines2=cfg.annotations[i+1].split('\n');
      g.font="bold 11px 'Microsoft JhengHei',sans-serif";g.fillStyle='#e74c3c';g.textAlign='center';g.textBaseline='bottom';
      lines2.forEach(function(line,li){g.fillText(line,bx+bw/2,by-4-li*14);});
    }
  }
  g.font=FONT;g.fillStyle='#333';g.setLineDash([]);g.textAlign='center';g.textBaseline='bottom';g.fillText(cfg.xLabel,L.l+L.w/2,L.b+58);
  g.save();g.translate(L.l-52,L.t+L.h/2);g.rotate(-Math.PI/2);g.textAlign='center';g.fillText(cfg.yLabel,0,0);g.restore();
  chartTitle(g,w,cfg.title);
}
function drawGroupedBar(g,w,h,cfg,p){
  var L=lay(w,h,78,22,52,70),n=cfg.labels.length,nDs=cfg.datasets.length;
  var yMin=cfg.yMin,yMax=cfg.yMax,barsShown=Math.round(p*n),gap=L.w/n,groupW=gap*0.82,bw=groupW/nDs;
  g.strokeStyle='#555';g.lineWidth=1.5;g.setLineDash([]);g.beginPath();g.moveTo(L.l,L.t);g.lineTo(L.l,L.b);g.lineTo(L.r,L.b);g.stroke();
  var niceStep=100;
  for(var yv=Math.ceil(yMin/niceStep)*niceStep;yv<=yMax+5;yv+=niceStep){
    var py=mapY(yv,yMin,yMax,L.t,L.b);
    g.strokeStyle='#e0e0e0';g.lineWidth=0.5;g.setLineDash([]);g.beginPath();g.moveTo(L.l,py);g.lineTo(L.r,py);g.stroke();
    g.strokeStyle='#777';g.lineWidth=1;g.beginPath();g.moveTo(L.l-4,py);g.lineTo(L.l,py);g.stroke();
    g.fillStyle='#444';g.textAlign='right';g.textBaseline='middle';g.fillText(yv,L.l-8,py);
  }
  if(cfg.shadeFrom!=null&&barsShown>cfg.shadeFrom){
    var sx=L.l+gap*cfg.shadeFrom;
    g.save();g.globalAlpha=0.1;g.fillStyle='#e74c3c';g.fillRect(sx,L.t,L.r-sx,L.h);g.restore();
    g.font="bold 11px 'Microsoft JhengHei',sans-serif";g.fillStyle='#e74c3c';g.textAlign='left';g.textBaseline='top';g.fillText(cfg.note||'分布偏移區',sx+6,L.t+8);
  }
  var py0=mapY(0,yMin,yMax,L.t,L.b);
  g.strokeStyle='#999';g.lineWidth=1;g.setLineDash([]);g.beginPath();g.moveTo(L.l,py0);g.lineTo(L.r,py0);g.stroke();
  for(var i=0;i<barsShown;i++){
    var gx=L.l+gap*i+(gap-groupW)/2;
    cfg.datasets.forEach(function(ds,di){
      var v=ds.values[i],bx2=gx+bw*di,py1=mapY(v,yMin,yMax,L.t,L.b),barH=Math.abs(py1-py0),barY=Math.min(py0,py1);
      g.fillStyle=ds.color;g.globalAlpha=0.85;g.fillRect(bx2,barY,bw-2,barH);g.globalAlpha=1;
    });
    g.font=FONTS;g.fillStyle='#444';g.textAlign='center';g.textBaseline='top';g.fillText(cfg.labels[i],gx+groupW/2,L.b+7);
  }
  g.font=FONT;g.fillStyle='#333';g.setLineDash([]);g.textAlign='center';g.textBaseline='bottom';g.fillText(cfg.xLabel,L.l+L.w/2,L.b+58);
  g.save();g.translate(L.l-58,L.t+L.h/2);g.rotate(-Math.PI/2);g.textAlign='center';g.fillText(cfg.yLabel,0,0);g.restore();
  legend(g,L,cfg.datasets.map(function(ds){return{label:ds.label,color:ds.color};}));chartTitle(g,w,cfg.title);
}

/* ── dispatch ── */
function rlDraw(p){
  var cv=$id('rl-canvas');if(!cv)return;
  var g=cv.getContext('2d');g.clearRect(0,0,cv.width,cv.height);
  var cfg=ANIM[curPage];if(!cfg)return;
  switch(cfg.type){
    case 'line':        drawLine(g,cv.width,cv.height,cfg,p);break;
    case 'line_band':   drawLineBand(g,cv.width,cv.height,cfg,p);break;
    case 'multi_line':  drawMultiLine(g,cv.width,cv.height,cfg,p);break;
    case 'dual_line':   drawDualLine(g,cv.width,cv.height,cfg,p);break;
    case 'bar_log':     drawBarLog(g,cv.width,cv.height,cfg,p);break;
    case 'grouped_bar': drawGroupedBar(g,cv.width,cv.height,cfg,p);break;
  }
  progressBar(g,cv.width,cv.height,p);
}

/* ── animation loop ── */
function tick(ts){
  if(!playing){rafId=null;return;}
  if(lastTs===null)lastTs=ts;
  var dt=(ts-lastTs)*speed;lastTs=ts;
  progress=Math.min(1,progress+dt/DURATION);
  rlDraw(progress);
  if(progress<1){rafId=requestAnimationFrame(tick);}else{playing=false;rafId=null;updateBtn();}
}

/* ════════ LASER POINTER ════════ */
function drawLaserDot(x,y){
  if(!laserCtx)return;
  laserCtx.clearRect(0,0,laserCv.width,laserCv.height);
  var g1=laserCtx.createRadialGradient(x,y,0,x,y,30);
  g1.addColorStop(0,'rgba(255,30,30,0.65)');g1.addColorStop(0.45,'rgba(255,50,50,0.22)');g1.addColorStop(1,'rgba(255,0,0,0)');
  laserCtx.fillStyle=g1;laserCtx.beginPath();laserCtx.arc(x,y,30,0,Math.PI*2);laserCtx.fill();
  var g2=laserCtx.createRadialGradient(x,y,0,x,y,9);
  g2.addColorStop(0,'rgba(255,255,255,0.96)');g2.addColorStop(0.4,'rgba(255,90,90,0.9)');g2.addColorStop(1,'rgba(220,20,20,0.6)');
  laserCtx.fillStyle=g2;laserCtx.beginPath();laserCtx.arc(x,y,9,0,Math.PI*2);laserCtx.fill();
}
function hideLaser(){if(laserCtx)laserCtx.clearRect(0,0,laserCv.width,laserCv.height);}

/* ════════ DRAWING ════════ */
function drawBegin(x,y,color,size,eraser){
  if(!drawCtx)return;
  localDrawing=true;drawPts=[{x:x,y:y}];
  drawCtx.globalCompositeOperation=eraser?'destination-out':'source-over';
  drawCtx.strokeStyle=eraser?'rgba(0,0,0,1)':color;
  drawCtx.lineWidth=eraser?size*4:size;
  drawCtx.lineCap='round';drawCtx.lineJoin='round';
  drawCtx.beginPath();drawCtx.moveTo(x,y);
}
function drawCont(x,y){
  if(!localDrawing||!drawCtx)return;
  drawPts.push({x:x,y:y});var n=drawPts.length;
  if(n>=3){var p1=drawPts[n-2],p2=drawPts[n-1],mx=(p1.x+p2.x)/2,my=(p1.y+p2.y)/2;drawCtx.quadraticCurveTo(p1.x,p1.y,mx,my);}
  else{drawCtx.lineTo(x,y);}
  drawCtx.stroke();drawCtx.beginPath();drawCtx.moveTo(x,y);
}
function drawEnd2(){if(!localDrawing)return;localDrawing=false;if(drawCtx)drawCtx.globalCompositeOperation='source-over';}
function clearDraw(){if(drawCtx)drawCtx.clearRect(0,0,drawCv.width,drawCv.height);}

/* ════════ SLIDE COORD HELPERS ════════ */
function getSlideRect(){
  var svg=document.querySelector('svg.bespoke-marp-slide.bespoke-marp-active');
  return svg?svg.getBoundingClientRect():null;
}
function toNorm(cx,cy){var r=getSlideRect();if(!r||!r.width)return null;return{nx:(cx-r.left)/r.width,ny:(cy-r.top)/r.height};}
function fromNorm(nx,ny){var r=getSlideRect();if(!r)return null;return{x:r.left+nx*r.width,y:r.top+ny*r.height};}

/* ════════ MODE MANAGEMENT ════════ */
function setMode(m){
  toolMode=(toolMode===m)?null:m;
  updateToolbarState();
  if(drawCv){
    if(toolMode==='draw'||toolMode==='eraser'){
      drawCv.style.pointerEvents='auto';drawCv.style.display='block';
    }else{
      drawCv.style.pointerEvents='none';
    }
  }
  if(!toolMode){hideLaser();bcSend({type:'laser-off'});}
}
function updateToolbarState(){
  ['laser','draw','eraser'].forEach(function(m){
    var btn=$id('rl-tb-'+m);if(!btn)return;
    var on=toolMode===m;
    btn.style.background=on?'rgba(255,255,255,0.18)':'transparent';
    btn.style.color=on?'#fff':'#bbb';
    btn.style.borderColor=on?'rgba(255,255,255,0.35)':'transparent';
  });
}

/* ════════ TOAST ════════ */
function showToast(msg){
  var t=$id('rl-toast');if(!t)return;
  t.textContent=msg;t.style.display='block';t.style.opacity='1';
  clearTimeout(t._tmr);
  t._tmr=setTimeout(function(){t.style.opacity='0';setTimeout(function(){t.style.display='none';},400);},2200);
}

/* ════════ ANIMATION PUBLIC API ════════ */
function localOpen(){
  var ov=$id('rl-overlay');if(!ov||!ANIM[curPage])return;
  ov.style.display='flex';
  var tl=$id('rl-title');if(tl)tl.textContent=ANIM[curPage].title;
  progress=0;playing=true;lastTs=null;
  if(rafId)cancelAnimationFrame(rafId);
  rafId=requestAnimationFrame(tick);updateBtn();
}
function _rlOpenFn(){
  if(!curPage||!ANIM[curPage]){showToast('此頁無動態圖表');return;}
  if(isPresenter){bcSend({type:'open-anim',page:curPage});showToast('已發送至主畫面 📊');}
  else{localOpen();}
}
window.rlTbAnim=_rlOpenFn;
window.rlOpen=_rlOpenFn;
window.rlClose=function(){
  var ov=$id('rl-overlay');if(ov)ov.style.display='none';
  playing=false;if(rafId){cancelAnimationFrame(rafId);rafId=null;}
};
window.rlTogglePlay=function(){
  if(playing){playing=false;updateBtn();}
  else{if(progress>=1)progress=0;playing=true;lastTs=null;if(!rafId)rafId=requestAnimationFrame(tick);updateBtn();}
};
window.rlRestart=function(){progress=0;playing=true;lastTs=null;if(!rafId)rafId=requestAnimationFrame(tick);updateBtn();};
window.rlSetSpeed=function(s){speed=parseFloat(s);var l=$id('rl-speed-lbl');if(l)l.textContent=s+'×';};
window.rlSetMode=setMode;
window.rlClearDraw=function(){clearDraw();bcSend({type:'draw-clear'});};

/* ════════ BROADCAST RECEIVER (main window only) ════════ */
function handleMsg(msg){
  if(!msg||!msg.type)return;
  switch(msg.type){
    case 'open-anim': curPage=msg.page;localOpen();break;
    case 'laser-move':
      if(laserCv)laserCv.style.display='block';
      var lp=fromNorm(msg.nx,msg.ny);if(lp)drawLaserDot(lp.x,lp.y);break;
    case 'laser-off': hideLaser();break;
    case 'draw-start':
      var dp=fromNorm(msg.nx,msg.ny);if(!dp)break;
      if(drawCv)drawCv.style.display='block';
      drawBegin(dp.x,dp.y,msg.color||'#ff3333',msg.size||5,!!msg.eraser);break;
    case 'draw-move':
      var dp2=fromNorm(msg.nx,msg.ny);if(dp2)drawCont(dp2.x,dp2.y);break;
    case 'draw-end': drawEnd2();break;
    case 'draw-clear': clearDraw();break;
  }
}
if(!isPresenter){
  if(bc)bc.onmessage=function(e){handleMsg(e.data);};
  window.addEventListener('storage',function(e){
    if(e.key!=='rl-sync-msg')return;
    try{var msg=JSON.parse(e.newValue);if(Date.now()-msg._ts<2000)handleMsg(msg);}catch(ex){}
  });
}

/* ════════ MOUSE / TOUCH HANDLERS ════════ */
function onDocMove(e){
  if(toolMode!=='laser')return;
  if(laserCv)laserCv.style.display='block';
  drawLaserDot(e.clientX,e.clientY);
  var n=toNorm(e.clientX,e.clientY);
  if(n)bcSend({type:'laser-move',nx:n.nx,ny:n.ny});
}
function onDocLeave(){if(toolMode==='laser'){hideLaser();bcSend({type:'laser-off'});}}

function onDcDown(e){
  if(toolMode!=='draw'&&toolMode!=='eraser')return;
  e.preventDefault();
  if(drawCv)drawCv.style.display='block';
  var eraser=toolMode==='eraser';
  drawBegin(e.clientX,e.clientY,drawColor,drawSize,eraser);
  var n=toNorm(e.clientX,e.clientY);
  if(n)bcSend({type:'draw-start',nx:n.nx,ny:n.ny,color:drawColor,size:drawSize,eraser:eraser});
}
function onDcMove(e){
  if(!localDrawing)return;e.preventDefault();
  drawCont(e.clientX,e.clientY);
  var n=toNorm(e.clientX,e.clientY);if(n)bcSend({type:'draw-move',nx:n.nx,ny:n.ny});
}
function onDcUp(){if(!localDrawing)return;drawEnd2();bcSend({type:'draw-end'});}

/* ════════ KEYBOARD SHORTCUTS ════════ */
document.addEventListener('keydown',function(e){
  setTimeout(detectPage,50);
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
  switch(e.key.toLowerCase()){
    case 'l':setMode('laser');break;
    case 'd':setMode('draw');break;
    case 'e':setMode('eraser');break;
    case 'c':if(toolMode==='draw'||toolMode==='eraser'){clearDraw();bcSend({type:'draw-clear'});}break;
    case 'escape':
      toolMode=null;updateToolbarState();
      if(drawCv)drawCv.style.pointerEvents='none';
      hideLaser();bcSend({type:'laser-off'});break;
  }
});

/* ════════ SLIDE DETECTION ════════ */
function detectPage(){
  var svg=document.querySelector('svg.bespoke-marp-slide.bespoke-marp-active');if(!svg)return;
  var sec=svg.querySelector('section[data-marpit-pagination]');if(!sec)return;
  var page=parseInt(sec.getAttribute('data-marpit-pagination'));
  var btn=$id('rl-tb-anim');
  if(ANIM_PAGES.has(page)){
    curPage=page;
    if(btn){btn.style.opacity='1';btn.style.cursor='pointer';}
  }else{
    curPage=null;
    if(btn){btn.style.opacity='0.35';btn.style.cursor='default';}
  }
}
setInterval(detectPage,300);
document.addEventListener('click',function(){setTimeout(detectPage,50);});
document.addEventListener('touchend',function(){setTimeout(detectPage,50);});

/* ════════ INIT ════════ */
function init(){
  laserCv=$id('rl-laser-cv');
  if(laserCv){laserCv.width=window.innerWidth;laserCv.height=window.innerHeight;laserCtx=laserCv.getContext('2d');}
  drawCv=$id('rl-draw-cv');
  if(drawCv){drawCv.width=window.innerWidth;drawCv.height=window.innerHeight;drawCtx=drawCv.getContext('2d');}

  window.addEventListener('resize',function(){
    if(laserCv){laserCv.width=window.innerWidth;laserCv.height=window.innerHeight;}
    if(drawCv){
      var imgData=drawCtx.getImageData(0,0,drawCv.width,drawCv.height);
      drawCv.width=window.innerWidth;drawCv.height=window.innerHeight;
      drawCtx.putImageData(imgData,0,0);
    }
  });

  document.addEventListener('mousemove',onDocMove);
  document.addEventListener('mouseleave',onDocLeave);

  if(drawCv){
    drawCv.addEventListener('mousedown',onDcDown);
    drawCv.addEventListener('mousemove',onDcMove);
    drawCv.addEventListener('mouseup',onDcUp);
    drawCv.addEventListener('mouseleave',onDcUp);
    drawCv.addEventListener('touchstart',function(e){var t=e.touches[0];onDcDown({clientX:t.clientX,clientY:t.clientY,preventDefault:function(){e.preventDefault();}});},{passive:false});
    drawCv.addEventListener('touchmove',function(e){e.preventDefault();var t=e.touches[0];onDcMove({clientX:t.clientX,clientY:t.clientY,preventDefault:function(){}});},{passive:false});
    drawCv.addEventListener('touchend',onDcUp);
  }

  var cl=$id('rl-color');if(cl)cl.addEventListener('input',function(){drawColor=this.value;});
  var si=$id('rl-size'); if(si)si.addEventListener('input',function(){drawSize=parseInt(this.value);});

  /* 只在講者模式（presenter view）顯示工具列；主畫面（投影幕）不顯示 */
  if(isPresenter){
    var tb=$id('rl-toolbar');
    if(tb)tb.style.display='flex';
  }

  updateToolbarState();
  detectPage();
}

if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}

})();
</script>
"""

# ── inject ────────────────────────────────────────────────────────────────────
def inject():
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    if 'rl-overlay' in html:
        print('Already injected — removing old version first...')
        html = re.sub(r'<!-- RL Animations.*?</script>', '', html, flags=re.DOTALL)

    data_json = json.dumps(ANIM, ensure_ascii=False)
    js_block  = JS_TEMPLATE.replace('__DATA__', data_json)
    injection = OVERLAY_HTML + js_block

    html = html.replace('</body>', injection + '\n</body>', 1)

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'✓ Animations + laser + draw injected → {HTML_PATH}')
    print(f'  Slides with animations: {sorted(ANIM.keys())}')
    print(f'  Tools: L=雷射 D=畫筆 E=橡皮擦 C=清除 Esc=離開工具')

if __name__ == '__main__':
    inject()
