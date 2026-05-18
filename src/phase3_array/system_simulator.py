"""Build an interactive visual simulator for the Phase 3 array system.

The generated HTML runs a lightweight browser-side physics simulation:
far-field source TDOA, microphone signals, delay-and-sum beam scanning, and a
simple confidence proxy. It is meant for visual understanding of the real array
pipeline; trained PyTorch CNN inference stays in the offline Phase 3 evaluator.
"""

from pathlib import Path

from . import config_phase3 as config


SIM_DIR = config.results_dir / "system_simulator"


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 3 Array System Simulator</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171b22;
      --panel2: #202733;
      --text: #edf2f5;
      --muted: #9ca9b4;
      --line: #33404b;
      --green: #39c172;
      --cyan: #36b6d8;
      --amber: #e0a83c;
      --red: #dc5a57;
      --blue: #77a7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 14px 18px 10px;
      border-bottom: 1px solid var(--line);
      background: #131820;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 23px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(620px, 1fr);
      min-height: calc(100vh - 69px);
    }
    aside {
      padding: 14px;
      background: #12161c;
      border-right: 1px solid var(--line);
    }
    .content {
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .row {
      display: grid;
      gap: 6px;
      margin-bottom: 11px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
    }
    select, button, input[type="range"] {
      width: 100%;
    }
    select, button {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid #40505d;
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      font-weight: 700;
    }
    button.primary {
      border-color: #2787a3;
      background: #163241;
    }
    input[type="checkbox"] {
      transform: translateY(1px);
    }
    .check {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--text);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .top {
      display: grid;
      grid-template-columns: minmax(560px, 1.2fr) minmax(310px, 0.8fr);
      gap: 12px;
    }
    canvas {
      width: 100%;
      display: block;
      background: #0b0e12;
      border-radius: 6px;
    }
    #worldCanvas { height: 560px; }
    #heatCanvas { height: 250px; }
    #waveCanvas { height: 220px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      min-height: 70px;
      border: 1px solid #2d3944;
      border-radius: 6px;
      padding: 9px;
      background: var(--panel2);
    }
    .metric .label {
      margin-bottom: 5px;
    }
    .metric .value {
      font-size: 20px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .good .value { color: var(--green); }
    .warn .value { color: var(--amber); }
    .bad .value { color: var(--red); }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .key {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }
    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--line); }
      .top { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Phase 3 Array System Simulator</h1>
    <div class="sub">Interactive receive-only microphone-array simulation: drone source, per-mic arrival delays, delay-and-sum beam scan, estimated direction, and detector confidence proxy.</div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Scenario</h2>
        <div class="row">
          <label for="scenario">Source mix</label>
          <select id="scenario">
            <option value="drone">drone alone</option>
            <option value="drone_tank">drone + tank</option>
            <option value="tank">tank alone</option>
            <option value="moving">moving drone</option>
          </select>
        </div>
        <div class="check"><input id="motion" type="checkbox"> <span>move drone</span></div>
        <div class="row">
          <label>true azimuth <span id="azLabel"></span></label>
          <input id="az" type="range" min="0" max="359" step="1" value="90">
        </div>
        <div class="row">
          <label>true elevation <span id="elLabel"></span></label>
          <input id="el" type="range" min="5" max="85" step="1" value="40">
        </div>
        <div class="row">
          <label>drone strength <span id="droneLabel"></span></label>
          <input id="droneAmp" type="range" min="0" max="1.5" step="0.01" value="1">
        </div>
        <div class="row">
          <label>tank interference <span id="tankLabel"></span></label>
          <input id="tankAmp" type="range" min="0" max="1.5" step="0.01" value="0">
        </div>
        <div class="row">
          <label>sensor noise <span id="noiseLabel"></span></label>
          <input id="noiseAmp" type="range" min="0" max="0.7" step="0.01" value="0.06">
        </div>
        <div class="row">
          <button id="play" class="primary">Pause</button>
          <button id="reset">Reset</button>
        </div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Array</h2>
        <div class="sub">
          8 microphones, cube geometry, 5 cm spacing.<br>
          Beam scan grid: 15 degree azimuth, 10-80 degree elevation.<br>
          The orange arrow is the beamformer estimate.
        </div>
      </div>
    </aside>
    <section class="content">
      <div class="top">
        <div class="panel">
          <h2>World View</h2>
          <canvas id="worldCanvas" width="1100" height="760"></canvas>
          <div class="legend">
            <span class="key"><span class="swatch" style="background:var(--green)"></span>microphones</span>
            <span class="key"><span class="swatch" style="background:var(--cyan)"></span>true drone direction</span>
            <span class="key"><span class="swatch" style="background:var(--amber)"></span>estimated beam direction</span>
            <span class="key"><span class="swatch" style="background:var(--red)"></span>tank/interference</span>
          </div>
        </div>
        <div class="panel">
          <h2>Detector And Direction</h2>
          <div id="metrics" class="metrics"></div>
        </div>
      </div>
      <div class="panel">
        <h2>Beam Scan Heatmap</h2>
        <canvas id="heatCanvas" width="1200" height="320"></canvas>
      </div>
      <div class="panel">
        <h2>Microphone Signals</h2>
        <canvas id="waveCanvas" width="1200" height="280"></canvas>
      </div>
    </section>
  </main>
<script>
const C = 343.0;
const FS = 16000;
const N = 640;
const spacing = 0.05;
const micPositions = [];
for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) {
  micPositions.push([x * spacing, y * spacing, z * spacing]);
}
const azGrid = Array.from({length: 24}, (_, i) => i * 15);
const elGrid = [10,20,30,40,50,60,70,80];
const COLORS = {
  text: '#edf2f5',
  muted: '#9ca9b4',
  grid: '#2b3541',
  green: '#39c172',
  cyan: '#36b6d8',
  amber: '#e0a83c',
  red: '#dc5a57',
  blue: '#77a7ff'
};
let running = true;
let sampleClock = 0;
let drawPhase = 0;
let lastScanMs = 0;
let state = {
  az: 90, el: 40,
  tankAz: 225, tankEl: 20,
  confidence: 0,
  detected: false,
  smoothed: false,
  bestAz: 90, bestEl: 40,
  scoreMap: new Map(),
  signals: Array.from({length: micPositions.length}, () => new Float32Array(N)),
  history: [],
  directionError: 0
};

const ids = ['scenario','motion','az','el','droneAmp','tankAmp','noiseAmp','play','reset'];
const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const labels = {
  az: document.getElementById('azLabel'),
  el: document.getElementById('elLabel'),
  drone: document.getElementById('droneLabel'),
  tank: document.getElementById('tankLabel'),
  noise: document.getElementById('noiseLabel')
};

function unit(azDeg, elDeg) {
  const az = azDeg * Math.PI / 180;
  const el = elDeg * Math.PI / 180;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
}
function delaysFor(azDeg, elDeg) {
  const u = unit(azDeg, elDeg);
  const raw = micPositions.map(r => -(r[0]*u[0] + r[1]*u[1] + r[2]*u[2]) / C);
  const mean = raw.reduce((a,b)=>a+b,0) / raw.length;
  return raw.map(v => v - mean);
}
function wrapDeg(a) {
  return ((a % 360) + 360) % 360;
}
function angleDiff(a, b) {
  let d = Math.abs(wrapDeg(a) - wrapDeg(b));
  return d > 180 ? 360 - d : d;
}
function noise(seed) {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}
function droneSignal(t) {
  const wob = 1 + 0.025 * Math.sin(2*Math.PI*3.2*t);
  const f = 215 * wob;
  const ph = 2 * Math.PI * f * t;
  const am = 0.76 + 0.24 * Math.sin(2*Math.PI*6.7*t + 0.4);
  return am * (0.58*Math.sin(ph) + 0.28*Math.sin(2.02*ph) + 0.13*Math.sin(3.04*ph) + 0.06*Math.sin(4.1*ph));
}
function tankSignal(t) {
  const ph = 2 * Math.PI * 48 * t;
  const clank = Math.max(0, Math.sin(2*Math.PI*7.3*t)) ** 18;
  return 0.82*Math.sin(ph) + 0.35*Math.sin(2*ph) + 0.17*Math.sin(3*ph) + clank * noise(Math.floor(t*80));
}
function interp(arr, idx) {
  if (idx < 0 || idx >= arr.length - 1) return 0;
  const i = Math.floor(idx);
  const f = idx - i;
  return arr[i] * (1 - f) + arr[i + 1] * f;
}
function generateSignals() {
  const scenario = ui.scenario.value;
  let droneAmp = Number(ui.droneAmp.value);
  let tankAmp = Number(ui.tankAmp.value);
  if (scenario === 'tank') droneAmp = 0;
  if (scenario === 'tank') tankAmp = Math.max(tankAmp, 0.9);
  if (scenario === 'drone') tankAmp = 0;
  if (scenario === 'drone_tank') tankAmp = Math.max(tankAmp, 0.55);
  if (scenario === 'moving') droneAmp = Math.max(droneAmp, 0.9);

  const srcDelays = delaysFor(state.az, state.el);
  const tankDelaysA = delaysFor(state.tankAz, state.tankEl);
  const tankDelaysB = delaysFor(wrapDeg(state.tankAz + 38), Math.max(10, state.tankEl - 8));
  const noiseAmp = Number(ui.noiseAmp.value);
  const signals = Array.from({length: micPositions.length}, () => new Float32Array(N));
  for (let m = 0; m < micPositions.length; m++) {
    for (let n = 0; n < N; n++) {
      const t = (sampleClock + n) / FS;
      const d = droneSignal(t - srcDelays[m]);
      const ta = tankSignal(t - tankDelaysA[m]);
      const tb = tankSignal(t * 0.997 - tankDelaysB[m] + 0.013);
      const sensor = noise((sampleClock + n) * (m + 3) + m * 97);
      signals[m][n] = droneAmp * d + tankAmp * (0.55*ta + 0.35*tb) + noiseAmp * sensor;
    }
  }
  state.signals = signals;
}
function beamScore(az, el) {
  const cand = delaysFor(az, el).map(v => v * FS);
  let prev = 0;
  let sumSq = 0;
  let rawSq = 0;
  let count = 0;
  for (let n = 12; n < N - 12; n += 2) {
    let y = 0;
    for (let m = 0; m < micPositions.length; m++) {
      y += interp(state.signals[m], n + cand[m]);
    }
    y /= micPositions.length;
    const hp = y - 0.985 * prev;
    prev = y;
    sumSq += hp * hp;
    rawSq += y * y;
    count++;
  }
  const band = Math.sqrt(sumSq / Math.max(1, count));
  const raw = Math.sqrt(rawSq / Math.max(1, count));
  return band * 1.55 + raw * 0.12;
}
function scanBeams() {
  generateSignals();
  let best = -Infinity;
  let worst = Infinity;
  let bestAz = 0;
  let bestEl = 0;
  const map = new Map();
  for (const el of elGrid) {
    for (const az of azGrid) {
      const score = beamScore(az, el);
      map.set(`${az}|${el}`, score);
      if (score > best) { best = score; bestAz = az; bestEl = el; }
      if (score < worst) worst = score;
    }
  }
  const span = Math.max(0.001, best - worst);
  const normalized = (best - worst) / (best + 0.10);
  const confidence = Math.max(0, Math.min(1, normalized * 1.35));
  const scenario = ui.scenario.value;
  const dronePresent = scenario !== 'tank';
  const detected = confidence > 0.47 && (dronePresent || best > 0.36);
  state.history.push(detected);
  while (state.history.length > 3) state.history.shift();
  state.smoothed = state.history.length === 3 && state.history.filter(Boolean).length >= 2;
  state.detected = detected;
  state.confidence = confidence;
  state.bestAz = bestAz;
  state.bestEl = bestEl;
  state.scoreMap = map;
  state.directionError = angleDiff(state.az, bestAz);
}
function resizeCanvases() {
  for (const id of ['worldCanvas','heatCanvas','waveCanvas']) {
    const c = document.getElementById(id);
    const r = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(300, Math.floor(r.width * dpr));
    c.height = Math.max(180, Math.floor(r.height * dpr));
  }
}
function drawArrow(ctx, cx, cy, len, az, color, label) {
  const a = az * Math.PI / 180;
  const x = cx + Math.cos(a) * len;
  const y = cy - Math.sin(a) * len;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = '15px Segoe UI, Arial';
  ctx.fillText(label, x + 10, y - 8);
}
function drawWorld() {
  const c = document.getElementById('worldCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0,0,w,h);
  const cx = w * 0.5;
  const cy = h * 0.55;
  const radius = Math.min(w,h) * 0.36;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let r = 70; r < radius * 1.35; r += 70) {
    ctx.beginPath();
    ctx.arc(cx, cy, (r + drawPhase) % (radius * 1.35), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(28, cy); ctx.lineTo(w - 28, cy);
  ctx.moveTo(cx, 28); ctx.lineTo(cx, h - 28);
  ctx.stroke();

  const scale = Math.min(w,h) * 4.6;
  for (const p of micPositions) {
    const x = cx + p[0] * scale;
    const y = cy - p[1] * scale;
    ctx.fillStyle = p[2] >= 0 ? COLORS.green : '#218952';
    ctx.beginPath();
    ctx.arc(x, y, p[2] >= 0 ? 8 : 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#dcfff0';
    ctx.stroke();
  }

  drawArrow(ctx, cx, cy, radius, state.az, COLORS.cyan, `true drone ${Math.round(state.az)} deg`);
  drawArrow(ctx, cx, cy, radius * 0.72, state.bestAz, COLORS.amber, `beam ${Math.round(state.bestAz)} deg`);
  const scenario = ui.scenario.value;
  if (scenario === 'drone_tank' || scenario === 'tank') {
    drawArrow(ctx, cx, cy, radius * 0.58, state.tankAz, COLORS.red, `tank ${state.tankAz} deg`);
  }

  const srcA = state.az * Math.PI / 180;
  const sx = cx + Math.cos(srcA) * radius;
  const sy = cy - Math.sin(srcA) * radius;
  ctx.strokeStyle = COLORS.cyan;
  ctx.lineWidth = 1.5;
  for (let r = 20 + drawPhase * 1.8; r < 170; r += 36) {
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = COLORS.text;
  ctx.font = '14px Segoe UI, Arial';
  ctx.fillText(`elevation true ${Math.round(state.el)} deg / estimated ${Math.round(state.bestEl)} deg`, 24, h - 22);
}
function drawHeat() {
  const c = document.getElementById('heatCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0,0,w,h);
  const vals = [...state.scoreMap.values()];
  const minV = vals.length ? Math.min(...vals) : 0;
  const maxV = vals.length ? Math.max(...vals) : 1;
  const padL = 50, padB = 34, padT = 18, padR = 12;
  const cw = (w - padL - padR) / azGrid.length;
  const ch = (h - padT - padB) / elGrid.length;
  for (let yi = 0; yi < elGrid.length; yi++) {
    for (let xi = 0; xi < azGrid.length; xi++) {
      const az = azGrid[xi];
      const el = elGrid[yi];
      const v = state.scoreMap.get(`${az}|${el}`) || 0;
      const t = Math.max(0, Math.min(1, (v - minV) / Math.max(0.001, maxV - minV)));
      ctx.fillStyle = `hsl(${210 - t*165}, 78%, ${20 + t*45}%)`;
      const x = padL + xi * cw;
      const y = padT + (elGrid.length - 1 - yi) * ch;
      ctx.fillRect(x+1, y+1, cw-2, ch-2);
      if (az === state.bestAz && el === state.bestEl) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x+2, y+2, cw-4, ch-4);
      }
    }
  }
  ctx.fillStyle = COLORS.text;
  ctx.font = '12px Segoe UI, Arial';
  for (let i=0;i<azGrid.length;i+=3) ctx.fillText(String(azGrid[i]), padL + i*cw + 4, h - 12);
  for (let i=0;i<elGrid.length;i++) ctx.fillText(String(elGrid[i]), 14, padT + (elGrid.length - 1 - i)*ch + ch*0.58);
}
function drawWaves() {
  const c = document.getElementById('waveCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0,0,w,h);
  const rows = micPositions.length;
  const left = 48;
  const rowH = (h - 24) / rows;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let m=0;m<rows;m++) {
    const y0 = 12 + rowH * (m + 0.5);
    ctx.beginPath();
    ctx.moveTo(left, y0);
    ctx.lineTo(w - 12, y0);
    ctx.stroke();
    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px Segoe UI, Arial';
    ctx.fillText(`mic ${m+1}`, 8, y0 + 4);
    ctx.strokeStyle = m % 2 ? COLORS.cyan : COLORS.blue;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const sig = state.signals[m];
    for (let i=0;i<sig.length;i+=4) {
      const x = left + (w - left - 12) * i / (sig.length - 1);
      const y = y0 - Math.max(-1.2, Math.min(1.2, sig[i])) * rowH * 0.32;
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
}
function renderMetrics() {
  const cls = state.smoothed ? 'good' : (state.detected ? 'warn' : 'bad');
  const errCls = state.directionError <= 15 ? 'good' : (state.directionError <= 45 ? 'warn' : 'bad');
  const metrics = [
    ['Drone Decision', state.smoothed ? 'DETECTED' : (state.detected ? 'candidate' : 'no drone'), cls],
    ['Confidence', state.confidence.toFixed(3), cls],
    ['True Direction', `${Math.round(state.az)} deg / ${Math.round(state.el)} deg`, ''],
    ['Estimated Direction', `${Math.round(state.bestAz)} deg / ${Math.round(state.bestEl)} deg`, errCls],
    ['Direction Error', `${Math.round(state.directionError)} deg`, errCls],
    ['Smoothing', `${state.history.filter(Boolean).length} of ${state.history.length}`, ''],
    ['Array Type', '8 mic cube', ''],
    ['Beamforming', 'delay and sum', '']
  ];
  document.getElementById('metrics').innerHTML = metrics.map(([label, value, cls]) =>
    `<div class="metric ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`
  ).join('');
}
function updateLabels() {
  labels.az.textContent = `${ui.az.value} deg`;
  labels.el.textContent = `${ui.el.value} deg`;
  labels.drone.textContent = Number(ui.droneAmp.value).toFixed(2);
  labels.tank.textContent = Number(ui.tankAmp.value).toFixed(2);
  labels.noise.textContent = Number(ui.noiseAmp.value).toFixed(2);
}
function applyScenarioDefaults() {
  const s = ui.scenario.value;
  ui.motion.checked = s === 'moving';
  if (s === 'drone') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.0; }
  if (s === 'drone_tank') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.65; }
  if (s === 'tank') { ui.droneAmp.value = 0.0; ui.tankAmp.value = 0.95; }
  if (s === 'moving') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.18; }
  state.history = [];
  updateLabels();
}
function tick(ts) {
  if (running) {
    sampleClock += 280;
    drawPhase = (drawPhase + 1.25) % 70;
    const moving = ui.motion.checked || ui.scenario.value === 'moving';
    if (moving) {
      state.az = wrapDeg(state.az + 0.22);
      ui.az.value = Math.round(state.az);
    } else {
      state.az = Number(ui.az.value);
    }
    state.el = Number(ui.el.value);
    if (ts - lastScanMs > 180) {
      scanBeams();
      lastScanMs = ts;
    }
  }
  updateLabels();
  drawWorld();
  drawHeat();
  drawWaves();
  renderMetrics();
  requestAnimationFrame(tick);
}
ui.play.onclick = () => {
  running = !running;
  ui.play.textContent = running ? 'Pause' : 'Run';
};
ui.reset.onclick = () => {
  sampleClock = 0;
  state.az = 90;
  state.el = 40;
  ui.az.value = 90;
  ui.el.value = 40;
  state.history = [];
};
ui.scenario.onchange = applyScenarioDefaults;
for (const id of ['az','el','droneAmp','tankAmp','noiseAmp']) ui[id].oninput = updateLabels;
window.addEventListener('resize', resizeCanvases);
resizeCanvases();
applyScenarioDefaults();
scanBeams();
requestAnimationFrame(tick);
</script>
</body>
</html>"""


def write_system_simulator(out_path: Path | None = None) -> Path:
    config.ensure_output_dirs()
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (SIM_DIR / "phase3_array_system_simulator.html")
    out_path.write_text(_html(), encoding="utf-8")
    return out_path


def main():
    print(write_system_simulator())


if __name__ == "__main__":
    main()
