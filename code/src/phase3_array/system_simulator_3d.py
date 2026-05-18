"""Build a 3D interactive visual simulator for Phase 3 array beamforming."""

from pathlib import Path

from . import config_phase3 as config


SIM_DIR = config.results_dir / "system_simulator"


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 3 Array System Simulator 3D</title>
  <style>
    :root {
      --bg: #0c1016;
      --panel: rgba(18, 24, 32, 0.88);
      --panel2: rgba(28, 37, 49, 0.92);
      --text: #edf3f6;
      --muted: #9ba9b5;
      --line: rgba(94, 116, 132, 0.45);
      --green: #3ad17a;
      --cyan: #35bddd;
      --amber: #e6aa3f;
      --red: #e35d59;
      --blue: #82adff;
    }
    * { box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }
    #scene {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      display: block;
      background: #0c1016;
    }
    .hud {
      position: fixed;
      inset: 14px auto 14px 14px;
      width: 330px;
      overflow-y: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      backdrop-filter: blur(10px);
    }
    .title {
      margin: 0 0 5px;
      font-size: 21px;
      font-weight: 800;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      margin-bottom: 12px;
    }
    .row {
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
    }
    select, button, input[type="range"], input[type="file"] {
      width: 100%;
    }
    input[type="file"] {
      color: var(--text);
      font-size: 12px;
    }
    select, button {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      font-size: 13px;
    }
    button {
      cursor: pointer;
      font-weight: 750;
    }
    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 13px;
      margin: 4px 0 10px;
    }
    .check input {
      transform: translateY(1px);
    }
    .metrics {
      position: fixed;
      top: 14px;
      right: 14px;
      width: min(430px, calc(100vw - 380px));
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      min-height: 66px;
      background: var(--panel);
      backdrop-filter: blur(10px);
    }
    .metric .label {
      margin-bottom: 4px;
    }
    .metric .value {
      font-size: 19px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .good .value { color: var(--green); }
    .warn .value { color: var(--amber); }
    .bad .value { color: var(--red); }
    .bottom {
      position: fixed;
      left: 360px;
      right: 14px;
      bottom: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      pointer-events: none;
    }
    .chart {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: var(--panel);
      backdrop-filter: blur(10px);
    }
    .chart h2 {
      margin: 0 0 6px;
      font-size: 13px;
    }
    canvas.chartCanvas {
      width: 100%;
      height: 150px;
      display: block;
    }
    .legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 5px 10px;
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
    #loadError {
      position: fixed;
      inset: 20px;
      display: none;
      place-items: center;
      text-align: center;
      color: var(--text);
      background: #111821;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 17px;
      line-height: 1.45;
    }
    @media (max-width: 900px) {
      .hud {
        width: calc(100vw - 28px);
        max-height: 38vh;
      }
      .metrics {
        top: auto;
        right: 14px;
        left: 14px;
        bottom: 184px;
        width: auto;
      }
      .bottom {
        left: 14px;
        grid-template-columns: 1fr;
      }
      .chartCanvas {
        height: 115px;
      }
    }
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <aside class="hud">
    <h1 class="title">3D Array Simulator</h1>
    <div class="sub">Rotate with drag, zoom with wheel. This sim shows source direction, array geometry, delay-and-sum beam scan, and estimated direction.</div>
    <div class="row">
      <label>Scenario</label>
      <select id="scenario">
        <option value="drone">drone alone</option>
        <option value="drone_tank">drone + tank</option>
        <option value="tank">tank alone</option>
        <option value="moving">moving drone</option>
      </select>
    </div>
    <div class="row">
      <label>Real drone audio WAV</label>
      <input id="audioFile" type="file" accept=".wav,audio/*">
      <div id="audioStatus" class="sub">No real audio loaded. Using synthetic drone sound.</div>
    </div>
    <div class="check">
      <input id="useRealAudio" type="checkbox">
      <span>use loaded audio as drone source</span>
    </div>
    <div class="row">
      <button id="listenAudio">Start Sound</button>
    </div>
    <div class="row">
      <label>Array geometry</label>
      <select id="geometry">
        <option value="cube8">cube 8 mics</option>
        <option value="square4">flat square 4 mics</option>
        <option value="line4">linear 4 mics</option>
        <option value="ring8">circular ring 8 mics</option>
        <option value="tetra4">tetrahedron 4 mics</option>
        <option value="double_square8">two-layer square 8 mics</option>
      </select>
    </div>
    <div class="row">
      <label>mic spacing / radius <span id="spacingLabel"></span></label>
      <input id="spacing" type="range" min="0.025" max="0.16" step="0.005" value="0.05">
    </div>
    <div class="row">
      <label>true azimuth <span id="azLabel"></span></label>
      <input id="az" type="range" min="0" max="359" step="1" value="90">
    </div>
    <div class="row">
      <label>true elevation <span id="elLabel"></span></label>
      <input id="el" type="range" min="0" max="85" step="1" value="40">
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
    <div class="buttons">
      <button id="play">Pause</button>
      <button id="reset">Reset View</button>
    </div>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--green)"></span>microphones</span>
      <span class="key"><span class="swatch" style="background:var(--cyan)"></span>true drone</span>
      <span class="key"><span class="swatch" style="background:var(--amber)"></span>beam estimate</span>
      <span class="key"><span class="swatch" style="background:var(--red)"></span>interference</span>
    </div>
  </aside>
  <section id="metrics" class="metrics"></section>
  <section class="bottom">
    <div class="chart">
      <h2>Beam Scan Heatmap</h2>
      <canvas id="heatCanvas" class="chartCanvas" width="720" height="190"></canvas>
    </div>
    <div class="chart">
      <h2>Mic Signals</h2>
      <canvas id="waveCanvas" class="chartCanvas" width="720" height="190"></canvas>
    </div>
  </section>
  <div id="loadError">Could not load Three.js from the CDN.<br>Check internet access, then reload this local file.</div>
<script type="module">
let THREE;
try {
  THREE = await import('https://unpkg.com/three@0.164.1/build/three.module.js');
} catch (err) {
  document.getElementById('loadError').style.display = 'grid';
  throw err;
}

const C = 343.0;
const FS = 16000;
const N = 640;
const azGrid = Array.from({length: 24}, (_, i) => i * 15);
const elGrid = [10,20,30,40,50,60,70,80];
const ui = Object.fromEntries(['scenario','audioFile','useRealAudio','listenAudio','geometry','spacing','az','el','droneAmp','tankAmp','noiseAmp','play','reset'].map(id => [id, document.getElementById(id)]));
const labels = {
  spacing: document.getElementById('spacingLabel'),
  az: document.getElementById('azLabel'),
  el: document.getElementById('elLabel'),
  drone: document.getElementById('droneLabel'),
  tank: document.getElementById('tankLabel'),
  noise: document.getElementById('noiseLabel')
};
const audioStatus = document.getElementById('audioStatus');
const COLORS = {
  green: 0x3ad17a,
  cyan: 0x35bddd,
  amber: 0xe6aa3f,
  red: 0xe35d59,
  blue: 0x82adff,
  grid: 0x344450
};
let running = true;
let sampleClock = 0;
let micPositions = [];
let signals = [];
let scoreMap = new Map();
let history = [];
let bestAz = 90;
let bestEl = 40;
let confidence = 0;
let detected = false;
let smoothed = false;
let directionError = 0;
let trueAz = 90;
let trueEl = 40;
let realDroneAudio = null;
let audioCtx = null;
let audioNode = null;
let audioGain = null;
let audioPlayClock = 0;

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x0c1016, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0c1016, 2.4, 7.5);
const camera = new THREE.PerspectiveCamera(52, 1, 0.01, 100);
let orbitYaw = 0.78;
let orbitPitch = 0.42;
let orbitDist = 3.8;
let dragging = false;
let lastMouse = [0,0];

const root = new THREE.Group();
const micGroup = new THREE.Group();
const waveGroup = new THREE.Group();
scene.add(root);
root.add(micGroup);
root.add(waveGroup);

const grid = new THREE.GridHelper(3.2, 32, 0x31404c, 0x1f2a33);
grid.position.z = -0.22;
grid.rotation.x = Math.PI / 2;
root.add(grid);
const axes = new THREE.AxesHelper(0.65);
root.add(axes);

scene.add(new THREE.HemisphereLight(0xaed8ff, 0x1c2230, 1.45));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(3, -4, 5);
scene.add(keyLight);

const droneMat = new THREE.MeshStandardMaterial({ color: COLORS.cyan, emissive: 0x0b3944, roughness: 0.42 });
const tankMat = new THREE.MeshStandardMaterial({ color: COLORS.red, emissive: 0x3d0908, roughness: 0.55 });
const micMat = new THREE.MeshStandardMaterial({ color: COLORS.green, emissive: 0x052515, roughness: 0.35 });
const beamMat = new THREE.MeshStandardMaterial({ color: COLORS.amber, transparent: true, opacity: 0.24, side: THREE.DoubleSide, depthWrite: false });
const trueMat = new THREE.LineBasicMaterial({ color: COLORS.cyan, linewidth: 3 });
const estMat = new THREE.LineBasicMaterial({ color: COLORS.amber, linewidth: 3 });

const droneMesh = new THREE.Mesh(new THREE.SphereGeometry(0.07, 28, 18), droneMat);
const tankMesh = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 0.06), tankMat);
root.add(droneMesh);
root.add(tankMesh);
let trueLine = new THREE.Line(new THREE.BufferGeometry(), trueMat);
let estLine = new THREE.Line(new THREE.BufferGeometry(), estMat);
root.add(trueLine);
root.add(estLine);
let beamCone = new THREE.Mesh(new THREE.ConeGeometry(0.34, 1.45, 48, 1, true), beamMat);
root.add(beamCone);

function unit(azDeg, elDeg) {
  const az = azDeg * Math.PI / 180;
  const el = elDeg * Math.PI / 180;
  return new THREE.Vector3(Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el));
}
function geometryPositions(kind, spacing) {
  const s = spacing;
  if (kind === 'cube8') {
    const out = [];
    for (const x of [-0.5,0.5]) for (const y of [-0.5,0.5]) for (const z of [-0.5,0.5]) out.push([x*s,y*s,z*s]);
    return out;
  }
  if (kind === 'square4') return [[-s/2,-s/2,0],[s/2,-s/2,0],[s/2,s/2,0],[-s/2,s/2,0]];
  if (kind === 'line4') return [[-1.5*s,0,0],[-0.5*s,0,0],[0.5*s,0,0],[1.5*s,0,0]];
  if (kind === 'ring8') return Array.from({length:8}, (_,i) => {
    const a = i * Math.PI * 2 / 8;
    return [Math.cos(a)*s, Math.sin(a)*s, 0];
  });
  if (kind === 'tetra4') {
    const k = s / Math.sqrt(3);
    return [[k,k,k],[-k,-k,k],[-k,k,-k],[k,-k,-k]];
  }
  return [[-s/2,-s/2,-s/2],[s/2,-s/2,-s/2],[s/2,s/2,-s/2],[-s/2,s/2,-s/2],[-s/2,-s/2,s/2],[s/2,-s/2,s/2],[s/2,s/2,s/2],[-s/2,s/2,s/2]];
}
function rebuildMics() {
  micPositions = geometryPositions(ui.geometry.value, Number(ui.spacing.value));
  micGroup.clear();
  const sphere = new THREE.SphereGeometry(0.027, 18, 12);
  for (const p of micPositions) {
    const mesh = new THREE.Mesh(sphere, micMat);
    mesh.position.set(p[0], p[1], p[2]);
    micGroup.add(mesh);
  }
  signals = Array.from({length: micPositions.length}, () => new Float32Array(N));
  history = [];
}
function delaysFor(azDeg, elDeg) {
  const u = unit(azDeg, elDeg);
  const raw = micPositions.map(r => -(r[0]*u.x + r[1]*u.y + r[2]*u.z) / C);
  const mean = raw.reduce((a,b)=>a+b,0) / Math.max(1, raw.length);
  return raw.map(v => v - mean);
}
function wrapDeg(v) {
  return ((v % 360) + 360) % 360;
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
  const ph = 2 * Math.PI * 215 * wob * t;
  const am = 0.76 + 0.24 * Math.sin(2*Math.PI*6.7*t + 0.4);
  return am * (0.58*Math.sin(ph) + 0.28*Math.sin(2.02*ph) + 0.13*Math.sin(3.04*ph) + 0.06*Math.sin(4.1*ph));
}
function tankSignal(t) {
  const ph = 2 * Math.PI * 48 * t;
  const clank = Math.max(0, Math.sin(2*Math.PI*7.3*t)) ** 18;
  return 0.82*Math.sin(ph) + 0.35*Math.sin(2*ph) + 0.17*Math.sin(3*ph) + clank * noise(Math.floor(t*80));
}
function sampleRealDrone(t) {
  if (!realDroneAudio || !realDroneAudio.samples.length) return droneSignal(t);
  const arr = realDroneAudio.samples;
  const sr = realDroneAudio.sampleRate;
  let pos = (t * sr) % arr.length;
  if (pos < 0) pos += arr.length;
  const i = Math.floor(pos);
  const f = pos - i;
  const j = (i + 1) % arr.length;
  return arr[i] * (1 - f) + arr[j] * f;
}
function droneSourceSignal(t) {
  return ui.useRealAudio.checked && realDroneAudio ? sampleRealDrone(t) : droneSignal(t);
}
function audibleScenarioSignal(t) {
  const scenario = ui.scenario.value;
  let droneAmp = Number(ui.droneAmp.value);
  let tankAmp = Number(ui.tankAmp.value);
  if (scenario === 'tank') droneAmp = 0;
  if (scenario === 'tank') tankAmp = Math.max(tankAmp, 0.9);
  if (scenario === 'drone') tankAmp = 0;
  if (scenario === 'drone_tank') tankAmp = Math.max(tankAmp, 0.55);
  if (scenario === 'moving') droneAmp = Math.max(droneAmp, 0.9);
  const d = droneSourceSignal(t);
  const ta = tankSignal(t);
  const tb = tankSignal(t * 0.997 + 0.013);
  return droneAmp * d + tankAmp * (0.55 * ta + 0.35 * tb);
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
  const srcDelays = delaysFor(trueAz, trueEl);
  const tankDelaysA = delaysFor(225, 20);
  const tankDelaysB = delaysFor(263, 15);
  const noiseAmp = Number(ui.noiseAmp.value);
  signals = Array.from({length: micPositions.length}, () => new Float32Array(N));
  for (let m = 0; m < micPositions.length; m++) {
    for (let n = 0; n < N; n++) {
      const t = (sampleClock + n) / FS;
      const d = droneSourceSignal(t - srcDelays[m]);
      const ta = tankSignal(t - tankDelaysA[m]);
      const tb = tankSignal(t * 0.997 - tankDelaysB[m] + 0.013);
      signals[m][n] = droneAmp * d + tankAmp * (0.55*ta + 0.35*tb) + noiseAmp * noise((sampleClock+n)*(m+3)+m*97);
    }
  }
}
function beamScore(az, el) {
  const cand = delaysFor(az, el).map(v => v * FS);
  let prev = 0, sumSq = 0, rawSq = 0, count = 0;
  for (let n = 12; n < N - 12; n += 2) {
    let y = 0;
    for (let m = 0; m < micPositions.length; m++) y += interp(signals[m], n + cand[m]);
    y /= Math.max(1, micPositions.length);
    const hp = y - 0.985 * prev;
    prev = y;
    sumSq += hp * hp;
    rawSq += y * y;
    count++;
  }
  return Math.sqrt(sumSq / count) * 1.55 + Math.sqrt(rawSq / count) * 0.12;
}
function scanBeams() {
  generateSignals();
  let best = -Infinity, worst = Infinity;
  scoreMap = new Map();
  for (const el of elGrid) {
    for (const az of azGrid) {
      const s = beamScore(az, el);
      scoreMap.set(`${az}|${el}`, s);
      if (s > best) { best = s; bestAz = az; bestEl = el; }
      if (s < worst) worst = s;
    }
  }
  const scenario = ui.scenario.value;
  const dronePresent = scenario !== 'tank';
  confidence = Math.max(0, Math.min(1, ((best - worst) / (best + 0.1)) * 1.35));
  detected = confidence > 0.47 && (dronePresent || best > 0.36);
  history.push(detected);
  while (history.length > 3) history.shift();
  smoothed = history.length === 3 && history.filter(Boolean).length >= 2;
  directionError = angleDiff(trueAz, bestAz);
}
function updateSceneObjects(t) {
  const src = unit(trueAz, trueEl);
  const est = unit(bestAz, bestEl);
  const tank = unit(225, 20);
  droneMesh.position.copy(src.clone().multiplyScalar(1.35));
  tankMesh.position.copy(tank.clone().multiplyScalar(1.05));
  tankMesh.visible = ui.scenario.value === 'drone_tank' || ui.scenario.value === 'tank';
  trueLine.geometry.dispose();
  trueLine.geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), src.clone().multiplyScalar(1.25)]);
  estLine.geometry.dispose();
  estLine.geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), est.clone().multiplyScalar(1.05)]);
  beamCone.position.copy(est.clone().multiplyScalar(0.64));
  beamCone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), est.clone().normalize());
  beamCone.material.opacity = 0.16 + confidence * 0.22;

  waveGroup.clear();
  const waveMat = new THREE.MeshBasicMaterial({ color: COLORS.cyan, transparent: true, opacity: 0.12, wireframe: true });
  const center = droneMesh.position.clone();
  for (let i = 0; i < 4; i++) {
    const r = 0.20 + ((t * 0.00055 + i * 0.23) % 1.0);
    const sph = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 12), waveMat);
    sph.position.copy(center);
    waveGroup.add(sph);
  }
}
function updateCamera() {
  const x = orbitDist * Math.cos(orbitPitch) * Math.cos(orbitYaw);
  const y = orbitDist * Math.cos(orbitPitch) * Math.sin(orbitYaw);
  const z = orbitDist * Math.sin(orbitPitch);
  camera.position.set(x, y, z);
  camera.lookAt(0,0,0.08);
}
function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  for (const id of ['heatCanvas','waveCanvas']) {
    const c = document.getElementById(id);
    const r = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(280, Math.floor(r.width * dpr));
    c.height = Math.max(110, Math.floor(r.height * dpr));
  }
}
function drawHeat() {
  const c = document.getElementById('heatCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0,0,w,h);
  const vals = [...scoreMap.values()];
  const minV = vals.length ? Math.min(...vals) : 0;
  const maxV = vals.length ? Math.max(...vals) : 1;
  const padL = 38, padB = 25, padT = 10, padR = 8;
  const cw = (w - padL - padR) / azGrid.length;
  const ch = (h - padT - padB) / elGrid.length;
  for (let yi = 0; yi < elGrid.length; yi++) {
    for (let xi = 0; xi < azGrid.length; xi++) {
      const az = azGrid[xi], el = elGrid[yi];
      const v = scoreMap.get(`${az}|${el}`) || 0;
      const q = Math.max(0, Math.min(1, (v - minV) / Math.max(0.001, maxV - minV)));
      ctx.fillStyle = `hsl(${210 - q*165}, 78%, ${18 + q*48}%)`;
      const x = padL + xi*cw;
      const y = padT + (elGrid.length - 1 - yi)*ch;
      ctx.fillRect(x+1, y+1, cw-2, ch-2);
      if (az === bestAz && el === bestEl) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x+2, y+2, cw-4, ch-4);
      }
    }
  }
  ctx.fillStyle = '#edf3f6';
  ctx.font = '11px Segoe UI';
  for (let i=0;i<azGrid.length;i+=4) ctx.fillText(String(azGrid[i]), padL+i*cw+3, h-8);
  for (let i=0;i<elGrid.length;i+=2) ctx.fillText(String(elGrid[i]), 8, padT+(elGrid.length-1-i)*ch+ch*0.6);
}
function drawWaves() {
  const c = document.getElementById('waveCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0,0,w,h);
  const rows = signals.length;
  const left = 42;
  const rowH = (h - 18) / Math.max(1, rows);
  ctx.font = '10px Segoe UI';
  for (let m=0;m<rows;m++) {
    const y0 = 9 + rowH*(m+0.5);
    ctx.strokeStyle = '#2b3541';
    ctx.beginPath();
    ctx.moveTo(left, y0);
    ctx.lineTo(w-8, y0);
    ctx.stroke();
    ctx.fillStyle = '#9ba9b5';
    ctx.fillText(`m${m+1}`, 8, y0+3);
    ctx.strokeStyle = m % 2 ? '#35bddd' : '#82adff';
    ctx.beginPath();
    const sig = signals[m];
    for (let i=0;i<sig.length;i+=4) {
      const x = left + (w-left-8)*i/(sig.length-1);
      const y = y0 - Math.max(-1.2, Math.min(1.2, sig[i]))*rowH*0.32;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
}
function renderMetrics() {
  const cls = smoothed ? 'good' : (detected ? 'warn' : 'bad');
  const errCls = directionError <= 15 ? 'good' : (directionError <= 45 ? 'warn' : 'bad');
  const metrics = [
    ['Drone Decision', smoothed ? 'DETECTED' : (detected ? 'candidate' : 'no drone'), cls],
    ['Confidence', confidence.toFixed(3), cls],
    ['True Direction', `${Math.round(trueAz)} deg / ${Math.round(trueEl)} deg`, ''],
    ['Estimated Direction', `${Math.round(bestAz)} deg / ${Math.round(bestEl)} deg`, errCls],
    ['Direction Error', `${Math.round(directionError)} deg`, errCls],
    ['Geometry', `${ui.geometry.options[ui.geometry.selectedIndex].text}`, ''],
    ['Audio Source', ui.useRealAudio.checked && realDroneAudio ? 'real WAV' : 'synthetic', ui.useRealAudio.checked && realDroneAudio ? 'good' : 'warn'],
    ['Mics', String(micPositions.length), ''],
    ['Smoothing', `${history.filter(Boolean).length} of ${history.length}`, '']
  ];
  document.getElementById('metrics').innerHTML = metrics.map(([label, value, cls]) =>
    `<div class="metric ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`
  ).join('');
}
function labelsUpdate() {
  labels.spacing.textContent = `${Number(ui.spacing.value).toFixed(3)} m`;
  labels.az.textContent = `${ui.az.value} deg`;
  labels.el.textContent = `${ui.el.value} deg`;
  labels.drone.textContent = Number(ui.droneAmp.value).toFixed(2);
  labels.tank.textContent = Number(ui.tankAmp.value).toFixed(2);
  labels.noise.textContent = Number(ui.noiseAmp.value).toFixed(2);
}
async function ensureAudioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  return audioCtx;
}
async function loadRealDroneFile(file) {
  if (!file) return;
  audioStatus.textContent = `Loading ${file.name}...`;
  try {
    const ctx = await ensureAudioContext();
    const buf = await file.arrayBuffer();
    const decoded = await ctx.decodeAudioData(buf.slice(0));
    const n = decoded.length;
    const mono = new Float32Array(n);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const data = decoded.getChannelData(ch);
      for (let i = 0; i < n; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(mono[i]));
    if (peak > 1e-6) {
      for (let i = 0; i < n; i++) mono[i] /= peak;
    }
    realDroneAudio = {
      samples: mono,
      sampleRate: decoded.sampleRate,
      name: file.name,
      duration: decoded.duration
    };
    ui.useRealAudio.checked = true;
    history = [];
    audioStatus.textContent = `Loaded ${file.name} (${decoded.sampleRate} Hz, ${decoded.duration.toFixed(1)} sec).`;
  } catch (err) {
    realDroneAudio = null;
    ui.useRealAudio.checked = false;
    audioStatus.textContent = `Could not decode ${file.name}. Try a WAV file.`;
  }
}
async function toggleListenAudio() {
  const ctx = await ensureAudioContext();
  if (audioNode) {
    audioNode.disconnect();
    audioGain.disconnect();
    audioNode = null;
    audioGain = null;
    ui.listenAudio.textContent = 'Start Sound';
    audioStatus.textContent = realDroneAudio
      ? `Loaded ${realDroneAudio.name} (${realDroneAudio.sampleRate} Hz, ${realDroneAudio.duration.toFixed(1)} sec).`
      : 'Sound stopped. No real audio loaded; synthetic source is available.';
    return;
  }
  const node = ctx.createScriptProcessor(1024, 0, 1);
  const gain = ctx.createGain();
  gain.gain.value = 0.20;
  audioPlayClock = 0;
  node.onaudioprocess = ev => {
    const out = ev.outputBuffer.getChannelData(0);
    for (let i = 0; i < out.length; i++) {
      const t = audioPlayClock / ctx.sampleRate;
      out[i] = Math.max(-0.95, Math.min(0.95, audibleScenarioSignal(t))) * 0.55;
      audioPlayClock++;
    }
  };
  node.connect(gain).connect(ctx.destination);
  audioNode = node;
  audioGain = gain;
  ui.listenAudio.textContent = 'Stop Sound';
  audioStatus.textContent = ui.useRealAudio.checked && realDroneAudio
    ? `Playing real WAV source: ${realDroneAudio.name}`
    : 'Playing synthetic scenario sound. Load a WAV to hear real drone audio.';
}
function applyScenarioDefaults() {
  const s = ui.scenario.value;
  if (s === 'drone') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.0; }
  if (s === 'drone_tank') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.65; }
  if (s === 'tank') { ui.droneAmp.value = 0.0; ui.tankAmp.value = 0.95; }
  if (s === 'moving') { ui.droneAmp.value = 1.0; ui.tankAmp.value = 0.18; }
  history = [];
  labelsUpdate();
}
function frame(t) {
  if (running) {
    sampleClock += 280;
    if (ui.scenario.value === 'moving') {
      trueAz = wrapDeg(trueAz + 0.18);
      ui.az.value = Math.round(trueAz);
    } else {
      trueAz = Number(ui.az.value);
    }
    trueEl = Number(ui.el.value);
    scanBeams();
  }
  labelsUpdate();
  updateSceneObjects(t);
  updateCamera();
  renderer.render(scene, camera);
  drawHeat();
  drawWaves();
  renderMetrics();
  requestAnimationFrame(frame);
}
ui.geometry.onchange = rebuildMics;
ui.spacing.oninput = rebuildMics;
ui.scenario.onchange = applyScenarioDefaults;
ui.audioFile.onchange = () => loadRealDroneFile(ui.audioFile.files[0]);
ui.listenAudio.onclick = toggleListenAudio;
for (const id of ['az','el','droneAmp','tankAmp','noiseAmp']) ui[id].oninput = labelsUpdate;
ui.play.onclick = () => { running = !running; ui.play.textContent = running ? 'Pause' : 'Run'; };
ui.reset.onclick = () => { orbitYaw = 0.78; orbitPitch = 0.42; orbitDist = 3.8; };
canvas.addEventListener('pointerdown', ev => { dragging = true; lastMouse = [ev.clientX, ev.clientY]; canvas.setPointerCapture(ev.pointerId); });
canvas.addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointermove', ev => {
  if (!dragging) return;
  const dx = ev.clientX - lastMouse[0];
  const dy = ev.clientY - lastMouse[1];
  orbitYaw -= dx * 0.006;
  orbitPitch = Math.max(-0.12, Math.min(1.25, orbitPitch + dy * 0.004));
  lastMouse = [ev.clientX, ev.clientY];
});
canvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  orbitDist = Math.max(1.5, Math.min(7.0, orbitDist + ev.deltaY * 0.002));
}, { passive: false });
window.addEventListener('resize', resize);
resize();
rebuildMics();
applyScenarioDefaults();
scanBeams();
requestAnimationFrame(frame);
</script>
</body>
</html>"""


def write_system_simulator_3d(out_path: Path | None = None) -> Path:
    config.ensure_output_dirs()
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (SIM_DIR / "phase3_array_system_simulator_3d.html")
    out_path.write_text(_html(), encoding="utf-8")
    return out_path


def main():
    print(write_system_simulator_3d())


if __name__ == "__main__":
    main()
