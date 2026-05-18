"""Generate a standalone visual HTML dashboard for Phase 3 simulations."""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from . import config_phase3 as config


VIS_DIR = config.results_dir / "visual_simulator"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_sim_tag() -> str | None:
    pattern = re.compile(r"array_sim_test_(\d{8}_\d{6})_.+\.json$")
    tags = []
    for path in Path(config.array_raw_dir).glob("array_sim_test_*.json"):
        m = pattern.match(path.name)
        if m:
            tags.append(m.group(1))
    return max(tags) if tags else None


def _scenario_sort_key(item):
    order = {
        "drone": 0,
        "drone_tank": 1,
        "drone_engine": 2,
        "drone_crowd": 3,
        "tank": 4,
        "engine": 5,
        "crowd": 6,
        "pure_noise": 7,
        "noise": 8,
    }
    return order.get(item.get("scenario", ""), 99)


def collect_visual_data(tag: str | None = None) -> dict:
    tag = tag or _latest_sim_tag()
    if tag is None:
        raise FileNotFoundError("No array_sim_test_*.json files found in data/array_raw.")

    scenarios = []
    for truth_path in sorted(Path(config.array_raw_dir).glob(f"array_sim_test_{tag}_*.json")):
        truth = _read_json(truth_path)
        stem = truth_path.stem
        summary_path = config.beam_scan_results_dir / f"{stem}_summary.json"
        per_window_path = config.beam_scan_results_dir / f"{stem}_per_window.csv"
        direction_path = config.beam_scan_results_dir / f"{stem}_direction_scores.csv"
        comparison_path = config.comparisons_dir / f"{stem}_array_vs_single_channel_summary.json"
        comparison_csv = config.comparisons_dir / f"{stem}_array_vs_single_channel.csv"

        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        comparison = _read_json(comparison_path) if comparison_path.exists() else {}
        scenarios.append({
            "stem": stem,
            "file": f"{stem}.wav",
            "scenario": truth.get("scenario", stem),
            "truth": truth,
            "summary": summary,
            "perWindow": _read_csv(per_window_path),
            "directionScores": _read_csv(direction_path),
            "comparison": comparison,
            "comparisonTimeline": _read_csv(comparison_csv),
        })

    if not scenarios:
        raise FileNotFoundError(
            f"No evaluated Phase 3 results found for tag {tag}. Run the array evaluator first."
        )

    scenarios.sort(key=_scenario_sort_key)
    return {
        "tag": tag,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "geometryMode": config.geometry_mode,
            "micSpacingM": config.mic_spacing_m,
            "azimuthGridDeg": config.azimuth_grid_deg,
            "elevationGridDeg": config.elevation_grid_deg,
            "smoothingMode": config.smoothing_mode,
            "topKBeamsForHybrid": config.top_k_beams_for_hybrid,
        },
        "scenarios": scenarios,
    }


def _html_template(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 3 Array Visual Simulator</title>
  <style>
    :root {{
      --bg: #101215;
      --panel: #181c21;
      --panel-2: #20262d;
      --text: #eef3f6;
      --muted: #9aa8b1;
      --line: #34414b;
      --green: #38c172;
      --cyan: #37b7d8;
      --amber: #e2a93b;
      --red: #df5b57;
      --blue: #6ea8fe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    header {{
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: #14181d;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      font-weight: 700;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    main {{
      display: grid;
      grid-template-columns: 310px minmax(520px, 1fr);
      min-height: calc(100vh - 78px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      padding: 16px;
      background: #12161a;
    }}
    .scenario-list {{
      display: grid;
      gap: 8px;
    }}
    button.scenario {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.scenario.active {{
      border-color: var(--cyan);
      background: #18303a;
    }}
    button.scenario .meta {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .content {{
      padding: 16px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .top-grid {{
      display: grid;
      grid-template-columns: minmax(420px, 1.1fr) minmax(300px, 0.9fr);
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 700;
    }}
    canvas {{
      width: 100%;
      display: block;
      background: #0d1013;
      border-radius: 6px;
    }}
    #arrayCanvas {{ height: 420px; }}
    #timelineCanvas {{ height: 220px; }}
    #heatCanvas {{ height: 260px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid #2d3740;
      border-radius: 6px;
      padding: 10px;
      min-height: 72px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric .value {{
      font-size: 20px;
      font-weight: 700;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .metric.good .value {{ color: var(--green); }}
    .metric.warn .value {{ color: var(--amber); }}
    .metric.bad .value {{ color: var(--red); }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .intervals {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 8px;
      overflow-wrap: anywhere;
    }}
    .footer-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: none; border-bottom: 1px solid var(--line); }}
      .top-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Phase 3 Array Visual Simulator</h1>
    <div class="sub">Animated passive delay-and-sum beamforming view. Tag <span id="tag"></span>. No model training, no transmission, recorded/simulated receive-only audio.</div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Scenarios</h2>
        <div id="scenarioList" class="scenario-list"></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <h2>Run Info</h2>
        <div class="sub" id="runInfo"></div>
      </div>
    </aside>
    <section class="content">
      <div class="top-grid">
        <div class="panel">
          <h2>Array And Direction</h2>
          <canvas id="arrayCanvas" width="900" height="700"></canvas>
          <div class="legend">
            <span class="key"><span class="swatch" style="background:var(--cyan)"></span>true source</span>
            <span class="key"><span class="swatch" style="background:var(--amber)"></span>estimated beam</span>
            <span class="key"><span class="swatch" style="background:var(--green)"></span>microphones</span>
          </div>
        </div>
        <div class="panel">
          <h2>Decision Summary</h2>
          <div class="metrics" id="metrics"></div>
          <div class="intervals" id="intervals"></div>
        </div>
      </div>
      <div class="panel">
        <h2>Score Timeline</h2>
        <canvas id="timelineCanvas" width="1200" height="320"></canvas>
      </div>
      <div class="panel">
        <h2>Direction Heatmap</h2>
        <canvas id="heatCanvas" width="1200" height="360"></canvas>
        <div class="footer-note">Each cell is average hybrid score for an azimuth/elevation direction; brighter means stronger.</div>
      </div>
    </section>
  </main>
<script>
const DATA = {data};
const COLORS = {{
  text: '#eef3f6',
  muted: '#9aa8b1',
  grid: '#2a333b',
  green: '#38c172',
  cyan: '#37b7d8',
  amber: '#e2a93b',
  red: '#df5b57',
  blue: '#6ea8fe'
}};
let activeIndex = 0;
let phase = 0;

function fmtPct(v) {{ return `${{Number(v || 0).toFixed(1)}}%`; }}
function fmt(v, d=3) {{ return Number(v || 0).toFixed(d); }}
function meanKey(rows, key) {{
  const vals = rows.map(r => Number(r[key])).filter(Number.isFinite);
  return vals.length ? vals.reduce((a,b)=>a+b,0) / vals.length : 0;
}}
function countKey(rows, key) {{
  return rows.reduce((n, r) => n + (Number(r[key] || 0) ? 1 : 0), 0);
}}
function topReason(rows) {{
  const counts = new Map();
  rows.forEach(r => {{
    const v = String(r.fusion_reason_best || '').trim();
    if (v) counts.set(v, (counts.get(v) || 0) + 1);
  }});
  let best = 'n/a', bestN = 0;
  counts.forEach((n, k) => {{ if (n > bestN) {{ best = k; bestN = n; }} }});
  return bestN ? `${{best}} (${{bestN}})` : 'n/a';
}}
function scenarioLabel(s) {{ return s.scenario.replaceAll('_', '+'); }}
function truthSource(s) {{
  const sources = s.truth.sources || [];
  return sources.find(x => x.kind === 'drone') || sources[0] || {{}};
}}
function sourceId(s, kind) {{
  const src = (s.truth.sources || []).find(x => x.kind === kind);
  return src && src.source_id ? String(src.source_id) : 'n/a';
}}
function estimatedDir(s) {{
  const d = s.summary.most_common_direction || {{}};
  return {{ az: Number(d.az || 0), el: Number(d.el || 0), count: Number(d.count || 0) }};
}}
function isPositive(s) {{
  return (s.truth.sources || []).some(x => x.kind === 'drone');
}}

function init() {{
  document.getElementById('tag').textContent = DATA.tag;
  document.getElementById('runInfo').innerHTML =
    `Geometry: <b>${{DATA.config.geometryMode}}</b><br>` +
    `Mic spacing: <b>${{DATA.config.micSpacingM}} m</b><br>` +
    `Grid: <b>${{DATA.config.azimuthGridDeg.length}} az x ${{DATA.config.elevationGridDeg.length}} el</b><br>` +
    `Smoothing: <b>${{DATA.config.smoothingMode}}</b><br>` +
    `Created: <b>${{DATA.createdAt}}</b>`;
  const list = document.getElementById('scenarioList');
  DATA.scenarios.forEach((s, i) => {{
    const btn = document.createElement('button');
    btn.className = 'scenario' + (i === activeIndex ? ' active' : '');
    btn.innerHTML = `${{scenarioLabel(s)}}<span class="meta">${{s.file}}</span>`;
    btn.onclick = () => {{
      activeIndex = i;
      document.querySelectorAll('button.scenario').forEach((b, bi) => b.classList.toggle('active', bi === i));
      renderAll();
    }};
    list.appendChild(btn);
  }});
  resizeCanvases();
  renderAll();
  requestAnimationFrame(tick);
}}

function resizeCanvases() {{
  for (const id of ['arrayCanvas', 'timelineCanvas', 'heatCanvas']) {{
    const c = document.getElementById(id);
    const rect = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(320, Math.floor(rect.width * dpr));
    c.height = Math.max(180, Math.floor(rect.height * dpr));
  }}
}}

function drawArrow(ctx, cx, cy, len, azDeg, color, label) {{
  const a = (Number(azDeg) || 0) * Math.PI / 180;
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
}}

function drawArray() {{
  const s = DATA.scenarios[activeIndex];
  const c = document.getElementById('arrayCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0d1013';
  ctx.fillRect(0,0,w,h);
  const cx = w * 0.5, cy = h * 0.54;
  const scale = Math.min(w, h) * 3.7;

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let r = 60; r < Math.min(w,h)*0.55; r += 60) {{
    ctx.beginPath();
    ctx.arc(cx, cy, r + (phase % 60), 0, Math.PI * 2);
    ctx.stroke();
  }}
  ctx.beginPath();
  ctx.moveTo(30, cy); ctx.lineTo(w-30, cy);
  ctx.moveTo(cx, 30); ctx.lineTo(cx, h-30);
  ctx.stroke();

  const mics = s.truth.mic_positions || [];
  for (const p of mics) {{
    const x = cx + Number(p[0]) * scale;
    const y = cy - Number(p[1]) * scale;
    const z = Number(p[2] || 0);
    ctx.fillStyle = z >= 0 ? COLORS.green : '#1f8b52';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#d9fff0';
    ctx.lineWidth = 1;
    ctx.stroke();
  }}

  const src = truthSource(s);
  const est = estimatedDir(s);
  drawArrow(ctx, cx, cy, Math.min(w,h)*0.35, src.az_deg || s.truth.primary_az_deg || 0, COLORS.cyan, `truth az ${{src.az_deg ?? s.truth.primary_az_deg}}`);
  drawArrow(ctx, cx, cy, Math.min(w,h)*0.26, est.az, COLORS.amber, `beam az ${{est.az}}`);

  ctx.fillStyle = COLORS.muted;
  ctx.font = '14px Segoe UI, Arial';
  ctx.fillText(`elevation truth ${{src.el_deg ?? s.truth.primary_el_deg}} deg / estimated ${{est.el}} deg`, 24, h - 24);
}}

function metricClass(label, value, positive) {{
  if (label.includes('False') || label.includes('Tank')) return value > 0 ? 'bad' : 'good';
  if (label.includes('Detected')) return value > 0 ? 'good' : (positive ? 'bad' : 'good');
  return '';
}}

function renderMetrics() {{
  const s = DATA.scenarios[activeIndex];
  const sum = s.summary;
  const comp = s.comparison || {{}};
  const rows = s.perWindow || [];
  const positive = isPositive(s);
  const det = Number(sum.smoothed_detected_windows || 0);
  const total = Number(sum.num_windows || 1);
  const metrics = [
    ['Scenario', scenarioLabel(s), ''],
    ['Smoothed Detected', `${{det}} / ${{total}}`, metricClass('Detected', det, positive)],
    ['Max Score', fmt(sum.max_score), ''],
    ['Average Score', fmt(sum.average_score), ''],
    ['Mean Option 2', fmt(meanKey(rows, 'option2_score_best')), ''],
    ['Mean Option 3', fmt(meanKey(rows, 'option3_score_best')), ''],
    ['Vetoed Windows', String(countKey(rows, 'vetoed_best')), countKey(rows, 'vetoed_best') ? 'warn' : 'good'],
    ['Top Fusion Reason', topReason(rows), ''],
    ['Truth Direction', `az ${{truthSource(s).az_deg ?? s.truth.primary_az_deg}} / el ${{truthSource(s).el_deg ?? s.truth.primary_el_deg}}`, ''],
    ['Estimated Direction', `az ${{estimatedDir(s).az}} / el ${{estimatedDir(s).el}}`, ''],
    ['Drone Source ID', sourceId(s, 'drone'), ''],
    ['Tank Source ID', sourceId(s, 'tank'), ''],
    ['Single Detection Rate', fmtPct(comp.single_channel_detection_rate_percent), ''],
    ['Beam Detection Rate', fmtPct(comp.beamformed_smoothed_detection_rate_percent), ''],
    ['Verdict', comp.verdict || 'n/a', (comp.verdict || '').includes('helped') ? 'good' : 'warn'],
    ['Mean Score Delta', fmt(comp.mean_score_delta_beam_minus_single || 0), ''],
  ];
  document.getElementById('metrics').innerHTML = metrics.map(([label, value, cls]) =>
    `<div class="metric ${{cls}}"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`
  ).join('');
  const intervals = sum.detection_intervals || [];
  document.getElementById('intervals').textContent = intervals.length
    ? `Detection intervals: ${{intervals.map(x => `${{fmt(x[0],1)}}-${{fmt(x[1],1)}}s`).join(', ')}}`
    : 'Detection intervals: none';
}}

function drawTimeline() {{
  const s = DATA.scenarios[activeIndex];
  const rows = s.perWindow || [];
  const c = document.getElementById('timelineCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0d1013';
  ctx.fillRect(0,0,w,h);
  const pad = 42;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {{
    const y = pad + (h - 2*pad) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(w-pad,y); ctx.stroke();
  }}
  function xy(i, val) {{
    const x = pad + (w - 2*pad) * i / Math.max(1, rows.length - 1);
    const y = h - pad - (h - 2*pad) * Number(val || 0);
    return [x,y];
  }}
  function line(key, color, width=3) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    rows.forEach((r,i) => {{
      const v = Number(r[key]);
      if (!Number.isFinite(v)) return;
      const [x,y] = xy(i, v);
      if (!started) {{ ctx.moveTo(x,y); started = true; }} else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }}
  line('best_score', COLORS.blue, 3);
  line('option2_score_best', COLORS.cyan, 2);
  line('option3_score_best', COLORS.amber, 2);
  ctx.strokeStyle = COLORS.green;
  ctx.lineWidth = 4;
  ctx.beginPath();
  rows.forEach((r,i) => {{
    const [x,y] = xy(i, Number(r.smoothed_detected || 0) * 0.95);
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }});
  ctx.stroke();
  ctx.fillStyle = COLORS.text;
  ctx.font = '14px Segoe UI, Arial';
  ctx.fillText('best hybrid score', pad, 22);
  ctx.fillStyle = COLORS.cyan;
  ctx.fillText('option 2', pad + 150, 22);
  ctx.fillStyle = COLORS.amber;
  ctx.fillText('option 3', pad + 230, 22);
  ctx.fillStyle = COLORS.green;
  ctx.fillText('smoothed detection', pad + 310, 22);
}}

function drawHeat() {{
  const s = DATA.scenarios[activeIndex];
  const rows = s.directionScores || [];
  const c = document.getElementById('heatCanvas');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0d1013';
  ctx.fillRect(0,0,w,h);
  const azs = [...new Set(rows.map(r => Number(r.az)))].sort((a,b)=>a-b);
  const els = [...new Set(rows.map(r => Number(r.el)))].sort((a,b)=>a-b);
  const map = new Map();
  for (const r of rows) {{
    const key = `${{Number(r.az)}}|${{Number(r.el)}}`;
    const v = r.hybrid_score === '' ? Number(r.beam_energy || 0) : Number(r.hybrid_score || 0);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(v);
  }}
  const padL = 62, padB = 42, padT = 20, padR = 20;
  const cellW = (w - padL - padR) / Math.max(1, azs.length);
  const cellH = (h - padT - padB) / Math.max(1, els.length);
  for (let yi=0; yi<els.length; yi++) {{
    for (let xi=0; xi<azs.length; xi++) {{
      const arr = map.get(`${{azs[xi]}}|${{els[yi]}}`) || [0];
      const val = Math.max(0, Math.min(1, arr.reduce((a,b)=>a+b,0)/arr.length));
      const hue = 210 - val * 160;
      ctx.fillStyle = `hsl(${{hue}}, 75%, ${{22 + val*42}}%)`;
      const x = padL + xi * cellW;
      const y = padT + (els.length - 1 - yi) * cellH;
      ctx.fillRect(x+2, y+2, cellW-4, cellH-4);
    }}
  }}
  ctx.fillStyle = COLORS.text;
  ctx.font = '13px Segoe UI, Arial';
  azs.forEach((az, xi) => ctx.fillText(String(az), padL + xi*cellW + cellW*0.3, h - 14));
  els.forEach((el, yi) => ctx.fillText(String(el), 18, padT + (els.length - 1 - yi)*cellH + cellH*0.55));
  ctx.fillStyle = COLORS.muted;
  ctx.fillText('azimuth', w/2 - 24, h - 14);
  ctx.save();
  ctx.translate(14, h/2 + 20);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('elevation', 0, 0);
  ctx.restore();
}}

function renderAll() {{
  renderMetrics();
  drawArray();
  drawTimeline();
  drawHeat();
}}

function tick() {{
  phase += 1.1;
  drawArray();
  requestAnimationFrame(tick);
}}

window.addEventListener('resize', () => {{ resizeCanvases(); renderAll(); }});
init();
</script>
</body>
</html>"""


def write_visual_html(tag: str | None = None, out_path: Path | None = None) -> Path:
    payload = collect_visual_data(tag)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (VIS_DIR / f"phase3_visual_sim_{payload['tag']}.html")
    out_path.write_text(_html_template(payload), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build Phase 3 visual simulator HTML.")
    parser.add_argument("--tag", default=None, help="Simulation tag, e.g. 20260514_135723. Defaults to latest.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = write_visual_html(args.tag, args.out)
    print(path)


if __name__ == "__main__":
    main()
