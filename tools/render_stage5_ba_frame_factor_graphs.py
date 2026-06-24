#!/usr/bin/env python3
"""Render Stage5 BA frame/board factor graph viewer from cost-parity CSVs."""

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def rmse_from_rows(rows):
    if not rows:
        return float("nan")
    total = 0.0
    count = 0
    for row in rows:
        dx = fnum(row.get("backend_residual_x"))
        dy = fnum(row.get("backend_residual_y"))
        if not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        total += dx * dx + dy * dy
        count += 1
    if count == 0:
        return float("nan")
    return math.sqrt(total / count)


def point_type_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row.get("point_type", "unknown")] += 1
    return dict(counts)


def load_points(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def point_key(row):
    return (
        inum(row.get("frame_index")),
        row.get("frame_label", ""),
        row.get("board_id", ""),
        row.get("point_id", ""),
        row.get("point_type", ""),
    )


def load_summary(path):
    p = Path(path)
    if not p.exists():
        return {}
    values = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def aggregate(initial_rows, optimized_rows):
    initial_by_key = {point_key(row): row for row in initial_rows}
    optimized_by_key = {point_key(row): row for row in optimized_rows}
    keys = sorted(set(initial_by_key) & set(optimized_by_key))

    frame_initial = defaultdict(list)
    frame_final = defaultdict(list)
    board_initial = defaultdict(list)
    board_final = defaultdict(list)

    for key in keys:
        frame_index, frame_label, board_id, _, _ = key
        frame_key = (frame_index, frame_label)
        board_key = (frame_index, frame_label, str(board_id))
        frame_initial[frame_key].append(initial_by_key[key])
        frame_final[frame_key].append(optimized_by_key[key])
        board_initial[board_key].append(initial_by_key[key])
        board_final[board_key].append(optimized_by_key[key])

    frames = []
    for frame_key in sorted(frame_final.keys(), key=lambda k: k[0]):
        frame_index, frame_label = frame_key
        board_ids = sorted(
            {board_key[2] for board_key in board_final.keys() if board_key[:2] == frame_key},
            key=lambda x: inum(x),
        )
        board_records = []
        for board_id in board_ids:
            board_key = (frame_index, frame_label, board_id)
            initial_rmse = rmse_from_rows(board_initial[board_key])
            final_rmse = rmse_from_rows(board_final[board_key])
            reduction = initial_rmse - final_rmse if finite(initial_rmse) and finite(final_rmse) else float("nan")
            counts = point_type_counts(board_final[board_key])
            corner_count = len(board_final[board_key])
            weight_sum = sum(fnum(row.get("backend_m_estimator_weight"), 1.0) for row in board_final[board_key])
            board_records.append(
                {
                    "boardId": str(board_id),
                    "cornerCount": corner_count,
                    "outerCount": counts.get("outer", 0),
                    "internalCount": counts.get("internal", 0),
                    "initialRmse": initial_rmse,
                    "finalRmse": final_rmse,
                    "residualReduction": reduction,
                    "constraintStrength": weight_sum,
                }
            )

        initial_rmse = rmse_from_rows(frame_initial[frame_key])
        final_rmse = rmse_from_rows(frame_final[frame_key])
        reduction = initial_rmse - final_rmse if finite(initial_rmse) and finite(final_rmse) else float("nan")
        counts = point_type_counts(frame_final[frame_key])
        timestamp = ""
        parts = frame_label.split("_")
        if len(parts) >= 3:
            timestamp = parts[2]
        frames.append(
            {
                "frameIndex": frame_index,
                "frameId": f"Frame_{frame_index}",
                "frameLabel": frame_label,
                "timestamp": timestamp,
                "boardCount": len(board_records),
                "cornerCount": len(frame_final[frame_key]),
                "outerCount": counts.get("outer", 0),
                "internalCount": counts.get("internal", 0),
                "initialRmse": initial_rmse,
                "finalRmse": final_rmse,
                "residualReduction": reduction,
                "boardFactors": board_records,
            }
        )
    return frames


def distribution(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def write_trace_csv(path, frames):
    fields = [
        "frame_index",
        "frame_label",
        "timestamp",
        "board_count",
        "corner_count",
        "outer_count",
        "internal_count",
        "initial_rmse",
        "final_rmse",
        "residual_reduction",
        "board_ids",
        "board_factor_stats",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            board_ids = ";".join(board["boardId"] for board in frame["boardFactors"])
            board_stats = ";".join(
                (
                    f"{board['boardId']}:"
                    f"{board['cornerCount']}/"
                    f"{board['initialRmse']:.6g}/"
                    f"{board['finalRmse']:.6g}/"
                    f"{board['residualReduction']:.6g}"
                )
                for board in frame["boardFactors"]
            )
            writer.writerow(
                {
                    "frame_index": frame["frameIndex"],
                    "frame_label": frame["frameLabel"],
                    "timestamp": frame["timestamp"],
                    "board_count": frame["boardCount"],
                    "corner_count": frame["cornerCount"],
                    "outer_count": frame["outerCount"],
                    "internal_count": frame["internalCount"],
                    "initial_rmse": f"{frame['initialRmse']:.9g}",
                    "final_rmse": f"{frame['finalRmse']:.9g}",
                    "residual_reduction": f"{frame['residualReduction']:.9g}",
                    "board_ids": board_ids,
                    "board_factor_stats": board_stats,
                }
            )


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False)


def sanitize_for_json(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    return obj


def render_html(path, frames, metadata):
    payload = {
        "frames": sanitize_for_json(frames),
        "metadata": sanitize_for_json(metadata),
    }
    data_json = json_dumps(payload)
    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage5 BA Frame Factor Graph Viewer</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #687386;
      --line: #d8deea;
      --accent: #2563eb;
      --good: #16803c;
      --warn: #b7791f;
      --bad: #c53030;
      --violet: #7c3aed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: 280px 1fr 330px;
      grid-template-rows: auto 1fr 260px;
      height: 100vh;
      min-height: 760px;
    }
    header {
      grid-column: 1 / 4;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }
    h1 { font-size: 17px; margin: 0; }
    .meta { color: var(--muted); font-size: 12px; }
    aside, .right, .bottom, .graph-wrap {
      background: var(--panel);
      border-color: var(--line);
    }
    aside {
      border-right: 1px solid var(--line);
      overflow: auto;
      padding: 14px;
    }
    .right {
      border-left: 1px solid var(--line);
      overflow: auto;
      padding: 14px;
    }
    .graph-wrap {
      position: relative;
      overflow: hidden;
    }
    .bottom {
      grid-column: 1 / 4;
      border-top: 1px solid var(--line);
      padding: 10px 16px 14px;
      overflow: hidden;
    }
    .controls { display: grid; gap: 10px; }
    label { color: var(--muted); font-size: 12px; display: grid; gap: 4px; }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      background: #fbfcff;
    }
    #graph {
      width: 100%;
      height: 100%;
      display: block;
      background:
        linear-gradient(#eef2f8 1px, transparent 1px),
        linear-gradient(90deg, #eef2f8 1px, transparent 1px);
      background-size: 28px 28px;
    }
    .node { cursor: pointer; }
    .node rect, .node circle, .node polygon { stroke: #243047; stroke-width: 1.2; }
    .node text { pointer-events: none; fill: #111827; font-size: 12px; font-weight: 650; }
    .node .sub { fill: #4b5563; font-size: 10px; font-weight: 500; }
    .edge { stroke: #8d99ae; stroke-opacity: .7; fill: none; }
    .edge.strong { stroke: #475569; }
    .edge.factor { stroke: #7c3aed; }
    .selected-outline { stroke: #111827 !important; stroke-width: 2.7 !important; }
    .tooltip {
      position: absolute;
      pointer-events: none;
      background: #111827;
      color: white;
      padding: 8px 10px;
      border-radius: 7px;
      font-size: 12px;
      max-width: 320px;
      opacity: 0;
      transform: translate(10px, 10px);
      transition: opacity .08s;
      white-space: pre-line;
    }
    .section { margin-bottom: 16px; }
    .section h2 {
      margin: 0 0 8px;
      font-size: 13px;
      letter-spacing: 0;
    }
    .kv { display: grid; grid-template-columns: 1fr auto; gap: 6px 10px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .board-list { display: grid; gap: 8px; }
    .board-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      background: #fbfcff;
    }
    .legend { display: grid; gap: 8px; }
    .legend-row { display: flex; gap: 8px; align-items: center; color: var(--muted); }
    .swatch { width: 16px; height: 16px; border: 1px solid #222; border-radius: 4px; }
    .chart-title { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      height: 100%;
    }
    .chart-panel { min-width: 0; }
    .bar { cursor: pointer; }
    .bar:hover { opacity: .75; }
    .toolbar {
      position: absolute;
      right: 12px;
      top: 12px;
      display: flex;
      gap: 6px;
      background: rgba(255,255,255,.92);
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Stage5 BA Frame Factor Graph Viewer</h1>
        <div class="meta" id="headerMeta"></div>
      </div>
      <div class="meta">K / T_camera_reference(frame) / T_reference_board(board) / reprojection factors</div>
    </header>

    <aside>
      <div class="controls">
        <label>搜索 frame_id / timestamp / label
          <input id="search" placeholder="例如 48 或 112699...">
        </label>
        <label>排序
          <select id="sortMode">
            <option value="frame">frame index</option>
            <option value="finalRmse">final RMSE desc</option>
            <option value="reduction">residual reduction desc</option>
            <option value="boardCount">board count desc</option>
            <option value="cornerCount">corner count desc</option>
          </select>
        </label>
        <label>High residual metric
          <select id="highMetric">
            <option value="finalRmse">final RMSE</option>
            <option value="initialRmse">initial RMSE</option>
            <option value="residualReduction">low reduction</option>
          </select>
        </label>
        <label>阈值
          <input id="threshold" type="number" step="0.1" value="1.0">
        </label>
        <label>
          <span><input id="onlyHigh" type="checkbox"> 只显示 high residual</span>
        </label>
        <button id="collapseAll">折叠全部</button>
        <button id="fit">Fit to screen</button>
        <button id="exportSvg">导出 SVG</button>
      </div>

      <div class="chips" id="statsChips"></div>

      <div class="section" style="margin-top:16px">
        <h2>Legend</h2>
        <div class="legend">
          <div class="legend-row"><span class="swatch" style="background:#dbeafe"></span>K 内参变量</div>
          <div class="legend-row"><span class="swatch" style="background:#ede9fe"></span>T_camera_reference(frame)</div>
          <div class="legend-row"><span class="swatch" style="background:#dcfce7"></span>T_reference_board(board)</div>
          <div class="legend-row"><span class="swatch" style="background:#fff7ed"></span>frame-board 重投影因子</div>
          <div class="legend-row"><span class="swatch" style="background:#fecaca"></span>高 final RMSE</div>
          <div class="legend-row">线越粗，corner count / 约束强度越高</div>
        </div>
      </div>
    </aside>

    <main class="graph-wrap">
      <div class="toolbar">
        <button id="zoomOut">-</button>
        <button id="zoomIn">+</button>
        <button id="resetView">Reset</button>
      </div>
      <svg id="graph" role="img" aria-label="Stage5 BA factor graph"></svg>
      <div class="tooltip" id="tooltip"></div>
    </main>

    <section class="right">
      <div class="section">
        <h2>Selected Frame</h2>
        <div class="kv" id="selectedKv"></div>
      </div>
      <div class="section">
        <h2>Board-Level Factors</h2>
        <div class="board-list" id="boardList"></div>
      </div>
      <div class="section">
        <h2>Interpretation</h2>
        <div class="meta">
          这个图展示的是 Stage5 内参 BA 中约束如何进入优化，以及每个 frame/board 聚合残差从 initial 到 final 的变化。
          连接关系本身不能证明参数一定最优，需要结合 residual reduction、最终 RMSE、变量更新量或不确定性一起判断。
        </div>
      </div>
    </section>

    <section class="bottom">
      <div class="charts">
        <div class="chart-panel">
          <div class="chart-title">Per-frame RMSE: initial vs final</div>
          <svg id="rmseChart" width="100%" height="210"></svg>
        </div>
        <div class="chart-panel">
          <div class="chart-title">Residual reduction and final RMSE heat</div>
          <svg id="reductionChart" width="100%" height="210"></svg>
        </div>
      </div>
    </section>
  </div>

<script>
const DATA = __DATA_JSON__;
const frames = DATA.frames;
const metadata = DATA.metadata;
const state = { expanded: new Set(), selected: frames[0]?.frameIndex ?? null, scale: 1, tx: 0, ty: 0 };

const svg = document.getElementById('graph');
const tooltip = document.getElementById('tooltip');
const ns = 'http://www.w3.org/2000/svg';

function fmt(v, digits=3) {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : 'n/a';
}

function colorForRmse(v) {
  if (!Number.isFinite(v)) return '#e5e7eb';
  if (v < 0.35) return '#dcfce7';
  if (v < 0.8) return '#fef9c3';
  if (v < 1.5) return '#fed7aa';
  return '#fecaca';
}

function strokeWidth(count) {
  return Math.max(1.2, Math.min(7, 1 + Math.sqrt(Math.max(0, count)) / 5));
}

function frameTooltip(f) {
  return [
    `${f.frameId}`,
    `label: ${f.frameLabel}`,
    `timestamp: ${f.timestamp || 'n/a'}`,
    `boards: ${f.boardCount}`,
    `corners: ${f.cornerCount}`,
    `RMSE initial/final: ${fmt(f.initialRmse)} -> ${fmt(f.finalRmse)}`,
    `reduction: ${fmt(f.residualReduction)}`
  ].join('\\n');
}

function boardTooltip(f, b) {
  return [
    `${f.frameId} / Board_${b.boardId}`,
    `corners: ${b.cornerCount} outer=${b.outerCount} internal=${b.internalCount}`,
    `RMSE initial/final: ${fmt(b.initialRmse)} -> ${fmt(b.finalRmse)}`,
    `reduction: ${fmt(b.residualReduction)}`
  ].join('\\n');
}

function getVisibleFrames() {
  const query = document.getElementById('search').value.trim().toLowerCase();
  const onlyHigh = document.getElementById('onlyHigh').checked;
  const metric = document.getElementById('highMetric').value;
  const threshold = Number(document.getElementById('threshold').value);
  const sortMode = document.getElementById('sortMode').value;
  let out = frames.filter(f => {
    const text = `${f.frameIndex} ${f.frameId} ${f.timestamp} ${f.frameLabel}`.toLowerCase();
    if (query && !text.includes(query)) return false;
    if (onlyHigh) {
      if (metric === 'residualReduction') {
        if (!Number.isFinite(f.residualReduction) || f.residualReduction >= threshold) return false;
      } else {
        if (!Number.isFinite(f[metric]) || f[metric] < threshold) return false;
      }
    }
    return true;
  });
  out.sort((a, b) => {
    if (sortMode === 'frame') return a.frameIndex - b.frameIndex;
    if (sortMode === 'reduction') return (b.residualReduction ?? -1e9) - (a.residualReduction ?? -1e9);
    return (b[sortMode] ?? -1e9) - (a[sortMode] ?? -1e9);
  });
  return out;
}

function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function make(tag, attrs={}) {
  const el = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  return el;
}

function addText(group, x, y, text, cls='') {
  const el = make('text', {x, y, 'text-anchor': 'middle', class: cls});
  el.textContent = text;
  group.appendChild(el);
  return el;
}

function showTooltip(evt, text) {
  tooltip.textContent = text;
  tooltip.style.left = `${evt.clientX}px`;
  tooltip.style.top = `${evt.clientY}px`;
  tooltip.style.opacity = '1';
}

function hideTooltip() {
  tooltip.style.opacity = '0';
}

function nodeEvents(el, tip, onClick) {
  el.addEventListener('mousemove', e => showTooltip(e, tip()));
  el.addEventListener('mouseleave', hideTooltip);
  el.addEventListener('click', e => {
    e.stopPropagation();
    onClick();
  });
}

function renderGraph() {
  clear(svg);
  const visible = getVisibleFrames();
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 600;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const root = make('g', {transform: `translate(${state.tx},${state.ty}) scale(${state.scale})`});
  svg.appendChild(root);

  const kx = 95;
  const yGap = 84;
  const top = 74;
  const kNode = make('g', {class: 'node'});
  kNode.appendChild(make('rect', {x: kx - 48, y: 18, width: 96, height: 44, rx: 8, fill: '#dbeafe'}));
  addText(kNode, kx, 45, 'K');
  root.appendChild(kNode);

  visible.forEach((f, idx) => {
    const y = top + idx * yGap;
    const expanded = state.expanded.has(f.frameIndex);
    const selected = state.selected === f.frameIndex;
    const frameX = 250;
    const poseX = expanded ? 470 : 520;
    const frameGroup = make('g', {class: 'node'});
    frameGroup.appendChild(make('rect', {
      x: frameX - 62, y: y - 24, width: 124, height: 48, rx: 8,
      fill: colorForRmse(f.finalRmse),
      class: selected ? 'selected-outline' : ''
    }));
    addText(frameGroup, frameX, y - 4, f.frameId);
    addText(frameGroup, frameX, y + 13, `${fmt(f.initialRmse,2)} -> ${fmt(f.finalRmse,2)}`, 'sub');
    root.appendChild(frameGroup);
    nodeEvents(frameGroup, () => frameTooltip(f), () => {
      state.selected = f.frameIndex;
      if (state.expanded.has(f.frameIndex)) state.expanded.delete(f.frameIndex);
      else state.expanded.add(f.frameIndex);
      renderAll();
    });

    const poseGroup = make('g', {class: 'node'});
    poseGroup.appendChild(make('rect', {
      x: poseX - 88, y: y - 22, width: 176, height: 44, rx: 8,
      fill: '#ede9fe',
      class: selected ? 'selected-outline' : ''
    }));
    addText(poseGroup, poseX, y - 3, `T_cam_ref(${f.frameIndex})`);
    addText(poseGroup, poseX, y + 13, `${f.boardCount} boards / ${f.cornerCount} pts`, 'sub');
    root.appendChild(poseGroup);

    const edge = make('line', {
      x1: kx + 48, y1: 40, x2: frameX - 62, y2: y,
      class: 'edge', 'stroke-width': strokeWidth(f.cornerCount)
    });
    root.insertBefore(edge, root.firstChild);

    if (!expanded) {
      const e2 = make('line', {
        x1: frameX + 62, y1: y, x2: poseX - 88, y2: y,
        class: 'edge factor', 'stroke-width': strokeWidth(f.cornerCount)
      });
      root.insertBefore(e2, root.firstChild);
      return;
    }

    const boardStartX = 690;
    const boardGap = 114;
    f.boardFactors.forEach((b, bi) => {
      const bx = boardStartX + bi * boardGap;
      const fy = y - 20;
      const by = y + 26;
      const factor = make('g', {class: 'node'});
      factor.appendChild(make('circle', {
        cx: bx, cy: fy, r: 20, fill: '#fff7ed',
        class: selected ? 'selected-outline' : ''
      }));
      addText(factor, bx, fy + 4, `f${b.boardId}`);
      root.appendChild(factor);
      nodeEvents(factor, () => boardTooltip(f, b), () => {
        state.selected = f.frameIndex;
        renderAll();
      });

      const board = make('g', {class: 'node'});
      board.appendChild(make('rect', {
        x: bx - 48, y: by - 18, width: 96, height: 36, rx: 8,
        fill: '#dcfce7'
      }));
      addText(board, bx, by + 4, `Board_${b.boardId}`);
      root.appendChild(board);
      nodeEvents(board, () => boardTooltip(f, b), () => {
        state.selected = f.frameIndex;
        renderAll();
      });

      for (const line of [
        [kx + 48, 40, bx - 20, fy],
        [poseX + 88, y, bx - 20, fy],
        [bx, fy + 20, bx, by - 18],
      ]) {
        root.insertBefore(make('line', {
          x1: line[0], y1: line[1], x2: line[2], y2: line[3],
          class: 'edge factor', 'stroke-width': strokeWidth(b.cornerCount)
        }), root.firstChild);
      }
    });
  });

  const contentHeight = top + visible.length * yGap + 40;
  const contentWidth = 760 + Math.max(...visible.map(f => f.boardCount), 1) * 114;
  svg.dataset.contentWidth = contentWidth;
  svg.dataset.contentHeight = contentHeight;
}

function renderStats() {
  document.getElementById('headerMeta').textContent =
    `${metadata.runName} · frames=${metadata.frameCount} · frame-board=${metadata.frameBoardCount} · distribution=${metadata.boardDistribution}`;
  const chips = document.getElementById('statsChips');
  chips.innerHTML = '';
  [
    `frames ${metadata.frameCount}`,
    `frame-board ${metadata.frameBoardCount}`,
    `dist ${metadata.boardDistribution}`,
    `overall ${fmt(metadata.initialOverallRmse)} -> ${fmt(metadata.finalOverallRmse)}`
  ].forEach(text => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = text;
    chips.appendChild(chip);
  });
}

function selectedFrame() {
  return frames.find(f => f.frameIndex === state.selected) || frames[0];
}

function renderSidePanel() {
  const f = selectedFrame();
  const kv = document.getElementById('selectedKv');
  kv.innerHTML = '';
  if (!f) return;
  const rows = [
    ['frame', f.frameId],
    ['label', f.frameLabel],
    ['timestamp', f.timestamp || 'n/a'],
    ['boards', f.boardCount],
    ['corners', f.cornerCount],
    ['outer/internal', `${f.outerCount}/${f.internalCount}`],
    ['initial RMSE', fmt(f.initialRmse)],
    ['final RMSE', fmt(f.finalRmse)],
    ['residual reduction', fmt(f.residualReduction)],
  ];
  rows.forEach(([k, v]) => {
    const kd = document.createElement('div');
    kd.textContent = k;
    const vd = document.createElement('div');
    vd.textContent = v;
    kv.appendChild(kd);
    kv.appendChild(vd);
  });

  const boardList = document.getElementById('boardList');
  boardList.innerHTML = '';
  f.boardFactors.forEach(b => {
    const card = document.createElement('div');
    card.className = 'board-card';
    card.innerHTML = `
      <strong>Board_${b.boardId}</strong>
      <div class="kv" style="margin-top:6px">
        <div>corners</div><div>${b.cornerCount}</div>
        <div>outer/internal</div><div>${b.outerCount}/${b.internalCount}</div>
        <div>RMSE</div><div>${fmt(b.initialRmse)} -> ${fmt(b.finalRmse)}</div>
        <div>reduction</div><div>${fmt(b.residualReduction)}</div>
      </div>`;
    boardList.appendChild(card);
  });
}

function renderCharts() {
  renderRmseChart();
  renderReductionChart();
}

function chartDims(svgEl) {
  const width = svgEl.clientWidth || 600;
  const height = svgEl.clientHeight || 210;
  svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
  clear(svgEl);
  return {width, height, padL: 36, padT: 12, padB: 34, padR: 10};
}

function renderRmseChart() {
  const el = document.getElementById('rmseChart');
  const d = chartDims(el);
  const maxV = Math.max(0.1, ...frames.flatMap(f => [f.initialRmse || 0, f.finalRmse || 0]));
  const innerW = d.width - d.padL - d.padR;
  const innerH = d.height - d.padT - d.padB;
  const slot = innerW / Math.max(1, frames.length);
  frames.forEach((f, i) => {
    const x = d.padL + i * slot;
    const w = Math.max(2, slot * 0.32);
    const h0 = innerH * ((f.initialRmse || 0) / maxV);
    const h1 = innerH * ((f.finalRmse || 0) / maxV);
    const g0 = make('rect', {x, y: d.padT + innerH - h0, width: w, height: h0, fill: '#94a3b8', class: 'bar'});
    const g1 = make('rect', {x: x + w + 1, y: d.padT + innerH - h1, width: w, height: h1, fill: colorForRmse(f.finalRmse), class: 'bar'});
    [g0, g1].forEach(bar => bar.addEventListener('click', () => {
      state.selected = f.frameIndex;
      state.expanded.add(f.frameIndex);
      renderAll();
    }));
    el.appendChild(g0);
    el.appendChild(g1);
    if (slot > 18) {
      const t = make('text', {x: x + slot / 2, y: d.height - 10, 'text-anchor': 'middle', fill: '#64748b', 'font-size': '10'});
      t.textContent = f.frameIndex;
      el.appendChild(t);
    }
  });
}

function renderReductionChart() {
  const el = document.getElementById('reductionChart');
  const d = chartDims(el);
  const vals = frames.map(f => Math.max(0, f.residualReduction || 0));
  const maxV = Math.max(0.1, ...vals);
  const innerW = d.width - d.padL - d.padR;
  const innerH = d.height - d.padT - d.padB;
  const slot = innerW / Math.max(1, frames.length);
  frames.forEach((f, i) => {
    const x = d.padL + i * slot;
    const w = Math.max(3, slot * 0.72);
    const h = innerH * (Math.max(0, f.residualReduction || 0) / maxV);
    const bar = make('rect', {x, y: d.padT + innerH - h, width: w, height: h, fill: '#2563eb', class: 'bar'});
    bar.addEventListener('click', () => {
      state.selected = f.frameIndex;
      state.expanded.add(f.frameIndex);
      renderAll();
    });
    el.appendChild(bar);
    const heat = make('rect', {x, y: d.height - 26, width: w, height: 10, fill: colorForRmse(f.finalRmse), class: 'bar'});
    heat.addEventListener('click', () => {
      state.selected = f.frameIndex;
      state.expanded.add(f.frameIndex);
      renderAll();
    });
    el.appendChild(heat);
    if (slot > 18) {
      const t = make('text', {x: x + slot / 2, y: d.height - 10, 'text-anchor': 'middle', fill: '#64748b', 'font-size': '10'});
      t.textContent = f.frameIndex;
      el.appendChild(t);
    }
  });
}

function fitToScreen() {
  const visible = getVisibleFrames();
  const maxBoards = Math.max(...visible.map(f => f.boardCount), 1);
  const contentW = 760 + maxBoards * 114;
  const contentH = 120 + visible.length * 84;
  const scaleX = (svg.clientWidth - 40) / contentW;
  const scaleY = (svg.clientHeight - 40) / contentH;
  state.scale = Math.max(0.25, Math.min(1.25, Math.min(scaleX, scaleY)));
  state.tx = 20;
  state.ty = 10;
  renderGraph();
}

function exportSvg() {
  const clone = svg.cloneNode(true);
  clone.setAttribute('xmlns', ns);
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], {type: 'image/svg+xml'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'stage5_ba_frame_factor_graph.svg';
  a.click();
  URL.revokeObjectURL(a.href);
}

let dragging = false;
let last = null;
svg.addEventListener('mousedown', e => { dragging = true; last = [e.clientX, e.clientY]; });
window.addEventListener('mouseup', () => { dragging = false; });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  state.tx += e.clientX - last[0];
  state.ty += e.clientY - last[1];
  last = [e.clientX, e.clientY];
  renderGraph();
});
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.08 : 0.92;
  state.scale = Math.max(0.15, Math.min(3, state.scale * factor));
  renderGraph();
}, {passive: false});

function renderAll() {
  renderStats();
  renderGraph();
  renderSidePanel();
  renderCharts();
}

for (const id of ['search', 'sortMode', 'highMetric', 'threshold', 'onlyHigh']) {
  document.getElementById(id).addEventListener('input', renderAll);
}
document.getElementById('collapseAll').addEventListener('click', () => { state.expanded.clear(); renderAll(); });
document.getElementById('fit').addEventListener('click', fitToScreen);
document.getElementById('zoomIn').addEventListener('click', () => { state.scale *= 1.15; renderGraph(); });
document.getElementById('zoomOut').addEventListener('click', () => { state.scale *= 0.85; renderGraph(); });
document.getElementById('resetView').addEventListener('click', () => { state.scale = 1; state.tx = 0; state.ty = 0; renderGraph(); });
document.getElementById('exportSvg').addEventListener('click', exportSvg);
window.addEventListener('resize', renderAll);

renderAll();
setTimeout(fitToScreen, 50);
</script>
</body>
</html>
"""
    path.write_text(html_text.replace("__DATA_JSON__", data_json), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-points-csv", required=True)
    parser.add_argument("--optimized-points-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    initial_path = Path(args.initial_points_csv)
    optimized_path = Path(args.optimized_points_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = load_points(initial_path)
    optimized_rows = load_points(optimized_path)
    frames = aggregate(initial_rows, optimized_rows)

    initial_all_rmse = rmse_from_rows(initial_rows)
    final_all_rmse = rmse_from_rows(optimized_rows)
    metadata = {
        "runName": args.run_name or optimized_path.parent.name,
        "frameCount": len(frames),
        "frameBoardCount": sum(frame["boardCount"] for frame in frames),
        "boardDistribution": distribution(frame["boardCount"] for frame in frames),
        "initialOverallRmse": initial_all_rmse,
        "finalOverallRmse": final_all_rmse,
        "sourceInitial": str(initial_path),
        "sourceOptimized": str(optimized_path),
    }

    write_trace_csv(output_dir / "stage5_ba_frame_factor_trace.csv", frames)
    render_html(output_dir / "viewer.html", frames, metadata)
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"run_name: {metadata['runName']}",
                f"frame_count: {metadata['frameCount']}",
                f"frame_board_count: {metadata['frameBoardCount']}",
                f"per_frame_board_distribution: {metadata['boardDistribution']}",
                f"initial_overall_rmse: {metadata['initialOverallRmse']:.9g}",
                f"final_overall_rmse: {metadata['finalOverallRmse']:.9g}",
                f"viewer_html: {output_dir / 'viewer.html'}",
                f"trace_csv: {output_dir / 'stage5_ba_frame_factor_trace.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote viewer: {output_dir / 'viewer.html'}")
    print(f"Wrote trace: {output_dir / 'stage5_ba_frame_factor_trace.csv'}")
    print(f"frames={metadata['frameCount']} frame_board={metadata['frameBoardCount']} dist={metadata['boardDistribution']}")
    print(f"rmse={metadata['initialOverallRmse']:.6g}->{metadata['finalOverallRmse']:.6g}")


if __name__ == "__main__":
    main()
