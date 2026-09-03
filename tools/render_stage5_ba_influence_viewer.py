#!/usr/bin/env python3
"""Render a Stage5 intrinsics BA influence/factor graph viewer.

This viewer is intentionally diagnostic: it combines factor-graph structure with
residual reduction and variable-update proxies available in existing Stage5
outputs. It does not claim exact Hessian marginal contribution unless those
fields are explicitly recorded by the solver.
"""

import argparse
import csv
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


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def load_summary(path):
    values = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        values[k.strip()] = v.strip()
    return values


def rmse(rows, prefix="backend_residual"):
    s = 0.0
    n = 0
    for r in rows:
        dx = fnum(r.get(prefix + "_x"))
        dy = fnum(r.get(prefix + "_y"))
        if finite(dx) and finite(dy):
            s += dx * dx + dy * dy
            n += 1
    return math.sqrt(s / n) if n else float("nan")


def weighted_cost(rows):
    total = 0.0
    for r in rows:
        v = fnum(r.get("backend_weighted_squared_error"), 0.0)
        if finite(v):
            total += v
    return total


def point_key(row):
    return (
        inum(row.get("frame_index")),
        row.get("frame_label", ""),
        str(row.get("board_id", "")),
        str(row.get("point_id", "")),
        row.get("point_type", ""),
    )


def residual_centroid(rows):
    xs, ys = [], []
    for r in rows:
        x = fnum(r.get("observed_x"))
        y = fnum(r.get("observed_y"))
        if finite(x) and finite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return {
        "meanX": sum(xs) / len(xs),
        "meanY": sum(ys) / len(ys),
        "minX": min(xs),
        "maxX": max(xs),
        "minY": min(ys),
        "maxY": max(ys),
    }


def image_coverage_proxy(rows):
    c = residual_centroid(rows)
    if not c:
        return 0.0
    # Assumes square-ish 4512 images for current datasets; still works as a
    # relative proxy if resolution differs.
    w = 4512.0
    h = 4512.0
    area = max(0.0, c["maxX"] - c["minX"]) * max(0.0, c["maxY"] - c["minY"])
    return min(1.0, area / (w * h))


def polar_proxy(rows):
    c = residual_centroid(rows)
    if not c:
        return 0.0
    cx, cy = 2256.0, 2256.0
    vals = []
    for x, y in [
        (c["minX"], c["minY"]),
        (c["minX"], c["maxY"]),
        (c["maxX"], c["minY"]),
        (c["maxX"], c["maxY"]),
        (c["meanX"], c["meanY"]),
    ]:
        vals.append(math.hypot(x - cx, y - cy) / math.hypot(cx, cy))
    return max(vals)


def load_variable_block_influence(run_dir):
    rows = []
    for suffix in ["initial", "optimized"]:
        path = run_dir / f"backend_optimization_variable_block_influence_{suffix}.csv"
        for row in load_csv(path):
            row["_file_stage"] = suffix
            rows.append(row)
    return rows


def aggregate_block_influence(rows):
    by_frame = defaultdict(lambda: defaultdict(float))
    by_board = defaultdict(lambda: defaultdict(float))
    by_frame_block = defaultdict(lambda: defaultdict(float))
    available = False
    for row in rows:
        if row.get("stage_label") != "optimized":
            continue
        scope = row.get("variable_scope", "")
        frame_index = inum(row.get("frame_index"), -1)
        frame_label = row.get("frame_label", "")
        board_id = str(row.get("board_id", ""))
        block = row.get("variable_block", "")
        if frame_index < 0 or not board_id:
            continue
        available = True
        trace = fnum(row.get("hessian_trace"), 0.0)
        logdet = fnum(row.get("hessian_logdet"), 0.0)
        rank = fnum(row.get("hessian_rank_proxy"), 0.0)
        grad = fnum(row.get("gradient_norm"), 0.0)
        cost = fnum(row.get("weighted_cost"), 0.0)
        residual_count = fnum(row.get("residual_count"), 0.0)
        fkey = (frame_index, frame_label)
        bkey = (frame_index, frame_label, board_id)
        for target in (by_frame[fkey], by_board[bkey], by_frame_block[(frame_index, frame_label, block)]):
            target["hessian_trace_total"] += trace
            target["hessian_logdet_total"] += logdet
            target["hessian_rank_proxy_total"] += rank
            target["gradient_norm_total"] += grad
            target["weighted_cost_total"] += cost
            target["residual_count_total"] += residual_count
        if scope == "camera_model":
            by_frame[fkey]["camera_hessian_trace"] += trace
            by_frame[fkey]["camera_hessian_logdet"] += logdet
            by_frame[fkey]["camera_hessian_rank_proxy"] += rank
            by_frame[fkey]["camera_gradient_norm"] += grad
            by_board[bkey]["camera_hessian_trace"] += trace
            by_board[bkey]["camera_hessian_logdet"] += logdet
            by_board[bkey]["camera_hessian_rank_proxy"] += rank
            by_board[bkey]["camera_gradient_norm"] += grad
        elif scope == "T_camera_reference":
            by_frame[fkey]["frame_pose_hessian_trace"] += trace
            by_frame[fkey]["frame_pose_hessian_logdet"] += logdet
            by_frame[fkey]["frame_pose_hessian_rank_proxy"] += rank
            by_frame[fkey]["frame_pose_gradient_norm"] += grad
            by_board[bkey]["frame_pose_hessian_trace"] += trace
            by_board[bkey]["frame_pose_hessian_logdet"] += logdet
            by_board[bkey]["frame_pose_hessian_rank_proxy"] += rank
            by_board[bkey]["frame_pose_gradient_norm"] += grad
        elif scope == "T_reference_board":
            by_frame[fkey]["board_layout_hessian_trace"] += trace
            by_frame[fkey]["board_layout_hessian_logdet"] += logdet
            by_frame[fkey]["board_layout_hessian_rank_proxy"] += rank
            by_frame[fkey]["board_layout_gradient_norm"] += grad
            by_board[bkey]["board_layout_hessian_trace"] += trace
            by_board[bkey]["board_layout_hessian_logdet"] += logdet
            by_board[bkey]["board_layout_hessian_rank_proxy"] += rank
            by_board[bkey]["board_layout_gradient_norm"] += grad
    return {
        "available": available,
        "by_frame": by_frame,
        "by_board": by_board,
        "by_frame_block": by_frame_block,
    }


def parse_stage_summaries(summary):
    stages = []
    current = None
    for key, value in summary.items():
        if key == "stage_label":
            if current:
                stages.append(current)
            current = {"stage_label": value}
        elif key.startswith("stage_") and current is not None:
            current[key] = value
    if current:
        stages.append(current)
    return stages


def load_stage_summaries(path):
    p = Path(path)
    if not p.exists():
        return []
    stages = []
    current = None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "stage_label":
            if current:
                stages.append(current)
            current = {"stage_label": value}
        elif key.startswith("stage_") and current is not None:
            current[key] = value
    if current:
        stages.append(current)
    return stages


def camera_delta(anchor, optimized):
    deltas = {}
    for key in ["xi", "alpha", "fu", "fv", "cu", "cv"]:
        a = anchor.get(key)
        b = optimized.get(key)
        deltas[key] = b - a if finite(a) and finite(b) else float("nan")
    return deltas


def stage_trace_points(stages):
    points = []
    if not stages:
        return points
    cursor = 0
    for stage in stages:
        start = fnum(stage.get("stage_objective_start"))
        final = fnum(stage.get("stage_objective_final"))
        iterations = max(1, inum(stage.get("stage_iterations"), 1))
        if finite(start):
            points.append({
                "x": cursor,
                "stage": stage.get("stage_label", ""),
                "kind": "start",
                "cost": start,
                "deltaX": None,
                "deltaJ": None,
            })
        cursor += iterations
        if finite(final):
            points.append({
                "x": cursor,
                "stage": stage.get("stage_label", ""),
                "kind": "final",
                "cost": final,
                "deltaX": fnum(stage.get("stage_delta_x_final")),
                "deltaJ": fnum(stage.get("stage_delta_j_final")),
            })
    return points


def build_payload(run_dir):
    run_dir = Path(run_dir)
    initial_rows = load_csv(run_dir / "backend_optimization_cost_parity_initial_points.csv")
    final_rows = load_csv(run_dir / "backend_optimization_cost_parity_optimized_points.csv")
    block_influence = aggregate_block_influence(load_variable_block_influence(run_dir))
    initial_by_key = {point_key(r): r for r in initial_rows}
    final_by_key = {point_key(r): r for r in final_rows}
    keys = sorted(set(initial_by_key) & set(final_by_key))

    frame_initial = defaultdict(list)
    frame_final = defaultdict(list)
    board_initial = defaultdict(list)
    board_final = defaultdict(list)
    layout_rows = {str(r.get("board_id")): r for r in load_csv(run_dir / "board_layout_pose_delta.csv")}

    for key in keys:
        fi, fl, bid, _, _ = key
        frame_key = (fi, fl)
        board_key = (fi, fl, bid)
        frame_initial[frame_key].append(initial_by_key[key])
        frame_final[frame_key].append(final_by_key[key])
        board_initial[board_key].append(initial_by_key[key])
        board_final[board_key].append(final_by_key[key])

    global_initial_cost = weighted_cost([initial_by_key[k] for k in keys])
    global_final_cost = weighted_cost([final_by_key[k] for k in keys])
    global_cost_drop = max(0.0, global_initial_cost - global_final_cost)

    frames = []
    for frame_key in sorted(frame_final.keys()):
        fi, fl = frame_key
        irows = frame_initial[frame_key]
        frows = frame_final[frame_key]
        i_rmse = rmse(irows)
        f_rmse = rmse(frows)
        i_cost = weighted_cost(irows)
        f_cost = weighted_cost(frows)
        cost_drop = max(0.0, i_cost - f_cost)
        board_ids = sorted(
            {bk[2] for bk in board_final if bk[:2] == frame_key},
            key=lambda x: inum(x),
        )
        boards = []
        for bid in board_ids:
            bk = (fi, fl, bid)
            bi = board_initial[bk]
            bf = board_final[bk]
            li = layout_rows.get(str(bid), {})
            bi_rmse = rmse(bi)
            bf_rmse = rmse(bf)
            bi_cost = weighted_cost(bi)
            bf_cost = weighted_cost(bf)
            b_drop = max(0.0, bi_cost - bf_cost)
            b_inf = block_influence["by_board"].get(bk, {})
            boards.append({
                "boardId": str(bid),
                "cornerCount": len(bf),
                "initialRmse": bi_rmse,
                "finalRmse": bf_rmse,
                "residualReduction": bi_rmse - bf_rmse if finite(bi_rmse) and finite(bf_rmse) else None,
                "costDrop": b_drop,
                "framePoseInfluenceProxy": b_drop / global_cost_drop if global_cost_drop > 0 else 0.0,
                "constraintStrength": sum(fnum(r.get("backend_inv_r_scale"), 1.0) for r in bf),
                "layoutTranslationDeltaMm": fnum(li.get("translation_delta_mm"), 0.0),
                "layoutRotationDeltaDeg": fnum(li.get("rotation_delta_deg"), 0.0),
                "cameraHessianTrace": b_inf.get("camera_hessian_trace", 0.0),
                "cameraHessianLogdet": b_inf.get("camera_hessian_logdet", 0.0),
                "cameraHessianRankProxy": b_inf.get("camera_hessian_rank_proxy", 0.0),
                "cameraGradientNorm": b_inf.get("camera_gradient_norm", 0.0),
                "framePoseHessianTrace": b_inf.get("frame_pose_hessian_trace", 0.0),
                "framePoseHessianLogdet": b_inf.get("frame_pose_hessian_logdet", 0.0),
                "framePoseHessianRankProxy": b_inf.get("frame_pose_hessian_rank_proxy", 0.0),
                "framePoseGradientNorm": b_inf.get("frame_pose_gradient_norm", 0.0),
                "boardLayoutHessianTrace": b_inf.get("board_layout_hessian_trace", 0.0),
                "boardLayoutHessianLogdet": b_inf.get("board_layout_hessian_logdet", 0.0),
                "boardLayoutHessianRankProxy": b_inf.get("board_layout_hessian_rank_proxy", 0.0),
                "boardLayoutGradientNorm": b_inf.get("board_layout_gradient_norm", 0.0),
            })
        coverage = image_coverage_proxy(frows)
        polar = polar_proxy(frows)
        f_inf = block_influence["by_frame"].get(frame_key, {})
        frames.append({
            "frameIndex": fi,
            "frameLabel": fl,
            "timestamp": fl.split("_")[2] if len(fl.split("_")) >= 3 else "",
            "boardCount": len(boards),
            "cornerCount": len(frows),
            "initialRmse": i_rmse,
            "finalRmse": f_rmse,
            "residualReduction": i_rmse - f_rmse if finite(i_rmse) and finite(f_rmse) else None,
            "costDrop": cost_drop,
            "cameraInfluenceProxy": cost_drop / global_cost_drop if global_cost_drop > 0 else 0.0,
            "coverageProxy": coverage,
            "polarProxy": polar,
            "intrinsicsSensitivityProxy": math.log1p(len(frows)) * (0.5 + coverage + polar),
            "jacobianBlockInfluenceAvailable": block_influence["available"],
            "cameraHessianTrace": f_inf.get("camera_hessian_trace", 0.0),
            "cameraHessianLogdet": f_inf.get("camera_hessian_logdet", 0.0),
            "cameraHessianRankProxy": f_inf.get("camera_hessian_rank_proxy", 0.0),
            "cameraGradientNorm": f_inf.get("camera_gradient_norm", 0.0),
            "framePoseHessianTrace": f_inf.get("frame_pose_hessian_trace", 0.0),
            "framePoseHessianLogdet": f_inf.get("frame_pose_hessian_logdet", 0.0),
            "framePoseHessianRankProxy": f_inf.get("frame_pose_hessian_rank_proxy", 0.0),
            "framePoseGradientNorm": f_inf.get("frame_pose_gradient_norm", 0.0),
            "boardLayoutHessianTrace": f_inf.get("board_layout_hessian_trace", 0.0),
            "boardLayoutHessianLogdet": f_inf.get("board_layout_hessian_logdet", 0.0),
            "boardLayoutHessianRankProxy": f_inf.get("board_layout_hessian_rank_proxy", 0.0),
            "boardLayoutGradientNorm": f_inf.get("board_layout_gradient_norm", 0.0),
            "boards": boards,
        })

    backend_summary = load_summary(run_dir / "backend_optimization_summary.txt")
    problem_summary = load_summary(run_dir / "stage5_backend_problem_summary.txt")
    cost_initial = load_summary(run_dir / "backend_optimization_cost_parity_initial_summary.txt")
    cost_final = load_summary(run_dir / "backend_optimization_cost_parity_optimized_summary.txt")
    stages = load_stage_summaries(run_dir / "backend_optimization_summary.txt")
    if not stages:
        stages = parse_stage_summaries(backend_summary)

    anchor_camera = {
        "xi": fnum(backend_summary.get("anchor_camera_xi")),
        "alpha": fnum(backend_summary.get("anchor_camera_alpha")),
        "fu": fnum(backend_summary.get("anchor_camera_fu")),
        "fv": fnum(backend_summary.get("anchor_camera_fv")),
        "cu": fnum(backend_summary.get("anchor_camera_cu")),
        "cv": fnum(backend_summary.get("anchor_camera_cv")),
    }
    optimized_camera = {
        "xi": fnum(backend_summary.get("optimized_camera_xi")),
        "alpha": fnum(backend_summary.get("optimized_camera_alpha")),
        "fu": fnum(backend_summary.get("optimized_camera_fu")),
        "fv": fnum(backend_summary.get("optimized_camera_fv")),
        "cu": fnum(backend_summary.get("optimized_camera_cu")),
        "cv": fnum(backend_summary.get("optimized_camera_cv")),
    }

    metadata = {
        "runName": run_dir.name,
        "runDir": str(run_dir),
        "frameCount": len(frames),
        "frameBoardCount": sum(f["boardCount"] for f in frames),
        "initialOverallRmse": fnum(backend_summary.get("initial_overall_rmse")),
        "optimizedOverallRmse": fnum(backend_summary.get("optimized_overall_rmse")),
        "initialCost": fnum(cost_initial.get("backend_problem_total_weighted_cost")),
        "optimizedCost": fnum(cost_final.get("backend_problem_total_weighted_cost")),
        "designVariableCount": inum(backend_summary.get("design_variable_count")),
        "errorTermCount": inum(backend_summary.get("error_term_count")),
        "backendMaxIterations": inum(backend_summary.get("backend_max_iterations")),
        "optimizeFramePoses": inum(problem_summary.get("optimize_frame_poses")),
        "optimizeBoardPoses": inum(problem_summary.get("optimize_board_poses")),
        "optimizeIntrinsics": inum(problem_summary.get("optimize_intrinsics")),
        "delayedIntrinsicsRelease": inum(problem_summary.get("delayed_intrinsics_release")),
        "intrinsicsReleaseIteration": inum(problem_summary.get("intrinsics_release_iteration")),
        "anchorCamera": anchor_camera,
        "optimizedCamera": optimized_camera,
        "cameraDelta": camera_delta(anchor_camera, optimized_camera),
        "stages": stages,
        "stageTrace": stage_trace_points(stages),
        "jacobianBlockInfluenceAvailable": block_influence["available"],
        "localOptimumCaveat": (
            "若 jacobian block influence 可用，图中 JᵀJ trace/logdet/rank 与 Jᵀr norm "
            "来自 ASLAM backend weighted Jacobian，按 frame-board 与变量块聚合。"
            "它仍是 block Fisher/Hessian proxy，不是完整 marginal covariance；"
            "严谨局部最优判断仍需多初值扰动或 leave-one-out ablation。"
        ),
    }
    return {"frames": frames, "metadata": metadata}


def sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage5 内参 BA 因子影响图</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d7deea;--blue:#2563eb;--green:#15803d;--violet:#7c3aed;--soft:#f8fafc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{display:grid;grid-template-columns:320px 1fr 390px;grid-template-rows:auto 1fr 310px;height:100vh;min-height:820px}
header{grid-column:1/4;background:var(--panel);border-bottom:1px solid var(--line);padding:12px 18px;display:flex;justify-content:space-between;gap:16px;align-items:center}h1{font-size:18px;margin:0}.meta{color:var(--muted);font-size:12px}
aside,.right,.bottom,.graphWrap{background:var(--panel)}aside{border-right:1px solid var(--line);padding:14px;overflow:auto}.right{border-left:1px solid var(--line);padding:14px;overflow:auto}.bottom{grid-column:1/4;border-top:1px solid var(--line);padding:10px 16px;overflow:hidden}
label{display:grid;gap:4px;color:var(--muted);font-size:12px;margin-bottom:10px}input,select,button{border:1px solid var(--line);border-radius:6px;padding:7px 8px;background:#fff;color:var(--ink);font:inherit}button{cursor:pointer}.chips{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}.chip{border:1px solid var(--line);border-radius:999px;padding:4px 8px;background:#fbfcff;color:var(--muted)}
.graphWrap{position:relative;overflow:hidden}#graph{width:100%;height:100%;display:block;background:linear-gradient(#eef2f8 1px,transparent 1px),linear-gradient(90deg,#eef2f8 1px,transparent 1px);background-size:28px 28px}.node{cursor:pointer}.node rect,.node circle,.node polygon{stroke:#243047;stroke-width:1.15}.node text{pointer-events:none;fill:#111827;font-size:11.5px;font-weight:650}.sub{fill:#536176!important;font-size:10px!important;font-weight:500!important}
.edge{stroke:#8d99ae;stroke-opacity:.75;fill:none}.edge.k{stroke:#2563eb}.edge.pose{stroke:#7c3aed}.edge.layout{stroke:#15803d}.selected{stroke:#111827!important;stroke-width:2.8!important}.toolbar{position:absolute;right:12px;top:12px;display:flex;gap:6px;background:rgba(255,255,255,.93);padding:6px;border:1px solid var(--line);border-radius:8px}.tip{position:absolute;pointer-events:none;background:#111827;color:#fff;padding:8px 10px;border-radius:7px;white-space:pre-line;max-width:390px;opacity:0;transform:translate(10px,10px)}
.section{margin-bottom:16px}.section h2{font-size:13px;margin:0 0 8px}.kv{display:grid;grid-template-columns:minmax(136px,1fr) auto;gap:6px 10px}.kv div:nth-child(odd){color:var(--muted)}.boardCard{border:1px solid var(--line);border-radius:7px;padding:8px;background:#fbfcff;margin-bottom:8px}.legendRow{display:flex;align-items:center;gap:8px;color:var(--muted);margin-bottom:6px}.sw{width:16px;height:16px;border:1px solid #222;border-radius:4px}.callout{background:var(--soft);border:1px solid var(--line);border-radius:7px;padding:9px;color:#394457}.smallTable{width:100%;border-collapse:collapse}.smallTable td{border-bottom:1px solid #edf1f7;padding:4px 2px}.smallTable td:nth-child(1){color:var(--muted)}.smallTable td:nth-child(2),.smallTable td:nth-child(3){text-align:right;font-variant-numeric:tabular-nums}.charts{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;height:100%}.chartTitle{font-size:12px;color:var(--muted);margin-bottom:5px}.bar{cursor:pointer}.bar:hover{opacity:.75}.axisText{fill:#64748b;font-size:10px}
</style>
</head>
<body>
<div class="app">
<header><div><h1>Stage5 内参 BA 因子影响图</h1><div class="meta" id="head"></div></div><div class="meta">相机模型 K / 每帧位姿 T_camera_reference / 多板布局 T_reference_board / JᵀJ block proxy</div></header>
<aside>
<label>搜索 frame / timestamp / label<input id="search" placeholder="例如 48 或 166703"></label>
<label>排序<select id="sort"><option value="frame">frame index</option><option value="cameraInfluenceProxy">帧→K 影响 从大到小</option><option value="finalRmse">最终 RMSE 从大到小</option><option value="reduction">残差下降 从大到小</option><option value="polarProxy">边缘/偏移视角 从大到小</option></select></label>
<label>筛选指标<select id="metric"><option value="finalRmse">最终 RMSE</option><option value="initialRmse">初始 RMSE</option><option value="cameraInfluenceProxy">帧→K 影响</option><option value="polarProxy">边缘/偏移视角 proxy</option></select></label>
<label>阈值<input id="threshold" type="number" step="0.1" value="1.0"></label>
<label><span><input id="onlyHigh" type="checkbox"> 只显示筛选项</span></label>
<button id="fit">适配屏幕</button> <button id="collapse">折叠全部</button> <button id="svg">导出 SVG</button>
<div class="chips" id="chips"></div>
<div class="section"><h2>图例</h2>
<div class="legendRow"><span class="sw" style="background:#dbeafe"></span>相机模型 K / 畸变 DS</div>
<div class="legendRow"><span class="sw" style="background:#ede9fe"></span>每帧位姿 T_camera_reference</div>
<div class="legendRow"><span class="sw" style="background:#dcfce7"></span>多板布局 T_reference_board</div>
<div class="legendRow"><span class="sw" style="background:#fff7ed"></span>重投影因子 f(frame, board)</div>
<div class="legendRow">颜色：Frame 节点表示最终 RMSE，绿/黄/橙/红依次变差。</div>
<div class="legendRow">线宽：蓝边=帧对 K 的 JᵀJ trace / fallback proxy；紫边=board 对 T_camera_reference 的 JᵀJ trace；绿边=board layout 更新量。</div>
</div>
</aside>
<main class="graphWrap"><div class="toolbar"><button id="minus">-</button><button id="plus">+</button><button id="reset">Reset</button></div><svg id="graph"></svg><div class="tip" id="tip"></div></main>
<section class="right"><div class="section"><h2>当前选中帧</h2><div class="kv" id="kv"></div></div><div class="section"><h2>该帧内的 Board 因子</h2><div id="boards"></div></div><div class="section"><h2>相机模型更新</h2><div id="camera"></div></div><div class="section"><h2>BA 收敛 / 局部最优诊断</h2><div class="kv" id="health"></div><p class="callout" id="caveat"></p></div></section>
<section class="bottom"><div class="charts"><div><div class="chartTitle">帧→相机模型 K 影响 proxy</div><svg id="chartK" width="100%" height="245"></svg></div><div><div class="chartTitle">最终 RMSE / 残差健康度</div><svg id="chartR" width="100%" height="245"></svg></div><div><div class="chartTitle">最大 Board layout 更新</div><svg id="chartB" width="100%" height="245"></svg></div><div><div class="chartTitle">Stage 级 cost 收敛</div><svg id="chartS" width="100%" height="245"></svg></div></div></section>
</div>
<script>
const DATA=__DATA_JSON__;const frames=DATA.frames;const meta=DATA.metadata;const ns='http://www.w3.org/2000/svg';const st={expanded:new Set(),selected:frames[0]?.frameIndex??0,scale:1,tx:0,ty:0};const svg=document.getElementById('graph'),tip=document.getElementById('tip');
function fmt(v,d=3){if(!Number.isFinite(v))return'n/a';const a=Math.abs(v);if(a!==0&&a<1e-3)return Number(v).toExponential(2);if(a>9999)return Number(v).toExponential(2);return Number(v).toFixed(d)}function fmtPx(v){return `${fmt(v,3)} px`}function fmtMm(v){return `${fmt(v,2)} mm`}function fmtDeg(v){return `${fmt(v,2)} deg`}function make(t,a={}){const e=document.createElementNS(ns,t);for(const[k,v]of Object.entries(a))if(v!==undefined&&v!==null)e.setAttribute(k,v);return e}function clear(e){while(e.firstChild)e.removeChild(e.firstChild)}
function color(v){if(!Number.isFinite(v))return'#e5e7eb';if(v<.5)return'#dcfce7';if(v<1)return'#fef9c3';if(v<2)return'#fed7aa';return'#fecaca'}function sw(v){return Math.max(1.2,Math.min(8,1+Math.sqrt(Math.max(0,v))*18))}function visible(){const q=document.getElementById('search').value.toLowerCase().trim(),m=document.getElementById('metric').value,t=Number(document.getElementById('threshold').value),oh=document.getElementById('onlyHigh').checked,s=document.getElementById('sort').value;let out=frames.filter(f=>{const txt=`${f.frameIndex} ${f.frameLabel} ${f.timestamp}`.toLowerCase();if(q&&!txt.includes(q))return false;if(oh&&!((f[m]??-1)>=t))return false;return true});out.sort((a,b)=>s==='frame'?a.frameIndex-b.frameIndex:(b[s==='reduction'?'residualReduction':s]??-1e9)-(a[s==='reduction'?'residualReduction':s]??-1e9));return out}
function show(e,text){tip.textContent=text;tip.style.left=e.clientX+'px';tip.style.top=e.clientY+'px';tip.style.opacity=1}function hide(){tip.style.opacity=0}function txt(g,x,y,s,c=''){const e=make('text',{x,y,'text-anchor':'middle',class:c});e.textContent=s;g.appendChild(e);return e}function line(g,x1,y1,x2,y2,cls,w){g.appendChild(make('path',{d:`M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`,class:'edge '+cls,'stroke-width':w}))}
function influenceWidth(v,fallback){const x=Number.isFinite(v)&&v>0?Math.log1p(v)/4:fallback;return sw(x)}function renderGraph(){clear(svg);const W=svg.clientWidth||900,H=svg.clientHeight||600;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const root=make('g',{transform:`translate(${st.tx},${st.ty}) scale(${st.scale})`});svg.appendChild(root);const fs=visible(),top=84,gap=94,kx=92,fx=315,bx=565;const kg=make('g',{class:'node'});kg.appendChild(make('rect',{x:kx-58,y:20,width:116,height:52,rx:8,fill:'#dbeafe'}));txt(kg,kx,42,'相机模型 K');txt(kg,kx,59,'xi α fu fv cu cv','sub');root.appendChild(kg);fs.forEach((f,i)=>{const y=top+i*gap,sel=f.frameIndex===st.selected;const fg=make('g',{class:'node'});fg.appendChild(make('rect',{x:fx-78,y:y-27,width:156,height:56,rx:8,fill:color(f.finalRmse),class:sel?'selected':''}));txt(fg,fx,y-6,`Frame ${f.frameIndex}`);txt(fg,fx,y+12,`T位姿 JtJ ${fmt(f.framePoseHessianTrace,1)} | ${fmtPx(f.finalRmse)}`,'sub');fg.onmousemove=e=>show(e,`Frame ${f.frameIndex}\n${f.frameLabel}\nboards ${f.boardCount}, corners ${f.cornerCount}\nRMSE ${fmtPx(f.initialRmse)} -> ${fmtPx(f.finalRmse)}\n残差下降 ${fmtPx(f.residualReduction)}\nK block JtJ trace ${fmt(f.cameraHessianTrace,3)}, rank ${fmt(f.cameraHessianRankProxy,3)}, |Jtr| ${fmt(f.cameraGradientNorm,3)}\nT_camera_reference JtJ trace ${fmt(f.framePoseHessianTrace,3)}, rank ${fmt(f.framePoseHessianRankProxy,3)}, |Jtr| ${fmt(f.framePoseGradientNorm,3)}\nT_reference_board JtJ trace ${fmt(f.boardLayoutHessianTrace,3)}, rank ${fmt(f.boardLayoutHessianRankProxy,3)}\n边缘/偏移视角 proxy ${fmt(f.polarProxy,3)}`);fg.onmouseleave=hide;fg.onclick=e=>{e.stopPropagation();st.selected=f.frameIndex;if(st.expanded.has(f.frameIndex))st.expanded.delete(f.frameIndex);else st.expanded.add(f.frameIndex);renderAll()};root.appendChild(fg);line(root,kx+58,46,fx-78,y,'k',influenceWidth(f.cameraHessianTrace,f.cameraInfluenceProxy));if(st.expanded.has(f.frameIndex)){f.boards.forEach((b,j)=>{const x=bx+j*116;const bg=make('g',{class:'node'});bg.appendChild(make('rect',{x:x-46,y:y-24,width:92,height:48,rx:7,fill:'#dcfce7'}));txt(bg,x,y-4,`Board ${b.boardId}`);txt(bg,x,y+13,`T位姿 ${fmt(b.framePoseHessianTrace,1)}`,'sub');bg.onmousemove=e=>show(e,`Frame ${f.frameIndex} / Board ${b.boardId}\nRMSE ${fmtPx(b.initialRmse)} -> ${fmtPx(b.finalRmse)}\nK block JtJ trace ${fmt(b.cameraHessianTrace,3)}, rank ${fmt(b.cameraHessianRankProxy,3)}, |Jtr| ${fmt(b.cameraGradientNorm,3)}\nT_camera_reference JtJ trace ${fmt(b.framePoseHessianTrace,3)}, rank ${fmt(b.framePoseHessianRankProxy,3)}, |Jtr| ${fmt(b.framePoseGradientNorm,3)}\nT_reference_board JtJ trace ${fmt(b.boardLayoutHessianTrace,3)}, rank ${fmt(b.boardLayoutHessianRankProxy,3)}, |Jtr| ${fmt(b.boardLayoutGradientNorm,3)}\nlayout update ${fmtMm(b.layoutTranslationDeltaMm)} / ${fmtDeg(b.layoutRotationDeltaDeg)}\ncorners ${b.cornerCount}`);bg.onmouseleave=hide;root.appendChild(bg);line(root,fx+78,y,x-46,y,'pose',influenceWidth(b.framePoseHessianTrace,b.framePoseInfluenceProxy));line(root,x,y+24,x,y+60,'layout',1+Math.min(6,b.layoutTranslationDeltaMm/4));const fac=make('g',{class:'node'});fac.appendChild(make('polygon',{points:`${x-26},${y+68} ${x},${y+54} ${x+26},${y+68} ${x},${y+82}`,fill:'#fff7ed'}));txt(fac,x,y+72,'重投影 f','sub');root.appendChild(fac)})}})}
function renderSide(){const f=frames.find(x=>x.frameIndex===st.selected)||frames[0];const kv=document.getElementById('kv');clear(kv);[['frame',f.frameIndex],['label',f.frameLabel],['boards',f.boardCount],['corners',f.cornerCount],['RMSE',`${fmtPx(f.initialRmse)} -> ${fmtPx(f.finalRmse)}`],['残差下降',fmtPx(f.residualReduction)],['K: JtJ trace / rank',`${fmt(f.cameraHessianTrace,3)} / ${fmt(f.cameraHessianRankProxy,3)}`],['K: |Jtr|',fmt(f.cameraGradientNorm,3)],['T_camera_reference: JtJ trace / rank',`${fmt(f.framePoseHessianTrace,3)} / ${fmt(f.framePoseHessianRankProxy,3)}`],['T_camera_reference: |Jtr|',fmt(f.framePoseGradientNorm,3)],['T_reference_board: JtJ trace / rank',`${fmt(f.boardLayoutHessianTrace,3)} / ${fmt(f.boardLayoutHessianRankProxy,3)}`],['边缘/偏移视角 proxy',fmt(f.polarProxy)]].forEach(([k,v])=>{kv.append(k);kv.append(String(v))});const bs=document.getElementById('boards');clear(bs);f.boards.forEach(b=>{const d=document.createElement('div');d.className='boardCard';d.innerHTML=`<b>Board ${b.boardId}</b><div class="meta">RMSE ${fmtPx(b.initialRmse)} -> ${fmtPx(b.finalRmse)}</div><table class="smallTable"><tbody><tr><td>K JtJ/rank</td><td>${fmt(b.cameraHessianTrace,3)}</td><td>${fmt(b.cameraHessianRankProxy,3)}</td></tr><tr><td>T_camera_reference JtJ/rank</td><td>${fmt(b.framePoseHessianTrace,3)}</td><td>${fmt(b.framePoseHessianRankProxy,3)}</td></tr><tr><td>T_reference_board JtJ/rank</td><td>${fmt(b.boardLayoutHessianTrace,3)}</td><td>${fmt(b.boardLayoutHessianRankProxy,3)}</td></tr><tr><td>layout Δ</td><td>${fmtMm(b.layoutTranslationDeltaMm)}</td><td>${fmtDeg(b.layoutRotationDeltaDeg)}</td></tr></tbody></table>`;bs.appendChild(d)});renderCamera()}
function renderCamera(){const el=document.getElementById('camera');const a=meta.anchorCamera||{},o=meta.optimizedCamera||{},d=meta.cameraDelta||{};let html='<table class="smallTable"><tbody><tr><td>参数</td><td>初值</td><td>优化后 / Δ</td></tr>';['xi','alpha','fu','fv','cu','cv'].forEach(k=>{html+=`<tr><td>${k}</td><td>${fmt(a[k],6)}</td><td>${fmt(o[k],6)} / ${fmt(d[k],6)}</td></tr>`});html+='</tbody></table>';el.innerHTML=html}
function renderHealth(){const h=document.getElementById('health');clear(h);const stages=meta.stages||[];[['变量数',meta.designVariableCount],['误差项数',meta.errorTermCount],['overall RMSE',`${fmtPx(meta.initialOverallRmse)} -> ${fmtPx(meta.optimizedOverallRmse)}`],['problem cost',`${fmt(meta.initialCost)} -> ${fmt(meta.optimizedCost)}`],['Jacobian block influence',meta.jacobianBlockInfluenceAvailable?'已记录':'未记录，使用残差 proxy'],['优化 K / frame / board',`${meta.optimizeIntrinsics}/${meta.optimizeFramePoses}/${meta.optimizeBoardPoses}`],['延迟释放内参',`${meta.delayedIntrinsicsRelease} @ iter ${meta.intrinsicsReleaseIteration}`]].forEach(([k,v])=>{h.append(k);h.append(String(v))});stages.forEach(s=>{h.append(`stage: ${s.stage_label}`);h.append(`iter ${s.stage_iterations||'n/a'}, Δx ${s.stage_delta_x_final||'n/a'}, ΔJ ${s.stage_delta_j_final||'n/a'}`)});document.getElementById('caveat').textContent=meta.localOptimumCaveat}
function chart(id,vals,colorFn){const el=document.getElementById(id);clear(el);const W=el.clientWidth||300,H=230,p=28,max=Math.max(.001,...vals.map(v=>Math.max(0,v.val)));vals.forEach((v,i)=>{const slot=(W-2*p)/vals.length,x=p+i*slot,w=Math.max(3,slot*.72),h=(H-2*p)*Math.max(0,v.val)/max;const r=make('rect',{x,y:H-p-h,width:w,height:h,fill:colorFn?colorFn(v):'#2563eb',class:'bar'});r.onclick=()=>{st.selected=v.frame;st.expanded.add(v.frame);renderAll()};el.appendChild(r);if(slot>16){const t=make('text',{x:x+slot/2,y:H-8,'text-anchor':'middle',class:'axisText'});t.textContent=v.frame;el.appendChild(t)}})}
function stageChart(){const el=document.getElementById('chartS');clear(el);const pts=(meta.stageTrace||[]).filter(p=>Number.isFinite(p.cost));const W=el.clientWidth||300,H=230,p=32;if(!pts.length)return;const max=Math.max(...pts.map(p=>p.cost)),min=Math.min(...pts.map(p=>p.cost)),xmax=Math.max(...pts.map(p=>p.x),1);let last=null;pts.forEach(pt=>{const x=p+(W-2*p)*pt.x/xmax,y=H-p-(H-2*p)*(pt.cost-min)/(max-min||1);if(last)el.appendChild(make('path',{d:`M${last.x},${last.y} L${x},${y}`,stroke:'#7c3aed','stroke-width':2,fill:'none'}));el.appendChild(make('circle',{cx:x,cy:y,r:4,fill:'#7c3aed'}));const t=make('text',{x,y:y-8,'text-anchor':'middle',class:'axisText'});t.textContent=`${pt.stage}:${pt.kind}`;el.appendChild(t);last={x,y}})}
function renderCharts(){chart('chartK',frames.map(f=>({frame:f.frameIndex,val:meta.jacobianBlockInfluenceAvailable?Math.log1p(f.cameraHessianTrace):f.cameraInfluenceProxy})));chart('chartR',frames.map(f=>({frame:f.frameIndex,val:f.finalRmse})),v=>color(v.val));chart('chartB',frames.map(f=>({frame:f.frameIndex,val:Math.max(...f.boards.map(b=>b.layoutTranslationDeltaMm),0)})),v=>'#15803d');stageChart()}
function renderStats(){document.getElementById('head').textContent=`${meta.runName} | backend frames ${meta.frameCount}, frame-board ${meta.frameBoardCount}`;const c=document.getElementById('chips');clear(c);[['RMSE',`${fmt(meta.initialOverallRmse)}→${fmt(meta.optimizedOverallRmse)} px`],['cost',`${fmt(meta.initialCost)}→${fmt(meta.optimizedCost)}`],['变量',meta.designVariableCount],['误差项',meta.errorTermCount]].forEach(([k,v])=>{const e=document.createElement('span');e.className='chip';e.textContent=`${k}: ${v}`;c.appendChild(e)})}
function fit(){const fs=visible(),maxB=Math.max(...fs.map(f=>f.boardCount),1);st.scale=Math.max(.18,Math.min(1.15,(svg.clientHeight-40)/(120+fs.length*94),(svg.clientWidth-40)/(760+maxB*116)));st.tx=20;st.ty=10;renderGraph()}function exportSvg(){const c=svg.cloneNode(true);c.setAttribute('xmlns',ns);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(c)],{type:'image/svg+xml'}));a.download='stage5_ba_influence_graph.svg';a.click();URL.revokeObjectURL(a.href)}function renderAll(){renderStats();renderGraph();renderSide();renderHealth();renderCharts()}
['search','sort','metric','threshold','onlyHigh'].forEach(id=>document.getElementById(id).oninput=renderAll);document.getElementById('fit').onclick=fit;document.getElementById('collapse').onclick=()=>{st.expanded.clear();renderAll()};document.getElementById('plus').onclick=()=>{st.scale*=1.15;renderGraph()};document.getElementById('minus').onclick=()=>{st.scale*=.85;renderGraph()};document.getElementById('reset').onclick=()=>{st.scale=1;st.tx=0;st.ty=0;renderGraph()};document.getElementById('svg').onclick=exportSvg;let drag=false,last=null;svg.onmousedown=e=>{drag=true;last=[e.clientX,e.clientY]};window.onmouseup=()=>drag=false;window.onmousemove=e=>{if(!drag)return;st.tx+=e.clientX-last[0];st.ty+=e.clientY-last[1];last=[e.clientX,e.clientY];renderGraph()};svg.onwheel=e=>{e.preventDefault();st.scale*=e.deltaY<0?1.08:.92;renderGraph()};renderAll();setTimeout(fit,50);
</script>
</body>
</html>
"""


def write_trace(path, payload):
    fields = [
        "frame_index", "frame_label", "board_count", "corner_count",
        "initial_rmse", "final_rmse", "residual_reduction",
        "camera_influence_proxy", "intrinsics_sensitivity_proxy",
        "coverage_proxy", "polar_proxy", "board_stats",
    ]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fr in payload["frames"]:
            stats = ";".join(
                f"{b['boardId']}:{b['initialRmse']:.6g}->{b['finalRmse']:.6g}:"
                f"Tproxy={b['framePoseInfluenceProxy']:.6g}:"
                f"layout={b['layoutTranslationDeltaMm']:.6g}mm/{b['layoutRotationDeltaDeg']:.6g}deg"
                for b in fr["boards"]
            )
            w.writerow({
                "frame_index": fr["frameIndex"],
                "frame_label": fr["frameLabel"],
                "board_count": fr["boardCount"],
                "corner_count": fr["cornerCount"],
                "initial_rmse": fr["initialRmse"],
                "final_rmse": fr["finalRmse"],
                "residual_reduction": fr["residualReduction"],
                "camera_influence_proxy": fr["cameraInfluenceProxy"],
                "intrinsics_sensitivity_proxy": fr["intrinsicsSensitivityProxy"],
                "coverage_proxy": fr["coverageProxy"],
                "polar_proxy": fr["polarProxy"],
                "board_stats": stats,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(Path(args.run_dir))
    data = json.dumps(sanitize(payload), ensure_ascii=False, allow_nan=False)
    (out / "viewer.html").write_text(HTML.replace("__DATA_JSON__", data), encoding="utf-8")
    write_trace(out / "stage5_ba_influence_trace.csv", payload)
    (out / "summary.txt").write_text(
        "\n".join([
            f"run_name: {payload['metadata']['runName']}",
            f"frame_count: {payload['metadata']['frameCount']}",
            f"frame_board_count: {payload['metadata']['frameBoardCount']}",
            f"initial_overall_rmse: {payload['metadata']['initialOverallRmse']:.9g}",
            f"optimized_overall_rmse: {payload['metadata']['optimizedOverallRmse']:.9g}",
            f"initial_cost: {payload['metadata']['initialCost']:.9g}",
            f"optimized_cost: {payload['metadata']['optimizedCost']:.9g}",
            f"viewer_html: {out / 'viewer.html'}",
            f"trace_csv: {out / 'stage5_ba_influence_trace.csv'}",
            "note: influence values are residual/cost and geometry proxies, not exact Hessian marginal contribution.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote viewer: {out / 'viewer.html'}")
    print(f"Wrote trace: {out / 'stage5_ba_influence_trace.csv'}")


if __name__ == "__main__":
    main()
