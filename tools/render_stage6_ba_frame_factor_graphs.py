#!/usr/bin/env python3
"""Render Stage6 frame-expanded BA factor traces as SVG figures."""

import argparse
import csv
import html
import json
import math
import os
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


def esc(text):
    return html.escape(str(text), quote=True)


def color_for_delta(delta):
    if not math.isfinite(delta):
        return "#8a8f98"
    if delta < -1e-6:
        return "#2e7d32"
    if delta > 1e-6:
        return "#c62828"
    return "#6b7280"


def parse_board_factor_counts(text):
    result = []
    if not text:
        return result
    for part in text.split(";"):
        if ":" not in part or "/" not in part:
            continue
        board, counts = part.split(":", 1)
        cam0, cam1 = counts.split("/", 1)
        result.append((board, inum(cam0), inum(cam1)))
    return result


def parse_board_residual_stats(text):
    result = {}
    if not text:
        return result
    for part in text.split(";"):
        if ":" not in part:
            continue
        board, payload = part.split(":", 1)
        sections = payload.split("|")
        if len(sections) < 5:
            continue
        counts = sections[0].split("/")
        overall = sections[1].split("/")
        cam0 = sections[2].split("/")
        cam1 = sections[3].split("/")
        result[board] = {
            "boardId": board,
            "cam0": inum(counts[0]) if len(counts) > 0 else 0,
            "cam1": inum(counts[1]) if len(counts) > 1 else 0,
            "outer": inum(counts[2]) if len(counts) > 2 else 0,
            "internal": inum(counts[3]) if len(counts) > 3 else 0,
            "initialRmse": fnum(overall[0]) if len(overall) > 0 else float("nan"),
            "finalRmse": fnum(overall[1]) if len(overall) > 1 else float("nan"),
            "rmseDelta": fnum(overall[2]) if len(overall) > 2 else float("nan"),
            "initialCam0Rmse": fnum(cam0[0]) if len(cam0) > 0 else float("nan"),
            "finalCam0Rmse": fnum(cam0[1]) if len(cam0) > 1 else float("nan"),
            "cam0Delta": fnum(cam0[2]) if len(cam0) > 2 else float("nan"),
            "initialCam1Rmse": fnum(cam1[0]) if len(cam1) > 0 else float("nan"),
            "finalCam1Rmse": fnum(cam1[1]) if len(cam1) > 1 else float("nan"),
            "cam1Delta": fnum(cam1[2]) if len(cam1) > 2 else float("nan"),
            "tSensitivityProxy": fnum(sections[4]),
            "boardPoseInitToFinalRotationDeg": (
                fnum(sections[5].split("/")[0]) if len(sections) > 5 else float("nan")
            ),
            "boardPoseInitToFinalTranslationM": (
                fnum(sections[5].split("/")[1]) if len(sections) > 5 and "/" in sections[5] else float("nan")
            ),
            "boardPoseFinalBaRotationDeg": (
                fnum(sections[6].split("/")[0]) if len(sections) > 6 else float("nan")
            ),
            "boardPoseFinalBaTranslationM": (
                fnum(sections[6].split("/")[1]) if len(sections) > 6 and "/" in sections[6] else float("nan")
            ),
        }
    return result


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: inum(r.get("pair_index", "0")))
    return rows


def empty_jacobian_metrics():
    return {
        "hessianTrace": float("nan"),
        "gradientNorm": float("nan"),
        "rankProxy": 0,
        "conditionNumber": float("nan"),
        "rmseLike": float("nan"),
        "residualDimension": 0,
    }


def summarize_jacobian_row(row):
    return {
        "hessianTrace": fnum(row.get("hessian_trace")),
        "gradientNorm": fnum(row.get("gradient_norm")),
        "rankProxy": inum(row.get("hessian_rank_proxy")),
        "conditionNumber": fnum(row.get("condition_number")),
        "rmseLike": fnum(row.get("rmse_like")),
        "residualDimension": inum(row.get("residual_dimension")),
    }


def load_jacobian_diagnostics(trace_csv):
    p = Path(trace_csv).resolve().parent / "stereo_jacobian_block_diagnostics.csv"
    result = {
        "available": False,
        "path": str(p),
        "residualMode": "",
        "frame": {},
        "boards": {},
    }
    if not p.exists():
        return result
    result["available"] = True
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            pair = inum(row.get("pair_index"), -1)
            board_id = row.get("board_id", "-1")
            scope = row.get("scope", "")
            block = row.get("variable_block", "")
            if row.get("residual_mode"):
                result["residualMode"] = row.get("residual_mode", "")
            metrics = summarize_jacobian_row(row)
            if scope == "frame":
                frame = result["frame"].setdefault(pair, {})
                if block == "T_1_0":
                    frame["tStereo"] = metrics
                elif block == "T_cam0_world":
                    frame["framePose"] = metrics
            elif scope == "pair_board":
                boards = result["boards"].setdefault(pair, {})
                board = boards.setdefault(str(inum(board_id)), {})
                if block == "T_1_0":
                    board["tStereo"] = metrics
                elif block == "T_cam0_world":
                    board["framePose"] = metrics
                elif block == "T_world_board":
                    board["boardPose"] = metrics
    return result


def row_to_viewer_record(row, jacobian=None):
    jacobian = jacobian or {}
    pair = inum(row.get("pair_index"))
    left = row.get("left_frame_label", "")
    right = row.get("right_frame_label", "")
    timestamp = ""
    for label in (left, right):
        parts = label.split("_")
        if len(parts) >= 3:
            timestamp = parts[2]
            break
    boards = parse_board_factor_counts(row.get("board_factor_counts", ""))
    residual_stats = parse_board_residual_stats(row.get("board_residual_stats", ""))
    board_ids = sorted(
        set([board_id for board_id, _, _ in boards]) | set(residual_stats.keys()),
        key=lambda x: inum(x),
    )
    board_records = []
    count_by_board = {board_id: (cam0, cam1) for board_id, cam0, cam1 in boards}
    board_jac = (jacobian.get("boards") or {}).get(pair, {})
    for board_id in board_ids:
        record = residual_stats.get(board_id, {"boardId": board_id})
        if board_id in count_by_board:
            record = dict(record)
            record["cam0"], record["cam1"] = count_by_board[board_id]
        record.setdefault("cam0", 0)
        record.setdefault("cam1", 0)
        record.setdefault("outer", 0)
        record.setdefault("internal", 0)
        record.setdefault("initialRmse", float("nan"))
        record.setdefault("finalRmse", float("nan"))
        record.setdefault("rmseDelta", float("nan"))
        record.setdefault("initialCam0Rmse", float("nan"))
        record.setdefault("finalCam0Rmse", float("nan"))
        record.setdefault("cam0Delta", float("nan"))
        record.setdefault("initialCam1Rmse", float("nan"))
        record.setdefault("finalCam1Rmse", float("nan"))
        record.setdefault("cam1Delta", float("nan"))
        record.setdefault("tSensitivityProxy", float("nan"))
        record.setdefault("boardPoseInitToFinalRotationDeg", float("nan"))
        record.setdefault("boardPoseInitToFinalTranslationM", float("nan"))
        record.setdefault("boardPoseFinalBaRotationDeg", float("nan"))
        record.setdefault("boardPoseFinalBaTranslationM", float("nan"))
        jac_record = board_jac.get(str(inum(board_id)), {})
        record.setdefault("jacobian", {
            "tStereo": jac_record.get("tStereo", empty_jacobian_metrics()),
            "framePose": jac_record.get("framePose", empty_jacobian_metrics()),
            "boardPose": jac_record.get("boardPose", empty_jacobian_metrics()),
        })
        board_records.append(record)
    frame_jac = (jacobian.get("frame") or {}).get(pair, {})
    return {
        "pairIndex": pair,
        "frameId": f"Frame_{pair}",
        "leftFrame": left,
        "rightFrame": right,
        "timestamp": timestamp,
        "selectedBoardIds": [b for b in row.get("selected_board_ids", "").split(";") if b],
        "boardFactors": board_records,
        "boardCount": inum(row.get("selected_board_count")),
        "cam0Factors": inum(row.get("cam0_reprojection_factor_count")),
        "cam1Factors": inum(row.get("cam1_reprojection_factor_count")),
        "tFactors": inum(row.get("t_1_0_reprojection_factor_count")),
        "totalFactors": inum(row.get("total_reprojection_factor_count")),
        "outerFactors": inum(row.get("outer_factor_count")),
        "internalFactors": inum(row.get("internal_factor_count")),
        "initialRmse": fnum(row.get("initial_overall_rmse")),
        "finalRmse": fnum(row.get("final_overall_rmse")),
        "rmseDelta": fnum(row.get("overall_rmse_delta")),
        "initialCam0Rmse": fnum(row.get("initial_cam0_rmse")),
        "finalCam0Rmse": fnum(row.get("final_cam0_rmse")),
        "cam0Delta": fnum(row.get("cam0_rmse_delta")),
        "initialCam1Rmse": fnum(row.get("initial_cam1_rmse")),
        "finalCam1Rmse": fnum(row.get("final_cam1_rmse")),
        "cam1Delta": fnum(row.get("cam1_rmse_delta")),
        "state": row.get("residual_state", ""),
        "tConstrained": row.get("t_1_0_constrained_by_frame", "0") == "1",
        "tSensitivityProxy": fnum(row.get("t_1_0_sensitivity_proxy")),
        "tInitToFinalRotationDeg": fnum(row.get("t_1_0_init_to_final_rotation_deg")),
        "tInitToFinalTranslationM": fnum(row.get("t_1_0_init_to_final_translation_m")),
        "tFinalBaRotationDeg": fnum(row.get("t_1_0_final_ba_rotation_deg")),
        "tFinalBaTranslationM": fnum(row.get("t_1_0_final_ba_translation_m")),
        "framePoseInitToFinalRotationDeg": fnum(row.get("frame_pose_init_to_final_rotation_deg")),
        "framePoseInitToFinalTranslationM": fnum(row.get("frame_pose_init_to_final_translation_m")),
        "framePoseFinalBaRotationDeg": fnum(row.get("frame_pose_final_ba_rotation_deg")),
        "framePoseFinalBaTranslationM": fnum(row.get("frame_pose_final_ba_translation_m")),
        "optimizedVariableSummary": row.get("optimized_variable_summary", ""),
        "factorSource": row.get("factor_count_source", ""),
        "jacobian": {
            "available": bool(jacobian.get("available")),
            "residualMode": jacobian.get("residualMode", ""),
            "tStereo": frame_jac.get("tStereo", empty_jacobian_metrics()),
            "framePose": frame_jac.get("framePose", empty_jacobian_metrics()),
        },
    }


def load_key_value_file(path):
    values = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def load_metric_initial_final(path):
    p = Path(path)
    metrics = {}
    if not p.exists():
        return metrics
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            metric = row.get("metric")
            if not metric:
                continue
            metrics[metric] = {
                "initial": fnum(row.get("initial")),
                "final": fnum(row.get("final")),
            }
    return metrics


def load_iteration_log(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "iteration": inum(row.get("iteration")),
                "totalCost": fnum(row.get("objective", row.get("total_cost"))),
                "cam1Rmse": fnum(row.get("cam1_rmse", row.get("selected_cam1_rmse"))),
                "totalRmse": fnum(row.get("total_rmse", row.get("selected_total_rmse"))),
                "baselineLength": fnum(row.get("baseline_length")),
                "rotationAngleDeg": fnum(row.get("rotation_angle_deg")),
            })
    return rows


def is_truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_run_diagnostics(trace_csv):
    trace_dir = Path(trace_csv).resolve().parent
    ba = load_key_value_file(trace_dir / "stereo_global_sparse_ba_summary.txt")
    ext = load_key_value_file(trace_dir / "stereo_extrinsic_summary.txt")
    pair_init = load_key_value_file(trace_dir / "stereo_pair_init_summary.txt")
    initial_final = load_metric_initial_final(
        trace_dir / "stereo_global_sparse_ba_initial_vs_final.txt"
    )
    convergence = load_iteration_log(trace_dir / "stereo_global_sparse_ba_iteration_log.csv")
    jacobian = load_jacobian_diagnostics(trace_csv)
    if initial_final:
        if not convergence:
            convergence.append({
                "iteration": 0,
                "totalCost": initial_final.get("objective", {}).get("initial"),
                "cam1Rmse": initial_final.get("selected_cam1_rmse", {}).get("initial"),
                "totalRmse": initial_final.get("selected_total_rmse", {}).get("initial"),
                "baselineLength": fnum(pair_init.get("medoid_baseline_length")),
                "rotationAngleDeg": 0.0,
            })
            convergence.append({
                "iteration": inum(ba.get("iterations"), 1),
                "totalCost": initial_final.get("objective", {}).get("final"),
                "cam1Rmse": initial_final.get("selected_cam1_rmse", {}).get("final"),
                "totalRmse": initial_final.get("selected_total_rmse", {}).get("final"),
                "baselineLength": fnum(ext.get("baseline_length")),
                "rotationAngleDeg": fnum(ext.get("rotation_angle_deg")),
            })
    skip_final = is_truthy(ba.get("skip_final_global_ba"))
    iterations = inum(ba.get("iterations"))
    final_state_label = ba.get("final_state_label", "")
    no_final_state = skip_final or (
        iterations == 0
        and final_state_label
        and final_state_label != "after_final_global_ba"
    )
    return {
        "traceCsv": str(trace_csv),
        "sourceDir": str(trace_dir),
        "iterationsAvailable": len(convergence) > 2,
        "noFinalBa": no_final_state,
        "convergence": convergence,
        "summary": {
            "skipFinalGlobalBa": 1 if skip_final else 0,
            "finalStateLabel": final_state_label,
            "residualMode": ba.get("residual_mode", ""),
            "selectionBaResidualMode": ba.get("selection_ba_residual_mode", ""),
            "solverMode": ba.get("solver_mode", ""),
            "baselineLength": fnum(ext.get("baseline_length")),
            "rotationAngleDeg": fnum(ext.get("rotation_angle_deg")),
            "objectiveStart": fnum(ba.get("objective_start")),
            "objectiveFinal": fnum(ba.get("objective_final")),
            "iterations": inum(ba.get("iterations")),
            "selectedPairCount": inum(ba.get("selected_pair_count")),
            "eligiblePairCount": inum(ba.get("eligible_pair_count")),
            "initialCam1Rmse": fnum(ba.get("initial_selected_cam1_rmse")),
            "finalCam1Rmse": fnum(ba.get("final_selected_cam1_rmse")),
            "jacobianDiagnosticsAvailable": 1 if jacobian.get("available") else 0,
            "jacobianResidualMode": jacobian.get("residualMode", ""),
        },
        "jacobian": {
            "available": jacobian.get("available", False),
            "path": jacobian.get("path", ""),
            "residualMode": jacobian.get("residualMode", ""),
        },
    }


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#15171a}",
        ".small{font-size:12px}.label{font-size:14px;font-weight:700}.title{font-size:20px;font-weight:700}",
        ".node{stroke:#30343b;stroke-width:1.5}.factor{stroke:#9a6a00;stroke-width:1.4}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]


def rect(x, y, w, h, fill, stroke="#30343b", sw=1.5, rx=6):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1, y1, x2, y2, color="#555", sw=2, dash=False):
    dash_attr = ' stroke-dasharray="5 4"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{dash_attr}/>'
    )


def text(x, y, value, cls="small", anchor="start", color=None):
    style = f' style="fill:{color}"' if color else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{style}>{esc(value)}</text>'


def render_overview(rows, out_path, diagnostics=None):
    diagnostics = diagnostics or {}
    no_final = bool(diagnostics.get("noFinalBa"))
    summary = diagnostics.get("summary", {})
    width = 1280
    row_h = 34
    top = 92
    height = top + max(1, len(rows)) * row_h + 80
    max_rmse = max(
        [fnum(r.get("initial_overall_rmse")) for r in rows]
        + [fnum(r.get("final_overall_rmse")) for r in rows]
        + [1.0]
    )
    max_factors = max([inum(r.get("total_reprojection_factor_count")) for r in rows] + [1])
    chart_x = 360
    chart_w = 520
    factor_x = 950
    factor_w = 180
    parts = svg_header(width, height)
    title = "Stage6 Selection-State Backend Trace" if no_final else "Stage6 Frame-Expanded BA Factor Trace"
    parts.append(text(36, 38, title, "title"))
    parts.append(text(36, 64, f"Frames entering backend: {len(rows)}", "small"))
    if no_final:
        parts.append(text(36, 86, "No final BA: values show the backend state produced by selection / trial BA. Initial/final deltas are intentionally not used as convergence evidence.", "small", color="#9a3412"))
    parts.append(text(chart_x, 64, "selection-state RMSE" if no_final else "RMSE before / after final BA", "label"))
    parts.append(text(factor_x, 64, "factor count", "label"))
    if not no_final:
        parts.append(text(36, 86, "green = improved, red = worse; cam1 reprojection factors directly constrain T_1_0", "small"))

    for i, row in enumerate(rows):
        y = top + i * row_h
        pair = row.get("pair_index", "")
        boards = row.get("selected_board_ids", "")
        initial = fnum(row.get("initial_overall_rmse"))
        final = fnum(row.get("final_overall_rmse"))
        delta = fnum(row.get("overall_rmse_delta"))
        total_factors = inum(row.get("total_reprojection_factor_count"))
        cam1_factors = inum(row.get("t_1_0_reprojection_factor_count"))
        state_color = "#3867b7" if no_final else color_for_delta(delta)
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        parts.append(rect(24, y - 19, width - 48, row_h - 4, bg, "#e5e7eb", 1, 4))
        parts.append(text(40, y + 2, f"pair {pair}", "label"))
        parts.append(text(115, y + 2, f"boards {boards}", "small"))

        init_w = 0 if not math.isfinite(initial) else max(1, initial / max_rmse * chart_w)
        final_w = 0 if not math.isfinite(final) else max(1, final / max_rmse * chart_w)
        if no_final:
            parts.append(rect(chart_x, y - 8, final_w, 13, state_color, state_color, 1, 2))
            parts.append(text(chart_x + chart_w + 14, y + 2, f"{final:.3f} selection-state", "small", color=state_color))
        else:
            parts.append(rect(chart_x, y - 13, init_w, 8, "#d0d7de", "#d0d7de", 1, 2))
            parts.append(rect(chart_x, y - 2, final_w, 8, state_color, state_color, 1, 2))
            parts.append(text(chart_x + chart_w + 14, y + 2, f"{initial:.3f} -> {final:.3f} ({delta:+.3f})", "small", color=state_color))

        fw = max(1, total_factors / max_factors * factor_w)
        t_w = max(1, cam1_factors / max_factors * factor_w)
        parts.append(rect(factor_x, y - 13, fw, 19, "#e8f1ff", "#8aa6d8", 1, 3))
        parts.append(rect(factor_x, y - 13, t_w, 19, "#ffe8e8", "#d88a8a", 1, 3))
        parts.append(text(factor_x + factor_w + 14, y + 2, f"all {total_factors}, T {cam1_factors}", "small"))

    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def render_frame_graph(row, out_path, diagnostics=None):
    diagnostics = diagnostics or {}
    no_final = bool(diagnostics.get("noFinalBa"))
    boards = parse_board_factor_counts(row.get("board_factor_counts", ""))
    if not boards:
        selected = [b for b in row.get("selected_board_ids", "").split(";") if b]
        boards = [(b, 0, 0) for b in selected]
    board_count = max(1, len(boards))
    width = 1180
    height = max(560, 230 + board_count * 70)
    parts = svg_header(width, height)

    pair = row.get("pair_index", "")
    left = row.get("left_frame_label", "")
    right = row.get("right_frame_label", "")
    initial = fnum(row.get("initial_overall_rmse"))
    final = fnum(row.get("final_overall_rmse"))
    delta = fnum(row.get("overall_rmse_delta"))
    cam0_delta = fnum(row.get("cam0_rmse_delta"))
    cam1_delta = fnum(row.get("cam1_rmse_delta"))
    state_color = "#3867b7" if no_final else color_for_delta(delta)

    title = "Selection-state backend factor graph" if no_final else "Frame-expanded BA factor graph"
    parts.append(text(34, 38, f"{title}: pair {pair}", "title"))
    parts.append(text(34, 64, f"{left}  |  {right}", "small"))
    if no_final:
        parts.append(text(34, 88, f"selection-state overall RMSE {final:.4f}", "label", color=state_color))
        parts.append(text(34, 112, "No final BA: this graph shows the backend state after trial selection, not a final optimizer before/after trace.", "small", color="#9a3412"))
    else:
        parts.append(text(34, 88, f"overall RMSE {initial:.4f} -> {final:.4f} ({delta:+.4f})", "label", color=state_color))
        parts.append(text(34, 112, f"cam0 delta {cam0_delta:+.4f} px, cam1 delta {cam1_delta:+.4f} px", "small"))

    k0 = (80, 178, 120, 48)
    k1 = (80, 312, 120, 48)
    t = (80, 410, 150, 54)
    b = (456, 116, 150, 54)
    parts.append(rect(*k0, "#e8f1ff"))
    parts.append(text(k0[0] + k0[2] / 2, k0[1] + 29, "K0 fixed", "label", "middle"))
    parts.append(rect(*k1, "#e8f1ff"))
    parts.append(text(k1[0] + k1[2] / 2, k1[1] + 29, "K1 fixed", "label", "middle"))
    parts.append(rect(*t, "#ffe8e8", "#c62828", 2.5))
    parts.append(text(t[0] + t[2] / 2, t[1] + 24, "T_1_0", "label", "middle", "#9f1d1d"))
    parts.append(text(t[0] + t[2] / 2, t[1] + 43, f"{row.get('t_1_0_reprojection_factor_count','0')} cam1 factors", "small", "middle", "#9f1d1d"))
    parts.append(rect(*b, "#eaf7ea", "#2e7d32", 1.8))
    parts.append(text(b[0] + b[2] / 2, b[1] + 24, f"B_i / pair {pair}", "label", "middle"))
    parts.append(text(b[0] + b[2] / 2, b[1] + 43, f"{row.get('selected_board_count','0')} selected boards", "small", "middle"))

    parts.append(text(350, 204, "cam0 factors f0_i_j", "label"))
    parts.append(text(350, 338, "cam1 factors f1_i_j", "label", color="#9f1d1d"))
    parts.append(text(740, 112, "per-board micro factor groups", "label"))
    parts.append(text(740, 132, "label format: board_id cam0/cam1 factor count", "small"))

    max_count = max([c0 + c1 for _, c0, c1 in boards] + [1])
    for i, (board_id, c0, c1) in enumerate(boards):
        y = 170 + i * 70
        f0 = (365, y, 150, 36)
        f1 = (365, y + 38, 150, 36)
        board_node = (740, y + 16, 180, 42)
        sw0 = 1.2 + 5.0 * c0 / max_count
        sw1 = 1.2 + 5.0 * c1 / max_count
        parts.append(rect(*f0, "#fff5cc", "#b58900", 1.3, 18))
        parts.append(text(f0[0] + f0[2] / 2, f0[1] + 23, f"f0 board {board_id}", "small", "middle"))
        parts.append(rect(*f1, "#ffe6d6", "#c95f14", 1.3, 18))
        parts.append(text(f1[0] + f1[2] / 2, f1[1] + 23, f"f1 board {board_id}", "small", "middle"))
        parts.append(rect(*board_node, "#eef7ff", "#4b83c4", 1.4, 6))
        parts.append(text(board_node[0] + board_node[2] / 2, board_node[1] + 18, f"board {board_id}", "label", "middle"))
        parts.append(text(board_node[0] + board_node[2] / 2, board_node[1] + 36, f"{c0}/{c1} factors", "small", "middle"))

        parts.append(line(k0[0] + k0[2], k0[1] + k0[3] / 2, f0[0], f0[1] + 18, "#4b83c4", sw0))
        parts.append(line(b[0] + b[2] / 2, b[1] + b[3], f0[0] + f0[2] / 2, f0[1], "#2e7d32", 1.8, True))
        parts.append(line(f0[0] + f0[2], f0[1] + 18, board_node[0], board_node[1] + 14, "#8aa6d8", sw0))

        parts.append(line(k1[0] + k1[2], k1[1] + k1[3] / 2, f1[0], f1[1] + 18, "#c95f14", sw1))
        parts.append(line(t[0] + t[2], t[1] + t[3] / 2, f1[0], f1[1] + 18, "#c62828", sw1))
        parts.append(line(b[0] + b[2] / 2, b[1] + b[3], f1[0] + f1[2] / 2, f1[1], "#2e7d32", 1.8, True))
        parts.append(line(f1[0] + f1[2], f1[1] + 18, board_node[0], board_node[1] + 30, "#d88a8a", sw1))

    legend_y = height - 70
    parts.append(rect(34, legend_y - 22, width - 68, 48, "#fafafa", "#e5e7eb", 1, 6))
    parts.append(text(54, legend_y + 2, "Edge width is proportional to reprojection factor count. Red cam1 edges are the direct T_1_0 constraints.", "small"))
    state_label = "selection-state backend" if no_final else row.get("residual_state", "")
    parts.append(text(54, legend_y + 22, f"state: {state_label} | factor source: {row.get('factor_count_source','')}", "small", color=state_color))
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def render_index(rows, output_dir, diagnostics=None):
    diagnostics = diagnostics or {}
    no_final = bool(diagnostics.get("noFinalBa"))
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Stage6 BA Frame Factor Graphs</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#15171a}"
        "a{color:#0b5cad}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}"
        ".card{border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#fff}"
        ".small{color:#57606a;font-size:13px}</style></head><body>",
        "<h1>Stage6 Selection-State Backend Factor Graphs</h1>" if no_final else "<h1>Stage6 BA Frame Factor Graphs</h1>",
        "<p class='small'>No final BA: RMSE values are selection-state diagnostics; initial/final deltas are not optimizer progress.</p>" if no_final else "",
        "<p><a href='overview.svg'>Open overview.svg</a></p>",
        "<div class='grid'>",
    ]
    for row in rows:
        pair = row.get("pair_index", "")
        delta = fnum(row.get("overall_rmse_delta"))
        color = "#3867b7" if no_final else color_for_delta(delta)
        name = f"frame_graphs/frame_{inum(pair):06d}.svg"
        lines.append(
            f"<div class='card'><a href='{name}'>pair {esc(pair)}</a>"
            f"<div class='small'>boards {esc(row.get('selected_board_ids',''))}</div>"
            + (
                f"<div class='small' style='color:{color}'>selection-state RMSE "
                f"{fnum(row.get('final_overall_rmse')):.3f}</div></div>"
                if no_final else
                f"<div class='small' style='color:{color}'>RMSE "
                f"{fnum(row.get('initial_overall_rmse')):.3f} -> "
                f"{fnum(row.get('final_overall_rmse')):.3f} "
                f"({delta:+.3f})</div></div>"
            )
        )
    lines += ["</div></body></html>"]
    Path(output_dir, "index.html").write_text("\n".join(lines), encoding="utf-8")


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def render_viewer_html(rows, output_dir, trace_csv):
    jacobian = load_jacobian_diagnostics(trace_csv)
    records = [row_to_viewer_record(row, jacobian) for row in rows]
    diagnostics = load_run_diagnostics(trace_csv)
    payload = json.dumps(
        json_safe({"frames": records, "diagnostics": diagnostics}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    template = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage6 Frame-Level BA Factor Graph Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d7dde8;
      --blue: #2563eb;
      --blue-soft: #dbeafe;
      --green: #15803d;
      --green-soft: #dcfce7;
      --red: #b91c1c;
      --red-soft: #fee2e2;
      --orange: #c05621;
      --orange-soft: #ffedd5;
      --yellow: #fef3c7;
      --purple: #7c3aed;
      --purple-soft: #ede9fe;
      --teal: #0f766e;
      --teal-soft: #ccfbf1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .subtle { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .run-banner {
      margin: 10px 12px 0;
      padding: 10px 12px;
      border: 1px solid #fdba74;
      border-radius: 8px;
      background: #fff7ed;
      color: #9a3412;
      font-size: 13px;
      display: none;
      line-height: 1.5;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    label { font-size: 12px; color: #344054; display: inline-flex; align-items: center; gap: 6px; }
    input, select, button {
      font: inherit;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 6px 8px;
    }
    input[type="number"] { width: 76px; }
    button { cursor: pointer; color: #263244; }
    button:hover { border-color: #9aa7bb; background: #f8fafc; }
    button.primary { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }
    button.primary:hover { background: #1e40af; border-color: #1e40af; }
    #app {
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 330px;
      grid-template-rows: minmax(440px, 58vh) minmax(280px, 32vh);
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 70px);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .panel-title {
      height: 38px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    #graphPanel { min-width: 0; min-height: 0; }
    #graphScroll { height: calc(100% - 38px); overflow: auto; background: #fbfcfe; }
    #graphSvg { display: block; min-width: 1100px; }
    #detailPanel { min-height: 0; }
    #details { padding: 12px; overflow: auto; height: calc(100% - 38px); }
    .metric {
      display: grid;
      grid-template-columns: minmax(96px, 1fr) minmax(110px, auto);
      gap: 8px;
      border-bottom: 1px solid #eef2f6;
      padding: 7px 0;
      font-size: 13px;
      align-items: start;
    }
    .metric span:first-child { color: var(--muted); }
    .metric b { color: #172033; text-align: right; font-weight: 700; overflow-wrap: anywhere; }
    .node { cursor: pointer; }
    .node rect, .node circle { stroke: #354052; stroke-width: 1.2; }
    .node.selected rect, .node.selected circle { stroke: #1d4ed8; stroke-width: 3; }
    .node.expanded rect { stroke: #2563eb; stroke-width: 2.2; }
    .edge { stroke-linecap: round; opacity: 0.82; }
    .edge-muted { stroke: #b8c0cc; stroke-width: 1.6; stroke-dasharray: 4 4; }
    .factor rect { fill: var(--yellow); stroke: #b58900; stroke-width: 1.3; }
    .factor.cam1 rect { fill: var(--orange-soft); stroke: #c95f14; }
    .t-node rect { fill: var(--red-soft); stroke: var(--red); stroke-width: 2.6; }
    .k-node rect { fill: var(--blue-soft); stroke: var(--blue); }
    .b-node rect { fill: var(--green-soft); stroke: var(--green); }
    .tiny { font-size: 11px; fill: #667085; }
    .label { font-size: 12px; font-weight: 700; fill: #172033; }
    .chipText { font-size: 10px; font-weight: 700; fill: #172033; }
    .chartGrid {
      display: grid;
      grid-template-columns: 1.2fr 1.2fr 1fr;
      height: calc(100% - 38px);
    }
    .chartBox { border-right: 1px solid var(--line); min-width: 0; overflow: hidden; }
    .chartBox:last-child { border-right: 0; }
    .chartBox h3 { margin: 10px 12px 0; font-size: 13px; color: #172033; }
    .chartBox svg { width: 100%; height: calc(100% - 28px); display: block; }
    #tooltip {
      position: fixed;
      pointer-events: none;
      background: rgba(17, 24, 39, 0.94);
      color: white;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.4;
      max-width: 360px;
      display: none;
      z-index: 10;
      box-shadow: 0 8px 22px rgba(0,0,0,.18);
    }
    .empty {
      padding: 18px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 58vh 320px 420px; height: auto; }
      #chartsPanel { grid-column: 1; }
      .chartGrid { grid-template-columns: 1fr; overflow: auto; }
      .chartBox { min-height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="viewerTitle">Frame-Level BA Factor Graph Viewer</h1>
      <div id="viewerSubtitle" class="subtle">Frame_i defaults to collapsed. Click to expand into K0 - f0_i - B_i - f1_i - K1, with every f1_i connected to T_1_0.</div>
    </div>
    <div class="toolbar">
      <label>Sort
        <select id="sortSelect">
          <option value="pair">frame id</option>
          <option value="rmse_desc">final RMSE high to low</option>
          <option value="rmse_asc">final RMSE low to high</option>
          <option value="reduction_desc">residual reduction high to low</option>
          <option value="board_desc">board count high to low</option>
        </select>
      </label>
      <label>Search <input id="searchInput" type="search" placeholder="frame / timestamp"></label>
      <label><input id="highOnly" type="checkbox"> high residual only</label>
      <label>metric
        <select id="residualMetric">
          <option value="final">final RMSE</option>
          <option value="initial">initial RMSE</option>
          <option value="cam1_final">cam1 final RMSE</option>
          <option value="cam1_initial">cam1 initial RMSE</option>
          <option value="reduction_below">residual reduction below</option>
        </select>
      </label>
      <label>threshold <input id="thresholdInput" type="number" step="0.1" value="2.0"></label>
      <button id="collapseAll">Collapse</button>
      <button id="fitView">Fit</button>
      <button id="zoomIn">Zoom +</button>
      <button id="zoomOut">Zoom -</button>
      <button id="exportSvg">Export SVG</button>
      <button id="exportPng">Export PNG</button>
      <button id="exportHtml" class="primary">Export HTML</button>
    </div>
  </header>
  <div id="runBanner" class="run-banner"></div>
  <main id="app">
    <section id="graphPanel" class="panel">
      <div class="panel-title">
        <span id="graphPanelTitle">BA Factor Graph</span>
        <span id="visibleCount" class="subtle"></span>
      </div>
      <div id="graphScroll"><svg id="graphSvg"></svg></div>
    </section>
    <aside id="detailPanel" class="panel">
      <div class="panel-title">
        <span>Selected Frame</span>
        <span class="subtle">T_1_0 always highlighted</span>
      </div>
      <div id="details"></div>
    </aside>
    <section id="chartsPanel" class="panel" style="grid-column: 1 / span 2;">
      <div class="panel-title">
        <span>Residual Statistics</span>
        <span id="runSummary" class="subtle"></span>
      </div>
      <div class="chartGrid">
        <div class="chartBox"><h3 id="rmseChartTitle">Per-frame final RMSE</h3><svg id="rmseChart"></svg></div>
        <div class="chartBox"><h3 id="secondChartTitle">Residual reduction</h3><svg id="reductionChart"></svg></div>
        <div class="chartBox"><h3 id="heatmapTitle">Residual heatmap / convergence</h3><svg id="heatmapChart"></svg></div>
      </div>
    </section>
  </main>
  <div id="tooltip"></div>
  <script id="viewer-data" type="application/json">__DATA__</script>
  <script>
    const payload = JSON.parse(document.getElementById('viewer-data').textContent);
    const allFrames = payload.frames || [];
    const diagnostics = payload.diagnostics || {};
    const runSummaryData = diagnostics.summary || {};
    const noFinalBa = Boolean(diagnostics.noFinalBa || runSummaryData.skipFinalGlobalBa);
    const expanded = new Set();
    const boardExpanded = new Set();
    let selectedId = allFrames.length ? allFrames[0].pairIndex : null;
    let zoomScale = 1.0;
    let lastGraphSize = {width: 1280, height: 720};
    const graphSvg = document.getElementById('graphSvg');
    const graphScroll = document.getElementById('graphScroll');
    const tooltip = document.getElementById('tooltip');
    const sortSelect = document.getElementById('sortSelect');
    const searchInput = document.getElementById('searchInput');
    const highOnly = document.getElementById('highOnly');
    const residualMetric = document.getElementById('residualMetric');
    const thresholdInput = document.getElementById('thresholdInput');
    if (noFinalBa) {
      document.getElementById('viewerTitle').textContent = 'Stage6 Selection-State Backend Factor Graph';
      document.getElementById('viewerSubtitle').textContent =
        '当前 baseline 没有额外 final BA；这里展示 selection / trial BA 结束后的 backend state、结构约束和残差健康度。';
      document.getElementById('graphPanelTitle').textContent = 'Backend 输入结构图';
      document.getElementById('rmseChartTitle').textContent = 'Per-frame selection-state RMSE';
      document.getElementById('secondChartTitle').textContent = '左右相机 RMSE 不平衡';
      document.getElementById('heatmapTitle').textContent = '残差 / board / 相机分布';
      const banner = document.getElementById('runBanner');
      banner.style.display = 'block';
      banner.textContent =
        `No final BA：final_state=${runSummaryData.finalStateLabel || 'selection-state backend'}，` +
        `selection residual=${runSummaryData.selectionBaResidualMode || 'n/a'}。` +
        `这里不再看 initial→final 下降量，而是看 backend 选中了哪些 frame/board、残差健康度、cam0/cam1 平衡以及 T_1_0 约束强度。` +
        (runSummaryData.jacobianDiagnosticsAvailable
          ? ` 已加载 Jacobian diagnostics（${runSummaryData.jacobianResidualMode || 'residual'}）：显示局部 JᵀJ / Jᵀr proxy。`
          : ` 当前输出未包含 Jacobian diagnostics，T 指标会回退到 factor-count proxy。`);
      const reductionOption = [...sortSelect.options].find(o => o.value === 'reduction_desc');
      if (reductionOption) reductionOption.textContent = 'cam imbalance high to low';
      [...residualMetric.options].forEach(option => {
        if (option.value === 'final') option.textContent = 'selection-state RMSE';
        if (option.value === 'initial') option.textContent = 'same backend-state RMSE';
        if (option.value === 'cam1_final') option.textContent = 'cam1 selection-state RMSE';
        if (option.value === 'cam1_initial') option.textContent = 'cam1 同一状态';
        if (option.value === 'reduction_below') option.textContent = '左右不平衡低于';
      });
    }

    function finite(v) { return typeof v === 'number' && Number.isFinite(v); }
    function fmt(v, digits = 3) { return finite(v) ? v.toFixed(digits) : 'n/a'; }
    function fmtSci(v, digits = 2) { return finite(v) ? v.toExponential(digits) : 'n/a'; }
    function reduction(f) { return finite(f.initialRmse) && finite(f.finalRmse) ? f.initialRmse - f.finalRmse : 0; }
    function camImbalance(f) {
      return finite(f.finalCam0Rmse) && finite(f.finalCam1Rmse) ? Math.abs(f.finalCam0Rmse - f.finalCam1Rmse) : 0;
    }
    function hexToRgb(hex) {
      const v = hex.replace('#', '');
      return {
        r: parseInt(v.slice(0, 2), 16),
        g: parseInt(v.slice(2, 4), 16),
        b: parseInt(v.slice(4, 6), 16)
      };
    }
    function mixColor(a, b, t) {
      const ca = hexToRgb(a), cb = hexToRgb(b);
      const u = Math.max(0, Math.min(1, t));
      return `rgb(${Math.round(ca.r + (cb.r - ca.r) * u)},${Math.round(ca.g + (cb.g - ca.g) * u)},${Math.round(ca.b + (cb.b - ca.b) * u)})`;
    }
    function colorForRmse(v) {
      const values = allFrames.map(f => f.finalRmse).filter(finite);
      const min = Math.min(...values, 0);
      const max = Math.max(...values, 1);
      const t = Math.max(0, Math.min(1, (v - min) / Math.max(1e-9, max - min)));
      if (t < 0.55) return mixColor('#0f766e', '#d97706', t / 0.55);
      return mixColor('#d97706', '#b91c1c', (t - 0.55) / 0.45);
    }
    function softFillForRmse(v) {
      const values = allFrames.map(f => f.finalRmse).filter(finite);
      const min = Math.min(...values, 0);
      const max = Math.max(...values, 1);
      const t = Math.max(0, Math.min(1, (v - min) / Math.max(1e-9, max - min)));
      if (t < 0.55) return '#ecfdf5';
      if (t < 0.8) return '#fff7ed';
      return '#fef2f2';
    }
    function imbalanceColor(f) {
      const v = camImbalance(f);
      if (v > 0.8) return '#b91c1c';
      if (v > 0.45) return '#d97706';
      return '#0f766e';
    }
    function deltaColor(delta) {
      if (!finite(delta)) return '#8a8f98';
      if (delta < -1e-6) return '#2e7d32';
      if (delta > 1e-6) return '#c62828';
      return '#6b7280';
    }
    function edgeWidth(count) {
      const max = Math.max(...allFrames.map(f => f.totalFactors || 0), 1);
      return 1.2 + 6.0 * Math.max(0, count) / max;
    }
    function tEdgeStyle(frame) {
      const hasExpandedFrame = expanded.size > 0;
      const focused = expanded.has(frame.pairIndex);
      if (!hasExpandedFrame || focused) {
        return {color: '#b91c1c', width: edgeWidth(frame.tFactors), dash: true, opacity: 0.9};
      }
      return {color: '#cbd5e1', width: 1.1, dash: true, opacity: 0.32};
    }
    function residualMetricValue(f) {
      const mode = residualMetric.value;
      if (mode === 'initial') return f.initialRmse || 0;
      if (mode === 'cam1_final') return f.finalCam1Rmse || 0;
      if (mode === 'cam1_initial') return f.initialCam1Rmse || 0;
      if (mode === 'reduction_below') return noFinalBa ? camImbalance(f) : reduction(f);
      return f.finalRmse || 0;
    }
    function passesHighResidualFilter(f, threshold) {
      const v = residualMetricValue(f);
      if (residualMetric.value === 'reduction_below') return v <= threshold;
      return v >= threshold;
    }
    function matchesSearch(f, needle) {
      if (!needle) return true;
      const haystack = [
        f.frameId, String(f.pairIndex), f.timestamp, f.leftFrame, f.rightFrame,
        ...(f.selectedBoardIds || [])
      ].join(' ').toLowerCase();
      return haystack.includes(needle);
    }
    function getVisibleFrames() {
      const threshold = Number(thresholdInput.value || 0);
      const needle = (searchInput.value || '').trim().toLowerCase();
      let frames = allFrames.filter(f => matchesSearch(f, needle));
      frames = frames.filter(f => !highOnly.checked || passesHighResidualFilter(f, threshold));
      const mode = sortSelect.value;
      frames = frames.slice();
      if (mode === 'rmse_desc') frames.sort((a, b) => (b.finalRmse || 0) - (a.finalRmse || 0));
      else if (mode === 'rmse_asc') frames.sort((a, b) => (a.finalRmse || 0) - (b.finalRmse || 0));
      else if (mode === 'reduction_desc') frames.sort((a, b) => (noFinalBa ? camImbalance(b) - camImbalance(a) : reduction(b) - reduction(a)));
      else if (mode === 'board_desc') frames.sort((a, b) => (b.boardCount || 0) - (a.boardCount || 0));
      else frames.sort((a, b) => a.pairIndex - b.pairIndex);
      return frames;
    }
    function svgEl(tag, attrs = {}, children = []) {
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [k, v] of Object.entries(attrs)) {
        if (v !== undefined && v !== null) el.setAttribute(k, String(v));
      }
      for (const child of children) {
        if (typeof child === 'string') el.appendChild(document.createTextNode(child));
        else el.appendChild(child);
      }
      return el;
    }
    function addText(parent, x, y, value, cls = 'tiny', anchor = 'start', color = null) {
      const attrs = {x, y, class: cls, 'text-anchor': anchor};
      if (color) attrs.fill = color;
      parent.appendChild(svgEl('text', attrs, [String(value)]));
    }
    function addRect(parent, x, y, w, h, fill, stroke = '#30343b', rx = 6, cls = '') {
      parent.appendChild(svgEl('rect', {x, y, width: w, height: h, rx, fill, stroke, class: cls}));
    }
    function addPill(parent, x, y, w, textValue, fill, stroke = '#d7dde8', color = '#172033') {
      addRect(parent, x, y, w, 18, fill, stroke, 9);
      addText(parent, x + w / 2, y + 12.5, textValue, 'chipText', 'middle', color);
    }
    function addLine(parent, x1, y1, x2, y2, stroke, width, dash = false) {
      parent.appendChild(svgEl('line', {
        x1, y1, x2, y2, stroke, 'stroke-width': width,
        'stroke-linecap': 'round',
        'stroke-dasharray': dash ? '5 4' : null,
        class: 'edge'
      }));
    }
    function addStyledLine(parent, x1, y1, x2, y2, style) {
      const el = svgEl('line', {
        x1, y1, x2, y2, stroke: style.color, 'stroke-width': style.width,
        'stroke-linecap': 'round',
        'stroke-dasharray': style.dash ? '5 4' : null,
        opacity: style.opacity == null ? 0.9 : style.opacity,
        class: 'edge'
      });
      parent.appendChild(el);
    }
    function showTip(evt, frame) {
      const jac = frame.jacobian || {};
      const tJac = jac.tStereo || {};
      const frameJac = jac.framePose || {};
      tooltip.innerHTML =
        `<b>${frame.frameId}</b><br>` +
        `timestamp: ${frame.timestamp || 'n/a'}<br>` +
        `boards: ${frame.boardCount}, corners/factors: ${frame.totalFactors}<br>` +
        (noFinalBa
          ? `selection-state RMSE: ${fmt(frame.finalRmse)}<br>` +
            `cam0/cam1: ${fmt(frame.finalCam0Rmse)} / ${fmt(frame.finalCam1Rmse)}<br>` +
            `cam imbalance: ${fmt(camImbalance(frame))}<br>`
          : `RMSE: ${fmt(frame.initialRmse)} -> ${fmt(frame.finalRmse)} (${fmt(frame.rmseDelta)})<br>` +
            `cam0: ${fmt(frame.initialCam0Rmse)} -> ${fmt(frame.finalCam0Rmse)}<br>` +
            `cam1: ${fmt(frame.initialCam1Rmse)} -> ${fmt(frame.finalCam1Rmse)}<br>`) +
        (jac.available
          ? `T_1_0 JtJ trace: ${fmtSci(tJac.hessianTrace)} | |Jtr| ${fmtSci(tJac.gradientNorm)} | rank ${tJac.rankProxy || 0}<br>` +
            `Frame pose JtJ trace: ${fmtSci(frameJac.hessianTrace)} | |Jtr| ${fmtSci(frameJac.gradientNorm)}<br>`
          : `T_1_0 sensitivity fallback proxy: ${fmt(frame.tSensitivityProxy)}<br>`) +
        (noFinalBa
          ? `selection update proxy: ${fmt(frame.framePoseInitToFinalRotationDeg)} deg, ${fmt(frame.framePoseInitToFinalTranslationM, 5)} m`
          : `frame pose init→final: ${fmt(frame.framePoseInitToFinalRotationDeg)} deg, ${fmt(frame.framePoseInitToFinalTranslationM, 5)} m`);
      tooltip.style.display = 'block';
      tooltip.style.left = `${evt.clientX + 14}px`;
      tooltip.style.top = `${evt.clientY + 14}px`;
    }
    function hideTip() { tooltip.style.display = 'none'; }
    function renderGraph() {
      const frames = getVisibleFrames();
      document.getElementById('visibleCount').textContent = `${frames.length} / ${allFrames.length} frames`;
      graphSvg.innerHTML = '';
      const width = 1280;
      let y = 132;
      const rowGap = 16;
      const tNode = {x: 42, y: 44, w: 150, h: 62};
      const content = svgEl('g');
      graphSvg.appendChild(content);

      const tGroup = svgEl('g', {class: 't-node'});
      addRect(tGroup, tNode.x, tNode.y, tNode.w, tNode.h, '#fee2e2', '#b91c1c', 8);
      addText(tGroup, tNode.x + tNode.w / 2, tNode.y + 25, 'T_1_0', 'label', 'middle', '#7f1d1d');
      const tUpdate = allFrames.length ? allFrames[0] : {};
      addText(tGroup, tNode.x + tNode.w / 2, tNode.y + 43, '双目外参', 'tiny', 'middle', '#7f1d1d');
      addText(tGroup, tNode.x + tNode.w / 2, tNode.y + 60,
        `Δ ${fmt(tUpdate.tInitToFinalRotationDeg, 2)}deg/${fmt(tUpdate.tInitToFinalTranslationM, 4)}m`,
        'tiny', 'middle', '#7f1d1d');
      content.appendChild(tGroup);

      const legend = svgEl('g');
      addRect(legend, 250, 22, 890, 88, '#ffffff', '#d7dde8', 8);
      addText(legend, 270, 47, '图例 / How to read', 'label');
      addRect(legend, 270, 60, 34, 18, '#dbeafe', '#2563eb', 4);
      addText(legend, 312, 74, 'K0/K1 固定内参', 'tiny');
      addRect(legend, 445, 60, 34, 18, '#fef3c7', '#b58900', 9);
      addText(legend, 487, 74, 'cam0 因子 f0', 'tiny');
      addRect(legend, 600, 60, 34, 18, '#ffedd5', '#c05621', 9);
      addText(legend, 642, 74, 'cam1 因子 f1', 'tiny');
      addRect(legend, 760, 60, 34, 18, '#fee2e2', '#b91c1c', 4);
      addText(legend, 802, 74, 'T_1_0 双目外参', 'tiny');
      addLine(legend, 270, 94, 330, 94, '#b91c1c', 5);
      addText(legend, 342, 96,
        noFinalBa
          ? '边越粗 = 该 frame 对变量约束越多；左侧色条 = selection-state RMSE 从绿到红；Δ = selection/trial BA 后的状态变化'
          : 'edge width = factor count; frame color = final RMSE; Δ pose shows optimized variable update',
        'tiny');
      content.appendChild(legend);

      const laneX = 250;
      for (const frame of frames) {
        const isExpanded = expanded.has(frame.pairIndex);
        const isBoardExpanded = boardExpanded.has(frame.pairIndex);
        const hasJac = Boolean(frame.jacobian && frame.jacobian.available);
        const boardDetailH = isExpanded && isBoardExpanded
          ? Math.max(40, frame.boardFactors.length * 46 + (hasJac ? 58 : 30))
          : 0;
        const rowH = isExpanded ? (hasJac ? 222 : 204) + boardDetailH : 72;
        const group = svgEl('g', {
          id: `frame-node-${frame.pairIndex}`,
          class: `node ${isExpanded ? 'expanded' : ''} ${selectedId === frame.pairIndex ? 'selected' : ''}`
        });
        group.addEventListener('click', () => {
          selectedId = frame.pairIndex;
          if (expanded.has(frame.pairIndex)) expanded.delete(frame.pairIndex);
          else expanded.add(frame.pairIndex);
          renderAll();
        });
        group.addEventListener('mousemove', evt => showTip(evt, frame));
        group.addEventListener('mouseleave', hideTip);
        addRect(group, laneX, y, 260, 58, softFillForRmse(frame.finalRmse), '#cfd8e3', 8);
        addRect(group, laneX, y, 9, 58, colorForRmse(frame.finalRmse), colorForRmse(frame.finalRmse), 8);
        addText(group, laneX + 20, y + 21, `${frame.frameId}`, 'label', 'start', '#172033');
        addText(group, laneX + 20, y + 41,
          `${noFinalBa ? 'selection' : 'final'} RMSE ${fmt(frame.finalRmse)} | T约束 ${frame.tFactors}`,
          'tiny', 'start', '#344054');
        addPill(group, laneX + 126, y + 10, 52, `${frame.boardCount}板`, '#dcfce7', '#86efac', '#14532d');
        addPill(group, laneX + 184, y + 10, 66, `camΔ ${fmt(camImbalance(frame), 2)}`, '#fff7ed', imbalanceColor(frame), imbalanceColor(frame));
        addStyledLine(content, tNode.x + tNode.w, tNode.y + tNode.h / 2, laneX, y + 29, tEdgeStyle(frame));
        content.appendChild(group);

        if (isExpanded) {
          const ey = y + 86;
          const k0 = {x: laneX, y: ey, w: 90, h: 38};
          const f0 = {x: laneX + 130, y: ey, w: 90, h: 38};
          const b = {x: laneX + 260, y: ey, w: 98, h: 38};
          const f1 = {x: laneX + 398, y: ey, w: 90, h: 38};
          const k1 = {x: laneX + 528, y: ey, w: 90, h: 38};
          const eg = svgEl('g');
          addRect(eg, k0.x, k0.y, k0.w, k0.h, '#dbeafe', '#2563eb');
          addText(eg, k0.x + k0.w / 2, k0.y + 24, 'K0', 'label', 'middle');
          addText(eg, k0.x + k0.w / 2, k0.y + 36, 'fixed', 'tiny', 'middle');
          addRect(eg, f0.x, f0.y, f0.w, f0.h, '#fef3c7', '#b58900', 18);
          addText(eg, f0.x + f0.w / 2, f0.y + 24, 'cam0因子', 'label', 'middle');
          const bGroup = svgEl('g', {class: 'b-node node'});
          bGroup.addEventListener('click', evt => {
            evt.stopPropagation();
            selectedId = frame.pairIndex;
            if (boardExpanded.has(frame.pairIndex)) boardExpanded.delete(frame.pairIndex);
            else boardExpanded.add(frame.pairIndex);
            expanded.add(frame.pairIndex);
            renderAll();
          });
          addRect(bGroup, b.x, b.y, b.w, b.h, '#dcfce7', '#15803d');
          addText(bGroup, b.x + b.w / 2, b.y + 18, 'B_i', 'label', 'middle');
          addText(bGroup, b.x + b.w / 2, b.y + 32, `${frame.boardCount} boards`, 'tiny', 'middle');
          eg.appendChild(bGroup);
          addRect(eg, f1.x, f1.y, f1.w, f1.h, '#ffedd5', '#c05621', 18);
          addText(eg, f1.x + f1.w / 2, f1.y + 24, 'cam1因子', 'label', 'middle');
          addRect(eg, k1.x, k1.y, k1.w, k1.h, '#dbeafe', '#2563eb');
          addText(eg, k1.x + k1.w / 2, k1.y + 24, 'K1', 'label', 'middle');
          addText(eg, k1.x + k1.w / 2, k1.y + 36, 'fixed', 'tiny', 'middle');
          addLine(eg, k0.x + k0.w, k0.y + 19, f0.x, f0.y + 19, '#2563eb', edgeWidth(frame.cam0Factors));
          addLine(eg, f0.x + f0.w, f0.y + 19, b.x, b.y + 19, '#15803d', edgeWidth(frame.cam0Factors));
          addLine(eg, b.x + b.w, b.y + 19, f1.x, f1.y + 19, '#15803d', edgeWidth(frame.cam1Factors));
          addLine(eg, f1.x + f1.w, f1.y + 19, k1.x, k1.y + 19, '#c05621', edgeWidth(frame.cam1Factors));
          addLine(eg, tNode.x + tNode.w, tNode.y + tNode.h, f1.x + f1.w / 2, f1.y, '#b91c1c', edgeWidth(frame.tFactors));
          addText(eg, laneX, ey + 64, `因子数：cam0 ${frame.cam0Factors}，cam1/T ${frame.cam1Factors}，total ${frame.totalFactors}`, 'tiny');
          addText(eg, laneX, ey + 78,
            (noFinalBa ? 'selection-state vars: ' : 'optimized vars: ') +
            `T_1_0 Δ ${fmt(frame.tInitToFinalRotationDeg, 3)}deg/${fmt(frame.tInitToFinalTranslationM, 5)}m; ` +
            `T_cam0_world(frame) Δ ${fmt(frame.framePoseInitToFinalRotationDeg, 3)}deg/${fmt(frame.framePoseInitToFinalTranslationM, 5)}m`,
            'tiny');
          const jac = frame.jacobian || {};
          const boardLabels = frame.boardFactors.length
            ? frame.boardFactors.map(bf => {
                const tj = ((bf.jacobian || {}).tStereo || {});
                return jac.available
                  ? `${bf.boardId}:${bf.cam0}/${bf.cam1},T-JtJ=${fmtSci(tj.hessianTrace, 1)}`
                  : `${bf.boardId}:${bf.cam0}/${bf.cam1},T=${fmt(bf.tSensitivityProxy, 2)}`;
              }).join('  ')
            : frame.selectedBoardIds.join(' ');
          addText(eg, laneX, ey + 94,
            jac.available
              ? `点击 B_i 展开 board 级别信息；board: cam0/cam1,T-JtJ = ${boardLabels}`
              : `点击 B_i 展开 board 级别信息；board: cam0/cam1,T fallback proxy = ${boardLabels}`,
            'tiny');
          if (jac.available) {
            const tj = jac.tStereo || {};
            const fj = jac.framePose || {};
            addText(eg, laneX, ey + 110,
              `Jacobian：T_1_0 trace(JᵀJ) ${fmtSci(tj.hessianTrace)} | |Jᵀr| ${fmtSci(tj.gradientNorm)} | rank ${tj.rankProxy || 0}; ` +
              `frame pose trace ${fmtSci(fj.hessianTrace)} | |Jᵀr| ${fmtSci(fj.gradientNorm)} | rank ${fj.rankProxy || 0}`,
              'tiny', 'start', '#475569');
          }
          if (isBoardExpanded) {
            const by = ey + (jac.available ? 138 : 122);
            addText(eg, laneX, by,
              noFinalBa
                ? 'board 级别：selection-state 残差健康度 + 局部 Jacobian block diagnostics'
                : 'board-level lazy expansion: residual before/after and local Jacobian block diagnostics',
              'label');
            frame.boardFactors.forEach((bf, idx) => {
              const yy = by + 20 + idx * 46;
              const x0 = laneX + 22;
              const x1 = laneX + 138;
              const x2 = laneX + 250;
              const x3 = laneX + 382;
              const x4 = laneX + 548;
              addRect(eg, x0, yy, 88, 22, '#dbeafe', '#2563eb', 4);
              addText(eg, x0 + 44, yy + 15, `board ${bf.boardId}`, 'tiny', 'middle');
              addRect(eg, x1, yy, 82, 22, '#fef3c7', '#b58900', 11);
              addText(eg, x1 + 41, yy + 15, `f0 ${bf.cam0}`, 'tiny', 'middle');
              addRect(eg, x2, yy, 82, 22, '#ffedd5', '#c05621', 11);
              addText(eg, x2 + 41, yy + 15, `f1 ${bf.cam1}`, 'tiny', 'middle');
              addRect(eg, x3, yy, 130, 22, softFillForRmse(bf.finalRmse), colorForRmse(bf.finalRmse), 4);
              addText(eg, x3 + 65, yy + 15,
                noFinalBa ? `rmse ${fmt(bf.finalRmse, 2)}` : `rmse ${fmt(bf.initialRmse, 2)}→${fmt(bf.finalRmse, 2)}`,
                'tiny', 'middle', '#172033');
              addRect(eg, x4, yy, 130, 22, '#fee2e2', '#b91c1c', 4);
              const bj = (bf.jacobian || {}).tStereo || {};
              addText(eg, x4 + 65, yy + 15,
                jac.available ? `T JtJ ${fmtSci(bj.hessianTrace, 1)}` : `T proxy ${fmt(bf.tSensitivityProxy, 2)}`,
                'tiny', 'middle', '#7f1d1d');
              const bpj = (bf.jacobian || {}).boardPose || {};
              addText(eg, x3, yy + 37,
                (noFinalBa
                  ? `cam0 ${fmt(bf.finalCam0Rmse, 2)} | cam1 ${fmt(bf.finalCam1Rmse, 2)} | `
                  : `cam0 ${fmt(bf.initialCam0Rmse, 2)}→${fmt(bf.finalCam0Rmse, 2)} | ` +
                    `cam1 ${fmt(bf.initialCam1Rmse, 2)}→${fmt(bf.finalCam1Rmse, 2)} | `) +
                (jac.available
                  ? `board-pose JtJ ${fmtSci(bpj.hessianTrace, 1)} | |Jtr| ${fmtSci(bpj.gradientNorm, 1)} | rank ${bpj.rankProxy || 0}`
                  : `T_world_board Δ ${fmt(bf.boardPoseInitToFinalRotationDeg, 2)}deg/${fmt(bf.boardPoseInitToFinalTranslationM, 5)}m`),
                'tiny');
              addLine(eg, x0 + 88, yy + 11, x1, yy + 11, '#2563eb', edgeWidth(bf.cam0));
              addLine(eg, x1 + 82, yy + 11, x2, yy + 11, '#15803d', 1.6);
              addLine(eg, x2 + 82, yy + 11, x3, yy + 11, '#b91c1c', edgeWidth(bf.cam1));
            });
          }
          content.appendChild(eg);
        }
        y += rowH + rowGap;
      }
      const height = Math.max(420, y + 40);
      lastGraphSize = {width, height};
      graphSvg.setAttribute('width', width * zoomScale);
      graphSvg.setAttribute('height', height * zoomScale);
      graphSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    function selectedFrame() {
      return allFrames.find(f => f.pairIndex === selectedId) || allFrames[0] || null;
    }
    function renderDetails() {
      const f = selectedFrame();
      const box = document.getElementById('details');
      if (!f) { box.innerHTML = '<div class="empty">No frame selected.</div>'; return; }
      const jac = f.jacobian || {};
      const tJac = jac.tStereo || {};
      const frameJac = jac.framePose || {};
      const rows = [
        ['frame', f.frameId],
        ['timestamp', f.timestamp || 'n/a'],
        ['left', f.leftFrame],
        ['right', f.rightFrame],
        [noFinalBa ? 'selection-state overall RMSE' : 'overall RMSE',
          noFinalBa ? fmt(f.finalRmse) : `${fmt(f.initialRmse)} -> ${fmt(f.finalRmse)} (${fmt(f.rmseDelta)})`],
        [noFinalBa ? 'selection-state cam0 RMSE' : 'cam0 RMSE',
          noFinalBa ? fmt(f.finalCam0Rmse) : `${fmt(f.initialCam0Rmse)} -> ${fmt(f.finalCam0Rmse)} (${fmt(f.cam0Delta)})`],
        [noFinalBa ? 'selection-state cam1 RMSE' : 'cam1 RMSE',
          noFinalBa ? fmt(f.finalCam1Rmse) : `${fmt(f.initialCam1Rmse)} -> ${fmt(f.finalCam1Rmse)} (${fmt(f.cam1Delta)})`],
        [noFinalBa ? '左右相机不平衡' : 'residual reduction', noFinalBa ? fmt(camImbalance(f)) : fmt(reduction(f))],
        [jac.available ? 'T_1_0 trace(JᵀJ)' : 'T_1_0 fallback proxy',
          jac.available ? fmtSci(tJac.hessianTrace) : fmt(f.tSensitivityProxy)],
        [jac.available ? 'T_1_0 |Jᵀr| / rank' : 'Jacobian diagnostics',
          jac.available ? `${fmtSci(tJac.gradientNorm)} / ${tJac.rankProxy || 0}` : 'not available'],
        [jac.available ? 'Frame pose trace(JᵀJ)' : 'Frame pose JᵀJ',
          jac.available ? fmtSci(frameJac.hessianTrace) : 'not available'],
        [jac.available ? 'Frame pose |Jᵀr| / rank' : 'Frame pose |Jᵀr|',
          jac.available ? `${fmtSci(frameJac.gradientNorm)} / ${frameJac.rankProxy || 0}` : 'not available'],
        ['K0 / K1', 'fixed intrinsics in Stage6'],
        [noFinalBa ? 'T_1_0 selection 更新' : 'T_1_0 init -> final', `${fmt(f.tInitToFinalRotationDeg, 4)} deg / ${fmt(f.tInitToFinalTranslationM, 6)} m`],
        [noFinalBa ? 'T_1_0 final BA 更新' : 'T_1_0 final BA only', noFinalBa ? 'not run' : `${fmt(f.tFinalBaRotationDeg, 4)} deg / ${fmt(f.tFinalBaTranslationM, 6)} m`],
        [noFinalBa ? 'T_cam0_world selection 更新' : 'T_cam0_world(frame) init -> final', `${fmt(f.framePoseInitToFinalRotationDeg, 4)} deg / ${fmt(f.framePoseInitToFinalTranslationM, 6)} m`],
        [noFinalBa ? 'T_cam0_world final BA 更新' : 'T_cam0_world(frame) final BA only', noFinalBa ? 'not run' : `${fmt(f.framePoseFinalBaRotationDeg, 4)} deg / ${fmt(f.framePoseFinalBaTranslationM, 6)} m`],
        ['优化变量', f.optimizedVariableSummary || 'n/a'],
        ['board 数', f.boardCount],
        ['selected boards', f.selectedBoardIds.join('; ') || 'n/a'],
        ['角点/因子数', f.totalFactors],
        ['cam0 因子', f.cam0Factors],
        ['cam1 / T_1_0 因子', f.tFactors],
        ['outer/internal 因子', `${f.outerFactors} / ${f.internalFactors}`],
        ['factor source', f.factorSource],
      ];
      const boardRows = (f.boardFactors || []).map(bf => (
        (() => {
          const bj = bf.jacobian || {};
          const tj = bj.tStereo || {};
          const bpj = bj.boardPose || {};
          const jacText = jac.available
            ? `T-JtJ ${fmtSci(tj.hessianTrace, 1)} | board-JtJ ${fmtSci(bpj.hessianTrace, 1)} | board |Jtr| ${fmtSci(bpj.gradientNorm, 1)}`
            : `T ${fmt(bf.tSensitivityProxy)}`;
          return (
        `<div class="metric"><span>board ${bf.boardId}</span>` +
        (noFinalBa
          ? `<b>rmse ${fmt(bf.finalRmse)} | cam0 ${fmt(bf.finalCam0Rmse)} | cam1 ${fmt(bf.finalCam1Rmse)} | `
          : `<b>rmse ${fmt(bf.initialRmse)}→${fmt(bf.finalRmse)} | ` +
            `cam0 ${fmt(bf.initialCam0Rmse)}→${fmt(bf.finalCam0Rmse)} | ` +
            `cam1 ${fmt(bf.initialCam1Rmse)}→${fmt(bf.finalCam1Rmse)} | `) +
        `${jacText} | ` +
        `board Δ ${fmt(bf.boardPoseInitToFinalRotationDeg, 3)}deg/${fmt(bf.boardPoseInitToFinalTranslationM, 5)}m</b></div>`
          );
        })()
      )).join('');
      box.innerHTML = rows.map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('') +
        `<div class="metric"><span>board 级别指标</span><b>${(f.boardFactors || []).length} boards</b></div>` +
        boardRows;
    }
    function renderBarChart(svgId, frames, valueFn, colorFn, labelFn) {
      const svg = document.getElementById(svgId);
      svg.innerHTML = '';
      const w = svg.clientWidth || 420, h = svg.clientHeight || 220;
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      const pad = {l: 42, r: 14, t: 18, b: 34};
      const values = frames.map(valueFn);
      const max = Math.max(...values.map(v => Math.abs(v)).filter(finite), 1);
      const barW = Math.max(4, (w - pad.l - pad.r) / Math.max(1, frames.length) - 3);
      frames.forEach((f, i) => {
        const v = valueFn(f);
        const x = pad.l + i * ((w - pad.l - pad.r) / Math.max(1, frames.length));
        const y0 = h - pad.b;
        const bh = Math.abs(v) / max * (h - pad.t - pad.b - 6);
        const y = v >= 0 ? y0 - bh : y0;
        const bar = svgEl('rect', {
          x, y, width: barW, height: Math.max(1, bh), fill: colorFn(f), rx: 2,
          style: 'cursor:pointer'
        });
        bar.addEventListener('click', () => {
          selectedId = f.pairIndex;
          expanded.add(f.pairIndex);
          renderAll();
          setTimeout(() => {
            const node = document.getElementById(`frame-node-${f.pairIndex}`);
            if (node && node.scrollIntoView) {
              node.scrollIntoView({block: 'center', inline: 'center', behavior: 'smooth'});
            }
          }, 30);
        });
        svg.appendChild(bar);
        if (frames.length <= 28) addText(svg, x + barW / 2, h - 12, f.pairIndex, 'tiny', 'middle');
        const title = svgEl('title', {}, [`${f.frameId}: ${labelFn(f)}`]);
        svg.lastChild.appendChild(title);
      });
      addLine(svg, pad.l, h - pad.b, w - pad.r, h - pad.b, '#9aa4b2', 1);
      addText(svg, 8, 18, `max ${fmt(max)}`, 'tiny');
    }
    function renderHeatmap(frames) {
      const svg = document.getElementById('heatmapChart');
      svg.innerHTML = '';
      const w = svg.clientWidth || 360, h = svg.clientHeight || 220;
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      const top = 18, left = 42;
      const rows = noFinalBa ? ['rmse', 'boards', 'cam0', 'cam1'] : ['final', 'delta', 'cam0', 'cam1'];
      const cellW = Math.max(6, (w - left - 12) / Math.max(1, frames.length));
      const cellH = 26;
      rows.forEach((name, ri) => addText(svg, 7, top + ri * cellH + 18, name, 'tiny'));
      const maxRmse = Math.max(...frames.map(f => f.finalRmse || 0), 1);
      const maxDelta = Math.max(...frames.map(f => Math.abs(f.rmseDelta || 0)), 1);
      const maxBoards = Math.max(...frames.map(f => f.boardCount || 0), 1);
      frames.forEach((f, i) => {
        const values = noFinalBa
          ? [
              {v: (f.finalRmse || 0) / maxRmse, c: colorForRmse(f.finalRmse)},
              {v: (f.boardCount || 0) / maxBoards, c: '#3867b7'},
              {v: (f.finalCam0Rmse || 0) / maxRmse, c: colorForRmse(f.finalCam0Rmse)},
              {v: (f.finalCam1Rmse || 0) / maxRmse, c: colorForRmse(f.finalCam1Rmse)},
            ]
          : [
              {v: (f.finalRmse || 0) / maxRmse, c: colorForRmse(f.finalRmse)},
              {v: Math.abs(f.rmseDelta || 0) / maxDelta, c: deltaColor(f.rmseDelta)},
              {v: Math.abs(f.cam0Delta || 0) / maxDelta, c: deltaColor(f.cam0Delta)},
              {v: Math.abs(f.cam1Delta || 0) / maxDelta, c: deltaColor(f.cam1Delta)},
            ];
        values.forEach((item, ri) => {
          svg.appendChild(svgEl('rect', {
            x: left + i * cellW, y: top + ri * cellH, width: Math.max(3, cellW - 1),
            height: cellH - 2, fill: item.c, opacity: 0.35 + 0.65 * item.v
          }, [svgEl('title', {}, [`${f.frameId} ${rows[ri]}`])]));
        });
      });
      const conv = diagnostics.convergence || [];
      const baseY = top + rows.length * cellH + 34;
      if (conv.length >= 2) {
        addText(svg, 7, baseY,
          noFinalBa ? 'no final BA: selection-state only' :
            (diagnostics.iterationsAvailable ? 'iteration convergence' : 'initial/final BA summary'),
          'tiny');
        const metrics = [
          ['cost', 'totalCost', '#3867b7'],
          ['cam1', 'cam1Rmse', '#c62828'],
          ['base', 'baselineLength', '#2e7d32'],
          ['rot', 'rotationAngleDeg', '#b85c00'],
        ];
        metrics.forEach((m, mi) => {
          const y = baseY + 20 + mi * 22;
          addText(svg, 7, y + 4, m[0], 'tiny');
          const vals = conv.map(p => p[m[1]]).filter(finite);
          const max = Math.max(...vals, 1e-9);
          let prev = null;
          conv.forEach((p, pi) => {
            const x = left + pi * ((w - left - 18) / Math.max(1, conv.length - 1));
            const yy = y + 10 - ((p[m[1]] || 0) / max) * 16;
            svg.appendChild(svgEl('circle', {cx: x, cy: yy, r: 3.5, fill: m[2]}));
            if (prev) addLine(svg, prev.x, prev.y, x, yy, m[2], 1.5);
            prev = {x, y: yy};
          });
        });
      }
    }
    function renderCharts() {
      const frames = getVisibleFrames();
      renderBarChart('rmseChart', frames, f => f.finalRmse || 0, f => colorForRmse(f.finalRmse), f => `${noFinalBa ? 'selection-state' : 'final'} RMSE ${fmt(f.finalRmse)}`);
      if (noFinalBa) {
        renderBarChart('reductionChart', frames, f => camImbalance(f), f => camImbalance(f) > 0.6 ? '#c62828' : '#3867b7', f => `cam imbalance ${fmt(camImbalance(f))}`);
      } else {
        renderBarChart('reductionChart', frames, f => reduction(f), f => reduction(f) >= 0 ? '#2e7d32' : '#c62828', f => `reduction ${fmt(reduction(f))}`);
      }
      renderHeatmap(frames);
    }
    function renderSummary() {
      const s = diagnostics.summary || {};
      document.getElementById('runSummary').textContent =
        noFinalBa
          ? `state ${s.finalStateLabel || 'selection-state'} | selection ${s.selectionBaResidualMode || 'n/a'} | pairs ${s.selectedPairCount || allFrames.length}`
          : `baseline ${fmt(s.baselineLength, 5)} m | rot ${fmt(s.rotationAngleDeg)} deg | cost ${fmt(s.objectiveStart, 1)} -> ${fmt(s.objectiveFinal, 1)}`;
    }
    function renderAll() {
      renderGraph();
      renderDetails();
      renderCharts();
      renderSummary();
    }
    function download(name, content, mime) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([content], {type: mime}));
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 800);
    }
    function fitToScreen() {
      const visibleWidth = Math.max(320, graphScroll.clientWidth - 24);
      zoomScale = Math.max(0.25, Math.min(1.5, visibleWidth / Math.max(1, lastGraphSize.width)));
      renderGraph();
    }
    function setZoom(next) {
      zoomScale = Math.max(0.25, Math.min(3.0, next));
      renderGraph();
    }
    document.getElementById('collapseAll').addEventListener('click', () => {
      expanded.clear();
      boardExpanded.clear();
      renderAll();
    });
    sortSelect.addEventListener('change', renderAll);
    searchInput.addEventListener('input', renderAll);
    highOnly.addEventListener('change', renderAll);
    residualMetric.addEventListener('change', renderAll);
    thresholdInput.addEventListener('input', renderAll);
    document.getElementById('fitView').addEventListener('click', fitToScreen);
    document.getElementById('zoomIn').addEventListener('click', () => setZoom(zoomScale * 1.18));
    document.getElementById('zoomOut').addEventListener('click', () => setZoom(zoomScale / 1.18));
    let dragState = null;
    graphScroll.addEventListener('mousedown', evt => {
      if (evt.button !== 0) return;
      dragState = {
        x: evt.clientX,
        y: evt.clientY,
        left: graphScroll.scrollLeft,
        top: graphScroll.scrollTop
      };
      graphScroll.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', evt => {
      if (!dragState) return;
      graphScroll.scrollLeft = dragState.left - (evt.clientX - dragState.x);
      graphScroll.scrollTop = dragState.top - (evt.clientY - dragState.y);
    });
    window.addEventListener('mouseup', () => {
      dragState = null;
      graphScroll.style.cursor = '';
    });
    graphScroll.addEventListener('wheel', evt => {
      if (!evt.ctrlKey && !evt.metaKey) return;
      evt.preventDefault();
      setZoom(zoomScale * (evt.deltaY < 0 ? 1.08 : 1 / 1.08));
    }, {passive: false});
    document.getElementById('exportSvg').addEventListener('click', () => {
      download('ba_frame_factor_graph_view.svg', new XMLSerializer().serializeToString(graphSvg), 'image/svg+xml');
    });
    document.getElementById('exportHtml').addEventListener('click', () => {
      download('ba_frame_factor_graph_viewer.html', '<!doctype html>\\n' + document.documentElement.outerHTML, 'text/html');
    });
    document.getElementById('exportPng').addEventListener('click', () => {
      const svgText = new XMLSerializer().serializeToString(graphSvg);
      const img = new Image();
      const url = URL.createObjectURL(new Blob([svgText], {type: 'image/svg+xml'}));
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = graphSvg.viewBox.baseVal.width || graphSvg.clientWidth || 1280;
        canvas.height = graphSvg.viewBox.baseVal.height || graphSvg.clientHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        canvas.toBlob(blob => {
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = 'ba_frame_factor_graph_view.png';
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 800);
        });
      };
      img.src = url;
    });
    window.addEventListener('resize', renderCharts);
    renderAll();
  </script>
</body>
</html>
"""
    Path(output_dir, "viewer.html").write_text(
        template.replace("__DATA__", payload), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 renders every frame; otherwise render this many largest absolute deltas.")
    args = parser.parse_args()

    rows = load_rows(args.trace_csv)
    diagnostics = load_run_diagnostics(args.trace_csv)
    output_dir = Path(args.output_dir)
    graph_dir = output_dir / "frame_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    render_overview(rows, output_dir / "overview.svg", diagnostics)
    frame_rows = rows
    if args.max_frames > 0:
        frame_rows = sorted(
            rows,
            key=lambda r: abs(fnum(r.get("overall_rmse_delta"), 0.0)),
            reverse=True,
        )[:args.max_frames]
        frame_rows.sort(key=lambda r: inum(r.get("pair_index", "0")))
    for row in frame_rows:
        render_frame_graph(row, graph_dir / f"frame_{inum(row.get('pair_index')):06d}.svg", diagnostics)
    render_index(frame_rows, output_dir, diagnostics)
    render_viewer_html(frame_rows, output_dir, args.trace_csv)
    print(f"rendered overview: {output_dir / 'overview.svg'}")
    print(f"rendered frame graphs: {graph_dir}")
    print(f"rendered index: {output_dir / 'index.html'}")
    print(f"rendered viewer: {output_dir / 'viewer.html'}")


if __name__ == "__main__":
    main()
