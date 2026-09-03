#!/usr/bin/env python3
"""Build and run the paired semi-synthetic DS P1 recovery experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAT = REPO / "image/babelcalib_export/mul-board/babelcalib_multiboard_export_144928clear_frontend_seed1337/all.mat"
CONFIG = REPO / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
CAMCHAIN = REPO / "config/mono_fisheye_calib_3_25_right-camchain.yaml"
STRENGTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("reference", "smoke", "formal", "all"), default="all")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--smoke-trials", type=int, default=3)
    p.add_argument(
        "--strengths",
        help=(
            "Comma-separated P1 strengths to run. Include 0 when collecting "
            "the recovery-success diagnostic so its clean baseline is defined."
        ),
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--model", choices=("ds-none", "kb"), default="ds-none")
    p.add_argument("--noise-mode", choices=("mad", "none"), default="mad")
    p.add_argument(
        "--noise-aware-weights", action="store_true",
        help=(
            "Write semi-synthetic observation_weight values calibrated from "
            "the outer/internal noise variances."
        ),
    )
    p.add_argument(
        "--fixed-backend-input", action="store_true",
        help=(
            "Use each trial's sampled frame-board set as the exact persistent "
            "backend input in both paired branches."
        ),
    )
    p.add_argument("--backend", type=Path, default=REPO / "build/run_stage5_backend")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_strengths(raw):
    if raw is None:
        return None
    values = tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values or any(not math.isfinite(value) or value < 0.0 or value > 2.0
                         for value in values):
        raise ValueError("--strengths must contain finite values in [0, 2]")
    return values


def csv_read(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k: f"{v:.12f}" if isinstance(v, float) else v for k, v in row.items()})


def kv(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1); out[k.strip()] = v.strip()
    return out


def fingerprint(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def command(cmd, dry):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if not dry: subprocess.run(cmd, cwd=REPO, check=True)


def clean_reference(a, root):
    out = root / "reference_gt" / "clean_outer_internal"
    snapshot = out / "final_persistent_backend_scene.txt"
    if snapshot.is_file() and a.resume: return out
    cmd = [sys.executable, str(Path(__file__).with_name("run_stage5_from_mat.py")), "--mat", str(MAT), "--all-training", "--config", str(CONFIG), "--models", a.model, "--target-mode", "multi_board", "--kalibr-camchain", str(CAMCHAIN), "--backend", str(a.backend), "--output", str(out), "--stage5-disable-selected-case-visualizations"]
    command(cmd, a.dry_run)
    if not a.dry_run and not snapshot.is_file(): raise RuntimeError("reference did not export final_persistent_backend_scene.txt")
    return out


def scene(path):
    cam = None; distortion = (); frames = {}; boards = {}
    for line in path.read_text().splitlines():
        x = line.split()
        if not x: continue
        if x[0] == "camera": cam = tuple(map(float, x[1:7]))
        if x[0] == "distortion": distortion = tuple(map(float, x[2:]))
        if x[0] in ("frame", "board"):
            T = np.asarray(list(map(float, x[5:21]))).reshape(4, 4)
            (frames if x[0] == "frame" else boards)[int(x[1])] = T
    if cam is None or not frames or not boards: raise RuntimeError("malformed GT scene")
    return cam + distortion, frames, boards


def project(cam, p):
    xi, alpha, fu, fv, cu, cv = cam[:6]; x, y, z = p
    if len(cam) == 10:
        radius = math.hypot(x, y)
        theta = math.atan2(radius, z)
        k1, k2, k3, k4 = cam[6:]
        theta2 = theta * theta
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 +
                           k3 * theta2**3 + k4 * theta2**4)
        if radius < 1e-12:
            return np.asarray([cu, cv])
        return np.asarray([fu * theta_d * x / radius + cu,
                           fv * theta_d * y / radius + cv])
    d1 = np.linalg.norm(p); z1 = xi * d1 + z; d2 = math.sqrt(x*x+y*y+z1*z1); den = alpha*d2 + (1-alpha)*z1
    if not np.isfinite(den) or abs(den) < 1e-12: return None
    return np.asarray([fu*x/den+cu, fv*y/den+cv])


def load_source(reference):
    source = reference / "precomputed_input" / "training"; meta = kv(source / "metadata.yaml")
    frames, boards, points = csv_read(source / "frames.csv"), csv_read(source / "boards.csv"), csv_read(source / "points.csv")
    groups = defaultdict(list)
    for point in points: groups[(int(point["frame_index"]), int(point["board_id"]))].append(point)
    return source, frames, boards, groups, int(meta["image_width"]), int(meta["image_height"])


def sigmas(reference):
    rows = csv_read(reference / "benchmark_training_points.csv"); ans = {}
    for kind in ("outer", "internal"):
        v = []
        for r in rows:
            if r.get("method") == "ours" and r.get("point_type") == kind:
                v += [float(r.get("residual_x", 0)), float(r.get("residual_y", 0))]
        x = np.asarray(v); ans[kind] = float(1.4826 * np.median(np.abs(x-np.median(x))))
    if not all(np.isfinite(list(ans.values()))): raise RuntimeError("unable to estimate clean residual MAD")
    return ans


def connected(keys):
    if {b for _, b in keys} != {1,2,3,4,5}: return False
    graph = defaultdict(set)
    for f, b in keys: graph[("f",f)].add(("b",b)); graph[("b",b)].add(("f",f))
    seen, todo = set(), [next(iter(graph))]
    while todo:
        n = todo.pop()
        if n not in seen: seen.add(n); todo += list(graph[n]-seen)
    return len(seen) == len(graph)


def trial_selection(frames, groups, trial, seed, cam, fpose, bpose, width, height):
    ids = [int(x["frame_index"]) for x in frames]; rng = np.random.default_rng(seed + trial * 1009)
    for _ in range(2000):
        pose = set(rng.choice(ids, size=max(1, round(.2*len(ids))), replace=False).tolist())
        strata = defaultdict(list)
        radius = min(cam[4], cam[5], width-1-cam[4], height-1-cam[5])
        for key, points in groups.items():
            if key[0] in pose: continue
            radii = []
            for point in points:
                if point["point_type"] != "outer": continue
                xyz = np.array([float(point["target_x"]),float(point["target_y"]),float(point["target_z"]),1.])
                uv = project(cam, (fpose[key[0]] @ bpose[key[1]] @ xyz)[:3])
                if uv is not None: radii.append(np.linalg.norm(uv-np.array(cam[4:6]))/radius)
            rho = float(np.median(radii)) if radii else 0.0
            strata[(key[1], 0 if rho < .4 else 1 if rho < .7 else 2)].append(key)
        selected = set()
        for entries in strata.values():
            chosen = rng.choice(entries, size=max(1,round(.7*len(entries))), replace=False)
            selected.update(tuple(key) for key in chosen.tolist())
        counts = [sum(b == i for _, b in selected) for i in range(1,6)]
        if min(counts) >= 4 and connected(selected): return pose, selected
    raise RuntimeError("unable to construct connected 70% sampled scene")


def write_scene(path, cam, frames, boards):
    lines = ["camera " + " ".join(f"{x:.17g}" for x in cam[:6]),
             "distortion %d%s" % (len(cam[6:]), "" if len(cam) == 6 else " " + " ".join(f"{x:.17g}" for x in cam[6:]))]
    for i,T in sorted(frames.items()): lines.append("frame %d 1 0 0 %s" % (i, " ".join(f"{x:.17g}" for x in T.ravel())))
    for i,T in sorted(boards.items()): lines.append("board %d 1 0 0 %s" % (i, " ".join(f"{x:.17g}" for x in T.ravel())))
    path.write_text("\n".join(lines) + "\n")


def frozen_input(dest, source, frames, groups, selected, cam, frame_pose, board_pose, sigma, noise_seed, width, height, noise_aware_weights):
    shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True)
    shutil.copyfile(source / "boards.csv", dest / "boards.csv")
    rng = np.random.default_rng(noise_seed); rows = []; excluded = 0
    for f,b in sorted(selected):
        for p in groups[(f,b)]:
            xyz = np.array([float(p["target_x"]),float(p["target_y"]),float(p["target_z"]),1.])
            uv = project(cam, (frame_pose[f] @ board_pose[b] @ xyz)[:3])
            if uv is None: excluded += 1; continue
            uv += rng.normal(0, sigma[p["point_type"]], 2)
            if not (0 <= uv[0] < width and 0 <= uv[1] < height): excluded += 1; continue
            q = dict(p); q["observed_x"] = f"{uv[0]:.12f}"; q["observed_y"] = f"{uv[1]:.12f}"; q["quality"] = "1.0"; rows.append(q)
    if noise_aware_weights:
        outer_count = sum(p["point_type"] == "outer" for p in rows)
        internal_count = sum(p["point_type"] == "internal" for p in rows)
        outer_sigma = sigma["outer"]
        internal_sigma = sigma["internal"]
        internal_weight = 1.0
        if outer_count > 0 and internal_count > 0 and outer_sigma > 0.0 and internal_sigma > 0.0:
            # Stage5's mixed-role balance normalizes each role by its count.
            # This factor restores the two roles' aggregate inverse-variance ratio.
            internal_weight = (internal_count / (internal_sigma * internal_sigma)) / (outer_count / (outer_sigma * outer_sigma))
        for point in rows:
            point["observation_weight"] = f"{1.0 if point['point_type'] == 'outer' else internal_weight:.12f}"
    else:
        for point in rows:
            point["observation_weight"] = "1.0"
    used = {int(p["frame_index"]) for p in rows}; out_frames=[]
    for frame in frames:
        if int(frame["frame_index"]) in used:
            q=dict(frame); q["point_count"] = str(sum(int(p["frame_index"]) == int(frame["frame_index"]) for p in rows)); out_frames.append(q)
    (dest / "metadata.yaml").write_text("%%YAML:1.0\n---\nschema_version: \"stage5_precomputed_observations_v1\"\nimage_width: %d\nimage_height: %d\nreference_board_id: 1\n" % (width,height))
    csv_write(dest / "frames.csv", out_frames); csv_write(dest / "points.csv", rows)
    return fingerprint([(p["frame_index"],p["board_id"],p["point_id"],p["observed_x"],p["observed_y"]) for p in rows]), excluded


def prepare(root, trial, source, frames, groups, cam, fpose, bpose, sigma, width, height, seed, noise_aware_weights):
    d = root / "trials" / f"trial_{trial:03d}"; manifest = d / "trial_manifest.json"
    if manifest.is_file(): return json.loads(manifest.read_text())
    pose, selected = trial_selection(frames, groups, trial, seed, cam, fpose, bpose, width, height); d.mkdir(parents=True, exist_ok=True)
    oh, oe = frozen_input(d/"train", source, frames, groups, selected, cam, fpose,bpose,sigma,seed+trial*4001,width,height,noise_aware_weights)
    eh, ee = frozen_input(d/"pose_eval", source, frames, groups,{k for k in groups if k[0] in pose},cam,fpose,bpose,sigma,seed+trial*4001+1,width,height,noise_aware_weights)
    write_scene(d/"ground_truth_scene.txt",cam,fpose,bpose)
    fixed_input = d / "fixed_backend_input.csv"
    fixed_input.write_text("frame_index,board_id\n" + "".join(f"{frame},{board}\n" for frame, board in sorted(selected)))
    m={"sequence_id":"144928-clear-right","trial_id":trial,"pose_eval_frame_ids":sorted(pose),"selected_frame_board":sorted(selected),"gt_scene_fingerprint":fingerprint([cam,{k:v.tolist() for k,v in fpose.items()},{k:v.tolist() for k,v in bpose.items()}]),"sampled_scene_fingerprint":fingerprint(sorted(selected)),"outer_observation_fingerprint":oh,"pose_eval_observation_fingerprint":eh,"outer_projection_excluded":oe,"pose_eval_projection_excluded":ee,"noise_seed":seed+trial*4001}
    manifest.write_text(json.dumps(m,indent=2)+"\n"); return m


def stage5(a, trial_dir, s, method, output, selection_seed):
    cmd=[str(a.backend),"--config",str(CONFIG),"--models",a.model,"--kalibr-camchain",str(CAMCHAIN),"--runtime-mode","research","--output",str(output),"--cache-dir",str(output/".cache"),"--stage5-precomputed-observations-dir",str(trial_dir/"train"),"--stage5-precomputed-holdout-observations-dir",str(trial_dir/"pose_eval"),"--stage5-precomputed-target-mode","multi_board","--stage5-enable-trial-backend-frame-board-selection","--stage5-trial-backend-selection-candidate-shuffle-seed",str(selection_seed),"--stage5-large-intrinsic-perturbation","P1","--stage5-large-intrinsic-perturbation-scale",str(s),"--stage5-large-intrinsic-perturbation-strict-scale","--stage5-large-intrinsic-perturbation-reference-scene",str(trial_dir/"ground_truth_scene.txt"),"--stage5-disable-selected-case-visualizations","--pre-backend-filter-mode","off"]
    if method == "outer_only": cmd += ["--stage5-large-intrinsic-perturbation-outer-only-after-application"]
    if a.fixed_backend_input:
        fixed_input = trial_dir / "fixed_backend_input.csv"
        cmd += [
            "--stage5-trial-backend-selection-force-include-frame-board-list", str(fixed_input),
            "--stage5-trial-backend-selection-seed-frame-board-list", str(fixed_input),
            "--stage5-trial-backend-selection-force-include-list-is-exact-input", "1",
            "--stage5-trial-backend-selection-candidate-order", "random_shuffle",
            "--stage5-trial-backend-selection-candidate-shuffle-seed", str(selection_seed),
            "--stage5-trial-backend-selection-mi-tol", "1e12",
        ]
    command(cmd,a.dry_run)


def unproject(cam,p):
    xi,a,fu,fv,cu,cv=cam[:6]; mx=(p[:,0]-cu)/fu; my=(p[:,1]-cv)/fv
    if len(cam) == 10:
        rd = np.hypot(mx, my); k1, k2, k3, k4 = cam[6:]
        def distorted(theta):
            theta2 = theta * theta
            return theta * (1.0 + k1 * theta2 + k2 * theta2**2 +
                            k3 * theta2**3 + k4 * theta2**4)
        max_theta = 0.5 * math.pi - 1e-9
        ok = np.isfinite(rd) & (rd <= distorted(max_theta) + 1e-9)
        low = np.zeros_like(rd); high = np.full_like(rd, max_theta)
        for _ in range(80):
            mid = .5 * (low + high)
            low = np.where(distorted(mid) < rd, mid, low)
            high = np.where(distorted(mid) < rd, high, mid)
        theta = .5 * (low + high); scale = np.zeros_like(rd); nonzero = rd >= 1e-12
        scale[nonzero] = np.sin(theta[nonzero]) / rd[nonzero]
        r = np.c_[mx * scale, my * scale, np.cos(theta)]
        r[~nonzero] = np.asarray([0.0, 0.0, 1.0])
        r[~ok] = np.nan
        return r, ok
    r2=mx*mx+my*my; inner=1-(2*a-1)*r2; ok=inner>1e-12; den=a*np.sqrt(np.maximum(inner,0))+1-a; ok &= np.abs(den)>1e-12; mz=np.full_like(mx,np.nan); mz[ok]=(1-a*a*r2[ok])/den[ok]; q=mz*mz+r2; ok &= q>1e-12; k=np.full_like(mx,np.nan); k[ok]=(mz[ok]*xi+np.sqrt(np.maximum(mz[ok]*mz[ok]+(1-xi*xi)*r2[ok],0)))/q[ok]; r=np.c_[k*mx,k*my,k*mz-xi]; n=np.linalg.norm(r,axis=1); ok &= np.isfinite(n)&(n>1e-12); r[ok]/=n[ok,None]; return r,ok


def rays(cam,gt,w,h):
    x,y=np.meshgrid(np.linspace(0,w-1,181),np.linspace(0,h-1,181)); p=np.c_[x.ravel(),y.ravel()]; rg,vg=unproject(gt,p); r,v=unproject(cam,p); rad=min(gt[4],gt[5],w-1-gt[4],h-1-gt[5]); rho=np.linalg.norm(p-np.array(gt[4:6]),axis=1)/rad; fixed=vg&(rho<=1); ok=fixed&v; angle=np.degrees(np.arccos(np.clip(np.sum(r[ok]*rg[ok],axis=1),-1,1))); peri=angle[rho[ok]>=.7]; q=lambda z,k:float(np.percentile(z,k)) if len(z) else math.nan; return {"valid_grid_ratio":float(ok.sum()/fixed.sum()),"invalid_grid_count":int(fixed.sum()-ok.sum()),"full_ray_p95_deg":q(angle,95),"full_ray_median_deg":q(angle,50),"peripheral_ray_p95_deg":q(peri,95),"peripheral_ray_median_deg":q(peri,50)}


def collect(root, records, gt, w,h):
    rows=[]
    for trial,s,method,output,m in records:
        p,t,hold,pose=kv(output/"large_intrinsic_perturbation_summary.txt"),kv(output/"backend_training_summary.txt"),kv(output/"backend_holdout_summary.txt"),kv(output/"large_perturbation_pose_orientation_summary.txt")
        if not p or not t: continue
        if len(gt) == 10:
            initial=(0.0,0.0)+tuple(map(float,p["perturbed_camera_intrinsics"].split(",")))+tuple(map(float,p["perturbed_camera_distortion"].split(",")))
            final=(0.0,0.0)+tuple(map(float,t["camera_intrinsics_csv"].split(",")))+tuple(map(float,t["camera_distortion_csv"].split(",")))
        else:
            initial=tuple(map(float,p["perturbed_camera_intrinsics"].split(","))); final=tuple(map(float,t["camera_intrinsics_csv"].split(",")))
        row={**m,"strength":s,"method":"Outer-only" if method=="outer_only" else "Outer+Internal","run_dir":str(output),"initial_camera_fingerprint":fingerprint(initial),"final_camera_fingerprint":fingerprint(final),"solver_status":t.get("success","0"),"heldout_overall_rmse":hold.get("overall_rmse",""),"orientation_median_deg":pose.get("orientation_median_deg",""),"orientation_p95_deg":pose.get("orientation_p95_deg",""),"pose_eval_success_rate":pose.get("pose_success_rate",""),"projection_center_error_px":math.hypot(final[4]-gt[4],final[5]-gt[5])}
        for n,v in zip(("xi","alpha","fu","fv","cu","cv","k1","k2","k3","k4"),initial): row["initial_"+n]=v
        for n,v in zip(("xi","alpha","fu","fv","cu","cv","k1","k2","k3","k4"),final): row["final_"+n]=v
        row.update({"initial_"+k:v for k,v in rays(initial,gt,w,h).items()}); row.update({"final_"+k:v for k,v in rays(final,gt,w,h).items()}); rows.append(row)
    clean={}
    for r in rows:
        d=Path(r["run_dir"])/"trial_backend_frame_board_selection_decisions.csv"; selected={(x["frame_index"],x["board_id"]) for x in csv_read(d)} if d.is_file() else set()
        r["selected_frame_board_count"]=len(selected)
        if r["strength"]==0: clean[(r["trial_id"],r["method"])]=selected
        r["_selected"]=selected
    threshold=float(np.quantile([r["final_peripheral_ray_p95_deg"] for r in rows if r["strength"]==0],.95))
    for r in rows:
        base=clean.get((r["trial_id"],r["method"]),set()); union=base|r["_selected"]; r["selection_jaccard"]=len(base&r["_selected"])/len(union) if union else math.nan; r["recovery_success_threshold_peripheral"]=threshold; r["recovery_success"]=int(r["solver_status"]=="1" and r["final_valid_grid_ratio"]>=.99 and r["final_peripheral_ray_p95_deg"]<=threshold and float(r["pose_eval_success_rate"] or 0)>=.9); del r["_selected"]
    csv_write(root/"p1_intrinsic_perturbation_all_runs.csv",rows); csv_write(root/"p1_pose_orientation_metrics.csv",rows); csv_write(root/"p1_selection_metrics.csv",rows); csv_write(root/"p1_failures.csv",[r for r in rows if not r["recovery_success"]])
    summary=[]
    for s in sorted(set(r["strength"] for r in rows)):
        for method in ("Outer-only","Outer+Internal"):
            q=[r for r in rows if r["strength"]==s and r["method"]==method]
            if q:
                z={"sequence_id":"144928-clear-right","strength":s,"method":method,"trial_count":len(q),"success_rate":float(np.mean([r["recovery_success"] for r in q]))}
                for k in ("final_full_ray_p95_deg","final_peripheral_ray_p95_deg","projection_center_error_px","orientation_p95_deg","selection_jaccard"):
                    v=np.asarray([float(r[k]) for r in q if r[k] not in ("",None)],float); z[k+"_median"]=float(np.median(v)) if len(v) else math.nan; z[k+"_q25"]=float(np.quantile(v,.25)) if len(v) else math.nan; z[k+"_q75"]=float(np.quantile(v,.75)) if len(v) else math.nan
                summary.append(z)
    csv_write(root/"p1_intrinsic_perturbation_summary.csv",summary); plot(root,summary)


def plot(root, summary):
    plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    metrics=[("final_full_ray_p95_deg","Full-field Ray P95 [deg]"),("final_peripheral_ray_p95_deg","Peripheral Ray P95 [deg]"),("projection_center_error_px","Projection-center error [px]"),("orientation_p95_deg","Camera orientation P95 [deg]"),("success_rate","Recovery success rate"),("selection_jaccard","Selection Jaccard overlap")]; fig,axes=plt.subplots(2,3,figsize=(9,5.2),sharex=True); colors={"Outer-only":"#D55E00","Outer+Internal":"#0072B2"}
    for ax,(m,label) in zip(axes.ravel(),metrics):
        for method,c in colors.items():
            q=sorted((r for r in summary if r["method"]==method),key=lambda r:r["strength"]); x=[r["strength"] for r in q]; y=[r[m+"_median"] if m!="success_rate" else r[m] for r in q]; ax.plot(x,y,"o-",ms=3,lw=1.5,color=c,label=method)
            if m!="success_rate": ax.fill_between(x,[r[m+"_q25"] for r in q],[r[m+"_q75"] for r in q],color=c,alpha=.18)
        ax.set_ylabel(label); ax.grid(axis="y",color="#ddd",lw=.6)
    for ax in axes[1]: ax.set_xlabel("P1 perturbation strength")
    axes[0,0].legend(frameon=False); fig.tight_layout(); fig.savefig(root/"p1_perturbation_figure.png",dpi=300); fig.savefig(root/"p1_perturbation_figure.pdf"); plt.close(fig)


def main():
    a=args(); root=a.output.resolve(); root.mkdir(parents=True,exist_ok=True); ref=clean_reference(a,root)
    if a.phase=="reference": return
    source,frames,boards,groups,w,h=load_source(ref); cam,fpose,bpose=scene(ref/"final_persistent_backend_scene.txt"); sigma=sigmas(ref)
    if a.noise_mode == "none":
        sigma = {"outer": 0.0, "internal": 0.0}
    selected_strengths = parse_strengths(a.strengths)
    n=a.smoke_trials if a.phase=="smoke" else a.trials
    strengths = (selected_strengths if selected_strengths is not None else
                 ((0.,1.,1.4) if a.phase=="smoke" else STRENGTHS))
    (root/"p1_ground_truth_scenes.yaml").write_text("sequence_id: 144928-clear-right\nmodel: %s\nscene: %s\n"%(a.model,ref/"final_persistent_backend_scene.txt")); (root/"p1_experiment_config.yaml").write_text("model: %s\nnoise_mode: %s\nnoise_aware_weights: %d\nfixed_backend_input: %d\npre_backend_filter_mode: off\npose_eval_ratio: 0.20\ncalibration_frame_board_ratio: 0.70\nstrengths: [%s]\nsigma_outer: %.12g\nsigma_internal: %.12g\n"%(a.model,a.noise_mode,a.noise_aware_weights,a.fixed_backend_input,", ".join(map(str,strengths)),sigma["outer"],sigma["internal"]))
    records=[]
    for i in range(n):
        m=prepare(root,i,source,frames,groups,cam,fpose,bpose,sigma,w,h,a.seed,a.noise_aware_weights); t=root/"trials"/f"trial_{i:03d}"
        for s in strengths:
            for method in ("outer_only","outer_internal"):
                out=root/"runs"/f"trial_{i:03d}"/f"s_{s:.3f}"/method
                if not (a.resume and (out/"backend_training_summary.txt").is_file()): stage5(a,t,s,method,out,a.seed+i)
                records.append((i,s,method,out,m))
    if not a.dry_run: collect(root,records,cam,w,h)


if __name__=="__main__": main()
