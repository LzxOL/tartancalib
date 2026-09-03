#!/usr/bin/env python3
"""Export all valid frontend outer-cache detections as a pose-evaluation MAT."""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from scipy.io import savemat

def struct(records, fields):
    out=np.empty((1,len(records)),dtype=[(x,'O') for x in fields])
    for i,r in enumerate(records):
        for f in fields: out[0,i][f]=r[f]
    return out

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache-dir',type=Path,required=True); p.add_argument('--source-run',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--test-ratio',type=float,default=.30); p.add_argument('--seed',type=int,default=1337); p.add_argument('--all-cache-frames',action='store_true',help='include valid cached frames not present in the backend selection log'); a=p.parse_args()
 labels={r['frame_label']:int(r['frame_index']) for r in csv.DictReader((a.source_run/'trial_backend_frame_board_selection_decisions.csv').open())}
 # Outer detector order is canonical: (0,0),(pitch,0),(pitch,pitch),(0,pitch).
 targets={}
 for r in csv.DictReader((a.source_run/'backend_training_points.csv').open()):
  if r['point_type']=='outer': targets.setdefault(int(r['board_id']),[]).append((int(r['point_id']),float(r['target_x']),float(r['target_y'])))
 targets={b:[next((x,y) for pid,x,y in v if pid==want) for want in (0,10,120,110)] for b,v in targets.items()}
 poses={}
 for r in csv.DictReader((a.source_run/'backend_board_poses.csv').open()):
  if r['initialized']=='1': poses[int(r['board_id'])]=np.array([float(r['T_reference_board_16'])]+[float(x) for x in r[None]],float).reshape(4,4)
 groups=defaultdict(list)
 for path in a.cache_dir.rglob('*.yml'):
  if path.name=='manifest.yaml': continue
  fs=cv2.FileStorage(str(path),cv2.FileStorage_READ); image_path=fs.getNode('absolute_image_path').string(); label=Path(image_path).stem
  if a.all_cache_frames and not Path(image_path).is_file():
   fs.release(); continue
  if a.all_cache_frames and label not in labels:
   try: labels[label]=int(label.split('_',1)[0]) - 1
   except (ValueError, IndexError): pass
  if label not in labels: continue
  ds=fs.getNode('detections')
  for i in range(ds.size()):
   d=ds.at(i); b=int(d.getNode('board_id').real()); valid=d.getNode('refined_valid')
   if b not in targets or int(d.getNode('success').real())!=1 or int(d.getNode('good').real())!=1 or valid.size()!=4 or not all(int(valid.at(j).real()) for j in range(4)): continue
   xy=np.array([[d.getNode('refined_corners_original_image').at(j).at(k).real() for j in range(4)] for k in range(2)],float)
   groups[(labels[label],label)].append((b,xy))
  fs.release()
 board_ids=sorted({b for v in groups.values() for b,_ in v}); board_index={b:i+1 for i,b in enumerate(board_ids)}
 boards=struct([{'X':np.array(targets[b],float).T,'Rt':poses[b][:3,:4],'board_id':np.array([[b]],np.int32)} for b in board_ids],('X','Rt','board_id'))
 keys=sorted(groups); rng=random.Random(a.seed); shuffled=keys[:];rng.shuffle(shuffled); test=set(shuffled[:round(len(keys)*a.test_ratio)])
 def corners(keys):
  rec=[]
  for key in keys:
   xs=[];cs=[]
   for b,xy in groups[key]: xs.append(xy); cs.append(np.vstack((np.arange(1,5),np.full(4,board_index[b]))))
   rec.append({'x':np.hstack(xs),'cspond':np.hstack(cs).astype(np.uint16)})
  return struct(rec,('x','cspond'))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for name,ks in [('all',keys),('test',sorted(test)),('train',sorted(set(keys)-test))]: savemat(a.output_dir/f'{name}.mat',{'corners':corners(ks),'boards':boards,'imgsize':np.array([[4512,4512]],float)},do_compression=True)
 (a.output_dir/'frames_test.jsonl').write_text(''.join(json.dumps({'frame_index':i,'frame_label':l})+'\n' for i,l in sorted(test)))
 (a.output_dir/'summary.json').write_text(json.dumps({'all_frames':len(keys),'test_frames':len(test),'groups':sum(map(len,groups.values()))},indent=2)+'\n')
 print((a.output_dir/'summary.json').read_text())
if __name__=='__main__': main()
