"""Compare the actual installed detector against the shared-weight path.

NPZ input: ranges [streams,ticks,rays], optional reset [streams,ticks] bool.
Use recorded or Gazebo scans, including occlusion and faults; no fake data
are generated here. Compare two different streams simultaneously.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from waffle_pi.scan_geometry import ScanGeometry
from waffle_pi.detector_engine import DetectorEngine
from gazebo_sim.batched_detector import DetectorPool


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scans',required=True)
    p.add_argument('--profile',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--atol-m',type=float,default=.0005)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    profile=json.loads(Path(a.profile).read_text())
    import hashlib
    if hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()!=profile['detector']['weights_sha256']:
        p.error('Weight hash differs from profile')
    laser=profile['laser']
    settings=profile['detector']
    with np.load(a.scans,allow_pickle=False) as archive:
        scans=np.asarray(archive['ranges'],np.float32)
        reset=np.asarray(archive['reset'],bool) if 'reset' in archive else np.zeros(scans.shape[:2],bool)
    if scans.ndim!=3 or scans.shape[0]<2 or scans.shape[1]<2 or scans.shape[-1]!=laser['rays'] or reset.shape!=scans.shape[:2]:
        p.error('Need at least two independent streams with chronological scans')
    n,t,r=scans.shape
    inc=2*math.pi/r
    start=laser['angle_min']
    geom=ScanGeometry.from_metadata(r,start,start+(r-1)*inc,inc)
    ref=[DetectorEngine(a.weights,settings['model'],a.device,settings['stride'],settings['confidence'],settings['max_range'],geom) for _ in range(n)]
    pool=DetectorPool(a.weights,settings,a.device,max_batch=min(n,16),wait_ms=2.)
    streams=[pool.stream(geom) for _ in range(n)]
    largest=0.
    mismatch=0
    matched=0
    faults=0
    def detect(engine,scan):
        try:
            return engine.detect(scan,laser['range_min'],laser['range_max'])
        except (ValueError,RuntimeError):
            engine.reset()
            return None
    try:
        with ThreadPoolExecutor(max_workers=n) as workers:
            for tick in range(t):
                for i in range(n):
                    if reset[i,tick]:
                        ref[i].reset()
                        streams[i].reset()
                baseline=[detect(e,scans[i,tick]) for i,e in enumerate(ref)]
                result=list(workers.map(lambda i:detect(streams[i],scans[i,tick]),range(n)))
                for x,y in zip(baseline,result):
                    if x is None or y is None:
                        faults+=1
                        mismatch+=int((x is None)!=(y is None))
                        continue
                    if len(x)!=len(y):
                        mismatch+=1
                        continue
                    if len(x):
                        d=np.linalg.norm(x[:,None]-y[None],axis=-1)
                        ix,jx=linear_sum_assignment(d)
                        largest=max(largest,float(d[ix,jx].max()))
                        matched+=len(ix)
        passed=mismatch==0 and largest<=a.atol_m and pool.stats['max_batch']>1 and matched>0
        report=dict(passed=passed,streams=n,ticks=t,max_center_error_m=largest,mismatched_frames=mismatch,
            matched_detections=matched,fault_frames=faults,atol_m=a.atol_m,detector_pool=pool.stats,
            note='Detection parity only; independent policy-quality and physical tests remain required')
        Path(a.output).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if not passed:
            raise SystemExit('Detector parity not established')
    finally:
        pool.close()


if __name__=='__main__':
    main()
