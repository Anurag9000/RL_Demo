"""Offline calibration from manually commissioned, timestamped CSV logs.

No robot commands are published. Fit wheel response using SI body velocities.
The result is a model fit, not a certified physical stopping-distance bound.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load_csv(path, columns):
    with open(path,newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows or any(k not in rows[0] for k in columns):
        raise ValueError("CSV requires columns "+','.join(columns))
    data=np.asarray([[float(row[k]) for k in columns] for row in rows],dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite calibration input")
    return data


def body_wheels(v,w,r=.033,L=.287):
    return np.stack((v-L*w/2,v+L*w/2),axis=-1)/r


def fit_motor(data):
    """Fit tau, wheel acceleration and command delay, with temporal holdout."""
    t,cv,cw,mv,mw=data.T
    if len(t)<60 or np.any(np.diff(t)<=0) or t[-1]-t[0]<5.:
        raise ValueError("Need >=60 strictly timestamped samples spanning >=5 s")
    if np.max(np.abs(cv))>.261 or np.max(np.abs(cw))>1.821:
        raise ValueError("Command exceeds nominal Waffle body limits")
    target=body_wheels(cv,cw)
    measured=body_wheels(mv,mw)
    limit=.26/.033
    target=target/np.maximum(1.,np.max(abs(target),axis=-1,keepdims=True)/limit)
    split=int(.7*len(t))
    if np.max(np.ptp(target[:split],axis=0))<1. or np.max(np.ptp(target[split:],axis=0))<1.:
        raise ValueError("Training and holdout segments both require varied commands")
    measured_accel=np.abs(np.diff(measured[:split],axis=0)/np.diff(t[:split])[:,None])
    base=max(.5,float(np.quantile(measured_accel,.9)))
    tau,acc,delay=np.meshgrid(np.geomspace(.025,1.,20),base*np.array([.5,1.,2.,4.]),
                              np.linspace(0.,.5,21),indexing='ij')
    tau,acc,delay=(a.reshape(-1) for a in (tau,acc,delay))
    # Parallel over candidate parameters; time evolution remains causal.
    pred=np.broadcast_to(measured[0],(len(tau),2)).copy()
    errors=np.zeros(len(tau))
    for i in range(1,split):
        dt=t[i]-t[i-1]
        ix=np.searchsorted(t,t[i-1]-delay,side='right')-1
        desired=np.where((ix>=0)[:,None],target[np.maximum(ix,0)],0.)
        delta=-np.expm1(-dt/tau[:,None])*(desired-pred)
        pred+=np.clip(delta,-acc[:,None]*dt,acc[:,None]*dt)
        errors+=np.sum((pred-measured[i])**2,axis=-1)
    j=int(np.argmin(errors))
    tau,acc,delay=float(tau[j]),float(acc[j]),float(delay[j])
    pred=measured[split-1].copy()
    err=0.
    for i in range(split,len(t)):
        dt=t[i]-t[i-1]
        ix=np.searchsorted(t,t[i-1]-delay,side='right')-1
        desired=target[ix] if ix>=0 else np.zeros(2)
        pred+=np.clip(-np.expm1(-dt/tau)*(desired-pred),-acc*dt,acc*dt)
        err+=float(np.sum((pred-measured[i])**2))
    return dict(schema='waffle_motor_fit_v1',motor_tau=tau,wheel_accel=acc,
                command_delay_s=delay,train_rmse_rad_s=float(np.sqrt(errors[j]/(2*(split-1)))),
                holdout_rmse_rad_s=float(np.sqrt(err/(2*(len(t)-split)))),
                samples=len(t),safety_validated=False)


def fit_detector(data):
    distance,seen,tx,ty,dx,dy=data.T
    if np.any(distance<0) or not np.isin(seen,[0.,1.]).all():
        raise ValueError("Invalid ground-truth range or detected flag")
    valid=seen.astype(bool)
    errors=np.stack((dx-tx,dy-ty),-1)
    bins=[]
    for lo,hi in zip([0.,1.,2.,3.,4.],[1.,2.,3.,4.,5.]):
        group=(distance>=lo)&(distance<hi)
        hits=group&valid
        n=int(group.sum())
        if not n:
            continue
        bins.append(dict(range_m=[lo,hi],samples=n,recall=float(hits.sum()/n),
                         mean_error_m=errors[hits].mean(0).tolist() if hits.any() else None,
                         rms_error_m=float(np.sqrt(np.mean(errors[hits]**2))) if hits.any() else None))
    return dict(schema='waffle_detector_fit_v1',bins=bins,
                recall=float(valid.mean()),samples=len(data),
                note='Matched truth rows only; unmatched false positives need a separate count')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['motor','detector'])
    parser.add_argument('csv')
    parser.add_argument('output')
    a=parser.parse_args()
    path=Path(a.output)
    if path.exists():
        raise ValueError("Choose a new output file")
    if a.kind=='motor':
        result=fit_motor(load_csv(a.csv,['t_s','cmd_v','cmd_w','meas_v','meas_w']))
    else:
        result=fit_detector(load_csv(a.csv,['distance_m','detected','truth_x','truth_y','det_x','det_y']))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
