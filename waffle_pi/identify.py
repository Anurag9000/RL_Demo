"""Offline system identification with separate fitting and holdout records.

All inputs are recorded, synchronized SI values. No hardware commands.
Fits a sampled actuator model, effective rolling geometry, laser residuals
and conditional detection statistics; none constitutes safety certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .calibration import load_csv


WHEEL_COLUMNS=('t_s','target_l_rad_s','target_r_rad_s','wheel_l_rad_s','wheel_r_rad_s')
GEOM_COLUMNS=('wheel_l_rad_s','wheel_r_rad_s','truth_v_m_s','truth_w_rad_s')


def rmse(a):
    return np.sqrt(np.mean(np.square(a),axis=0))


def check_wheel(x,dt):
    if x.ndim!=2 or x.shape[1]!=5 or len(x)<60 or not np.isfinite(x).all():
        raise ValueError('Need at least 60 finite synchronized wheel observations')
    if not np.allclose(np.diff(x[:,0]),dt,rtol=.01,atol=1e-5):
        raise ValueError('Wheel samples must use the declared uniform physics timestep')
    if np.any(np.ptp(x[:,1:3],axis=0)<1.) or np.any(np.max(x[:,1:3],axis=0)<.5) or np.any(np.min(x[:,1:3],axis=0)>-.5):
        raise ValueError('Each recording needs varied forward, reverse and turn commands')
    if np.max(np.abs(x[:,1:3]))>8.0:
        raise ValueError('Wheel targets outside nominal Waffle Pi range')


def wheel_predict(record,parameters,delay,dt):
    # Same response equation and braking criterion as fidelity.motor_step.
    tau,acc,brake,gain,deadband=np.asarray(parameters).T
    out=np.empty_like(record[:,3:5])
    out[0]=record[0,3:5]
    for i in range(1,len(record)):
        j=i-1-delay
        target=record[j,1:3] if j>=0 else np.zeros(2)
        desired=np.where(np.abs(target)<deadband,0.,target*gain)
        slowing=(out[i-1]*desired<=0)|(np.abs(desired)<np.abs(out[i-1]))
        bound=np.where(slowing,brake,acc)*dt
        delta=-np.expm1(-dt/tau)*(desired-out[i-1])
        out[i]=out[i-1]+np.clip(delta,-bound,bound)
    return out


def fit_wheels(train,holdout,dt,max_delay=.5):
    if not .005<=dt<=.05 or not 0<=max_delay<=.5:
        raise ValueError('Invalid identification step or delay')
    check_wheel(train,dt)
    check_wheel(holdout,dt)
    lo=np.tile([.005,.1,.1,.3,0.],(2,1))
    hi=np.tile([2.,200.,200.,1.5,2.],(2,1))
    # Deadband is discontinuous: grid search its values, then optimize the
    # continuous tau, acceleration, braking and gain for each wheel.
    # Delay also remains a discrete sampled-data parameter.
    initial=np.tile([.15,12.,12.,1.,0.],(2,1))
    candidates=[]
    for delay in range(round(max_delay/dt)+1):
        # Fit using one-step prediction first (vectorized, cheap), then
        # evaluate open-loop residual across the entire fit recording.
        last=train[:-1,3:5]
        measured=train[1:,3:5]
        ix=np.arange(len(last))-delay
        target=np.where((ix>=0)[:,None],train[np.maximum(ix,0),1:3],0.)
        params=[]
        for side in range(2):
            best=None
            for band in (0.,.1,.2,.4,.8,1.2,1.6,2.):
                desired=np.where(np.abs(target[:,side])<band,0.,target[:,side])
                prev=last[:,side]
                def residual(p):
                    tau,acc,brake,gain=p
                    target_speed=desired*gain
                    slowing=(prev*target_speed<=0)|(np.abs(target_speed)<np.abs(prev))
                    bound=np.where(slowing,brake,acc)*dt
                    pred=prev+np.clip(-np.expm1(-dt/tau)*(target_speed-prev),-bound,bound)
                    return pred-measured[:,side]
                fit=least_squares(residual,initial[side,:4],bounds=(lo[side,:4],hi[side,:4]),
                                  loss='soft_l1',f_scale=.05,max_nfev=150)
                err=float(np.mean(residual(fit.x)**2))
                if best is None or err<best[0]:
                    best=err,np.r_[fit.x,band]
            params.append(best[1])
        params=np.asarray(params)
        pred=wheel_predict(train,params,delay,dt)
        candidates.append((float(np.mean((pred-train[:,3:5])**2)),delay,params))
    _,delay,params=min(candidates,key=lambda c:c[0])
    # Refine continuous parameters jointly against open-loop training rollout.
    def residual(v):
        p=np.c_[v.reshape(2,4),params[:,4]]
        return (wheel_predict(train,p,delay,dt)-train[:,3:5]).ravel()
    fit=least_squares(residual,params[:,:4].ravel(),bounds=(lo[:,:4].ravel(),hi[:,:4].ravel()),
                     loss='soft_l1',f_scale=.1,max_nfev=100)
    params=np.c_[fit.x.reshape(2,4),params[:,4]]
    tr=wheel_predict(train,params,delay,dt)-train[:,3:5]
    va=wheel_predict(holdout,params,delay,dt)-holdout[:,3:5]
    return dict(motor=dict(tau_s=params[:,0].tolist(),accel_rad_s2=params[:,1].tolist(),
        brake_rad_s2=params[:,2].tolist(),gain=params[:,3].tolist(),
        deadband_rad_s=params[:,4].tolist(),delay_s=delay*dt),
        train_rmse_rad_s=rmse(tr).tolist(),holdout_rmse_rad_s=rmse(va).tolist(),
        holdout_p95_abs_rad_s=np.quantile(abs(va),.95,axis=0).tolist(),
        delay_grid_resolution_s=dt,fit_samples=len(train),holdout_samples=len(holdout),
        note='Quantization, slip, voltage and load not separately identifiable from wheel velocity alone')


def fit_geometry(train,holdout):
    for x in (train,holdout):
        if x.ndim!=2 or x.shape[1]!=4 or len(x)<30 or not np.isfinite(x).all():
            raise ValueError('Need synchronized wheel and independent external velocity truth')
        if np.linalg.matrix_rank(x[:,:2])!=2 or np.max(abs(x[:,3]))<.1:
            raise ValueError('Both straight and turning excitation required')
    w=train[:,:2]
    def predict(p,x):
        ground=x[:,:2]*p[:2]
        return np.column_stack((ground.sum(1)/2,(ground[:,1]-ground[:,0])/p[2]))
    scale=np.array([.1,.5])
    fit=least_squares(lambda p:((predict(p,train)-train[:,2:])/scale).ravel(),
                      [.033,.033,.287],bounds=([.02,.02,.20],[.05,.05,.36]),
                      loss='soft_l1',f_scale=.1)
    return dict(motor=dict(radius_m=fit.x[:2].tolist(),track_m=float(fit.x[2])),
                train_rmse_v_w=rmse(predict(fit.x,train)-train[:,2:]).tolist(),
                holdout_rmse_v_w=rmse(predict(fit.x,holdout)-holdout[:,2:]).tolist(),
                note='Ground truth must be independent of the wheel-derived ROS odometry')


def fit_laser(train,holdout):
    # truth range, measured range, valid return (0/1); both sets must
    # contain only in-range, independently measured visible targets.
    def stats(x):
        if len(x)<30 or not np.isfinite(x).all() or not np.isin(x[:,2],[0.,1.]).all() or np.any(x[:,0]<=0):
            raise ValueError('Invalid laser calibration samples')
        valid=x[:,2].astype(bool)
        if valid.sum()<20:
            raise ValueError('Insufficient valid laser returns')
        error=x[valid,1]-x[valid,0]
        return float(error.mean()),float(error.std(ddof=1)),float(1-valid.mean()),int(valid.sum())
    bias,std,drop,n=stats(train)
    vb,vs,vd,vn=stats(holdout)
    return dict(laser=dict(range_bias_m=bias,noise_std_m=std,drop_probability=drop),
                holdout_bias_m=vb,holdout_std_m=vs,holdout_drop_probability=vd,
                fit_valid=n,holdout_valid=vn)


def fit_detector_bins(train,holdout,max_range):
    # [true range, matched flag, true x/y, detected x/y] in LASER axes;
    # input rows should represent geometrically visible people only.
    bins=[]
    if not 0<max_range<=5:
        raise ValueError('Invalid policy range')
    edges=np.r_[np.arange(0.,max_range,1.),max_range]
    for x in (train,holdout):
        if not np.isfinite(x).all() or not np.isin(x[:,1],[0.,1.]).all() or np.any(x[:,0]<0):
            raise ValueError('Invalid detector calibration')
    for lo,hi in zip(edges[:-1],edges[1:]):
        def residual(x):
            group=(x[:,0]>=lo)&(x[:,0]<hi)
            hits=group&(x[:,1]==1)
            if group.sum()<20 or hits.sum()<10:
                raise ValueError('Insufficient fit/holdout labels in range bin '+str((lo,hi)))
            return x[hits,4:6]-x[hits,2:4],float(hits.sum()/group.sum()),int(group.sum())
        e,rec,n=residual(train)
        v,vrec,vn=residual(holdout)
        bins.append(dict(range_m=[float(lo),float(hi)],recall=rec,bias_m=e.mean(0).tolist(),
            std_m=e.std(0,ddof=1).tolist(),fit_samples=n,
            holdout_recall=vrec,holdout_samples=vn,holdout_rmse_m=rmse(v).tolist()))
    return dict(detector_bins=bins,note='Conditional on visible people; unmatched false positives must be separately counted')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['wheel','geometry','laser','detector'])
    p.add_argument('fit_csv')
    p.add_argument('holdout_csv')
    p.add_argument('output')
    p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--max-delay',type=float,default=.5)
    p.add_argument('--max-range',type=float,default=3.)
    a=p.parse_args()
    fit,holdout,out=map(Path,(a.fit_csv,a.holdout_csv,a.output))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if fit.resolve()==holdout.resolve() or digest(fit)==digest(holdout):
        p.error('Fit and holdout must be independent records')
    cols={'wheel':WHEEL_COLUMNS,'geometry':GEOM_COLUMNS,
          'laser':('truth_range_m','measured_range_m','valid'),
          'detector':('distance_m','detected','truth_x','truth_y','det_x','det_y')}[a.kind]
    x,y=load_csv(fit,cols),load_csv(holdout,cols)
    if a.kind=='wheel':
        result=fit_wheels(x,y,a.dt,a.max_delay)
    elif a.kind=='geometry':
        result=fit_geometry(x,y)
    elif a.kind=='laser':
        result=fit_laser(x,y)
    else:
        result=fit_detector_bins(x,y,a.max_range)
    result.update(schema='waffle_identification_v1',kind=a.kind,
                  fit_sha256=digest(fit),holdout_sha256=digest(holdout),
                  measured_profile_certified=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
