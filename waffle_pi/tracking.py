"""Fixed-capacity constant-velocity Kalman tracker with gated Hungarian matching.
Provisional measurement/process noise; validate against labeled LDS recordings.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self,n,gate=.6,expiry=1.,measurement_std=.08,process_std=.5):
        self.n,self.gate,self.expiry=n,gate,expiry
        self.measurement_std,self.process_std=measurement_std,process_std
        self.state=np.zeros((n,4),dtype=float)
        self.cov=np.repeat(np.eye(4)[None],n,axis=0)
        self.pos=np.zeros((n,2),np.float32)
        self.last=np.full(n,-np.inf)
        self.stamp=None
        self.seen=np.zeros(n,bool)
        self.overflow=False

    def update(self,points,stamp):
        points=np.asarray(points,dtype=float).reshape(-1,2)
        if not np.isfinite(points).all() or not np.isfinite(stamp):
            raise ValueError("Invalid detection")
        if self.stamp is not None and stamp<=self.stamp:
            raise ValueError("Detection timestamps must increase")
        dt=0. if self.stamp is None else stamp-self.stamp
        self.stamp=stamp
        active=stamp-self.last<=self.expiry
        ids=np.flatnonzero(active)
        F=np.eye(4)
        F[:2,2:]=np.eye(2)*dt
        G=np.vstack((np.eye(2)*dt**2/2,np.eye(2)*dt))
        Q=G@G.T*self.process_std**2
        self.state[ids]=self.state[ids]@F.T
        self.cov[ids]=F@self.cov[ids]@F.T+Q
        R=np.eye(2)*self.measurement_std**2
        self.seen[:]=False
        reused=np.zeros(self.n,bool)
        remaining=set(range(len(points)))
        if len(ids) and len(points):
            diff=points[None]-self.state[ids,None,:2]
            S=self.cov[ids,:2,:2]+R
            mahal=np.einsum('ijk,ikl,ijl->ij',diff,np.linalg.inv(S),diff)
            allowed=(mahal<=9.21)&(np.linalg.norm(diff,axis=-1)<=self.gate)
            costs=np.where(allowed,mahal,1e9)
            rows,cols=linear_sum_assignment(costs)
            for row,j in zip(rows,cols):
                if not allowed[row,j]:
                    continue
                i=ids[row]
                K=np.linalg.solve(S[row],self.cov[i,:2,:]).T
                self.state[i]+=K@diff[row,j]
                A=np.eye(4)
                A[:,:2]-=K
                self.cov[i]=A@self.cov[i]@A.T+K@R@K.T
                self.last[i],self.seen[i]=stamp,True
                remaining.remove(j)
        for i,j in zip(np.flatnonzero(~active),sorted(remaining)):
            self.state[i]=(*points[j],0.,0.)
            self.cov[i]=np.diag([self.measurement_std**2]*2+[1.]*2)
            self.last[i],self.seen[i],reused[i]=stamp,True,True
            remaining.remove(j)
        self.overflow=bool(remaining)
        self.pos[:]=self.state[:,:2]
        return reused


class TrajectoryHistory:
    """Timestamped history with interpolation only between valid observations."""
    def __init__(self,n,length,dt=.25,max_gap=.45):
        self.n,self.length,self.dt,self.max_gap=n,length,dt,max_gap
        self.frames=[]

    def clear(self):
        self.frames.clear()

    def append(self,stamp,points,visible,reused):
        points=np.asarray(points,dtype=np.float32).reshape(self.n,2)
        visible=np.asarray(visible,dtype=bool).reshape(self.n)
        reused=np.asarray(reused,dtype=bool).reshape(self.n)
        if self.frames and stamp<=self.frames[-1][0]:
            raise ValueError("Non-increasing trajectory timestamp")
        for _,_,valid in self.frames:
            valid[reused]=False
        self.frames.append((stamp,points.copy(),visible.copy()))
        cutoff=stamp-(self.length-1)*self.dt-self.max_gap
        while len(self.frames)>1 and self.frames[1][0]<cutoff:
            self.frames.pop(0)

    def sample(self,stamp):
        positions=np.full((self.n,self.length,2),-999.,np.float32)
        masks=np.zeros((self.n,self.length,1),np.float32)
        times=np.array([f[0] for f in self.frames])
        for k,t in enumerate(stamp-(self.length-1-np.arange(self.length))*self.dt):
            i=np.searchsorted(times,t-1e-7)
            if i<len(times) and abs(times[i]-t)<1e-6:
                _,p,valid=self.frames[i]
            elif 0<i<len(times) and times[i]-times[i-1]<=self.max_gap:
                _,p0,m0=self.frames[i-1]
                _,p1,m1=self.frames[i]
                weight=(t-times[i-1])/(times[i]-times[i-1])
                p=p0+(p1-p0)*weight
                valid=m0&m1
            else:
                continue
            positions[valid,k]=p[valid]
            masks[valid,k,0]=1.
        return positions[None],masks[None]
