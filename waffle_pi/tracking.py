"""Small fixed-capacity tracker for demonstration; greedy gated association.

Not a replacement for a validated multi-person tracker. Slots persist
through short occlusion; expiry/reuse clears GST history for that slot.
"""
import numpy as np


class Tracker:
    def __init__(self,n,gate=0.6,expiry=1.0):
        self.n,self.gate,self.expiry = n,gate,expiry
        self.pos = np.zeros((n,2),np.float32)
        self.last = np.full(n,-np.inf)
        self.seen = np.zeros(n,bool)

    def update(self,points,stamp):
        points = np.asarray(points,dtype=np.float32).reshape(-1,2)
        points = points[np.isfinite(points).all(axis=1)]
        active = stamp-self.last<=self.expiry
        reused = np.zeros(self.n,bool)
        self.seen[:] = False
        available = set(range(len(points)))
        candidates = sorted((float(np.linalg.norm(self.pos[i]-p)),i,j)
                            for i in np.flatnonzero(active) for j,p in enumerate(points))
        for d,i,j in candidates:
            if d>self.gate:
                break
            if self.seen[i] or j not in available:
                continue
            self.pos[i],self.last[i],self.seen[i] = points[j],stamp,True
            available.remove(j)
        for i,j in zip(np.flatnonzero(~active),sorted(available)):
            self.pos[i],self.last[i],self.seen[i] = points[j],stamp,True
            reused[i] = True
        return reused
