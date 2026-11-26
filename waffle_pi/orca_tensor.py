"""Experimental dense batched agent-agent ORCA, no RVO2 or host sync.

RVO2-style half-plane construction and incremental 2D linear programs.
Parallel over worlds and agents; constraint iteration is sequential.
No obstacle-line solver. Treat numerical degeneracies and tie ordering
as differences from native RVO2, and validate before replacing it.
"""
import torch


def dot(a,b):
    return (a*b).sum(-1)


def det(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def perp(a):
    return torch.stack((-a[...,1],a[...,0]),-1)


def unit(a,eps=1e-8):
    return a/torch.linalg.vector_norm(a,dim=-1,keepdim=True).clamp_min(eps)


def clip_circle(v,r):
    return v/(torch.linalg.vector_norm(v,dim=-1)/r.clamp_min(1e-8)).clamp_min(1.)[...,None]


def lines(pos,vel,radius,dt,horizon,neighbor_dist,margin):
    """pos, vel [B,N,2], radii [B,N]; outputs lines sorted by distance."""
    p=pos[:,None,:,:]-pos[:,:,None,:]
    v=vel[:,:,None,:]-vel[:,None,:,:]
    d2=dot(p,p)
    r=radius[:,:,None]+radius[:,None,:]+2*margin
    r2=r.square()
    n=pos.shape[1]
    ids=torch.arange(n,device=pos.device)
    valid=(ids[:,None]!=ids[None,:])[None] & (d2<neighbor_dist**2)
    w=v-p/horizon
    w2=dot(w,w)
    pdot=dot(w,p)
    cutoff=(pdot<0)&(pdot.square()>r2*w2)
    uw=unit(w)
    d_cut=-perp(uw)
    u_cut=(r/horizon-w2.sqrt())[...,None]*uw
    leg=(d2-r2).clamp_min(0).sqrt()
    px,py=p.unbind(-1)
    left=torch.stack((px*leg-py*r,px*r+py*leg),-1)/d2.clamp_min(1e-8)[...,None]
    right=-torch.stack((px*leg+py*r,-px*r+py*leg),-1)/d2.clamp_min(1e-8)[...,None]
    d_leg=torch.where((det(p,w)>0)[...,None],left,right)
    u_leg=dot(v,d_leg)[...,None]*d_leg-v
    direction=torch.where(cutoff[...,None],d_cut,d_leg)
    u=torch.where(cutoff[...,None],u_cut,u_leg)
    wc=v-p/dt
    wc_norm=torch.linalg.vector_norm(wc,dim=-1)
    # deterministic antisymmetric fallback for coincident, equal-velocity agents
    sign=torch.where(ids[:,None]<ids[None,:],1.,-1.).expand_as(wc_norm)
    fallback=torch.stack((sign,torch.zeros_like(sign)),-1)
    uwc=torch.where((wc_norm>1e-8)[...,None],unit(wc),fallback)
    overlap=d2<=r2
    direction=torch.where(overlap[...,None],-perp(uwc),direction)
    u=torch.where(overlap[...,None],(r/dt-wc_norm)[...,None]*uwc,u)
    point=vel[:,:,None,:]+0.5*u
    order=torch.argsort(d2.masked_fill(~valid,float('inf')),dim=-1)
    ix=order[...,None].expand_as(point)
    return point.gather(2,ix),direction.gather(2,ix),valid.gather(2,order)


def lp1(points,dirs,valid,i,r,opt,directional):
    p,d=points[...,i,:],dirs[...,i,:]
    proj=dot(p,d)
    discr=proj.square()+r.square()-dot(p,p)
    root=discr.clamp_min(0).sqrt()
    lo,hi=-proj-root,-proj+root
    denom=det(d[...,None,:],dirs)
    numer=det(dirs,p[...,None,:]-points)
    prev=valid & (torch.arange(points.shape[-2],device=points.device)<i)
    parallel=denom.abs()<1e-6
    feasible=~(prev&parallel&(numer<0)).any(-1)
    bound=numer/torch.where(parallel,1.,denom)
    lower=torch.where(prev&~parallel&(denom<0),bound,-float('inf')).amax(-1)
    upper=torch.where(prev&~parallel&(denom>0),bound,float('inf')).amin(-1)
    lo,hi=torch.maximum(lo,lower),torch.minimum(hi,upper)
    t=dot(opt-p,d)
    if directional:
        t=torch.where(dot(opt,d)>0,hi,lo)
    else:
        t=torch.minimum(torch.maximum(t,lo),hi)
    return p+t[...,None]*d,feasible&(discr>=0)&(lo<=hi)


def lp2(points,dirs,valid,r,opt,directional=False):
    result=opt*r[...,None] if directional else clip_circle(opt,r)
    count=points.shape[-2]
    fail=torch.full_like(r,count,dtype=torch.long)
    for i in range(count):
        active=valid[...,i]&(fail==count)&(det(dirs[...,i,:],points[...,i,:]-result)>0)
        proposal,feasible=lp1(points,dirs,valid,i,r,opt,directional)
        result=torch.where((active&feasible)[...,None],proposal,result)
        fail=torch.where(active&~feasible,i,fail)
    return result,fail


def lp3(points,dirs,valid,r,result,fail):
    """Least-violation fallback for infeasible half-plane intersections."""
    distance=torch.zeros_like(r)
    count=points.shape[-2]
    indices=torch.arange(count,device=points.device)
    for i in range(count):
        pi,di=points[...,i,:],dirs[...,i,:]
        violation=det(di,pi-result)
        active=valid[...,i]&(fail<=i)&(violation>distance)
        denom=det(di[...,None,:],dirs)
        parallel=denom.abs()<1e-6
        opposite=dot(di[...,None,:],dirs)<=0
        proj_valid=valid&(indices<i)&(~parallel|opposite)
        ratio=det(dirs,pi[...,None,:]-points)/torch.where(parallel,1.,denom)
        intersect=pi[...,None,:]+ratio[...,None]*di[...,None,:]
        proj_points=torch.where(parallel[...,None],(pi[...,None,:]+points)/2,intersect)
        proj_dirs=unit(dirs-di[...,None,:])
        candidate,bad=lp2(proj_points,proj_dirs,proj_valid,r,perp(di),True)
        result=torch.where((active&(bad==count))[...,None],candidate,result)
        distance=torch.where(active,det(di,pi-result),distance)
    return result


def orca_velocity(pos,vel,radius,preferred,max_speed,dt=0.25,horizon=5.,
                  neighbor_dist=10.,margin=.16):
    points,dirs,valid=lines(pos,vel,radius,dt,horizon,neighbor_dist,margin)
    result,fail=lp2(points,dirs,valid,max_speed,preferred)
    return lp3(points,dirs,valid,max_speed,result,fail)
