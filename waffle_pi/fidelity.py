"""Measured-profile substep dynamics and geometric 2D laser simulation.

Physics is planar, not a motor electrical/firmware emulator. Human cross
sections are disks. Rays are instantaneous, not scan-deskewed. Calibration
and physical commissioning remain mandatory. No ROS or motor output here.
"""
import hashlib
import json
import math
from pathlib import Path
import torch

CONTROL_DT=.25


def read_profile(path,require_measured=False):
    raw=Path(path).read_bytes()
    data=json.loads(raw)
    if data.get('schema')!='waffle_fidelity_v1':
        raise ValueError('Invalid fidelity schema')
    if data.get('status') not in ('illustrative','measured'):
        raise ValueError('Declare profile provenance')
    if require_measured:
        if data['status']!='measured' or not all(data.get('hardware',{}).get(k)
                for k in ('serial_or_lab_id','lds_model','firmware','floor','measurement_record')):
            raise ValueError('Measured profile and hardware evidence references required')
    dt=float(data['physics_dt'])
    if not math.isfinite(dt) or not .005<=dt<=.05 or abs(CONTROL_DT/dt-round(CONTROL_DT/dt))>1e-6:
        raise ValueError('physics_dt must divide .25 s and be in [.005,.05]')
    m=data['motor']
    for k,lo,hi in [('tau_s',.005,2.),('accel_rad_s2',.1,200.),('brake_rad_s2',.1,200.),
                    ('gain',.3,1.5),('radius_m',.02,.05),('deadband_rad_s',0.,2.)]:
        v=m[k]
        if len(v)!=2 or any(not math.isfinite(x) or not lo<=x<=hi for x in v):
            raise ValueError('Invalid left/right motor '+k)
    if not .20<=m['track_m']<=.36 or not 0<=m['delay_s']<=.5:
        raise ValueError('Invalid track or delay')
    s=data['laser']
    if not isinstance(s['rays'],int) or not 32<=s['rays']<=4096:
        raise ValueError('Invalid ray count')
    if not 0<s['range_min']<s['range_max']<=30 or not 1<=s['hz']<=20:
        raise ValueError('Invalid laser range/frequency')
    if not all(math.isfinite(s[k]) for k in ('angle_min','offset_x','offset_y','yaw','noise_std_m','drop_probability')):
        raise ValueError('Invalid laser metadata')
    if math.hypot(s['offset_x'],s['offset_y'])>.25 or not 0<=s['noise_std_m']<=.2 or not 0<=s['drop_probability']<=1:
        raise ValueError('Invalid laser error or extrinsics')
    bins=data.get('detector_bins',[])
    end=0.
    for b in bins:
        lo,hi=b['range_m']
        if lo!=end or not lo<hi or not 0<=b['recall']<=1:
            raise ValueError('Detector bins must cover contiguous ranges from zero')
        if len(b['bias_m'])!=2 or len(b['std_m'])!=2 or any(not math.isfinite(x) for x in b['bias_m']+b['std_m']) or min(b['std_m'])<0:
            raise ValueError('Invalid detector error fit')
        end=hi
    return data,hashlib.sha256(raw).hexdigest()


def laser_origin(pose,laser):
    c,s=pose[:,2].cos(),pose[:,2].sin()
    x,y=laser['offset_x'],laser['offset_y']
    return pose[:,:2]+torch.stack((c*x-s*y,s*x+c*y),-1)


def raycast(pose,humans,arena,laser,rectangles=None):
    """Nearest disk/room/rectangle intersections. Angles in ROS convention.

    Returns ideal ranges and human hit IDs (-1 for static surfaces).
    A sensor inside geometry returns zero instead of a distant surface.
    """
    origin=laser_origin(pose,laser)
    angle=pose[:,2:]+laser['yaw']+laser['angle_min']+torch.arange(laser['rays'],device=pose.device)* (2*math.pi/laser['rays'])
    direction=torch.stack((angle.cos(),angle.sin()),-1)
    rel=humans[:,None,:,:2]-origin[:,None,None,:]
    projection=(rel*direction[:,:,None,:]).sum(-1)
    d2=rel.square().sum(-1)
    r2=humans[:,None,:,2].square()
    discriminant=r2-(d2-projection.square())
    near=projection-discriminant.clamp_min(0).sqrt()
    inside=d2<=r2
    valid=(discriminant>=0)&(projection>=0)&(near>=0)
    human_distance=torch.where(inside,0.,torch.where(valid,near,float('inf')))
    human_range,idx=human_distance.min(-1)
    boundary=torch.where(direction>=0,arena,-arena)
    div=torch.where(direction.abs()>1e-8,direction,1.)
    wall=torch.where(direction.abs()>1e-8,(boundary-origin[:,None,:])/div,float('inf')).amin(-1).clamp_min(0.)
    if rectangles is not None and rectangles.numel():
        # Ray/slab intersection with axis-aligned closed rectangles.
        o=origin[:,None,None,:]
        d=direction[:,:,None,:]
        lo,hi=rectangles[None,None,:,:2],rectangles[None,None,:,2:]
        parallel=d.abs()<1e-8
        miss=(parallel&((o<lo)|(o>hi))).any(-1)
        inv=torch.where(parallel,1.,d)
        t0,t1=(lo-o)/inv,(hi-o)/inv
        enter=torch.where(parallel,-float('inf'),torch.minimum(t0,t1)).amax(-1)
        leave=torch.where(parallel,float('inf'),torch.maximum(t0,t1)).amin(-1)
        hit=(~miss)&(leave>=enter.clamp_min(0.))
        obstacle=torch.where(hit,enter.clamp_min(0.),float('inf')).amin(-1)
        wall=torch.minimum(wall,obstacle)
    return torch.minimum(human_range,wall),torch.where(human_range<wall,idx,-1)


def disk_clearance(points,radius,arena,rectangles=None):
    """Signed distance from circular body to static geometry."""
    wall=arena-points.abs().amax(-1)-radius
    if rectangles is None or not rectangles.numel():
        return wall
    center=(rectangles[:,:2]+rectangles[:,2:])/2
    half=(rectangles[:,2:]-rectangles[:,:2])/2
    q=(points[...,None,:]-center).abs()-half
    signed=q.clamp_min(0.).norm(dim=-1)+q.amax(-1).clamp_max(0.)
    return torch.minimum(wall,signed.amin(-1)-radius)


def clearance_gate(scan,measured,target,limits,params,laser):
    """Match physical gate's raw scan octant coverage and stopping test."""
    valid=torch.isfinite(scan)&(scan>=laser['range_min'])&(scan<=laser['range_max'])
    coverage=torch.stack([valid[:,i*scan.shape[1]//8:(i+1)*scan.shape[1]//8].float().mean(-1)
                          for i in range(8)],-1).amin(-1)>=.5
    nearest=scan.masked_fill(~valid,float('inf')).amin(-1)
    v=target.sum(-1)*limits.wheel_radius/2
    speed=torch.maximum(v.clamp_min(0.),measured[:,0].abs())
    required=params['footprint']+params['sensor_offset']+params['margin']+speed*params['latency']+speed.square()/(2*params['braking'])
    blocked=(~coverage)|(nearest<=required)|(torch.isfinite(scan)&(scan<laser['range_min'])).any(-1)
    blocked|=(measured[:,0].abs()>.26)|(measured[:,1].abs()>1.82)|(v<0.)
    return torch.where(blocked[:,None],0.,target),blocked


def motor_step(wheels,target,dt,tau,accel,brake,gain,deadband):
    """Asymmetric speed response. Exponential update, bounded acceleration.

    Braking rate applies to slowing down or reversal. Position integration
    uses average wheel speed over the substep, not final velocity alone.
    """
    desired=torch.where(target.abs()<deadband,0.,target*gain)
    slowing=(wheels*desired<=0)|(desired.abs()<wheels.abs())
    bound=torch.where(slowing,brake,accel)*dt
    delta=-torch.expm1(-dt/tau)*(desired-wheels)
    return wheels+torch.maximum(-bound,torch.minimum(bound,delta))


class FidelityDrive:
    def __init__(self,profile,n,device,rng,rectangles=None):
        self.profile,self.n,self.rng=profile,n,rng
        self.rectangles=rectangles
        self.dt=profile['physics_dt']
        self.steps=round(CONTROL_DT/self.dt)
        m=profile['motor']
        for name,key in [('tau','tau_s'),('accel','accel_rad_s2'),('brake','brake_rad_s2'),
                         ('gain','gain'),('radius','radius_m'),('deadband','deadband_rad_s')]:
            setattr(self,name,torch.tensor(m[key],device=device))
        self.track=m['track_m']
        self.delay=round(m['delay_s']/self.dt)
        self.queue=torch.zeros(self.delay+1,n,2,device=device)
        self.cursor=0
        self.laser=profile['laser']
        self.scan=torch.full((n,self.laser['rays']),float('inf'),device=device)
        self.scan_elapsed=torch.full((n,),float('inf'),device=device)
        self.hits=torch.full_like(self.scan,-1,dtype=torch.long)
        self.scan_pose=torch.zeros(n,3,device=device)
        self.scan_humans=None
        self.clock=torch.zeros(n,device=device,dtype=torch.float64)
        self.on_scan=None

    def reset(self,done):
        self.queue.masked_fill_(done[None,:,None],0.)
        self.scan.masked_fill_(done[:,None],float('inf'))
        self.hits.masked_fill_(done[:,None],-1)
        self.scan_elapsed.masked_fill_(done,float('inf'))
        self.clock.masked_fill_(done,0.)

    def twist(self,wheels):
        ground=wheels*self.radius
        return torch.stack((ground.sum(-1)/2,(ground[:,1]-ground[:,0])/self.track),-1)

    def sample_scan(self,pose,humans,arena):
        ranges,ids=raycast(pose,humans,arena,self.laser,self.rectangles)
        if self.laser['noise_std_m']:
            ranges=ranges+torch.randn(ranges.shape,device=ranges.device,generator=self.rng)*self.laser['noise_std_m']
        valid=ranges<=self.laser['range_max']
        if self.laser['drop_probability']:
            valid&=torch.rand(ranges.shape,device=ranges.device,generator=self.rng)>=self.laser['drop_probability']
        return torch.where(valid,ranges.clamp_min(0.),float('inf')),torch.where(valid,ids,-1)

    def update_scan(self,pose,humans,arena,sample):
        if self.scan_humans is None:
            self.scan_humans=torch.zeros_like(humans)
        ranges,ids=self.sample_scan(pose,humans,arena)
        self.scan=torch.where(sample[:,None],ranges,self.scan)
        self.hits=torch.where(sample[:,None],ids,self.hits)
        self.scan_pose=torch.where(sample[:,None],pose,self.scan_pose)
        self.scan_humans=torch.where(sample[:,None,None],humans,self.scan_humans)
        self.scan_elapsed=torch.where(sample,torch.zeros_like(self.scan_elapsed),self.scan_elapsed)
        if self.on_scan is not None:
            self.on_scan(self.scan,self.scan_pose,self.clock,sample)

    def ensure_scan(self,pose,humans,arena):
        self.update_scan(pose,humans,arena,torch.isinf(self.scan_elapsed))

    def advance(self,pose,wheels,target,before,after,arena,limits,params,tau_scale,accel_scale,robot_radius,perception_fault=None):
        initial_heading=pose[:,2]
        rotation=torch.zeros_like(initial_heading)
        blocked_any=torch.zeros_like(initial_heading,dtype=torch.bool)
        clearance=torch.full_like(initial_heading,float('inf'))
        static_clearance=disk_clearance(pose[:,:2],robot_radius,arena,self.rectangles)
        path=torch.zeros_like(clearance)
        for k in range(self.steps):
            humans=before+(after-before)*(k/self.steps)
            sample=self.scan_elapsed>=1/self.laser['hz']-1e-7
            # Per-world phase; compute full batch and replace only due scans.
            self.update_scan(pose,humans,arena,sample)
            self.queue[self.cursor].copy_(target)
            self.cursor=(self.cursor+1)%len(self.queue)
            cmd=self.queue[self.cursor]
            cmd,blocked=clearance_gate(self.scan,self.twist(wheels),cmd,limits,params,self.laser)
            if perception_fault is not None:
                cmd=cmd.masked_fill(perception_fault[:,None],0.)
                blocked|=perception_fault
            blocked_any|=blocked
            new=motor_step(wheels,cmd,self.dt,self.tau*tau_scale[:,None],
                           self.accel*accel_scale[:,None],self.brake*accel_scale[:,None],self.gain,self.deadband)
            v,w=self.twist((wheels+new)/2).unbind(-1)
            dtheta=w*self.dt
            rotation+=dtheta
            mid=pose[:,2]+dtheta/2
            d=v*self.dt*torch.sinc(dtheta/(2*torch.pi))
            xy=pose[:,:2]+d[:,None]*torch.stack((mid.cos(),mid.sin()),-1)
            next_h=before+(after-before)*((k+1)/self.steps)
            a=humans[...,:2]-pose[:,None,:2]
            b=next_h[...,:2]-xy[:,None,:]
            change=b-a
            t=(-(a*change).sum(-1)/change.square().sum(-1).clamp_min(1e-10)).clamp(0.,1.)
            c=(a+t[...,None]*change).norm(dim=-1)-torch.maximum(humans[...,2],next_h[...,2])-robot_radius
            clearance=torch.minimum(clearance,c.amin(-1))
            # Conservative Lipschitz bound over each short body segment.
            c0=disk_clearance(pose[:,:2],robot_radius,arena,self.rectangles)
            c1=disk_clearance(xy,robot_radius,arena,self.rectangles)
            static_clearance=torch.minimum(static_clearance,torch.minimum(c0,c1)-d.abs()/2)
            pose=torch.cat((xy,((pose[:,2]+dtheta)%(2*torch.pi))[:,None]),-1)
            path+=d.abs()
            wheels=new
            self.scan_elapsed+=self.dt
            self.clock+=self.dt
        v=self.twist(wheels)[:,0]
        vel=v[:,None]*torch.stack((pose[:,2].cos(),pose[:,2].sin()),-1)
        return pose,wheels,vel,rotation,blocked_any,clearance,static_clearance,path
