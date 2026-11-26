"""Synthetic detector observation model; calibrate before real use."""
import math
import torch


class Perception:
    def __init__(self,args,b,n,device,rng):
        self.args,self.device,self.rng=args,device,rng
        self.fidelity=None
        self.bias=torch.zeros(b,2,device=device)
        self.dropped=torch.zeros(b,n,device=device,dtype=torch.bool)
        self.delay=args.waffle_detection_delay
        if not 0<=self.delay<=8 or not 0<=args.waffle_dropout<=1:
            raise ValueError("Invalid detector delay/dropout")
        if min(args.waffle_noise,args.waffle_pose_drift)<0:
            raise ValueError("Noise must be nonnegative")
        self.history=torch.zeros(self.delay+1,b,n,2,device=device)
        self.masks=torch.zeros(self.delay+1,b,n,device=device,dtype=torch.bool)
        self.cursor=0

    def reset(self,done):
        self.bias.masked_fill_(done[:,None],0.)
        self.dropped.masked_fill_(done[:,None],False)
        self.history.masked_fill_(done[None,:,None,None],0.)
        self.masks.masked_fill_(done[None,:,None],False)

    def observe(self,pose,humans,limits,robot_radius,sensor_range):
        a=self.args
        if a.waffle_pose_drift:
            self.bias+=torch.randn(self.bias.shape,device=self.device,generator=self.rng)*a.waffle_pose_drift
        center=pose[:,:2]
        # Nominal rear-offset laser origin.
        scan=center-.064*torch.stack((pose[:,2].cos(),pose[:,2].sin()),-1)
        if self.fidelity is not None:
            from .fidelity import laser_origin
            scan=laser_origin(pose,self.fidelity['laser'])
        rel=humans[...,:2]-scan[:,None,:]
        dist=torch.linalg.vector_norm(rel,dim=-1)
        visible=dist<=sensor_range
        if self.fidelity is not None:
            from .fidelity import raycast
            laser=self.fidelity['laser']
            ranges,ids=raycast(pose,humans,a.waffle_arena,laser)
            # Geometric hit eligibility; not an actual DR-SPAAM inference.
            hit=(ids[:,:,None]==torch.arange(humans.shape[1],device=pose.device)[None,None,:])
            visible&=(hit&(ranges[:,:,None]>=laser['range_min'])&(ranges[:,:,None]<=laser['range_max'])).any(1)
        elif a.waffle_occlusion:
            # A target center ray intersects a nearer disk. No leg geometry.
            u=rel/dist.clamp_min(1e-6)[...,None]
            proj=(u[:,:,None,:]*rel[:,None,:,:]).sum(-1)
            perp2=(dist[:,None,:].square()-proj.square()).clamp_min(0.)
            rad=humans[:,None,:,2]
            blocked=(proj>0)&(proj<dist[:,:,None])&(perp2<rad.square())
            eye=torch.eye(rel.shape[1],device=self.device,dtype=torch.bool)
            visible=visible&~(blocked&~eye[None]).any(-1)
        if a.waffle_dropout:
            # A drop burst persists with probability 0.5, otherwise restart.
            u=torch.rand(visible.shape,device=self.device,generator=self.rng)
            self.dropped=u<torch.where(self.dropped,0.5+0.5*a.waffle_dropout,a.waffle_dropout)
            visible=visible&~self.dropped
        measured=humans[...,:2]
        if self.fidelity is not None and self.fidelity.get('detector_bins'):
            eligible=torch.zeros_like(visible)
            for b in self.fidelity['detector_bins']:
                lo,hi=b['range_m']
                group=(dist>=lo)&(dist<hi)
                hit=torch.rand(dist.shape,device=pose.device,generator=self.rng)<b['recall']
                eligible|=group&hit
                bias=torch.as_tensor(b['bias_m'],device=pose.device)
                std=torch.as_tensor(b['std_m'],device=pose.device)
                # Detector residuals specified in laser-local axes.
                noise=torch.randn(measured.shape,device=pose.device,generator=self.rng)*std+bias
                angle=pose[:,2:]+self.fidelity['laser']['yaw']
                c,s=angle.cos(),angle.sin()
                dx,dy=noise.unbind(-1)
                world=torch.stack((c*dx-s*dy,s*dx+c*dy),-1)
                measured=torch.where(group[...,None],measured+world,measured)
            visible&=eligible
        if a.waffle_noise:
            measured=measured+torch.randn(measured.shape,device=self.device,generator=self.rng)*a.waffle_noise*(1+dist[...,None]/sensor_range)
        self.history[self.cursor].copy_(measured)
        self.masks[self.cursor].copy_(visible)
        self.cursor=(self.cursor+1)%(self.delay+1)
        delayed=self.history[self.cursor]
        valid=self.masks[self.cursor]
        estimated=center+self.bias
        relative=delayed-center[:,None,:]
        # Detection and robot share odometry error; GST receives absolute
        # detections in the estimated frame via robot node + displacement.
        return estimated,torch.where(valid[...,None],relative,15.),valid
