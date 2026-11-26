"""Batched constant-twist Waffle Pi transition; no per-world Python loops."""
import math
import torch


def wheel_targets(pose, action, limits):
    # action is the unchanged Gaussian Cartesian action, not the executed twist.
    speed = torch.linalg.vector_norm(action, dim=-1)
    angle = torch.atan2(action[..., 1], action[..., 0]) - pose[..., 2]
    error = torch.atan2(torch.sin(angle), torch.cos(angle))
    v = speed.clamp(max=limits.max_linear)*torch.cos(error).clamp_min(0.)
    w = (limits.heading_gain*error).clamp(-limits.max_angular, limits.max_angular)
    v = torch.where(speed < 1e-4, 0., v)
    w = torch.where(speed < 1e-4, 0., w)
    target = torch.stack((v-limits.wheel_separation*w/2,
                          v+limits.wheel_separation*w/2), -1)/limits.wheel_radius
    target = target/(target.abs().amax(-1, keepdim=True)/limits.max_wheel).clamp_min(1.)
    return target


def native_targets(action,limits):
    v=action[...,0].tanh().clamp_min(0.)*limits.max_linear
    w=action[...,1].tanh()*limits.max_angular
    target=torch.stack((v-limits.wheel_separation*w/2,
                        v+limits.wheel_separation*w/2),-1)/limits.wheel_radius
    return target/(target.abs().amax(-1,keepdim=True)/limits.max_wheel).clamp_min(1.)


def approximate_gate(pose,wheels,target,humans,arena,limits,params):
    # Ideal 360-degree raw range: nearest disk surface or square boundary.
    # Real blind spots, stale scans, nonfinite returns and coverage are not
    # modeled here. The physical gate remains independent and authoritative.
    origin=pose[:,:2]-params['sensor_offset']*torch.stack(
        (pose[:,2].cos(),pose[:,2].sin()),-1)
    ranges=(torch.linalg.vector_norm(humans[...,:2]-origin[:,None,:],dim=-1)-humans[...,2]).amin(-1)
    wall=arena-origin.abs().amax(-1)
    nearest=torch.minimum(ranges,wall)
    v=(target.sum(-1)*limits.wheel_radius/2).clamp_min(0.)
    measured=(wheels.sum(-1)*limits.wheel_radius/2).abs()
    speed=torch.maximum(v,measured)
    clearance=params['footprint']+params['sensor_offset']+params['margin']+speed*params['latency']+speed.square()/(2*params['braking'])
    blocked=nearest<=clearance
    return torch.where(blocked[:,None],0.,target),blocked


def advance(pose,wheels,target,limits,dt,tau_scale=None,accel_scale=None):
    if tau_scale is None:
        alpha = 1. - math.exp(-dt/limits.motor_tau)
        bound = limits.wheel_accel*dt
        delta = (alpha*(target-wheels)).clamp(-bound,bound)
    else:
        alpha = -torch.expm1(-dt/(limits.motor_tau*tau_scale[:,None]))
        bound = limits.wheel_accel*dt*accel_scale[:,None]
        delta = torch.maximum(-bound,torch.minimum(bound,alpha*(target-wheels)))
    wheels = wheels+delta
    v = limits.wheel_radius*wheels.sum(-1)/2
    w = limits.wheel_radius*(wheels[..., 1]-wheels[..., 0])/limits.wheel_separation
    dtheta = w*dt
    mid = pose[..., 2]+dtheta/2
    d = v*dt*torch.sinc(dtheta/(2*torch.pi))
    pose = torch.stack((pose[..., 0]+d*torch.cos(mid), pose[..., 1]+d*torch.sin(mid),
                        (pose[..., 2]+dtheta)%(2*torch.pi)), -1)
    velocity = torch.stack((v*torch.cos(pose[..., 2]), v*torch.sin(pose[..., 2])), -1)
    return pose, wheels, velocity, dtheta


def step(pose,wheels,action,limits,dt):
    return advance(pose,wheels,wheel_targets(pose,action,limits),limits,dt)
