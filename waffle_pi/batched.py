"""Batched constant-twist Waffle Pi transition; no per-world Python loops."""
import math
import torch


def step(pose, wheels, action, limits, dt):
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
    alpha = 1. - math.exp(-dt/limits.motor_tau)
    wheels = wheels + (alpha*(target-wheels)).clamp(-limits.wheel_accel*dt, limits.wheel_accel*dt)
    v = limits.wheel_radius*wheels.sum(-1)/2
    w = limits.wheel_radius*(wheels[..., 1]-wheels[..., 0])/limits.wheel_separation
    dtheta = w*dt
    mid = pose[..., 2]+dtheta/2
    d = v*dt*torch.sinc(dtheta/(2*torch.pi))
    pose = torch.stack((pose[..., 0]+d*torch.cos(mid), pose[..., 1]+d*torch.sin(mid),
                        (pose[..., 2]+dtheta)%(2*torch.pi)), -1)
    velocity = torch.stack((v*torch.cos(pose[..., 2]), v*torch.sin(pose[..., 2])), -1)
    return pose, wheels, velocity, dtheta
