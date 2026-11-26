"""Explicit navigation objective and evaluation statistics; no safety certification."""
import math
import torch


def validate_reward(reward):
    keys=('progress','time','turn','intervention','human_clearance','wall_clearance','human_margin','wall_margin')
    if set(reward)!=set(keys) or any(not math.isfinite(reward[k]) or reward[k]<0 for k in keys):
        raise ValueError('Reward requires finite nonnegative explicit coefficients')
    if reward['progress']<=0 or reward['human_margin']<=0 or reward['wall_margin']<=0:
        raise ValueError('Positive progress and clearance margins required')


def navigation_reward(old_dist,dist,human_clearance,wall_clearance,rotation,intervened,dt,reward):
    # Terms are additive, unlike the released discomfort/progress switch.
    # Rotation cost uses integrated squared yaw-rate (per-second coefficient).
    return (reward['progress']*(old_dist-dist)-reward['time']*dt
            -reward['turn']*rotation.square()/dt
            -reward['intervention']*intervened.float()*dt
            -reward['human_clearance']*(reward['human_margin']-human_clearance).clamp_min(0.)*dt
            -reward['wall_clearance']*(reward['wall_margin']-wall_clearance).clamp_min(0.)*dt)


def wilson(success,n,z=1.959963984540054):
    if n<1 or not 0<=success<=n:
        raise ValueError('Invalid binomial count')
    p=success/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0.,center-half),min(1.,center+half)
