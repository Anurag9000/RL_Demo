"""Shared differential-drive action adapter. SI units."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DriveLimits:
    wheel_radius: float = 0.033
    wheel_separation: float = 0.287
    max_linear: float = 0.26
    max_angular: float = 1.82
    # Provisional limits, not measured hardware calibration.
    max_wheel: float = 0.26 / 0.033
    heading_gain: float = 2.0
    motor_tau: float = 0.15
    wheel_accel: float = 12.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in vars(self).values()):
            raise ValueError("Limits must be finite and positive")
        if self.max_linear > 0.26 or self.max_angular > 1.82:
            raise ValueError("Exceeds nominal Waffle Pi body limits")


def cartesian_to_wheels(action, heading, limits):
    vx, vy = map(float, action)
    if not all(map(math.isfinite, (vx, vy, heading))):
        raise ValueError("Non-finite action or heading")
    speed = math.hypot(vx, vy)
    if speed < 1e-4:
        return 0., 0.
    e = math.atan2(math.sin(math.atan2(vy, vx)-heading),
                   math.cos(math.atan2(vy, vx)-heading))
    v = min(speed, limits.max_linear)*max(0., math.cos(e))
    w = max(-limits.max_angular, min(limits.max_angular, limits.heading_gain*e))
    l = (v-limits.wheel_separation*w/2)/limits.wheel_radius
    r = (v+limits.wheel_separation*w/2)/limits.wheel_radius
    scale = max(1., abs(l)/limits.max_wheel, abs(r)/limits.max_wheel)
    return l/scale, r/scale


def wheels_to_twist(wheels, limits):
    l, r = wheels
    return (limits.wheel_radius*(l+r)/2,
            limits.wheel_radius*(r-l)/limits.wheel_separation)


def command_twist(action, heading, limits):
    return wheels_to_twist(cartesian_to_wheels(action, heading, limits), limits)


def advance_wheels(actual, target, dt, limits):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time step")
    alpha = -math.expm1(-dt/limits.motor_tau)
    bound = limits.wheel_accel*dt
    return tuple(a+max(-bound, min(bound, alpha*(t-a))) for a,t in zip(actual,target))


def integrate_pose(pose, twist, dt):
    x,y,theta = pose
    v,w = twist
    half = w*dt/2
    sinc = math.sin(half)/half if abs(half)>1e-8 else 1.
    d = v*dt*sinc
    return x+d*math.cos(theta+half), y+d*math.sin(theta+half), (theta+2*half)%(2*math.pi)
