"""Conservative command gate logic, independent of the neural policy.

Not a certified safety system. Clearances, braking, latency and sensor
coverage require supervised measurement. Fails closed on invalid input.
"""
import math
from .dynamics import DriveLimits, wheels_to_twist


def safe_command(v, w, ranges, range_min, range_max, *, fresh, armed,
                 estop, footprint=0.25, sensor_offset=0.064,
                 margin=0.15, latency=0.3, braking=0.3, measured_speed=0., measured_angular=0.):
    params = (v,w,range_min,range_max,footprint,sensor_offset,margin,latency,braking,measured_speed,measured_angular)
    if (not fresh or not armed or estop or not all(map(math.isfinite,params))
            or range_min < 0 or range_max <= range_min
            or min(footprint,margin,braking) <= 0 or min(sensor_offset,latency)<0):
        return 0.,0.
    # Initial commissioning caps. Reject reverse or invalid command.
    if v < 0:
        return 0.,0.
    v,w = min(v,0.10),max(-0.50,min(w,0.50))
    valid = []
    for x in ranges:
        if math.isfinite(x):
            if x < range_min:
                return 0.,0.
            if x <= range_max:
                valid.append(x)
    if not valid:
        return 0.,0.
    # All-around clearance: deliberately conservative, including rotation.
    speed=max(v,abs(measured_speed))
    # Bound translation conservatively using measured speed, not just the
    # requested command. The circumscribed footprint covers rotation.
    clearance=footprint+sensor_offset+margin+speed*latency+speed*speed/(2*braking)
    if abs(measured_speed)>.26 or abs(measured_angular)>1.82:
        return 0.,0.
    if min(valid) <= clearance:
        return 0.,0.
    limits = DriveLimits()
    l = (v-limits.wheel_separation*w/2)/limits.wheel_radius
    r = (v+limits.wheel_separation*w/2)/limits.wheel_radius
    scale = max(1.,abs(l)/limits.max_wheel,abs(r)/limits.max_wheel)
    return wheels_to_twist((l/scale,r/scale),limits)


def scan_coverage(ranges,angle_increment,range_min,range_max,min_fraction=.5):
    """Reject partial, sparse or invalid scan coverage conservatively."""
    if (len(ranges)<32 or not all(map(math.isfinite,(angle_increment,range_min,range_max,min_fraction)))
            or not 0<min_fraction<=1 or range_min<0 or range_max<=range_min
            or abs(angle_increment)*len(ranges)<2*math.pi-.10):
        return False
    # Reject unknown returns when they occupy too much of any octant.
    for i in range(8):
        section=ranges[i*len(ranges)//8:(i+1)*len(ranges)//8]
        if sum(math.isfinite(x) and range_min<=x<=range_max for x in section)/len(section)<min_fraction:
            return False
    return True
