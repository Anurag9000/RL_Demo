"""Native Waffle policy contract and hardware gate approximation.

Gaussian samples remain latent z in the PPO buffer. A deterministic
environment transformation maps z to forward-only velocity and yaw rate.
No Jacobian is needed when PPO is defined on the latent action distribution.
"""
import json
import math
from pathlib import Path

COMMISSION_V = .10
COMMISSION_W = .50


def native_twist(action,limits):
    z0,z1=map(float,action)
    if not all(map(math.isfinite,(z0,z1))):
        raise ValueError("Non-finite native action")
    return limits.max_linear*max(0.,math.tanh(z0)),limits.max_angular*math.tanh(z1)


def reset_head(model):
    import torch
    with torch.no_grad():
        torch.nn.init.orthogonal_(model.dist.fc_mean.weight,gain=.01)
        model.dist.fc_mean.bias.copy_(model.dist.fc_mean.bias.new_tensor([.5,0.]))
        for p in model.dist.logstd.parameters():
            p.fill_(-.7)


def check_checkpoint(path,native,expected=None):
    """Never silently reinterpret a Cartesian checkpoint as native or vice versa."""
    manifest=Path(path).resolve().parent.parent/'run_args.json'
    data=json.loads(manifest.read_text()) if manifest.is_file() else {}
    recorded=bool(data.get('waffle_native_action',False))
    if native!=recorded:
        raise ValueError("Checkpoint action contract differs from requested mode")
    if native and expected:
        for key,value in expected.items():
            if data.get(key)!=value:
                raise ValueError("Checkpoint profile mismatch: "+key)
    return data


def safety_parameters(path,radius):
    defaults=dict(footprint=radius,sensor_offset=.064,margin=.15,latency=.5,braking=.3)
    if path:
        data=json.loads(Path(path).read_text())
        if data.get('schema')!='waffle_safety_commissioning_v1':
            raise ValueError("Expected safety commissioning schema")
        defaults={k:float(data[k]) for k in defaults}
    if any(not math.isfinite(x) or x<=0 for x in defaults.values()) or defaults['latency']<.5 or defaults['footprint']<.20:
        raise ValueError("Invalid stopping model")
    if defaults['footprint']>radius:
        raise ValueError("Simulation radius must cover safety footprint")
    return defaults


def profile_contract(args):
    result={k:getattr(args,k) for k in ('waffle_speed','waffle_angular_speed',
        'waffle_radius','waffle_sensor_range','waffle_humans','gst_adjacent_mask')}
    if getattr(args,'waffle_fidelity_sha',None):
        result['waffle_fidelity_sha']=args.waffle_fidelity_sha
    return result
