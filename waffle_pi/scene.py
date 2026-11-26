"""Surveyed planar geometry and explicit scenario banks.

Coordinates are in metres, in one local map frame. Rectangles are closed,
axis-aligned static solids; the enclosing square is also solid. Human goals
are ordered cyclic waypoints. This is not a general polygonal map planner.
"""
import torch


def scene_tensors(profile,arena,humans,robot_radius,device,split='train',require_measured=False):
    from .fidelity import disk_clearance
    scene=profile.get('scene')
    if scene is None:
        return None
    if scene['arena_half_extent_m']!=arena:
        raise ValueError('Scene arena differs from simulator')
    rect=torch.tensor(scene['rectangles_m'],device=device,dtype=torch.float32).reshape(-1,4)
    if not torch.isfinite(rect).all() or (rect[:,:2]>=rect[:,2:]).any() or (rect.abs()>arena).any():
        raise ValueError('Invalid surveyed rectangles')
    scenarios=scene['scenarios']
    if not scenarios:
        raise ValueError('Empty scenario bank')
    # Use only explicitly supplied layouts for this crowd size. No silent
    # resizing or truncation of surveyed scenarios during curriculum.
    if require_measured:
        import json
        signatures=set()
        ids=set()
        for s in scenarios:
            if s.get('split') not in ('train','validation','test') or not s.get('id') or s['id'] in ids:
                raise ValueError('Measured scene requires unique IDs and explicit splits')
            ids.add(s['id'])
            signature=json.dumps({k:s[k] for k in ('robot_start','robot_goal','humans')},sort_keys=True)
            if signature in signatures:
                raise ValueError('Duplicate measured scenario across bank')
            signatures.add(signature)
        for name in ('train','validation','test'):
            if not any(s['split']==name and len(s['humans'])==humans for s in scenarios):
                raise ValueError('Measured scene missing '+name+' cases for crowd size')
    cases=[s for s in scenarios if len(s['humans'])==humans and s.get('split','train')==split]
    if not cases:
        raise ValueError('No surveyed scenario for requested crowd size')
    starts=torch.tensor([s['robot_start'] for s in cases],device=device)
    goals=torch.tensor([s['robot_goal'] for s in cases],device=device)
    pos=torch.tensor([[h['start'] for h in s['humans']] for s in cases],device=device)
    rad=torch.tensor([[h['radius'] for h in s['humans']] for s in cases],device=device)
    speed=torch.tensor([[h['speed'] for h in s['humans']] for s in cases],device=device)
    # All pedestrians have a fixed number of waypoints to enable batching.
    routes=torch.tensor([[h['waypoints'] for h in s['humans']] for s in cases],device=device)
    if starts.shape!=(len(cases),3) or goals.shape!=(len(cases),2) or pos.shape!=(len(cases),humans,2) or routes.ndim!=4 or routes.shape[:2]!=pos.shape[:2] or routes.shape[-1]!=2 or routes.shape[2]<2:
        raise ValueError('Invalid scene tensor shapes')
    if not all(torch.isfinite(t).all() for t in (starts,goals,pos,rad,speed,routes)) or (rad<=0).any() or (speed<=0).any():
        raise ValueError('Invalid scenario values')
    if (disk_clearance(starts[:,:2],robot_radius,arena,rect)<=0).any() or (disk_clearance(goals,robot_radius,arena,rect)<=0).any():
        raise ValueError('Robot start/goal intersects geometry')
    if (disk_clearance(pos,rad,arena,rect)<=0).any() or (disk_clearance(routes,rad[:,:,None],arena,rect)<=0).any():
        raise ValueError('Pedestrian start/waypoint intersects geometry')
    pair=(pos[:,:,None]-pos[:,None]).norm(dim=-1)-rad[:,:,None]-rad[:,None,:]
    if (pair.masked_fill(torch.eye(humans,device=device,dtype=torch.bool)[None],float('inf'))<=0).any():
        raise ValueError('Overlapping pedestrian starts')
    if ((pos-starts[:,None,:2]).norm(dim=-1)<=rad+robot_radius).any():
        raise ValueError('Robot overlaps pedestrian start')
    # Check the straight route segments with a conservative distance bound.
    # Waypoints must describe unobstructed pedestrian corridors.
    route_start=torch.cat((pos[:,:,None],routes),2)
    route_end=torch.cat((routes,routes[:,:,:1]),2)
    length=(route_end-route_start).norm(dim=-1)
    count=max(1,int(torch.ceil(length.max()/.025).item()))
    u=torch.linspace(0,1,count+1,device=device)
    samples=route_start[:,:,:,None]+u[None,None,None,:,None]*(route_end-route_start)[:,:,:,None]
    if (disk_clearance(samples,rad[:,:,None,None],arena,rect)<=.0125).any():
        raise ValueError('Pedestrian route intersects surveyed geometry')
    return dict(rectangles=rect,starts=starts.float(),goals=goals.float(),pos=pos.float(),
                radius=rad.float(),speed=speed.float(),routes=routes.float())


def constrain_human_step(old,new,radius,arena,rectangles):
    """Conservative static collision rejection, not wall ORCA."""
    from .fidelity import disk_clearance
    distance=(new-old).norm(dim=-1)
    clearance=torch.minimum(disk_clearance(old,radius,arena,rectangles),
                            disk_clearance(new,radius,arena,rectangles))-distance/2
    return torch.where((clearance>0)[...,None],new,old)
