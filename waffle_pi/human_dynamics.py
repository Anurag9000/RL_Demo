"""Hidden scheduled goal changes for paper-style crowd regimes."""
import math
import numpy as np


def validate_human_dynamics(config):
    if not isinstance(config, dict):
        raise ValueError('human_dynamics must be a mapping')
    regime=config.get('regime')
    if regime not in ('randomized','nonrandomized'):
        raise ValueError('human_dynamics.regime must be randomized or nonrandomized')
    interval=float(config.get('random_goal_change_interval_s',0.0))
    chance=float(config.get('random_goal_change_chance',0.0))
    clearance=float(config.get('goal_clearance_m',0.2))
    if not math.isfinite(interval) or interval<0:
        raise ValueError('random goal interval must be finite and nonnegative')
    if not 0<=chance<=1:
        raise ValueError('random goal change chance must be in [0,1]')
    if 'end_goal_changing' in config or 'end_goal_change_chance' in config:
        raise ValueError('arrival goal changes are unsupported; use the scenario cyclic waypoint route')
    route_policy=config.get('route_policy','cyclic_waypoints')
    if route_policy!='cyclic_waypoints':
        raise ValueError('human_dynamics.route_policy must be cyclic_waypoints')
    if not math.isfinite(clearance) or clearance<0:
        raise ValueError('goal clearance must be finite and nonnegative')
    random_changing=bool(config.get('random_goal_changing',False))
    if random_changing and interval<=0:
        raise ValueError('random goal changes require a positive interval')
    if regime=='nonrandomized' and random_changing:
        raise ValueError('nonrandomized regime cannot enable scheduled random goal changes')
    if regime=='randomized' and not random_changing:
        raise ValueError('randomized regime must enable scheduled random goal changes')
    return dict(regime=regime,
                route_policy=route_policy,
                random_goal_changing=random_changing,
                random_goal_change_interval_s=interval,
                random_goal_change_chance=chance,
                goal_clearance_m=clearance)


def _valid_candidate(candidate,index,positions,goals,radii,robot_position,
                     robot_goal,robot_radius,half_extent,clearance,rectangles):
    margin=float(radii[index])+clearance
    if np.any(np.abs(candidate)>half_extent-margin):
        return False
    if np.linalg.norm(candidate-robot_position)<margin+robot_radius:
        return False
    if np.linalg.norm(candidate-robot_goal)<margin+robot_radius:
        return False
    for j in range(len(positions)):
        if j==index:
            continue
        separation=margin+float(radii[j])
        if np.linalg.norm(candidate-positions[j])<separation:
            return False
        if np.linalg.norm(candidate-goals[j])<separation:
            return False
    for xmin,xmax,ymin,ymax in rectangles:
        if xmin-margin<=candidate[0]<=xmax+margin and ymin-margin<=candidate[1]<=ymax+margin:
            return False
    return True


def update_human_goals(positions,goals,radii,robot_position,robot_goal,
                       robot_radius,arena_half_extent,policy_tick,policy_dt,
                       dynamics,rng,rectangles=(),max_attempts=256):
    """Apply scheduled random changes to hidden simulator goals.

    The caller advances the deterministic scenario waypoint route on arrival.
    Returns ``(goals, changed_count, failed_samples)``. Random changes occur
    only on the configured policy-step cadence. Goal sampling is bounded and rejects human/robot overlaps,
    other humans' destinations, arena-boundary violations and rectangles.
    """
    cfg=validate_human_dynamics(dynamics)
    pos=np.asarray(positions,dtype=np.float64)
    target=np.asarray(goals,dtype=np.float64).copy()
    rad=np.asarray(radii,dtype=np.float64)
    robot=np.asarray(robot_position,dtype=np.float64)
    destination=np.asarray(robot_goal,dtype=np.float64)
    n=len(pos)
    if pos.shape!=(n,2) or target.shape!=(n,2) or rad.shape!=(n,) or n<1:
        raise ValueError('human positions, goals and radii have inconsistent shapes')
    if not all(np.isfinite(x).all() for x in (pos,target,rad,robot,destination)) or (rad<=0).any():
        raise ValueError('human goal inputs must be finite with positive radii')
    robot_radius=float(robot_radius)
    arena_half_extent=float(arena_half_extent)
    policy_dt=float(policy_dt)
    if (robot.shape!=(2,) or destination.shape!=(2,) or
            not all(math.isfinite(x) and x>0 for x in (robot_radius,arena_half_extent,policy_dt))):
        raise ValueError('invalid robot, arena or timestep parameters')
    if isinstance(policy_tick,bool) or int(policy_tick)!=policy_tick or policy_tick<0:
        raise ValueError('policy_tick must be a nonnegative integer')
    if isinstance(max_attempts,bool) or int(max_attempts)!=max_attempts or max_attempts<1:
        raise ValueError('max_attempts must be a positive integer')
    rect=np.asarray(rectangles,dtype=np.float64).reshape(-1,4)
    if not np.isfinite(rect).all() or (rect[:,0]>rect[:,1]).any() or (rect[:,2]>rect[:,3]).any():
        raise ValueError('static rectangle coordinates must be finite')
    interval_steps=round(cfg['random_goal_change_interval_s']/policy_dt) if cfg['random_goal_change_interval_s'] else 0
    if cfg['random_goal_changing'] and (interval_steps<1 or abs(interval_steps*policy_dt-cfg['random_goal_change_interval_s'])>1e-7):
        raise ValueError('random goal interval must be a whole number of policy steps')
    # policy_tick counts completed calls, so call 20 (index 19) changes goals
    # at the 20th 0.25-second boundary.
    random_event=(cfg['random_goal_changing'] and interval_steps>0 and
                  (int(policy_tick)+1)%interval_steps==0)
    changed=failed=0
    for i in range(n):
        want_random=random_event and rng.random()<=cfg['random_goal_change_chance']
        if not want_random:
            continue
        accepted=None
        for _ in range(max_attempts):
            candidate=rng.uniform(-arena_half_extent,arena_half_extent,size=2)
            if _valid_candidate(candidate,i,pos,target,rad,robot,destination,
                                robot_radius,arena_half_extent,cfg['goal_clearance_m'],rect):
                accepted=candidate
                break
        if accepted is None:
            failed+=1
        else:
            target[i]=accepted
            changed+=1
    return target,changed,failed
