"""Offline parallel RVO2 trajectory generation for a non-reactive crowd.

The robot is stationary during generation. Upstream respawn/goal sampling
can depend on its location, so replay is an approximate experimental backend,
not an exact replacement for live ORCA. Each scenario retains its robot start
and goal. No pickle is used for the resulting pool.
"""
import multiprocessing as mp
import json
from pathlib import Path
import numpy as np

_ENV = None
_STEPS = None


def _init(args):
    global _ENV, _STEPS
    from crowd_nav.configs.config import Config
    from crowd_sim.envs.crowd_sim_pred_real_gst import CrowdSimPredRealGST
    cfg = Config(args)
    _ENV = CrowdSimPredRealGST()
    _ENV.configure(cfg)
    _ENV.phase = 'train'
    _ENV.nenv = 1
    _STEPS = int(round(cfg.env.time_limit/cfg.env.time_step))+1


def _scenario(seed):
    env = _ENV
    env.thisSeed = seed
    env.case_counter['train'] = 0
    env.reset()
    start = np.array([env.robot.px, env.robot.py, env.robot.theta], np.float32)
    goal = np.array([env.robot.gx, env.robot.gy], np.float32)
    humans = np.empty((_STEPS, env.human_num, 3), np.float32)
    for t in range(_STEPS):
        humans[t] = [(h.px,h.py,h.radius) for h in env.humans]
        if t+1 < _STEPS:
            # No vector wrapper here: keep advancing even after robot done.
            env.step(np.zeros(2, dtype=np.float32))
    return start,goal,humans


def build(args):
    from crowd_nav.configs.config import Config
    cfg = Config(args)
    if not args.waffle_pi or cfg.robot.visible or cfg.sim.human_num_range:
        raise ValueError("Replay requires a Waffle profile and fixed non-reactive crowd")
    if args.waffle_pool_count < 1 or args.num_processes < 1:
        raise ValueError("Positive pool count and worker count required")
    out = Path(args.waffle_build_pool)
    if out.suffix != '.npz' or out.exists():
        raise ValueError("Choose a new .npz pool file")
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed, args.seed+args.waffle_pool_count)
    with mp.get_context('fork').Pool(args.num_processes, initializer=_init,initargs=(args,)) as pool:
        rows = list(pool.imap(_scenario, seeds, chunksize=4))
    starts, goals, humans = map(np.stack, zip(*rows))
    meta = json.dumps(dict(dt=cfg.env.time_step, timeout=cfg.env.time_limit,
                           arena=cfg.sim.arena_size, humans=cfg.sim.human_num,
                           radius=cfg.robot.radius, seed=args.seed, count=len(rows)))
    np.savez(out, starts=starts, goals=goals, humans=humans, metadata=meta)
    print("Saved", out, humans.shape)
