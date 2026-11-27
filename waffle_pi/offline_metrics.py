"""Independent paper-metric recomputation from immutable Gazebo truth traces."""
from pathlib import Path
import json
import numpy as np


def score_trace_file(path):
    """Recompute paper ITR/SD without calling the online metric implementation."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as trace:
        robot=np.asarray(trace['robot_positions'],dtype=np.float64)
        humans=np.asarray(trace['human_positions'],dtype=np.float64)
        radii=np.asarray(trace['human_radii'],dtype=np.float64)
        human_ids=np.asarray(trace['human_ids']).astype(str)
        robot_times=np.asarray(trace['robot_times'],dtype=np.float64)
        human_times=np.asarray(trace['human_times'],dtype=np.float64)
        robot_radius=float(trace['robot_radius'])
        future_steps=int(trace['future_steps'])
        episode_steps=int(trace['episode_steps'])
        metadata={k:trace[k].item() for k in ('scenario_id','scenario_index','split','seed','lane','episode_number','profile_sha256','checkpoint_sha256','detector_sha256','model_sdf_sha256','worker_sha256') if k in trace}
        online_itr=float(trace['online_itr']) if 'online_itr' in trace else None
        online_sd_valid=bool(trace['online_sd_valid']) if 'online_sd_valid' in trace else None
        online_sd=float(trace['online_sd_m']) if online_sd_valid else None
    if robot.ndim!=2 or robot.shape[1]!=2 or not np.isfinite(robot).all():
        raise ValueError(f'{path}: invalid robot truth trace')
    if humans.ndim!=3 or humans.shape[2]!=2 or not np.isfinite(humans).all():
        raise ValueError(f'{path}: invalid human truth trace')
    if episode_steps<1 or robot.shape[0]!=episode_steps+1 or future_steps<1:
        raise ValueError(f'{path}: inconsistent episode length/horizon')
    if humans.shape[1]!=len(radii) or humans.shape[1]!=len(human_ids) or humans.shape[1]<1 or len(set(human_ids.tolist()))!=len(human_ids) or not np.isfinite(radii).all() or (radii<=0).any():
        raise ValueError(f'{path}: invalid human radii/IDs/slots')
    if not np.isfinite(robot_radius) or robot_radius<=0 or humans.shape[0]<episode_steps+future_steps:
        raise ValueError(f'{path}: incomplete future truth horizon')
    if robot_times.shape!=(episode_steps+1,) or human_times.shape!=(humans.shape[0],) or not np.isfinite(robot_times).all() or not np.isfinite(human_times).all():
        raise ValueError(f'{path}: invalid simulator timestamps')
    if np.any(np.diff(robot_times)<=0) or np.any(np.diff(human_times)<=0) or not np.allclose(human_times[:len(robot_times)],robot_times,rtol=0.,atol=1e-9):
        raise ValueError(f'{path}: nonmonotonic or misaligned simulator timestamps')
    overlap_limit=radii+robot_radius
    event_count=0
    social_distance_sum=0.0
    for decision in range(episode_steps):
        robot_xy=robot[decision]
        future_xy=humans[decision+1:decision+1+future_steps]
        future_separation=np.sqrt(np.sum((future_xy-robot_xy)**2,axis=-1))
        if np.any(future_separation<overlap_limit[None,:]):
            current_separation=np.sqrt(np.sum((humans[decision]-robot_xy)**2,axis=-1))
            social_distance_sum+=float(np.min(current_separation))
            event_count+=1
    result=dict(path=str(path),scenario_id=str(metadata.get('scenario_id','')),
        scenario_index=int(metadata.get('scenario_index',-1)),split=str(metadata.get('split','')),
        seed=int(metadata.get('seed',-1)),
        profile_sha256=str(metadata.get('profile_sha256','')),
        checkpoint_sha256=str(metadata.get('checkpoint_sha256','')),
        detector_sha256=str(metadata.get('detector_sha256','')),
        model_sdf_sha256=str(metadata.get('model_sdf_sha256','')),
        worker_sha256=str(metadata.get('worker_sha256','')),episode_steps=episode_steps,
        human_ids=human_ids.tolist(),robot_time_start=float(robot_times[0]),
        robot_time_end=float(robot_times[-1]),human_time_end=float(human_times[-1]),
        intrusion_steps=event_count,itr=event_count/episode_steps,
        sd_sum_m=social_distance_sum,sd_count=event_count,
        sd_m=social_distance_sum/event_count if event_count else None,
        online_itr=online_itr,online_sd_m=online_sd)
    if online_itr is not None:
        result['online_parity']=(abs(result['itr']-online_itr)<=1e-12 and
            ((online_sd is None and result['sd_m'] is None) or
             (online_sd is not None and result['sd_m'] is not None and abs(result['sd_m']-online_sd)<=1e-6)))
    return result


def score_trace_directory(directory):
    files=sorted(Path(directory).glob('*.npz'))
    if not files:
        raise ValueError(f'no trace .npz files in {directory}')
    episodes=[score_trace_file(p) for p in files]
    events=sum(x['intrusion_steps'] for x in episodes)
    sd_sum=sum(x['sd_sum_m'] for x in episodes)
    return dict(trace_count=len(episodes),episodes=episodes,
        mean_episode_itr=float(np.mean([x['itr'] for x in episodes])),
        intrusion_steps=events,episode_steps=sum(x['episode_steps'] for x in episodes),
        pooled_sd_m=sd_sum/events if events else None,
        all_online_parity=all(x.get('online_parity',False) for x in episodes))


def verify_and_apply_offline_metrics(episode_metrics, offline, online_aggregate, atol=1e-6):
    """Check float32 online aggregates against traces, then publish trace scores."""
    if len(episode_metrics)!=offline['trace_count'] or len(offline['episodes'])!=len(episode_metrics):
        raise RuntimeError('saved truth trace count does not match completed episodes')
    if not offline['all_online_parity']:
        raise RuntimeError('per-episode online/offline paper-metric parity check failed')
    if abs(float(offline['mean_episode_itr'])-float(online_aggregate['paper_itr']))>atol:
        raise RuntimeError('offline and online ITR aggregates disagree')
    online_sd=online_aggregate['paper_sd_m']
    offline_sd=offline['pooled_sd_m']
    if (online_sd is None)!=(offline_sd is None) or (online_sd is not None and abs(float(online_sd)-float(offline_sd))>atol):
        raise RuntimeError('offline and online SD aggregates disagree')
    by_scenario={}
    for scored in offline['episodes']:
        scenario_id=scored['scenario_id']
        if not scenario_id or scenario_id in by_scenario:
            raise RuntimeError('offline traces have missing or duplicate scenario IDs')
        by_scenario[scenario_id]=scored
    seen=set()
    for episode in episode_metrics:
        scenario_id=episode.get('scenario_id')
        if scenario_id not in by_scenario or scenario_id in seen:
            raise RuntimeError('completed episode IDs do not exactly match unique offline traces')
        seen.add(scenario_id)
        scored=by_scenario[scenario_id]
        episode.update(paper_itr=scored['itr'],paper_sd_m=scored['sd_m'],
            paper_intrusion_steps=scored['intrusion_steps'],paper_episode_steps=scored['episode_steps'],
            paper_sd_sum_m=scored['sd_sum_m'],paper_sd_count=scored['sd_count'])
    if seen!=set(by_scenario):
        raise RuntimeError('offline traces contain cases absent from evaluator results')
    return offline['mean_episode_itr'],offline['pooled_sd_m']


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir',required=True)
    parser.add_argument('--output',default=None)
    args=parser.parse_args(argv)
    result=score_trace_directory(args.trace_dir)
    payload=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        out=Path(args.output)
        if out.exists(): raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(payload+'\n')
    print(payload)


if __name__=='__main__':
    main()
