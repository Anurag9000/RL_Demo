"""Deterministic actor evaluation on reference or batched Waffle simulation."""
from collections import Counter
import json
import math
from pathlib import Path
import time
import torch


def run(args):
    from .quality import wilson
    from crowd_nav.configs.config import Config
    from rl.networks.model import Policy
    if not args.waffle_pi or args.waffle_eval_episodes < 1:
        raise ValueError("Use --waffle-pi and a positive evaluation count")
    from gazebo_sim.hardware import bind_rank
    bind_rank()
    torch.set_num_threads(args.num_threads)
    if args.cuda:
        torch.backends.cuda.matmul.allow_tf32=bool(args.tf32)
        torch.backends.cudnn.allow_tf32=bool(args.tf32)
    config=Config(args)
    tensor=bool(args.waffle_live_orca or args.waffle_replay or args.waffle_gazebo)
    if args.waffle_live_orca and args.waffle_replay:
        raise ValueError("Select one tensor environment")
    b=min(args.num_processes,args.waffle_eval_episodes) if tensor else 1
    args.num_processes=config.env.num_processes=b
    config.args=args
    device=torch.device('cuda' if args.cuda else 'cpu')
    config.training.device=str(device)
    torch.manual_seed(args.seed)
    if args.waffle_gazebo:
        from gazebo_sim.vector_env import make_gazebo
        env=make_gazebo(config,args,device)
    elif tensor:
        from .replay import make_replay
        env=make_replay(config,args,device)
    else:
        import crowd_sim
        from rl.networks.envs import make_vec_envs
        env=make_vec_envs(args.env_name,args.seed,1,args.gamma,None,device,
                         True,config=config,pretext_wrapper=True)
    try:
        model=Policy(env.observation_space.spaces,env.action_space,
                     base=config.robot.policy,base_kwargs=args).to(device)
        from .checkpoints import load_weights
        from .native import check_checkpoint,profile_contract
        check_checkpoint(args.load_path,args.waffle_native_action,profile_contract(args) if args.waffle_native_action else None)
        model.load_state_dict(load_weights(args.load_path,device),strict=True)
        model.eval()
        obs=env.reset()
        hidden={'human_node_rnn':torch.zeros(b,1,args.human_node_rnn_size,device=device),
                'human_human_edge_rnn':torch.zeros(b,1,args.human_human_edge_rnn_size,device=device)}
        masks=torch.zeros(b,1,device=device)
        outcomes,rewards=Counter(),[]
        total=torch.zeros(b,device=device)
        steps=0
        interventions=0
        counted_transitions=0
        episode_metrics=[]
        scenario_ids=set()
        travel=torch.zeros(b,device=device)
        turns=torch.zeros_like(travel)
        duration=torch.zeros_like(travel)
        intrusion=torch.zeros_like(travel)
        minimum=torch.full_like(travel,float('inf'))
        minimum_wall=torch.full_like(travel,float('inf'))
        initial_distance=torch.zeros_like(travel)
        gate_ticks=torch.zeros_like(travel)
        quota=torch.full((b,),args.waffle_eval_episodes//b,device=device,dtype=torch.long)
        quota[:args.waffle_eval_episodes%b]+=1
        completed=torch.zeros_like(quota)
        if args.waffle_gazebo:
            env.venv.set_eval_quotas(quota.detach().cpu().numpy())
        start=time.perf_counter()
        limit=args.waffle_eval_episodes*(int(config.env.time_limit/config.env.time_step)+5)
        with torch.no_grad():
            while len(rewards)<args.waffle_eval_episodes:
                if steps>=limit:
                    raise RuntimeError("Evaluation exceeded maximum transition rounds")
                _,action,_,hidden=model.act(obs,hidden,masks,deterministic=True)
                obs,reward,done,infos=env.step(action)
                total=total+reward.to(device).reshape(b)
                done=torch.as_tensor(done,device=device,dtype=torch.bool)
                masks=(~done).float().unsqueeze(-1)
                if args.waffle_gazebo:
                    masks=masks*(~env.venv.policy_reset).float().unsqueeze(-1)
                active=completed<quota
                counted_transitions+=int(active.sum().item())
                accepted=done&active
                if tensor:
                    if args.waffle_native_action:
                        interventions+=int((env.venv.intervened&active).sum().item())
                    venv=env.venv
                    initial_distance=torch.where(duration==0,venv.last_goal_distance,initial_distance)
                    duration+=config.env.time_step
                    travel+=venv.last_path
                    turns+=venv.last_turn
                    intrusion+=(venv.last_clearance<config.reward.discomfort_dist).float()*config.env.time_step
                    minimum=torch.minimum(minimum,venv.last_clearance)
                    minimum_wall=torch.minimum(minimum_wall,venv.last_static_clearance)
                    gate_ticks+=venv.intervened.float()
                    codes=venv.outcome
                    for i in accepted.nonzero().flatten().tolist():
                        scenario_ids.add(venv.last_scenario_ids[i])
                        shortest=float(initial_distance[i])
                        actual=float(travel[i])
                        success=int(codes[i])==1
                        episode=dict(success=success,path_m=actual,time_s=float(duration[i]),
                            rotation_rad=float(turns[i]),min_human_clearance_m=float(minimum[i]),
                            min_wall_clearance_m=float(minimum_wall[i]),intrusion_s=float(intrusion[i]),
                            gate_fraction=float(gate_ticks[i])*config.env.time_step/float(duration[i]),
                            straight_line_spl=float(success)*shortest/max(shortest,actual,1e-8),
                            final_goal_distance_m=float(venv.last_goal_distance_after[i]))
                        if args.waffle_gazebo:
                            paper_itr=float(venv.paper_itr[i])
                            paper_sd=float(venv.paper_sd_m[i])
                            episode.update(paper_itr=paper_itr,
                                paper_itr_percent=100.0*paper_itr,
                                paper_sd_m=paper_sd if math.isfinite(paper_sd) else None,
                                paper_intrusion_steps=int(venv.paper_intrusion_steps[i]),
                                paper_episode_steps=int(venv.paper_episode_steps[i]),
                                paper_sd_sum_m=float(venv.paper_sd_sum_m[i]),
                                paper_sd_count=int(venv.paper_sd_count[i]),
                                scenario_id=venv.last_scenario_ids[i])
                        episode_metrics.append(episode)
                    for t in (travel,turns,duration,intrusion,initial_distance,gate_ticks):
                        t.masked_fill_(done,0.)
                    minimum.masked_fill_(done,float('inf'))
                    minimum_wall.masked_fill_(done,float('inf'))
                    finished=list(zip(codes[accepted].cpu().tolist(),total[accepted].cpu().tolist()))
                    for code,value in finished:
                        outcomes[{1:'ReachGoal',2:'Collision',3:'Timeout'}[code]]+=1
                        rewards.append(value)
                elif bool(done[0]):
                    outcomes[type(infos[0]['info']).__name__]+=1
                    rewards.append(float(total[0]))
                completed+=accepted.long()
                total=torch.where(done,0.,total)
                steps+=1
        elapsed=time.perf_counter()-start
        result={'episodes':len(rewards),'outcomes':dict(outcomes),
                'success_rate':outcomes['ReachGoal']/len(rewards),
                'collision_rate':outcomes['Collision']/len(rewards),
                'timeout_rate':outcomes['Timeout']/len(rewards),
                'mean_reward':sum(rewards)/len(rewards),'seed':args.seed,
                'gate_intervention_fraction':interventions/counted_transitions if args.waffle_native_action else None,
                'backend':'gazebo' if args.waffle_gazebo else 'live_tensor' if args.waffle_live_orca else 'replay' if args.waffle_replay else 'reference',
                'transitions_per_sec':steps*b/elapsed,'counted_transitions':counted_transitions,'elapsed_s':elapsed}
        result['success_ci95']=wilson(outcomes['ReachGoal'],len(rewards))
        result['collision_ci95']=wilson(outcomes['Collision'],len(rewards))
        if episode_metrics:
            import numpy as np
            result['scene_split']=args.waffle_scene_split
            result['unique_scenarios']=len(scenario_ids)
            result['episode_metrics']=episode_metrics
            for k in ('path_m','time_s','rotation_rad','intrusion_s','straight_line_spl','final_goal_distance_m'):
                result['mean_'+k]=float(np.mean([e[k] for e in episode_metrics]))
            result['mean_all_episode_path_m']=result['mean_path_m']
            result['mean_all_episode_time_s']=result['mean_time_s']
            from .paper_metrics import aggregate_success_navigation_metrics
            result.update(aggregate_success_navigation_metrics(episode_metrics))
            result['paper_navigation_metric_definition']='NT and PL are mean simulated time and Gazebo truth path length over successful episodes only; both are null when there are no successes. mean_time_s and mean_path_m retain all-outcome means.'
            for k in ('min_human_clearance_m','min_wall_clearance_m'):
                result['p05_'+k]=float(np.quantile([e[k] for e in episode_metrics],.05))
            result['gate_episode_fraction']=sum(e['gate_fraction']>0 for e in episode_metrics)/len(episode_metrics)
            if args.waffle_gazebo:
                from .paper_metrics import aggregate_intrusion_metrics
                pooled=aggregate_intrusion_metrics([{
                    'paper_itr':e['paper_itr'],
                    'paper_intrusion_steps':e['paper_intrusion_steps'],
                    'paper_episode_steps':e['paper_episode_steps'],
                    'paper_sd_sum_m':e['paper_sd_sum_m'],
                } for e in episode_metrics])
                result['mean_paper_itr']=pooled['paper_itr']
                result['paper_sd_intrusion_steps']=pooled['paper_intrusion_steps']
                result['mean_paper_sd_m']=pooled['paper_sd_m']
                result['paper_metric_definition']='ITR is the unweighted mean across episodes of each episode fraction whose robot center intrudes into any human ground-truth position from t+1 through t+5; SD is the pooled mean current-time closest-human center distance over intrusion timesteps.'
                from .offline_metrics import score_trace_directory,verify_and_apply_offline_metrics
                trace_dir=env.venv.eval_trace_dir
                offline=score_trace_directory(trace_dir)
                paper_itr,paper_sd=verify_and_apply_offline_metrics(episode_metrics,offline,pooled)
                result['mean_paper_itr']=paper_itr
                result['mean_paper_itr_percent']=100.0*paper_itr
                result['mean_paper_sd_m']=paper_sd
                result['paper_trace_dir']=str(trace_dir)
                result['paper_trace_count']=offline['trace_count']
                result['paper_trace_offline_verification']={
                    'passed':True,'mean_episode_itr':offline['mean_episode_itr'],
                    'pooled_sd_m':offline['pooled_sd_m'],'intrusion_steps':offline['intrusion_steps'],
                    'episode_steps':offline['episode_steps']}
        print(json.dumps(result,indent=2,allow_nan=False))
        if args.waffle_eval_json:
            path=Path(args.waffle_eval_json)
            if path.exists():
                raise ValueError("Choose a new evaluation JSON path")
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        env.close()
