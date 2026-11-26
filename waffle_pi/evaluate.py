"""Deterministic actor evaluation on reference or batched Waffle simulation."""
from collections import Counter
import json
from pathlib import Path
import time
import torch


def run(args):
    from crowd_nav.configs.config import Config
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy
    if not args.waffle_pi or args.waffle_eval_episodes < 1:
        raise ValueError("Use --waffle-pi and a positive evaluation count")
    config=Config(args)
    tensor=bool(args.waffle_live_orca or args.waffle_replay)
    if args.waffle_live_orca and args.waffle_replay:
        raise ValueError("Select one tensor environment")
    b=min(args.num_processes,args.waffle_eval_episodes) if tensor else 1
    args.num_processes=config.env.num_processes=b
    config.args=args
    device=torch.device('cuda' if args.cuda else 'cpu')
    config.training.device=str(device)
    torch.manual_seed(args.seed)
    if tensor:
        from .replay import make_replay
        env=make_replay(config,args,device)
    else:
        env=make_vec_envs(args.env_name,args.seed,1,args.gamma,None,device,
                         True,config=config,pretext_wrapper=True)
    try:
        model=Policy(env.observation_space.spaces,env.action_space,
                     base=config.robot.policy,base_kwargs=args).to(device)
        model.load_state_dict(torch.load(args.load_path,map_location=device),strict=True)
        model.eval()
        obs=env.reset()
        hidden={'human_node_rnn':torch.zeros(b,1,args.human_node_rnn_size,device=device),
                'human_human_edge_rnn':torch.zeros(b,1,args.human_human_edge_rnn_size,device=device)}
        masks=torch.zeros(b,1,device=device)
        outcomes,rewards=Counter(),[]
        total=torch.zeros(b,device=device)
        steps=0
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
                if tensor:
                    codes=env.venv.outcome
                    finished=list(zip(codes[done].cpu().tolist(),total[done].cpu().tolist()))
                    for code,value in finished[:args.waffle_eval_episodes-len(rewards)]:
                        outcomes[{1:'ReachGoal',2:'Collision',3:'Timeout'}[code]]+=1
                        rewards.append(value)
                elif bool(done[0]):
                    outcomes[type(infos[0]['info']).__name__]+=1
                    rewards.append(float(total[0]))
                total=torch.where(done,0.,total)
                steps+=1
        elapsed=time.perf_counter()-start
        result={'episodes':len(rewards),'outcomes':dict(outcomes),
                'success_rate':outcomes['ReachGoal']/len(rewards),
                'collision_rate':outcomes['Collision']/len(rewards),
                'timeout_rate':outcomes['Timeout']/len(rewards),
                'mean_reward':sum(rewards)/len(rewards),'seed':args.seed,
                'backend':'live_tensor' if args.waffle_live_orca else 'replay' if args.waffle_replay else 'reference',
                'transitions_per_sec':steps*b/elapsed,'elapsed_s':elapsed}
        print(json.dumps(result,indent=2))
        if args.waffle_eval_json:
            path=Path(args.waffle_eval_json)
            if path.exists():
                raise ValueError("Choose a new evaluation JSON path")
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        env.close()
