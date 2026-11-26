"""Zero-shot or fine-tuned Waffle simulation checkpoint evaluation."""
from collections import Counter
import json
import torch


def run(args):
    from crowd_nav.configs.config import Config
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy
    if not args.waffle_pi or args.waffle_eval_episodes < 1:
        raise ValueError("Use --waffle-pi and a positive evaluation count")
    config = Config()
    args.num_processes = config.env.num_processes = 1
    device = torch.device('cuda' if args.cuda else 'cpu')
    config.training.device = str(device)
    torch.manual_seed(args.seed)
    env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, None, device,
                       True, config=config, pretext_wrapper=True)
    try:
        model = Policy(env.observation_space.spaces, env.action_space,
                       base=config.robot.policy, base_kwargs=args).to(device)
        model.load_state_dict(torch.load(args.load_path, map_location=device), strict=True)
        model.eval()
        obs = env.reset()
        hidden = {'human_node_rnn':torch.zeros(1,1,args.human_node_rnn_size,device=device),
                  'human_human_edge_rnn':torch.zeros(1,config.sim.human_num+1,args.human_human_edge_rnn_size,device=device)}
        masks = torch.ones(1,1,device=device)
        outcomes, rewards, total = Counter(), [], 0.
        with torch.no_grad():
            while len(rewards) < args.waffle_eval_episodes:
                _, action, _, hidden = model.act(obs, hidden, masks, deterministic=True)
                obs, reward, done, infos = env.step(action)
                total += float(reward[0])
                masks.fill_(0. if done[0] else 1.)
                if done[0]:
                    outcomes[type(infos[0]['info']).__name__] += 1
                    rewards.append(total)
                    total = 0.
        print(json.dumps({'episodes':len(rewards),'outcomes':dict(outcomes),
                          'mean_reward':sum(rewards)/len(rewards)},indent=2))
    finally:
        env.close()
