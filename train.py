import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.model import Policy
from rl.networks.storage import RolloutStorage


from crowd_nav.configs.config import Config


def main():
	"""
	main function for training a robot policy network
	"""
	# read arguments
	algo_args = get_args()

	if algo_args.algo != 'ppo' or algo_args.seq_length != algo_args.num_steps:
		raise ValueError("Use PPO and match seq_length to num_steps")
	if algo_args.num_steps<=0 or algo_args.num_processes<=0 or algo_args.num_mini_batch<=0 or algo_args.num_processes % algo_args.num_mini_batch:
		raise ValueError("num_processes must be divisible by num_mini_batch")
	if algo_args.waffle_eval_episodes:
		from waffle_pi.evaluate import run
		return run(algo_args)

	if algo_args.waffle_build_pool:
		from waffle_pi.trajectory_pool import build
		return build(algo_args)

	from waffle_pi.distributed import setup
	rank,world,device = setup(algo_args)
	if world>1 and (not algo_args.waffle_pi or algo_args.compile_policy or not (algo_args.waffle_replay or algo_args.waffle_live_orca)):
		raise ValueError("Distributed path requires Waffle mode; use eager DDP policy initially")
	if algo_args.amp_bf16 and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
		raise ValueError("BF16 requires a supported CUDA device")
	if algo_args.fused_adam and device.type!='cuda':
		raise ValueError("Fused Adam requires CUDA")
	if algo_args.tf32 and device.type=='cuda':
		torch.backends.cuda.matmul.allow_tf32=True
		torch.backends.cudnn.allow_tf32=True
	if (algo_args.cuda_graph_actor or algo_args.cuda_graph_gst) and device.type!='cuda':
		raise ValueError("Explicit CUDA graphs require a CUDA training device")
	if algo_args.cuda_graph_actor and (algo_args.compile_policy or world>1):
		raise ValueError("Explicit actor graph is initially single-GPU and mutually exclusive with compile-policy")
	if algo_args.cuda_graph_gst and (algo_args.compile_gst or world>1):
		raise ValueError("Explicit GST graph is initially single-GPU and mutually exclusive with compile-gst")
	if algo_args.compile_policy:
		algo_args.masked_gru=True
		if not hasattr(torch,'compile'):
			raise RuntimeError("Policy compilation requires PyTorch 2.x")

	# create a directory for saving the logs and weights
	if rank==0:
		if algo_args.waffle_pi and os.path.exists(algo_args.output_dir):
			raise ValueError("Waffle runs require a new output directory")
		if not os.path.exists(algo_args.output_dir):
			os.makedirs(algo_args.output_dir)
		elif not algo_args.overwrite:
			raise ValueError("output_dir already exists")
		save_config_dir = os.path.join(algo_args.output_dir,'configs')
		os.makedirs(save_config_dir,exist_ok=True)
		shutil.copy('crowd_nav/configs/config.py',save_config_dir)
		shutil.copy('crowd_nav/configs/__init__.py',save_config_dir)
		shutil.copy('arguments.py',algo_args.output_dir)
		import json
		with open(os.path.join(algo_args.output_dir,'run_args.json'),'w') as handle:
			json.dump(dict(vars(algo_args),world_size=world),handle,indent=2)


	env_config = config = Config(algo_args)
	config.env.num_processes = algo_args.num_processes
	config.training.device = str(device)
	if rank==0:
		with open(os.path.join(algo_args.output_dir,'run_args.json'),'w') as handle:
			json.dump(dict(vars(algo_args),world_size=world),handle,indent=2)

	# Shared policy initialization; distinct rollout seeds set after DDP broadcasts.
	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	seed = algo_args.seed + rank*100003


	env_name = algo_args.env_name

	if config.sim.render:
		algo_args.num_processes = 1
		algo_args.num_mini_batch = 1

	# for visualization
	if config.sim.render:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	# Create a wrapped, monitored VecEnv
	tensor_env = bool(algo_args.waffle_replay or algo_args.waffle_live_orca)
	if algo_args.waffle_live_orca and algo_args.waffle_replay:
		raise ValueError("Choose either live tensor ORCA or replay")
	if algo_args.waffle_reactive_humans and not algo_args.waffle_live_orca:
		raise ValueError("Reactive humans require live tensor ORCA")
	if tensor_env:
		from waffle_pi.replay import make_replay
		envs = make_replay(config, algo_args, device)
	else:
		import crowd_sim  # register legacy Gym environments
		from rl.networks.envs import make_vec_envs
		envs = make_vec_envs(env_name, seed, algo_args.num_processes,
						 algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)


	# create a policy network
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=algo_args,
		base=config.robot.policy)

	# storage buffer to store the agent's experience
	rollouts = RolloutStorage(algo_args.num_steps,
							  algo_args.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  algo_args.human_node_rnn_size,
							  algo_args.human_human_edge_rnn_size,
								  compact_edges=algo_args.fast_training and config.robot.policy == 'selfAttn_merge_srnn')

	# continue training from an existing model if resume = True
	if algo_args.resume:
		load_path = algo_args.load_path
		from waffle_pi.checkpoints import load_weights
		actor_critic.load_state_dict(load_weights(load_path,'cpu'), strict=True)
		print("Loaded the following checkpoint:", load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	actor_critic.to(device)
	if algo_args.freeze_human_attention:
		for name in ('spatial_attn', 'spatial_linear'):
			module = getattr(actor_critic.base, name, None)
			if module is not None:
				module.requires_grad_(False)

	train_model = None
	if world>1:
		from torch.nn.parallel import DistributedDataParallel
		train_model = DistributedDataParallel(actor_critic,device_ids=[device.index] if device.type=='cuda' else None,
			broadcast_buffers=False,find_unused_parameters=True)
	act = actor_critic.act
	get_value = actor_critic.get_value
	evaluate = actor_critic.evaluate_actions
	if algo_args.compile_policy:
		act = torch.compile(act,mode=algo_args.compile_mode)
		get_value = torch.compile(get_value,mode=algo_args.compile_mode)
		evaluate = torch.compile(evaluate,mode=algo_args.compile_mode)
	if algo_args.cuda_graph_actor:
		from waffle_pi.cuda_graph import StaticGraphCall
		act=StaticGraphCall(actor_critic.act,device,amp_bf16=algo_args.amp_bf16)
	torch.manual_seed(seed)
	# create the ppo optimizer
	agent = ppo.PPO(
		actor_critic,
		algo_args.clip_param,
		algo_args.ppo_epoch,
		algo_args.num_mini_batch,
		algo_args.value_loss_coef,
		algo_args.entropy_coef,
		lr=algo_args.lr,
		eps=algo_args.eps,
		max_grad_norm=algo_args.max_grad_norm,train_model=train_model,evaluate=evaluate,
		amp_bf16=algo_args.amp_bf16,fused_adam=algo_args.fused_adam)



	rollouts.to(device)
	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)


	episode_rewards = deque(maxlen=100)
	if tensor_env:
		running_rewards = torch.zeros(algo_args.num_processes, device=device)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // (algo_args.num_processes*world)
	if num_updates<1:
		raise ValueError('Budget too small for one global rollout')

	if algo_args.benchmark_updates<0 or algo_args.benchmark_warmup<0:
		raise ValueError("Benchmark counts must be nonnegative")
	if algo_args.benchmark_updates:
		num_updates=algo_args.benchmark_warmup+algo_args.benchmark_updates
		benchmark_rows=[]
		def sync_clock():
			if device.type=='cuda':
				torch.cuda.synchronize(device)
			return time.perf_counter()

	# start the training loop
	for j in range(num_updates):
		if algo_args.benchmark_updates:
			start_rollout=sync_clock()
		# schedule learning rate if needed
		if algo_args.use_linear_lr_decay:
			network_utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)

		if tensor_env:
			completed_rewards, completed_masks = [], []

		# step the environment for a few times
		for step in range(algo_args.num_steps):
			# Sample actions
			with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=algo_args.amp_bf16):

				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]
				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
				value, action, action_log_prob, recurrent_hidden_states = act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])

			# if we use real prediction, send predictions to env for rendering
			if config.sim.render and env_name == 'CrowdSimPredRealGST-v0' and env_config.env.use_wrapper:
				# [nenv, max_human_num, 2*(pred_steps+1)] -> [nenv, max_human_num, 2*pred_steps]
				out_pred = rollouts_obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
				# send manager action to all processes
				ack = envs.talk2Env(out_pred)
				assert all(ack)

			if config.sim.render:
				envs.render()
			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)


			if tensor_env:
				running_rewards = running_rewards + reward.squeeze(-1)
				completed_rewards.append(running_rewards)
				completed_masks.append(done)
				running_rewards = torch.where(done, 0., running_rewards)
				masks = (~done).float().unsqueeze(-1)
				bad_masks = torch.ones_like(masks)
			else:
				for info in infos:
					if 'episode' in info.keys():
						episode_rewards.append(info['episode']['r'])
				masks = torch.FloatTensor([[0.] if d else [1.] for d in done])
				bad_masks = torch.FloatTensor([[0.] if 'bad_transition' in info else [1.] for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)
		if tensor_env:
			episode_rewards.extend(torch.stack(completed_rewards)[torch.stack(completed_masks)].cpu().tolist())
		if algo_args.benchmark_updates:
			end_rollout=sync_clock()
		# store the stepped experience to buffer
		with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=algo_args.amp_bf16):
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]
			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()

		# compute advantage and gradient, and update the network parameters
		rollouts.compute_returns(next_value, algo_args.use_gae, algo_args.gamma,
								 algo_args.gae_lambda, algo_args.use_proper_time_limits,algo_args.gae_backend)

		if algo_args.benchmark_updates:
			end_gae=sync_clock()
		value_loss, action_loss, dist_entropy = agent.update(rollouts)
		if algo_args.benchmark_updates:
			end_learner=sync_clock()
			elapsed=torch.tensor([end_rollout-start_rollout,end_gae-end_rollout,end_learner-end_gae],device=device)
			if world>1:
				torch.distributed.all_reduce(elapsed,op=torch.distributed.ReduceOp.MAX)
			if rank==0 and j>=algo_args.benchmark_warmup:
				benchmark_rows.append(elapsed.cpu().tolist())

		rollouts.after_update()

		# save the model for every interval-th episode or for the last epoch
		if rank==0 and (j % algo_args.save_interval == 0
			or j == num_updates - 1):
			save_path = os.path.join(algo_args.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i'%j + ".pt"))

		if rank==0 and j % algo_args.log_interval == 0:
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps * world
			end = time.time()
			if not episode_rewards:
				print("Updates {}, global timesteps {}, FPS {} (no local completed episodes)".format(j,total_num_steps,int(total_num_steps/max(end-start,1e-6))))
				continue
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')):
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='w', header=True, index=False)

	if algo_args.benchmark_updates and rank==0:
		rows=np.asarray(benchmark_rows)
		seconds=float(rows.sum())
		steps=len(rows)*algo_args.num_steps*algo_args.num_processes*world
		result=dict(measured_updates=len(rows),global_transitions=steps,
			transitions_per_sec=steps/seconds,rollout_s=float(rows[:,0].sum()),
			gae_and_bootstrap_s=float(rows[:,1].sum()),learner_s=float(rows[:,2].sum()),
			note='Excludes checkpoint/logging time and warmup; phase sync overhead included')
		print(result)
		with open(os.path.join(algo_args.output_dir,'benchmark.json'),'w') as handle:
			json.dump(result,handle,indent=2)
	envs.close()
	if world>1:
		torch.distributed.destroy_process_group()


if __name__ == '__main__':
	main()
