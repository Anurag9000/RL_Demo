import torch
import torch.nn as nn
import torch.optim as optim


class PPO():
    """ Class for the PPO optimizer """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True, train_model=None, evaluate=None,
                 amp_bf16=False, fused_adam=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.train_model = train_model
        self.evaluate = evaluate or actor_critic.evaluate_actions
        self.amp_bf16 = amp_bf16
        kwargs = {'fused':True} if fused_adam else {}
        self.optimizer = optim.Adam([p for p in actor_critic.parameters() if p.requires_grad], lr=lr, eps=eps, **kwargs)



    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if self.train_model is not None:
            import torch.distributed as dist
            stats = torch.stack((advantages.sum(),advantages.square().sum(),
                                 advantages.new_tensor(advantages.numel())))
            dist.all_reduce(stats)
            mean = stats[0]/stats[2]
            var = (stats[1]-stats[0]*mean)/(stats[2]-1).clamp_min(1)
            advantages = (advantages-mean)/(var.clamp_min(0).sqrt()+1e-5)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=advantages.numel()>1) + 1e-5)

        metrics = torch.zeros(3, device=advantages.device)
        count = 0


        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                evaluate = self.train_model if self.train_model is not None else self.evaluate
                # Keep distribution probabilities and PPO ratios in FP32.
                with torch.autocast(device_type=advantages.device.type, dtype=torch.bfloat16, enabled=self.amp_bf16):
                    values, action_log_probs, dist_entropy, _ = evaluate(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                values,action_log_probs,dist_entropy = values.float(),action_log_probs.float(),dist_entropy.float()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad(set_to_none=True)
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                metrics += torch.stack((value_loss.detach(), action_loss.detach(), dist_entropy.detach()))
                count += 1
        if count == 0:
            raise RuntimeError("No PPO minibatches")
        return tuple((metrics/count).cpu().tolist())
