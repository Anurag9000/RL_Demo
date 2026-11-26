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
                 amp_bf16=False, fused_adam=False,cuda_graph_ppo=False, adaptive_kl=0., lr_min=1e-6, lr_max=3e-4):

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
        self.adaptive_kl,self.lr_min,self.lr_max=adaptive_kl,lr_min,lr_max
        self.last_kl=0.
        kwargs = {'fused':True} if fused_adam else {}
        self.params=[p for p in actor_critic.parameters() if p.requires_grad]
        if cuda_graph_ppo:
            if train_model is not None or not all(p.is_cuda for p in self.params):
                raise ValueError("Captured learner requires one CUDA device")
            kwargs['capturable']=True
        self.optimizer = optim.Adam(self.params,lr=lr,eps=eps,**kwargs)
        self.captured=None
        self.capture_mode=cuda_graph_ppo
        if cuda_graph_ppo:
            from waffle_pi.cuda_learner import StaticLearner
            self.captured=StaticLearner(self._loss,self.optimizer,self.params,
                                        max_grad_norm,next(actor_critic.parameters()).device)

    def _loss(self,sample):
        obs,hidden,actions,old_values,returns,masks,old_logp,advantages=sample
        evaluate=self.train_model if self.train_model is not None else self.evaluate
        with torch.autocast(device_type=advantages.device.type,dtype=torch.bfloat16,
                            enabled=self.amp_bf16,cache_enabled=not self.capture_mode):
            values,logp,entropy,_=evaluate(obs,hidden,masks,actions)
        values,logp,entropy=values.float(),logp.float(),entropy.float()
        ratio=torch.exp(logp-old_logp)
        action_loss=-torch.minimum(ratio*advantages,
                         ratio.clamp(1.-self.clip_param,1.+self.clip_param)*advantages).mean()
        if self.use_clipped_value_loss:
            clipped=old_values+(values-old_values).clamp(-self.clip_param,self.clip_param)
            value_loss=.5*torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
        else:
            value_loss=.5*(returns-values).square().mean()
        total=value_loss*self.value_loss_coef+action_loss-entropy*self.entropy_coef
        logratio=(logp-old_logp).detach()
        kl=(logratio.exp()-1.-logratio).mean()
        stats=torch.stack((value_loss.detach(),action_loss.detach(),entropy.detach(),kl))
        return total,stats



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

        metrics = torch.zeros(4, device=advantages.device)
        count = 0


        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                if self.captured is not None:
                    stats=self.captured(sample)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss,stats=self._loss(sample)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.params,self.max_grad_norm)
                    self.optimizer.step()
                metrics += stats
                count += 1
        if count == 0:
            raise RuntimeError("No PPO minibatches")
        result=metrics/count
        if self.train_model is not None:
            dist.all_reduce(result)
            result/=dist.get_world_size()
        result=result.cpu()
        if not torch.isfinite(result).all():
            raise RuntimeError('Non-finite PPO loss; no checkpoint written for this update')
        self.last_kl=float(result[3])
        if self.adaptive_kl:
            for group in self.optimizer.param_groups:
                lr=group['lr']
                if self.last_kl>1.5*self.adaptive_kl:
                    lr/=1.5
                elif self.last_kl<self.adaptive_kl/1.5:
                    lr*=1.5
                group['lr']=min(self.lr_max,max(self.lr_min,lr))
        return tuple(result[:3].tolist())
