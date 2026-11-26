import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].view(T * N, *(_tensor[key].size()[2:]))
        return _tensor
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    """ The rollout buffer to store the agent's experience for PPO """
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 human_node_rnn_size, human_human_edge_rnn_size, compact_edges=False, separate_rnn_masks=False):

        if isinstance(obs_shape, dict):
            self.obs = {}
            for key in obs_shape:
                self.obs[key] = torch.zeros(num_steps + 1, num_processes, *(obs_shape[key].shape))
            self.human_num = obs_shape['spatial_edges'].shape[0]
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)

        self.recurrent_hidden_states = {} # a dict of tuple(hidden state, cell state)

        node_num = 1
        # todo: uncomment the next line for previous models!
        edge_num = 1 if compact_edges else self.human_num + 1
        # edge_num = 2

        self.recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_steps + 1, num_processes, node_num, human_node_rnn_size)
        self.recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_steps + 1, num_processes, edge_num, human_human_edge_rnn_size)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rnn_masks = torch.ones_like(self.masks) if separate_rnn_masks else None

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    @property
    def policy_masks(self):
        return self.masks if self.rnn_masks is None else self.rnn_masks

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key] = self.recurrent_hidden_states[key].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.rnn_masks is not None:
            self.rnn_masks=self.rnn_masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, rnn_masks=None):


        for key in self.obs:
            self.obs[key][self.step + 1].copy_(obs[key])
        for key in recurrent_hidden_states:
            self.recurrent_hidden_states[key][self.step + 1].copy_(recurrent_hidden_states[key])

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if self.rnn_masks is not None:
            self.rnn_masks[self.step+1].copy_(masks if rnn_masks is None else rnn_masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):

        for key in self.obs:
            self.obs[key][0].copy_(self.obs[key][-1])
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key][0].copy_(self.recurrent_hidden_states[key][-1])

        self.masks[0].copy_(self.masks[-1])
        if self.rnn_masks is not None:
            self.rnn_masks[0].copy_(self.rnn_masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True, gae_backend='reference'):
        if use_gae and gae_backend != 'reference':
            self.value_preds[-1].copy_(next_value)
            if gae_backend == 'scan':
                from waffle_pi.fast_ops import gae_scan as calculate
            elif gae_backend == 'triton':
                from waffle_pi.triton_gae import gae_triton as calculate
            else:
                raise ValueError("Unknown GAE backend")
            self.returns[:-1].copy_(calculate(self.rewards,self.value_preds,self.masks,
                self.bad_masks,gamma,gae_lambda,use_proper_time_limits))
            self.returns[-1].copy_(next_value)
            return
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:


            obs_batch = {}
            for key in self.obs:
                obs_batch[key] = self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])[indices]
            recurrent_hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = self.recurrent_hidden_states[key][:-1].view(
                -1, self.recurrent_hidden_states[key].size(-1))[indices]

            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.policy_masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        n = self.rewards.size(1)
        if num_mini_batch <= 0 or n % num_mini_batch:
            raise ValueError("Recurrent batches must divide environments evenly")
        T, N = self.num_steps, n//num_mini_batch
        perm = torch.randperm(n, device=self.rewards.device)
        def gather(t, ids):
            return t.index_select(1, ids).reshape(T*N, *t.shape[2:])
        for ids in perm.split(N):
            obs = {k:gather(v[:-1],ids) for k,v in self.obs.items()}
            hidden = {k:v[0].index_select(0,ids) for k,v in self.recurrent_hidden_states.items()}
            yield (obs, hidden, gather(self.actions,ids), gather(self.value_preds[:-1],ids),
                   gather(self.returns[:-1],ids), gather(self.policy_masks[:-1],ids),
                   gather(self.action_log_probs,ids), gather(advantages,ids))
