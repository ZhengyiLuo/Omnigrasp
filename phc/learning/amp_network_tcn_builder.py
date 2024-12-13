
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from phc.learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext
from phc.learning.tcn import TemporalConvNet
from phc.learning.kin_tcn import TemporalModelOptimized1f
DISC_LOGIT_INIT_SCALE = 1.0


class AMPTCNBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPTCNBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.fut_tracks = self.task_obs_size_detail['fut_tracks']
            self.obs_v = self.task_obs_size_detail['obs_v']
            self.num_traj_samples = self.task_obs_size_detail['num_traj_samples']
            self.track_bodies = self.task_obs_size_detail['track_bodies']
            out_features = 256
            kwargs['input_shape'] = (self.self_obs_size + out_features,)  #

            super().__init__(params, **kwargs)
            self.tcn = TemporalModelOptimized1f(in_features = self.task_obs_size // self.num_traj_samples, out_features = out_features, filter_widths = [3, 3], causal=False, dropout=0.2, channels=1024)

            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var
            
        def eval_critic(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:]
            task_obs_steps = task_obs.view(task_obs.size(0), self.num_traj_samples, -1)
            tcn_out = self.tcn(task_obs_steps)
            c_out = torch.cat((self_obs, tcn_out[:, 0]), dim=1)
            c_out = self.critic_mlp(c_out)
            value = self.value(c_out)
            return value

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']

            a_out = self.actor_cnn(obs)  # This is empty
            # a_out = a_out.contiguous().view(a_out.size(0), -1)
            
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:]
            task_obs_steps = task_obs.view(task_obs.size(0), self.num_traj_samples, -1)
            tcn_out = self.tcn(task_obs_steps)
            
            a_out = torch.cat((self_obs, tcn_out[:, 0]), dim=1)
            a_out = self.actor_mlp(a_out)
            

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return
