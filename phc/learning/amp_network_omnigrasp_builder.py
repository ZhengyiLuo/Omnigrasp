
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


class AMPOmniGraspBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPOmniGraspBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.fixed_latent = self.task_obs_size_detail['fixed_latent']
            self.res_hand = self.task_obs_size_detail.get("res_hand", False)
            
            self.use_image_obs = self.task_obs_size_detail.get("use_image_obs", False)
            self.camera_config = self.task_obs_size_detail.get("camera_config", {})
            
            self.use_part = params['use_part']
            if self.use_part:
                self.body_embedding_size = int(kwargs['actions_num'] * 2/3)
                self.left_embedding_size = int(kwargs['actions_num'] * 1/6)
                self.right_embedding_size = int(kwargs['actions_num'] * 1/6)
                
            if self.res_hand:
                self.res_hand_dim = self.task_obs_size_detail['res_hand_dim']
            
            if self.fixed_latent: # latent code does not change
                self.env_to_obj_code = self.task_obs_size_detail['env_to_obj_code'] # Always aligned input size
                self.obj_embedding_input_size = self.env_to_obj_code.shape[-1] 
                self.obj_embedding_output_size = 256 # hard coded for now
                kwargs['input_shape'] = (self.self_obs_size + self.task_obs_size - 1 + self.obj_embedding_output_size,)  # -1 for the env id. 
                
                
            super().__init__(params, **kwargs)
            
            if self.res_hand:
                self.sigma[-self.res_hand_dim:] = -3.5
            
            if self.fixed_latent:
                self._build_z_mlp()
                
                if self.separate:
                    self._build_critic_z_mlp()
                    
                self.running_mean = kwargs['mean_std'].running_mean
                self.running_var = kwargs['mean_std'].running_var
            
        def load(self, params):
            super().load(params)
            self._task_units = params['task_mlp']['units']
            self._task_activation = params['task_mlp']['activation']
            self._task_initializer = params['task_mlp']['initializer']
            return
        
        def _build_critic_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self.obj_embedding_input_size  # target

            self.critic_z_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.critic_z_mlp = self._build_mlp(**mlp_args)
            
            if not self.has_rnn:
                self.critic_z_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=self.obj_embedding_output_size))
            else:
                self.critic_z_proj_linear = nn.Linear(in_features=self._task_units[-1], out_features=self.obj_embedding_output_size)
            
            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.critic_z_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return
        
        def _build_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self.obj_embedding_input_size  # target

            self.z_reader_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.z_reader_mlp = self._build_mlp(**mlp_args)
            self.z_reader_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=self.obj_embedding_output_size))
            
            
            if self.use_part:
                part_input_size = self.units[-1] if self.is_rnn_before_mlp else self.rnn_units
                mlp_args = {'input_size': part_input_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                self.body_mlp = self._build_mlp(**mlp_args)
                self.body_mlp.append(nn.Linear(in_features=self._task_units[0], out_features=self.body_embedding_size))
                
                mlp_args = {'input_size': part_input_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                self.left_mlp = self._build_mlp(**mlp_args)
                self.left_mlp.append(nn.Linear(in_features=self._task_units[0], out_features=self.left_embedding_size))
                
                mlp_args = {'input_size': part_input_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                self.right_mlp = self._build_mlp(**mlp_args)
                self.right_mlp.append(nn.Linear(in_features=self._task_units[0], out_features=self.right_embedding_size))
                

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.z_reader_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return
            
        def eval_critic(self, obs_dict):
            obs = obs_dict['obs']
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)
            if self.fixed_latent:
                env_ids = obs[:, -1].long() # no longer part of the filtering
                obs = obs[:, :-1]
                
            
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(-1, c_out.size(-1))
            
            if self.fixed_latent:
                obj_embedding = self.critic_z_mlp(self.env_to_obj_code[env_ids])
                c_out = torch.cat((c_out, obj_embedding), dim=1)
            
            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    c_out_in = c_out
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = c_out.size()[0]
                num_seqs = batch_size // seq_length
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                ################# New RNN
                if len(states) == 2:
                    c_states = states[1].reshape(num_seqs, seq_length, -1)
                else:
                    c_states = states[2:].reshape(num_seqs, seq_length, -1)
                c_out, c_states = self.c_rnn(c_out, c_states[:, 0:1].transpose(0, 1).contiguous()) # ZL: only pass the first state, others are ignored. ???            
                
                ################# Old RNN
                # if len(states) == 2:	
                #     c_states = states[1]	
                # else:	
                #     c_states = states[2:]	
                # c_out, c_states = self.c_rnn(c_out, c_states)
                
                
                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if type(c_states) is not tuple:
                    c_states = (c_states,)

                if self.is_rnn_before_mlp:
                    c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                c_out = self.critic_mlp(c_out)

                value = self.value_act(self.value(c_out))
                return value
        
        
        

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            
            if self.fixed_latent:
                env_ids = obs[:, -1].long() # no longer part of the filtering
                obs = obs[:, :-1]
            
            a_out = self.actor_cnn(obs)  # This is empty
            # a_out = a_out.contiguous().view(a_out.size(0), -1)
            if self.fixed_latent:
                obj_embedding = self.z_reader_mlp(self.env_to_obj_code[env_ids])
                a_out = torch.cat((obs, obj_embedding), dim=1)
            
            
            if self.has_rnn:
                if not self.is_rnn_before_mlp: # set to is_rnn_before_mlp = False, so usually rnn is later than the MLP. 
                    
                    a_out_in = a_out
                    a_out = self.actor_mlp(a_out_in)
                    
                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)

                ################# New RNN
                if len(states) == 2:
                    a_states = states[0].reshape(num_seqs, seq_length, -1)
                else:
                    a_states = states[:2].reshape(num_seqs, seq_length, -1)
                a_out, a_states = self.a_rnn(a_out, a_states[:, 0:1].transpose(0, 1).contiguous())
                
                ################ Old RNN
                # if len(states) == 2:	
                #     a_states = states[0]	
                # else:	
                #     a_states = states[:2]	
                # a_out, a_states = self.a_rnn(a_out, a_states)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)

                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)

                if type(a_states) is not tuple:
                    a_states = (a_states,)

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, a_states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, a_states

                if self.is_continuous:
                    if self.use_part:
                        body_out = self.body_mlp(a_out)
                        left_out = self.left_mlp(a_out)
                        right_out = self.right_mlp(a_out)
                        mu_out = torch.cat((body_out, left_out, right_out), dim=1)
                        mu = self.mu_act(mu_out)
                    else:
                        mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, a_states

            else:
                a_out = self.actor_mlp(a_out)
                
                # mlp_out = self.actor_mlp(a_out[:1])
                # (self.actor_mlp(a_out[:5])[0] - self.actor_mlp(a_out[:2])[0]).abs()


                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, 

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, 

                if self.is_continuous:
                    
                    if self.use_part:    
                        
                        body_out = self.body_mlp(a_out)
                        left_out = self.left_mlp(a_out)
                        right_out = self.right_mlp(a_out)
                        mu_out = torch.cat((body_out, left_out, right_out), dim=1)
                        mu = self.mu_act(mu_out)
                        
                    else:
                        mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    
                    return mu, sigma
                    # return torch.round(mu, decimals=3), sigma

            return