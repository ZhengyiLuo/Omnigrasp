from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from phc.learning.amp_network_builder import AMPBuilder
from phc.learning.network_builder import init_mlp
import torch
import torch.nn as nn
import numpy as np
from phc.utils.torch_utils import project_to_norm
from phc.learning.vq_quantizer import EMAVectorQuantizer, Quantizer
from phc.utils.flags import flags
DISC_LOGIT_INIT_SCALE = 1.0


class AMPZPartBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPZPartBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']

            self.proj_norm = self.task_obs_size_detail["proj_norm"]
            self.embedding_size = self.task_obs_size_detail['embedding_size']
            self.embedding_norm = self.task_obs_size_detail['embedding_norm']
            self.z_readout = self.task_obs_size_detail.get("z_readout", False)
            self.z_type = self.task_obs_size_detail.get("z_type", "sphere")
            self.dict_size = self.task_obs_size_detail.get("dict_size", 512)
            self.z_all = self.task_obs_size_detail.get("z_all", False)
            self.embedding_partion = self.task_obs_size_detail.get("embedding_partion", 1)
            
            self.use_vae_prior = self.task_obs_size_detail.get("use_vae_prior", False)
            self.use_vae_fixed_prior = self.task_obs_size_detail.get("use_vae_fixed_prior", False)
            self.use_vae_clamped_prior = self.task_obs_size_detail.get("use_vae_clamped_prior", False)
            self.use_vae_sphere_prior = self.task_obs_size_detail.get("use_vae_sphere_prior", False)
            self.use_vae_sphere_posterior = self.task_obs_size_detail.get("use_vae_sphere_posterior", False)
            self.vae_prior_fixed_logvar = self.task_obs_size_detail.get("vae_prior_fixed_logvar", 0)
            self.vae_var_clamp_max = self.task_obs_size_detail.get("vae_var_clamp_max", 0)
            
            self.embedding_size_body = int(self.embedding_size * 2/3)
            self.embedding_size_part = int(self.embedding_size * 1/6)
            self.self_obs_left_hand_idxes = self.task_obs_size_detail['self_obs_left_hand_idxes']
            self.self_obs_right_hand_idxes = self.task_obs_size_detail['self_obs_right_hand_idxes']
            self.im_obs_right_hand_idxes = self.task_obs_size_detail['im_obs_right_hand_idxes']
            self.im_obs_left_hand_idxes = self.task_obs_size_detail['im_obs_left_hand_idxes']
            
            self.dof_names = self.task_obs_size_detail['dof_names']
            self.left_hand_names = self.task_obs_size_detail['left_hand_names']
            self.right_hand_names = self.task_obs_size_detail['right_hand_names']
            
            dof_names_expand = sum([[n + "_x", n + "_z", n + "_y"] for n in self.dof_names], [])
            body_names_expand = sum([[n + "_x", n + "_z", n + "_y"] for n in self.dof_names if (not n in self.left_hand_names and not n in self.right_hand_names)], [])
            left_names_expand = sum([[n + "_x", n + "_z", n + "_y"] for n in self.left_hand_names], [])
            right_names_expand = sum([[n + "_x", n + "_z", n + "_y"] for n in self.right_hand_names], [])
            self.body_name_idxes = [dof_names_expand.index(n) for n in body_names_expand]
            self.left_dof_idxes = [dof_names_expand.index(n) for n in left_names_expand]
            self.right_dof_idxes = [dof_names_expand.index(n) for n in right_names_expand]
            
            
            ##### Debug utils
            flags.idx = 0
            self.debug_idxes = [0] * self.embedding_partion
            
            if self.z_all:
                kwargs['input_shape'] = (self.embedding_size,)  # Task embedding size + self_obs
            else:
                kwargs['input_shape'] = (kwargs['self_obs_size'] + self.embedding_size,)  # Task embedding size + self_obs
                
            super().__init__(params, **kwargs)
            
            ##### Override mu 
            self.mu = nn.Linear(in_features=self.units[-1], out_features=kwargs['actions_num'] - (len(self.left_hand_names) + len(self.right_hand_names)) * 3)
            
            
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

            self._build_z_mlp()
            
            if self.z_readout:
                self._build_z_reader()
            if self.separate:
                self._build_critic_z_mlp()
                
                
        def load(self, params):
            super().load(params)
            self._task_units = params['task_mlp']['units']
            self._task_activation = params['task_mlp']['activation']
            self._task_initializer = params['task_mlp']['initializer']
            
            self._part_units = params['part_mlp']['units']
            self._part_activation = params['part_mlp']['activation']
            self._part_initializer = params['part_mlp']['initializer']
            return
        
        def form_embedding(self, task_out_z, obs_dict = None):
            extra_dict = {}
            B, N = task_out_z.shape
            
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:]
            
            if self.z_type == 'vae':
                
                vae_mu = self.z_mu(task_out_z)
                vae_log_var = self.z_logvar(task_out_z)
                
                if self.use_vae_clamped_prior:
                    vae_log_var = torch.clamp(vae_log_var, min = -5, max = self.vae_var_clamp_max)
                    
                
                if "z_noise"  in obs_dict and self.training: # bypass reparatzation and use the noise sampled during training. 
                    task_out_proj = vae_mu + torch.exp(0.5*vae_log_var) * obs_dict['z_noise']
                else:
                    task_out_proj, self.z_noise = self.reparameterize(vae_mu, vae_log_var)
                    
                if flags.test:
                    task_out_proj = vae_mu
                    
                if flags.trigger_input:
                    flags.trigger_input = False
                    flags.debug = not flags.debug
                    
                if flags.debug:
                    if self.use_vae_prior or self.use_vae_fixed_prior:
                        prior_mu, prior_logvar = self.compute_prior(obs_dict)
                        # if flags.trigger_input:
                        #     ### Trigger input
                        #     task_out_proj[:], noise = self.reparameterize(prior_mu, prior_logvar) ; print("\n   debugging",  end='')
                        #     flags.trigger_input = False
                        # else:
                        #     task_out_proj[:] = prior_mu
                        # task_out_proj[:], noise = self.reparameterize(prior_mu, torch.ones_like(prior_logvar) * -2.3 ) ; print("\r  debugging with prior using -2.3 std.",  end='')
                        # task_out_proj[:], noise = self.reparameterize(prior_mu, torch.ones_like(prior_logvar) * -1.5 ) ; print("\r  debugging with prior using -1.5 std.",  end='')
                        task_out_proj, noise = self.reparameterize(prior_mu, prior_logvar ) ; print(f"\r prior_mu {prior_mu.abs().max():.3f} {prior_logvar.exp().max():.3f}",  end='')
                        # task_out_proj[:] = torch.randn_like(vae_mu) ; print("\r   debugging randn",  end='')
                        
                        task_out_proj[:, :40] = vae_mu[:, :40]
                        enhance = 0
                    else:
                        task_out_proj[:] = torch.randn_like(vae_mu) ; print("\r   debugging",  end='')
                        
                if self.use_vae_sphere_posterior:
                    task_out_proj = project_to_norm(task_out_proj, norm=self.embedding_norm, z_type="sphere")
                
                extra_dict = {"vae_mu": vae_mu, "vae_log_var": vae_log_var, "noise": self.z_noise}
                
                
                # prior_mu, prior_logvar = self.compute_prior(obs_dict)
                # print(prior_logvar.exp().max())
                # np.set_printoptions(precision=4, suppress=1)
                # if "prev_task_out_proj" in self.__dict__:
                #     diff = self.prev_task_out_proj.cpu().numpy() - task_out_proj.cpu().numpy()
                #     print(f"{np.abs(diff).max():.4f}", diff)
                #     if np.abs(diff).max() > 0.5:
                #         import ipdb; ipdb.set_trace()
                #         print('...')
                # self.prev_task_out_proj = task_out_proj
                
                # prior_mu, prior_logvar = self.compute_prior(obs_dict)
                # import ipdb; ipdb.set_trace()
                # print(prior_mu.abs().argmax(), prior_mu.abs().max(), task_out_proj.abs().argmax(), task_out_proj.abs().max())
                
                # print(task_out_proj.abs().max(), task_out_proj.abs().argmax(), task_out_proj.cpu().numpy())
                # torch.exp(prior_logvar * 0.5), torch.exp(0.5 * vae_log_var)
                # print(torch.exp(prior_logvar * 0.5).mean(), torch.exp(0.5 * vae_log_var).mean())
                # print(task_out_proj.abs().max(), (prior_mu - task_out_proj).abs().cpu().numpy().max(), (prior_mu - task_out_proj).cpu().numpy())
                # import ipdb; ipdb.set_trace()
                
                
            elif self.z_type == "sphere":
                task_out_proj = project_to_norm(task_out_z, norm=self.embedding_norm, z_type=self.z_type)
                
            # print(task_out_proj.max(), task_out_proj.min())
            return task_out_proj, extra_dict
        
        
        def compute_prior(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:]
            
            prior_latent = self.z_prior(self_obs)
            prior_mu = self.z_prior_mu(prior_latent)
            
            prior_latent_left = self.z_left_prior(self_obs[:, self.self_obs_left_hand_idxes])
            prior_mu_left = self.z_left_prior_mu(prior_latent_left)
            
            prior_latent_right = self.z_right_prior(self_obs[:, self.self_obs_right_hand_idxes])
            prior_mu_right = self.z_right_prior_mu(prior_latent_right)
            
            prior_mu = torch.cat([prior_mu, prior_mu_left, prior_mu_right], dim = -1)
            
            if self.use_vae_prior:
                prior_logvar = self.z_prior_logvar(prior_latent)
                prior_logvar_left = self.z_left_prior_logvar(prior_latent_left)
                prior_logvar_right = self.z_right_prior_logvar(prior_latent_right)
                
                prior_logvar = torch.cat([prior_logvar, prior_logvar_left, prior_logvar_right], dim = -1)
                
                if self.use_vae_clamped_prior:
                    prior_logvar = torch.clamp(prior_logvar, min = -5, max = self.vae_var_clamp_max)
                    
                return prior_mu, prior_logvar
            
            elif self.use_vae_fixed_prior:
                if self.use_vae_sphere_prior:
                    return project_to_norm(prior_mu, z_type="sphere", norm = self.embedding_norm), torch.ones_like(prior_mu) * self.vae_prior_fixed_logvar
                else:
                    return prior_mu, torch.ones_like(prior_mu ) * self.vae_prior_fixed_logvar
                    
           
        
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, eps

        def eval_z(self, obs_dict):
            obs = obs_dict['obs']

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            z_out = self.z_mlp(obs)
            if self.proj_norm:
                z_out, extra_dict = self.form_embedding(z_out, obs_dict)
            return z_out
        
        def read_z(self, z):
            z_readout = self.z_reader_mlp(z)
            return z_readout
        
        def eval_critic(self, obs_dict):

            obs = obs_dict['obs']
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)

            self_obs = obs[:, :self.self_obs_size]
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here
            
            if self.has_rnn:
                c_out_in = c_out
                c_out = self.critic_z_mlp(c_out_in)

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
                
                c_out = self.critic_z_proj_linear(c_out)
                # c_out, extra_dict = self.form_embedding(c_out) # do not form VAE embedding for cirtic. 
                if self.z_type == "sphere":
                    c_out = project_to_norm(c_out, norm=self.embedding_norm, z_type=self.z_type)
                
                c_out = torch.cat([self_obs, c_out], dim=-1)

                c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                task_out = self.critic_z_mlp(obs)
                
                # c_out, extra_dict = self.form_embedding(c_out) # do not form VAE embedding for cirtic. 
                if self.z_type == "sphere": # but we do project for z sphere....
                    task_out = project_to_norm(task_out, norm=self.embedding_norm, z_type=self.z_type)
                    
                if self.z_all:
                    c_input = task_out
                else:
                    c_input = torch.cat([self_obs, task_out], dim=-1)
                c_out = self.critic_mlp(c_input)
                value = self.value_act(self.value(c_out))
                return value
            
        def eval_actor(self, obs_dict, return_extra = False):
            # ZL: uglllly code. Refractor asap. 
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:]
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            
            if self.has_rnn:
                
                a_out_in = a_out
                
                a_out = self.z_mlp(obs)
                    
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
                
                z_out = self.z_proj_linear(a_out)
                
                if self.proj_norm:
                    z_out, extra_dict = self.form_embedding(z_out, obs_dict)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                
                actor_input = torch.cat([self_obs, z_out], dim=-1)
                a_out = self.actor_mlp(actor_input)

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, a_states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, a_states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    
                    if return_extra:
                        return mu, sigma, a_states, extra_dict
                    else:
                        return mu, sigma, a_states
            else:
                # if self.z_all:
                #     task_out_z = self.z_mlp(task_obs)
                #     self_out_z = self.z_self_mlp(self_obs)
                #     # self_out_z[:] = 0
                #     task_out_z = torch.cat([task_out_z, self_out_z], dim=-1)
                # else:
                #     task_out_z = self.z_mlp(obs)
                
                task_out_z = self.z_mlp(obs)
                
                if self.proj_norm:
                    z_out, extra_dict = self.form_embedding(task_out_z, obs_dict)
                
                # if "z_acc" not in self.__dict__.keys():
                #     self.z_acc = []
                # self.z_acc.append(z_out)
                # if len(self.z_acc) > 500:
                #     import ipdb; ipdb.set_trace()
                #     import joblib;joblib.dump(self.z_acc, "z_acc_compare_3.pkl")
                if self.z_all:
                    actor_input = z_out
                else:
                    actor_input = torch.cat([self_obs, z_out], dim=-1)
         
                a_out = self.actor_mlp(actor_input) # Decoder 
                a_left_out = self.actor_left_mlp(torch.cat([self_obs[:, self.self_obs_left_hand_idxes], z_out[:, self.embedding_size_body:self.embedding_size_body + self.embedding_size_part]], dim = -1))
                a_right_out = self.actor_right_mlp(torch.cat([self_obs[:, self.self_obs_right_hand_idxes], z_out[:, -self.embedding_size_part:]], dim = -1))
                
                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits
                
                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    mu_left = self.mu_act(a_left_out)
                    mu_right = self.mu_act(a_right_out)
                    
                    mu_action = torch.zeros([mu.shape[0], len(self.dof_names) * 3], device = mu.device)
                    
                    mu_action[:, self.body_name_idxes] = mu
                    mu_action[:, self.left_dof_idxes] = mu_left
                    mu_action[:, self.right_dof_idxes] = mu_right
                    
                    mu = mu_action
                    
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                        
                    if return_extra:
                        return mu, sigma, extra_dict
                    else:
                        return mu, sigma

        def _build_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            
            if self.z_type == "vae":
                out_size = self.embedding_size * 5
            else:
                out_size = self.embedding_size

            mlp_input_shape = self_obs_size + task_obs_size  # target

            # Encoder
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.z_mlp = self._build_mlp(**mlp_args)
            self.z_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=out_size))
            
            left_hand_input_shape = len(self.self_obs_left_hand_idxes) + len(self.im_obs_left_hand_idxes)
            mlp_args = {'input_size': left_hand_input_shape, 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
            self.z_left_mlp = self._build_mlp(**mlp_args)
            self.z_left_mlp.append(nn.Linear(in_features=self._part_units[-1], out_features=out_size))
            
            right_hand_input_shape = len(self.self_obs_right_hand_idxes) + len(self.im_obs_right_hand_idxes)
            mlp_args = {'input_size': right_hand_input_shape, 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
            self.z_right_mlp = self._build_mlp(**mlp_args)
            self.z_right_mlp.append(nn.Linear(in_features=self._part_units[-1], out_features=out_size))
            
            mlp_init = self.init_factory.create(**self._task_initializer)
            init_mlp(self.z_mlp, mlp_init)
            init_mlp(self.z_left_mlp, mlp_init)
            init_mlp(self.z_right_mlp, mlp_init)
            
            
            # Decoder
            left_hand_input_shape = len(self.self_obs_right_hand_idxes) + self.embedding_size_part
            mlp_args = {'input_size': left_hand_input_shape, 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
            self.actor_left_mlp = self._build_mlp(**mlp_args)
            self.actor_left_mlp.append(nn.Linear(in_features=self._part_units[-1], out_features=len(self.left_hand_names) * 3))
            init_mlp(self.actor_left_mlp, mlp_init)
            
            right_hand_input_shape = len(self.self_obs_right_hand_idxes) + self.embedding_size_part
            mlp_args = {'input_size': right_hand_input_shape, 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
            self.actor_right_mlp = self._build_mlp(**mlp_args)
            self.actor_right_mlp.append(nn.Linear(in_features=self._part_units[-1], out_features=len(self.right_hand_names) * 3))
            init_mlp(self.actor_right_mlp, mlp_init)
            
                        
            if self.z_type == "vae":
                self.z_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                self.z_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                init_mlp(self.z_mu, mlp_init); init_mlp(self.z_logvar, mlp_init)
                
                
                
                # self.z_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_body)
                # self.z_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_body)
                # init_mlp(self.z_mu, mlp_init); init_mlp(self.z_logvar, mlp_init)
                
                # self.z_left_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_part)
                # self.z_left_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_part)
                # init_mlp(self.z_left_mu, mlp_init); init_mlp(self.z_left_logvar, mlp_init)
                
                # self.z_right_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_part)
                # self.z_right_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size_part)
                # init_mlp(self.z_right_mu, mlp_init); init_mlp(self.z_right_logvar, mlp_init)
                
                if self.use_vae_prior:
                    mlp_args = {'input_size': self_obs_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                    self.z_prior = self._build_mlp(**mlp_args)
                    self.z_prior_mu = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size_body)
                    self.z_prior_logvar = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size_body)
                    init_mlp(self.z_prior, mlp_init); init_mlp(self.z_prior_mu, mlp_init); init_mlp(self.z_prior_logvar, mlp_init)
                    
                    mlp_args = {'input_size': len(self.self_obs_left_hand_idxes), 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
                    self.z_left_prior = self._build_mlp(**mlp_args)
                    self.z_left_prior_mu = nn.Linear(in_features=self._part_units[-1], out_features=self.embedding_size_part)
                    self.z_left_prior_logvar = nn.Linear(in_features=self._part_units[-1], out_features=self.embedding_size_part)
                    init_mlp(self.z_left_prior, mlp_init); init_mlp(self.z_left_prior_mu, mlp_init); init_mlp(self.z_left_prior_logvar, mlp_init)
                    
                    mlp_args = {'input_size': len(self.self_obs_right_hand_idxes), 'units': self._part_units, 'activation': self._part_activation, 'dense_func': torch.nn.Linear}
                    self.z_right_prior = self._build_mlp(**mlp_args)
                    self.z_right_prior_mu = nn.Linear(in_features=self._part_units[-1], out_features=self.embedding_size_part)
                    self.z_right_prior_logvar = nn.Linear(in_features=self._part_units[-1], out_features=self.embedding_size_part)
                    init_mlp(self.z_right_prior, mlp_init); init_mlp(self.z_right_prior_mu, mlp_init); init_mlp(self.z_right_prior_logvar, mlp_init)
                    
                    
                    ############ Debugging ############
                    # self.z_left_prior_mu.requires_grad_(False)
                    # self.z_left_mlp.requires_grad_(False)
                    # self.z_left_prior.requires_grad_(False)
                    # self.z_right_prior_mu.requires_grad_(False)
                    # self.z_right_mlp.requires_grad_(False)
                    # self.z_right_prior.requires_grad_(False)
                    
                elif self.use_vae_fixed_prior:
                    mlp_args = {'input_size': self_obs_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                    self.z_prior = self._build_mlp(**mlp_args)
                    self.z_prior_mu = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                    init_mlp(self.z_prior, mlp_init); init_mlp(self.z_prior_mu, mlp_init)
                    
            return
        
        def _build_critic_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self_obs_size + task_obs_size  # target

            self.critic_z_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.critic_z_mlp = self._build_mlp(**mlp_args)
            
            if not self.has_rnn:
                self.critic_z_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size))
            else:
                self.critic_z_proj_linear = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
            

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.critic_z_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return

        def _build_z_reader(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self.embedding_size  # target

            self.z_reader_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.z_reader_mlp = self._build_mlp(**mlp_args)
            self.z_reader_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=72))

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.z_reader_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return
