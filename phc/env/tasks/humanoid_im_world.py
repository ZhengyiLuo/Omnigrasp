import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from phc.env.util import gym_util
from phc.env.tasks.humanoid_im import HumanoidIm
from phc.utils.isaacgym_torch_utils import *

from utils import torch_utils
from phc.utils.flags import flags
from typing import OrderedDict

from phc.learning.world_model import WorldModel
from easydict import EasyDict 
import poselib.poselib.core.rotation3d as rot3d

class HumanoidImWorld(HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        # self.world_model.load_state_dict(torch.load("output/HumanoidIm/phc_3_world/world_model.pth"))
        
    def _update_marker(self):
        if flags.show_traj:
            
            motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times + self._motion_start_times_offset # + 1 for target. 
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
            self._marker_pos[:] = self.world_model_sim

            if self._occl_training:
                self._marker_pos[self.random_occlu_idx] = 0

        else:
            self._marker_pos[:] = 1000

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return
    
    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # apply actions
        self.pre_physics_step(actions)
        
        if self.save_kin_info: # this needs to happen after pre_physics_step to get the correctly scaled actions
            self.update_kin_info()
        
        ########## World model testing
        
        if self.cfg.test:
            kin_dict = self.kin_dict
            kin_dict['self_obs_and_action'] = torch.cat([self.obs_buf[:, :self.get_self_obs_size()], kin_dict['dof_pos'], kin_dict['dof_vel'], kin_dict['actions']], dim=-1)
            kin_dict = {k: v[:, None] for k, v in kin_dict.items()}
            next_state = self.world_model.forward(kin_dict)
            self.world_model_sim = next_state.body_pos[0]
            
        #     self._rigid_body_ang_vel = next_state.ang_vel[0]
        #     self._rigid_body_vel = next_state.lin_vel[0]
        #     self._rigid_body_pos = next_state.body_pos[0]
        #     self._rigid_body_rot = next_state.body_quat[0]
        #     self._dof_vel = next_state.dof_vel.reshape(self.num_envs, -1)
        #     self._dof_pos = next_state.dof_pos.reshape(self.num_envs, -1)
            
        # step physics and render each frame
        # self.render()
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
            
    
    def setup_kin_info(self):
        if self.cfg.env.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({ 
                "rg_pos": self._rigid_body_pos.clone(),
                "rg_rot": self._rigid_body_rot.clone(),
                "rg_vel": self._rigid_body_vel.clone(),
                "rg_ang_vel": self._rigid_body_ang_vel.clone(),
                
                "ref_body_pos": self.ref_body_pos.clone(),
                "ref_body_rot": self.ref_body_rot.clone(), 
                "ref_body_vel": self.ref_body_vel.clone(),
                "ref_body_ang_vel": self.ref_body_ang_vel.clone(),
                
                "dof_pos": self._dof_pos.clone(),
                "dof_vel": self._dof_vel.clone(),
                "actions": torch.zeros_like(self._dof_pos),
            }) # Full kinemaitc state. 
        config_dict = EasyDict({
            "num_dofs": 69, 
            "num_rigid_bodies": 24,
            "dim_in": self.get_self_obs_size() + 3 * self._dof_size , # max coordinates self obs + dof pos and dof_vel and actions
            "upright": self._has_upright_start, 
            "device": self.device,
            "skeleton_trees": self.skeleton_trees,
        })
        
        self.world_model = WorldModel(config_dict)
        
    def update_kin_info(self):
        
        # this is called right before the stepping function, right before the actions. 
        self.kin_dict.update({ 
                "rg_pos": self._rigid_body_pos.clone(),
                "rg_rot": self._rigid_body_rot.clone(),
                "rg_vel": self._rigid_body_vel.clone(),
                "rg_ang_vel": self._rigid_body_ang_vel.clone(),
                
                "ref_body_pos": self.ref_body_pos.clone(),
                "ref_body_rot": self.ref_body_rot.clone(), 
                "ref_body_vel": self.ref_body_vel.clone(),
                "ref_body_ang_vel": self.ref_body_ang_vel.clone(),
                
                "dof_pos": self._dof_pos.clone(),
                "dof_vel": self._dof_vel.clone(),
                "actions": self.actions.clone(),
            }) # Full kinemaitc state. 
        
   