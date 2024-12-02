# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Tuple, OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from easydict import EasyDict

from phc.env.tasks.humanoid import remove_base_rot
from phc.utils import torch_utils
import poselib.poselib.core.rotation3d as rot3d
from phc.utils.torch_smpl_humanoid_sk_batch import Humanoid_Batch


class WorldModel(torch.nn.Module):
    """A Simulator equivalent."""

    def __init__(self, config):
        super().__init__()

        self.upright = config.upright
        n_hidden = 512 

        # architecture
        self.output_dim_dict = OrderedDict({
            "pos_delta": 3, 
            "rot_delta": 3,
            "dof_delta": config.num_dofs,
            "lin_vel": 72, # not integrated, just directly predicted
            "ang_vel": 72, # not integrated, just directly predicted
            "dof_vel": config.num_dofs, # not integrated, just directly predicted
        })
        
        dim_in = config.dim_in # max coordinates self obs + dof pos and dof_vel
        self.num_joints = config.num_rigid_bodies
        self.num_dof = config.num_dofs
        self.hum_fk = Humanoid_Batch(config.skeleton_trees, device=config.device)
        self.local_translation_batch = self.hum_fk.local_translation_batch[0:1]
        self.parent_indices = self.hum_fk.parent_indices
        dim_out = sum(self.output_dim_dict.values())
        
        self.layers = nn.Sequential(
        #   nn.BatchNorm1d(dim_in, momentum=mom), 
          # input layer
          nn.Linear(dim_in,  n_hidden, bias=True),  
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.SiLU(),
          # hidden layer 1
          nn.Linear(n_hidden, n_hidden, bias=True), 
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.SiLU(),
          # hidden layer 2
          nn.Linear(n_hidden, n_hidden, bias=True), 
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.SiLU(),
          # output layer
          nn.Linear(n_hidden, dim_out),
        )
        self.layers.to(config.device)
        # set last layer weights low to not immediately blow up orientation
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.001)
            self.layers[-1].bias.zero_()

        self.render_frame = False

    def forward(self, data_dict):
        """Simulated current state one step forward"""
        self_obs_orig = data_dict["self_obs_and_action"] # pre computed self obs + dof pos and velocity
        out = self.layers(self_obs_orig)
        out_dict = {}
        for k, v in self.output_dim_dict.items():
            out_dict[k] = out[..., :v]
            out = out[..., v:]
        next_state = self.local_step(out_dict, data_dict)
        
        return next_state

    def local_step(self, out_dict, data_dict):
        B, T, _ = data_dict["dof_pos"].shape
        rg_pos = data_dict["rg_pos"].view(B * T, self.num_joints, 3)
        rg_rot = data_dict["rg_rot"].view(B * T, self.num_joints, 4)
        rg_vel = data_dict["rg_vel"].view(B * T, self.num_joints, 3)
        rg_ang_vel = data_dict["rg_ang_vel"].view(B * T, self.num_joints, 3)
        dof_pos = data_dict["dof_pos"].view(B * T, -1, 3) # each dof is 3 dof in SMPL humanoid. 
        root_pos = rg_pos[:, 0]
        root_rot = rg_rot[:, 0]
        
        pos_delta_local = out_dict["pos_delta"].view(B * T, 3)
        rot_delta_local = out_dict["rot_delta"].view(B * T, 3)
        dof_dleta = out_dict["dof_delta"].view(B * T, -1, 3)
        
        next_lin_vel = out_dict["lin_vel"].reshape(-1, 3)
        next_ang_vel = out_dict["ang_vel"].reshape(-1, 3)
        dof_vel = out_dict["dof_vel"].view(B, T, -1, 3)
        
        if not self.upright:
            root_rot = remove_base_rot(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        heading_rot_expand = heading_rot.unsqueeze(1).repeat(1, self.num_joints, 1).view(-1, 4)
        new_root_pos = root_pos + torch_utils.my_quat_rotate(heading_rot, pos_delta_local)
        rot_delta = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, torch_utils.exp_map_to_quat(rot_delta_local)), heading_inv_rot)
        new_root_rot = torch_utils.quat_mul(rot_delta, root_rot)
        new_dof_pos_quat = torch_utils.quat_mul(torch_utils.exp_map_to_quat(dof_dleta), torch_utils.exp_map_to_quat(dof_pos)) # local dof_quat. 
        new_dof_pos = torch_utils.quat_to_exp_map(new_dof_pos_quat)
        rot_quats = torch.cat([new_root_rot.unsqueeze(1), new_dof_pos_quat], dim=1)
        next_body_pos, next_body_rot_quat = forward_kinematics_batch(rot_quats, new_root_pos, self.local_translation_batch, self.parent_indices) # assuming fixed offsets.
        

        next_lin_vel = torch_utils.my_quat_rotate(heading_rot_expand, next_lin_vel)
        next_ang_vel = torch_utils.my_quat_rotate(heading_rot_expand, next_ang_vel)
        
        return_dict = EasyDict()
        return_dict.body_pos = next_body_pos.view(B, T, -1, 3)
        return_dict.body_quat = next_body_rot_quat.view(B, T, -1, 4)
        return_dict.dof_pos = new_dof_pos.view(B, T, -1, 3)
        return_dict.lin_vel = next_lin_vel.view(B, T, -1, 3)
        return_dict.ang_vel = next_ang_vel.view(B, T, -1, 3)
        return_dict.dof_vel = dof_vel
        
       
        return return_dict


@torch.jit.script
def forward_kinematics_batch(rotations, root_positions, local_trans, parent_indices):
    device, dtype = rotations.device, rotations.dtype
    B, J, = rotations.shape[:2]
    positions_world = []
    rotations_world = []
    local_trans = local_trans.repeat(B, 1, 1)

    for i in range(J):
        if parent_indices[i] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, 0])
        else:
            
            jpos = (torch_utils.my_quat_rotate(rotations_world[parent_indices[i]], local_trans[:, i, :]) + positions_world[parent_indices[i]])
            rot_quat = torch_utils.quat_mul(rotations_world[parent_indices[i]], rotations[:, i, :])

            positions_world.append(jpos)
            rotations_world.append(rot_quat)

    positions_world = torch.stack(positions_world, dim=1)
    rotations_world = torch.stack(rotations_world, dim=1)
    return positions_world, rotations_world