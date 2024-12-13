import numpy as np
import torch
import joblib
import random
from phc.utils.flags import flags
from phc.utils import torch_utils
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonMotion


class TrajGenerator3D():

    def __init__(self, num_envs, episode_dur, num_verts, device, traj_config, starting_still_dt=1.5):

        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        
        self._dtheta_max = traj_config['dtheta_max']
        self._speed_min =  traj_config['speed_min']
        self._speed_max = traj_config['speed_max']
        self._accel_max = traj_config['accel_max']
        self._sharp_turn_prob = traj_config['sharp_turn_prob']

        self.gts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.grs_flat = torch.zeros((num_envs * num_verts, 4), dtype=torch.float32, device=self._device)
        self.gavs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.gvs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        
        self.gts = self.gts_flat.view((num_envs, num_verts, 3))
        self.grs = self.grs_flat.view((num_envs, num_verts, 4))
        self.gavs = self.gavs_flat.view((num_envs, num_verts, 3))
        self.gvs = self.gvs_flat.view((num_envs, num_verts, 3))

        

        env_ids = torch.arange(self.get_num_envs(), dtype=int)

        self.heading = torch.zeros(num_envs, 1)
        
        self.starting_still_frames = int(starting_still_dt/self._dt)
        return
        
    def sample_dtheat_dspeed(self, n):
        num_verts = self.get_num_verts()
        dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0  # Sample the angles at each waypoint
        dtheta *= self._dtheta_max * self._dt

        dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0)  # Sharp Angles Angle
        sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
        sharp_mask = torch.bernoulli(sharp_probs) == 1.0
        dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

        dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0)  # Heading

        dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
        dspeed *= self._accel_max * self._dt
        dspeed[:, 0] = (self._speed_max - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min  # Speed

        speed = torch.zeros_like(dspeed)
        speed[:, 0] = dspeed[:, 0]
        for i in range(1, dspeed.shape[-1]):
            speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, self._speed_max)

        dtheta = torch.cumsum(dtheta, dim=-1)
        return speed, dspeed, dtheta
    
    def reset(self, env_ids, init_pos, init_rot):
        n = len(env_ids)
        if (n > 0):
            
            speed, dspeed, dtheta = self.sample_dtheat_dspeed(n)
            speed_z, dspeed_z, dtheta_z = self.sample_dtheat_dspeed(n)

            seg_len = speed * self._dt

            dpos = torch.stack([torch.cos(dtheta), -torch.sin(dtheta), torch.cos(dtheta_z)], dim=-1)
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:3] += init_pos[..., 0:3]
            vert_pos = torch.cumsum(dpos, dim=-2)
            
            self.gts[env_ids, :self.starting_still_frames] = init_pos[..., None, :] # first 30 Frames should be still. 
            self.gts[env_ids, (self.starting_still_frames):] = vert_pos[:, :-(self.starting_still_frames - 1)]
            self.gts[env_ids, ..., 2] = torch.clamp(self.gts[env_ids, ..., 2], min=0.3, max=1.6)

            ending_rot = torch.from_numpy(sRot.random(n).as_quat()).to(init_pos)

            num_verts = self.get_num_verts()
            slerp_weights = (torch.arange(num_verts - self.starting_still_frames)/(num_verts - self.starting_still_frames)).repeat(n, 1).to(init_pos)
            slerp_weights = torch.cat([torch.zeros([n, self.starting_still_frames]).to(init_pos), slerp_weights], dim = -1)
            
            self.grs[env_ids] = torch_utils.slerp(init_rot[:, None].repeat(1, num_verts, 1).view(-1, 4), ending_rot[:, None].repeat(1, num_verts, 1).view(-1, 4), slerp_weights.view(-1, 1)).view(n, self.get_num_verts(), -1)            
            self.gavs[env_ids] = SkeletonMotion._compute_velocity(p=self.gts[env_ids, :, None, :], time_delta=self._dt, guassian_filter = False)[:, :, 0]
            self.gvs[env_ids] = SkeletonMotion._compute_angular_velocity(r=self.grs[env_ids, :, None, :], time_delta=self._dt, guassian_filter = False)[:, :, 0]
            
 
        return

    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')

        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num=coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num=coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis=-1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis=-2)
        self.gts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self.gts[env_ids]

    def get_num_verts(self):
        return self.gts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self.gts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt 
        return dur

    def get_traj_verts(self, traj_id):
        return self.gts[traj_id]

    def get_motion_state(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.gts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.gts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        o_rb_pos = (1.0 - lerp) * pos0 + lerp * pos1


        rb_rot0 = self.grs_flat[traj_ids * num_verts + seg_id0]
        rb_rot1 = self.grs_flat[traj_ids * num_verts + seg_id1]
        
        o_rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, lerp)

        ang_vel0, ang_vel1 = self.gavs_flat[traj_ids * num_verts + seg_id0], self.gavs_flat[traj_ids * num_verts + seg_id1]
        lin_vel0, lin_vel1 = self.gvs_flat[traj_ids * num_verts + seg_id0], self.gvs_flat[traj_ids * num_verts + seg_id1]

        o_lin_vel = (1.0 - lerp) * lin_vel0 + lerp * lin_vel1
        o_ang_vel = (1.0 - lerp) * ang_vel0 + lerp * ang_vel1
        

        return {
            "o_ang_vel": o_ang_vel[:, None].clone(),
            "o_rb_rot": o_rb_rot[:, None].clone(),
            "o_rb_pos": o_rb_pos[:, None].clone(),
            "o_lin_vel": o_lin_vel[:, None].clone(),
        }

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.gts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.gts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)
                disc_grad = torch.autograd.grad(new_val, new_obs, grad_outputs=torch.ones_like(new_val), create_graph=False, retain_graph=True, only_inputs=True)

        return pos
