

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

from phc.utils import torch_utils
import joblib
import torch
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from collections import defaultdict

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from phc.utils.motion_lib_smpl import MotionLibSMPL
from smpl_sim.utils.torch_ext import to_torch

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy




class MotionLibSMPLObj(MotionLibSMPL):
    
    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg = motion_lib_cfg)

        self.obj_to_seq_dict = defaultdict(list)
        for k in self._motion_data_keys:
            obj_name = self._motion_data_load[k]['obj_data']['obj_info'][0].split("/")[-1].split(".")[0]
            self.obj_to_seq_dict[obj_name].append(k)

    @staticmethod
    def load_motion_with_skeleton(cfg, ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
        np.random.seed(random.randint(0, 5000)* (pid + 1))
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = shape_params[f]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
            
            pose_quat_global = curr_file['pose_quat_global'][start:end]
            B, J, N = pose_quat_global.shape
            
            trans = curr_file['root_trans_offset'].clone()[start:end].float()
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).reshape(B, J, -1)

            ##### Do not randomize the heading for this task.  ######
            # if (not flags.im_eval) and (not flags.test):
            #     # if True:
            #     random_rot = np.zeros(3)
            #     random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
            #     random_heading_rot = sRot.from_euler("xyz", random_rot)
            #     pose_aa[:, 0, :] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, 0, :])).as_rotvec())
            #     pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
            #     trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T).float())
            ##### ZL: randomize the heading ######

            trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)

            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            
            ##### Loading Object data #####
            obj_data = curr_file['obj_data']
            obj_pose = obj_data['obj_pose'].reshape(B, -1, 7)
            obj_info = obj_data['obj_info']
            
            obj_trans = torch.from_numpy(obj_pose[..., :3]).float()
            obj_rot = torch.from_numpy(obj_pose[..., 3:7]).float()
            hand_trans = torch.from_numpy(obj_data['hand_trans']).float() # Pregrasp hand position
            hand_rot = torch.from_numpy(obj_data['hand_rot']).float() # Pregrasp hand rotation (global)

            ############# Random rotation and translate of the object pose #############
            if cfg.get("obj_rand_pos", False) and (not flags.im_eval) and (not flags.test):
            # if cfg.get("obj_rand_pos", False) and (not flags.im_eval) :

                random_heading = sRot.from_euler("xyz", [0, 0, np.random.random() * 2 * np.pi])
                
                random_pos_delta = np.random.random(3) # randomrize position in the range of -0.5 to 0.5
                
                if cfg.get("obj_rand_pos_extreme", False):
                    random_pos_delta[0] -= 0.5 # x drecation can be anything 
                    random_pos_delta[1] -= 1 # y direction can't be negative. 
                    random_pos_delta[:2] *= 0.25 # Too extreme
                    random_pos_delta[2] = np.random.uniform(low=-obj_trans[0, 1, 2], high=0.05)
                else:
                    random_pos_delta[:2] -= 0.5
                    random_pos_delta[2] -= 0.75 # lower better. Higher can lead to not plausible poses
                    random_pos_delta *= 0.1
                
                obj_rot = torch.from_numpy((random_heading * sRot.from_quat(obj_rot.view(-1, 4))).as_quat().reshape(obj_rot.shape)).float()
                hand_rot = torch.from_numpy((random_heading * sRot.from_quat(hand_rot.view(-1, 4))).as_quat().reshape(hand_rot.shape)).float()
                
                hand_trans = (hand_trans - obj_trans[:, 0:1]) @ torch.from_numpy(random_heading.as_matrix().T).float() + obj_trans[:, 0:1] # Change
                
                obj_trans = obj_trans + torch.from_numpy(random_pos_delta).float()
                hand_trans = hand_trans + torch.from_numpy(random_pos_delta).float()
            ############# Random rotation and translate of the object pose #############

            
            
            global_angular_vel = SkeletonMotion._compute_angular_velocity(obj_rot, time_delta=1 / curr_file['fps'])
            linear_vel = SkeletonMotion._compute_velocity(obj_trans, time_delta=1 / curr_file['fps'])

            quest_motion = {"obj_angular_vel": global_angular_vel, "obj_linear_vel": linear_vel, "obj_trans": obj_trans, "obj_rot": obj_rot}
            hand_global_angular_vel = SkeletonMotion._compute_angular_velocity(hand_rot, time_delta=1 / curr_file['fps'])
            hand_linear_vel = SkeletonMotion._compute_velocity(hand_trans, time_delta=1 / curr_file['fps'])
            quest_motion["hand_angular_vel"] = hand_global_angular_vel
            quest_motion["hand_linear_vel"] = hand_linear_vel
            quest_motion["hand_trans"] = hand_trans
            quest_motion["hand_rot"] = hand_rot
            quest_motion["contact_idx"] = obj_data['contact_info'].nonzero()[0][0]

            curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)
            
        if not queue is None:
            queue.put(res)
        else:
            return res

    def compute_per_obj_fail_rate(self, failed_keys):
        per_obj_success_rate = {}
        obj_fail_dict = defaultdict(list)
        for k in failed_keys:
            if k in self.obj_to_seq_dict:
                obj_fail_dict[k].append(k)
            else:
                obj_fail_dict[k.split("_")[1]].append(k)
            
        for k, v in self.obj_to_seq_dict.items():
            per_obj_success_rate[k] = len(obj_fail_dict[k])/len(v)
        return per_obj_success_rate
    
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1):
        # load motion load the same number of motions as there are skeletons (humanoids)
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        
        o_gts, o_grs, o_gavs, o_gvs, h_gts, h_grs, h_gavs, h_gvs = [], [], [], [], [], [], [], []
        contact_idx, lift_idx = [], []
        

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)
        
        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)
        
        if (random_sample or (flags.test and not flags.im_eval)) :
            
            assert(len(self.m_cfg.env_to_obj_id) == len(skeleton_trees))
            for obj_id in self.m_cfg.obj_ids_unique_load: # for each unqiuely loaded object id
                data_seq_allowed = self.m_cfg.data_seq_to_obj_id == obj_id # first fine the sequences for this object 
                data_seq_allowed_ids = torch.arange(len(self._motion_data_list))[data_seq_allowed].cuda() # get the ids of the sequences
                env_picks = self.m_cfg.env_to_obj_id == obj_id # then find the environments that have this object
                
                if self._sampling_prob[data_seq_allowed].sum() > 0:
                    data_seq_picks_idx = torch.multinomial(self._sampling_prob[data_seq_allowed]/(self._sampling_prob[data_seq_allowed].sum() + 1e-20), num_samples = env_picks.sum(), replacement=True)
                else:
                    data_seq_picks_idx = torch.multinomial((self._sampling_prob[data_seq_allowed] + 1)/(len(self._sampling_prob[data_seq_allowed])), num_samples = env_picks.sum(), replacement=True)
                
                data_seq_picks = data_seq_allowed_ids[data_seq_picks_idx] # find data for each enviorment
                sample_idxes[env_picks] = data_seq_picks # set data sample for each env
                
        
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:10])
        if len(self.curr_motion_keys) < 30:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:10], ".....")
        print("*********************************************************************************\n")


        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 32)

        if num_jobs <= 8 or not self.multi_thread:
            num_jobs = 1
        if flags.debug:
            num_jobs = 1
        # num_jobs = 1
        
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(self.m_cfg, ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk],  self.mesh_parsers, self.m_cfg) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            
            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            o_gts.append(curr_motion.quest_motion['obj_trans'])
            o_grs.append(curr_motion.quest_motion['obj_rot'])
            o_gavs.append(curr_motion.quest_motion['obj_angular_vel'])
            o_gvs.append(curr_motion.quest_motion['obj_linear_vel'])

            if "hand_trans" in curr_motion.quest_motion:
                h_gts.append(curr_motion.quest_motion['hand_trans'])
                h_grs.append(curr_motion.quest_motion['hand_rot'])
                h_gavs.append(curr_motion.quest_motion['hand_angular_vel'])
                h_gvs.append(curr_motion.quest_motion['hand_linear_vel'])
                contact_idx.append(curr_motion.quest_motion['contact_idx'])
                # lift_idx.append(curr_motion.quest_motion['lift_idx'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)

        self.h_gts = torch.cat(h_gts, dim=0).float().to(self._device)
        self.h_grs = torch.cat(h_grs, dim=0).float().to(self._device)
        self.h_gavs = torch.cat(h_gavs, dim=0).float().to(self._device)
        self.h_gvs = torch.cat(h_gvs, dim=0).float().to(self._device)
        
        self.o_gts = torch.cat(o_gts, dim=0).float().to(self._device)
        self.o_grs = torch.cat(o_grs, dim=0).float().to(self._device)
        self.o_gavs = torch.cat(o_gavs, dim=0).float().to(self._device)
        self.o_gvs = torch.cat(o_gvs, dim=0).float().to(self._device)
        self.contact_idx = torch.tensor(contact_idx, device=self._device)
        # self.lift_idx = torch.tensor(lift_idx, device=self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        
        self.get_contact_hand_pose_all()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def get_use_hand_label(self, motion_ids):
        return self.use_hand_flag[motion_ids]
    
    def get_contact_idxes(self, motion_ids):
        return self.contact_idx[motion_ids]
    
    def get_contact_time(self, motion_ids):
        return self.contact_idx[motion_ids] * self._motion_dt[motion_ids]

    def get_lift_time(self, motion_ids):
        return self.lift_idx[motion_ids] * self._motion_dt[motion_ids]

    def get_contact_hand_pose_all(self):
        if "contact_h_rb_rot" in self.__dict__:
            del self.contact_h_rb_rot, self.contact_h_rb_pos, self.contact_h_lin_vel, self.contact_h_ang_vel
        
        motion_ids = torch.arange(self._num_motions, device=self._device)
        n = len(motion_ids)
        dt = self._motion_dt[motion_ids]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        motion_times = self.contact_idx[motion_ids].float() * dt
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        h_ang_vel0, h_ang_vel1 = self.h_gavs[f0l], self.h_gavs[f1l]
        h_rb_rot0, h_rb_rot1 = self.h_grs[f0l], self.h_grs[f1l]
        h_rb_pos0, h_rb_pos1 = self.h_gts[f0l, :], self.h_gts[f1l, :]
        h_lin_vel0, h_lin_vel1 = self.h_gvs[f0l], self.h_gvs[f1l]
        
        ref_obj_pos0, ref_obj_pos1 = self.o_gts[f0l, :1, :], self.o_gts[f1l, :1, :]
        self.contact_ref_obj_pos = (1.0 - blend_exp) * ref_obj_pos0 + blend_exp * ref_obj_pos1
        self.contact_h_rb_rot = torch_utils.slerp(h_rb_rot0, h_rb_rot1, blend_exp)
        self.contact_h_rb_pos = (1.0 - blend_exp) * h_rb_pos0 + blend_exp * h_rb_pos1
        self.contact_h_lin_vel = (1.0 - blend_exp) * h_lin_vel0 + blend_exp * h_lin_vel1
        self.contact_h_ang_vel = (1.0 - blend_exp) * h_ang_vel0 + blend_exp * h_ang_vel1

        dist_hand_to_ref = torch.norm(self.contact_h_rb_pos - self.contact_ref_obj_pos, dim = -1, p = 2) < self.m_cfg.get("close_distance_pregrasp", 0.2)
        hand_dim  = dist_hand_to_ref.shape[-1]//2
        self.use_hand_flag = (dist_hand_to_ref.reshape(-1, 2, hand_dim).sum(dim = -1) > 0).float()
        
        if (self.use_hand_flag.sum(dim = -1) == 0).sum() > 0:
            print("No Hand is used! Is this a demo pkl? ")
        
        
        
        
        
    
    def get_contact_hand_pose(self, motion_ids):
        return {
            "contact_hand_rot": self.contact_h_rb_rot[motion_ids],
            "contact_hand_trans": self.contact_h_rb_pos[motion_ids],
            "contact_hand_ang_vel": self.contact_h_ang_vel[motion_ids],
            "contact_hand_vel": self.contact_h_lin_vel[motion_ids],
            "contact_ref_obj_pos": self.contact_ref_obj_pos[motion_ids],
        }


    
    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        o_rb_rot0, o_rb_rot1 = self.o_grs[f0l], self.o_grs[f1l]
        o_rb_pos0, o_rb_pos1 = self.o_gts[f0l, :], self.o_gts[f1l, :]
        o_ang_vel0, o_ang_vel1 = self.o_gavs[f0l], self.o_gavs[f1l]
        o_lin_vel0, o_lin_vel1 = self.o_gvs[f0l], self.o_gvs[f1l]

        o_rb_rot = torch_utils.slerp(o_rb_rot0, o_rb_rot1, blend_exp)
        o_rb_pos = (1.0 - blend_exp) * o_rb_pos0 + blend_exp * o_rb_pos1
        o_lin_vel = (1.0 - blend_exp) * o_lin_vel0 + blend_exp * o_lin_vel1
        o_ang_vel = (1.0 - blend_exp) * o_ang_vel0 + blend_exp * o_ang_vel1


        h_ang_vel0, h_ang_vel1 = self.h_gavs[f0l], self.h_gavs[f1l]
        h_rb_rot0, h_rb_rot1 = self.h_grs[f0l], self.h_grs[f1l]
        h_rb_pos0, h_rb_pos1 = self.h_gts[f0l, :], self.h_gts[f1l, :]
        h_lin_vel0, h_lin_vel1 = self.h_gvs[f0l], self.h_gvs[f1l]

        h_rb_rot = torch_utils.slerp(h_rb_rot0, h_rb_rot1, blend_exp)
        h_rb_pos = (1.0 - blend_exp) * h_rb_pos0 + blend_exp * h_rb_pos1
        h_lin_vel = (1.0 - blend_exp) * h_lin_vel0 + blend_exp * h_lin_vel1
        h_ang_vel = (1.0 - blend_exp) * h_ang_vel0 + blend_exp * h_ang_vel1
        

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
            
            "o_ang_vel": o_ang_vel,
            "o_rb_rot": o_rb_rot,
            "o_rb_pos": o_rb_pos,
            "o_lin_vel": o_lin_vel,

            "h_ang_vel": h_ang_vel,
            "h_rb_rot": h_rb_rot,
            "h_rb_pos": h_rb_pos,
            "h_lin_vel": h_lin_vel,
        }

