import torch

from isaacgym import gymtorch, gymapi, gymutil
from phc.utils.isaacgym_torch_utils import *
from easydict import EasyDict
from scipy.spatial.transform import Rotation as sRot

import phc.env.tasks.humanoid as humanoid
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
import phc.env.tasks.humanoid_im as humanoid_im
import phc.env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
from uuid import uuid4

from phc.utils.flags import flags
import joblib
from phc.utils.motion_lib_smpl_obj import MotionLibSMPLObj
from phc.utils.motion_lib_base import FixHeightMode



TAR_ACTOR_ID = 1

class HumanoidGrab(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.load_humanoid_configs(cfg)
        self._fut_tracks = cfg["env"].get("fut_tracks", False)
        if self._fut_tracks:
            self._num_traj_samples = cfg["env"]["numTrajSamples"]
        else:
            self._num_traj_samples = 1
        self._traj_sample_timestep = 1 / cfg["env"].get("trajSampleTimestepInv", 30)
        self.reward_specs = cfg["env"].get("reward_specs", {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1})
        self.reward_specs_im = {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1}
        
        self.num_envs = cfg["env"]["num_envs"]
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        
        self._track_bodies = cfg["env"].get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_key_body_ids_tensor(self._track_bodies)
        self._reset_bodies = cfg["env"].get("reset_bodies", self._track_bodies)
        self._reset_bodies_id = self._build_key_body_ids_tensor(self._reset_bodies)
        self._contact_sensor_body_ids = self._build_key_body_ids_tensor(cfg.env.contact_sensor_bodies)
        
        # ZL Debugging putting objects. Fix asap. Very very very very janky 
        data_seq = joblib.load(cfg.env.motion_file)
        self.data_key = data_key = list(data_seq.keys())[0]
        self.gender  = data_seq[data_key]["gender"]
        self.gender_number = torch.zeros(1)
        if self.gender == "neutral": self.gender_number[:] = 0
        if self.gender == "male": self.gender_number[:] = 1
        if self.gender == "female": self.gender_number[:] = 2
        
        
        self.obj_name = data_seq[data_key]["obj_data"]["obj_info"][0].split("/")[-1].split(".")[0]
        self.v_template = torch.from_numpy(data_seq[data_key]['v_template']).float()
        

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._sampled_motion_ids = self.all_env_ids
        self._obj_contact_forces = self.contact_force_tensor.view(self.num_envs, self.bodies_per_env, 3)[..., -2:, :] # gater both table and object contact forces.
        
        self._cycle_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        grab_body_names = cfg.env.hand_bodies
        self._grab_body_ids = self._build_key_body_ids_tensor(grab_body_names)
        
        self._build_target_tensors()
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        return
    
    def _create_smpl_humanoid_xml(self, num_humanoids, smpl_robot, queue, pid):
        np.random.seed(np.random.randint(5002) * (pid + 1))
        res = {}
        for idx in num_humanoids:
            if self.has_shape_variation:
                gender_beta = self._amass_gender_betas[idx % self._amass_gender_betas.shape[0]]
            else:
                gender_beta = np.zeros(17)

            if flags.im_eval:
                gender_beta = np.zeros(17)
                    
            asset_id = uuid4()
            
            if not smpl_robot is None:
                asset_id = uuid4()
                asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"
                
                if self.has_shape_variation and "v_template" in self.__dict__:
                    smpl_robot.load_from_skeleton(v_template = self.v_template, gender=self.gender_number, objs_info=None)
                else:
                    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
                
                smpl_robot.write_xml(asset_file_real)
                # smpl_robot.write_xml("test.xml")
            else:
                asset_file_real = f"phc/data/assets/mjcf/smpl_{int(gender_beta[0])}_humanoid.xml"

            res[idx] = (gender_beta, asset_file_real)

        if not queue is None:
            queue.put(res)
        else:
            return res
    
    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            fix_height_mode = self.cfg.env.get("fix_height_mode", "full_fix")
            
            if fix_height_mode == "full_fix":
                fix_height_mode = FixHeightMode.full_fix
            elif fix_height_mode == "ankle_fix":
                fix_height_mode = FixHeightMode.ankle_fix
            elif fix_height_mode == "no_fix":
                fix_height_mode = FixHeightMode.no_fix
                
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": fix_height_mode,
                "min_length": self._min_motion_len,
                "max_length": self.max_len,
                "im_eval": flags.im_eval,
                "multi_thread": True ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
            })
            motion_eval_file = motion_train_file
            self._motion_train_lib = MotionLibSMPLObj(motion_lib_cfg)
            motion_lib_cfg.im_eval = True
            self._motion_eval_lib = MotionLibSMPLObj(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions), max_len=-1 if flags.test else self.max_len)

        else:
            self._motion_lib = MotionLib(motion_file=motion_train_file, dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets, device=self.device)

        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15 * self._num_traj_samples + 9 # local pos and orientation
            
            if self.cfg.env.has_im_obs:
                obs_size = obs_size + len(self._track_bodies) * 24
                
            if self.cfg.env.get("contact_obs", False):
                obs_size = obs_size + len(self._contact_sensor_body_ids) * 3
        return obs_size
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return
    
    def _load_target_asset(self): 
        ##### Load object config files here ######
        self._target_assets = {}
        
        asset_root = "phc/data/assets/urdf/grab/"
        asset_file = f"{self.obj_name}.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 100
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.vhacd_params.resolution = 300000
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.fix_base_link = False

        _target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self._target_assets[self.obj_name] = _target_asset
        
        
        asset_options_fix = gymapi.AssetOptions()
        asset_options_fix.angular_damping = 0.01
        asset_options_fix.linear_damping = 0.01
        asset_options_fix.max_angular_velocity = 100.0
        asset_options_fix.density = 1000.0
        asset_options_fix.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options_fix.vhacd_enabled = True
        asset_options_fix.vhacd_params.max_convex_hulls = 10
        asset_options_fix.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options_fix.vhacd_params.resolution = 300000
        asset_options_fix.fix_base_link = True
        asset_file = "table.urdf"
        
        _target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options_fix)
        self._target_assets['table'] = _target_asset
         
        return

    def _build_target(self, env_id, env_ptr):
        
        
        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_assets[self.obj_name], default_pose, "target", env_id, 0)
        self._target_handles.append(target_handle)
        
        target_handle = self.gym.create_actor(env_ptr, self._target_assets['table'], default_pose, "target", env_id, 0) # Tablle 
        self._target_handles.append(target_handle)
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._obj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._table_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        
        self._tar_actor_ids = torch.stack([to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1, to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + +2], dim = -1)
        
        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids].flatten()
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)
        
        
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], self._motion_start_times[env_ids])  
            
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        self._obj_states[env_ids, :3] = ref_o_rb_pos[:, 0]
        self._obj_states[env_ids, 3:7] = ref_o_rb_rot[:, 0]
        self._obj_states[env_ids, 7:10] = ref_o_lin_vel[:, 0]
        self._obj_states[env_ids, 10:13] = ref_o_ang_vel[:, 0]
        
        
        static_o_ang_vel, static_o_lin_vel, static_o_rb_rot, static_o_rb_pos = motion_res['o_ang_vel'][:, 1:], motion_res['o_lin_vel'][:, 1:], motion_res['o_rb_rot'][:, 1:], motion_res['o_rb_pos'][:, 1:]
        self._table_states[env_ids, :3] = static_o_rb_pos[:, 0] 
        self._table_states[env_ids, 2] -= (0.1 - 0.005481)/2
        self._table_states[env_ids, 3:7] = static_o_rb_rot[:, 0]
        self._table_states[env_ids, 7:10] = static_o_lin_vel[:, 0]
        self._table_states[env_ids, 10:13] = static_o_ang_vel[:, 0]
        
        return
    
    def _sample_time(self, motion_ids):
        # Motion imitation, no more blending and only sample at certain locations
        return self._motion_lib.sample_time_interval(motion_ids)
        # return self._motion_lib.sample_time(motion_ids)
    
    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]

        if (self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
            # motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
            # motion_times = torch.clamp(motion_times, 0, 40 * self.dt) # Start as in the first 40 frames. 
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
            
        # if flags.test:
            # motion_times[:] = 0
        
        if self.humanoid_type in ["smpl", "smplh", "smplx"] :
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], motion_times)
            rb_pos, rb_rot = None, None
            
        return self._sampled_motion_ids[env_ids], motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel


    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            
            root_states = self._humanoid_root_states
            obj_states = self._obj_states
            contact_forces = self._contact_forces
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            root_states = self._humanoid_root_states[env_ids]
            obj_states = self._obj_states[env_ids]
            contact_forces = self._contact_forces[env_ids]
            
        if self._fut_tracks:
            time_steps = self._num_traj_samples
            B = env_ids.shape[0]
            time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * self._traj_sample_timestep
            motion_times_steps = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_internals + self._motion_start_times[env_ids, None]).flatten()  # Next frame, so +1
            env_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
            motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps)  # pass in the env_ids such that the motion is in synced.
        else:
            motion_times = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids]
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times)  # pass in the env_ids such that the motion is in synced.
            
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        root_pos = root_states[..., 0:3]
        root_rot = root_states[..., 3:7]
        obj_pos = obj_states[..., None,   0:3]
        obj_rot = obj_states[...,  None,  3:7]
        obj_lin_vel = obj_states[..., None,  7:10]
        obj_ang_vel = obj_states[..., None,  10:13]
        
        if self.cfg.env.has_im_obs:
            grab_obs = compute_grab_observations(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
            # One step for im_obs. 
            motion_times = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids]
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times)  # pass in the env_ids such that the motion is in synced.
            
            ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
            body_pos_subset = body_pos[..., self._track_bodies_id, :]
            body_rot_subset = body_rot[..., self._track_bodies_id, :]
            body_vel_subset = body_vel[..., self._track_bodies_id, :]
            body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
            im_obs = humanoid_im.compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 1, self._has_upright_start)
            
            obs = torch.cat([grab_obs, im_obs], dim=-1)
        else:
            obs = compute_grab_observations(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        
        if self.cfg.env.get("contact_obs", False):
            im_obs = compute_contact_force_obs(root_pos, root_rot, contact_forces[:, self._contact_sensor_body_ids], self._has_upright_start)
            obs = torch.cat([obs, im_obs], dim=-1)
        
        return obs

    def _compute_reward(self, actions):
        obj_pos = self._obj_states[..., 0:3]
        obj_rot = self._obj_states[..., 3:7]
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        
        obj_pos = self._obj_states[..., None,   0:3]
        obj_rot = self._obj_states[...,  None,  3:7]
        obj_lin_vel = self._obj_states[..., None,  7:10]
        obj_ang_vel = self._obj_states[..., None,  10:13]
        obj_contact_forces = self._obj_contact_forces
        
        hand_pos = self._rigid_body_pos[:, self._grab_body_ids, :]
        
        motion_times = self.progress_buf * self.dt + self._motion_start_times 
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, None) 
        
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        ref_hand_pos = ref_rb_pos[:, self._grab_body_ids, :]
        hand_contact_force = self._contact_forces[:, self._grab_body_ids, :]
        
        
        grab_reward, grab_reward_raw  = compute_grab_reward(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, hand_contact_force, obj_contact_forces, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, hand_pos, ref_hand_pos, self.reward_specs)
        if self.cfg.env.get("im_reward", False):
            im_reward, im_rewad_raw = humanoid_im.compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel, self.reward_specs_im)
            self.rew_buf[:], self.reward_raw = im_reward * 0.3 + grab_reward * 0.7, torch.cat([grab_reward_raw, im_rewad_raw], dim=-1)
        else:
            self.rew_buf[:], self.reward_raw =  grab_reward * 1, torch.cat([grab_reward_raw], dim=-1)
        return

    def _compute_reset(self):
        time = (self.progress_buf) * self.dt + self._motion_start_times 
        pass_time = time >= self._motion_lib._motion_lengths
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time)

        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        obj_pos = self._obj_states[..., None, 0:3]
        obj_rot = self._obj_states[..., None, 3:7]
        hand_pos = self._rigid_body_pos[:, self._grab_body_ids, :]
        grab_reset, grab_terminate = compute_humanoid_grab_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, \
                                                                               obj_pos, obj_rot,  ref_o_rb_pos, ref_o_rb_rot,  hand_pos, pass_time, self._enable_early_termination,
                                                                               self.grab_termination_disatnce, flags.no_collision_check, flags.im_eval and (not self.strict_eval))
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, grab_reset), torch.logical_or(self._terminate_buf, grab_terminate)

        if self.cfg.env.get("im_reset", False):
            ref_rb_pos = motion_res['rg_pos']
            body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
            ref_body_pos = ref_rb_pos[..., self._reset_bodies_id, :].clone()
            im_reset, im_terminate = humanoid_im.compute_humanoid_im_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, \
                                                                                body_pos, ref_body_pos, pass_time, self._enable_early_termination,
                                                                                self._termination_distances[..., self._reset_bodies_id], flags.no_collision_check, flags.im_eval and (not self.strict_eval))
            
            self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, im_reset), torch.logical_or(self._terminate_buf, im_terminate)
        
        is_recovery = torch.logical_and(~pass_time, self._cycle_counter > 0)  # pass time should override the cycle counter.
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        
        return

    def _reset_ref_state_init(self, env_ids):
        self._cycle_counter[env_ids] = 0 # always give 60 frames for it to pick up and catch up. 
        super()._reset_ref_state_init(env_ids)  # This function does not use the offset
        return
    
    def _update_cycle_count(self):
        self._cycle_counter -= 1
        self._cycle_counter = torch.clamp_min(self._cycle_counter, 0)
        return
    
    def pre_physics_step(self, actions):

        super().pre_physics_step(actions)
        self._update_cycle_count()

        if self._occl_training:
            self._update_occl_training()

        return
    
    def _build_termination_heights(self):
        super()._build_termination_heights()
        termination_distance = self.cfg["env"].get("terminationDistance", 0.5)
        if type(termination_distance) is list:
            assert(len(termination_distance) == self.num_bodies)
            self._termination_distances = to_torch(np.array(termination_distance), device=self.device)
        else:
            self._termination_distances = to_torch(np.array([termination_distance] * self.num_bodies), device=self.device)
            
        self.grab_termination_disatnce = to_torch(np.array(self.cfg["env"].get("grab_termination_distance", 0.5)), device=self.device)
        return
    
    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._obj_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)
            
        
        motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times 
        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times)
        o_rb_pos = motion_res['o_rb_pos'][:, :1].cpu().numpy()
        rg_pos = motion_res['rg_pos'].cpu().numpy()
        
        for env_id in range(self.num_envs):
            sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04, 5, 5, None, color=(0.0, 1, 0.0) )
            sphere_pose = gymapi.Transform(gymapi.Vec3(o_rb_pos[env_id, :, 0], o_rb_pos[env_id, :, 1], o_rb_pos[env_id, :, 2]), r=None)
            gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
            
            if flags.show_traj:
                for jt_num in range(rg_pos.shape[1]):
                    sphere_geom_marker = gymutil.WireframeSphereGeometry(0.005, 5, 5, None, color=(1.0, 0, 0.0) )
                    sphere_pose = gymapi.Transform(gymapi.Vec3(rg_pos[env_id, jt_num, 0], rg_pos[env_id, jt_num, 1], rg_pos[env_id, jt_num, 2]), r=None)
                    gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                
        return

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_pos')):
            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        tar_pos = self._obj_states[0, 0:3].cpu().numpy()
        self._output_motion_target_pos.append(tar_pos)

        tar_rot = self._obj_states[0, 3:7].cpu().numpy()
        self._output_motion_target_rot.append(tar_rot)
        
        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_pos) > 1):
            output_tar_pos = np.array(self._output_motion_target_pos)
            output_tar_rot = np.array(self._output_motion_target_rot)
            output_data = np.concatenate([output_tar_pos, output_tar_rot], axis=-1)
            np.save('output/record_tar_motion.npy', output_data)

            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        return
 


class HumanoidGrabZ(HumanoidGrab):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        super().step_z(actions)
        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        
        # if self.cfg.env.res_hand:
            # self._num_actions += 
        return
    
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_grab_observations(root_pos, root_rot, o_pos, o_rot, o_lin_vel, o_ang_vel, ref_o_pos, ref_o_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = o_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_o_pos.view(B, time_steps, J, 3) - o_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_o_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(o_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_o_ang_vel.view(B, time_steps, J, 3) - o_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_o_body_pos = o_pos.view(B, 1, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))

    local_o_body_rot = torch_utils.quat_mul(heading_inv_rot.view(-1, 4), o_rot.view(-1, 4))
    local_o_body_rot = torch_utils.quat_to_tan_norm(local_o_body_rot)

    # make some changes to how futures are appended.
    
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_rot.view(B, -1))  # timestep  * 24 * 6
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

@torch.jit.script
def compute_contact_force_obs(root_pos, root_rot, contact_forces_subset, upright):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor
    B, J, _ = contact_forces_subset.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1))
    contact_force_local = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), contact_forces_subset.view(-1, 3)).view(B, -1)
    
    return contact_force_local
    

@torch.jit.script
def compute_grab_reward(root_pos, root_rot, obj_pos, obj_rot, obj_vel, obj_ang_vel, hand_contact_force, obj_contact_forces, ref_obj_pos, ref_obj_rot, ref_body_vel, ref_body_ang_vel, hand_pos, ref_hand_pos, rwd_specs):
     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    k_pos, k_rot, k_vel, k_ang_vel = 100, 10, 0.1, 0.1
    # w_pos, w_rot, w_vel, w_ang_vel = 0.4, 0.3, 0.05, 0.05
    w_pos, w_rot, w_vel, w_ang_vel = 0.25, 0.25, 0.05, 0.05
    # w_cos,  w_dist= 0.1, 0.1
    k_cos, k_dist = 50, 50
    # w_conctact, w_close = 0.2, 0
    w_conctact, w_close = 0.1, 0
    
    # object position tracking reward
    diff_global_body_pos = ref_obj_pos - obj_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_obj_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # object rotation tracking reward
    diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot, torch_utils.quat_conjugate(obj_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_obj_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # object linear velocity tracking reward
    diff_global_vel = ref_body_vel - obj_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_lin_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # object angular velocity tracking reward
    diff_global_ang_vel = ref_body_ang_vel - obj_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    obj_contact_force_sum = obj_contact_forces.sum(dim = -2).abs().sum(dim = -1) > 0
    hand_pos_diff = (hand_pos - obj_pos).norm(dim=-1, p = 2)
    
    table_no_contact = obj_contact_forces[:, -1].abs().sum(dim = -1) == 0 
    obj_has_contact = obj_contact_forces[:, 0].abs().sum(dim = -1) > 0 
    object_lifted = torch.logical_and(obj_has_contact, table_no_contact)
    
    pos_filter = (hand_pos_diff < 0.1).sum(dim = -1) > 0
    vel_filter = (torch.norm(obj_vel, dim= -1, p = 2) > 0.01)[:, 0]
    
    vel_filter = torch.logical_or(vel_filter, object_lifted) # velocity means the object is moved. The object lifted is for when the object is mid-air and stationary. 
    
    contact_filter = torch.logical_and(obj_contact_force_sum, torch.logical_and(pos_filter, vel_filter)) # 
    r_contact_lifted = contact_filter.float() 
    
    
    # print(vel_filter, contact_filter)
    
    # if contact_filter.sum() > 0:
        # import ipdb; ipdb.set_trace()

    # # r_close = torch.exp(-k_pos * (hand_pos_diff.min(dim = -1).values **2))

    # ##### pos_filter makes sure that no reward is given if the hand is too far from the object.
    # # reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  + r_close * w_close
    # reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  
    # # reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel, r_close], dim=-1)
    # reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel], dim=-1)

    w_cos,  w_dist= 0.15, 0.15
    k_cos, k_dist = 50, 50
    B, H, C = hand_pos.shape

    ref_hand_pos_diff = (ref_hand_pos - ref_obj_pos)
    hand_pos_diff = (hand_pos - obj_pos)
    ref_diff_norm = ref_hand_pos_diff.norm(dim = -1, keepdim = True, p = 2)
    hand_diff_norm = hand_pos_diff.norm(dim = -1, keepdim = True, p = 2)
    # pos_filter = (ref_hand_pos_diff.norm(dim=-1) < 0.03).sum(dim = -1) > 0
    diff_pos_diff = (ref_hand_pos_diff - hand_pos_diff)

    normalized_reference_vector = ref_hand_pos_diff / ref_diff_norm
    normalized_current_vector = hand_pos_diff / hand_diff_norm
    cosine_similarity = normalized_current_vector.view(B, H, 1, C) @  normalized_reference_vector.view(B, H, C, 1)
    cosine_similarity = cosine_similarity.squeeze(-1).squeeze(-1)

    
    r_ref_v_dist = torch.exp(-k_dist * diff_pos_diff.norm(dim=-1, p = 2).mean(dim=-1))
    r_ref_v_cos = torch.exp(-k_cos * (1 - cosine_similarity).mean(dim=-1))

    # reward = w_dist * r_ref_v_dist + w_cos * r_ref_v_cos
    # reward_raw = torch.stack([r_ref_v_dist, r_ref_v_cos], dim=-1)
    
    reward = w_dist * r_ref_v_dist + w_cos * r_ref_v_cos + (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  
    reward_raw = torch.stack([r_ref_v_dist, r_ref_v_cos], dim=-1)
    
    return reward, reward_raw

@torch.jit.script
def compute_humanoid_grab_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, obj_pos, obj_rot, ref_obj_pos, ref_obj_rot, hand_pos, pass_time, enable_early_termination, termination_distance, disableCollision, use_mean):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool, bool) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)
    if (enable_early_termination):
        if use_mean:
            has_fallen = torch.any(torch.norm(obj_pos - ref_obj_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0], dim=-1)  # using average, same as UHC"s termination condition
        else:
            has_fallen = torch.any(torch.norm(obj_pos - ref_obj_pos, dim=-1) > termination_distance, dim=-1)  # using max
            
        distance = ((hand_pos - obj_pos)**2).mean(dim=-1)
        min_distance = distance.min(dim = -1).values
        has_fallen = torch.logical_or(has_fallen, min_distance > 0.5) # Hand should be really close to the object. Hard coded for now. 

        diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot, torch_utils.quat_conjugate(obj_rot))
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        has_fallen = torch.logical_or(has_fallen, diff_global_body_angle_dist > 1) # Hand should be really close to the object. Hard coded for now. 
        
        
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        if disableCollision:
            has_fallen[:] = False
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
        
        # if (contact_buf.abs().sum(dim=-1)[0] > 0).sum() > 2:
        #     np.set_printoptions(precision=4, suppress=1)
        #     print(contact_buf.numpy(), contact_buf.abs().sum(dim=-1)[0].nonzero().squeeze())
        #     import ipdb; ipdb.set_trace()
            
        # import ipdb; ipdb.set_trace()
        # if terminated.sum() > 0:
        #     import ipdb; ipdb.set_trace()
        #     print("Fallen")

    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)
    # import ipdb
    # ipdb.set_trace()

    return reset, terminated