import torch

from typing import OrderedDict
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
import phc.utils.traj_generator_3d as traj_generator_3d
from phc.utils.draw_utils import agt_color, get_color_gradient
import gc
import time
import cv2

TAR_ACTOR_ID = 1

class HumanoidOmniGrasp(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.load_humanoid_configs(cfg)
        self._fut_tracks = cfg["env"].get("fut_tracks", False)
        if self._fut_tracks:
            self._num_traj_samples = cfg["env"]["numTrajSamples"]
        else:
            self._num_traj_samples = 1
        self._traj_sample_timestep = 1 / cfg["env"].get("trajSampleTimestepInv", 30)
        self.reward_specs = cfg["env"].get("reward_specs", {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1})

        self.reward_specs_im = {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0, "w_rot": 0.7, "w_vel": 0.15, "w_ang_vel": 0.15}
        
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
        self.hand_onehot = torch.zeros([self.num_envs, 2]).float().to(self.device)
        self.start_idx = 0
        self._target_assets = {}
        self.bounding_box = torch.tensor([[-0.1000, -0.1000, -0.1000], [ 0.1000, -0.1000, -0.1000], [ 0.1000,  0.1000, -0.1000], [-0.1000,  0.1000, -0.1000], [-0.1000, -0.1000,  0.1000], [ 0.1000, -0.1000,  0.1000], [ 0.1000,  0.1000,  0.1000], [-0.1000,  0.1000,  0.1000]]).float().to(self.device)
        self.bounding_box_batch = self.bounding_box.unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        # ZL Debugging putting objects. Fix asap. Very very very very janky 
        data_seq = joblib.load(cfg.env.motion_file)
        self.data_key = data_key = list(data_seq.keys())[0]
        
        self.gender  = data_seq[data_key]["gender"]
        self.gender_number = torch.zeros(1)
        if self.gender == "neutral": self.gender_number[:] = 0
        if self.gender == "male": self.gender_number[:] = 1
        if self.gender == "female": self.gender_number[:] = 2
        
        self.has_data = True
        self.obj_names, self.env_to_obj_name, self.obj_name_to_code, self.obj_asset_root = [], {}, {}, {}
        for data_key in list(data_seq.keys()):
            obj_name = data_seq[data_key]["obj_data"]["obj_info"][0].split("/")[-1].split(".")[0]
            self.obj_names.append(obj_name)
            self.obj_name_to_code[obj_name] = data_seq[data_key]["obj_data"]['object_code']
            
            if "oakink" in data_seq[data_key]["obj_data"]["obj_info"][0] or "omomo" in data_seq[data_key]["obj_data"]["obj_info"][0]:
                self.obj_asset_root[obj_name] = "/".join(data_seq[data_key]["obj_data"]["obj_info"][0].split("/")[:-1]) 
                self.has_data = False
            else:
                self.obj_asset_root[obj_name] = "phc/data/assets/urdf/grab/"
        
        
        self.obj_names_current_envs = list(self.obj_names)[:self.num_envs] # only load the same number of objects as the number of envs.
        self.obj_names_unique = sorted(list(set(self.obj_names))) # all unique object names
        # self.obj_names_unique = list(self.obj_names) # all unique object names
        
        
        if cfg.env.get("tile_env", False):
            self.obj_names_current_envs = [self.obj_names_unique[i % len(self.obj_names_unique)] for i in range(self.num_envs)]
        
        self.obj_names_unique_load = sorted(list(set(self.obj_names_current_envs))) # unique object names for loading 
        
        self.obj_ids_unique_load = torch.tensor([self.obj_names_unique.index(name) for name in self.obj_names_unique_load])
        self.data_seq_to_obj_id = torch.tensor([self.obj_names_unique.index(name) for name in self.obj_names])
        self.env_to_obj_name = [self.obj_names_current_envs[i % len(self.obj_names_current_envs)] for i in range(self.num_envs)] # 
        self.env_to_obj_id = torch.tensor([self.obj_names_unique.index(name) for name in self.env_to_obj_name])
        
        if self.cfg.env.get("add_noise", False):
            noise_scale = self.cfg.env.get("noise_scale", 0.01)
            self.env_to_obj_code = torch.cat([self.obj_name_to_code[self.env_to_obj_name[i]] + (2 * torch.rand_like(self.obj_name_to_code[obj_name]) - 1) * noise_scale for i in range(self.num_envs)], dim = 0).to(self.device)
        else:
            self.env_to_obj_code = torch.cat([self.obj_name_to_code[self.env_to_obj_name[i]] for i in range(self.num_envs)], dim = 0).to(self.device)
        
        
        self.v_template = torch.from_numpy(data_seq[data_key]['v_template']).float()

        self.table_remove_frame = torch.zeros(self.num_envs).to(self.device) # 45 frames, 1.5 second
        self.table_remove_frame[:] = cfg["env"].get("table_remove_frame", 45)
        self.grasp_start_frame = cfg["env"].get("grasp_start_frame", 30)
        self.check_rot_reset = cfg["env"].get("check_rot_reset", False)
        self.close_distance_pregrasp = cfg["env"].get("close_distance_pregrasp", 0.2)
        self.close_distance_contact = cfg["env"].get("close_distance_contact", 0.1)
        
        print({k:v for k,v in np.asarray(np.unique(self.env_to_obj_name, return_counts=True)).T})
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._build_traj_generator()
        self._sampled_motion_ids = self.all_env_ids
        self._obj_contact_forces = self.contact_force_tensor.view(self.num_envs, self.bodies_per_env, 3)[..., -2:, :] # gater both table and object contact forces.
        
        self._cycle_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.eval_time_coutner =torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._hand_body_ids = self._build_key_body_ids_tensor(cfg.env.hand_bodies)
        self._fingertip_body_ids = self._build_key_body_ids_tensor(cfg.env.fingertip_bodies)
        self._hand_pos_prev = torch.zeros((self.num_envs, len(self._hand_body_ids), 3)).to(self.device)    
        self._hand_dof_ids = [[self._dof_names.index(hand_name) * 3, self._dof_names.index(hand_name) * 3 + 1, self._dof_names.index(hand_name) * 3 + 2] for hand_name in self.cfg.env.hand_bodies]
        self._hand_dof_ids = torch.from_numpy(np.concatenate(self._hand_dof_ids)).to(self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self._contact_body_ids), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, len(self._contact_body_ids), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        
        self.head_idx = self._build_key_body_ids_tensor(['Head'])
        self.eye_offset = [0.0, 0.075, 0.1]
        if self.cfg.env.get("use_image", False):
            print("Setting up camera")
            camera_config = self.cfg.env.get("camera_config", {})
            camera_config["width"] = camera_config.get("width", 32)
            camera_config["height"] = camera_config.get("height", 32)
            self.image_tensor = torch.zeros([self.num_envs, camera_config["height"], camera_config["width"]   , 3], device=self.device, dtype=torch.float32)
            self.setup_camera(camera_config)
        
        return
    
    
    
    def _setup_tensors(self):
        super()._setup_tensors()
        self._build_target_tensors()

    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length * self.dt
        num_verts = int(self.max_episode_length/2)
        
        
        if self.cfg.env.get("use_orig_traj", False):
            import phc.utils.traj_generator_3d_orig as traj_generator_3d_orig
            
            self._traj_gen = traj_generator_3d_orig.TrajGenerator3D(num_envs, episode_dur, num_verts, self.device, self.cfg.env.traj_gen, starting_still_dt=self.table_remove_frame * self.dt)
        else:
            self._traj_gen = traj_generator_3d.TrajGenerator3D(num_envs, episode_dur, num_verts, self.device, self.cfg.env.traj_gen, starting_still_dt=self.table_remove_frame * self.dt)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._traj_gen.reset(env_ids, init_pos = self._obj_states[:, 0:3], init_rot = self._obj_states[:, 3:7], time_till_table_removal=self.table_remove_frame.max() * self.dt)

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
        
    def forward_motion_samples(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=False, start_idx=self.start_idx)
        self.reset()
        
    def recreate_sim(self, failed_keys, epoch=0):
        if self.cfg.env.obj_pmcp and epoch >= self.cfg.env.get("obj_pmcp_start", 999):
            print("############################### Recreating Sim ################################")
            
            # if True:
            if epoch // self.cfg.learning.params.config.eval_frequency % 2 == 0:
                print("Extreme::::::")
                tau = 500
                per_obj_fail_rate = self._motion_lib.compute_per_obj_fail_rate(failed_keys)
                obj_fail_array = np.array([per_obj_fail_rate[n] for n in self.obj_names_unique])
                all_prob = np.exp(tau * np.array(obj_fail_array))
                max_idx = all_prob.argmax()
                all_prob[:] = 0
                all_prob[max_idx] = 1
                all_prob = all_prob/all_prob.sum()
                
                obj_multiplier = 1
                if len(self.obj_names) < 200 and self.num_envs > len(self.obj_names) * 10:
                    obj_multiplier = 10

                num_available_envs = self.num_envs - len(self.obj_names) * obj_multiplier
                new_env_to_obj_ids = np.random.choice(np.arange(len(self.obj_names_unique)), num_available_envs, p=all_prob, replace=True)
                new_env_to_obj_ids = np.sort(new_env_to_obj_ids)
                self.env_to_obj_name = self.obj_names * obj_multiplier + [self.obj_names_unique[i] for i in new_env_to_obj_ids] # 
            else:
                print("Random::::::")
                obj_multiplier = 1
                num_available_envs = self.num_envs - len(self.obj_names) * obj_multiplier
                new_env_to_obj_ids = np.array([i % len(self.obj_names_unique) for i in range(num_available_envs)])
                new_env_to_obj_ids = np.sort(new_env_to_obj_ids)
                self.env_to_obj_name = self.obj_names * obj_multiplier + [self.obj_names_unique[i] for i in new_env_to_obj_ids] # 

                
            print("--> new env to obj name")
            print({k:v for k,v in np.asarray(np.unique(self.env_to_obj_name, return_counts=True)).T})
            
            if num_available_envs > 0:
                
                if self.cfg.env.get("add_noise", False):
                    noise_scale = self.cfg.env.get("noise_scale", 0.01)
                    self.env_to_obj_id[-num_available_envs:] = torch.from_numpy(new_env_to_obj_ids) # this will also update motion lib's copy
                    self.env_to_obj_code[:] = torch.cat([self.obj_name_to_code[self.env_to_obj_name[i]] + (2 * torch.rand_like(self.obj_name_to_code[self.env_to_obj_name[i]]) - 1) * noise_scale for i in range(self.num_envs)], dim = 0).to(self.device)
                else:
                    self.env_to_obj_id[-num_available_envs:] = torch.from_numpy(new_env_to_obj_ids) # this will also update motion lib's copy
                    self.env_to_obj_code[-num_available_envs:] = torch.cat([self.obj_name_to_code[self.obj_names_unique[new_env_to_obj_ids[i]]] for i in range(num_available_envs)], dim = 0).to(self.device) # this will also update network's copy
                    
                    
            self.gym.destroy_sim(self.sim)
            del self.sim
            if not self.headless:
                self.gym.destroy_viewer(self.viewer)
            torch.cuda.empty_cache()
            gc.collect()
            self.create_sim()
            self.gym.prepare_sim(self.sim)
            self.create_viewer()
            self._setup_tensors()
            self.resample_motions()
                
        
    def resample_motions(self):

        print("Partial solution, only resample motions...")
        # if self.hard_negative:
            # self._motion_lib.update_sampling_weight()

        if flags.test:
            self.forward_motion_samples()
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, limb_weights=self.humanoid_limb_and_weights.cpu(), gender_betas=self.humanoid_shapes.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                          max_len=-1 if flags.test else self.max_len)  # For now, only need to sample motions since there are only 400 hmanoids

            # self.reset() #
            # print("Reasmpling and resett!!!.")

            time = self.progress_buf * self.dt + self._motion_start_times 
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            # self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res['root_pos'][:, :2]
            self.reset()

    
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
                "im_eval": False, # no im_eval flag for envs that have obj. Otherwise the sampling will be wrong. im_eval sorts the keys by length
                "multi_thread": True ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
                "env_to_obj_id": self.env_to_obj_id,
                "data_seq_to_obj_id": self.data_seq_to_obj_id,
                "obj_ids_unique_load" : self.obj_ids_unique_load,
                "obj_rand_pos": self.cfg.env.get("obj_rand_pos", False),
                "obj_rand_pos_extreme": self.cfg.env.get("obj_rand_pos_extreme", False),
                "close_distance_pregrasp": self.close_distance_pregrasp, 
            })
            
            self._motion_train_lib = MotionLibSMPLObj(motion_lib_cfg)
            self._motion_eval_lib = MotionLibSMPLObj(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions), max_len=-1 if flags.test else self.max_len)

        else:
            self._motion_lib = MotionLib(motion_file=motion_train_file, dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets, device=self.device)

        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            if self.cfg.env.obs_v in [1, 1.5, 2]: 
                obs_size = 15 * self._num_traj_samples + 9 + 10 * 3 # local pos and orientation 10 ; 9 is object position and orientation in root.  ; 10 * 3 for finger to object root position. 
            elif self.cfg.env.obs_v in  [1.4]:
                obs_size = 15 * self._num_traj_samples + 9  # local pos and orientation 10 ; 9 is object position and orientation in root.  ; 10 * 3 for finger to object root position. 
            elif self.cfg.env.obs_v in  [1.6]:
                obs_size = 6 * self._num_traj_samples + 3 + 10 * 3  # local pos and orientation 10 ; 9 is object position and orientation in root.  ; 10 * 3 for finger to object root position. 
            elif self.cfg.env.obs_v in  [3]:
                obs_size = 3 * 8 * self._num_traj_samples + 6 * self._num_traj_samples + 8 * 3 + 10 * 3 # box ref, velocity + angular velocity + box + fingers
            elif self.cfg.env.obs_v in  [4]:
                obs_size = 15 * self._num_traj_samples  # local pos and orientation 10 ; 9 is object position and orientation in root.  
            
            if self.cfg.env.has_im_obs:
                obs_size = obs_size + len(self._track_bodies) * 24 

            if self.cfg.env.get("contact_obs", False):
                obs_size = obs_size + len(self._contact_sensor_body_ids) * 3
                
            if self.cfg.env.get("contact_obs_bi", False):
                obs_size = obs_size + len(self._contact_sensor_body_ids) * 1
                
            if self.cfg.env.get("use_hand_flag", False):
                obs_size = obs_size + 2
                
            if self.cfg.env.get("use_image_obs", False):
                camera_config = self.cfg.env.get("camera_config", {})
                width = camera_config.get("width", 224)
                height = camera_config.get("width", 224)
                obs_size = obs_size + width * height * 3
            
            if self.cfg.env.fixed_latent:
                obs_size += 1
                
        return obs_size
    
    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()
        
        task_obs_detail["fixed_latent"] = self.cfg.env.get("fixed_latent", False)
        task_obs_detail['env_to_obj_code'] = self.env_to_obj_code
        task_obs_detail["res_hand"] = self.cfg.env.get("res_hand", False)
        task_obs_detail["res_hand_dim"] = len(self.cfg.env.hand_bodies) * 3
        task_obs_detail["use_image_obs"] = self.cfg.env.get("use_image_obs", False)
        task_obs_detail["camera_config"] = self.cfg.env.get("camera_config", {})


        return task_obs_detail
    
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
        self._target_assets = {}
        
        for obj_name in self.obj_names_unique_load:
            asset_file = f"{obj_name}.urdf"
            obj_asset_root = self.obj_asset_root[obj_name]

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 1000.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            
            # asset_options.vhacd_params.max_convex_hulls = 20 # NeurIPS submission
            # asset_options.vhacd_params.max_num_vertices_per_ch = 64 # NeurIPS submission
            
            asset_options.vhacd_params.max_convex_hulls = 32 
            asset_options.vhacd_params.max_num_vertices_per_ch = 72
            
            if "oakink" in obj_asset_root:
                asset_options.vhacd_params.resolution = 512 # Oakink object will trigger a werid bug and segfault. 
            elif "omomo" in obj_asset_root:
                # asset_options.vhacd_params.max_convex_hulls = 100
                asset_options.vhacd_params.resolution = 100000
            else:
                asset_options.vhacd_params.resolution = 300000
                
                # Paramters from Jona
                # asset_options.vhacd_params.max_convex_hulls = 32
                # asset_options.vhacd_params.min_volume_per_ch = 0.0001
                # asset_options.vhacd_params.max_num_vertices_per_ch = 72
                # asset_options.vhacd_params.convex_hull_downsampling = 8
                # asset_options.vhacd_params.plane_downsampling = 8
                # asset_options.vhacd_params.resolution = 20000000 
            
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.fix_base_link = False

            _target_asset = self.gym.load_asset(self.sim, obj_asset_root, asset_file, asset_options)
            self._target_assets[obj_name] = _target_asset
        
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
        
        _target_asset = self.gym.load_asset(self.sim, "phc/data/assets/urdf/grab/", asset_file, asset_options_fix)
        self._target_assets['table'] = _target_asset
         
        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_assets[self.env_to_obj_name[env_id]], default_pose, "target", env_id, 0)
        self._target_handles.append(target_handle)
        
        col = agt_color(env_id + 1)
        # col = agt_color(3)
        color_vec = gymapi.Vec3(col[0], col[1], col[2])
        # if flags.real_traj:
            # color_vec = gymapDi.Vec3(1, 0.41015625, 0.703125)
        self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, color_vec)

        target_handle = self.gym.create_actor(env_ptr, self._target_assets['table'], default_pose, "target", env_id, 0) # Table 
        self._target_handles.append(target_handle)
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._obj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._table_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        
        self._all_obj_ids = torch.stack([to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1, to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2], dim = -1)
        self._table_obj_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2
        self._obj_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._all_obj_ids[env_ids].flatten()
        
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
        self._traj_gen.reset(env_ids, init_pos = self._obj_states[env_ids, 0:3], init_rot = self._obj_states[env_ids, 3:7], time_till_table_removal=self.table_remove_frame[env_ids].max() * self.dt)
        
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
            if self.cfg.env.get("flex_start", True):
                motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
                
                if self.cfg.env.get("use_stage_reward", False):
                    motion_times = torch.clamp(motion_times, torch.zeros_like(motion_times),  self._motion_lib.get_contact_time(self._sampled_motion_ids[env_ids]) - 1) # Start before contact happens.
                else:
                    motion_times = torch.clamp(motion_times, torch.zeros_like(motion_times),  self._motion_lib.get_contact_time(self._sampled_motion_ids[env_ids]) - 0.5) # Start before contact happens.
                    
                if flags.test:
                    motion_times[:] = 0
            elif self.cfg.env.get("contact_start", False):
                motion_times = self._motion_lib.get_contact_time(self._sampled_motion_ids[env_ids])
            else:
                motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        
        
            
        if self.cfg.env.get("use_stage_reward", False):
            self.table_remove_frame[env_ids] = (self._motion_lib.get_contact_time(self._sampled_motion_ids[env_ids]) - motion_times)/self.dt # update table remove frame based on the sample start time of the motion. 0.25 is the raise time of the object. Bascially, the object will be raised within 0.25 second, and the table should be safe to be removed. 
        
        if self.humanoid_type in ["smpl", "smplh", "smplx"] :
            
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        else:
            raise NotImplementedError
            
        motion_times[:] = 0 # Even though you sampled at a certain time, the internal timer should start at 0 
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
            motion_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
            
        else:
            motion_times_steps = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids]
            motion_ids_steps = self._sampled_motion_ids[env_ids]
            time_steps = 1
        
        if flags.im_eval and self.has_data:
            motion_res = self._get_state_from_motionlib_cache(motion_ids_steps, motion_times_steps)
        else:
            motion_res = self._traj_gen.get_motion_state(motion_ids_steps, motion_times_steps)
        
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        
        root_pos = root_states[..., 0:3]
        root_rot = root_states[..., 3:7]
        obj_pos = obj_states[..., None,   0:3]
        obj_rot = obj_states[...,  None,  3:7]
        obj_lin_vel = obj_states[..., None,  7:10]
        obj_ang_vel = obj_states[..., None,  10:13]
        
        finger_pos = body_pos[:, self._fingertip_body_ids, :]
        finger_rot = body_rot[:, self._fingertip_body_ids, :]

        
        if self.obs_v == 1:
            obs = compute_grab_observations(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        elif self.obs_v == 1.4:
            obs = compute_grab_observations_v1_4(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        elif self.obs_v == 1.5:
            obs = compute_grab_observations_v1_5(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        elif self.obs_v == 1.6:
            obs = compute_grab_observations_v1_6(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        elif self.obs_v == 2:
            obs = compute_grab_observations_v2(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
        elif self.obs_v == 3:
            obs = compute_grab_observations_v3(root_pos, root_rot, finger_pos, finger_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, self.bounding_box_batch[env_ids], time_steps, self._has_upright_start)
        elif self.obs_v == 4:
            obs = compute_grab_observations_v4(root_pos, root_rot, finger_pos, finger_rot, ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, time_steps, self._has_upright_start)
    
        if self.cfg.env.get("contact_obs", False):
            contact_forces_fingers  = contact_forces[:, self._contact_sensor_body_ids]
            if self.cfg.env.get("normalize_contact", False):
                norm = contact_forces_fingers.norm(dim = -1, keepdim = True)
                contact_forces_fingers /= (norm + 1e-6)
                contact_forces_fingers *= torch.log(norm+ 1) # log scale forces
                
                
            contact_obs = compute_contact_force_obs(root_pos, root_rot, contact_forces_fingers, self._has_upright_start)
            obs = torch.cat([obs, contact_obs], dim=-1)
            if self.cfg.env.get("contact_obs_bi", False):
                import ipdb; ipdb.set_trace()
                print('shouldnt have both')
                
        elif self.cfg.env.get("contact_obs_bi", False):
            contact_forces_fingers  = contact_forces[:, self._contact_sensor_body_ids]
            contact_obs = (contact_forces_fingers.abs().sum(dim=-1) > 0).float()
            obs = torch.cat([obs, contact_obs], dim=-1)
            
        if self.cfg.env.get("use_hand_flag", False):
            use_hand_label = self._motion_lib.get_use_hand_label(self._sampled_motion_ids[env_ids])
            
            if flags.trigger_input:
                use_hand_label[:, 0] = 1
                use_hand_label[:, 1] = 1
                print(use_hand_label)
                
            
            obs = torch.cat([obs, use_hand_label], dim=-1)
            
        if self.cfg.env.get("use_image", False):
            self.render_camera(env_ids)
            if self.cfg.env.get("use_image_obs", False):
                obs = torch.cat([obs, self.image_tensor[env_ids].reshape(B, -1)], dim=-1)
        
        if self.cfg.env.fixed_latent:
            obs = torch.cat([obs, env_ids[:, None]], dim=-1)
            
            
        return obs
    
    def get_running_mean_size(self):
        if self.cfg.env.fixed_latent:
            return (self.get_obs_size() - 1, ) # need multi object index to not be clipped
        else:
            return (self.get_obs_size() , )

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
        
        hand_pos = self._rigid_body_pos[:, self._hand_body_ids, :]
        hand_rot = self._rigid_body_rot[:, self._hand_body_ids, :]
        hand_vel = self._rigid_body_vel[:, self._hand_body_ids, :]
        hand_ang_vel = self._rigid_body_ang_vel[:, self._hand_body_ids, :]
        
        motion_times = self.progress_buf * self.dt + self._motion_start_times 
        
        motion_res = self._traj_gen.get_motion_state(self._sampled_motion_ids, motion_times)
        
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        hand_contact_force = self._contact_forces[:, self._hand_body_ids, :]
        table_removed_flag = self.progress_buf > self.table_remove_frame
        table_removed = self.all_env_ids[table_removed_flag]
        
        contact_filter = check_contact(hand_contact_force, obj_contact_forces, hand_pos, obj_pos, obj_lin_vel, table_removed, self.close_distance_contact)
        
        
        if self.cfg.env.get("r_v", 1)  in [1]:
            grab_reward, grab_reward_raw  = compute_grab_reward(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel,  ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel,  contact_filter, self.reward_specs)
        if self.cfg.env.get("r_v", 1.5)  in [1.5]:
            assert(self.cfg.env.get("use_stage_reward", False) and  self.reward_specs['w_pos'] == 0.6)
            grab_reward, grab_reward_raw  = compute_grab_reward_v15(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel,  ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel,  contact_filter, self.reward_specs)
        elif self.cfg.env.get("r_v", 1)  in [2]:
            grab_reward, grab_reward_raw  = compute_grab_reward_v2(root_pos, root_rot, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel,  ref_o_rb_pos, ref_o_rb_rot, ref_o_lin_vel, ref_o_ang_vel, self.bounding_box_batch, contact_filter, self.reward_specs)
        
        
        if self.cfg.env.get("pregrasp_reward", True):
            if self.cfg.env.get("use_stage_reward", False): # contact based 
                contact_hand_dict = self._motion_lib.get_contact_hand_pose(self._sampled_motion_ids)
                ref_contact_hand_pos, ref_contact_hand_rot, ref_contact_hand_vel, ref_contact_hand_ang_vel, contact_ref_obj_pos = contact_hand_dict['contact_hand_trans'],  contact_hand_dict['contact_hand_rot'], contact_hand_dict['contact_hand_vel'], contact_hand_dict['contact_hand_ang_vel'], contact_hand_dict['contact_ref_obj_pos']
                pregrasp_reward, pregrasp_reward_raw = compute_pregrasp_reward_contact(root_pos, root_rot, hand_pos, hand_rot, hand_vel, hand_ang_vel, ref_contact_hand_pos, ref_contact_hand_rot, ref_contact_hand_vel, ref_contact_hand_ang_vel,  contact_ref_obj_pos, obj_pos, self._hand_pos_prev, self.close_distance_pregrasp,  table_removed_flag, self.reward_specs)
                grab_reward[~contact_filter] = pregrasp_reward[~contact_filter]
                grab_reward[contact_filter] += 0.5 # Advance stage reward. 
                
            else: # Time based. 
                contact_hand_dict = self._motion_lib.get_contact_hand_pose(self._sampled_motion_ids)
                ref_contact_hand_pos, ref_contact_hand_rot, ref_contact_hand_vel, ref_contact_hand_ang_vel, contact_ref_obj_pos = contact_hand_dict['contact_hand_trans'],  contact_hand_dict['contact_hand_rot'], contact_hand_dict['contact_hand_vel'], contact_hand_dict['contact_hand_ang_vel'], contact_hand_dict['contact_ref_obj_pos']
                pregrasp_reward, pregrasp_reward_raw = compute_pregrasp_reward_time(root_pos, root_rot, hand_pos, hand_rot, hand_vel, hand_ang_vel, ref_contact_hand_pos, ref_contact_hand_rot, ref_contact_hand_vel, ref_contact_hand_ang_vel,  contact_ref_obj_pos, self._hand_pos_prev, self.close_distance_pregrasp,  self.reward_specs)
                pass_contact_time = motion_times > self.grasp_start_frame * self.dt
                grab_reward[~pass_contact_time] = pregrasp_reward[~pass_contact_time]

        # if torch.isnan(grab_reward).any():
        #     import ipdb; ipdb.set_trace()
        #     print('....')
            
        self.rew_buf[:], self.reward_raw =  grab_reward , torch.cat([grab_reward_raw], dim=-1)
        
        if self.cfg.env.get("penality_slippage", False):
            slipage = torch.clamp(self._penality_slippage() * self.cfg.env.get("slippage_coefficient", 0.3), 0, 1)
            self.rew_buf[:] = self.rew_buf - slipage
            self.reward_raw = torch.cat([self.reward_raw, -slipage[:, None]], dim=-1)
        
        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
            # power_reward = -0.00005 * (power ** 2)
            power_reward = torch.clamp(-self.power_coefficient * power, -1, 1)
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        if self.cfg.env.get("reward_airtime", False):
            airtime_reward = torch.clamp(self._reward_feet_air_time() * self.cfg.env.get("airtime_coefficient", 0.5), 0, 1)
            self.rew_buf[:] = self.rew_buf + airtime_reward
            self.reward_raw = torch.cat([self.reward_raw, airtime_reward[:, None]], dim=-1)
        
        return

    def _compute_reset(self):
        time = (self.progress_buf) * self.dt + self._motion_start_times 
        
        if flags.im_eval and self.has_data:
            pass_time = time >= self._motion_lib._motion_lengths
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time)
        else:
            pass_time = time >= self.max_episode_length * self.dt
            motion_res = self._traj_gen.get_motion_state(self._sampled_motion_ids, time)


        
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        obj_pos = self._obj_states[..., None, 0:3]
        obj_rot = self._obj_states[..., None, 3:7]
        hand_pos = self._rigid_body_pos[:, self._hand_body_ids, :]
        
        grab_reset, grab_terminate = compute_humanoid_grab_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, \
                                                                               obj_pos, obj_rot,  ref_o_rb_pos, ref_o_rb_rot,  hand_pos, pass_time, self._enable_early_termination,
                                                                               self.grab_termination_disatnce, flags.no_collision_check, self.check_rot_reset and (not flags.im_eval))
        if flags.im_eval:
            if self.has_data:
                motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, torch.zeros_like(time))
            else:
                motion_res = self._traj_gen.get_motion_state(self._sampled_motion_ids, torch.zeros_like(time))

            ref_o_rb_pos_init = motion_res['o_rb_pos'][:, :1]
            obj_hand_has_contact = (self._obj_contact_forces[:, 0].abs() > 1).sum(dim = -1) + (self._contact_forces[:, self._hand_body_ids].abs() > 1).sum(dim=-1).sum(dim=-1) > 0
            lifted = torch.logical_and(((obj_pos - ref_o_rb_pos_init)[..., 2] > 0.03)[:, 0], obj_hand_has_contact)
            
            self.eval_time_coutner[lifted] += 1
            if not "success_lift" in self.__dict__:
                self.success_lift = torch.zeros(self.num_envs).to(self.device).bool()
            
            self.success_lift = torch.logical_or(self.eval_time_coutner > (0.5 / self.dt), self.success_lift) 
            
        
        
        
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, grab_reset), torch.logical_or(self._terminate_buf, grab_terminate)
        
        
        is_recovery = torch.logical_and(~pass_time, self._cycle_counter > 0)  # pass time should override the cycle counter.
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        
        return

    def _reset_ref_state_init(self, env_ids):
        self._cycle_counter[env_ids] = 0 # always give 60 frames for it to pick up and catch up. 
        self.eval_time_coutner[env_ids] = 0
        super()._reset_ref_state_init(env_ids)  # This function does not use the offset
        return
    
    def _update_cycle_count(self):
        self._cycle_counter -= 1
        self._cycle_counter = torch.clamp_min(self._cycle_counter, 0)
        return
    
    def remove_table(self, env_ids = None):
        if env_ids is None:
            env_ids = self.all_env_ids
            
        self._table_states[env_ids, 2] = 100
        env_ids_int32 = self._table_obj_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def debug_obj_pose(self, env_ids = None):
        if env_ids is None:
            env_ids = self.all_env_ids
        
        
        motion_times = self.progress_buf * self.dt + self._motion_start_times 
        motion_res = self._traj_gen.get_motion_state(self._sampled_motion_ids, motion_times)
        ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
        
        
        self._obj_states[env_ids, :3] = ref_o_rb_pos[:, 0]
        self._obj_states[env_ids, 3:7] = ref_o_rb_rot[:, 0]
        self._obj_states[env_ids, 7:10] = ref_o_lin_vel[:, 0]
        self._obj_states[env_ids, 10:13] = ref_o_ang_vel[:, 0]
        
        env_ids_int32 = self._obj_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def pre_physics_step(self, actions):
        

        super().pre_physics_step(actions)
        self._update_cycle_count()
        
        self._hand_pos_prev = self._rigid_body_pos[:, self._hand_body_ids, :].clone()

        if self._occl_training:
            self._update_occl_training()

        return
    
    def post_physics_step(self):

        # if (self.progress_buf > 60).sum() > 0:
        #     import ipdb; ipdb.set_trace()
        #     joblib.dump({k: self._obj_states.cpu().numpy()[i] for i, k in  enumerate(self._motion_lib.curr_motion_keys)}, "obj_states_virtual.pkl")
        #     print('....')
            
        if self.save_kin_info: # this needs to happen BEFORE the next time-step observation is computed, to collect the "current time-step target"
            self.extras['kin_dict'] = self.kin_dict
            
        self.remove_table(env_ids=self.all_env_ids[self.progress_buf > self.table_remove_frame ])
        
        
        super().post_physics_step()
        
        if flags.im_eval:
            motion_times = (self.progress_buf) * self.dt + self._motion_start_times   # already has time + 1, so don't need to + 1 to get the target for "this frame"
            
            if self.has_data:
                motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, None)  # pass in the env_ids such that the motion is in synced.
            else:
                motion_res = self._traj_gen.get_motion_state(self._sampled_motion_ids, motion_times)

            body_pos = self._rigid_body_pos
            
            ref_o_ang_vel, ref_o_lin_vel, ref_o_rb_rot, ref_o_rb_pos = motion_res['o_ang_vel'][:, :1], motion_res['o_lin_vel'][:, :1], motion_res['o_rb_rot'][:, :1], motion_res['o_rb_pos'][:, :1]
            
            obj_pos = self._obj_states[..., None, 0:3]
            obj_rot = self._obj_states[..., None, 3:7]
        
            self.extras['body_pos'] = obj_pos.cpu().numpy()
            self.extras['body_pos_gt'] = ref_o_rb_pos.cpu().numpy()
            self.extras['body_rot'] = obj_rot.cpu().numpy()
            self.extras['body_rot_gt'] = ref_o_rb_rot.cpu().numpy()

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
        if not flags.real_traj:
            self.gym.clear_lines(self.viewer)        
            
            
        head_rot = self._rigid_body_rot[:, self.head_idx, :]
        head_pos = self._rigid_body_pos[:, self.head_idx, :]
        eye_offset = torch.tensor([self.eye_offset] * self.num_envs, device=head_rot.device)
        pos_near = head_pos[:, 0] + torch_utils.my_quat_rotate(head_rot[:, 0], eye_offset)
        
        eyesigt_direction = torch.tensor([[0.0, 0, 0.5]] * self.num_envs, device=head_rot.device)
        target = pos_near + torch_utils.my_quat_rotate(head_rot[:, 0], eyesigt_direction)
        
        if not flags.real_traj:
            for env_id in range(self.num_envs):
                line = torch.cat([pos_near[env_id].unsqueeze(0), target[env_id].unsqueeze(0)], dim=0).cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[env_id], 1, line, (1.0, 0, 0.0))
                
                sphere_geom_marker_pos_near = gymutil.WireframeSphereGeometry(0.025, 10, 10, None, color=(1, 0, 0.0))
                sphere_pose_pos_near = gymapi.Transform(gymapi.Vec3(pos_near[env_id, 0], pos_near[env_id, 1], pos_near[env_id, 2]), r=None)
                gymutil.draw_lines(sphere_geom_marker_pos_near, self.gym, self.viewer, self.envs[env_id], sphere_pose_pos_near)
                
                sphere_geom_marker_target = gymutil.WireframeSphereGeometry(0.025, 10, 10, None, color=(0, 1, 0.0))
                sphere_pose_target = gymapi.Transform(gymapi.Vec3(target[env_id, 0], target[env_id, 1], target[env_id, 2]), r=None)
                gymutil.draw_lines(sphere_geom_marker_target, self.gym, self.viewer, self.envs[env_id], sphere_pose_target)
        
            
        if flags.show_traj:
            cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

            starts = self._humanoid_root_states[..., 0:3]
            ends = self._obj_states[..., 0:3]
            verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

            # if not flags.real_traj:
            #     for i, env_ptr in enumerate(self.envs):
            #         curr_verts = verts[i]
            #         curr_verts = curr_verts.reshape([1, 6])
            #         self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)
                
            # import ipdb; ipdb.set_trace()
            contact_hand_trans = self._motion_lib.get_contact_hand_pose(self._sampled_motion_ids)['contact_hand_trans']

            env_ids = self._sampled_motion_ids
            time_steps = self._num_traj_samples
            
            B = env_ids.shape[0]
            time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * 0.1
            
            if flags.real_traj:
                time_steps = 50
                time_internals = -torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * 0.075
            
            
            motion_times_steps = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_internals + self._motion_start_times[env_ids, None]).flatten()  # Future poses
            env_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
                
            if flags.im_eval and self.has_data:
                motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps)  # pass in the env_ids such that the motion is in synced.
            else:
                motion_res = self._traj_gen.get_motion_state(env_ids_steps, motion_times_steps)
                
            o_rb_pos = motion_res['o_rb_pos'].cpu().numpy().reshape(B, time_steps, -1, 3)
        
            for env_id in range(self.num_envs):
                if flags.real_traj:
                    
                    # if self.progress_buf[env_id] > self.table_remove_frame:
                    #     if self.progress_buf[env_id] == self.table_remove_frame + 1:
                    #         self.save_pos = []
                    #     else:
                    #         self.save_pos.append(self._obj_states[0, :3].cpu().numpy())
                        
                    #         sphere_geom_marker = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 0.41015625, 0.703125) )
                            
                    #         for obj_pos in self.save_pos:
                    #             pos = obj_pos
                    #             sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
                    #             gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                        
                        
                    # for time_step in range(time_steps):
                    #     # sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04 * (1 - time_step/len(o_rb_pos)), 5, 5, None, color=(0.0, 1 * (1 - time_step/len(o_rb_pos)), 0.0) )
                    #     sphere_geom_marker = gymutil.WireframeSphereGeometry(0.025, 10, 10, None, color=(0.0, 1, 0.0) )
                    #     sphere_pose = gymapi.Transform(gymapi.Vec3(o_rb_pos[env_id, time_step, 0, 0], o_rb_pos[env_id, time_step, 0, 1], o_rb_pos[env_id, time_step, 0, 2]), r=None)
                    #     gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                    
                    for time_step in range(time_steps):
                        # sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04 * (1 - time_step/len(o_rb_pos)), 5, 5, None, color=(0.0, 1 * (1 - time_step/len(o_rb_pos)), 0.0) )
                        sphere_geom_marker = gymutil.WireframeSphereGeometry(0.025, 10, 10, None, color=(0.0, 1, 0.0) )
                        sphere_pose = gymapi.Transform(gymapi.Vec3(o_rb_pos[env_id, time_step, 0, 0], o_rb_pos[env_id, time_step, 0, 1], o_rb_pos[env_id, time_step, 0, 2]), r=None)
                        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                        break
                        
                else:
                    for time_step in range(time_steps):
                        # sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04 * (1 - time_step/len(o_rb_pos)), 5, 5, None, color=(0.0, 1 * (1 - time_step/len(o_rb_pos)), 0.0) )
                        if time_step == 0:
                            sphere_geom_marker = gymutil.WireframeSphereGeometry(0.025, 10, 10, None, color=(1, 0, 0.0) )
                        else:
                            sphere_geom_marker = gymutil.WireframeSphereGeometry(0.015, 10, 10, None, color=(0.0, 1 * ((time_steps - time_step)/time_steps + 0.5), 0.0) )
                        sphere_pose = gymapi.Transform(gymapi.Vec3(o_rb_pos[env_id, time_step, 0, 0], o_rb_pos[env_id, time_step, 0, 1], o_rb_pos[env_id, time_step, 0, 2]), r=None)
                        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                    
                    # ###### Drawing the boxes. 
                    # obj_pos = self._obj_states[..., 0:3]
                    # obj_rot = self._obj_states[..., 3:7]    
                    # new_boxes = (torch_utils.my_quat_rotate(obj_rot[:, None].repeat(1, 8, 1).view(-1, 4), self.bounding_box_batch .view(-1, 3))).view(-1, 8, 3) + obj_pos[:, None]
                        
                    # for i in range(new_boxes.shape[1]):
                    #     bbox = new_boxes[env_id, i]
                        
                    #     sphere_geom_marker = gymutil.WireframeSphereGeometry(0.015, 10, 10, None, color=(1.0, 0, 0.0) )
                    #     sphere_pose = gymapi.Transform(gymapi.Vec3(bbox[0], bbox[1], bbox[2]), r=None)
                    #     gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                        
                    # new_box = new_boxes[env_id].cpu().numpy()
                    # lines = np.array([
                    #         [new_box[0], new_box[1]],
                    #         [new_box[1], new_box[2]],
                    #         [new_box[2], new_box[3]],
                    #         [new_box[3], new_box[0]],
                    #         [new_box[4], new_box[5]],
                    #         [new_box[5], new_box[6]],
                    #         [new_box[6], new_box[7]],
                    #         [new_box[7], new_box[4]],
                    #         [new_box[0], new_box[4]],
                    #         [new_box[1], new_box[5]],
                    #         [new_box[2], new_box[6]],
                    #         [new_box[3], new_box[7]]
                    # ])
                    # for line in lines:
                    #     self.gym.add_lines(self.viewer, self.envs[env_id], 1, line, (1.0, 0, 0.0))
                    
                
                    # for jt_num in range(contact_hand_trans.shape[1]):
                    #     sphere_geom_marker = gymutil.WireframeSphereGeometry(0.005, 5, 5, None, color=(0.0, 0, 0.0) )
                    #     sphere_pose = gymapi.Transform(gymapi.Vec3(contact_hand_trans[env_id, jt_num, 0], contact_hand_trans[env_id,  jt_num, 1], contact_hand_trans[env_id,  jt_num, 2]), r=None)
                    #     gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                    
            
                
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
    
    
    def _penality_slippage(self):
        foot_vel = self._rigid_body_vel[:, self._contact_body_ids]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self._contact_forces[:, self._contact_body_ids, :], dim=-1) > 1.), dim=1)
 
    def _penality_stumble(self):
        return torch.any(torch.norm(self._contact_forces[:, self._contact_body_ids, :2], dim=2) >\
             5 *torch.abs(self._contact_forces[:, self._contact_body_ids, 2]), dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self._contact_forces[:, self._contact_body_ids, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def setup_camera(self, camera_config):
        self.render_camera_handles = []
        camera_props = gymapi.CameraProperties()
        camera_props.width = camera_config['width']
        camera_props.height = camera_config['height']
        camera_props.enable_tensors = True
        
        for env_ptr in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0))
            self.render_camera_handles.append(camera_handle)
            
        pass
            
            
    def render_camera(self, env_ids):
        head_pos = self._rigid_body_pos[:, self.head_idx]
        head_rot = self._rigid_body_rot[:, self.head_idx]
        obj_pos = self._obj_states[:, 0:3].cpu().numpy()
        start = time.time()
        
        pos = head_pos[:, 0, :]
        # target = obj_pos[env_ids, :].reshape(-1, obj_pos.shape[-1])
        eye_offset = torch.tensor([self.eye_offset] * self.num_envs, device=head_rot.device)
        pos_near = pos + torch_utils.my_quat_rotate(head_rot[:, 0], eye_offset)
        
        eyesigt_direction = torch.tensor([[0.0, 0, 0.5]] * self.num_envs, device=head_rot.device)
        target = pos_near + torch_utils.my_quat_rotate(head_rot[:, 0], eyesigt_direction)
        
        for env_id in env_ids:
            self.gym.set_camera_location(self.render_camera_handles[env_id], self.envs[env_id], gymapi.Vec3(pos_near[env_id][0], pos_near[env_id][1], pos_near[env_id][2]), gymapi.Vec3(target[env_id][0], target[env_id][1], target[env_id][2]))
            
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # todo: THE CAMERA VIEW CHANGE STEP BY STEP

        for env_id in env_ids:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.render_camera_handles[env_id], gymapi.IMAGE_COLOR)
            self.image_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :, :3].float()  
        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        import ipdb; ipdb.set_trace()
        self.gym.end_access_image_tensors(self.sim)
        return 
    
    def render(self, sync_frame_time = False):
        super().render(sync_frame_time=sync_frame_time)
        # if ((not self.headless) ) and self.cfg.get("use_image", False):
        
        # if (not self.headless) and self.cfg.env.get("use_image", False):
        #     ref_image = self.image_tensor.squeeze().numpy()/255
            
        #     if self.num_envs == 1:
        #         cv2.imshow("mono image", ref_image)
        #         cv2.waitKey(1)


            

class HumanoidOmniGraspZ(HumanoidOmniGrasp):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        self.step_z(actions)
        return
    
    def step_z(self, action_z):
        self.action_z = action_z
        if self.cfg.env.get("res_hand", False):
            actions = self.compute_z_actions(action_z[:, :self.cfg.env.embedding_size])
            actions_res_hand = action_z[:, self.cfg.env.embedding_size:]
            actions[:, self._hand_dof_ids] += actions_res_hand * 0
        else:
            actions = self.compute_z_actions(action_z)
        
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        if self.cfg.env.get("res_hand", False):
            self._num_actions += len(self.cfg.env.hand_bodies) * 3
        
        return
    
    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        
        if self.cfg.env.get("res_hand", False):
            # here actions is the hand actions, not actions_z
            actions_res_hand = self.action_z[:, self.cfg.env.embedding_size:]
            
            action_res_reward =  -torch.abs(actions_res_hand).mean(dim = -1)  
            self.rew_buf[:] += action_res_reward
            self.reward_raw = torch.cat([self.reward_raw, action_res_reward[:, None]], dim=-1)
            
        else:
            pass


    
    
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_grab_observations(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = obj_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_obj_pos.view(B, time_steps, J, 3) - obj_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(obj_rot[:, None]).repeat_interleave(time_steps, 1))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_o_ang_vel.view(B, time_steps, J, 3) - o_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    
    ##### Object position and rotation in body frame
    local_o_body_pos = obj_pos.view(B, 1, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))

    local_o_body_rot = torch_utils.quat_mul(heading_inv_rot.view(-1, 4), obj_rot.view(-1, 4))
    local_o_body_rot = torch_utils.quat_to_tan_norm(local_o_body_rot)


    B, J_f, _ = fingertip_pos.shape
    diff_global_finger_to_obj_pos = fingertip_pos - obj_pos
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J_f, 1))
    local_finger_to_obj_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_finger_to_obj_pos.view(-1, 3))

    
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_rot.view(B, -1))  # timestep  * 24 * 6
    obs.append(local_finger_to_obj_pos.view(B, -1))  # 10 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


@torch.jit.script
def compute_grab_observations_v1_4(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # No hand object relative position information. 
    obs = []
    B, J, _ = obj_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_obj_pos.view(B, time_steps, J, 3) - obj_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(obj_rot[:, None]).repeat_interleave(time_steps, 1))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_o_ang_vel.view(B, time_steps, J, 3) - o_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    
    ##### Object position and rotation in body frame
    local_o_body_pos = obj_pos.view(B, 1, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))

    local_o_body_rot = torch_utils.quat_mul(heading_inv_rot.view(-1, 4), obj_rot.view(-1, 4))
    local_o_body_rot = torch_utils.quat_to_tan_norm(local_o_body_rot)
    
    
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_rot.view(B, -1))  # timestep  * 24 * 6
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs



@torch.jit.script
def compute_grab_observations_v1_5(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = obj_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    
    ##### Body position and rotation differences
    diff_global_body_pos = ref_obj_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, J, 3) # Object future trajectory, in the body frame. 
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(root_rot[:, None, None]).repeat_interleave(time_steps, 1))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### reference linear and angular Velocity 
    local_obj_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_o_vel.view(-1, 3))

    local_obj_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_o_ang_vel.view(-1, 3))
    
    ##### Object position and rotation in body frame
    local_o_body_pos = obj_pos.view(B, 1, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))

    local_o_body_rot = torch_utils.quat_mul(heading_inv_rot.view(-1, 4), obj_rot.view(-1, 4))
    local_o_body_rot = torch_utils.quat_to_tan_norm(local_o_body_rot)


    B, J_f, _ = fingertip_pos.shape
    diff_global_finger_to_obj_pos = fingertip_pos - obj_pos
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J_f, 1))
    local_finger_to_obj_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_finger_to_obj_pos.view(-1, 3))

    
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * timestep * 24 * 6
    obs.append(local_obj_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_obj_ang_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_rot.view(B, -1))  # timestep  * 24 * 6
    obs.append(local_finger_to_obj_pos.view(B, -1))  # 10 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

@torch.jit.script
def compute_grab_observations_v1_6(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Rotation only
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = obj_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_obj_pos.view(B, time_steps, J, 3) - obj_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))

    ##### Object position and rotation in body frame
    local_o_body_pos = obj_pos.view(B, 1, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))


    B, J_f, _ = fingertip_pos.shape
    diff_global_finger_to_obj_pos = fingertip_pos - obj_pos
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J_f, 1))
    local_finger_to_obj_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_finger_to_obj_pos.view(-1, 3))

    
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_finger_to_obj_pos.view(B, -1))  # 10 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs



@torch.jit.script
def compute_grab_observations_v2(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Object centric
    obs = []
    B, J, _ = obj_pos.shape
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    # heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # heading_rot = torch_utils.calc_heading_quat(root_rot)
    obj_inv_rot = torch_utils.quat_conjugate(obj_rot)
    
    obj_rot_inv_rot_expand = obj_inv_rot.repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    obj_rot_expand = obj_rot.repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_obj_pos = ref_obj_pos.view(B, time_steps, J, 3) - obj_pos.view(B, 1, J, 3)
    diff_local_obj_pos_flat = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_obj_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot.view(B, time_steps, J, 4), obj_rot_inv_rot_expand.view(B, time_steps, J, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(obj_rot_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), obj_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_o_ang_vel.view(B, time_steps, J, 3) - o_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    
    ##### Body bosition and orientation in object frame
    local_o_body_pos = root_pos.view(B, 1, 1, 3) - obj_pos.view(B, 1, J, 3)   # preserves the body position
    local_o_body_pos = torch_utils.my_quat_rotate(obj_inv_rot.view(-1, 4), local_o_body_pos.view(-1, 3))

    local_o_body_rot = torch_utils.quat_mul(obj_inv_rot.view(-1, 4), root_rot.view(-1, 4))
    local_o_body_rot = torch_utils.quat_to_tan_norm(local_o_body_rot)


    B, J_f, _ = fingertip_pos.shape
    diff_global_finger_to_obj_pos = fingertip_pos - obj_pos
    obj_rot_inv_rot_expand = obj_inv_rot.repeat((1, J_f, 1))
    local_finger_to_obj_pos = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_finger_to_obj_pos.view(-1, 3))

    
    obs.append(diff_local_obj_pos_flat.view(B, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_pos.view(B, -1))  # timestep  * 24 * 3
    obs.append(local_o_body_rot.view(B, -1))  # timestep  * 24 * 6
    obs.append(local_finger_to_obj_pos.view(B, -1))  # 10 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

#### BBox 
@torch.jit.script
def compute_grab_observations_v3(root_pos, root_rot, fingertip_pos, fingertip_rot, obj_pos, obj_rot, o_lin_vel, o_ang_vel, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, bounding_boxes, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, Tensor, int, bool) -> Tensor
    # Object centric
    obs = []
    B, J, _ = obj_pos.shape
    bbox_j = 8
    if not upright:
        root_rot = remove_base_rot(root_rot)
    
    obj_inv_rot = torch_utils.quat_conjugate(obj_rot)
    
    obj_rot_inv_rot_expand = obj_inv_rot.repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    obj_rot_inv_rot_expand_bbox = obj_inv_rot.repeat((1, bbox_j, 1))
    obj_rot_inv_rot_expand_bbox_timestep = obj_rot_inv_rot_expand_bbox.repeat_interleave(time_steps, 0)
    
    # new_boxes = (torch_utils.my_quat_rotate(obj_rot[:, None].repeat(1, 8, 1).view(-1, 4), self.bounding_box_batch .view(-1, 3))).view(-1, 8, 3) + obj_pos[:, None]
    boungding_box_obj = torch_utils.my_quat_rotate(obj_rot.repeat(1, 8, 1).view(-1, 4), bounding_boxes.view(-1, 3)).view(B, 8, 3) + obj_pos
    boungding_box_ref = torch_utils.my_quat_rotate(ref_obj_rot.repeat(1, 8, 1).view(-1, 4), bounding_boxes.repeat(time_steps, 1, 1).view(-1, 3)).view(-1, 8, 3) + ref_obj_pos
    
    ##### Body position and rotation differences
    diff_global_obj_pos = boungding_box_ref.view(B, time_steps, bbox_j, 3) - boungding_box_obj.view(B, 1, bbox_j, 3)
    diff_local_bbox_flat = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand_bbox_timestep.view(-1, 4), diff_global_obj_pos.view(-1, 3))
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_o_vel.view(B, time_steps, J, 3) - o_lin_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_o_ang_vel.view(B, time_steps, J, 3) - o_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    
    ##### Body bosition and orientation in object frame
    local_obj_bbox_pos = root_pos.view(B, 1, 3) - boungding_box_obj   # preserves the body position
    local_obj_bbox_pos = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand_bbox.view(-1, 4), local_obj_bbox_pos.view(-1, 3))

    B, J_f, _ = fingertip_pos.shape
    diff_global_finger_to_obj_pos = fingertip_pos - obj_pos
    obj_rot_inv_rot_expand = obj_inv_rot.repeat((1, J_f, 1))
    local_finger_to_obj_pos = torch_utils.my_quat_rotate(obj_rot_inv_rot_expand.view(-1, 4), diff_global_finger_to_obj_pos.view(-1, 3))

    
    obs.append(diff_local_bbox_flat.view(B, -1))  # 1 * timestep * 8 * 3
    obs.append(diff_local_vel.view(B, -1))  # timestep  * 1 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # timestep  * 1 * 3
    obs.append(local_obj_bbox_pos.view(B, -1))  # timestep  * 8 * 3
    obs.append(local_finger_to_obj_pos.view(B, -1))  # 10 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

# @torch.jit.script
def compute_pregrasp_reward_time(root_pos, root_rot, hand_pos, hand_rot, hand_vel, hand_ang_vel, ref_hand_pos, ref_hand_rot, ref_hand_vel, ref_hand_ang_vel, ref_obj_pos, hand_pos_prev, close_distance, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, float, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]
    w_pos, w_rot = 0.9, 0.1
    
    # body position reward
    diff_hand_to_object = torch.norm(ref_hand_pos - ref_obj_pos, dim = -1, p = 2)
    close_hand_flag = diff_hand_to_object < close_distance # This flag decides whether the reference hand should be used in computing the pregrasp reward; is it close? 

    prev_dist = torch.norm(hand_pos_prev - ref_hand_pos, dim=-1, p = 2)
    curr_dist = torch.norm(hand_pos - ref_hand_pos, dim=-1, p = 2)
    prev_dist[~close_hand_flag] = 0
    curr_dist[~close_hand_flag] = 0
    distance_filter = curr_dist.sum(dim = -1)/close_hand_flag.sum(dim=-1) > close_distance # distance filter computes whether the hand is close to reference hand pose. If not close enough, it will be replaced with the "getting closer" reward. 
    
    closer_to_hand_r = torch.clamp(prev_dist - curr_dist, min=0, max=1/10).sum(dim=-1)/close_hand_flag.sum(dim=-1)   # cap max at 1/10, encourage the hand that is close enough in referect to get closer to the reference
    
    # Hand position reward
    diff_global_body_pos = ref_hand_pos - hand_pos
    distance = (diff_global_body_pos**2).mean(dim=-1)
    distance[~close_hand_flag] = 0
    diff_body_pos_dist = distance.sum(dim=-1)/close_hand_flag.sum(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # hand rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_hand_rot, torch_utils.quat_conjugate(hand_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle[~close_hand_flag] = 0
    diff_global_body_angle_dist = (diff_global_body_angle**2).sum(dim=-1)/close_hand_flag.sum(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    r_body_pos[distance_filter] = closer_to_hand_r[distance_filter]
    r_body_rot[distance_filter] = closer_to_hand_r[distance_filter]
    
    reward = w_pos * r_body_pos + w_rot * r_body_rot 
    reward_raw = torch.stack([r_body_pos, r_body_rot], dim=-1)
    
    return reward, reward_raw


@torch.jit.script
def compute_pregrasp_reward_contact(root_pos, root_rot, hand_pos, hand_rot, hand_vel, hand_ang_vel, ref_hand_pos, ref_hand_rot, ref_hand_vel, ref_hand_ang_vel, ref_obj_pos, obj_pos, hand_pos_prev, close_distance, table_removed_flag, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]
    w_pos, w_rot = 0.9, 0.1
    
    # body position reward
    diff_hand_to_object = torch.norm(ref_hand_pos - obj_pos, dim = -1, p = 2)
    close_hand_flag = diff_hand_to_object < close_distance # This flag decides whether the reference hand should be used in computing the pregrasp reward; is it close? 

    prev_dist = torch.norm(hand_pos_prev - obj_pos, dim=-1).min(dim = -1).values
    curr_dist = torch.norm(hand_pos - obj_pos, dim=-1).min(dim = -1).values
    
    distance_filter = curr_dist > close_distance # distance filter computes whether the hand is close to object. If not close enough, it will be replaced with the "getting closer" reward. 
    distance_filter = torch.logical_or(distance_filter, table_removed_flag) # if table is removed, replace with "getting closer" reward.
    
    closer_to_hand_r = torch.clamp(prev_dist - curr_dist, min=0, max=1/5)  # cap max at 1/5
    
    # Hand position reward
    diff_global_body_pos = ref_hand_pos - hand_pos
    distance = (diff_global_body_pos**2).mean(dim=-1)
    distance[~close_hand_flag] = 0
    diff_body_pos_dist = distance.sum(dim=-1)/(close_hand_flag.sum(dim=-1) + 1e-6)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)
    
    distance_filter = torch.logical_or(distance_filter, close_hand_flag.sum(dim=-1) == 0) # replace with "getting closer" reward if the reference hand pose is no longer close to the object

    # hand rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_hand_rot, torch_utils.quat_conjugate(hand_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle[~close_hand_flag] = 0
    diff_global_body_angle_dist = (diff_global_body_angle**2).sum(dim=-1)/close_hand_flag.sum(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    r_body_pos[distance_filter] = closer_to_hand_r[distance_filter]
    r_body_rot[distance_filter] = closer_to_hand_r[distance_filter]
    
    reward = w_pos * r_body_pos + w_rot * r_body_rot 
    reward_raw = torch.stack([r_body_pos, r_body_rot], dim=-1)
    
    return reward, reward_raw


@torch.jit.script
def check_contact(hand_contact_force, obj_contact_forces, hand_pos, obj_pos, obj_vel, table_removed, close_distance_contact):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    
    obj_contact_force_sum = obj_contact_forces.sum(dim = -2).abs().sum(dim = -1) > 0
    hand_pos_diff = (hand_pos - obj_pos).norm(dim=-1, p = 2)
    
    table_no_contact = obj_contact_forces[:, -1].abs().sum(dim = -1) == 0 
    obj_has_contact = obj_contact_forces[:, 0].abs().sum(dim = -1) > 0 
    object_lifted = torch.logical_and(obj_has_contact, table_no_contact)
    
    pos_filter = (hand_pos_diff < close_distance_contact).sum(dim = -1) > 0
    vel_filter = (torch.norm(obj_vel, dim= -1, p = 2) > 0.01)[:, 0]
    
    vel_filter = torch.logical_or(vel_filter, object_lifted) # velocity means the object is moved. The object lifted is for when the object is mid-air and stationary. 
    vel_filter[table_removed]  = True # if table is removed, we do not need the proxy check anymore. 
    proxy_check = torch.logical_and(pos_filter, vel_filter)
    
    contact_filter = torch.logical_and(obj_contact_force_sum, proxy_check) # 

    return contact_filter

@torch.jit.script
def compute_grab_reward(root_pos, root_rot, obj_pos, obj_rot, obj_vel, obj_ang_vel,  ref_obj_pos, ref_obj_rot, ref_body_vel, ref_body_ang_vel, contact_filter, rwd_specs):
     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel, w_conctact = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"], rwd_specs["w_conctact"]

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

    r_contact_lifted = contact_filter.float() 

    # # r_close = torch.exp(-k_pos * (hand_pos_diff.min(dim = -1).values **2))

    # ##### pos_filter makes sure that no reward is given if the hand is too far from the object.
    # # reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  + r_close * w_close
    reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  
    # # reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel, r_close], dim=-1)
    reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel], dim=-1)
    
    # np.set_printoptions(precision=4, suppress=1)
    # print(reward_raw.detach().numpy())
    
    return reward, reward_raw

# V 1.5, no contact reward, use with stage. 
@torch.jit.script
def compute_grab_reward_v15(root_pos, root_rot, obj_pos, obj_rot, obj_vel, obj_ang_vel,  ref_obj_pos, ref_obj_rot, ref_body_vel, ref_body_ang_vel, contact_filter, rwd_specs):
     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel, w_conctact = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"], rwd_specs["w_conctact"]

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

    r_contact_lifted = contact_filter.float() 

    # # r_close = torch.exp(-k_pos * (hand_pos_diff.min(dim = -1).values **2))

    # ##### pos_filter makes sure that no reward is given if the hand is too far from the object.
    # # reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  + r_close * w_close
    reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter
    # # reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel, r_close], dim=-1)
    reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel], dim=-1)
    
    # np.set_printoptions(precision=4, suppress=1)
    # print(reward_raw.detach().numpy())
    
    return reward, reward_raw


##### V2 reward, using 8 points instead of rotation reward. 
@torch.jit.script
def compute_grab_reward_v2(root_pos, root_rot, obj_pos, obj_rot, obj_vel, obj_ang_vel,  ref_obj_pos, ref_obj_rot, ref_body_vel, ref_body_ang_vel,  bounding_boxes, contact_filter, rwd_specs):
     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel, w_conctact = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"], rwd_specs["w_conctact"]
    
    B = obj_pos.shape[0]
    
    boungding_box_obj = torch_utils.my_quat_rotate(obj_rot.repeat(1, 8, 1).view(-1, 4), bounding_boxes.view(-1, 3)).view(B, 8, 3) + obj_pos
    boungding_box_ref = torch_utils.my_quat_rotate(ref_obj_rot.repeat(1, 8, 1).view(-1, 4), bounding_boxes.view(-1, 3)).view(-1, 8, 3) + ref_obj_pos

    alpha, beta = 30, 2
    diff_global_body_pos = boungding_box_ref - boungding_box_obj
    diff_body_pos_dist = diff_global_body_pos.norm(dim = -1, p = 2)
    r_obj_pos = (1/(torch.exp(alpha * diff_body_pos_dist) + beta + torch.exp(-alpha * diff_body_pos_dist))).sum(dim = -1)/2
    

    # object linear velocity tracking reward
    diff_global_vel = ref_body_vel - obj_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_lin_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # object angular velocity tracking reward
    diff_global_ang_vel = ref_body_ang_vel - obj_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    r_contact_lifted = contact_filter.float() 
    # # r_close = torch.exp(-k_pos * (hand_pos_diff.min(dim = -1).values **2))

    # ##### pos_filter makes sure that no reward is given if the hand is too far from the object.
    # # reward = (w_pos * r_obj_pos + w_rot * r_obj_rot + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  + r_close * w_close
    reward = (w_pos * r_obj_pos + w_vel * r_lin_vel + w_ang_vel * r_ang_vel) * contact_filter + r_contact_lifted * w_conctact  
    # # reward_raw = torch.stack([r_obj_pos, r_obj_rot, r_lin_vel, r_ang_vel, r_close], dim=-1)
    reward_raw = torch.stack([r_obj_pos, r_lin_vel, r_ang_vel], dim=-1)
    
    return reward, reward_raw

@torch.jit.script
def compute_grab_observations_v4(root_pos, root_rot, fingertip_pos, fingertip_rot, ref_obj_pos, ref_obj_rot, ref_o_vel, ref_o_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, int, bool) -> Tensor
    # Image based. Only future object tracks. 
    obs = []
    B, _, _ = fingertip_pos.shape
    J = 1
    if not upright:
        root_rot = remove_base_rot(root_rot)
        
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    ref_global_body_pos = ref_obj_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, J, 3)
    ref_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_global_body_pos.view(-1, 3))

    ref_global_body_rot = torch_utils.quat_mul(ref_obj_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(root_rot[:, None, None]).repeat_interleave(time_steps, 1))
    ref_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    ref_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_o_vel.view(-1, 3))
    ref_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_o_ang_vel.view(-1, 3))
    

    
    obs.append(ref_local_body_pos_flat.view(B, -1))  # 1 * timestep * 1 * 3
    obs.append(torch_utils.quat_to_tan_norm(ref_local_body_rot_flat).view(B, -1))  #  1 * timestep * 1 * 6
    obs.append(ref_local_vel.view(B, -1))  # timestep  * 1 * 3
    obs.append(ref_local_ang_vel.view(B, -1))  # timestep  * 1 * 3
    
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

@torch.jit.script
def compute_humanoid_grab_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, obj_pos, obj_rot, ref_obj_pos, ref_obj_rot, hand_pos, pass_time, enable_early_termination, termination_distance, disableCollision, check_rot_reset):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool, bool) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)
    if (enable_early_termination):
        has_fallen = torch.any(torch.norm(obj_pos - ref_obj_pos, dim=-1) > termination_distance, dim=-1) 
            
        
        if check_rot_reset:
            diff_global_body_rot = torch_utils.quat_mul(ref_obj_rot, torch_utils.quat_conjugate(obj_rot))
            diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
            diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
            has_fallen = torch.logical_or(has_fallen, diff_global_body_angle_dist > 1) # object orientation check
        
        
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