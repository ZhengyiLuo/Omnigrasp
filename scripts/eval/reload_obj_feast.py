"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict
import argparse

flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

def create_sim_and_env():
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    
    asset_file = asset_descriptors[args.asset_id].file_name

    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.flip_visual_attachments = asset_descriptors[
    #     args.asset_id].flip_visual_attachments
    asset_options.use_mesh_materials = True

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    target_assets = {}
    motion_file = "data/amass_x/oakink/singles_virtual/oakink_virtual_grasp_train_1024_bps.pkl"
    motion_data = joblib.load(motion_file)
    
    ####################################
    
    for k, v in motion_data.items():
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
        
        obj_asset_root = "/".join(v["obj_data"]["obj_info"][0].split("/")[:-1])
        asset_file = v["obj_data"]["obj_info"][0].split("/")[-1]
        target_assets[k] = gym.load_asset(sim, obj_asset_root, asset_file, asset_options)
    ####################################

    # set up the env grid
    num_envs = len(target_assets)
    num_per_row = 5
    spacing = 5
    env_lower = gymapi.Vec3(-spacing, spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)

    # cache useful handles
    envs = []
    actor_handles = []

    num_dofs = gym.get_asset_dof_count(asset)
    print("Creating %d environments" % num_envs)
    
    all_assets = list(target_assets.values())
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        actor_handle = gym.create_actor(env, all_assets[i], pose, "actor", 0, 1, 0) # col_group, col_filter, 0
        actor_handles.append(actor_handle)

    gym.prepare_sim(sim)
    
    
    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)
    
    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)
    
    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)
    return sim

if __name__ == "__main__":
    
    robot_cfg = {
        "mesh": True,
        "rel_joint_lm": False,
        "upright_start": False,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smplx",
        "big_ankle": True, 
        "freeze_hand": False,
        "box_body": True, 
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    # gender_beta = np.array([1.0000, -0.2141, -0.1140, 0.3848, 0.9583, 1.7619, 1.5040, 0.5765, 0.9636, 0.2636, -0.4202, 0.5075, -0.7371, -2.6490, 0.0867, 1.4699, -1.1865])
    asset_root = "/"
    gender_beta = np.zeros((21))
    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
    test_good = f"/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_good)
    # test_good = f"test.xml"
    # asset_root = "./"
    sk_tree = SkeletonTree.from_mjcf(test_good)

    asset_descriptors = [
        AssetDesc(test_good, False),
    ]

    

    # parse arguments
    args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                                custom_parameters=[{
                                    "name": "--asset_id",
                                    "type": int,
                                    "default": 0,
                                    "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
                                }, {
                                    "name": "--speed_scale",
                                    "type": float,
                                    "default": 1.0,
                                    "help": "Animation speed scale"
                                }, {
                                    "name": "--show_axis",
                                    "action": "store_true",
                                    "help": "Visualize DOF axis"
                                }, 
                                {
                                    "name": "--motion_file",
                                    "type": str,
                                    "default": "data/amass_x/pkls/singles/smplx_failed_1.pkl"
                                }
                                ])
    motion_file = args.motion_file
    
    if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
        print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
        quit()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if not args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    
    sim = create_sim_and_env()
    gym.destroy_sim(sim)
    del sim
    sim = create_sim_and_env()
    gym.destroy_sim(sim)
    del sim
    sim = create_sim_and_env()
    gym.destroy_sim(sim)
    del sim
    sim = create_sim_and_env()
    gym.destroy_sim(sim)
    del sim
    sim = create_sim_and_env()
    
    

    