import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_games.algos_torch import torch_ext
import cv2
from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import pickle as pk
from typing import OrderedDict
import copy

mujoco_hand_joints = ['L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']
hand_jt_idx = [SMPLH_MUJOCO_NAMES.index(q) for q in mujoco_hand_joints]
left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

flip_left_right = True
upright_start = False
trim = False
print("upright_start", upright_start, "trim", trim)
print("upright_start", upright_start, "trim", trim)
print("upright_start", upright_start, "trim", trim)

robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, 
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False, 
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
    }

smpl_local_robot = LocalRobot(robot_cfg,)
data_load = joblib.load("data/grab/raw/grab.pkl")

print("using neutral model")

objects = set()
for k, v in data_load.items():
    objects.add(k.split("_")[1])

objects.remove("doorknob")
object_code_dict = {}
for object_name in objects:
    object_code_dict[object_name] = torch.load(f"data/amass_x/grab/grab_codes_v1/{object_name}.pkl")

data_dump_full = OrderedDict()
for object_name in objects:
    
    data_dump = OrderedDict()
    pbar = tqdm(data_load.items())
    for data_key, entry_data in pbar:
        if data_key in  ["s5_flashlight_on_2", "s5_flashlight_on_1", "s1_watch_set_1"] + ["s10_pyramidsmall_inspect_1"]: # unfixable initial state. Remove sequences for now + corrupted data 
            continue
        
        # if not data_key == "s9_waterbottle_drink_1":
        #     continue
        
        pbar.set_description(f"Processing {object_name}")
        if not f"{object_name}" == data_key.split("_")[1]:
            continue
        start  = entry_data['contact_info'].sum(axis = -1).nonzero()[0][0]
        if not trim:
            start = 0
        pose_aa = entry_data['pose_aa'][start:]
        root_trans = entry_data['trans_orig'] if "trans_orig" in entry_data else entry_data['trans'][start:]
        betas = entry_data['beta']
        gender = entry_data['gender']
        N = pose_aa.shape[0] 
        
        
        gender_number = np.zeros([1])
        if gender == "male": gender_number[:] = 1
        if gender == "female": gender_number[:] = 2
        
        smpl_local_robot.load_from_skeleton(v_template=torch.from_numpy(entry_data['v_template']).float(), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"/tmp/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"/tmp/{robot_cfg['model']}_humanoid.xml")

        N = pose_aa.shape[0]

        smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
        mujoco_2_smpl = [SMPLH_MUJOCO_NAMES.index(q) for q in SMPLH_BONE_ORDER_NAMES if q in SMPLH_MUJOCO_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 52, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
        
        root_trans_offset = torch.from_numpy(root_trans).float() + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True)
        
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()

        pose_quat_global = new_sk_state.global_rotation.numpy()
        
        pose_quat = new_sk_state.local_rotation.numpy()
        fps = 30
        
        
        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset
        new_motion_out['beta'] = betas
        new_motion_out['gender'] = gender
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['fps'] = fps

        
        if data_key == "s1_cubesmall_offhand_1" and start == 0:
            entry_data['obj_pose'][0, 3:7] = entry_data['obj_pose'][1, 3:7]
            
        if data_key == "s6_flute_lift" and start == 0:
            entry_data['obj_pose'][:3, :7] = entry_data['obj_pose'][3, :7]
            
        
        if "obj_info" in entry_data:
            new_motion_out['obj_data'] = {
                "hand_rot": new_sk_state.global_rotation[:, hand_jt_idx].numpy(),
                "hand_trans": new_sk_state.global_translation[:, hand_jt_idx].numpy(),
                "contact_info": entry_data['contact_info'][start:].sum(axis = -1),
                "obj_pose": entry_data['obj_pose'][start:],
                "obj_info": entry_data['obj_info'],
                "object_code": object_code_dict[object_name]
            }
            

        if "v_template" in entry_data:
            new_motion_out['v_template'] = entry_data['v_template']
        data_dump[data_key] = copy.deepcopy(new_motion_out)
        
        if flip_left_right:
            pose_quat_global_flip = pose_quat_global[:, left_to_right_index].copy()
            root_trans_flip = root_trans.copy() # here probably should add the offset back. 
            root_trans_flip[..., 0] *= -1
            
            root_trans_offset_flip = root_trans_offset.clone()
            root_trans_offset_flip[..., 0] *= -1
            
            pose_quat_global_flip[..., 1] *= -1
            pose_quat_global_flip[..., 2] *= -1
            
            flip_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global_flip), root_trans_offset_flip, is_local=False)
            
            pose_quat_flip = flip_sk_state.local_rotation.numpy()
            pose_aa_mj = sRot.from_quat(pose_quat.reshape(-1, 4)).as_rotvec().reshape(N, 52, 3)
            pose_aa  = pose_aa_mj[:, mujoco_2_smpl]
            
            
            
            flip_motion_out = {}
            flip_motion_out['pose_quat_global'] = pose_quat_global_flip
            flip_motion_out['pose_quat'] = pose_quat_flip
            flip_motion_out['trans_orig'] = root_trans_flip
            flip_motion_out['root_trans_offset'] = root_trans_offset_flip
            flip_motion_out['beta'] = betas
            flip_motion_out['gender'] = gender
            flip_motion_out['pose_aa'] = pose_aa
            flip_motion_out['fps'] = fps
            
            obj_pose_flip = entry_data['obj_pose'].copy()
            obj_pose_flip[:, [0, 7]] *= -1
            obj_pose_flip[:, [4, 11]] *= -1
            obj_pose_flip[:, [5, 12]] *= -1
            
            if data_key == "s1_cubesmall_offhand_1" and start == 0:
                obj_pose_flip[0, 3:7] = obj_pose_flip[1, 3:7]
                
            if data_key == "s6_flute_lift" and start == 0:
                obj_pose_flip[:3, :7] = obj_pose_flip[3, :7]
                
            if "obj_info" in entry_data:
                flip_motion_out['obj_data'] = {
                    "hand_rot": flip_sk_state.global_rotation[:, hand_jt_idx].numpy(),
                    "hand_trans": flip_sk_state.global_translation[:, hand_jt_idx].numpy(),
                    "contact_info": entry_data['contact_info'][start:].sum(axis = -1),
                    "obj_pose": obj_pose_flip[start:],
                    "obj_info": entry_data['obj_info'],
                    "object_code": object_code_dict[object_name]
                }
                

            if "v_template" in entry_data:
                flip_motion_out['v_template'] = entry_data['v_template']
            flip_key = data_key + "_flip"
            data_dump[flip_key] = copy.deepcopy(flip_motion_out)
            joblib.dump({flip_key:  flip_motion_out}, f"data/amass_x/grab_flip/singles/{flip_key}.pkl", compress=True)
        
    
        # if trim:
        #     joblib.dump({data_key: new_motion_out}, f"data/amass_x/grab/singles_trim/{data_key}.pkl", compress=True)
        # else:
        #     joblib.dump({data_key: new_motion_out}, f"data/amass_x/grab/singles/{data_key}.pkl", compress=True)
        
    data_dump_full.update(data_dump)      
    
    print(f"Total number of sequences {len(data_dump)}")
    # if trim:
    #     joblib.dump(data_dump, f"data/amass_x/grab/grab_{object_name}_trim.pkl", compress=True)
    # else:
    #     joblib.dump(data_dump, f"data/amass_x/grab/grab_{object_name}.pkl", compress=True)

import ipdb; ipdb.set_trace()

# joblib.dump(data_dump_full, f"data/amass_x/grab/grab_full.pkl", compress=True)
joblib.dump(data_dump_full, f"data/amass_x/grab_flip/grab_flip.pkl", compress=True)