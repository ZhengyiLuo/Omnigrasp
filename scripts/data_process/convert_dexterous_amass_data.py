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
import uuid

np.random.seed(0)

smpl_hand_joints = ["L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3","L_Middle1","L_Middle2","L_Middle3","L_Pinky1","L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",]

hands_joints_idx = [SMPLH_BONE_ORDER_NAMES.index(x) for x in smpl_hand_joints]

upright_start = False
print("upright_start", upright_start)
print("upright_start", upright_start)
print("upright_start", upright_start)

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
        "sim": "isaacgym", 
    }

smpl_local_robot = LocalRobot(robot_cfg,)
hands_data = joblib.load("data/amass_x/hands_data.pkl")
hands_data_keys = list(hands_data.keys())
all_pkls = glob.glob("/hdd/zen/data/ActBound/AMASS/AMASS_X_G_Complete/**/*.npz", recursive=True)
amass_occlusion = joblib.load("/hdd/zen/data/ActBound/AMASS/amassx_occlusion_v1.pkl")
# raw_data_entry = "/hdd/zen/data/ActBound/AMASS/AMASS_X_G_download/ACCAD/Male2MartialArtsKicks_c3d/G12-__cresent_left_stageii.npz"
amass_full_motion_dict = {}
pbar = tqdm(all_pkls)

betas = np.zeros((16))
gender_number, betas[:], gender = [0], 0, "neutral"
print("using neutral model")
smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(betas[None,]), gender=gender_number, objs_info=None)
uuid_str = uuid.uuid4()
smpl_local_robot.write_xml(f"/tmp/smpl/{uuid_str}_humanoid.xml")
skeleton_tree = SkeletonTree.from_mjcf(f"/tmp/smpl/{uuid_str}_humanoid.xml")

for data_path in pbar:
    bound = 0
    
    key_name_dump = "_".join(data_path.split("/")[7:]).replace(".npz", "")
    # if not "Eyes_Japan_Dataset".lower() in key_name_dump.lower():
        # continue
    
    if key_name_dump in amass_occlusion:
        issue = amass_occlusion[key_name_dump]["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
            bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
            if bound < 10:
                pbar.set_description_str(f"bound too small {key_name_dump}, {bound}")
                continue
        else:
            pbar.set_description_str(f"issue irrecoverable, {key_name_dump}, {issue}")
            continue
    
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if not 'mocap_frame_rate' in  entry_data:
        continue
    framerate = entry_data['mocap_frame_rate']

    if "totalcapture" in key_name_dump.lower() or "ssm" in key_name_dump.lower():
        framerate = 60 # total capture's framerate is wrong. 
    elif "KIT".lower() in key_name_dump.lower():
        framerate = 100 # KIT's framerate
    elif "CMU".lower() in key_name_dump.lower():
        orig_file = data_path
        orig_file = orig_file.replace("AMASS_X_G_Complete", "AMASS_Complete")
        orig_file = orig_file.replace("stageii", "poses")
        if osp.isfile(orig_file):
            entry_data_orig = dict(np.load(open(orig_file, "rb"), allow_pickle=True))
            if (entry_data['mocap_frame_rate'] !=  entry_data_orig['mocap_framerate']):
                framerate = entry_data_orig['mocap_framerate']
        else:
            import ipdb; ipdb.set_trace()
            print('.....')
            
    elif "Eyes_Japan_Dataset".lower() in key_name_dump.lower():
        orig_file = data_path
        orig_file = orig_file.replace("AMASS_X_G_Complete", "AMASS_Complete")
        orig_file = orig_file.replace("stageii", "poses")
        splits = orig_file.split("-")
        splits[2] = splits[2].replace("_", " ")
        orig_file = "-".join(splits)
        
        if osp.isfile(orig_file):
            entry_data_orig = dict(np.load(open(orig_file, "rb"), allow_pickle=True))
            if (entry_data['mocap_frame_rate'] !=  entry_data_orig['mocap_framerate']):
                framerate = entry_data_orig['mocap_framerate']
        else:
            import ipdb; ipdb.set_trace()
            print('.....')
            
    skip = int(np.floor(framerate/30))
    pose_aa = np.concatenate([entry_data['poses'][::skip, :66], entry_data['poses'][::skip, 75:]], axis = -1)
    root_trans = entry_data['trans'][::skip, :]
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    
    if bound == 0:
        bound = N
            
    root_trans = root_trans[:bound]
    pose_aa = pose_aa[:bound]
    N = pose_aa.shape[0]
    if N < 10:
        continue

    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    pose_aa = pose_aa.reshape(N, 52, 3)
    
    ######################################## Hands augmentation ########################################
    if not "GRAB" in data_path:
        acc_len, hands_data_acc = 0, []
        while acc_len < N:
            random_key = np.random.choice(hands_data_keys)
            hands_data_pick = hands_data[random_key]['hands_aa']
            hands_data_acc.append(hands_data_pick)
            acc_len += hands_data_pick.shape[0]
            
        pose_aa[:, hands_joints_idx] = np.concatenate(hands_data_acc, axis = 0)[:N]
    
    ######################################## Hands augmentation ########################################
    
    
    
    pose_aa_mj = pose_aa[:, smpl_2_mujoco]
    
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 52, 4)
    
    # gender_number, beta[:], gender = [0], 0, "neutral" # For updating bodies
    # print("using neutral model")
    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(betas[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"/tmp/smpl/{uuid_str}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"/tmp/smpl/{uuid_str}_humanoid.xml")

    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

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
    amass_full_motion_dict[key_name_dump] = new_motion_out
    
import ipdb; ipdb.set_trace()    
print(f"Total number of sequences {len(amass_full_motion_dict)}")
joblib.dump(amass_full_motion_dict, "data/amass_x/amass_x_dex/amass_clean_dex_v1.pkl", compress=True)
# joblib.dump(amass_full_motion_dict, "data/amass_x/upright/amass_clean_upright_v1.pkl", compress=True)