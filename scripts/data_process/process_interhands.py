import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import json
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from smplx import MANO

layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
mano_layer_right = MANO(model_path = "data/smpl", is_rhand=True, use_pca=False, flat_hand_mean=False, **layer_arg)
mano_layer_left = MANO(model_path = "data/smpl", is_rhand=False, use_pca=False, flat_hand_mean=False, **layer_arg)
left_mean = mano_layer_left.hand_mean.clone()
right_mean = mano_layer_right.hand_mean.clone()
mano_layer_right = MANO(model_path = "data/smpl", is_rhand=True, use_pca=False, flat_hand_mean=True, **layer_arg)
mano_layer_left = MANO(model_path = "data/smpl", is_rhand=False, use_pca=False, flat_hand_mean=True, **layer_arg)


orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
orig_root_joint_idx = orig_joints_name.index('Wrist')

interhand_keypoints_data = joblib.load("/hdd/zen/dev/meta/PHC_X/data/reinterhand/interhand_keypoints.pkl")
hand_data_dump = {}
for take_key in os.listdir(f"/hdd/zen/dev/meta/InterWild/data/ReInterHand/"):
    fitted_path = f"/hdd/zen/dev/meta/InterWild/data/ReInterHand/{take_key}/mano_fits/params/"
    subject_files = sorted(glob.glob(osp.join(fitted_path, "*.json")))
    
    if len(subject_files) < 1:
        print(take_key)
        continue
    
    with open(subject_files[0]) as f:
        mano_param = json.load(f)
    shape = torch.FloatTensor(np.array(mano_param['shape'])).view(-1, 10)
    trans = torch.FloatTensor(np.array(mano_param['trans'])).view(-1,3)
    pose = torch.FloatTensor(np.array(mano_param['pose'])).view(-1,3)


    take_key = subject_files[0].split("/")[-4]
    keypoint_pos = interhand_keypoints_data[take_key]['data_array'][:, :, 1:4]/1000
    hand_data_entry = defaultdict(list)


    for file in tqdm(subject_files):
        
        with open(file) as f:
            mano_param = json.load(f)
        file_name = file.split("/")[-1].split('.')[0]
        frame, hand = int(file_name.split('_')[0]), file_name.split('_')[1]
        
        pose = np.array(mano_param['pose']).reshape(-1, 3)
        root_pose = pose[orig_root_joint_idx].reshape(1,3)
        if hand == "left":
            mean = left_mean
        elif hand == "right":
            mean = right_mean
        hand_pose = np.concatenate((pose[:orig_root_joint_idx,:], pose[orig_root_joint_idx+1:,:])).reshape(1,-1) + mean.numpy()

        hand_data_entry[f'{hand}_shape'].append(np.array(mano_param['shape']).reshape(-1, 10))
        hand_data_entry[f'{hand}_trans'].append(np.array(mano_param['trans']).reshape(-1,3))
        hand_data_entry[f'{hand}_pose'].append(hand_pose)
        hand_data_entry[f'{hand}_root_pose'].append(root_pose)
        hand_data_entry[f'{hand}_frame_idx'].append(frame)
        


    hand_data_entry = {k: np.array(v).squeeze() for k, v in hand_data_entry.items()}
    hand_data_entry['keypoint_pos'] = keypoint_pos
    hand_data_dump[take_key] = hand_data_entry
    
joblib.dump(hand_data_dump, "data/reinterhand/interhand.pkl", compress = True)