import os
import glob
import os.path as osp
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot

sub_gender = {
    "s1": "male",
    "s2": "male",
    "s8": "male",
    "s9": "male",
    "s10": "male",
    
    "s3": "female",
    "s4": "female",
    "s5": "female",
    "s6": "female",
    "s7": "female",
}
grab_base = "/hdd/zen/data/ActBound/GRAB/grab"
subjects = os.listdir(grab_base)

out_base = "/hdd/zen/dev/copycat/smplx/output/"
dump = {}



start = 0
for subject in subjects:
    
    sbj_vtemp = np.array(Mesh(f"/hdd/zen/data/ActBound/GRAB/tools/subject_meshes/{sub_gender[subject]}/{subject}.ply").vertices)
    pbar = tqdm(glob.glob(osp.join(grab_base, f"{subject}/*")))
    for f in pbar:
        
        # print(f)
        seq_name = subject + "_" + f.split("/")[-1].split(".")[0]
        grab_item = dict(np.load(f, allow_pickle = True))
        skip = int(grab_item['framerate']/30)
        assert(skip == 4)
        body_params = grab_item['body'].tolist()['params']
        
        left_hand_pose = np.einsum("bi,ij->bj", body_params['left_hand_pose'], left_hand_components) + left_mean.numpy()[None, ]
        right_hand_pose = np.einsum("bi,ij->bj", body_params['right_hand_pose'], right_hand_components) + right_mean.numpy()[None, ]
        pose_aa = np.hstack([body_params['global_orient'], body_params['body_pose'], left_hand_pose, right_hand_pose])
    
        gender = grab_item['gender']
        
        
        obj_data = {k : v[::skip] for k, v in grab_item['object'].tolist()['params'].items()}
        table_data = {k : v[::skip] for k, v in grab_item['table'].tolist()['params'].items()}
        beta = np.hstack([np.load("/hdd/zen/data/ActBound/GRAB/tools/subject_meshes/male/s1_betas.npy"), np.zeros([1, 6])]).squeeze()
        table_pos = table_data['transl'].copy()
        table_pos[:, 2]/= 2
        seq_len = table_data['transl'].shape[0]
        dump[seq_name] = {
            "seq_name": seq_name, 
            "pose_aa": pose_aa[::skip][start:],
            "trans": body_params['transl'][::skip][start:], 
            "gender": gender, 
            "beta": beta,
            "v_template": sbj_vtemp,
            'obj_pose': np.hstack([
                np.hstack([obj_data['transl'], sRot.from_rotvec(obj_data['global_orient']).inv().as_quat()])[start:],
                # np.hstack([table_data['transl'], sRot.from_rotvec(table_data['global_orient']).as_quat()])[start:]
                np.hstack([table_data['transl'], np.repeat(sRot.from_euler("xyz", [-90, 0, 0], degrees = True).as_quat()[None,], seq_len, axis = 0)])
            ]), 
            "obj_info": [grab_item['object'].tolist()['object_mesh'], 
                         grab_item['table'].tolist()['table_mesh'], ], 
            "contact_info": grab_item['contact'].item()['body'][::skip][start:]
        }
        pbar.set_description_str(f"{seq_name}, {dump[seq_name]['pose_aa'].shape}, {gender}")
        
# joblib.dump(dump, "data/grab/raw/grab.pkl")