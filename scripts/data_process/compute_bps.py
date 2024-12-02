import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import time
import numpy as np
from bps_torch.bps import bps_torch
import open3d as o3d
import joblib
from tqdm import tqdm
from phc.utils.point_utils import normalize_points_byshapenet

np.random.seed(0)

custom_basis = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################
grab_data_item = joblib.load("data/amass_x/grab/grab_codes_bps/stamp.pkl")
custom_basis = grab_data_item['basis']
##################

# initiate the bps module
bps = bps_torch(bps_type='random_uniform',
                n_bps_points=512,
                radius=1.,
                n_dims=3,
                custom_basis=custom_basis)


# mesh_root = "phc/data/assets/mesh/"
# mesh_root = "phc/data/assets/mesh/oakink/"
# mesh_root = "phc/data/assets/mesh/oakink_virtual/"
mesh_root = "phc/data/assets/mesh/omomo/"
for ply_file in tqdm(sorted(glob.glob(osp.join(mesh_root, "*.stl")))):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    
    points = torch.Tensor(np.asarray(mesh.vertices)).to(device)
    points = normalize_points_byshapenet(points)
    

    bps_enc = bps.encode(points,
                         feature_type=['dists'],
                         x_features=None,
                         custom_basis=custom_basis)
    dists = bps_enc['dists']
    
    obj_name = ply_file.split('/')[-1].split('.')[0]
    print(obj_name,  points.max(),  points.min())
    obj_dict = {'bps_code': dists, 'basis': bps.bps.clone()}
    
    # joblib.dump(obj_dict, f"data/amass_x/grab/grab_codes_bps/{obj_name}.pkl", compress=True)
    # joblib.dump(obj_dict, f"data/oakink/oakink_latent/{obj_name}/{obj_name}_bps.pkl", compress=True)
    # joblib.dump(obj_dict, f"data/oakink/oakink_latent_virtual/{obj_name}/{obj_name}_bps.pkl", compress=True)
    joblib.dump(obj_dict, f"data/omomo/1x_smaller/{obj_name}/{obj_name}_bps.pkl", compress=True)
    
