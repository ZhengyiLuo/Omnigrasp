import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
from smplx import MANO
from scipy.ndimage import gaussian_filter1d
paused, reset, recording, image_list, writer, control, curr_zoom = False, False, False, [], None, None, 0.01


def pause_func(action):
    global paused
    paused = not paused
    print(f"Paused: {paused}")
    return True


def reset_func(action):
    global reset
    reset = not reset
    print(f"Reset: {reset}")
    return True


def record_func(action):
    global recording, writer
    if not recording:
        fps = 30
        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        curr_video_file_name = f"output/renderings/o3d/{curr_date_time}-test.mp4"
        writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
    elif not writer is None:
        writer.close()
        writer = None

    recording = not recording

    print(f"Recording: {recording}")
    return True


def zoom_func(action):
    global control, curr_zoom
    curr_zoom = curr_zoom * 0.9
    control.set_zoom(curr_zoom)
    print(f"Reset: {reset}")
    return True

colorpicker = mpl.colormaps['Blues']
mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']


data_dir = "data/smpl"
layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
mano_layer_right = MANO(model_path = "data/smpl", is_rhand=True, use_pca=False, flat_hand_mean=True, **layer_arg)
mano_layer_left = MANO(model_path = "data/smpl", is_rhand=False, use_pca=False, flat_hand_mean=True, **layer_arg)

# motion_file = "data/amass_x/pkls/singles/GRAB_s1_banana_eat_1_stageii.pkl"
motion_file = "data/reinterhand/test.pkl"
# motion_file = "data/amass_x/pkls/test.pkl"

Name = motion_file.split("/")[-1].split(".")[0]
pkl_data = joblib.load(motion_file)
head_up_rot = sRot.from_euler("xyz", [-np.pi/2, 0, 0])

def main():
    global reset, paused, recording, image_list, control
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    smpl_meshes = dict()
    items = list(pkl_data.items())
    
    for entry_key, data_seq in tqdm(items):
        data_seq = {k: torch.from_numpy(v).float() for k, v in data_seq.items()}
        start, end = 0, data_seq['left_pose'].shape[0]
        left_pose = data_seq['left_pose'][start:end]
        right_pose = data_seq['right_pose'][start:end]
        left_shape = data_seq['left_shape'][start:end]
        right_shape = data_seq['right_shape'][start:end]
        left_trans = data_seq['left_trans'][start:end]
        right_trans = data_seq['right_trans'][start:end]
        left_root_pose = data_seq['left_root_pose'][start:end]
        right_root_pose = data_seq['right_root_pose'][start:end]
        keypoint_pos = data_seq['keypoint_pos'][:, [235, 236, 237, 238]][start:end]
        keypoint_pos = torch.from_numpy(gaussian_filter1d(keypoint_pos, sigma = 3, axis = 0)).float()
        
        left_root_pose = torch.from_numpy((head_up_rot * sRot.from_rotvec(left_root_pose)).as_rotvec()).float()
        right_root_pose = torch.from_numpy((head_up_rot * sRot.from_rotvec(right_root_pose)).as_rotvec()).float()
        left_height, right_height = left_trans[0, :].clone(), right_trans[0, :].clone()
        left_trans = (left_trans @ head_up_rot.as_matrix().T).float()
        right_trans = (right_trans @ head_up_rot.as_matrix().T).float()
        keypoint_pos = keypoint_pos @ head_up_rot.as_matrix().T
        
        diff = (left_trans[0, :] - left_height)
        left_trans = left_trans - diff
        right_trans = right_trans - diff
        keypoint_pos = keypoint_pos - diff
        
        # import ipdb; ipdb.set_trace()
        
        with torch.no_grad():
            left_output = mano_layer_left(betas=left_shape, hand_pose=left_pose, global_orient=left_root_pose, transl=left_trans)
            right_output = mano_layer_right(betas=right_shape, hand_pose=right_pose, global_orient=right_root_pose, transl=right_trans)
            
            
        left_vertices = left_output.vertices.numpy()
        left_faces = mano_layer_left.faces
        left_mesh = o3d.geometry.TriangleMesh()
        left_mesh.vertices = o3d.utility.Vector3dVector(left_vertices[0])
        left_mesh.triangles = o3d.utility.Vector3iVector(left_faces)
        
        
        right_vertices = right_output.vertices.numpy()
        right_faces = mano_layer_right.faces
        right_mesh = o3d.geometry.TriangleMesh()
        right_mesh.vertices = o3d.utility.Vector3dVector(right_vertices[0])
        right_mesh.triangles = o3d.utility.Vector3iVector(right_faces)

        ######################## Smampling texture ########################
        vertex_colors = colorpicker(0.6)[:3]
        left_mesh.paint_uniform_color(vertex_colors)
        left_mesh.compute_vertex_normals()
        
        right_mesh.paint_uniform_color(vertex_colors)
        right_mesh.compute_vertex_normals()
        ######################## Smampling texture ########################
        vis.add_geometry(left_mesh); vis.add_geometry(right_mesh)
        smpl_meshes[entry_key] = {
            'left_mesh': left_mesh,
            'right_mesh': right_mesh,
            "left_vertices": left_vertices,
            "right_vertices": right_vertices,
        }
        break
    
    box = o3d.geometry.TriangleMesh()
    ground_size, height = 15,  0.01
    box = box.create_box(width=ground_size, height=height, depth=ground_size)
    box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
    box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
    box.compute_vertex_normals()
    box.vertex_colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]).repeat(8, axis=0))
    num_jts = 51
   
    ##################### Shperes ##########################
    
    spheres = []
    for idx in range(keypoint_pos.shape[1]):
        sphere = o3d.geometry.TriangleMesh()
        sphere = sphere.create_sphere(radius=0.01)
        sphere.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.9, 0.1]]).repeat(len(sphere.vertices), axis=0))
        
        # if idx in [235, 236, 237, 238]:
        #     sphere = sphere.create_sphere(radius=0.02)
        #     sphere.vertex_colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]]).repeat(len(sphere.vertices), axis=0))
        # else:
        #     sphere = sphere.create_sphere(radius=0.01)
        #     sphere.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.9, 0.1]]).repeat(len(sphere.vertices), axis=0))
        sphere.compute_vertex_normals()
        spheres.append(sphere)

    sphere_pos = np.zeros([keypoint_pos.shape[1], 3])
    [vis.add_geometry(sphere) for sphere in spheres]
    ##################### Shperes ##########################
    
    
    vis.add_geometry(box)

    control = vis.get_view_control()

    control.unset_constant_z_far()
    control.unset_constant_z_near()
    i = 0
    N = left_vertices.shape[0]

    vis.register_key_callback(32, pause_func)
    vis.register_key_callback(82, reset_func)
    vis.register_key_callback(76, record_func)
    vis.register_key_callback(90, zoom_func)

    control.set_up(np.array([0, 0, 1]))
    control.set_front(np.array([1, 0, 0]))
    control.set_lookat(np.array([0, 0, 0]))

    control.set_zoom(0.5)
    dt = 1 / 30

    
    # tracker_pos = data_seq['joints_gt']

    while True:
        vis.poll_events()
        for smpl_mesh_key, smpl_mesh_data in smpl_meshes.items():
            left_verts = smpl_mesh_data["left_vertices"]
            smpl_mesh_data["left_mesh"].vertices = o3d.utility.Vector3dVector(left_verts[i % left_verts.shape[0]])
            vis.update_geometry(smpl_mesh_data["left_mesh"])
            smpl_mesh_data["left_mesh"].compute_vertex_normals()
            
            
            right_verts = smpl_mesh_data["right_vertices"]
            smpl_mesh_data["right_mesh"].vertices = o3d.utility.Vector3dVector(right_verts[i % right_verts.shape[0]])
            vis.update_geometry(smpl_mesh_data["right_mesh"])
            smpl_mesh_data["right_mesh"].compute_vertex_normals()

            curr_pos = keypoint_pos[i % keypoint_pos.shape[0]]
            for idx, s in enumerate(spheres):
                s.translate((curr_pos - sphere_pos)[idx])
                vis.update_geometry(s)
            sphere_pos = curr_pos

        if not paused:
            i = (i + 1)

        if reset:
            i = 0
            reset = False
        if recording:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)
            writer.append_data(rgb)

        vis.update_renderer()


if __name__ == "__main__":
    main()