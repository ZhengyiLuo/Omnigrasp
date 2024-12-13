import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from phc.utils.motion_lib_smpl_obj import MotionLibSMPLObj
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
import joblib

from lxml import etree
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from io import BytesIO

def xml_add_obj(xml, obj_info, obj_pose):
    # tree = etree.parse("parsed_neutral.xml")
    parser = XMLParser(remove_blank_text=True)
    tree = parse(BytesIO(xml), parser=parser)
    for obj_info_curr in obj_info:
        obj_name = obj_info_curr.split("/")[-1].split(".")[0]
        worldroot = tree.getroot().find("worldbody")
        obj_node = Element("body", {"name": obj_name, "pos": "0 0 0"})
        SubElement(
            obj_node,
            "joint",
            {"limited": "false", "name": obj_name, "type": "free"},
        )
        SubElement(
            obj_node,
            "geom",
            {
                "type": "mesh",
                "mesh": obj_name,
                "contype": "1",
                "conaffinity": "1",
                "friction": "1 0.005 0.0001",
                "rgba": "0.8 0.6 .4 1",
            },
        )
        # if obj_name == "table":
        #     tb_pos = obj_pose[0, (len(obj_info) - 1) * 7 : (len(obj_info) - 1) * 7 + 3]
        #     SubElement(
        #         obj_node,
        #         "geom",
        #         {
        #             "type": "cylinder",
        #             "size": f"0.1 {tb_pos[2]/2:.5f}",
        #             "pos": f"0  {tb_pos[2]/2:.5f} 0",
        #             "euler": "90 0 0",
        #             "contype": "1",
        #             "conaffinity": "1",
        #             "friction": "1 0.005 0.0001",
        #             "mass": "1000",
        #         },
        #     )

        worldroot.append(obj_node)

        asset = tree.getroot().find("asset")
        asset.append(
            Element(
                "mesh",
                {
                    "file": f"/hdd/zen/data/ActBound/GRAB/tools/object_meshes/contact_meshes_stl/{obj_name}.stl",
                    # "file": f"/hdd/zen/dev/meta/PHC_X/phc/data/assets/mesh/oakink/{obj_name}.stl",
                },
            )
        )
    # tree.write("test.xml", pretty_print=True)
    return etree.tostring(tree)

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, rendering, motion_time, obj_pose, obj_info, seq_key, mj_model, mj_data, viewer
    if chr(keycode) == "T":
        curr_start += num_motions
        print("Next Motion", curr_start)
        motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
        seq_key = motion_lib.curr_motion_keys
        obj_pose = motion_file[seq_key]['obj_data']['obj_pose']
        obj_info = motion_file[seq_key]['obj_data']['obj_info']
        print(obj_info)

        xml_string_byte = smpl_robot.export_xml_string()
        new_string = xml_add_obj(xml_string_byte, obj_info, obj_pose)
        
        mj_model = mujoco.MjModel.from_xml_string(new_string)
        mj_data = mujoco.MjData(mj_model)
        viewer =  mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back)
        
        time_step = 0
    elif chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "L":
        print("Render")
        rendering = not rendering
    elif chr(keycode) == "P":
        print(f"time {motion_time} {motion_time/dt}")
    else:
        print("not mapped", chr(keycode))
    
    
        
if __name__ == "__main__":
    device = torch.device("cpu")
    # motion_file = "data/amass_x/grab/singles/s8_teapot_pour_1.pkl"
    # motion_file = "data/amass_x/grab/singles_trim/s8_teapot_pour_1.pkl"
    # motion_file = "data/amass_x/grab/grab_flute.pkl"
    # motion_file = "data/amass_x/grab_flip/flip_test.pkl"
    motion_file = "data/amass_x/grab_flip/singles/s4_binoculars_pass_1_flip.pkl"
    # motion_file = "data/amass_x/grab/singles/s2_cubemedium_lift.pkl"
    # motion_file = "data/amass_x/grab/singles/s7_cylinderlarge_lift.pkl"
    # motion_file = "data/amass_x/grab/singles/s1_mug_pass_1.pkl"
    # motion_file = "data/amass_x/oakink/singles/035_power_drill.pkl"
    # motion_file = "data/amass_x/oakink/oakink_bps.pkl"
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, rendering = 0, 1, 0, set(), 0, 1/30, False, False
    motion_lib_cfg = EasyDict({
                    "motion_file": motion_file,
                    "device": torch.device("cpu"),
                    "fix_height": FixHeightMode.no_fix,
                    "min_length": -1,
                    "max_length": -1,
                    "im_eval": False,
                    "multi_thread": False ,
                    "smpl_type": 'smplx',
                    "randomrize_heading": True,
                    "device": device,
                })
    
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
        "fix_height": False, 
        "sim": "mujoco",
    }
    smpl_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )
    
    gender_beta = np.zeros((21))
    ################## Janky ##################
    motion_file = joblib.load(motion_file)
    seq_key = list(motion_file.keys())[0]
    obj_pose = motion_file[seq_key]['obj_data']['obj_pose']
    obj_info = motion_file[seq_key]['obj_data']['obj_info']
    beta = motion_file[seq_key]['beta']
    gender = motion_file[seq_key]['gender']
    if gender == "male": gender_beta[0] = 1
    if gender == "female": gender_beta[0] = 2
    
    
    ################## Janky ##################
    v_template = motion_file[seq_key]['v_template']
    smpl_robot.load_from_skeleton(v_template=torch.from_numpy(v_template).float())
    # smpl_robot.load_from_skeleton(gender=gender_beta[0:1])
    
    test_good = f"/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_good)
    # smpl_robot.write_xml("test.xml")
    
    sk_tree = SkeletonTree.from_mjcf(test_good)
    motion_lib = MotionLibSMPLObj(motion_lib_cfg)
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(21)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
    
    
    xml_string_byte = smpl_robot.export_xml_string()
    new_string = xml_add_obj(xml_string_byte, obj_info, obj_pose)
    open(test_good, "wb").write(new_string)
    
    mj_model = mujoco.MjModel.from_xml_path(test_good)
    # mj_model = mujoco.MjModel.from_xml_string(new_string)
    mj_data = mujoco.MjData(mj_model)
    
    mj_model.opt.timestep = dt
    smplx_end = (7 + (len(sk_tree._node_indices) - 1) * 3)
    
    ######### Table position #########
    mj_data.qpos[smplx_end + 7:smplx_end+10] = obj_pose[0, 7:10]
    mj_data.qpos[smplx_end + 10:smplx_end+14] = obj_pose[0, 10:14][[3, 0, 1, 2]]
    ######### Table position #########
            
    viewer =  mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back)

    for _ in range(len(sk_tree._node_indices)):
        add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))
    # Close the viewer automatically after 30 wall-seconds.


    while viewer.is_running():
        step_start = time.time()
        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(device), torch.tensor([motion_time]).to(device))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        o_pos, o_rot = motion_res["o_rb_pos"].squeeze().numpy(), motion_res["o_rb_rot"].squeeze().numpy()
        

        mj_data.qpos[:3] = root_pos[0].cpu().numpy()
        mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
        mj_data.qpos[7:smplx_end] = sRot.from_rotvec(dof_pos[0].cpu().numpy().reshape(-1, 3)).as_euler("XYZ").flatten()
        # mj_data.qpos[smplx_end:smplx_end+3] = o_pos
        # mj_data.qpos[smplx_end + 3:smplx_end+7] = o_rot[[3, 0, 1, 2]]
        mj_data.qpos[smplx_end:smplx_end+3] = obj_pose[int(motion_time/dt) % len(obj_pose), :3]
        mj_data.qpos[smplx_end + 3:smplx_end+7] = obj_pose[int(motion_time/dt) % len(obj_pose), 3:7][[3, 0, 1, 2]]
        
        mujoco.mj_forward(mj_model, mj_data)
        if not paused:
            time_step += dt

        # for i in range(rb_pos.shape[1]):
        #     viewer.user_scn.geoms[i].pos = rb_pos[0, i]
        
        
        contact_hand_dict = motion_lib.get_contact_hand_pose(torch.tensor([motion_id]).to(device))
        contact_hand_pos = contact_hand_dict['contact_hand_trans']
        for i in range(contact_hand_pos.shape[1]):
            viewer.user_scn.geoms[i].pos = contact_hand_pos[0, i]
            
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
        if rendering:
            import ipdb; ipdb.set_trace()
            # viewer.render()
