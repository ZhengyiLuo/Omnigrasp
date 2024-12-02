import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import numpy as np
from phc.learning.im_amp_players import compute_metrics_obj
from collections import defaultdict


 
# exp_name = "omnigrasp_oakink_pregrasp_airtime"



# motion_file = "data/amass_x/omomo/omomo_bps.pkl"


learning = "omnigrasp_rnn"
fixed_latent="True"
latent_mode = 'output/HumanoidIm/phcx_vae_dex_1_3_resume/Humanoid_00016000.pth'
addition = ""
# addition = "+env.add_noise=True +env.noise_scale=0.025"
# exp_name = "omnigrasp_abla_noz"; epoch=78000 ; learning = "omnigrasp_noz"; task="HumanoidOmniGrab" # 25000
# exp_name = "omnigrasp_15_20_fix"; epoch=28000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" 
# exp_name = "omni_grab_abla_im"; epoch=9000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ"; addition="env.has_im_obs=True" # 25000
# exp_name = "omnigrasp_abla_no_obj_rand_pos"; epoch=15000 ; learning = "omnigrasp_rnn"; 
# exp_name = "omnigrasp_15_20_repo_vel"; epoch=28000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_0709_nodouble"; epoch=11000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_0722_nodouble"; epoch=34000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_0722_hasdouble"; epoch=34000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_0715_nodouble"; epoch=35000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
exp_name = "omnigrasp_0802_hasdouble"; epoch=26800 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_0805_vae"; epoch=17800 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" # 25000

# exp_name = "omnigrasp_abla_noz_amp"; epoch=18000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrab" # 25000
# exp_name = "omnigrasp_15_20_500"
# exp_name = "omnigrasp_abla_nornn"; epoch=27000 ; learning = "omnigrasp"
# exp_name = "omnigrasp_abla_no_pregrasp_standing"; epoch=20000 ; learning = "omnigrasp_rnn"# 25000
# exp_name = "omni_grab_15_20" ; epoch=15000 ; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_abla_no_obj_pmcp";  epoch=19000 ; task="HumanoidOmniGrabZ" # 25000
# exp_name = "omnigrasp_oakink_warm" ; epoch=20000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGraspZ" # 25000
# exp_name = "omnigrasp_oakink_pregrasp_airtime" ; epoch=29000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGraspZ" # 25000
# exp_name = "omnigrasp_abla_nodex";  latent_mode = 'output/HumanoidIm/phcx_vae_48/Humanoid_00015500.pth'; epoch=19000
# exp_name = "omnigrasp_abla_nolatent"; epoch=28000 ; learning = "omnigrasp_rnn"; task="HumanoidOmniGrabZ" ; fixed_latent="False"# 25000
# exp_name = "omnigrasp_omomo_warm"; epoch=26000 ; learning = "omnigrasp_rnn"# 25000

motion_file = "data/amass_x/oakink/oakink_virtual_grasp_train_bps.pkl"; task="HumanoidOmniGraspZ"
# motion_file = "data/amass_x/oakink/oakink_virtual_test_bps.pkl"; task="HumanoidOmniGraspZ"
# motion_file = "data/amass_x/grab/grab_train_bps.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_train_nodouble_bps.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_train_bps.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_test_goal.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_test_goal_bps_fixheight.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_test_imos_bps_fixheight.pkl" ; #task="HumanoidOmniGrabZ"
# motion_file = "data/amass_x/grab/grab_test_goal_bps_fixheight.pkl" ; #task="HumanoidOmniGrabZ"



# task="HumanoidOmniGraspZ"


num_tests = 10
test_termination_distances = 0.25

motion_data = joblib.load(motion_file)
motion_data_keys = list(motion_data.keys())
motion_file_name = motion_file.split("/")[-1].split(".")[0]
obj_to_seq_dict = defaultdict(list)
obj_succ_traj_dict, obj_succ_grasp_dict, obj_ttr_dict = defaultdict(list), defaultdict(list), defaultdict(list)
pred_pos_succ, gt_pos_succ, pred_rot_succ, gt_rot_succ, success_lift_all, succ_all = [], [], [], [], [], []
seq_name_success = defaultdict(list)


for k in motion_data_keys:
    obj_name = motion_data[k]['obj_data']['obj_info'][0].split("/")[-1].split(".")[0]
    obj_to_seq_dict[obj_name].append(k)
    
cmd = f"python phc/run_hydra.py  project_name=OmniGrasp   exp_name={exp_name} \
    learning={learning} env=env_x_grab_z env.task={task} env.motion_file={motion_file} env.embedding_size=48 \
    env.stateInit=Start env.models=['{latent_mode}'] robot=smplx_humanoid robot.has_mesh=False sim=hand_sim learning.params.network.task_mlp.units=[256,256]  \
    learning.params.config.eval_frequency=250 env.numTrajSamples=20 env.trajSampleTimestepInv=15 test=True env.num_envs={len(motion_data_keys)}  headless=True im_eval=True epoch={epoch} env.test_termination_distances={test_termination_distances} \
    env.fixed_latent={fixed_latent} " + addition

print(cmd)
for _ in range(num_tests):
    os.system(cmd)

for eval_file in glob.glob(f"output/HumanoidIm/{exp_name}/eval/res_{motion_file_name}_{epoch}_*"):
    eval_dict = joblib.load(eval_file)
    assert(len(motion_data_keys) == len(eval_dict['success_lift']))
    pred_pos_succ += eval_dict['pred_pos_all_succ']; gt_pos_succ += eval_dict['gt_pos_all_succ']
    pred_rot_succ += eval_dict['pred_rot_all_succ']; gt_rot_succ += eval_dict['gt_rot_all_succ']
    success_lift_all.append(eval_dict['success_lift']), succ_all.append(eval_dict['terminate_hist'])
    counter = 0
    for i, seq_name in enumerate(motion_data_keys):
        obj_name = motion_data[seq_name]['obj_data']['obj_info'][0].split("/")[-1].split(".")[0]
        obj_succ_traj_dict[obj_name].append(eval_dict['terminate_hist'][i])
        
        obj_succ_grasp_dict[obj_name].append(1 - eval_dict['success_lift'][i])
        seq_name_success[seq_name].append(1 - eval_dict['terminate_hist'][i])
        
        if not eval_dict['terminate_hist'][i]:
            TTR = np.linalg.norm(eval_dict['pred_pos_all_succ'][counter] - eval_dict['gt_pos_all_succ'][counter], axis=2) < 0.12
            obj_ttr_dict[obj_name].append(TTR)
            counter += 1
        


metrics_succ = compute_metrics_obj(pred_pos_succ, gt_pos_succ, pred_rot_succ, gt_rot_succ)
metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}


print(f'===================================Addition: {addition}=================================')
print("Metrics: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
print(f"Traj succ: {1 - np.concatenate(succ_all).sum()/np.concatenate(succ_all).shape[0]:.3f}")
print('---------------------------')
for k, v in obj_succ_traj_dict.items():
    print(f"{k}:  {1 - np.array(v).sum()/np.array(v).shape[0]:.3f}", end = " ")
    
print("\n")
print('---------------------------')
print(f"Grasp succ: {np.concatenate(success_lift_all).sum()/np.concatenate(success_lift_all).shape[0]:.3f}")
for k, v in obj_succ_grasp_dict.items():
    print(f"{k}:  {1 - np.array(v).sum()/np.array(v).shape[0]:.3f}", end = " ")

print("\n")
print(f"======>{exp_name} {epoch} {motion_file_name} {test_termination_distances}======>")

seq_name_success = {k: np.mean(v) for k, v in seq_name_success.items()}
seq_name_failed = [k for k in seq_name_success.keys() if seq_name_success[k] != 1]
import ipdb; ipdb.set_trace()

print('....')
for k, v in obj_ttr_dict.items():
    print(f"{k}:  { np.concatenate(v).sum()/np.concatenate(v).shape[0]:.3f}", end = " ")
print("\n")
