import glob
import os
import sys
import pdb
import os.path as osp
import time  # Import time module for adding delay
sys.path.append(os.getcwd())

command = "python phc/run_hydra.py project_name=OmniGrasp exp_name=omnigrasp_1018_amp learning=omnigrasp_rnn_amp env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=data/amass_x/grab/grab_train_bps.pkl env.embedding_size=48 env.stateInit=Start env.models=[output/HumanoidIm/phcx_vae_dex_1_3_resume/Humanoid_00016000.pth] robot=smplx_humanoid robot.has_mesh=False sim=hand_sim learning.params.network.task_mlp.units=[256,256] learning.params.config.eval_frequency=100 learning.params.config.save_frequency=200 env.numTrajSamples=20 env.trajSampleTimestepInv=15 env.obj_pmcp_start=2500 env.traj_gen.speed_max=1.5 epoch=-1"


try:
    print(command)
    time.sleep(5)  # Wait for 10 seconds before retrying
    while True:
        try:
            os.system(command)
        except Exception as e:
            print(f"Command failed with exception: {e}")
        time.sleep(10)  # Wait for 10 seconds before retrying
except KeyboardInterrupt:
    print("Script interrupted by user. Exiting...")