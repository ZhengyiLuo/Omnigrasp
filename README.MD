# Omnigrasp: Simulated Humanoid Grasping on Diverse Objects

Official implementation of NeurIPS 2024 paper: "Omnigrasp: Simulated Humanoid Grasping on Diverse Objects". In this project, we control a simulated humanoid to grasp diverse objects and follow diverse object trajectories. 

[[paper]](https://arxiv.org/abs/2407.11385) [[website]](https://zhengyiluo.github.io/Omnigrasp/) 

<div float="center">
  <img src="assets/omnigrasp_teaser.gif" />
</div>


## News 🚩


## TODOs


### Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytorch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```


Make sure you have the SMPL-X paramters properly setup by running the following scripts:
```
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

The SMPL model is used to adjust the height the humanoid robot to avoid penetnration with the ground during data loading. 

4. Use the following script to download trained models and sample data.

```
bash download_data.sh
```


## Evaluation 


### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| r | reset episode |
| j | apply large force to the humanoid |
| l | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| m | cancel termination based on imitation |

... more shortcut can be found in `phc/env/tasks/base_task.py`


To evaluate a trained policy on the GRAB dataset, run the following script:

```
python phc/run_hydra.py  \
    project_name=OmniGrasp   exp_name=omnigrasp_neurips_grab  \
    learning=omnigrasp_rnn \
    env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/hammer.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15\
    robot=smplx_humanoid sim=hand_sim  \
    epoch=-1 test=True env.num_envs=45 headless=False
```

Rendering the above trajectory might be slow, to speed up, use less number of environments or disable object trajectory rendering by press "k". 


## Training 

[This section is under construction]

### Data processing:

To train Omnigrasp, we need to process the GRAB dataset. Notice that while Omnigrasp does not depend on the GRAB dataset (we train on oakink and OMOMO as well), the GRAB dataset is used as a pre-grasp provider and inital pose provider. When training on oakink/OMOMO, a single sample motion from GRAB is used as the initial pose. 

To setup the GRAB dataset, please follow the following instructions:

1. Download the GRAB dataset from [here](https://grab.is.tue.mpg.de/).
2. Run the following script to process the dataset:

```
python scripts/data_process/convert_grab_smplx.py
```

