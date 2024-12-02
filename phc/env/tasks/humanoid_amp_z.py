import time
import torch
import phc.env.tasks.humanoid_amp as humanoid_amp
from phc.env.tasks.humanoid_amp import remove_base_rot
from phc.utils import torch_utils
from typing import OrderedDict

from phc.utils.isaacgym_torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.utils.torch_utils import project_to_norm

from phc.utils.motion_lib_smpl import MotionLibSMPL 
from phc.learning.network_loader import load_z_encoder, load_z_decoder

from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode

HACK_MOTION_SYNC = False

class HumanoidAMPZ(humanoid_amp.HumanoidAMP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        super().step_z(actions)
        return
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props_z(self, key_bodies)
        return