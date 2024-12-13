from phc.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from phc.utils.isaacgym_torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn
from phc.env.tasks.humanoid_amp_task import HumanoidAMPTask

import phc.learning.replay_buffer as replay_buffer
import phc.learning.common_agent as common_agent

from tensorboardX import SummaryWriter
import copy
from phc.utils.torch_utils import project_to_norm
import phc.learning.amp_datasets as amp_datasets
from phc.learning.loss_functions import kl_multi
from smpl_sim.utils.math_utils import LinearAnneal

def load_my_state_dict(target, saved_dict):
    for name, param in saved_dict.items():
        if name not in target:
            continue

        if target[name].shape == param.shape:
            target[name].copy_(param)


class TaskAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

