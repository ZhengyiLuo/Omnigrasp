

import os.path as osp
from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im as humanoid_im
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from phc.utils.isaacgym_torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from collections import defaultdict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import open3d as o3d
from datetime import datetime
import imageio
from collections import deque
from tqdm import tqdm
import copy


class HumanoidImInterhand(humanoid_im.HumanoidIm):
    pass