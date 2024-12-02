# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dyno.utils.quaternion import Quaternion, get_reference_frame, get_reference_frame_onnx


# device = "cuda:1"


def build_indices(num_dofs):
    idx = {}
    i = 0
    for name, length in [
        ("POS", 3),
        ("ORI", 4),
        ("VEL", 3),
        ("OMEGA", 3),
        ("Q", num_dofs),
        ("QD", num_dofs),
        ("SCALE", 1),
        ("CONTACT", 6),
        ("POS_L", 3),
        ("POS_R", 3)
    ]:
        idx[name] = list(range(i, i + length))
        i += length

    return idx, i



class WorldModel(torch.nn.Module):
    """A Simulator equivalent."""

    def __init__(self, idx_links):
        super().__init__()

        self.IDX, dim_state = build_indices(12)

        # masking input
        self.mask_x = torch.zeros(1, dim_state + 12, dtype=torch.bool)
        self.mask_x[:, self.IDX["POS_L"]] = True
        self.mask_x[:, self.IDX["POS_R"]] = True
        self.mask_x[:, self.IDX["CONTACT"]] = True


        self.dim_state = dim_state
        dim_torques = 12        
        n_hidden = 512 


        self.idx_links = idx_links

        # architecture
        dim_in = self.dim_state + dim_torques
        dim_out = self.dim_state
        self.layers = nn.Sequential(
        #   nn.BatchNorm1d(dim_in, momentum=mom), 
          # input layer
          nn.Linear(dim_in,  n_hidden, bias=True),  
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.Tanh(),
          # hidden layer 1
          nn.Linear(n_hidden, n_hidden, bias=True), 
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.Tanh(),
          # hidden layer 2
          nn.Linear(n_hidden, n_hidden, bias=True), 
        #   nn.BatchNorm1d(n_hidden, momentum=mom), 
          nn.Tanh(),
          # output layer
          nn.Linear(n_hidden, dim_out),
        )

        # set last layer weights low to not immediately blow up orientation
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.001)
            self.layers[-1].bias.zero_()

        self.render_frame = False


    # def forward(self, curr_W: Dict[str, torch.Tensor], actions: torch.Tensor) -> None:
    def forward(self, 
        q, qd, root_W, root_vel_W, W_Rq_root, root_omega_W, scale,
        actions: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Simulated current state one step forward"""

        curr_W = self.to_tensor(q, qd, root_W, root_vel_W, W_Rq_root, root_omega_W, scale, torch.zeros(1), torch.zeros(1))

        if self.training:
            ref_frame = get_reference_frame(root_W[:,0], W_Rq_root[:,0])
        else:
            ref_frame = get_reference_frame_onnx(root_W[:,0], W_Rq_root[:,0])
        curr_P = world_to_reference_frame(curr_W, self.IDX, ref_frame["pos_W"], ref_frame["Rmat_PW"])

        next_P = self.local_step(curr_P, actions)

        # reference to world frame
        # base position
        root_W = ref_frame["pos_W"] + change_frame(ref_frame["Rmat_WP"], next_P[:, self.IDX["POS"]])
        root_W = root_W.unsqueeze(1)
        # orientation
        Rmat_PB = Quaternion.batch_to_matrix(next_P[:, self.IDX["ORI"]])
        W_Rq_root = Quaternion.batch_from_matrix(ref_frame["Rmat_WP"] @ Rmat_PB) # Rq_WB
        W_Rq_root = W_Rq_root.unsqueeze(1)
        # linear velocities
        root_vel_W = change_frame(ref_frame["Rmat_WP"], next_P[:, self.IDX["VEL"]])
        root_vel_W = root_vel_W.unsqueeze(1)
        # angular root vel
        root_omega_W = change_frame(ref_frame["Rmat_WP"], next_P[:, self.IDX["OMEGA"]])
        root_omega_W = root_omega_W.unsqueeze(1)
        # joints (unchanged)
        q  = next_P[:, self.IDX["Q"]]
        qd = next_P[:, self.IDX["QD"]]

        return q, qd, root_W, root_vel_W, W_Rq_root, root_omega_W, scale


    @torch.jit.export
    def local_step(self, curr_P: torch.Tensor, action:torch.Tensor) -> torch.Tensor:

        x = torch.cat([curr_P, action], dim=-1)

        # don't use the position of the feet in the predictions
        x = torch.where(self.mask_x, torch.zeros(1).float(), x)

        # forward pass
        next_P = curr_P + self.layers(x)

        # some post processing
        # wedge this in the middle, since onnx opset=9 doesnt' allow index_put
        P_Rq_root = standardize(next_P[:, self.IDX["ORI"]])
        next_P = torch.cat((next_P[:, 0:self.IDX["ORI"][0]], P_Rq_root, next_P[:, self.IDX["ORI"][-1]+1:]), dim=1)

        if self.training:
            next_P[:, self.IDX["CONTACT"]] = torch.sigmoid(next_P[:, self.IDX["CONTACT"]]) # make probability [0,1]

        return next_P

    
    def to_tensor(self, q, qd, root_W, root_vel_W, W_Rq_root, root_omega_W, scale, contact_flag, link_W) -> torch.Tensor:
        """Convert a dictionary into a tensor"""

        B = q.shape[0]
        lst = {}
        lst["POS"] = root_W[:, 0]
        lst["ORI"] = W_Rq_root[:, 0]
        lst["VEL"] = root_vel_W[:, 0]
        lst["OMEGA"] = root_omega_W[:, 0]
        lst["Q"] = q
        lst["QD"] = qd
        lst["SCALE"] = scale

        if self.training:
            lst["CONTACT"] = contact_flag.float()
            lst["POS_L"] = link_W[:, self.idx_links["b_l_talocrural"]]
            lst["POS_R"] = link_W[:, self.idx_links["b_r_talocrural"]]
        else:
            lst["CONTACT"] = torch.zeros(B,6) # contact
            lst["POS_L"] = torch.zeros(B,3) # left foot
            lst["POS_R"] = torch.zeros(B,3) # right foot

        curr_W = torch.cat([lst[k] for k in self.IDX], dim=1)
        return curr_W


def change_frame(B_Rmat_A, link_A):
    return (B_Rmat_A @ link_A.unsqueeze(-1)).squeeze(-1)


def standardize(q):
    """Makes sure the real part is positive and quaternion is normalized"""
    norm = torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True)) + 1e-8
    return q * q[..., 3].sign().unsqueeze(-1) / norm


def world_to_reference_frame(curr_W: torch.Tensor, IDX: Dict[str, List[int]], pos_W: torch.Tensor, Rmat_PW: torch.Tensor) -> torch.Tensor:
    """Converts a tensor expressed in world frame (W) to a tensor expressed in P frame."""

    if len(curr_W.shape) == 2:

        lst = {}

        # cartesian positions
        lst["POS"] = change_frame(Rmat_PW, curr_W[:, IDX["POS"]] - pos_W)
        lst["POS_L"] = change_frame(Rmat_PW, curr_W[:, IDX["POS_L"]] - pos_W)
        lst["POS_R"] = change_frame(Rmat_PW, curr_W[:, IDX["POS_R"]] - pos_W)
        # root orientation
        Rmat_WB = Quaternion.batch_to_matrix(curr_W[:, IDX["ORI"]])
        lst["ORI"] = Quaternion.batch_from_matrix(Rmat_PW @ Rmat_WB) # Rmat_PB
        lst["ORI"] = standardize(lst["ORI"])
        # root velocities
        lst["VEL"] = change_frame(Rmat_PW, curr_W[:, IDX["VEL"]])
        lst["OMEGA"] = change_frame(Rmat_PW, curr_W[:, IDX["OMEGA"]])
        # joints (unchanged)
        lst["Q"] = curr_W[:,  IDX["Q"]]
        lst["QD"] = curr_W[:, IDX["QD"]]
        # scale (unchanged)
        lst["SCALE"] = curr_W[:, IDX["SCALE"]]
        # contact flag
        lst["CONTACT"] = curr_W[:, IDX["CONTACT"]]
        # combine
        curr_P = torch.cat([lst[k] for k in IDX], dim=-1)


    elif len(curr_W.shape) == 3:
        # hacky copy, horrible
        # root position
        curr_P = torch.zeros_like(curr_W)
        curr_P[:, :, IDX["POS"]] = change_frame(Rmat_PW, curr_W[:, :, IDX["POS"]] - pos_W)
        curr_P[:, :, IDX["POS_L"]] = change_frame(Rmat_PW, curr_W[:, :, IDX["POS_L"]] - pos_W)
        curr_P[:, :, IDX["POS_R"]] = change_frame(Rmat_PW, curr_W[:, :, IDX["POS_R"]] - pos_W)
        # root orientation
        Rmat_WB = Quaternion.batch_to_matrix(curr_W[:, :, IDX["ORI"]])
        curr_P[:, :, IDX["ORI"]] = Quaternion.batch_from_matrix(Rmat_PW @ Rmat_WB) # Rmat_PB
        curr_P[:, :, IDX["ORI"]] = standardize(curr_P[:, :, IDX["ORI"]])
        # root velocities
        curr_P[:, :, IDX["VEL"]] = change_frame(Rmat_PW, curr_W[:, :, IDX["VEL"]])
        curr_P[:, :, IDX["OMEGA"]] = change_frame(Rmat_PW, curr_W[:, :, IDX["OMEGA"]])
        # joints (unchanged)
        curr_P[:, :, IDX["Q"]] = curr_W[:, :,  IDX["Q"]]
        curr_P[:, :, IDX["QD"]] = curr_W[:, :, IDX["QD"]]
        # scale (unchanged)
        curr_P[:, :, IDX["SCALE"]] = curr_W[:, :, IDX["SCALE"]]
        # contact flag
        curr_P[:, :, IDX["CONTACT"]] = curr_W[:, :, IDX["CONTACT"]]
    else:
        raise ValueError("Unsupported number of dimensions")

    return curr_P


def reference_to_world_frame(next_P: torch.Tensor, IDX: Dict[str, List[int]], pos_W: torch.Tensor, Rmat_WP: torch.Tensor) -> torch.Tensor:
    """Converts a tensor expressed in P frame to a tensor expressed in world frame (W)."""
    assert len(next_P.shape) == 2

    lst = {}
    # base position
    lst["POS"] = pos_W + change_frame(Rmat_WP, next_P[:, IDX["POS"]])
    # orientation
    Rmat_PB = Quaternion.batch_to_matrix(next_P[:, IDX["ORI"]])
    lst["ORI"] = Quaternion.batch_from_matrix(Rmat_WP @ Rmat_PB) # Rq_WB
    # velocities
    lst["VEL"] = change_frame(Rmat_WP, next_P[:, IDX["VEL"]])
    lst["OMEGA"] = change_frame(Rmat_WP, next_P[:, IDX["OMEGA"]])
    # joints (unchanged)
    lst["Q"]  = next_P[:, IDX["Q"]]
    lst["QD"] = next_P[:, IDX["QD"]]
    # scale (unchanged)
    lst["SCALE"] = next_P[:, IDX["SCALE"]]
    # contact flag (unchanged)
    lst["CONTACT"] = next_P[:, IDX["CONTACT"]]
    # positions (WRONG FRAME)
    lst["POS_L"] = next_P[:, IDX["POS_L"]]
    lst["POS_R"] = next_P[:, IDX["POS_R"]]

    next_W = torch.cat([lst[k] for k in IDX], dim=1)


    return next_W