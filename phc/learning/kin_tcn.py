# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch.nn as nn
import torch
from phc.learning.running_norm import RunningNorm


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, in_features, out_features,
                 filter_widths, causal, dropout, channels, ):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.in_features = in_features
        self.out_features = out_features
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.silu = nn.SiLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.shrink = nn.Conv1d(channels, out_features , 1)
        # self.norm = RunningNorm(self.in_features * self.num_joints_in) # ZL: disable for now. 

    def set_bn_momentum(self, momentum):
        # self.expand_bn.momentum = momentum
        # for bn in self.layers_bn:
        #     bn.momentum = momentum
        pass

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.in_features

        B, T, F = x.shape

        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, self.out_features)

        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self,
                 num_joints_in,
                 in_features,
                 num_joints_out,
                 filter_widths,
                 causal=False,
                 dropout=0.25,
                 channels=1024,
                 dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out,
                         filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features,
                                     channels,
                                     filter_widths[0],
                                     bias=False)

        layers_conv = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 *
                                      next_dilation) if causal else 0)

            layers_conv.append(
                nn.Conv1d(channels,
                          channels,
                          filter_widths[i] if not dense else
                          (2 * self.pad[-1] + 1),
                          dilation=next_dilation if not dense else 1,
                          bias=False))
            layers_conv.append(
                nn.Conv1d(channels, channels, 1, dilation=1, bias=False))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)

    def _forward_blocks(self, x):
        x = self.drop(self.silu(self.expand_conv(x)))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift:x.shape[2] - pad + shift]

            x = self.drop(self.silu(self.layers_conv[2 * i](x)))
            x = res + self.drop(self.silu(self.layers_conv[2 * i + 1](x)))

        x = self.shrink(x)
        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 filter_widths,
                 causal=False,
                 dropout=0.25,
                 channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__( in_features, out_features,
                         filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d( in_features,
                                     channels,
                                     filter_widths[0],
                                     stride=filter_widths[0],
                                     bias=False)

        layers_conv = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(
                nn.Conv1d(channels,
                          channels,
                          filter_widths[i],
                          stride=filter_widths[i],
                          bias=False))
            layers_conv.append(
                nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)

    def _forward_blocks(self, x):
        x = self.drop(self.silu(self.expand_conv(x)))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] +
                    self.filter_widths[i + 1] // 2::self.filter_widths[i + 1]]

            x = self.drop(self.silu((self.layers_conv[2 * i](x))))
            x = res + self.drop(self.silu((self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x
    
    
if __name__ == '__main__':
    tcn = TemporalModelOptimized1f(in_features = 4 * 30, out_features = 23, filter_widths = [3, 3], causal=False, dropout=0.2, channels=1024)
    input_val = torch.zeros(3, 10, 120) ### Batch, Time, Feat
    out = tcn(input_val)
    print(out.shape) # Batch, Time, Feat
    import ipdb; ipdb.set_trace()
    print('...')
    
