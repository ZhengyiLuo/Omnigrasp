# This is the overall forward pass of the model. 

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import torch

class ModelTaskContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('task', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelTaskContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

