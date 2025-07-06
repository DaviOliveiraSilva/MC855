"""uavFL: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import flwr as fl
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector
from torchrl.modules import SafeModule, MLP, NormalParamWrapper
from uavfl.env_wrapper import make_torchrl_env

csv_path = 'uavfl/data/sensor_dataset.csv'

def get_weights(policy, value):
    return [val.detach().cpu().numpy() for val in list(policy.parameters()) + list(value.parameters())]


def set_weights(policy, value, parameters):
    policy_params = parameters[:len(list(policy.parameters()))]
    value_params = parameters[len(list(policy.parameters())):]
    for p, val in zip(policy.parameters(), policy_params):
        p.data = torch.tensor(val, dtype=torch.float32)
    for p, val in zip(value.parameters(), value_params):
        p.data = torch.tensor(val, dtype=torch.float32)
