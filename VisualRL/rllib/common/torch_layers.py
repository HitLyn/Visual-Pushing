import torch
import torch.nn as nn
from IPython import embed

import numpy as np



class GRU_Extractor(nn.Module):
    def __init__(self, observation_space, feature_dims, device):
        super(GRU_Extractor, self).__init__()
        if isinstance(observation_space, tuple):
            input_dim = observation_space[1]
        else:
            input_dim = observation_space

        self.input_dim = input_dim
        self.feature_dims = feature_dims
        self.device = device
        gru = nn.GRU(input_dim, self.feature_dims)
        self.gru = gru.to(self.device)

    def forward(self, obs):
        # obs :(t_len, batch_size, obs_shape)
        # return (batch_size, hidden_size)
        gru_out, hn = self.gru(obs)
        hn = hn[0]
        return hn


class Linear_Extractor(nn.Module):
    def __init__(self, observation_space, feature_dims, device):
        super(Linear_Extractor, self).__init__()
        if isinstance(observation_space, tuple):
            input_dim = observation_space[0] * observation_space[1]
        else:
            input_dim = observation_space

        self.input_dim = input_dim
        self.feature_dims = feature_dims
        self.device = device
        mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, feature_dims), nn.Tanh(),
        )
        self.mlp = mlp.to(self.device)

    def forward(self, obs):
        # obs: (t_len, batch_size, obs_shape)
        # swap axis and flatten
        obs = obs.clone()
        obs = obs.transpose(0, 1)
        batch_size = obs.shape[0]
        return self.mlp(obs.reshape(batch_size, -1))


def make_feature_extractor(net_class, observation_space, feature_dims, device):
    assert net_class in ('MLP', 'GRU', 'TCN')
    if net_class == 'MLP':
        return Linear_Extractor(observation_space, feature_dims, device)
    elif net_class == 'GRU':
        return GRU_Extractor(observation_space, feature_dims, device)
    else:
        return TCN_Extractor(observation_space, feature_dims, device)