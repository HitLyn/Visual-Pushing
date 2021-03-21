import torch
import torch.nn as nn
import numpy as np

from VisualRL.rllib.common.distributions import SquashedDiagGaussianDistribution


class Actor(nn.Module):
    def __init__(
            self,
            action_space,
            feature_extractor,
            feature_dims,
            ):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.feature_extractor = feature_extrator
        self.feature_dims = feature_dims

        self.latent_pi_net = nn.Sequential(
                nn.Linear(feature_dims, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                )
        self.action_dist = SquashedDiagGaussianDistribution(action_space)
        self.mu = nn.Linear(64, action_space)
        self.log_std = nn.Linear(64, action_space)

    def get_action_parameters(self, obs):
        features = self.feature_extractor(obs)
        latent_pi = self.latent_pi_net(features)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)

        return mean_actions, log_std

    def forward(self, obs, determinstic = False):
        mean_actions, log_std = self.get_action_parameters(obs)

        return self.action_dist.action_from_params(mean_actions, log_std, determinstic = determinstic)

    def action_log_prob(self, obs):
        mean_actions, log_std = self.get_action_parameters(obs)

        return self.action_dist.log_prob_from_params(mean_actions, log_std)









