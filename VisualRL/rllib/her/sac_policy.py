import torch
import torch.nn as nn

import numpy as np

from VisualRL.rllib.her.actor import Actor

class SACPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            feature_dims,
            feature_exactor,
            rollout_buffer,
            device,
            min_action,
            max_action,
            learning_rate
            ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.feature_dims = feature_dims
        self.feature_exactor = feature_exactor
        self.rollout_buffer = rollout_buffer
        self.device = device
        self.min_action = min_action
        self.max_action = max_action
        self.initial_learning_rate = learning_rate

        # entropy item
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
        self.log_ent_coef = torch.log(torch.ones(1, device = self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr = 1e-3)
        self.ent_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.ent_coef_optimizer, 0.999)
        # build actor and critic
        self.actor = self.make_actor(feature_extractor = self.feature_exatractor)
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.initial_learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor.optimizer, 0.999)

        self.critic = self.make_critic(feature_extractor = self.feature_extractor)
        self.critic.optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.initial_learning_rate)
        self.critic_schedulrt = torch.optim.lr_scheduler.ExponentialLR(self.critic.optimizer, 0.999)

    def _update_learning_rate(self, schedulers):
        if not isinstance(optimizers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def make_actor(self, feature_extractor):
        actor_kwargs = dict()
        actor_kwargs["actins_space"] = self.actions_space
        actor_kwargs["feature_extractor"] = self.feature_extractor
        actor_kwargs["feature_dims"] self.feature_dims

        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, feature_extractor):
        pass

    def train(self, gradient_steps, batch_size):
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        self._update_learning_rate(schedulers)

        # train

