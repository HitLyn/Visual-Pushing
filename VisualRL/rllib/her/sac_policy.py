import torch
import torch.nn as nn

import numpy as np

class SACPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            feature_exactor,
            rollout_buffer,
            device,
            min_action,
            max_action,
            learning_rate
            ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.feature_exactor = feature_exactor
        self.rollout_buffer = rollout_buffer
        self.device = device
        self.min_action = min_action
        self.max_action = max_action
        self.initial_learning_rate = learning_rate





    def _update_learning_rate(self, schedulers):
        if not isinstance(optimizers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def train(self, gradient_steps, batch_size):
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        self._update_learning_rate(schedulers)

        # train

