import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWritter

import numpy as np
from VisualRL.common.utils import get_device, set_seed_everywhere
from VisualRL.rllib.common.torch_layers import make_feature_extractor
from VisualRL.rllib.her.sac_policy import SACPolicy
from VisualRL.rllib.her.her_replay_buffer import HerReplayBuffer


class HER:
    def __init__(
            self,
            observation_space,
            action_space,
            max_episode_steps,
            feature_dims,
            normalizer,
            net_class,
            min_action,
            max_action,
            learning_rate = 3e-4,
            buffer_size = 1e6,
            learning_starts = 100,
            batch_size = 256,
            tau = 0.005,
            gamma = 0.99,
            device = None,
            seed = None,
            ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps
        self.feature_dims = feature_dims
        self.normalizer = normalizer
        self.net_class = net_class
        self.learning_rate = learning_rate
        self.min_action = min_action
        self.max_action = max_action
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.seed = seed

        self._episode_num = 0
        self.num_timesteps = 0

        set_seed_everywhere(self.seed)

        self.rollout_buffer = HerReplayBuffer(
                buffer_size,
                max_episode_steps,
                observation_space,
                action_space,
                device,
                )

        self.feature_extractor = make_feature_extractor(
                net_class,
                observation_space = observation_space,
                feature_dims = feature_dims,
                device = device
                )


        self.policy = SACPolicy(
                observation_space,
                action_space,
                self.feature_extractor,
                device,
                min_action,
                max_action,
                learning_rate = learning_rate,
                )
        self.policy.to(self.device)

        # entropy item
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
        self.log_ent_coef = torch.log(torch.ones(1, device = self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr = 1e-3)
        self.ent_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.ent_coef_optimizer, 0.999)

        # actor item
        self.actor = self.policy.actor
        self.actor_scheduler = self.policy.actor_scheduler
        # critic item
        self.critic = self.policy.cirtic
        self.critic_target = self.policy.critic_target
        self.critic_scheduler = self.policy.critic_scheduler

    def _update_learning_rate(self, schedulers):
        if not isinstance(optimizers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def to(self, device):
        self.device = device
        self.policy.to(device)

    def train(self, gradient_steps, batch_size):
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        self._update_learning_rate(schedulers)

        # train


    def _setup_learn(self):
        self._episode_num = 0
        self.num_timesteps = 0

    def collect_rollouts(self):
        pass


    def learn(self, env, total_timesteps, log_freq, eval_freq, num_eval_episodes):
        # setup model for learning process
        self._setup_learn()
        # rollout here
        self.collect_rollouts(self)
        # train








          



        



