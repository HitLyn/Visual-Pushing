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
            goal_space,
            feature_dims,
            net_class,
            min_action,
            max_action,
            max_episode_steps,
            train_freq,
            train_cycle,
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
        self.goal_space = goal_space
        self.max_episode_steps = max_episode_steps
        self.feature_dims = feature_dims
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

        self.dims = self.get_dims()

        self._episode_num = 0
        self.num_timesteps = 0
        self.train_freq = train_freq
        self.train_cycle = train_cycle

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
                self.feature_dims,
                self.feature_extractor,
                self.rollout_buffer,
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


    def to(self, device):
        self.device = device
        self.policy.to(device)

    def _setup_learn(self):
        self._episode_num = 0
        self.num_collected_episodes = 0

    def get_dims(self):
        self.dims = dict()
        self.dims['observation'] = self.observation_space
        self.dims['action'] = self.action_space
        self.dims['goal'] = self.goal_space
        return self.dims

    def _select_action(self, observation, achieved_goal, desired_goal):
        observation = observation.reshape(-1, self.dims['observation'])
        desired_goal = desired_goal.reshape(-1, self.dims['goal'])
        obs_input = np.concatenate([observation, desired_goal], axis = 1)
        scaled_action = self.policy.predict(obs_input, determinstic = True)

    def _sample_action(self, observation, achieved_goal, desired_goal):
        if self.num_collected_episodes < self.learning_starts:
            scaled_action = np.random.uniform(-1, 1, size = self.dims['action'])
        else:
            # reshape and normalize observations for network
            observation = observation.reshape(-1, self.dims['observation'])
            desired_goal = desired_goal.reshape(-1, self.dims['goal'])
            obs_input = np.concatenate([observation, desired_goal], axis = 1)
            scaled_action = self.policy.predict(obs_input, determinstic = False)

        return scaled_action

    def collect_rollouts(self, env):
        success_stats = []
        episode = 0
        while episode < self.train_freq:
            obs_dict = env.reset()
            observation = np.empty(self.dims['observation'], np.float32)
            achieved_goal = np.empty(self.dims['goal'], np.float32)
            desired_goal = np.empty(self.dims['goal'], np.float32)
            observation[:] = obs_dict['observation']
            achieved_goal[:] = obs_dict['achieved_goal']
            desired_goal[:] = obs_dict['desired_goal']

            obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
            for t in range(self.max_episode_steps):
                observation_new = np.empty(self.dims['observation'], np.float32)
                achieved_goal_new = np.empty(self.dims['goal'], np.float32)
                success = np.zeros(1)

                # step env
                action= self._sample_action(observation, achieved_goal, desired_goal) # action is squashed to [-1, 1] by tanh function
                obs_dict_new, reward, done, _ = env.step(ACTION_SCALE*action)
                observation_new = obs_dict_new['observation']
                achieved_goal_new = obs_dict_new['achieved_goal']
                success = np.array(obs_dict_new['is_success'])

                # store transitions
                dones.append(done)
                obs.append(observation.copy())
                a_goals.append(achieved_goal.copy())
                acts.append(action.copy())
                d_goals.append(desired_goal.copy())
                successes.append(success.copy())

                # update states
                observation[:] = observation_new.copy()
                achieved_goal[:] = achieved_goal.copy()

            obs.append(observation.copy())
            a_goals.append(achieved_goal.copy())

            episode_transition = dict(o = obs, u = acts, g = d_goals, ag = a_goals)
            # stats
            episode += 1
            self.num_collected_episodes += 1
            success_stats.append(successes[-1])
            success_rate = np.mean(np.array(success_stats))
            #TODO write success_rate to logger here

            # add transition to replay buffer
            self.roullout_buffer.add_episode_transition(episode_transition)


    def learn(self, env, total_episodes, log_freq, eval_freq, num_eval_episodes):
        # setup model for learning process
        self._setup_learn()
        # rollout and train model in turn
        while self.num_collected_episodes < total_episodes:
            self.collect_rollouts(env)
            if self.num_collected_episodes >= self.learning_starts:
                for i in self.train_cycle:
                    self.train()
                if self.num_collected_episodes//eval_freq == 0:
                    self.eval(env, num_eval_episodes)

    def eval(self, env, num_eval_episodes):
        reward_stats, success_rate_stats = [], []
        for episode in range(num_eval_episodes):
            obs_dict = env.reset()
            rewards = []
            for step in range(self.max_episode_steps):
                action = self._select_action(obs_dict['observation'], obs_dict['achieved_goal'], obs_dict['desired_goal'])
                obs_dict, reward, done, _ = env.step(ACTION_SCALE*action)
                rewards.append(reward)
            success = obs_dict['is_success']
            reward_stats.append(np.mean(np.array(rewards)))
            success_rate_stats.append(success)

        mean_reward = np.mean(np.array(reward_stats))
        mean_success_rate = np.mean(np.array(success_rate_stats))
        # TODO write stats to logger here

    def train(self, gradient_steps, batch_size):
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        self._update_learning_rate(schedulers)

        # train


    def _update_learning_rate(self, schedulers):
        if not isinstance(optimizers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def save(self, path, step):
        torch.save(self.policy.state_dict(), "%s/her_%s.pt" % (path, step))

    def load(self, path, step):
        self.policy.load_state_dict(torch.load("%s/her_%s.pt" % (path, step))
