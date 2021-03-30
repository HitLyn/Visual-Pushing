import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from IPython import embed

import numpy as np

from VisualRL.common.utils import get_device, set_seed_everywhere
from VisualRL.rllib.common.torch_layers import make_feature_extractor
from VisualRL.rllib.her.sac_policy import SACPolicy
from VisualRL.rllib.her.her_replay_buffer import HerReplayBuffer
from VisualRL.rllib.common.utils import polyak_update

ACTION_SCALE = 0.5
class HER:
    def __init__(
            self,
            observation_space,
            action_space,
            goal_space,
            feature_dims,
            min_action,
            max_action,
            max_episode_steps,
            train_freq,
            train_cycle,
            net_class = "MLP",
            target_update_interval = 2,
            save_interval = 50,
            gradient_steps = 5,
            learning_rate = 1e-3,
            buffer_size = 1e6,
            learning_starts = 10,
            batch_size = 256,
            tau = 0.005,
            gamma = 0.99,
            device = None,
            seed = 1,
            ):

        self.observation_space = observation_space # network size = 15
        self.action_space = action_space
        self.goal_space = goal_space
        self.buffer_obs_size = observation_space - goal_space # buffer obs shape = 9
        self.max_episode_steps = max_episode_steps
        self.feature_dims = feature_dims
        self.net_class = net_class
        self.learning_rate = learning_rate
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.min_action = min_action
        self.max_action = max_action
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.seed = seed
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.train_cycle = train_cycle

        self.dims = self.get_dims()

        self._episode_num = 0
        self.num_timesteps = 0
        self._n_updates = 0

        set_seed_everywhere(self.seed)

        self.rollout_buffer = HerReplayBuffer(
                buffer_size,
                max_episode_steps,
                self.buffer_obs_size,
                goal_space,
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
        self.target_entropy = -np.prod(self.action_space).astype(np.float32)
        self.log_ent_coef = torch.log(torch.ones(1, device = self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr = 1e-3)
        self.ent_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.ent_coef_optimizer, 0.999)

        # actor item
        self.actor = self.policy.actor
        self.actor_scheduler = self.policy.actor_scheduler
        # critic item
        self.critic = self.policy.critic
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
        self.dims['observation_space'] = self.observation_space
        self.dims['buffer_obs_size'] = self.buffer_obs_size
        self.dims['action'] = self.action_space
        self.dims['goal'] = self.goal_space
        return self.dims

    def _select_action(self, observation, achieved_goal, desired_goal):
        observation = observation.reshape(-1, self.dims['buffer_obs_size'])
        desired_goal = desired_goal.reshape(-1, self.dims['goal'])
        obs_input = np.concatenate([observation, desired_goal], axis = 1)
        obs_input = torch.as_tensor(obs_input).float().to(self.device)
        scaled_action = self.policy.predict(obs_input, determinstic = True).squeeze().cpu().numpy()
        return scaled_action

    def _sample_action(self, observation, achieved_goal, desired_goal):
        if self.num_collected_episodes < self.learning_starts:
            scaled_action = np.random.uniform(2*self.min_action, 2*self.max_action, size = self.dims['action'])
        else:
            # reshape and normalize observations for network
            observation = observation.reshape(-1, self.dims['buffer_obs_size'])
            desired_goal = desired_goal.reshape(-1, self.dims['goal'])
            obs_input = np.concatenate([observation, desired_goal], axis = 1)
            obs_input = torch.as_tensor(obs_input).float().to(self.device)
            scaled_action = self.policy.predict(obs_input, determinstic = False).squeeze().cpu().numpy()

        return scaled_action

    def collect_rollouts(self, env, writer):
        success_stats = []
        episode = 0
        print(f"collecting rollouts: {self.num_collected_episodes}")
        while episode < self.train_freq:
            obs_dict = env.reset()
            observation = np.empty(self.dims['buffer_obs_size'], np.float32)
            achieved_goal = np.empty(self.dims['goal'], np.float32)
            desired_goal = np.empty(self.dims['goal'], np.float32)
            observation[:] = obs_dict['observation']
            achieved_goal[:] = obs_dict['achieved_goal']
            desired_goal[:] = obs_dict['desired_goal']

            obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
            with torch.no_grad():
                for t in range(self.max_episode_steps):
                    observation_new = np.empty(self.dims['buffer_obs_size'], np.float32)
                    achieved_goal_new = np.empty(self.dims['goal'], np.float32)
                    # success = np.zeros(1)

                    # step env
                    action= self._sample_action(observation, achieved_goal, desired_goal) # action is squashed to [-1, 1] by tanh function
                    obs_dict_new, reward, done, _ = env.step(ACTION_SCALE*action)
                    observation_new[:] = obs_dict_new['observation']
                    achieved_goal_new[:] = obs_dict_new['achieved_goal']
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
                    achieved_goal[:] = achieved_goal_new.copy()

            obs.append(observation.copy())
            a_goals.append(achieved_goal.copy())

            episode_transition = dict(
                o = np.array(obs).copy(),
                u = np.array(acts).copy(),
                g = np.array(d_goals).copy(),
                ag = np.array(a_goals).copy())
            # stats
            episode += 1
            self.num_collected_episodes += 1
            success_stats.append(successes[-1])
            # add transition to replay buffer
            self.rollout_buffer.add_episode_transitions(episode_transition)

        success_rate = np.mean(np.array(success_stats))
        #TODO write success_rate to logger here
        writer.add_scalar("train/success_rate", success_rate, self._n_updates)


    def learn(self, env, total_episodes, eval_freq, num_eval_episodes, writer, model_path):
        # setup model for learning process
        self._setup_learn()
        # rollout and train model in turn
        while self.num_collected_episodes < total_episodes:
            self.collect_rollouts(env, writer)
            if self.num_collected_episodes >= self.learning_starts:
                for i in range(self.train_cycle):
                    self.train(self.gradient_steps, self.batch_size, writer)
                if self.num_collected_episodes % eval_freq == 0:
                    self.eval(env, num_eval_episodes, writer)
                    # save
                if self.num_collected_episodes % self.save_interval == 0:
                    self.save(model_path, self._n_updates)


    def eval(self, env, num_eval_episodes, writer):
        print(f"evaluate after {self.num_collected_episodes} episodes")
        reward_stats, success_rate_stats = [], []
        for episode in range(num_eval_episodes):
            obs_dict = env.reset()
            rewards = []
            with torch.no_grad():
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
        writer.add_scalar("eval/mean_reward", mean_reward, self._n_updates)
        writer.add_scalar("eval/success_rate", mean_success_rate, self._n_updates)
        print(f"success_rate: {mean_success_rate}")

    def train(self, gradient_steps, batch_size, writer):
        # optimizers
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        # train
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.rollout_buffer.sample(batch_size)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data["goal_obs_con"])
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef*(log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())
            # optimize entropy coefficient
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            with torch.no_grad():
                # select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data["next_goal_obs_con"])
                # compute next Q values
                next_q_values = torch.cat(self.critic_target(replay_data["next_goal_obs_con"], next_actions), dim = 1)
                next_q_values, _ = torch.min(next_q_values, dim = 1, keepdim = True)
                # add entropy term
                next_q_values = next_q_values - ent_coef*next_log_prob.reshape(-1, 1)
                # td error, entropy term
                target_q_values = replay_data["rewards"] + (1 - replay_data["dones"]) * self.gamma * next_q_values

            # get current Q values estimates for each critic net
            current_q_values = self.critic(replay_data["goal_obs_con"], replay_data["actions"])
            # compute critic loss
            # embed();exit()
            critic_loss = 0.5*sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())
            # optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # compute actor loss
            q_values_pi = torch.cat(self.critic(replay_data["goal_obs_con"], actions_pi), dim = 1)
            min_qf_pi, _ = torch.min(q_values_pi, dim = 1, keepdim = True)
            actor_loss = (ent_coef*log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            # optimize actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # updata target network
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        self._update_learning_rate(schedulers)
        # TODO write summary to logger here
        writer.add_scalar("train/ent_coef", np.mean(ent_coefs), self._n_updates)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), self._n_updates)
        writer.add_scalar("train/critic_loss", np.mean(critic_losses), self._n_updates)
        if len(ent_coef_losses) > 0:
            writer.add_scalar("train/ent_coef_loss", np.mean(ent_coef_losses), self._n_updates)


    def _update_learning_rate(self, schedulers):
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def save(self, path, step):
        torch.save(self.policy.state_dict(), "%s/her_%s.pt" % (path, step))

    def load(self, path, step):
        self.policy.load_state_dict(torch.load("%s/her_%s.pt" % (path, step)))
