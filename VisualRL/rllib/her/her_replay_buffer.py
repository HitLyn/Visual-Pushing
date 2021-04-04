import numpy as np
from robogym.utils import rotation
from IPython import embed
import torch
import torch.nn as nn

class HerReplayBuffer:
    def __init__(self,
            size_in_transitions,
            episode_steps,
            obs_shape,
            goal_shape,
            action_shape,
            device,
            pos_threshold = 0.04,
            rot_threshold = 0.3,
            relative_goal = True,
            ):
        self.size_in_transitions = size_in_transitions
        self.size = int(self.size_in_transitions//episode_steps)
        self.episode_steps = episode_steps
        self.obs_shape = obs_shape
        self.goal_shape = goal_shape
        self.action_shape = action_shape
        self.device = device
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold
        self.relative_goal = relative_goal

        # buffer
        self.obses = np.empty([self.size, self.episode_steps + 1, self.obs_shape], np.float32)
        self.a_goals = np.empty([self.size, self.episode_steps + 1, self.goal_shape], np.float32)
        self.d_goals = np.empty([self.size, self.episode_steps, self.goal_shape], np.float32)
        self.actions = np.empty([self.size, self.episode_steps, self.action_shape], np.float32)
        self.dones = np.zeros([self.size, self.episode_steps, 1])
        self.dones[:, -1] = 1.
        # counter
        self.current_size = 0
        self.n_transitions_stored = 0
        # her replay params
        self.replay_k = 4

    def full(self):
        return self.current_size == self.size

    def add_episode_transitions(self, transition_dict):
        # find idx to store transitions
        idx = self._get_storage_idx()
        self.obses[idx] = transition_dict["o"]
        self.a_goals[idx] = transition_dict["ag"]
        self.d_goals[idx] = transition_dict["g"]
        self.actions[idx] = transition_dict["u"]
        # update size counter
        self.current_size += 1
        self.n_transitions_stored += self.episode_steps

    def add_episode_transitions_list(self, transition_dict_list):
        # multiprocess store
        for transition_dict in transition_dict_list:
            self.add_episode_transitions(transition_dict)


    def _get_storage_idx(self):
        idx = self.current_size % self.size
        return idx

    def sample(self, batch_size):
        buffer_ = dict()
        buffer_["obses"] = self.obses[:self.current_size].copy()
        buffer_["a_goals"] = self.a_goals[:self.current_size].copy() # achieved goal before action
        buffer_["d_goals"] = self.d_goals[:self.current_size].copy()
        buffer_["actions"] = self.actions[:self.current_size].copy()
        buffer_["next_obses"] = buffer_["obses"][:, 1:, :].copy()
        buffer_["a_goals_"] = buffer_["a_goals"][:, 1:, :].copy() # achieved goal after action
        buffer_["dones"] = self.dones[:self.current_size].copy()

        transitions = self.sample_transitions(buffer_, batch_size)

        return transitions

    def sample_transitions(self, buffer_, batch_size, device = None):

        if device is None:
            device = self.device

        future_p = 1 - (1./(1 + self.replay_k))
        T = buffer_["actions"].shape[1]
        episode_nums = buffer_["actions"].shape[0]
        # select episode and timesteps to use
        episode_idxs = np.random.randint(0, episode_nums, batch_size)
        t_samples = np.random.randint(0, T, batch_size)
        transitions = {key: buffer_[key][episode_idxs, t_samples].copy()
                for key in buffer_.keys()}

        # substitute in future goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size = batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = buffer_["a_goals"][episode_idxs[her_indexes], future_t].copy()
        transitions["d_goals"][her_indexes] = future_ag

        # recompute rewards
        reward_params = {k: transitions[k] for k in ["a_goals_", "d_goals"]}
        # embed();exit()
        transitions["rewards"] = self.reward_function(**reward_params).reshape(-1, 1)
        # embed();exit()
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # concatenate desired goals with observation together for network
        if self.relative_goal:
            transitions["goal_obs_con"] = np.concatenate([transitions["obses"], transitions["d_goals"] - transitions["a_goals"]], axis = 1)
            transitions["next_goal_obs_con"] = np.concatenate([transitions["next_obses"], transitions["d_goals"] - transitions["a_goals_"]], axis = 1)

        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key]).float().to(device)

        return transitions

    def reward_function(self, **parameters):
        # calculate relative goal
        relative_goal = {}
        relative_goal['obj_pos'] = parameters['a_goals_'][:, :3] - parameters['d_goals'][:, :3]
        relative_goal['obj_rot'] = parameters['a_goals_'][:, 3:] - parameters['d_goals'][:, 3:]
        pos_distances = np.linalg.norm(relative_goal["obj_pos"], axis=-1)
        rot_distances = rotation.quat_magnitude(
            rotation.quat_normalize(rotation.euler2quat(relative_goal["obj_rot"]))
        )
        success = np.array((pos_distances < self.pos_threshold) * (rot_distances < self.rot_threshold))
        success = success.astype(float)
        return success
