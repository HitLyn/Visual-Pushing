import numpy as np

class HerReplayBuffer:
    def __init__(
            size_in_transitions,
            episode_steps,
            obs_shape,
            goal_shape,
            action_shape,
            device):

        self.size = self.size_in_transitions//eipsode_steps
        self.size_in_transitions = size_in_transitions
        self.episode_steps = episode_steps
        self.obs_shape = obs_shape
        self.goal_shape = goal_shape
        self.action_shape = action_shape
        self.device = device

        # buffer
        self.obses = np.empty([self.size, self.episode_steps + 1, self.obs_shape], np.float32)
        self.a_goals = np.empty([self.size, self.episode_steps + 1, self.goal_shape], np.float32)
        self.d_goals = np.empty([self.size, self.episode_steps, self.goal_shape], np.float32)
        self.actions = np.empty([self.size, self.episode_steps, self.actions_shape], np.float32)
        self.dones = np.zeros([self.size, self.episode_stpes])
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

    def _get_storage_idx(self):
        idx = self.current_size % self.size
        return idx

    def sample(self, batch_size):
        buffer_ = dict()
        buffer_["obses"] = self.obses[:self.current_size]
        buffer_["a_goals"] = self.a_goals[:self.current_size] # achieved goal before action
        buffer_["d_goals"] = self.d_goals[:self.current_size]
        buffer_["actions"] = self.actions[:self.current_size]
        buffer_["next_obses"] = buffer_["obses"][:, 1:, :]
        buffer_["a_goals_"] = buffer_["a_goals"][:, 1:, :] # achieved goal after action
        buffer_["dones"] = self.dones[:self.current_size]

        transitions = self.sample_transitions(buffer_, batch_size)

        return transitions

    def sample_transitions(self, buffer_, batch_size):
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
        future_offset = np.random.unifor(size = batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = buffer_["a_goals"][episode_idxs[her_indexes], future_t]
        transitions["d_goals"][her_indexes] = future_ag

        # recompute rewards
        reward_params = {k: transitions[k] for k in ["a_goals_", "d_goals"]}
        transitions["rewards"] = reward_function(**reward_params)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        # concatenate desired goals with observation together for network
        transitions["goal_obs_con"] = np.concatenate([transitions["obses"], transitions["d_goals"]], axis = 1)
        transitions["next_goal_obs_con"] = np.concatenate([transitions["next_obses"], transitions["d_goals"]], axis = 1)

        return transitions
