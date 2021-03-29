import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from IPython import embed
import argparse

from VisualRL.rllib.her.her import HER
from VisualRL.rllib.common.utils import get_device, set_seed_everywhere

from robogym.envs.push.push_env import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="YCB-Pushing")
parser.add_argument("--obs_size", default = 15)
parser.add_argument("--action_size", default = 2)
parser.add_argument("--feature_dims", default = 128)
parser.add_argument("--goal_size", default = 6)
parser.add_argument("--device", default="auto")
parser.add_argument("--min_action", default = -0.5)
parser.add_argument("--max_action", default = 0.5)
parser.add_argument("--max_episode_steps", default = 40)
parser.add_argument("--train_freq", default = 10)
parser.add_argument("--train_cycle", default = 2)
parser.add_argument("--gradient_steps", default = 5)
parser.add_argument("--batch_size", default = 128)
parser.add_argument("--total_episodes", default = 1e6)
parser.add_argument("--eval_freq", default = 100)
parser.add_argument("--num_eval_episode", default = 10)
args = parser.parse_args()

def main():
    observation_space = args.obs_size
    action_space = args.actions_size
    goal_space = args.goal_size
    feature_dims = args.feature_dims
    min_action = args.min_action
    max_action = args.max_action
    max_episode_steps = args.max_episode_steps
    train_freq = args.train_freq
    train_cycle = args.train_cycle
    gradient_steps = args.gradient_steps
    batch_size = args.batch_size
    total_episodes = args.total_episodes
    eval_freq = args.eval_freq
    num_eval_episode = args.num_eval_episode

    device = get_device(args.device)
    # save dir
    save_dir = os.path.join(os.environ["VISUAL_PUSHING_HOME"], "log_files/her")
    train_name = time.strftime("%m_%d-%H_%M", time.gmtime())
    os.makedirs(os.path.join(save_dir, train_name), exist_ok=True)
    save_path = os.path.join(save_dir, train_name)
    model_path = os.path.join(save_path, 'her_models')
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    # agent and env
    env = make_env()
    agent = HER(
        observation_space,
        action_space,
        goal_space,
        feature_dims,
        min_action,
        max_action,
        max_episode_steps,
        train_freq,
        train_cycle,
        device = device,
    )
    # train
    agent.learn(env, total_episodes, eval_freq, num_eval_episode, writer, model_path)


if __name__ == '__main__':
    main()
