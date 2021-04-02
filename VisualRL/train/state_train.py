import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from IPython import embed
import argparse
import os
import time
from IPython import embed

from VisualRL.rllib.her.her import HER
from VisualRL.rllib.common.utils import get_device, set_seed_everywhere

from robogym.envs.push.push_env import make_env

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="YCB-Pushing")
parser.add_argument("--obs_size", default = 15, type = int)
parser.add_argument("--action_size", default = 2, type = int)
parser.add_argument("--feature_dims", default = 128, type = int)
parser.add_argument("--goal_size", default = 6, type = int)
parser.add_argument("--device", default="auto", type = str)
parser.add_argument("--net_class", default="Flatten", type = str)
parser.add_argument("--min_action", default = -0.5, type = float)
parser.add_argument("--max_action", default = 0.5, type = float)
parser.add_argument("--max_episode_steps", default = 40, type = int)
parser.add_argument("--train_freq", default = 10, type = int)
parser.add_argument("--learning_starts", default = 20, type = int)
parser.add_argument("--save_interval", default = 100, type = int)
parser.add_argument("--train_cycle", default = 2, type = int)
parser.add_argument("--gradient_steps", default = 10, type = int)
parser.add_argument("--batch_size", default = 128, type = int)
parser.add_argument("--total_episodes", default = 1e6, type = int)
parser.add_argument("--eval_freq", default = 100, type = int)
parser.add_argument("--num_eval_episode", default = 10, type = int)
parser.add_argument("--relative_goal", default = True, type = bool)
args = parser.parse_args()

def main():
    observation_space = args.obs_size
    action_space = args.action_size
    goal_space = args.goal_size
    feature_dims = args.feature_dims
    min_action = args.min_action
    max_action = args.max_action
    max_episode_steps = args.max_episode_steps
    train_freq = args.train_freq
    # embed();exit()
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
        net_class = args.net_class,
        save_interval = args.save_interval,
        learning_starts = args.learning_starts,
        device = device,
        relative_goal = args.relative_goal,
    )
    # train
    agent.learn(env, total_episodes, eval_freq, num_eval_episode, writer, model_path, mp = args.mp)


if __name__ == '__main__':
    main()
