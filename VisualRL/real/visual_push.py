#!/usr/bin/env python

import torch
from torchvision import transforms
import numpy as np
import os
import cv2

import rospy
import tf
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from VisualRL.rllib.her.her import HER
from VisualRL.vae.model import VAE
from VisualRL.rllib.common.utils import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="YCB-Pushing")
parser.add_argument("--obs_size", default = 11, type = int)
parser.add_argument("--action_size", default = 2, type = int)
parser.add_argument("--feature_dims", default = 128, type = int)
parser.add_argument("--goal_size", default = 4, type = int)
parser.add_argument("--device", default="auto", type = str)
parser.add_argument("--net_class", default="Flatten", type = str)
parser.add_argument("--min_action", default = -1., type = float)
parser.add_argument("--max_action", default = 1., type = float)
parser.add_argument("--max_episode_steps", default = 50, type = int)
parser.add_argument("--train_freq", default = 1, type = int)
parser.add_argument("--learning_starts", default = 2, type = int)
parser.add_argument("--learning_rate", default = 0.0003, type = float)
parser.add_argument("--save_interval", default = 100, type = int)
parser.add_argument("--step", default = 51000, type = int)
parser.add_argument("--train_cycle", default = 1, type = int)
parser.add_argument("--gradient_steps", default = 50, type = int)
parser.add_argument("--batch_size", default = 256, type = int)
parser.add_argument("--total_episodes", default = 1e6, type = int)
parser.add_argument("--eval_freq", default = 50, type = int)
parser.add_argument("--num_eval_episode", default = 10, type = int)
parser.add_argument("--relative_goal", action = "store_false")
parser.add_argument("--mp", action = "store_true")
parser.add_argument("--seed", default = None, type = int)
parser.add_argument("--load_weights", default=1, type=int)
args = parser.parse_args()

WEIGHT_PATH = "/homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_30-14_25/her_models"
ACTION_SCALE = 0.5
STEP = 51000



class VisualPushing:
    def __init__(self, device = None):
        self.vel_pub = rospy.Publisher('/dynamic_pushing/velocity', geometry_msgs.msg.Vector3, queue_size=1)
        self.bridge = CvBridge()
        self.device = torch.device('cuda:0') if device == None else torch.device('cuda:1')
        # tf
        self.tf_listener = tf.TransformListener()
        # motion range
        self.x_range = np.array([0.43, 0.90])
        self.y_range = np.array([0.35, 1.10])
        # goal pose initialization
        self.vae = VAE(device = self.device, image_channels = 1, h_dim = 1024, z_dim = 4)
        self.vae.load("/homeL/cong/HitLyn/Visual-Pushing/results/vae/04_30-13_51/vae_model", 100, map_location=self.device)  # latent space = 4
        self.goal_latent = self.initialize_goal()


    def clip_action(self, gripper_pos, action):
        # x direction
        if gripper_pos[0] + action[0]*0.07 >= self.x_range[1] or gripper_pos[0] + action[0]*0.07 <= self.x_range[0]:
            action[0] = 0
        if gripper_pos[1] + action[1]*0.07 >= self.y_range[1] or gripper_pos[1] + action[1]*0.07 <= self.y_range[0]:
            action[1] = 0
        return action

    def initialize_goal(self):
        raw_image = rospy.wait_for_message("/some/channel/to/image", Image)
        try:
            goal_image = self.bridge.imgmsg_to_cv2(raw_image, "passthrough")
        except:
            print("Error from ROS Image to CV2")

        goal_latent = self.get_visual_latent(goal_image)
        return goal_latent

    def get_visual_latent(self, image):
        # TODO crop to square
        # image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # cv2.imshow(image)
        # cv2.waitKey(0)

        light_red = (0, 150, 0)
        dark_red = (20, 255, 255)
        mask = cv2.inRange(image, light_red, dark_red)
        tensor = transforms.ToTensor()(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, z, mu, _ = self.vae(goal_tensor)
        latent = mu[0].cpu().numpy()
        return latent


    def get_obs_dict(self):
        obs = dict()
        # image
        raw_image = rospy.wait_for_message("/some/channel/to/image", Image)
        try:
            object_image = self.bridge.imgmsg_to_cv2(raw_image, "passthrough")
        except:
            print("Error from ROS Image to CV2")

        object_latent = self.get_visual_latent(object_image)
        goal_latent = self.goal_latent
        # gripper pos
        obs["gripper_pos"] = self.get_gripper_pos()
        obs["observation"] = np.concatenate([object_latent.copy(), obs["gripper_pos"].squeeze().copy()])
        obs["achieved_goal"] = np.concatenate([object_latent.copy(), ])
        obs["desired_goal"] = np.concatenate([goal_latent.copy()])

        return obs

    def get_gripper_pos(self):
        pose_tool = None
        while (pose_tool is None):
            try:
                (trans_tool, rot_tool) = tf_listener.lookupTransform('world', 's_model_tool0', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("waiting for transform ....")
                continue
            pose_tool = np.asarray(trans_tool)[:2]

        return pose_tool

    def step(self, action):
        gripper_pos = self.get_gripper_pos()
        clipped_action = self.clip_action(gripper_pos, action)
        # send clipped action to gripper
        action_x = 0.1 * clipped_action[0]
        action_y = 0.1 * clipped_action[1]
        vel_msg = geometry_msgs.msg.Vector3()
        vel_msg.x = action_x
        vel_msg.y = action_y

        # now = rospy.Time.now()
        # target_time = now + rospy.Duration(TIME_DURATION)
        # print('start pushing step: ', step)
        # while(rospy.Time.now() < target_time):
        self.vel_pub.publish(vel_msg)

        # vel_msg.x = 0.0
        # vel_msg.y = 0.0
        # pub.publish(vel_msg)
        print('finish pushing one step')
        obs = self.get_obs_dict()
        return obs


def main():
    rospy.init_node('visual_pushing')
    env = VisualPushing()
    observation_space = args.obs_size
    action_space = args.action_size
    goal_space = args.goal_size
    feature_dims = args.feature_dims
    min_action = args.min_action
    max_action = args.max_action
    max_episode_steps = args.max_episode_steps
    train_freq = args.train_freq
    train_cycle = args.train_cycle

    device = get_device(args.device)
    # embed();exit()
    agent = HER(
        observation_space,
        action_space,
        goal_space,
        env,
        feature_dims,
        min_action,
        max_action,
        max_episode_steps,
        train_freq,
        train_cycle,
        net_class = args.net_class,
        gradient_steps=args.gradient_steps,
        save_interval = args.save_interval,
        learning_starts = args.learning_starts,
        learning_rate = args.learning_rate,
        device = device,
        relative_goal = args.relative_goal,
        batch_size = args.batch_size,
    )
    # TODO load model
    if args.load_weights:
        print("loading model ...")
        agent.load(WEIGHT_PATH, args.step, map_location='cuda:1')

    # initialization env
    obs_dict = env.get_obs_dict()
    observation = obs_dict['observation'].copy()
    achieved_goal = obs_dict['achieved_goal'].copy()
    desired_goal = obs_dict['desired_goal'].copy()

    with torch.no_grad():
        for step in range(100000):
            print(f"step: {step}")
            action = agent._select_action(observation, achieved_goal,
                                          desired_goal)  # action is squashed to [-1, 1] by tanh function
            obs_dict_new = env.step(ACTION_SCALE * action)
            observation_new = obs_dict_new['observation'].copy()
            achieved_goal_new = obs_dict_new['achieved_goal'].copy()
            # update stats
            observation[:] = observation_new.copy()
            achieved_goal[:] = achieved_goal_new.copy()





if __name__ == '__main__':
    main()



