import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

IMAGE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/red_original"
SAVE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/masks"

for i in range(10000):
    file_name = os.path.join(IMAGE_PATH, "{:0>5d}.png".format(i))
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    light_red = (0, 150, 0)
    bright_red = (20, 255, 255)
    mask = cv2.inRange(image, light_red, bright_red)
    crop_mask = mask[200:, 100:400]
    crop_mask = cv2.resize(crop_mask, (64, 64))
    save_name = os.path.join(SAVE_PATH, "{:0>5d}.png".format(i))
    cv2.imwrite(save_name, crop_mask)



