import cv2
import matplotlib.pyplot as plt
from IPython import embed
import os

IMAGE_PATH = os.path.join(os.environ["VISUAL_PUSHING_HOME"], "images/original/0001.png")

test = cv2.imread(IMAGE_PATH)
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
# embed()
light_red = (0, 150, 0)
bright_red = (20, 255, 255)
mask = cv2.inRange(test, light_red, bright_red)
result = cv2.bitwise_and(test, test, mask=mask)
cv2.imwrite("0002.png", mask)

plt.imshow(mask)
plt.show()
