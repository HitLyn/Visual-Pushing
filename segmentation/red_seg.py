import cv2
import matplotlib.pyplot as plt
from IPython import embed


test = cv2.imread('/homeL/cong/HitLyn/Visual-Pushing/images/push_sim_red/0_2')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
# embed()
light_red = (0, 150, 0)
bright_red = (20, 255, 255)
mask = cv2.inRange(test, light_red, bright_red)
result = cv2.bitwise_and(test, test, mask=mask)
# h,s,v = cv2.split(test)
# fig = plt.figure()
# axis = fig.add_subplot(1,1,1,projection='3d')
# axis.scatter(h.flatten(), s.flatten(), v.flatten(),  marker=".")
plt.imshow(mask)
plt.show()
