import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from IPython import embed


# object trajectory
fig = plt.figure(figsize = (15, 10))
ax1 = fig.add_subplot(131, title = 'object trajectory')

object_data = np.load('object_trajectory.npy')
object_x = object_data[:, :, 1].reshape(-1)
object_y = object_data[:, :, 0].reshape(-1)
xedges1 = np.linspace(0.4, 1.0, num = 200)
yedges1 = np.linspace(0.4, 1.0, num = 200)
heatmap1, xedges1, yedges1 = np.histogram2d(object_x, object_y, bins = (xedges1, yedges1))
heatmap1 = heatmap1.T
heatmap1 = gaussian_filter(heatmap1, sigma = 2)
ax1.imshow(heatmap1, extent = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], origin = 'lower', cmap = cm.jet)

# pusher trajectory
ax2 = fig.add_subplot(132, title = 'pusher trajectory')

pusher_data = np.load('pusher_trajectory.npy')
pusher_x = pusher_data[:, :, 1].reshape(-1)
pusher_y = pusher_data[:, :, 0].reshape(-1)
xedges2 = np.linspace(0.4, 1.0, num = 200)
yedges2 = np.linspace(0.4, 1.0, num = 200)
heatmap2, xedges2, yedges2 = np.histogram2d(pusher_x, pusher_y, bins = (xedges2, yedges2))
heatmap2 = heatmap2.T
heatmap2 = gaussian_filter(heatmap2, sigma = 2)
ax2.imshow(heatmap2, extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], origin = 'lower', cmap = cm.jet)

# goal error
ax3 = fig.add_subplot(133, title = 'goal error')
goal_data = np.load('goal.npy')
goal_x = goal_data[:, 0].reshape(-1)
goal_y = goal_data[:, 1].reshape(-1)
goal_z = goal_data[:, 2].reshape(-1)




plt.show()



