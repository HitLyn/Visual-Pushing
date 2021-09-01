import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D
PIXEL_SIZE = 200
# object trajectory
fig = plt.figure(figsize = (20, 5))
fig.tight_layout()
ax1 = fig.add_subplot(141,)

object_data = np.load('object_trajectory.npy')
# object_x = object_data[:, :, 1].reshape(-1)
# object_y = object_data[:, :, 0].reshape(-1)
object_x = object_data[:80, :, 1].reshape(-1)
object_y = object_data[:80, :, 0].reshape(-1)
# embed()
xedges1 = np.linspace(0.5, 1.0, num = PIXEL_SIZE)
yedges1 = np.linspace(0.4, 0.9, num = PIXEL_SIZE)
heatmap1, xedges1, yedges1 = np.histogram2d(object_x, object_y, bins = (xedges1, yedges1))
# embed()
heatmap1 /= heatmap1.sum()
heatmap1 = heatmap1.T
heatmap1 = gaussian_filter(heatmap1, sigma = 2)
ax1.imshow(heatmap1, extent = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], origin = 'lower', cmap = cm.jet)
ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax1.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_title('Object in Sim', fontdict={'fontsize': 20, 'fontweight': "medium"})
# ax1.plot(object_x, object_y, 'o')


# pusher trajectory
ax2 = fig.add_subplot(142,)

pusher_data = np.load('pusher_trajectory.npy')
pusher_x = pusher_data[:80, :, 1].reshape(-1)
pusher_y = pusher_data[:80, :, 0].reshape(-1)
# pusher_x = pusher_data[:, :, 1].reshape(-1)
# pusher_y = pusher_data[:, :, 0].reshape(-1)
xedges2 = np.linspace(0.5, 1.0, num = PIXEL_SIZE)
yedges2 = np.linspace(0.4, 0.9, num = PIXEL_SIZE)
heatmap2, xedges2, yedges2 = np.histogram2d(pusher_x, pusher_y, bins = (xedges2, yedges2))
heatmap2 /= heatmap2.sum()

heatmap2 = heatmap2.T
heatmap2 = gaussian_filter(heatmap2, sigma = 2)
pusher = ax2.imshow(heatmap2, extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], origin = 'lower', cmap = cm.jet)
ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax2.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_title('Pusher in Sim', fontdict={'fontsize': 20, 'fontweight': "medium"})

ax3 = fig.add_subplot(143,)
object_data_real = np.load('object_trajectory_.npy')
# object_x_real = object_data_real[:200, :200, 1].reshape(-1)
# object_y_real = object_data_real[:200, :200, 0].reshape(-1)
object_x_real = object_data_real[:, :12, 1].reshape(-1)
object_y_real = object_data_real[:, :12, 0].reshape(-1)
xedges3 = np.linspace(0.5, 1.0, num = PIXEL_SIZE)
yedges3 = np.linspace(0.4, 0.9, num = PIXEL_SIZE)
heatmap3, xedges3, yedges3 = np.histogram2d(object_x_real, object_y_real, bins = (xedges3, yedges3))
heatmap3 /= heatmap3.sum()

heatmap3 = heatmap3.T
heatmap3 = gaussian_filter(heatmap3, sigma = 2)
ax3.imshow(heatmap3, extent = [xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]], origin = 'lower', cmap = cm.jet)
ax3.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax3.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax3.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax3.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax3.tick_params(axis='both', which='major', labelsize=18)
ax3.set_title('Object in Real', fontdict={'fontsize': 20, 'fontweight': "medium"})
# ax1.plot(object_x, object_y, 'o')


# pusher trajectory
ax4 = fig.add_subplot(144,)

pusher_data_real = np.load('pusher_trajectory_.npy')
# pusher_x_real = pusher_data[:200, :200, 1].reshape(-1)
# pusher_y_real = pusher_data[:200, :200, 0].reshape(-1)
pusher_x_real = pusher_data_real[:, :12, 1].reshape(-1)
pusher_y_real = pusher_data_real[:, :12, 0].reshape(-1)
xedges4 = np.linspace(0.5, 1.0, num = PIXEL_SIZE)
yedges4 = np.linspace(0.4, 0.9, num = PIXEL_SIZE)
heatmap4, xedges4, yedges4 = np.histogram2d(pusher_x_real, pusher_y_real, bins = (xedges4, yedges4))
heatmap4 /= heatmap4.sum()
heatmap4 = heatmap4.T
heatmap4 = gaussian_filter(heatmap4, sigma = 2)
im = ax4.imshow(heatmap4, extent = [xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]], origin = 'lower', cmap = cm.jet)
ax4.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax4.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax4.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax4.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax4.tick_params(axis='both', which='major', labelsize=18)
ax4.set_title('Pusher in Real', fontdict={'fontsize': 20, 'fontweight': "medium"})
# embed()

cb = fig.colorbar(im)

# fig.tight_layout()
plt.savefig('trajectory_sim_real.pdf')
plt.show()

all_data = np.hstack([heatmap1, heatmap2, heatmap3, heatmap4]) *100
plt.imshow(all_data,   origin = 'lower', cmap = cm.jet)
plt.colorbar()
plt.savefig('together.pdf')
plt.show()
