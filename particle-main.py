import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

planck_data_file = '/home/mmtmn/sphere-cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits'

cmb_map = hp.read_map(planck_data_file, field=0)
NSIDE = 1024

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_axis_off()

threshold = np.mean(cmb_map) - np.std(cmb_map)  # Adjusted threshold
mask = hp.ma(cmb_map) < threshold
theta, phi = hp.pix2ang(NSIDE, mask.nonzero()[0])
black_spots = np.array([theta, phi]).T

num_particles = len(black_spots)
positions = black_spots
velocities = np.zeros((num_particles, 2))

def update_particles(dt):
    global positions, velocities
    G = 1.0
    accel = np.zeros_like(velocities)

    for i in range(num_particles):
        diffs = positions - positions[i]
        dist_squared = np.sum(diffs**2, axis=1)
        dist_squared[i] = np.inf
        nearest_neighbor_idx = np.argmin(dist_squared)
        force_direction = diffs[nearest_neighbor_idx]
        force_magnitude = G / dist_squared[nearest_neighbor_idx]
        accel[i] = force_magnitude * force_direction

    velocities += accel * dt
    positions += velocities * dt

    x, y, z = hp.ang2vec(positions[:,0], positions[:,1]).T
    ax.clear()
    ax.scatter(x, y, z, c='white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    fig.canvas.draw_idle()

def update_view(event):
    if event.key == 'left' or event.key == 'right' or event.key == 'up' or event.key == 'down':
        update_particles(0.01)

fig.canvas.mpl_connect('key_press_event', update_view)

x, y, z = hp.ang2vec(positions[:,0], positions[:,1]).T
ax.scatter(x, y, z, c='white')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()
