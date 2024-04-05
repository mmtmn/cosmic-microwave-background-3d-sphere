import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

planck_data_file = '/home/mmtmn/sphere-cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits'
cmb_map = hp.read_map(planck_data_file, field=0)
NSIDE = 1024
threshold = np.mean(cmb_map) - np.std(cmb_map)
mask = hp.ma(cmb_map) < threshold
theta, phi = hp.pix2ang(NSIDE, mask.nonzero()[0])
black_spots = np.array([theta, phi]).T
num_particles = len(black_spots)
positions = black_spots
velocities = np.zeros((num_particles, 2))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_axis_off()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

def init():
    x, y, z = hp.ang2vec(positions[:,0], positions[:,1]).T
    scatter = ax.scatter(x, y, z, c='white')
    return scatter,

def update(frame):
    global positions, velocities
    dt = 0.01
    G = 1.0
    for i in range(num_particles):
        diffs = positions - positions[i]
        dist_squared = np.sum(diffs**2, axis=1)
        dist_squared[i] = np.inf
        force_direction = np.nan_to_num(diffs / dist_squared[:, np.newaxis])
        force_magnitude = G / dist_squared
        accel = force_magnitude[:, np.newaxis] * force_direction
        velocities += accel * dt
    positions += velocities * dt
    positions[:,0] = np.clip(positions[:,0], 0, np.pi)
    positions[:,1] = np.mod(positions[:,1], 2*np.pi)
    positions = np.nan_to_num(positions)
    x, y, z = hp.ang2vec(positions[:,0], positions[:,1]).T
    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    scatter = ax.scatter(x, y, z, c='white')
    return scatter,

ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, blit=False)
plt.show()
