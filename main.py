import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Download the Planck CMB data (replace 'path/to/data' with the actual path)
planck_data_file = '/home/mmtmn/sphere-cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits'

# Read the CMB temperature map from the FITS file
cmb_map = hp.read_map(planck_data_file, field=0)

# Set the resolution of the map (NSIDE parameter)
NSIDE = 1024

# Create a figure and axis for the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initial rotation angles
rot_angles = [0, 90, 0]

def update_view(event):
    if event.key == 'left':
        rot_angles[0] += 10
    elif event.key == 'right':
        rot_angles[0] -= 10
    elif event.key == 'up':
        rot_angles[1] = min(rot_angles[1] + 10, 90)
    elif event.key == 'down':
        rot_angles[1] = max(rot_angles[1] - 10, -90)
    
    ax.clear()
    hp.orthview(cmb_map, rot=rot_angles, norm='hist', min=-1e-5, max=1e-5, title='Cosmic Microwave Background', unit='mK', fig=fig, half_sky=True)
    fig.canvas.draw_idle()

# Connect the keyboard event to the update_view function
fig.canvas.mpl_connect('key_press_event', update_view)

# Plot the initial view
hp.orthview(cmb_map, rot=rot_angles, norm='hist', min=-1e-5, max=1e-5, title='Cosmic Microwave Background', unit='mK', fig=fig, half_sky=True)

plt.tight_layout()
plt.show()
