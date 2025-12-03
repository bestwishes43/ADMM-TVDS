import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .common import check_and_trans

def implay(images, vmax=None, vmin=None):
    images = check_and_trans(images)
    images = np.transpose(images, [2, 0, 1])

    fig, ax = plt.subplots()
    if vmax is None:
        vmax = np.max(images)
    if vmin is None:
        vmin = np.min(images)
    image_ax = ax.imshow(np.zeros_like(images[0]), animated=True, cmap='gray',vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    def update(idx):
        image_ax.set_data(images[idx])
        return plt.gca(),
    # 创建动画
    ani = FuncAnimation(fig, update, frames=range(len(images)), interval=200, blit=True)
    plt.show()

def imshow(images):
    for image in images:
        image = check_and_trans(image)
        _, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_axis_off()
    plt.show()