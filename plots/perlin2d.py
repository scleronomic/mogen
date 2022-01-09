import numpy as np

from wzk.mpl.styles import set_style, set_borders
from wzk.mpl import new_fig, save_fig, imshow, turn_ticklabels_off, close_all

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from mopla.World.random_obstacles import create_perlin_image
set_style('ieee')

set_borders(no_borders=True, no_whitespace=True)


for i in range(20):
    fig, ax = new_fig(aspect=1, width=5, height=5)
    turn_ticklabels_off(ax=ax)

    img = create_perlin_image(shape=(64, 64), n=1, threshold=0.4)
    h = imshow(ax=ax, img=img, mask=~img, cmap='black')
    save_fig(file=f'perlin{i}', formats='pdf', fig=fig, bbox=None)
    close_all()


def update(i):
    if i < 100:
        t = (100-i) / 100
    else:
        t = (i-100) / 100

    img = create_perlin_image(shape=(64, 64), n=1, threshold=t, seed=0)
    if 99 < i < 101:
        print(i)
        img[:] = 1

    imshow(h=h, img=img, mask=~img, cmap='black')


# for i in range(200):
#     update(i)
#     plt.pause(0.1)


ani = FuncAnimation(fig, func=update, frames=200, repeat=False)

ani.save('perlin2d_threshold.gif', writer='ffmpeg', fps=30, metadata=dict(artist='Johannes Tenhumberg'))
