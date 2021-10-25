import numpy as np
from wzk import new_fig
from wzk import object2numeric_array
from wzk.sql2 import get_values_sql

import rokin.Vis.robot_2d as plt2


file = '2D/SR/2dof/path.db'
path_db = get_values_sql(file=file, table='paths')
x_start = object2numeric_array(path_db.x_start.values)
x_end = object2numeric_array(path_db.x_end.values)
x_path = object2numeric_array(path_db.x_path.values)

n_samples = int(5e6)

fig, ax = plt2.new_world_fig(limits=100)
plt2.plot_x_path(x_path[0].reshape(2, 22).T)
ax.scatter(x_start[:, 0], x_start[:, 1])

x_path = x_path.reshape(-1, 22, 2)

np.linalg.norm(np.diff(x_path3, axis=0), axis=-1)
fig, ax = new_fig(title='path_length distribution')
ax.hist(path_length, bins=100, range=[0, 21])

heatmap, xedges, yedges = np.histogram2d(x_start[:, 0], x_start[:, 1], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig, ax = new_fig(title='x_start distribution')
ax.imshow(heatmap.T, extent=extent, origin='lower')

heatmap, xedges, yedges = np.histogram2d(x_end[:, 0], x_end[:, 1], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig, ax = new_fig(title='x_end distribution')
ax.imshow(heatmap.T, extent=extent, origin='lower')

heatmap, xedges, yedges = np.histogram2d(x_path[..., 0].flatten(), x_path[..., 1].flatten(), bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig, ax = new_fig(title='x_path distribution')
ax.imshow(heatmap.T, extent=extent, origin='lower')

steps = np.diff(x_path, axis=1)
steps /= np.linalg.norm(steps, axis=-1, keepdims=True)

x_steps2 = steps[np.random.choice(np.arange(n_samples), size=200, replace=False), :, :]
x_steps2 = x_steps2[np.arange(200), np.random.randint(0, 21, size=200)]

fig, ax = plt2.new_world_fig(limits=2, title="Quiver, rand(rand))")
ax.quiver(np.ones(200), np.ones(200), x_steps2[:, 0], x_steps2[:, 1], scale=2)

fig, ax = plt2.new_world_fig(limits=2, title="Quiver, rand()")
i = np.random.choice(np.arange(n_samples), size=200, replace=False)
ax.quiver(np.ones(200), np.ones(200), steps[..., 0].flatten()[i], steps[..., 1].flatten()[i], scale=2)

angles = np.arctan2(steps[..., 1].flatten(), steps[..., 0].flatten())
angles *= 180 / np.pi
fig, ax = new_fig(title='step angles distribution')
ax.hist(angles, bins=360)

fig, ax = new_fig(title='step angles distribution (2000)')
ax.hist(angles[np.random.choice(np.arange(n_samples), size=2000, replace=False)], bins=100)
