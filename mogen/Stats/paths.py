import numpy as np

from wzk import trajectory
from wzk import new_fig


from mogen.Generation.load import sql2, get_samples, get_worlds, get_paths
from mogen.Generation.parameter import init_par
from mogen.Vis.main import plot_path_gif
from mogen.Cleaning.clean import update_cast_joint_errors


from mopla.Optimizer import feasibility_check


robot_id = 'StaticArm04'
par = init_par(robot_id=robot_id).par
file = f'/Users/jote/Documents/DLR/Data/mogen/{robot_id}/{robot_id}_hard2.db'

print(sql2.get_columns(file=file, table='paths'))
# sql2.alter_table(file, table='paths', columns=['world_i32', 'sample_i32', 'q_f32', 'objective_f32', 'feasible_b'],
#                  dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_REAL, sql2.TYPE_INTEGER])
n_wp = 20
n_dof = 4


i = np.arange(100)
i_w, i_s, q, o, f = get_paths(file, i=i)

q = q.reshape(-1, n_wp, n_dof)
q = update_cast_joint_errors(q=q, limits=par.robot.limits)

img = get_worlds(file=file, i_w=i_w, img_shape=(64, 64))


q_bee = trajectory.get_q_bee(q=q, n_wp=n_wp)
q_length = np.linalg.norm(np.diff(q, axis=-2), axis=-1).sum(axis=-1)
q_bee_length = np.linalg.norm(np.diff(q_bee, axis=-2), axis=-1).sum(axis=-1)
q_max_step = np.linalg.norm(np.diff(q, axis=-2), axis=-1).max(axis=-1)
q_bee_step = np.linalg.norm(np.diff(q_bee, axis=-2), axis=-1).max(axis=-1)

# fig, ax = new_fig()
# ax.hist(q_length / q_bee_length, bins=100)

# i = np.argsort(o)
# i = np.argsort(q_length/q_bee_length)
i = np.argsort(q_max_step/q_bee_step)
i = i[-10000:][::-1]

qq = q[i]
qq2 = trajectory.get_path_adjusted(qq, m=100)
ff = feasibility_check(q=qq2, par=par)
print(ff.mean())

qq2_length = np.linalg.norm(np.diff(qq2, axis=-2), axis=-1).sum(axis=-1)
dd = q_length[i] / qq2_length

fig, ax = new_fig()
ax.hist(dd, bins=100, alpha=0.5)

for ii in i[-100:][::-1]:
    plot_path_gif(file=file, robot_id=robot_id, i=ii)
