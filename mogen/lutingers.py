import numpy as np

from wzk import sql2, spatial
from wzk.mpl import plot_projections_2d

from rokin.Robots import Justin19
from rokin.Vis import robot_3d
from mogen.Generation.Data.data import T_PATHS

# TODO redo left hands
file = '/Users/jote/Documents/DLR/Data/mogen/Automatica2022/table_left_lut.db'
f_idx = 22

q, o = sql2.get_values_sql(file=file, table=T_PATHS(), rows=-1, columns=[T_PATHS.C_Q_F(), T_PATHS.C_OBJECTIVE_F()])

robot = Justin19()
f = robot.get_frames(q)[:, f_idx, :, :]

x, rv = spatial.frame2trans_rotvec(f=f)
xrv = np.concatenate((x, rv), axis=-1)

# plot_projections_2d(x=xrv, ls='', marker='o', markersize=1, alpha=0.1, color='k')
n = len(q)

i = np.random.choice(np.arange(n), 100, replace=False)
robot_3d.animate_path(q=q[i], robot=robot, kwargs_frames=dict(f_fix=f[i], f_idx_robot=[f_idx], scale=0.05))

