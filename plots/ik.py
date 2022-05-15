import numpy as np

from wzk.mpl import new_fig, plot_projections_2d
from wzk import sql2
from wzk.spatial.transform import frame2trans_rotvec
from wzk.spatial.difference import frame_difference

from rokin.Robots.Justin19.justin19_primitives import justin_primitives
from mopla.Optimizer.feasibility_check import feasibility_check
from mopla.Optimizer.length import len_close2q_cost
from mopla.main import ik_w_projection
from mogen.Generation import parameter

from molea.Util import comet, data, nets, inference, torch2

robot_id = 'Justin19'
file = data.get_file_ik(robot_id)
gen = parameter.init_par(robot_id=robot_id)
par, gd = gen.par, gen.gd
f_idx = 13

n = 100000
q = sql2.get_values_sql(file=file, table=data.T_PATHS, rows=np.arange(n), columns=[data.C_Q_F32])
f = par.robot.get_frames(q)[:, f_idx, :, :]


x, r = frame2trans_rotvec(f)
fi, ax = new_fig(aspect=1, n_cols=3, n_rows=2)
plot_projections_2d(ax=ax[0],
                    x=x, dim_labels='xyz', aspect=1, color='blue', marker='o', markersize=1, alpha=0.01,
                    limits=par.world.limits)
plot_projections_2d(ax=ax[1],
                    x=r, dim_labels='xyz', aspect=1, color='red', marker='o', markersize=1, alpha=0.01)