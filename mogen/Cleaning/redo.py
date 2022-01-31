import numpy as np
from wzk.sql2 import set_values_sql

from mopla.main import objective_feasibility
from mopla.Optimizer.gradient_descent import gd_chomp

from mogen.Loading.load import get_sample
from multiprocessing import Lock


def update_i(file, par, gd, i):
    i_w, i_s, q, img = get_sample(file=file, i_s=i, img_shape=par.world.shape)
    assert len(np.unique(i_w)) == 1
    q = q.reshape((-1, par.robot.n_dof))
    par.q_start = q[0]
    par.q_end = q[-1]
    q = q[np.newaxis, 1:-1, :]

    # q = gd_chomp(q0=q, par=par, gd=gd)
    o, f = objective_feasibility(q=q, img=img, par=par)

    lock = Lock()
    set_values_sql(file=file, table='paths', values=(q, o, f),
                   columns=['q_f32', 'objective_f32', 'feasible_b'], rows=i, lock=lock)


def main():
    par = None
    gd = None

    robot_id = 'SingleSphere02'
    file = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot_id}_hard2.db'

    update_i(file=file, par=par, gd=gd, i=0)