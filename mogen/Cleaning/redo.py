import numpy as np


# from multiprocessing import Lock
from wzk import sql2, trajectory
from wzk import safe_unify
# from mopla.main import objective_feasibility
from mopla.Optimizer.gradient_descent import gd_chomp
from mopla.Optimizer import feasibility_check
from mopla.Parameter import parameter

from mogen.Generation.load import get_samples, get_paths, get_worlds
from mogen.Generation.parameter import init_par


def get_samples_for_world(file, par, i=None, i_w=None):
    if i_w is not None:
        i_w, i_w_all = i_w
        i = np.nonzero(i_w_all == i_w)[0]

    i_w, i_s, q, o, f = get_paths(file, i)
    q = q.reshape((len(i), -1, par.robot.n_dof))
    i_w = safe_unify(i_w)
    img = get_worlds(file=file, i_w=i_w, img_shape=par.world.shape)

    parameter.initialize_oc(par=par, obstacle_img=img)
    return i, q, img


def update_objective(file, i, par):
    i_w, i_s, q, img = get_samples(file=file, i=i, img_shape=par.world.shape)
    assert len(np.unique(i_w)) == 1
    q = q.reshape((-1, par.robot.n_dof))
    par.q_start = q[0]
    par.q_end = q[-1]
    q = q[np.newaxis, 1:-1, :]

    # q = gd_chomp(q0=q, par=par, gd=gd)
    o, f = objective_feasibility(q=q, img=img, par=par)

    lock = Lock()
    sql2.set_values_sql(file=file, table='paths', values=(q, o.tolist(), f.tolist()),
                        columns=['q_f32', 'objective_f32', 'feasible_b'], rows=i, lock=lock)


def adjust_path(file, par, i=None, i_w=None):
    i, q, img = get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    q = trajectory.get_path_adjusted(q, m=50, is_periodic=par.robot.infinity_joints)
    f = feasibility_check(q=q, par=par)

    q = q[f > 0]
    i = i[f > 0]
    print(f"updated {f.sum()}/{f.size} samples")
    sql2.set_values_sql(file=file, table='paths', values=(q,),
                        columns=['q_f32'], rows=i, lock=None)


def main():
    robot_id = 'StaticArm04'
    file = f"/Users/jote/Documents/DLR/Data/mogen/{robot_id}/{robot_id}.db"

    par = init_par(robot_id).par

    i_w_all = sql2.get_values_sql(file=file, table='paths', columns='world_i32', rows=-1, values_only=True)
    for i_w in np.arange(10000):
        print(i_w)
        adjust_path(file=file, par=par, i=None, i_w=(i_w, i_w_all))


if __name__ == '__main__':
    main()
