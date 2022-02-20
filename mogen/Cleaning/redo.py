import numpy as np


# from multiprocessing import Lock
from wzk import sql2, trajectory
from wzk import safe_unify, tictoc
from wzk.mpl import new_fig

# from mopla.main import objective_feasibility
from mopla.Optimizer import feasibility_check, objectives
from mopla.Optimizer.gradient_descent import gd_chomp

from mopla.Parameter.parameter import initialize_oc

from mogen.Generation.load import get_samples, get_paths, get_worlds
from mogen.Generation.parameter import init_par

# TODO function which calculates objectives + feasibilities simultaneous


def get_samples_for_world(file, par, i=None, i_w=None):
    if i_w is not None:
        i_w, i_w_all = i_w
        i = np.nonzero(i_w_all == i_w)[0]

    i_w, i_s, q, o, f = get_paths(file, i)
    q = q.reshape((len(i), par.n_wp, par.robot.n_dof))
    i_w = safe_unify(i_w)
    img = get_worlds(file=file, i_w=i_w, img_shape=par.world.shape)

    q_start, q_end = q[..., 0, :], q[..., -1, :]
    par.q_start, par.q_end = q_start, q_end
    initialize_oc(par=par, obstacle_img=img)
    return i, q, img


def update_objective(file, par, i=None, i_w=None):
    i, q, img = get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    o = objectives.chomp_cost(q=q, par=par)
    sql2.set_values_sql(file=file, table='paths', columns=['objective_f32'], rows=i, values=(o.tolist(),))


def refine_chomp(file, par, gd,
                 q0_fun=None,
                 i=None, i_w=None):
    i, q0, img = get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    if q0_fun is None:
        q00 = trajectory.full2inner(q0)
    else:
        q0 = q0_fun(i=i)
        q00 = trajectory.full2inner(q0)

    par.weighting.length = 0
    par.weighting.collision = 1

    with tictoc() as _:
        q00 = q00[:10]
        par.q_start = par.q_start[:10]
        par.q_end = par.q_end[:10]
        q, o = gd_chomp(q0=q00, par=par, gd=gd)

    q = trajectory.inner2full(inner=q, start=par.q_start, end=par.q_end)

    f0 = feasibility_check(q=q0, par=par)
    f = feasibility_check(q=q, par=par)
    o0 = objectives.chomp_cost(q=q0, par=par)

    # print(o.mean())
    ff0 = np.round((f0 == 1).mean(), 3)
    ff = np.round((f == 1).mean(), 3)

    oo = np.round(o[f == 1].mean(), 4)
    oo0 = np.round(o0[f == 1].mean(), 4)
    try:
        j = np.nonzero(f != 1)[0][0]
    except IndexError:
        j = 0
    fig, ax = new_fig()
    for i_d in range(par.robot.n_dof):
        ax.plot(q0[j, :, i_d], color='b', marker='o')
        ax.plot(q[j, :, i_d], color='r', marker='o')

    print(f"n: {len(i)} | f0: {ff0}, 'f': {ff} | o0: {oo0}, o:{oo}")


def adjust_path(file, par, i=None, i_w=None):
    i, q0, img = get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    q = trajectory.get_path_adjusted(q0, m=50, is_periodic=par.robot.is_periodic)
    f = feasibility_check(q=q, par=par)
    o0 = objectives.chomp_cost(q=q0, par=par)
    o = objectives.chomp_cost(q=q, par=par)
    o_improvement = (o - o0) / o0
    o_improvement = o_improvement[f > 0].sum()
    q = q[f > 0]
    i = i[f > 0]
    print(f"updated {f.sum()}/{f.size} samples")
    print(f"objective improvement {o_improvement}")

    sql2.set_values_sql(file=file, table='paths', values=(q,),
                        columns=['q_f32'], rows=i, lock=None)


def main():
    robot_id = 'StaticArm04'
    robot_id = 'Justin19'
    file = f"/Users/jote/Documents/DLR/Data/mogen/{robot_id}/{robot_id}.db"

    gen = init_par(robot_id)
    par, gd = gen.par, gen.gd
    i_w_all = sql2.get_values_sql(file=file, table='paths', columns='world_i32', rows=-1, values_only=True)
    for i_w in np.arange(10):
        # adjust_path(file=file, par=par, i=None, i_w=(i_w, i_w_all))
        with tictoc(text=f'World {i_w}') as _:
            refine_chomp(file=file, par=par, gd=gd,
                         q0_fun=None, i_w=(i_w, i_w_all))


if __name__ == '__main__':
    main()
