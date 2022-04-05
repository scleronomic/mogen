from multiprocessing import Lock

import numpy as np
import fire

from wzk import sql2, trajectory
from wzk import tictoc
from wzk.mpl import new_fig, remove_duplicate_labels
from wzk.ray2 import ray, ray_init

from rokin.Vis.robot_2d import plot_img_patch_w_outlines
from mopla.Optimizer import feasibility_check, objectives
from mopla.Optimizer.gradient_descent import gd_chomp
from mopla.Parameter.parameter import initialize_oc

from mogen.Generation import data, parameter


def update_objective(file, par, i=None, i_w=None):
    i, q, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    o = objectives.chomp_cost(q=q, par=par)
    sql2.set_values_sql(file=file, table='paths', columns=[data.C_OBJECTIVE_F], rows=i, values=(o.tolist(),))


def plot_redo(q, q0, f, j,
              par):
    # TODO If f is False check, f0 again, than f0 with more substeps, if all is False, than the label was not
    #   correct to begin with
    #

    if par.robot.id == 'SingleSphere02':
        fig, ax = new_fig(aspect=1, title=f'Feasibility: {f[j]}')
        plot_img_patch_w_outlines(ax=ax, img=par.oc.img, limits=par.world.limits)
        ax.plot(*q0[j, :, :].T, color='b', marker='o', label='old', markersize=15)
        ax.plot(*q[j, :, :].T, color='r', marker='o', label='new', markersize=10)
        ax.legend()
        remove_duplicate_labels(ax=ax)


def recalculate_objective(file, par,
                          i=None, i_w=None,
                          lock=None):
    i, q, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    o = objectives.o_len.len_q_cost(q, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)
    sql2.set_values_sql(file=file, table='paths', values=(o,), columns=[data.C_OBJECTIVE_F], rows=i, lock=lock)


def refine_chomp(file, par, gd,
                 q0_fun=None,
                 i=None, i_w=None,
                 verbose=0,
                 lock=None):
    i, q0, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    if q0_fun is not None:
        q0 = q0_fun(i=i)

    q00 = trajectory.get_path_adjusted(q0, m=50, is_periodic=par.robot.is_periodic)
    q00 = trajectory.full2inner(q00)
    q, o = gd_chomp(q0=q00, par=par, gd=gd)

    q = trajectory.inner2full(inner=q, start=par.q_start, end=par.q_end)

    f = feasibility_check(q=q, par=par)
    f0 = feasibility_check(q=q0, par=par)

    # TODO use the correct metric, if i want to do the same thing for IK
    o0 = objectives.o_len.len_q_cost(q0, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)
    o = objectives.o_len.len_q_cost(q, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)

    def set_values(b, new=True):
        if b.sum() > 0:
            if new:
                sql2.set_values_sql(file=file, table=data.T_PATHS, lock=lock, rows=i[b], values=(q[b], o[b], f[b]),
                                    columns=[data.C_Q_F32, data.C_OBJECTIVE_F, data.C_FEASIBLE_I])
            else:
                sql2.set_values_sql(file=file, table=data.T_PATHS, lock=lock, rows=i[b], values=(q0[b], o0[b], f0[b]),
                                    columns=[data.C_Q_F32, data.C_OBJECTIVE_F, data.C_FEASIBLE_I])

    b_fb = np.logical_and(f == +1, o < o0)  # feasible and better
    b_nfb = np.logical_and(np.logical_and(f0 == -1, f == -1), o < o0)  # not feasible and better
    b_rest = ~np.logical_or(b_fb, b_nfb)  # rest
    assert b_fb.sum() + b_nfb.sum() + b_rest.sum() == len(i)

    if verbose > 0:
        ff0 = np.round((f0 == 1).mean(), 3)
        ff = np.round((f == 1).mean(), 3)

        oo = np.round(o[f == 1].mean(), 4)
        oo0 = np.round(o0.astype(float)[f == 1].mean(), 4)

        n = len(i)
        n_fb = b_fb.sum()
        n_nf = (~np.logical_or(f0 == +1, f == +1)).sum()
        print(f"# samples: {n}, # improvements {n_fb}  # infeasible {n_nf} | "
              f"f0: {ff0}, 'f': {ff} | "
              f"o0: {oo0}, o:{oo}")

        if verbose > 1:
            j = np.argmax((o0 - o) / 1)
            print(f'Largest Improvement {j}:', o0[j], o[j], f[j])

            if verbose > 10:
                plot_redo(q=q, q0=q0, f=f, j=j, par=par)

    set_values(b_fb)
    set_values(b_nfb)
    set_values(b_rest, new=False)

    sql2.set_values_sql(file=file, table=data.T_PATHS, lock=lock, rows=i, values=(np.arange(len(i)),),
                        columns=[data.C_SAMPLE_I])


def refine_adjust_steps(file, par, i=None, i_w=None):
    i, q0, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

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

    sql2.set_values_sql(file=file, table=data.T_PATHS, values=(q,),
                        columns=[data.C_Q_F32], rows=i, lock=None)


def main(robot_id):

    file = data.get_file(robot_id=robot_id)

    gen = parameter.init_par(robot_id)
    par, gd = gen.par, gen.gd

    par.oc.n_substeps_check += 2
    gd.n_steps = 100

    i_w_all = sql2.get_values_sql(file=file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    for i_w in np.arange(0, 10000):
        # print('World', i_w)
        with tictoc(text=f'World {i_w}') as _:
            refine_chomp(file=file, par=par, gd=gd,
                         q0_fun=None, i_w=(i_w, i_w_all), verbose=2)
            # adjust_path(file=file, par=par, i=None, i_w=(i_w, i_w_all))


def check_worlds_img_cmp():
    robot_id = 'SingleSphere02'

    file = f"/Users/jote/Documents/DLR/Data/mogen/{robot_id}/{robot_id}.db"

    img_cmp = sql2.get_values_sql(file=file, table=data.T_WORLDS, rows=-1,
                                  columns=data.C_IMG_CMP, values_only=True)
    img_shape = (64, 64)

    from wzk.image import compressed2img, zlib
    # print(img_cmp[1001])
    for i in range(2000):
        try:
            img = compressed2img(img_cmp=img_cmp[i], shape=img_shape, dtype=bool)
        except zlib.error:
            print(i)

    i_w = sql2.get_values_sql(file=file, table=data.T_PATHS, rows=-1, columns=data.C_WORLD_I, values_only=True)
    u, c = np.unique(i_w, return_counts=True)


def main_refine_chomp(file):
    lock = Lock()

    robot_id = parameter.get_robot_str(file)

    i_w_all = sql2.get_values_sql(file=file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    iw_list = np.unique(i_w_all)

    @ray.remote
    def refine_ray(i):
        gen = parameter.init_par(robot_id)
        par, gd = gen.par, gen.gd

        par.oc.n_substeps_check += 2
        gd.n_steps = 100

        with tictoc(text=f'World {i_w}') as _:
            refine_chomp(file=file, par=par, gd=gd,
                         q0_fun=None, i_w=(i, i_w_all), verbose=1, lock=lock)

        return 1

    futures = []
    for i_w in iw_list:
        futures.append(refine_ray.remote(i_w))

    res = ray.get(futures)

    print(f"{np.sum(res)} / {np.size(res)}")


def main_recalculate_objective(file):

    robot_id = parameter.get_robot_str(file)
    i_w_all = sql2.get_values_sql(file=file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    iw_list = np.unique(i_w_all)

    gen = parameter.init_par(robot_id)
    par, gd = gen.par, gen.gd

    for i_w in iw_list:
        with tictoc(text=f'World {i_w}') as _:
            recalculate_objective(file=file, par=par, i_w=(i_w, i_w_all), )


if __name__ == '__main__':
    # fire.Fire({
    #     'objective': main_recalculate_objective,
    #     'chomp': main_refine_chomp
    # })

    # main_recalculate_objective(robot_id='SingleSphere02')

    # ray_init(perc=100)

    main_refine_chomp(file='/home_local/tenh_jo/StaticArm04.db')
    # main(robot_id='SingleSphere02')


