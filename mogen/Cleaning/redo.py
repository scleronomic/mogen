import os.path

import numpy as np
# import fire

from wzk import sql2, trajectory
from wzk import tictoc, safe_rmdir
from wzk.mpl import new_fig, remove_duplicate_labels, save_fig, close_all
from wzk.ray2 import ray, ray_init

from rokin.Vis.robot_2d import plot_img_patch_w_outlines
from mopla.Optimizer import feasibility_check, objectives
from mopla.Optimizer.gradient_descent import gd_chomp

from mogen.Generation import data, parameter
from mogen.Vis.main import animate_path, input_wrapper


__directory_numpy_tmp = 'tmp_np'

batch_size_dict = {'SingleSphere02': 10000,
                   'StaticArm04': 10000,
                   'JustinArm07': 200,
                   'Justin19': 200}


def create_numpy_directory(file, replace=True):
    directory_np = file2numpy_directory(file)
    if replace:
        safe_rmdir(directory_np)
    os.makedirs(directory_np, exist_ok=True)


def file2numpy_directory(file):
    directory = os.path.split(file)[0]
    return f"{directory}/{__directory_numpy_tmp}"


def update_objective(file, par, i=None, i_w=None):
    i, q, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    o = objectives.chomp_cost(q=q, par=par)
    sql2.set_values_sql(file=file, table=data.T_PATHS, columns=[data.C_OBJECTIVE_F], rows=i, values=(o.tolist(),))


def plot_redo(q, q0, q_pred, f, i,
              par):

    if par.robot.id == 'SingleSphere02':
        fig, ax = new_fig(aspect=1, title=f'Feasibility: {f}')
        plot_img_patch_w_outlines(ax=ax, img=par.oc.img, limits=par.world.limits)
        ax.plot(*q0.T, color='blue', marker='o', label='old', markersize=15, alpha=0.8)
        if not np.allclose(q0, q_pred):
            ax.plot(*q_pred.T, color='orange', marker='o', label='new0', markersize=10, alpha=0.8)
        ax.plot(*q.T, color='red', marker='o', label='new', markersize=10)

        ax.legend()
        remove_duplicate_labels(ax=ax)

    else:
        file = data.get_file(robot_id=par.robot.id)
        _, _, img, file_out = input_wrapper(robot_id=par.robot.id, q=q, img=par.oc.img, file=file, i=i)  # TODO j not correct

        file_out = os.path.split(file_out)
        file_out1 = f"{file_out[0]}/redo/{file_out[1]}"
        file_out0 = f"{file_out[0]}/redo/{file_out[1]}_0"

        # animate_path(robot_id=par.robot.id, q=q0[j, :, :], img=par.oc.img, file_out=file_out0)
        # animate_path(robot_id=par.robot.id, q=q[j, :, :], img=par.oc.img, file_out=file_out1)

        # Configuration path
        colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:pink']

        q00 = trajectory.get_substeps(x=q0[[0, -1]], n=par.n_wp-1)
        qd = q - q00  # q[0, :]
        qd0 = q0 - q00  # q0[0, :]
        fig, ax = new_fig()
        for i in range(par.robot.n_dof):
            ax.plot(qd0[:, i], color=colors[i], ls='-', marker='o', markersize=3)
            ax.plot(qd[:, i], color=colors[i], ls='--', marker='s', markersize=3)
        save_fig(fig=fig, file=file_out1, formats='pdf')


        close_all()


def recalculate_objective(file, par,
                          i=None, i_w=None):
    i, q, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    o = objectives.o_len.len_q_cost(q, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)
    sql2.set_values_sql(file=file, table=data.T_PATHS, values=(o,), columns=[data.C_OBJECTIVE_F], rows=i)


def test_spline(file, par, gd, i=None, i_w=None):
    i, q0, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)

    c = trajectory.to_spline(x=q0, n_c=4)
    q1 = trajectory.from_spline(c=c, n_wp=par.n_wp)
    q, o = gd_chomp(q0=trajectory.full2inner(q1), par=par, gd=gd)
    f = feasibility_check(q=q, par=par)

    print(f"{f.sum()} / {f.size}")


def print_improvements(o0, o1, f0, f1,
                       b_fb,
                       verbose):
    ff0 = np.round((f0 == 1).mean(), 3)
    ff1 = np.round((f1 == 1).mean(), 3)

    oo1 = np.round(o1[f1 == 1].mean(), 4)
    oo0 = np.round(o0.astype(float)[f1 == 1].mean(), 4)

    n_fb = b_fb.sum()
    n_nf = (~np.logical_or(f0 == -1, f1 == -1)).sum()
    print(f"# samples: {len(f0)}, # improvements {n_fb}  # infeasible {n_nf} | "
          f"f0: {ff0}, f1: {ff1} | "
          f"o0: {oo0}, o1: {oo1}")

    if verbose > 1:
        improvement = o0 - o1
        improvement[f1 == -1] -= np.inf
        j = np.argmax(improvement)

        print(f'Largest Improvement {j}:', o0[j], o1[j], f1[j])


def get_b_improvements(o0, o1, f0, f1):
    b_fb = np.logical_or(np.logical_and(f1 == +1, o1 < o0),   # feasible and better
                         np.logical_and(f1 == +1, f0 == -1))  # feasible instead of unfeasible
    b_nfb = np.logical_and(np.logical_and(f0 == -1, f1 == -1), o1 < o0)  # not feasible and better
    b_rest = ~np.logical_or(b_fb, b_nfb)  # rest
    assert b_fb.sum() + b_nfb.sum() + b_rest.sum() == len(b_fb)

    return b_fb, b_nfb, b_rest


def refine_chomp(file, par, gd,
                 q_fun=None,
                 i=None, i_w=None,
                 batch_size=10000,
                 verbose=0,
                 mode=None):
    i, q0, img = data.get_samples_for_world(file=file, par=par, i=i, i_w=i_w)
    n = len(i)

    if q_fun is not None:
        q_pred = q_fun(i=i)

    else:
        q_pred = q0

    q_pred = trajectory.get_path_adjusted(q_pred, m=50, is_periodic=par.robot.is_periodic)

    if batch_size > n:
        q1, o1 = gd_chomp(q0=trajectory.full2inner(q_pred), par=par, gd=gd)

    else:
        i2 = np.array_split(np.arange(n), max(2, n//batch_size))
        q1 = np.zeros((n, par.n_wp-2, par.robot.n_dof))
        o1 = np.zeros(n)
        for ii2 in i2:
            par.q_start = q_pred[ii2, 0, :]
            par.q_end = q_pred[ii2, -1, :]
            q1[ii2], o1[ii2] = gd_chomp(q0=trajectory.full2inner(q_pred[ii2]), par=par, gd=gd)

        par.q_start = q0[:, 0, :]
        par.q_end = q0[:, -1, :]

    q1 = trajectory.inner2full(inner=q1, start=par.q_start, end=par.q_end)

    f0 = feasibility_check(q=q0, par=par)
    f1 = feasibility_check(q=q1, par=par)

    o0 = objectives.o_len.len_q_cost(q0, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)
    o1 = objectives.o_len.len_q_cost(q1, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)

    b_fb, b_nfb, b_rest = get_b_improvements(o0=o0, o1=o1, f0=f0, f1=f1)
    print_improvements(o0=o0, o1=o1, f0=f0, f1=f1, b_fb=b_fb, verbose=verbose)

    q1[b_rest] = q0[b_rest]
    o1[b_rest] = o0[b_rest]
    f1[b_rest] = f0[b_rest]

    if mode is None:
        print('no set_values')

    elif mode == 'set_sql':
        sql2.set_values_sql(file=file, table=data.T_PATHS, rows=i, values=(q1, o1, f1),
                            columns=[data.C_Q_F32, data.C_OBJECTIVE_F, data.C_FEASIBLE_I])

    elif mode == 'save_numpy':
        directory_np = file2numpy_directory(file=file)
        file2 = f"{directory_np}/s_{min(i)}-{max(i)}"

        np.savez(file2,
                 i=i, q=q1, o=o1, f=f1)

    else:
        raise ValueError


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
                        columns=[data.C_Q_F32], rows=i)


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
                         q_fun=None, i_w=(i_w, i_w_all), verbose=2)
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
            _ = compressed2img(img_cmp=img_cmp[i], shape=img_shape, dtype=bool)
        except zlib.error:
            print(i)

    i_w = sql2.get_values_sql(file=file, table=data.T_PATHS, rows=-1, columns=data.C_WORLD_I, values_only=True)
    u, c = np.unique(i_w, return_counts=True)
    assert u == np.arange(len(u))


def main_refine_chomp(file, q_fun=None, ray_perc=100, mode=None):
    ray_init(perc=ray_perc)

    robot_id = parameter.get_robot_str(file)

    batch_size = batch_size_dict[robot_id]
    iw_all = sql2.get_values_sql(file=file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    iw_list = np.unique(iw_all)

    create_numpy_directory(file=file, replace=True)

    q_fun_ray0 = ray.put(q_fun)

    @ray.remote
    def refine_ray(q_fun_ray, i):
        gen = parameter.init_par(robot_id)
        par, gd = gen.par, gen.gd

        par.oc.n_substeps_check += 2
        par.sc.n_substeps_check = par.oc.n_substeps_check
        gd.n_steps = 50

        with tictoc(text=f'World {min(i)}') as _:
            refine_chomp(file=file, par=par, gd=gd, q_fun=q_fun_ray, i=i, batch_size=batch_size, verbose=1, mode=mode)

        return 1

    futures = []
    for iw in iw_list:
        ii = data.iw2is_wrapper(iw=iw, iw_all=iw_all)
        futures.append(refine_ray.remote(q_fun_ray0, ii))

    res = ray.get(futures)
    print(f"{np.sum(res)} / {np.size(res)}")


def __setup(file):
    robot_id = parameter.get_robot_str(file)
    i_w_all = sql2.get_values_sql(file=file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    # iw_list = np.unique(i_w_all)

    gen = parameter.init_par(robot_id)
    par, gd = gen.par, gen.gd

    return par, gd, i_w_all


def main_recalculate_objective(file):
    par, gd, i_w_all = __setup(file)

    for i_w in np.unique(i_w_all):
        with tictoc(text=f'World {i_w}') as _:
            recalculate_objective(file=file, par=par, i_w=(i_w, i_w_all), )


def main_test_splines(file):

    par, gd, i_w_all = __setup(file)

    for i_w in np.unique(i_w_all):
        with tictoc(text=f'World {i_w}') as _:
            test_spline(file=file, par=par, i_w=(i_w, i_w_all), gd=gd)


def tmp_numpy2sql(file):
    directory_np = file2numpy_directory(file=file)
    file_list = os.listdir(directory_np)
    file_list.sort()
    count = 0
    for file_i in file_list:
        if not file_i.endswith('.npz'):
            continue

        data_i = np.load(f"{directory_np}/{file_i}", allow_pickle=True)
        i, q, o, f = data_i['i'], data_i['q'], data_i['o'], data_i['f']

        sql2.set_values_sql(file=file, table=data.T_PATHS, rows=i, values=(q, o, f),
                            columns=[data.C_Q_F32, data.C_OBJECTIVE_F, data.C_FEASIBLE_I])

        count += 1
        print(f"{count}: {min(i)}-{max(i)}")

    safe_rmdir(directory_np)


if __name__ == '__main__':
    pass
    # fire.Fire({
    #     'objective': main_recalculate_objective,
    #     'chomp': main_refine_chomp
    # })

    # main_recalculate_objective(robot_id='SingleSphere02')

    # ray_init(perc=100)

    # main(robot_id='SingleSphere02')

    # _file = '/Users/jote/Documents/DLR/Data/mogen/StaticArm04/StaticArm04.db'

    _robot_id = 'Justin19'
    _file = data.get_file(robot_id=_robot_id)

    # main_test_splines(file=_file)
    main_refine_chomp(file=_file, ray_perc=100, mode='save_numpy')
    tmp_numpy2sql(file=_file)
    from wzk.gcp.gcloud2 import gsutil_cp
    gsutil_cp(src=_file, dst=f'gs://tenh_jo/{_robot_id}/{_robot_id}_vX.db')

    # iw_all = sql2.get_values_sql(file=_file, table=data.T_PATHS, columns=data.C_WORLD_I, rows=-1, values_only=True)
    # iw_list = np.unique(iw_all)
    #
    # gen = parameter.init_par(_robot_id)
    # par, gd = gen.par, gen.gd
    #
    # par.oc.n_substeps_check += 2
    # par.sc.n_substeps_check = par.oc.n_substeps_check
    # gd.n_steps = 100
    #
    # with tictoc():
    #     for iw in range(10):
    #         ii = data.iw2is_wrapper(iw=iw, iw_all=iw_all)
    #         refine_chomp(file=_file, par=par, gd=gd, q_fun=None, i=ii, verbose=13, mode=None)


# TODO write automatic train and learn loop :)
