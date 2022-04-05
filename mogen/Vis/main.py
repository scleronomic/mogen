import os.path

import numpy as np

from wzk.mpl import new_fig, save_fig, close_all
from wzk import safe_makedir
from wzk import trajectory, sql2

from rokin.Vis import robot_2d, robot_3d

from mogen.Generation import data, parameter


def get_fig_file(file, i_w, i_s):
    fig_directory = f"{os.path.dirname(file)}/Vis"
    safe_makedir(fig_directory)
    fig_file = f"{fig_directory}/w{i_w}_s{i_s}"
    return fig_file


def input_wrapper(robot_id,
                  q=None, img=None,
                  file=None, i=None,
                  file_out=None):
    par = parameter.init_par(robot_id=robot_id).par

    if file is not None:
        i_w, i_s, q_, img_ = data.get_samples(file=file, i=i, img_shape=par.world.shape)
        if q is None:
            q = q_
        if img is None:
            img = img_

        file_out = file_out or get_fig_file(file=file, i_w=i_w, i_s=i_s)
    else:
        pass

    q = q.reshape(-1, par.robot.n_dof)
    return par, q, img, file_out


def plot_path_2d(robot_id, file, i):

    par = parameter.init_par(robot_id=robot_id).par
    i_w, i_s, q, img = data.get_samples(file=file, i=i, img_shape=par.world.shape)
    q = q.reshape(-1, par.robot.n_dof)
    q2 = trajectory.get_path_adjusted(q, m=100, is_periodic=par.robot.is_periodic)

    ax, h = robot_2d.plot_path(q=q2, img=img, par=par)
    fig_file = get_fig_file(file=file, i_w=i_w, i_s=i_s)
    save_fig(file=fig_file, fig=ax.figure, formats='pdf')
    close_all()


def plot_path(robot_id,
              q=None, img=None,
              file=None, i=None,
              file_out=None,
              formats=None,
              verbose=1):

    par, q, img, file_out = input_wrapper(robot_id=robot_id, q=q, img=img, file=file, i=i, file_out=file_out)

    if par.world.n_dim == 2:
        ax, h = robot_2d.plot_path(q=q, img=img, par=par)
        save_fig(file=file_out, fig=ax.figure, formats=formats, verbose=verbose)
        close_all()

    if par.world.n_dim == 3:
        raise NotImplementedError


def animate_path(robot_id,
                 q=None, img=None,
                 file=None, i=None,
                 file_out=None):

    par, q, img, file_out = input_wrapper(robot_id=robot_id, q=q, img=img, file=file, i=i, file_out=file_out)

    if par.world.n_dim == 2:
        robot_2d.animate_path(q=q, img=img, par=par, gif=file_out)

    elif par.world.n_dim == 3:
        robot_3d.animate_path(p=dict(off_screen=False, gif=file_out, window_size=(1024, 1024)),
                              q=q, robot=par.robot, gif=file_out,
                              kwargs_world=dict(limits=par.world.limits, img=img, mode='mesh'))

        # p = robot_3d.pv.Plotter()
        # p = dict(off_screen=False, gif=file_out, screen_size=(512, 384))
        # robot_3d.robot_path_interactive(p=p,
        #                                 gif=file_out,
        #                                 q=q, robot=par.robot,
        #                                 kwargs_robot=dict(mode=['mesh', 'sphere']),
        #                                 kwargs_world=dict(limits=par.world.limits, img=img, mode='mesh'))
        #
        # p = robot_3d.plotter_wrapper({})
        # robot_3d.plot_bool_vol(p=p, img=img, limits=par.world.limits, mode='voxel', opacity=0.5)
        # p.show()

#
# def plot_dist_to_q0(file, robot, i):
#
#     i_w, i_s, q, q0 = get_values_sql(file=file, table='paths',
#                                      rows=i, columns=['world_i32', 'sample_i32', 'q_f32', 'q0_f32'],
#                                      values_only=True)
#
#     file_hist = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/imgs/{robot.id}_hist'
#
#     q0 = q0.reshape(len(i), -1, 19)
#     q = q.reshape(len(i), -1, 19)
#
#     dq = (q0-q).max(axis=(-1, -2))
#     fig, ax = new_fig()
#     print(len(dq))
#     ax.hist(dq, bins=50)
#     print(np.sort(dq)[:10])
#     save_fig(fig=fig, file=file_hist, formats='png')


def plot_paths(file, i_w):
    robot_id = parameter.get_robot_str(file)
    for i in range(20):
        # plot_path_gif(file=file, robot_id=robot_id, i=i)
        plot_path(file=file, robot_id=robot_id, i=i)


def plot_all_paths_in_world(file, i_w):
    robot_id = parameter.get_robot_str(file)
    i_w_all = sql2.get_values_sql(file=file, table='paths', columns='world_i32', rows=-1, values_only=True)

    gen = parameter.init_par(robot_id)
    par, gd = gen.par, gen.gd

    for iwi in i_w:
        i, q, img = data.get_samples_for_world(file=file, par=par, i=None, i_w=(iwi, i_w_all))

        fig, ax = new_fig(aspect=1)
        robot_2d.plot_img_patch_w_outlines(ax=ax, img=par.oc.img, limits=par.world.limits)

        for qq in q:
            ax.plot(*qq.T, color='b', marker='o', markersize=10, alpha=0.1)

        fig_file = get_fig_file(file=file, i_w=iwi, i_s='')[:-2]
        save_fig(file=fig_file, fig=fig, formats='pdf')
        close_all()


if __name__ == '__main__':
    _robot_id = 'SingleSphere02'
    _file = f"/Users/jote/Documents/DLR/Data/mogen/{_robot_id}/{_robot_id}.db"

    plot_paths(file=_file, i_w=range(100))
    # plot_all_paths_in_world(file=file, i_w=range(10, 100))
