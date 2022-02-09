import os.path

import numpy as np

from wzk.mpl import new_fig, save_fig, close_all
from wzk import safe_makedir
from wzk import trajectory, sql2

from rokin.Vis import robot_2d, robot_3d
from mopla.Parameter.parameter import Parameter

from mogen.Generation.load import get_samples


def get_fig_file(file, i_w, i_s):
    fig_directory = f"{os.path.dirname(file)}/Vis"
    safe_makedir(fig_directory)
    fig_file = f"{fig_directory}/w{i_w}_s{i_s}"
    return fig_file


def plot_path_2d(file, robot_id, i_s):

    par = Parameter(robot=robot_id)
    i_w, i_s, q, img = get_samples(file=file, i=i_s, img_shape=par.world.shape)
    q = q.reshape(-1, par.robot.n_dof)

    q2 = trajectory.get_path_adjusted(q, m=100, is_periodic=par.robot.infinity_joints)

    fig, ax = robot_2d.new_world_fig(limits=par.world.limits)
    robot_2d.plot_img_patch_w_outlines(ax=ax, img=img, limits=par.world.limits)

    if robot_id == 'SingleSphere02':
        robot_2d.plot_x_path(ax=ax, x=q, r=par.robot.spheres_rad, marker='o', color='blue')
        robot_2d.plot_x_path(ax=ax, x=q2, r=par.robot.spheres_rad, marker='o', color='red')

    else:
        robot_2d.plot_x_path_arm(q=q, robot=par.robot, ax=ax)

    fig_file = get_fig_file(file=file, i_w=i_w, i_s=i_s)
    save_fig(file=fig_file, fig=fig, formats='png')
    close_all()


def plot_path_2d_gif(file, robot_id, i, file_out=None, qq=None):

    par = Parameter(robot=robot_id)
    i_w, i, q, img = get_samples(file=file, i=i, img_shape=par.world.shape)
    if qq is None:
        q = q.reshape(-1, par.robot.n_dof)
    else:
        q = qq.reshape(-1, par.robot.n_dof)
    fig, ax = robot_2d.new_world_fig(limits=par.world.limits)
    robot_2d.plot_img_patch_w_outlines(ax=ax, img=img, limits=par.world.limits)

    if robot_id == 'SingleSphere02':
        pass

    else:
        if file_out is None:
            file_out = get_fig_file(file=file, i_w=i_w, i_s=i)
        robot_2d.animate_arm(ax=ax, robot=par.robot, q=q, n_ss=1, gif=file_out)
        close_all()

    return


def plot_path_3d_gif(file, robot_id, i_s):

    par = Parameter(robot=robot_id)
    i_w, i_s, q, img = get_samples(file=file, i=i_s, img_shape=par.world.shape)
    q = q.reshape(-1, par.robot.n_dof)

    fig_file = get_fig_file(file=file, i_w=i_w, i_s=i_s)
    robot_3d.robot_path_interactive(p=dict(off_screen=True, gif=fig_file, screen_size=(1024, 768)), q=q, robot=par.robot,
                                    kwargs_world=dict(limits=par.world.limits, img=img))

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


def main():

    # robot = Justin19()
    # robot = SingleSphere02(radius=0.25)
    # robot = SingleSphere02(radius=0.25)
    # file_easy = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_easy.db'
    # file_hard = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_hard.db'

    robot_id = 'StaticArm04'
    file = f"/Users/jote/Documents/DLR/Data/mogen/{robot_id}/{robot_id}_hard2.db"

    # plot_dist_to_q0(file=file_easy, robot=robot, i=np.arange(10000))
    i = 2832983
    i = 310744
    i = 2527987
    i = 3449376
    i = 1570113
    i = 1669455
    i = 3033384
    i = 3130827
    # for i in range(100):
    #     print(i)
        # plot_path_2d(file=file, robot_id=robot_id, i_s=i)
    plot_path_2d_gif(file=file, robot_id=robot_id, i=i)


if __name__ == '__main__':
    main()
