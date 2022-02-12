import os.path

import numpy as np

from wzk.mpl import new_fig, save_fig, close_all
from wzk import safe_makedir
from wzk import trajectory, sql2

from rokin.Vis import robot_2d, robot_3d
from mogen.Generation.parameter import init_par

from mogen.Generation.load import get_samples


def get_fig_file(file, i_w, i_s):
    fig_directory = f"{os.path.dirname(file)}/Vis"
    safe_makedir(fig_directory)
    fig_file = f"{fig_directory}/w{i_w}_s{i_s}"
    return fig_file


def plot_path_2d(file, robot_id, i_s):

    par = init_par(robot_id=robot_id).par
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


def plot_path_gif(robot_id,
                  q=None, img=None,
                  file=None, i=None,
                  file_out=None):

    par = init_par(robot_id=robot_id).par
    if file is not None:
        i_w, i_s, q_, img_ = get_samples(file=file, i=i, img_shape=par.world.shape)
        if q is None:
            q = q_
        if img is None:
            img = img_

        file_out = file_out or get_fig_file(file=file, i_w=i_w, i_s=i_s)
    else:
        pass

    q = q.reshape(-1, par.robot.n_dof)

    if par.world.n_dim == 2:
        robot_2d.robot_path_interactive(q=q, img=img, par=par, gif=file_out)

    elif par.world.n_dim == 3:
        robot_3d.robot_path_interactive(p=dict(off_screen=True, gif=file_out, screen_size=(1024, 768)),
                                        q=q, robot=par.robot,
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

    for i in range(100):
        print(i)
        # plot_path_2d(file=file, robot_id=robot_id, i_s=i)
        plot_path_gif(file=file, robot_id=robot_id, i=i)


if __name__ == '__main__':
    main()
