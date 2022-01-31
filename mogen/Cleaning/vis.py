import numpy as np

from wzk.mpl import new_fig, save_fig, close_all

from wzk.trajectory import get_substeps_adjusted

from rokin.Vis import robot_2d, robot_3d
from rokin.Robots import *
from mopla.Parameter.parameter import Parameter

from mogen.Loading.load import get_sample, get_values_sql


def plot_path_2d(i_s, robot, file):

    par = Parameter(robot=robot)
    i_w, i_s, q, img = get_sample(file=file, i_s=i_s, img_shape=par.world.shape)
    # file_path = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/imgs/{robot.id}/w{i_w}_s{i_s}'
    file_path = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot.id}/w{i_w}_s{i_s}'

    q = q.reshape(-1, robot.n_dof)

    q2 = get_substeps_adjusted(x=q, n=1901)[::100]
    # print(len(q2))
    # q2 = q2[::10]
    # print(np.arange(200)[::10])
    # print(len(q2))
    fig, ax = robot_2d.new_world_fig(limits=par.world.limits)
    robot_2d.plot_img_patch_w_outlines(ax=ax, img=img, limits=par.world.limits)
    robot_2d.plot_x_path(ax=ax, x=q, r=par.robot.spheres_rad, marker='o', color='blue')
    robot_2d.plot_x_path(ax=ax, x=q2, r=par.robot.spheres_rad, marker='o', color='red')
    save_fig(file=file_path, fig=fig, formats='png')


def plot_gif_3d(i_s, file):

    robot = Justin19()
    par = Parameter(robot=robot)

    i_w, i_s, q, img = get_sample(file=file, i_s=i_s, img_shape=par.world.shape)

    file_gif = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot.id}_w{i_w}_s{i_s}.gif'

    q = q.reshape(-1, robot.n_dof)
    robot_3d.robot_path_interactive(p=dict(off_screen=True, gif=file_gif, screen_size=(1024, 768)), q=q, robot=robot,
                                    gif=file_gif,
                                    kwargs_world=dict(limits=par.world.limits, img=img))


def plot_dist_to_q0(file, robot, i):
    i_w, i_s, q, q0 = get_values_sql(file=file, table='paths',
                                     rows=i, columns=['world_i32', 'sample_i32', 'q_f32', 'q0_f32'],
                                     values_only=True)

    file_hist = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/imgs/{robot.id}_hist'

    q0 = q0.reshape(len(i), -1, 19)
    q = q.reshape(len(i), -1, 19)

    dq = (q0-q).max(axis=(-1, -2))
    fig, ax = new_fig()
    print(len(dq))
    ax.hist(dq, bins=50)
    print(np.sort(dq)[:10])
    save_fig(fig=fig, file=file_hist, formats='png')


def main():

    # robot = Justin19()
    robot = SingleSphere02(radius=0.25)
    # file_easy = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_easy.db'
    file_hard = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_hard.db'
    file = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot.id}_hard2.db'

    # plot_dist_to_q0(file=file_easy, robot=robot, i=np.arange(10000))

    for i in range(1000):
        plot_path_2d(file=file, robot=robot, i_s=i)
        close_all()
    # plot_path_2d(file=file_easy, robot=robot, i_s=1)


if __name__ == '__main__':
    main()
