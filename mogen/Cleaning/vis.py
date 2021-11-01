import numpy as np

from wzk.sql2 import get_values_sql
from wzk.mpl import new_fig, save_fig
from wzk.image import compressed2img

from rokin.Vis import robot_2d, robot_3d
from rokin.Robots import *
from mopla.parameter import Parameter


def get_sample(file, i_s, img_shape):
    i_w, i_s, q = get_values_sql(file=file, table='paths',
                                 rows=i_s, columns=['world_i32', 'sample_i32', 'q_f32'],
                                 values_only=True)
    i_w = int(np.squeeze(i_w))
    i_s = int(np.squeeze(i_s))
    img_cmp = get_values_sql(file=file, rows=i_w, table='worlds', columns='img_cmp', values_only=True)
    img = compressed2img(img_cmp=img_cmp, shape=img_shape, dtype=bool)

    return i_w, i_s, q, img


def plot_path_2d(i_s, robot, file):

    par = Parameter(robot=robot)
    i_w, i_s, q, img = get_sample(file=file, i_s=i_s, img_shape=par.world.shape)
    file_path = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot.id}_w{i_w}_s{i_s}'

    q = q.reshape(-1, robot.n_dof)

    fig, ax = robot_2d.new_world_fig(limits=par.world.limits)
    robot_2d.plot_img_patch_w_outlines(ax=ax, img=img, limits=par.world.limits)
    robot_2d.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax)
    save_fig(file=file_path, fig=fig, formats=('pdf', 'png'))


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
    from wzk import new_fig
    fig, ax = new_fig()
    print(len(dq))
    ax.hist(dq, bins=50)
    print(np.sort(dq)[:10])
    save_fig(fig=fig, file=file_hist, formats=('pdf', 'png'))


if __name__ == '__main__':

    # robot = Justin19()
    robot = SingleSphere02()
    file_easy = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_easy.db'
    file_hard = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_hard.db'

    # plot_dist_to_q0(file=file_easy, robot=robot, i=np.arange(10000))
    plot_path_2d(file=file_easy, robot=robot, i_s=0)
    plot_path_2d(file=file_easy, robot=robot, i_s=1)


# q = np.random.random((100, 20, 19))
# ax.plot()
# def a():
#     pass
#
#
# for i in range(0, 10000, 10):
#     sample_gif_3d(i_s=i, file=file_easy)


# p = robot_3d.pv.Plotter(off_screen=False)