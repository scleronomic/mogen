import numpy as np

from wzk import print_progress
from wzk.image import img2compressed, compressed2img
from wzk.sql2 import df2sql, get_values_sql

from rokin.Vis import robot_2d, robot_3d
from rokin.sample_configurations import sample_q

from mopla.World import create_perlin_image, create_rectangle_image
from mopla.Optimizer import feasibility_check
from mopla import parameter

from mogen.Loading.load import create_world_df


def sample_worlds(par, n_worlds, mode='perlin',
                  kwargs_perlin=None, kwargs_rectangles=None,
                  verbose=1):

    img_list = []
    while len(img_list) < n_worlds:
        if verbose > 0:
            print_progress(i=len(img_list), n=n_worlds)

        if mode == 'perlin':
            img = create_perlin_image(shape=par.world.shape, **kwargs_perlin)

        elif mode == 'rectangles':
            img = create_rectangle_image(shape=par.world.shape, **kwargs_rectangles)

        elif mode == 'both':
            img1 = create_perlin_image(shape=par.world.shape, **kwargs_perlin)
            img2 = create_rectangle_image(shape=par.world.shape, **kwargs_rectangles)
            img = np.logical_or(img1, img2)

        else:
            raise ValueError

        parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

        try:
            sample_q(par.robot, shape=10, feasibility_check=lambda q: feasibility_check(q=q, par=par))
            img_list.append(img)

        except RuntimeError:
            pass

        if verbose > 2:
            if par.robot.n_dim == 2:
                fig, ax = robot_2d.new_world_fig(limits=par.world.limits)
                robot_2d.plot_img_patch_w_outlines(ax=ax, img=par.oc.img, limits=par.world.limits)

            if par.robot.n_dim == 3:
                robot_3d.robot_path_interactive(q=par.robot.sample_q(100), robot=par.robot,
                                                kwargs_world=dict(img=par.oc.img, limits=par.world.limits))
    # fig, ax = new_fig()
    # hh = np.mean(img_list, axis=(1, 2, 3))
    # ax.hist(hh)

    img_list = img2compressed(img=np.array(img_list, dtype=bool), n_dim=par.world.n_dim)
    aa = compressed2img(img_list, shape=(64, 64, 64), dtype=bool)
    world_df = create_world_df(i_world=np.arange(n_worlds), img_cmp=img_list)
    bb = world_df.img_cmp.values
    bb = compressed2img(img_list, shape=(64, 64, 64), dtype=bool)

    return world_df


def get_robot_max_reach(robot):

    n = 1000
    m = 1000

    n_dim = robot.n_dim
    xmin = np.zeros(n_dim)
    xmax = np.zeros(n_dim)

    for i in range(n):
        print(i)
        q = robot.sample_q(m)
        f = robot.get_frames(q)
        x = f[..., :-1, -1]
        for j in range(3):
            xmin[j] = min(xmin[j], x[..., j].min())
            xmax[j] = max(xmax[j], x[..., j].max())

    limits = np.vstack((xmin, xmax)).T
    limits = np.round(limits, 4)
    return limits


def main():
    from rokin.Robots import Justin19, JustinArm07
    # robot = SingleSphere02(radius=0.25)
    robot = JustinArm07()  # threshold=0.35
    # robot = Justin19()  # threshold=0.40
    # print(get_robot_max_reach(robot))
    par = parameter.Parameter(robot=robot, obstacle_img=None)
    par.check.self_collision = False
    par.check.obstacle_collision = True
    df = sample_worlds(par=par, n_worlds=200,
                       mode='perlin', kwargs_perlin=dict(threshold=0.35), verbose=1)

    # for i in range(20):
    #     df = sample_worlds(par=par, n_worlds=5000,
    #                        mode='perlin', kwargs_perlin=dict(threshold=0.30+i/100), verbose=1)
        # df = sample_worlds(par=par, n_worlds=1000,
        #                    mode='rectangles', kwargs_rectangles=dict(n=20, size_limits=(1, 10)),
        #                    verbose=0)  #  special_dim=((0, 1, 2), 20)
        # df = sample_worlds(par=par, n_worlds=1000,
        #                    mode='both',
        #                    kwargs_rectangles=dict(n=20, size_limits=(1, 20)),
        #                    kwargs_perlin=dict(threshold=0.45))
        # df2sql(df=df, file='world.db', table='perlin', if_exists='append')

    file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}.db'
    file_easy = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_easy.db'
    file_hard = f'/net/rmc-lx0062/home_local/tenh_jo/{robot.id}_hard.db'

    # df2sql(df=df, file=f"{robot.id}.db", table='worlds', if_exists='append')
    ra = 'replace'
    df2sql(df=df, file=file, table='worlds', if_exists=ra)
    df2sql(df=df, file=file_easy, table='worlds', if_exists=ra)
    df2sql(df=df, file=file_hard, table='worlds', if_exists=ra)

    img = get_values_sql(file=file, table='worlds', columns='img_cmp', values_only=True)
    img = compressed2img(img_cmp=img, shape=(64, 64, 64), dtype=bool)
    print(img.shape)
    print(df)


def test_zlib():
    n = 100
    a = np.random.random((n, 64, 64, 64)) < 0.1
    a = a.astype(bool)
    b = img2compressed(img=a, n_dim=3)
    b2 = b.astype(bytes)
    for bb, bb2 in zip(b, b2):
        assert bb == bb2

    assert np.allclose(b, b2)
    a2 = compressed2img(img_cmp=b, shape=(64, 64, 64), dtype=bool)
    a2 = compressed2img(img_cmp=b2, shape=(64, 64, 64), dtype=bool)
    assert np.allclose(a, a2)

    df = create_world_df(i_world=np.arange(n), img_cmp=b)
    a3 = compressed2img(img_cmp=df.img_cmp.values, shape=(64, 64, 64), dtype=bool)
    assert np.allclose(a, a3)

    file = f'/net/rmc-lx0062/home_local/tenh_jo/zlib.db'
    df2sql(df=df, file=file, table='worlds', if_exists='replace')
    img = get_values_sql(file=file, table='worlds', columns='img_cmp', values_only=True)
    a3 = compressed2img(img_cmp=img, shape=(64, 64, 64), dtype=bool)
    print(a3.shape)
    np.allclose(a, a3)

if __name__ == '__main__':
    test_zlib()  #JustinArm07 | World 0-1000 | Samples 0-1000


# import numpy as np
# from rokin.Robots import JustinArm07
# from wzk.trajectory import get_substeps
# from wzk.mpl import new_fig
# robot = JustinArm07()
#
# q = robot.sample_q((1000, 2))
# q = get_substeps(x=q, n=20, include_start=True)
#
# f = robot.get_frames(q)
# x = f[..., :-1, -1]
#
# d = np.linalg.norm(x[:, 1:] - x[:, :-1], axis=-1).sum(axis=1)
#
# fig, ax = new_fig()
# for i in [2, 4, 7]:
#     ax.hist(d[:, i], alpha=0.5, bins=50)
#
# np.mean(d, axis=0)