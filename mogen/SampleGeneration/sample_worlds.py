import numpy as np

from wzk import print_progress
from wzk.image import img2compressed
from rokin.Vis import robot_3d
from rokin.sample_configurations import sample_q

from mopla.GridWorld import create_perlin_image, create_rectangle_image
from mopla.Optimizer import feasibility_check
from mopla import parameter

from mogen.Loading.load_pandas import create_world_df


def sample_worlds(par, n_worlds, mode='perlin',
                  kwargs_perlin=None, kwargs_rectangles=None,
                  verbose=1):

    img_list = []
    while len(img_list) < n_worlds:
        if verbose > 0:
            print_progress(i=len(img_list), n=n_worlds)

        if mode == 'perlin':
            img = create_perlin_image(n_voxels=par.world.n_voxels, **kwargs_perlin)

        elif mode == 'rectangles':
            img = create_rectangle_image(n_voxels=par.world.n_voxels, **kwargs_rectangles)

        elif mode == 'both':
            img1 = create_perlin_image(n_voxels=par.world.n_voxels, **kwargs_perlin)
            img2 = create_rectangle_image(n_voxels=par.world.n_voxels, **kwargs_rectangles)
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
            robot_3d.robot_path_interactive(q=par.robot.sample_q(100), robot=par.robot, mode='mesh',
                                            obstacle_img=par.oc.img,
                                            voxel_size=par.world.voxel_size, lower_left=par.world.limits[:, 0])

    img_list = img2compressed(img=np.array(img_list, dtype=bool), n_dim=par.world.n_dim)
    world_df = create_world_df(i_world=np.arange(n_worlds).tolist(), img_cmp=img_list)
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


def test():
    from mogen.Loading.load_sql import df2sql
    from rokin.Robots import Justin19
    robot = Justin19()

    par = parameter.Parameter(robot=robot, obstacle_img=None)
    par.check.self_collision = True
    par.check.obstacle_collision = True
    df = sample_worlds(par=par, n_worlds=1000,
                       mode='perlin', kwargs_perlin=dict(threshold=0.4))
    # df = sample_worlds(par=par, n_worlds=1000,
    #                    mode='rectangles', kwargs_rectangles=dict(n=20, size_limits=(1, 20)))  #  special_dim=((0, 1, 2), 20)
    # df = sample_worlds(par=par, n_worlds=1000,
    #                    mode='both',
    #                    kwargs_rectangles=dict(n=20, size_limits=(1, 20)),
    #                    kwargs_perlin=dict(threshold=0.45))

    # create_perlin_image()
    df2sql(df=df, file=f"{robot.id}.db", table_name='worlds', if_exists='replace')


test()
