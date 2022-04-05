import numpy as np

from wzk import print_progress
from wzk.image import img2compressed
from wzk.sql2 import df2sql

from rokin.Vis import robot_2d, robot_3d
from rokin.sample_configurations import sample_q

from mopla.World import create_perlin_image, create_rectangle_image
from mopla.Optimizer import feasibility_check
from mopla.Parameter import parameter

from mogen.Generation import data


def sample_worlds(par, n, mode='perlin',
                  kwargs_perlin=None, kwargs_rectangles=None,
                  verbose=1):

    if verbose > 0:
        print(f"generating {n} worlds for the robot {par.robot.id}...")
        print("limits:")
        print(par.world.limits)
        print("mode:")
        print(mode)

    img_list = []
    while len(img_list) < n:
        if verbose > 0:
            print_progress(i=len(img_list), n=n)

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

        parameter.initialize_oc(par=par, obstacle_img=img)

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
                robot_3d.animate_path(q=par.robot.sample_q(100), robot=par.robot,
                                      kwargs_world=dict(img=par.oc.img, limits=par.world.limits))

    img_list = img2compressed(img=np.array(img_list, dtype=bool), n_dim=par.world.n_dim)
    world_df = data.create_world_df(i_world=np.arange(n), img_cmp=img_list)
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
    from shutil import copy
    from mopla.Parameter import get_par_staticarm, get_par_justin19, get_par_singlesphere02, get_par_justinarm07
    # par = get_par_staticarm(n_dof=4, lengths=0.25, widths=0.1)[0]   # threshold=0.5
    # par = get_par_justin19()[0]  # threshold=0.40
    # par = get_par_justinarm07()[0]   # threshold=0.35
    par = get_par_singlesphere02()[0]  # threshold=0.40

    # file = f"/home/johannes_tenhumberg/sdb/{par.robot.id}.db"
    file = f"/Users/jote/Documents/DLR/Data/mogen/{par.robot.id}/{par.robot.id}_worlds.db"
    # copy(file, file2)

    # print(get_robot_max_reach(robot))
    df = sample_worlds(par=par, n=10000, mode='perlin', kwargs_perlin=dict(threshold=0.40, mode='old'), verbose=1)

    df2sql(df=df, file=file, table='worlds', dtype=data.world_df_dtypes, if_exists='replace')
    # TODO add meta table where all the robot parameters are listed


if __name__ == '__main__':
    main()

