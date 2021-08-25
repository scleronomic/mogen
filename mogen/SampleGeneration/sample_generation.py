import numpy as np

from wzk.mpl import new_fig, save_fig
from wzk.image import img2compressed

import mopla.Optimizer.InitialGuess.path as path_i
import mopla.Optimizer.gradient_descent as opt2
import mopla.Optimizer.objectives as cost

import mopla.GridWorld.obstacle_distance as cost_f
import mopla.GridWorld.random_obstacles as randrect
import mopla.GridWorld.swept_volume as sv


import mogen.Loading.load_pandas as ld
import mogen.Loading.load_sql as ld_sql


# Wrapper for a generator for multi starts between arbitrary start- and end points
def check_settings(directory, par, lock=None):

    try:
        world_df = ld_sql.get_values_sql(rows=0, file=directory + dfn.PATH_DB)
        path_df = ld_sql.get_values_sql(rows=0, file=directory + dfn.WORLD_DB)
    except FileNotFoundError:
        print(f"New Samples with n_dim: {par.world.n_dim}, robot: {par.robot.id}")
        return

    _n_dim = world_df.loc[0, 'rectangle_size'].shape[1]

    _n_dof = np.size(path_df.loc[0, dfn.START_Q])

    # _fixed_base = world_df.loc[0, 'fixed_base']
    # _lll = world_df.loc[0, 'lll']
    # _n_links = forward.lll2n_links(_lll)
    # _n_spheres = forward.lll2n_spheres(_lll)

    if _n_dim != par.world.n_dim:
        raise ValueError(f"Check the geometry settings for {directory}, -> n_dim")

    if np.all(path_df.n_waypoints.values != par.world.n_waypoints):
        raise ValueError('Check the geometry settings for {}, -> n_waypoints'.format(directory))

    if np.all(path_df.r_sphere.values != par.robot.r_sphere):
        raise ValueError('Check the geometry settings for {}, -> r_sphere'.format(directory))


def sample_generation_gd(par, gd,
                         x0_generator=None, obstacle_img=None, n_samples=1,
                         return_pandas=False, verbose=1):
    """
    Create n_samples of paths in a quadratic world with block obstacles (defined by 'obstacle_img', create new one
    if argument is None).
    Choose start end end point random for each sample, but use 'acceptance_rate' to ensure the overall Measurements is
    distributed as desired.
    """

    if x0_generator is None:
        x0_generator = path_i.generator_wrapper_rp(g=g, o=o)

    acceptance_rate_x = g.start_end_dist_x_ar
    acceptance_rate_a = g.start_end_dist_a_ar

    x_start, x_end = path_i.sample_start_end_pos(constraints_x=g.constraints_x, constraints_q=g.constraints_a,
                                                 acceptance_rate_x=acceptance_rate_x, n_dim=g.n_dim, n_samples=n_samples,
                                                 acceptance_rate_q=acceptance_rate_a, edt_cost_fun=obst_cost_fun,
                                                 lll=g.lll, fixed_base=g.fixed_base)

    res = opt2.gd_chomp(q_start=x_start, q_end=x_end, q0=1, par=par, gd=gd)



    x_inner_opt, objective = res

    # Inner -> Full
    x_opt = path.x_inner2x(inner=x_inner_opt, start=x_start, end=x_end, n_dof=g.n_dim + g.n_joints,
                           n_samples=n_samples, return_flat=True)  # Must be flat to be saved correctly as SQL

    # Check feasibility

    # Save samples to pandas
    path_df_new = ld.initialize_df()
    for i in range(n_samples):

            path_df_new = path_df_new.append(
                ld.create_path_df(i_world=-1, i_sample=-1, r_sphere=g.r_sphere, n_waypoints=g.n_waypoints,
                                  x_start=x_start[i], x_end=x_end[i], x_path=x_opt[i],
                                  objective=objective[i],
                                  start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp,
                                  path_img_cmp=path_img_cmp))

    if return_pandas:
        return path_df_new
    else:
        try:
            start_end_img_ar = np.array([i for i in path_df_new.start_end_img.values])
            path_img_ar = np.array([i for i in path_df_new.path_img.values])
            return obstacle_img_ar, start_end_img_ar, path_img_ar
        except AttributeError:
            return np.array([]), np.array([]), np.array([])


def new_world_samples(g, o,
                      n_new_world=1, n_samples_per_world=10, verbose=1, directory=None, evaluation_samples=False,
                      lock=None):
    directory = dfn.arg_wrapper__sample_dir(directory)
    check_settings(directory=directory, lock=lock, g=g)
    x0_generator = path_i.generator_wrapper_rp(g=g, o=o)

    try:
        world_df = ld.load_world_df(directory=directory)
    except FileNotFoundError:
        world_df = ld.initialize_df()

    for i in range(n_new_world):
        if verbose >= 1:
            print('new_world={}'.format(i))
        world_df_new, path_df_new = \
            sample_generation_gd(g=g, o=o,
                                 n_samples=n_samples_per_world, x0_generator=x0_generator, obstacle_img=None,
                                 return_pandas=True, evaluation_samples=evaluation_samples, verbose=verbose - 1)
        i_world = len(world_df)
        if len(path_df_new) == 0:
            continue

        path_df_new.loc[:, 'i_world'] = i_world
        path_df_new.loc[:, 'i_sample'] = np.arange(len(path_df_new), dtype=int)

        world_df_new.idx = range(i_world, i_world + 1)

        world_df = world_df.append(world_df_new)
        world_df.to_pickle(path=directory + dfn.WORLD_DB)

        # path_df = path_df.append(path_df_new)

        # Save to SQL-Database
        ld_sql.df2sql(df=path_df_new, directory=directory, if_exists='append', lock=lock)  # TODO make more efficient
