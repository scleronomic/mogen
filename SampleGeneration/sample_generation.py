import numpy as np
from wzk import new_fig, save_fig

import GridWorld.world2grid
import Kinematic.forward as forward
import Optimizer.InitialGuess.path as path_i
import Optimizer.Objective.gradient_descent as opt2
import Optimizer.Objective.objectives as cost
import Optimizer.path as path
import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import Util.Visualization.plotting_2 as plt2
import GridWorld.obstacle_distance as cost_f
import GridWorld.random_obstacles as randrect
import GridWorld.swept_volume as sv
import definitions as dfn



# g = par.Geometry('2D/FB/3dof')
# c = par.Optimizer(lll=g.lll, fixed_base=g.fixed_base, evaluation_samples=None)

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


    # print(f"n_dim: {_n_dim}, fixed_base: {_fixed_base}, robot: {par.robot.id}, "
    #       f"n_dof: {par.robot.n_dof}, spheres: {par.robot.n_spheres}")

    if _n_dim != par.world.n_dim:
        raise ValueError(f"Check the geometry settings for {directory}, -> n_dim")

    if np.all(path_df.n_waypoints.values != par.world.n_waypoints):
        raise ValueError('Check the geometry settings for {}, -> n_waypoints'.format(directory))

    if np.all(path_df.r_sphere.values != par.robot.r_sphere):
        raise ValueError('Check the geometry settings for {}, -> r_sphere'.format(directory))






def sample_generation_gd(g, o,
                         x0_generator=None, obstacle_img=None, n_samples=1,
                         return_pandas=False, verbose=1,
                         evaluation_samples=False):
    """
    Create n_samples of paths in a quadratic world with block obstacles (defined by 'obstacle_img', create new one
    if argument isNone).
    Choose start end end point random for each sample, but use 'acceptance_rate' to ensure the overall Measurements is
    distributed as desired.
    """

    if evaluation_samples:
        n_samples = 1

    if x0_generator is None:
        x0_generator = path_i.generator_wrapper_rp(g=g, o=o)

    acceptance_rate_x = g.start_end_dist_x_ar
    acceptance_rate_a = g.start_end_dist_a_ar

    x_start, x_end = path_i.sample_start_end_pos(constraints_x=g.constraints_x, constraints_q=g.constraints_a,
                                                 acceptance_rate_x=acceptance_rate_x, n_dim=g.n_dim, n_samples=n_samples,
                                                 acceptance_rate_q=acceptance_rate_a, edt_cost_fun=obst_cost_fun,
                                                 lll=g.lll, fixed_base=g.fixed_base)

    res = opt2.gd_chomp(q_start=x_start, q_end=x_end, q0=1, par=par, gd=gd)

    if evaluation_samples:
        pass
        # x_initial, x_ms, objective, i_opt_total = res
        # x_initial = x_initial[0]
        # x_opt = x_ms[0]
        # objective = objective[0]
        # n_samples = np.shape(objective)
        # x_start = x_start.repeat(n_samples, axis=0)
        # x_end = x_end.repeat(n_samples, axis=0)

    else:
        x_inner_opt, objective = res

        # Inner -> Full
        x_opt = path.x_inner2x(inner=x_inner_opt, start=x_start, end=x_end, n_dof=g.n_dim + g.n_joints,
                               n_samples=n_samples, return_flat=True)  # Must be flat to be saved correctly as SQL

    # Check feasibility
    feasible, path_img_arr = ld.check_sample_feasibility(x=x_opt, obstacle_img=obstacle_img,
                                                         r_sphere=g.r_sphere, n_dim=g.n_dim, n_samples=n_samples,
                                                         lll=g.lll, n_joints=g.n_joints, n_spheres_tot=g.n_spheres_tot,
                                                         fixed_base=g.fixed_base,
                                                         obst_cost_fun=obst_cost_fun,
                                                         obst_cost_threshold=o.obst_cost_threshold, squeeze=False)

    if g.lll is None:
        x_start_warm = None
        x_end_warm = None
    else:
        if g.n_dim == 2:
            x_start_warm = forward.xa2x_warm_2d(xa=x_start, lll=g.lll, n_joints=g.n_joints,
                                                n_spheres_tot=g.n_spheres_tot,
                                                n_dim=g.n_dim, n_samples=n_samples,
                                                warm_in_sample_dim=False, with_base=not np.any(g.fixed_base))
            x_end_warm = forward.xa2x_warm_2d(xa=x_end, lll=g.lll, n_joints=g.n_joints, n_spheres_tot=g.n_spheres_tot,
                                              n_dim=g.n_dim, n_samples=n_samples,
                                              warm_in_sample_dim=False, with_base=not np.any(g.fixed_base))
        else:  # n_dim
            a_start = path.q2x_q(xq=x_start, n_dim=g.n_dim, n_joints=g.n_joints, n_samples=n_samples)[1]
            a_end = path.q2x_q(xq=x_end, n_dim=g.n_dim, n_joints=g.n_joints, n_samples=n_samples)[1]
            x_start_warm = jstn.get_frames_right_arm_spheres(q=a_start, n_samples=n_samples, warm_in_sample_dim=False)
            x_end_warm = jstn.get_frames_right_arm_spheres(q=a_end, n_samples=n_samples, warm_in_sample_dim=False)

    # Save samples to pandas
    path_df_new = ld.initialize_dataframe()
    for i in range(n_samples):

        # GridWorld to Image
        if feasible[i] or evaluation_samples:
            if g.lll is None:
                start_img = w2i.sphere2grid_whole(x=x_start[i], r_sphere=g.r_sphere,)
                end_img = w2i.sphere2grid_whole(x=x_end[i], r_sphere=g.r_sphere,)
            else:
                start_img = w2i.sphere2grid_whole(x=x_start_warm[i], r_sphere=g.r_sphere,
                                                  n_samples=g.n_spheres_tot_wb)
                end_img = w2i.sphere2grid_whole(x=x_end_warm[i], r_sphere=g.r_sphere,
                                                n_samples=g.n_spheres_tot_wb)

            start_img_cmp = w2i.img2compressed(start_img)
            end_img_cmp = w2i.img2compressed(end_img)
            path_img_cmp = w2i.img2compressed(path_img_arr[i])

            if evaluation_samples:
                path_df_new = path_df_new.append(
                    ld.create_path_dataframe_eval(i_world=-1, i_sample=-1, r_sphere=g.r_sphere, n_waypoints=g.n_waypoints,
                                                  x_start=x_start[i], x_end=x_end[i], x_path=x_opt[i],
                                                  objective=objective[i], x0=x_initial[i], feasible=feasible[i],
                                                  start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp,
                                                  path_img_cmp=path_img_cmp))
            else:
                path_df_new = path_df_new.append(
                    ld.create_path_dataframe(i_world=-1, i_sample=-1, r_sphere=g.r_sphere, n_waypoints=g.n_waypoints,
                                             x_start=x_start[i], x_end=x_end[i], x_path=x_opt[i],
                                             objective=objective[i],
                                             start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp,
                                             path_img_cmp=path_img_cmp))
            if verbose >= 3:
                print('Collision: ', w2i.check_overlap(img_a=path_img_arr[i], img_b=obstacle_img))

                fig, ax = plt2.new_world_fig(g.world_size, scale=2, n_dim=g.n_dim)

                if g.lll is None:
                    plt2.plot_x_path(x=x_opt[i], n_dim=g.n_dim, ax=ax, marker='o')

                    if g.n_dim == 2:
                        plt2.plot_obstacle_path_world(obstacle_img=obstacle_img, path_img=path_img_arr[i], n_dim=g.n_dim,
                                                      world_size=g.world_size, n_voxels=g.n_voxels, ax=ax)

                    else:
                        plt2.plot_obstacle_path_world(obstacle_img=(rectangle_pos, rectangle_size),
                                                      path_img=path_img_arr[i], world_size=g.world_size,
                                                      n_voxels=g.n_voxels, ax=ax, n_dim=g.n_dim)

                else:
                    if g.n_dim == 2:
                        plt2.plot_obstacle_path_world(obstacle_img=obstacle_img, path_img=path_img_arr[i].sum(axis=-1),
                                                      world_size=g.world_size, ax=ax, n_dim=g.n_dim,
                                                      n_voxels=g.n_voxels)

                        x_warm = forward.xa2x_warm_2d(xa=x_opt[i], lll=g.lll, n_dim=g.n_dim, n_joints=g.n_joints,
                                                      warm_in_sample_dim=False, n_spheres_tot=g.n_spheres_tot,
                                                      n_samples=None)
                    else:
                        plt2.plot_obstacle_path_world(obstacle_img=(rectangle_pos, rectangle_size),
                                                      path_img=None, world_size=g.world_size,
                                                      n_voxels=g.n_voxels, ax=ax, n_dim=g.n_dim)

                        a_opt = path.q2x_q(xq=x_opt[i], n_dim=g.n_dim, n_joints=g.n_joints, n_samples=None)[1]
                        x_warm = jstn.get_frames_right_arm_linkss(q=a_opt, n_samples=None, warm_in_sample_dim=False)

                    plt2.plot_x_spheres(x_spheres=x_warm, n_dim=g.n_dim, ax=ax)

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
        world_df = ld.initialize_dataframe()

    # path_df = ld.initialize_dataframe()

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


def add_path_samples(g, o,
                     i_worlds, x0_generator=None, n_samples_miss=1, fill=None, directory=None, verbose=0,
                     n_samples_cur=None, lock=None, evaluation_samples=False):
    """
    :param i_worlds:
    :param x0_generator:
    :param n_samples_miss: (int) number of samples to generate
    :param fill: (int) if given, overwrite n_samples and add remaining number of samples to dataframe
    :param directory: (str) directory where results should be stored
    :param verbose: (int) flag for printing progress
    :param n_samples_cur: (int) number of samples for this world
    :param lock: (int) lock for SQL Database
    :param evaluation_samples: (bool)
    :param g
    :param o
    :return:
    """

    check_settings(directory=directory, lock=lock, g=g)

    i_world_list = ld.arg_wrapper__i_world(i_worlds, directory=directory)
    world_img_dict = ld.create_world_img_dict(i_worlds=i_world_list, directory=directory)

    for iw in i_world_list:

        if n_samples_cur is None:
            n_samples_cur = ld_sql.get_n_samples(i_worlds=iw, directory=directory, lock=lock)

        if fill is not None:
            n_samples_miss = max(0, fill - n_samples_cur)

        if verbose >= 1:
            print('i_world={}, len={}, n_samples={}'.format(iw, n_samples_cur, n_samples_miss))

        if n_samples_miss == 0:
            n_samples_cur = None
            continue

        _, path_df_new = sample_generation_gd(g=g, o=o,
                                              n_samples=n_samples_miss, x0_generator=x0_generator
                                              , obstacle_img=world_img_dict[iw],
                                              return_pandas=True, verbose=verbose - 1,
                                              evaluation_samples=evaluation_samples)
        if len(path_df_new) == 0:
            n_samples_cur = None
            continue

        path_df_new.loc[:, 'i_world'] = iw
        path_df_new.loc[:, 'i_sample'] = np.arange(n_samples_cur, n_samples_cur + len(path_df_new), dtype=int)

        # Connection to SQl database
        ld_sql.df2sql(df=path_df_new, directory=directory, if_exists='append', lock=lock)

        n_samples_cur = None


def update_objective(g, o,
                     i_worlds=-1, directory=None, lock=None, use_x_pred=False, verbose=1):
    check_settings(g=g, directory=directory, lock=lock)

    n_samples = dfn.n_samples_per_world
    i_world_list = ld.arg_wrapper__i_world(i_worlds, directory=directory)
    world_df = ld.load_world_df(directory=directory)

    if use_x_pred:
        path_type = 'x_pred'
        objective_type = 'objective_pred'
    else:
        path_type = PATH_Q
        objective_type = 'objective'

    for iw in i_world_list:
        _n_voxels, rectangle_pos, rectangle_size = world_df.loc[iw, ['n_voxels', 'rectangle_pos',
                                                                          'rectangle_size']]
        obstacle_img = GridWorld.random_obstacles.rectangles2image(n_voxels=g.n_voxels, rect_pos=rectangle_pos, rect_size=rectangle_size,
                                                                   safety_idx=GridWorld.world2grid.grid_x2i(x=g.fixed_base,
                                                                    safety_margin=g.safety_margin)

        obst_cost_fun = cost_f.obstacle_img2cost_fun(obstacle_img=obstacle_img, r=g.r_sphere,
                                                     eps=o.eps_obstacle_cost, world_size=g.world_size,
                                                     interp_order=o.interp_order_cost)

        # x, pred
        x_start, x_end, x_path, objective = ld_sql.get_values_sql(columns=[START_Q, END_Q, path_type, 'objective'],
                                                                  i_worlds=iw, directory=directory, values_only=True,
                                                                  lock=lock)
        x_inner = path.x2x_inner(x=x_path, n_dof=g.n_dim + g.n_joints, n_samples=dfn.n_samples_per_world)

        # Length norm
        n_wp = path.get_n_waypoints(x=x_path, n_dof=g.n_dim + g.n_joints, n_samples=n_samples)
        if o.length_norm:
            if g.lll is None:
                ln = path.get_start_end_normalization(start=x_start, end=x_end, n=n_wp)
            else:
                ln = forward.get_beeline_normalization(q_start=x_start, q_end=x_end, n_wp=n_wp,
                                                       lll=g.lll, n_joints=g.n_joints, n_spheres_tot=g.n_spheres_tot,
                                                       n_samples=n_samples, fixed_base=g.fixed_base,
                                                       infinity_joints=g.infinity_joints)
        else:
            ln = False

        objective2 = cost.chomp_cost(x_inner=x_inner, x_start=x_start, x_end=x_end, gamma_len=o.gamma_cost,
                                     length_norm=ln, obst_cost_fun=obst_cost_fun,
                                     n_substeps=o.n_obstacle_substeps_cost, n_dim=g.n_dim,
                                     n_samples=n_samples, return_separate=False,
                                     robot_id=g.lll, fixed_base=g.fixed_base, length_a_penalty=o.length_a_penalty,
                                     n_joints=g.n_joints, n_spheres_tot=g.n_spheres_tot, n_wp=g.n_waypoints,
                                     infinity_joints=g.infinity_joints)

        if verbose >= 1:
            print('GridWorld: {} | Updated Objective: {}'.format(iw, np.sum(objective != objective2)))
            # print('GridWorld: {} | Updated Objective: {}'.format(iw, np.sum(objective2 > 10)))

        if verbose >= 2:
            for i in np.nonzero(objective2 > 10)[0]:
                print(objective2[i])
                ax = plt2.plot_sample(i_sample_local=i, i_world=iw, directory='2D/2dof_redo/')
                save_fig(filename='test_w{}_s{}'.format(iw, i), ax=ax.get_figure())

        ld_sql.set_values_sql(values=(objective2,), columns=[objective_type], i_worlds=iw, directory=directory,
                              lock=lock)


def update_path_img(g,
                    i_worlds=-1, directory=None, lock=None, verbose=1):
    check_settings(directory=directory, lock=lock, g=g)

    n_samples = dfn.n_samples_per_world
    i_world_list = ld.arg_wrapper__i_world(i_worlds, directory=directory)

    n_channels = None
    for iw in i_world_list:
        if verbose >= 1:
            print('GridWorld:', iw)

        # x_path = ld_sql.get_values_sql(columns=PATH_Q, i_worlds=iw, directory=directory, values_only=True, lock=lock)

        # DEBUGGING
        x_start, x_end, x_path, start_img_base, end_img_base, path_img_base = \
            ld_sql.get_values_sql(columns=[START_Q, END_Q, PATH_Q, START_IMG_CMP, END_IMG_CMP,
                                           PATH_IMG_CMP],
                                  i_worlds=iw, directory=directory, values_only=True, lock=lock)

        feasible, path_img_arr = ld.check_sample_feasibility(x=x_path.copy(), obstacle_img=None,
                                                             r_sphere=g.r_sphere, n_dim=g.n_dim, n_samples=n_samples,
                                                             lll=g.lll, n_joints=g.n_joints,
                                                             n_spheres_tot=g.n_spheres_tot
                                                             , fixed_base=g.fixed_base)

        # Check if there are differences
        # count = 0
        # path_img_base = compressed2img(path_img_base, n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels)
        # for o in range(n_samples):
        #     if not np.allclose(path_img_base[o], path_img_arr[o]):
        #         # print(o)
        #         diff_img = path_img_base[o].astype(int) - path_img_arr[o].astype(int)
        #         if np.sum(diff_img) < 1:
        #             continue
        #         count += 1
        #
        # print(iw, count)
        #         # fig, ax = plt2.new_world_fig(world_size=world_size)
        #         # plt2.world_imshow(diff_img, ax=ax, world_size=world_size, alpha=0.4)
        #         # plt2.world_imshow(path_img_base[o], ax=ax, world_size=world_size, alpha=0.4)
        #         # plt2.world_imshow(path_img_arr[o], ax=ax, world_size=world_size, alpha=0.4)
        # if count == 0:
        #     continue

        # path_img_cmp = [w2i.img2compressed(path_img_arr[o]) for o in range(n_samples)]
        # #
        # ld_sql.set_values_sql(values=(path_img_cmp), columns=[PATH_IMG_CMP], i_worlds=iw, directory=directory,
        #                       lock=lock)


def redo_path_samples(g, o,
                      i_worlds=-1, directory=None, verbose=0, lock=None, redo_cost_threshold=5):
    # The column 'x_pred' must be updated before running this
    # The column 'objective' must be updated before running this

    # raise NotImplementedError('Look at this and repair')
    check_settings(directory=directory, lock=lock, g=g)

    x0_generator = path_i.generator_wrapper_rp(g=g, o=o)
    n_samples = dfn.n_samples_per_world

    i_world_list = ld.arg_wrapper__i_world(i_worlds=i_worlds, directory=directory)
    world_img_dict = ld.create_world_img_dict(i_worlds=i_world_list, directory=directory)
    for iw in i_world_list:

        # Get Obstacles of world
        obstacle_img = world_img_dict[iw]
        obst_cost_fun = cost_f.obstacle_img2cost_fun(
            obstacle_img=obstacle_img, r=g.r_sphere, eps=o.eps_obstacle_cost, world_size=g.world_size,
            interp_order=o.interp_order_cost)
        obst_cost_fun_grad = cost_f.obstacle_img2cost_fun_grad(obstacle_img=obstacle_img, r=g.r_sphere,
                                                               world_size=g.world_size, eps=o.eps_obstacle_cost_grad,
                                                               interp_order=o.interp_order_grad)
        # Get information of current paths
        x_start, x_end, x_path, objective, path_img_cmp = \
            ld_sql.get_values_sql(columns=[START_Q, END_Q, PATH_Q, 'objective', PATH_IMG_CMP],
                                  i_worlds=iw, directory=directory, values_only=True, lock=lock)

        x_inner = path.x2x_inner(x=x_path.copy(), n_samples=dfn.n_samples_per_world, return_flat=False,
                                 n_dof=g.n_dim + g.n_joints)

        # Length norm
        if o.length_norm:
            if g.lll is None:
                ln = path.get_start_end_normalization(start=x_start, end=x_end, n=g.n_waypoints)
            else:
                ln = forward.get_beeline_normalization(q_start=x_start, q_end=x_end,
                                                       n_wp=g.n_waypoints, n_samples=n_samples, n_dim=g.n_dim,
                                                       lll=g.lll, fixed_base=g.fixed_base, n_joints=g.n_joints,
                                                       n_spheres_tot=g.n_spheres_tot, infinity_joints=g.infinity_joints)
        else:
            ln = False

        objective2 = cost.chomp_cost(x_inner=x_inner, x_start=x_start, x_end=x_end, gamma_len=o.gamma_cost,
                                     length_norm=ln, obst_cost_fun=obst_cost_fun, return_separate=False,
                                     n_substeps=o.n_obstacle_substeps, n_dim=g.n_dim, fixed_base=g.fixed_base,
                                     n_samples=n_samples, robot_id=g.lll, length_a_penalty=o.length_a_penalty,
                                     infinity_joints=g.infinity_joints, n_wp=g.n_waypoints)
        if o.length_norm:
            ln = np.array(np.split(ln, n_samples))
        n_bad = np.sum(objective2 > redo_cost_threshold)
        n_solved = 0
        for i in np.nonzero(objective2 > redo_cost_threshold)[0]:
            if ln[i].min() < 0.1:
                continue
            # print(objective2[o])
            x_inner_opt, objective_opt = \
                opt2.gd_chomp(x0_generator=x0_generator, q_start=x_start[i], q_end=x_end[i],
                              n_samples=None, n_ss_obst_cost=o.n_obstacle_substeps_cost,
                              n_dim=g.n_dim, obst_cost_fun=obst_cost_fun, length_norm=True,
                              gamma_cost=o.gamma_cost, gamma_len=o.gamma_grad.copy(),
                              obst_cost_fun_grad=obst_cost_fun_grad, gd_step_number=o.gd_step_number,
                              gd_step_size=o.gd_step_size, adjust_gd_step_size=o.adjust_gd_step_size,
                              verbose=verbose - 1, robot_id=g.lll, fixed_base=g.fixed_base,
                              constraints_x=g.constraints_x, constraints_a=g.constraints_a)

            feasible, path_img_arr = ld.check_sample_feasibility(x=x_opt.copy(), obstacle_img=obstacle_img,
                                                                 r_sphere=g.r_sphere, n_dim=g.n_dim,
                                                                 n_samples=1, lll=g.lll, n_joints=g.n_joints,
                                                                 n_spheres_tot=g.n_spheres_tot,
                                                                 fixed_base=g.fixed_base,
                                                                 obst_cost_fun=obst_cost_fun,
                                                                 obst_cost_threshold=o.obst_cost_threshold,
                                                                 squeeze=True)

            # Inner -> Full + Flat
            x_opt = path.x_inner2x(inner=x_inner_opt, start=x_start[i], end=x_end[i], n_dof=g.n_dim + g.n_joints,
                                   n_samples=None, return_flat=True)  # flat for SQL

            # fig, ax = plt2.new_world_fig(world_size=world_size)
            # plt2.plot_world_hatch(obstacle_img,  ax=ax)
            # plt2.plot_path(x_opt)
            # plt2.plot_path(x_path[o])
            if feasible and objective_opt < objective2[i]:
                n_solved += 1
                x_path[i] = x_opt
                path_img_cmp[i] = w2i.img2compressed(path_img_arr)
                objective2[i] = objective_opt

        print(iw, n_bad, n_solved)
        if n_solved > 0:
            ld_sql.set_values_sql(values=(x_path, objective2, path_img_cmp),
                                  columns=[PATH_Q, 'objective', PATH_IMG_CMP],
                                  i_worlds=iw, directory=directory, lock=lock)


def improve_path_samples(g, o,
                         i_worlds=-1, directory=None, use_x_pred=False, verbose=0, lock=None):
    # The column 'x_pred' must be updated before running this
    # The column 'objective' must be updated before running this

    check_settings(directory=directory, lock=lock, g=g)

    n_samples = dfn.n_samples_per_world

    i_world_list = ld.arg_wrapper__i_world(i_worlds=i_worlds, directory=directory)
    world_df = ld.load_world_df(directory=directory)
    for iw in i_world_list:

        # Get information of world
        rectangle_pos, rectangle_size = world_df.loc[iw, ['rectangle_pos', 'rectangle_size']]
        obstacle_img = GridWorld.random_obstacles.rectangles2image(n_voxels=g.n_voxels, rect_pos=rectangle_pos, rect_size=rectangle_size,
                                                                   safety_idx=GridWorld.world2grid.grid_x2i(x=g.fixed_base,
                                                                    safety_margin=g.safety_margin)

        obst_cost_fun = \
            cost_f.obstacle_img2cost_fun(obstacle_img=obstacle_img, r=g.r_sphere, eps=o.eps_obstacle_cost,
                                         world_size=g.world_size, interp_order=o.interp_order_cost)
        obst_cost_fun_grad = cost_f.obstacle_img2cost_fun_grad(obstacle_img=obstacle_img, r=g.r_sphere,
                                                               eps=o.eps_obstacle_cost_grad, world_size=g.world_size,
                                                               interp_order=o.interp_order_grad)
        # Get information of current paths
        if use_x_pred:
            x_start, x_end, x_path, x_pred, objective, path_img_cmp = \
                ld_sql.get_values_sql(columns=[START_Q, END_Q, PATH_Q, 'x_pred', 'objective', PATH_IMG_CMP],
                                      i_worlds=iw, directory=directory, values_only=True, lock=lock)
            x_inner = path.x2x_inner(x=x_pred.copy(), n_dof=g.n_dim + g.n_joints, n_samples=n_samples, return_flat=False)

        else:
            x_start, x_end, x_path, objective, path_img_cmp = \
                ld_sql.get_values_sql(columns=[START_Q, END_Q, PATH_Q, 'objective', PATH_IMG_CMP],
                                      i_worlds=iw, directory=directory, values_only=True, lock=lock)

            x_inner = path.x2x_inner(x=x_path.copy(), n_dof=g.n_dim + g.n_joints, n_samples=n_samples,
                                     return_flat=False)

        objective_old = objective.mean()

        # Perform Optimization
        x_inner_opt, objective_opt = \
            opt2.gradient_descent_path_cost(constraints_x=g.constraints_x, constraints_q=g.constraints_a,
                                            infinity_joints=g.infinity_joints,
                                            x_inner=x_inner, x_start=x_start, x_end=x_end, gamma_cost=o.gamma_cost,
                                            gamma_grad=o.gamma_grad.copy(), n_wp=g.n_waypoints,
                                            n_ss_obst_cost=o.n_obstacle_substeps_cost,
                                            n_ss_obst_grad=o.n_obstacle_substeps_grad,
                                            obst_cost_fun=obst_cost_fun, obst_cost_fun_grad=obst_cost_fun_grad,
                                            n_dim=g.n_dim, n_samples=n_samples, lll=g.lll, fixed_base=g.fixed_base,
                                            gd_step_number=o.gd_step_number,
                                            gd_step_size=o.gd_step_size, adjust_gd_step_size=o.adjust_gd_step_size,
                                            length_norm=True,
                                            return_separate=False, verbose=verbose - 1)

        # Inner -> Full + Flat
        x_opt = path.x_inner2x(inner=x_inner_opt, start=x_start, end=x_end, n_dof=g.n_dim + g.n_joints,
                               n_samples=n_samples, return_flat=True)

        # Check improvement and feasibility
        idx_improvement = objective_opt < objective
        # Exclude the paths with no improvement from the feasibility check to speed things up
        x_opt[np.logical_not(idx_improvement), :] = -1
        feasible, path_img_arr = ld.check_sample_feasibility(x=x_opt.copy(), obstacle_img=obstacle_img,
                                                             r_sphere=g.r_sphere, n_dim=g.n_dim, n_samples=n_samples,
                                                             lll=g.lll, n_joints=g.n_joints,
                                                             n_spheres_tot=g.n_spheres_tot,
                                                             fixed_base=g.fixed_base,
                                                             obst_cost_fun=obst_cost_fun,
                                                             obst_cost_threshold=o.obst_cost_threshold, squeeze=False)
        idx_improvement = np.logical_and(idx_improvement, feasible)
        path_img_cmp_opt = w2i.img2compressed(img=path_img_arr, n_dim=path_img_arr.shape-1)

        # Save the better results
        x_path[idx_improvement] = x_opt[idx_improvement]
        objective[idx_improvement] = objective_opt[idx_improvement]
        path_img_cmp[idx_improvement] = path_img_cmp_opt[idx_improvement]

        if verbose >= 1:
            print('GridWorld: {} | Updated Paths: {} | Mean Cost: {} -> {}'.
                  format(iw, np.sum(feasible), objective_old, objective.mean()))

        ld_sql.set_values_sql(values=(x_path, objective, path_img_cmp), columns=[PATH_Q, 'objective', PATH_IMG_CMP],
                              i_worlds=iw, directory=directory, lock=lock)


def reduce_n_voxels_df(g, directory, i_worlds, kernel=2, lock=None,
                       kernel_old=1):
    # df = get_values_sql(directory=directory)
    # df.sort_values(by, inplace=True)
    # con = get_sql_connection(db_file=None, directory=directory)
    #
    # df.to_sql(name='path_db', con=con, if_exists='replace', idx=False)
    # con.close()
    # directory = '3D/SR/3dof/'
    #

    n_voxels_old = g.n_voxels // kernel_old
    if kernel_old > 1:
        dtype = float
    else:
        dtype = bool

    i_world_list = ld.arg_wrapper__i_world(i_worlds=i_worlds, directory=directory)
    n_samples = dfn.n_samples_per_world
    for iw in i_world_list:
        start_img, end_img, path_img = \
            ld_sql.get_values_sql(columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP],
                                  i_worlds=iw, directory=directory, values_only=True)

        start_img = compressed2img(start_img, n_voxels=n_voxels_old, n_dim=g.n_dim, n_channels=g.n_img_channels,
                                   dtype=dtype)
        end_img = compressed2img(end_img, n_voxels=n_voxels_old, n_dim=g.n_dim, n_channels=g.n_img_channels,
                                 dtype=dtype)
        path_img = compressed2img(path_img, n_voxels=n_voxels_old, n_dim=g.n_dim, n_channels=g.n_img_channels,
                                  dtype=dtype)

        start_img = ld.reduce_n_voxels(img=start_img, n_voxels=n_voxels_old, n_dim=g.n_dim,
                                       n_channels=g.n_img_channels,
                                       kernel=kernel, n_samples=n_samples, sample_dim=True, channel_dim=False)
        end_img = ld.reduce_n_voxels(img=end_img, n_voxels=n_voxels_old, n_dim=g.n_dim,
                                     n_channels=g.n_img_channels,
                                     kernel=kernel, n_samples=n_samples, sample_dim=True, channel_dim=False)
        path_img = ld.reduce_n_voxels(img=path_img, n_voxels=n_voxels_old, n_dim=g.n_dim,
                                      n_channels=g.n_img_channels,
                                      kernel=kernel, n_samples=n_samples, sample_dim=True, channel_dim=False)

        # fig, ax = plt2.new_world_fig(world_size=g.world_size)
        # plt2.world_imshow(img=path_img[0], ax=ax, world_size=g.world_size)

        start_img = w2i.img2compressed(img=start_img, n_dim=n_dim)
        end_img = w2i.img2compressed(img=end_img, n_dim=n_dim)
        path_img = w2i.img2compressed(img=path_img, n_dim=n_dim)

        ld_sql.set_values_sql(values=(start_img, end_img, path_img), i_worlds=iw, directory=directory, lock=lock,
                              columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])


