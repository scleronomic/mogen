import numpy as np
import scipy.interpolate as interp
from wzk import print_progress, compressed2img, img2compressed, object2numeric_array, numeric2object_array

import Kinematic.forward as forward
import Optimizer.InitialGuess.path as path_i
import Optimizer.path as path
import SampleGeneration.StartEnd.start_end_statistics as ses
import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import Util.Visualization.plotting_2 as plt2
import GridWorld.swept_volume as sv
import definitions as d

import parameter

par = parameter.initialize_par(robot_id='Stat_Arm_03')


def smooth_path(x, n_wp):
    tck, u = interp.splprep(x.T, s=1, k=min(3, x.size // n_dim - 1))  # Error when k > matrix TODO TUNE s, k
    u = np.linspace(0.0, 1.0, n_wp)
    x_smooth = np.column_stack(interp.splev(x=u, tck=tck))
    return x_smooth


def generate_samples(n_samples, directory, lock=None, verbose=1):
    # tic()
    # n_samples = 1000
    x_start, x_end = path_i.sample_start_end_pos(constraints_x=constraints_x, constraints_q=constraints_a,
                                                 acceptance_rate_x=acceptance_rate_x, n_dim=n_dim, n_samples=n_samples,
                                                 acceptance_rate_q=acceptance_rate_a, edt_cost_fun=None,
                                                 lll=lll, fixed_base=fixed_base, relative_angles=relative_angles)
    # diff = np.abs(x_end - x_start)
    # diff = diff[:, 0, n_dim:]
    # print(diff.mean())
    n_multi_start_a_temp = n_multi_start_rp_a * int(np.ceil(n_samples / len(n_multi_start_rp_a)))

    xa = np.zeros((n_samples, n_waypoints, n_dim + n_joints))
    for i in range(n_samples):
        x_path = path_i.initialize_xq0_random(q_start=x_start[i], q_end=x_end[i], constraints_x=constraints_x,
                                              n_waypoints=n_waypoints * 2, constraints_q=constraints_a,
                                              n_dim=n_dim, fixed_base=fixed_base, order_random=True,
                                              n_random_points_x=0,
                                              n_random_points_a=n_multi_start_a_temp[i],
                                              infinity_joints=infinity_joints)
        xa[i] = smooth_path(x=x_path, n_wp=n_waypoints)

    # TODO x_warm function for justin
    x_warm = forward.xa2x_warm_2d(relative_angles=relative_angles, lll=lll, n_joints=n_joints,
                                  n_spheres_tot=n_links_tot, xa=xa, with_base=True, n_dim=n_dim,
                                  warm_in_sample_dim=False, n_samples=n_samples)

    # Create DataFrames
    # world_df_new = ld.create_world_dataframe_path_img(world_size=world_size, n_voxels=n_voxels, r_sphere=r_sphere)

    path_df_new = ld.initialize_dataframe()

    for i in range(n_samples):
        if verbose >= 1:
            print_progress(i, total=n_samples, prefix='Path Image Generation')

        path_img = sv.sphere2grid_path(x=x_warm[i, 1:], r_sphere=r_sphere, n_dim=n_dim, n_spheres=n_joints)
        path_img_cmp = sv.img2compressed(img=path_img)

        if verbose >= 2:
            fig, ax = plt2.new_world_fig()
            plt2.plot_x_spheres(x_spheres=x_warm[i], n_dim=n_dim)
            plt2.imshow(img=path_img, limits=world_size, ax=ax)

        path_df_new = path_df_new.append(
            ld.create_path_dataframe_path_img(i_world=0, i_sample=i, x_path=path.x2x_flat(x=xa[i]),
                                              x_warm=path.x2x_flat(x=x_warm[i], n_samples=n_joints).flatten(),
                                              path_img_cmp=path_img_cmp))

    # Save to SQL
    ld_sql.df2sql(df=path_df_new, directory=directory, if_exists='append', lock=lock)


def direct_path2direct_path():
    n_joints = 2
    n_dim = 2
    path_df = ld_sql.get_values_sql(directory='PathImg/2D/FB/2dof_direct')

    x_path = path_df.loc[:, PATH_Q]
    x_path = object2numeric_array(x_path)
    x_path = path.x_flat2x(x_flat=x_path, n_dof=n_dof)

    x_start = x_path[:, 0, :]
    x_end = x_path[:, -1, :]

    path_df.loc[:, START_Q] = basic.numeric2object_array(x_start)
    path_df.loc[:, END_Q] = basic.numeric2object_array(x_end)
    # path_df.loc[:, 'x_start_warm'] = -1
    # path_df.loc[:, 'x_end_warm'] = -1

    path_df.loc[:, START_IMG_CMP] = -1
    path_df.loc[:, END_IMG_CMP] = -1

    del x_start, x_end, x_path

    x_warm = path_df.loc[:, 'x_warm']
    x_warm = object2numeric_array(x_warm)

    for i in range(len(path_df)):
        print_progress(i, total=len(path_df))
        x_warm_i = path.x_flat2x(x_flat=x_warm[i].reshape(n_joints, -1), n_dof=n_dim)

        x_warm_start_i = x_warm_i[:, 0, :]
        x_warm_end_i = x_warm_i[:, -1, :]

        start_img = sv.sphere2grid_whole(x=x_warm_start_i[1:, np.newaxis, :], r_sphere=r_sphere)
        end_img = sv.sphere2grid_whole(x=x_warm_end_i[1:, np.newaxis, :], r_sphere=r_sphere)
        start_img_cmp = img2compressed(img=start_img)
        end_img_cmp = img2compressed(img=end_img)

        path_df.loc[i, START_IMG_CMP] = start_img_cmp
        path_df.loc[i, END_IMG_CMP] = end_img_cmp

    ld_sql.df2sql(df=path_df, directory='PathImg/2D/FB/2_direct_new2', if_exists='append', lock=None)

