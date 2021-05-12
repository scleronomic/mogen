import numpy as np
import pandas as pd
from wzk import print_progress, object2numeric_array, numeric2object_array, compressed2img

import GridWorld.random_obstacles as randrect
from definitions import *


world_df_columns_old = np.array(['world_size', 'n_voxels', 'r_agent_max', 'n_obstacles', 'min_max_obstacle_length',
                                 'rectangle_pos', 'rectangle_wh', 'blocked_ratio', 'obstacle_img'])

world_df_columns = np.array(['world_size', 'n_voxels', 'n_obstacles', 'min_max_obstacle_size',
                             'rectangle_pos', 'rectangle_size'])

world_df_columns_new = np.array(['world_size', 'n_voxels', 'n_obstacles', 'min_max_obstacle_size',
                                 'rectangle_pos', 'rectangle_size', 'lll', 'fixed_base'])

world_df_columns_path_img = np.array(['world_size', 'n_voxels', 'r_sphere'])

path_df_columns_old = np.array(['i_world', 'r_sphere', 'n_waypoints', 'xy_start', 'xy_end', 'x0', 'x_final',
                                'tries', 'net_redo', 'objective', 'start_end_img', 'path_img'])

# path_df_columns = np.array(['i_world', 'r_sphere', 'n_waypoints', START_Q, END_Q, PATH_Q,
#                             'tries', 'objective', START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])

# TODO move r_sphere and n_waypoints to the world_df
# TODO rename, r_sphere is saved in the robot_id (the directory name)
# TODO add robot + weighting = etc config to the saved directory
path_df_columns = np.array(['i_world', 'i_sample', 'r_sphere', 'n_waypoints',
                            START_Q, END_Q,
                            PATH_Q, 'objective',
                            START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])

# Is used for path and for pose
path_df_columns_path_img = np.array(['i_world', 'i_sample', PATH_Q, 'x_warm', PATH_IMG_CMP])

path_df_columns_eval = np.array(['i_world', 'i_sample', 'r_sphere', 'n_waypoints',
                                 START_Q, END_Q,
                                 'x0', PATH_Q, 'objective', 'feasible',
                                 START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])


# Wrapper and Helper
#  Switching dimensions between net-arguments and mpl
def df_set_column_lists(df, column, ll):
    n_samples = len(df)
    df.loc[:, column] = 0
    df.loc[:, column] = df.loc[:, column].astype(object)
    for i in range(n_samples):
        df.at[i, column] = ll[i]

    return df


def initialize_dataframe():
    return pd.DataFrame(data=None)


def create_world_dataframe(*, world_size, n_voxels, n_obstacles, min_max_obstacle_size, rectangle_pos,
                           rectangle_size, lll, fixed_base):
    data = [[world_size, n_voxels, n_obstacles, min_max_obstacle_size, rectangle_pos,
             rectangle_size, lll, fixed_base]]
    return pd.DataFrame(data=data, columns=world_df_columns_new)


def create_world_dataframe_path_img(world_size, n_voxels, r_sphere):
    data = [[world_size, n_voxels, r_sphere]]
    return pd.DataFrame(data=data, columns=world_df_columns_path_img)


def create_path_dataframe(i_world, i_sample, r_sphere, n_waypoints, x_start, x_end, x_path, objective=-1,
                          start_img_cmp=None, end_img_cmp=None, path_img_cmp=None):
    data = [[i_world, i_sample, r_sphere, n_waypoints,
             x_start, x_end,
             x_path, objective,
             start_img_cmp, end_img_cmp, path_img_cmp]]
    return pd.DataFrame(data=data, columns=path_df_columns)


def create_path_dataframe_eval(i_world, i_sample, r_sphere, n_waypoints,
                               x_start, x_end, x0, x_path, objective, feasible,
                               start_img_cmp=None, end_img_cmp=None, path_img_cmp=None):
    data = [[i_world, i_sample, r_sphere, n_waypoints,
             x_start, x_end,
             x0, x_path, objective, feasible,
             start_img_cmp, end_img_cmp, path_img_cmp]]
    return pd.DataFrame(data=data, columns=path_df_columns_eval)


def create_path_dataframe_path_img(i_world, i_sample, x_path, x_warm, path_img_cmp):
    data = [[i_world, i_sample, x_path, x_warm, path_img_cmp]]
    return pd.DataFrame(data=data, columns=path_df_columns_path_img)


# TODO Can be deleted if everything is converted to .df
def load_world_df(file, decompress_edt=False):
    world_df = pd.read_pickle(file)

    if decompress_edt:
        n_voxels = world_df.loc[0, 'n_voxels']
        n_dim = np.size(world_df.loc[0, 'min_max_obstacle_size'])

        if 'edt_img' in world_df.columns:
            edt_img = world_df.loc[:, 'edt_img'].values
            edt_img = compressed2img(img_cmp=edt_img, n_voxels=n_voxels, n_dim=n_dim, dtype=float)
            world_df.loc[:, 'edt_img'] = numeric2object_array(edt_img)

    return world_df


# TODO Can be deleted if everything is converted to .df
def add_obstacle_img_column(world_df, values_only=False, verbose=0):
    obstacle_img_arr = []

    if isinstance(world_df, pd.Series):
        _range = [1]
    else:
        _range = world_df.index

    for iw in _range:
        if verbose >= 1:
            print_progress(iw, len(world_df), prefix='Add obstacle_image-column')

        if len(world_df) == 1:
            n_voxels = world_df['n_voxels'].values[0]
            rectangle_pos = world_df['rectangle_pos'].values[0]
            rectangle_size = world_df['rectangle_size'].values[0]

        elif isinstance(world_df, pd.Series):
            n_voxels = world_df['n_voxels']
            rectangle_pos = world_df['rectangle_pos']
            rectangle_size = world_df['rectangle_size']

        else:
            n_voxels, rectangle_pos, rectangle_size, fixed_base = \
                world_df.loc[iw, ['n_voxels', 'rectangle_pos', 'rectangle_size', 'fixed_base']]

        obstacle_img = randrect.rectangles2image(n_voxels=int(n_voxels), rect_pos=rectangle_pos,
                                                 rect_size=rectangle_size)
        obstacle_img_arr.append(obstacle_img)

    if values_only:
        return object2numeric_array(obstacle_img_arr)

    else:
        world_df.loc[:, 'obstacle_img'] = obstacle_img_arr
        return world_df
