from wzk import print_progress, object2numeric_array, save_pickle, image as img_basic

import Optimizer.path as path
from Util.Loading.load import arg_wrapper__i_world
import Util.Loading.load_pandas as load_pd
from Util.Loading.load_sql import *
from Util.Loading.dlr import get_sample_dir
from definitions import *


def rectangle_image_sanity_check(file):
    import GridWorld.random_obstacles as randrect
    world_db = get_values_sql(file=file, rows=-1)

    n_worlds = len(world_db)
    n_voxels = (64, 64)
    n_dim = 2

    rect_pos = [rp.reshape(-1, n_dim) for rp in world_db.rectangle_pos.values]
    rect_size = [rs.reshape(-1, n_dim) for rs in world_db.rectangle_size.values]

    s = np.array([rp.shape[0] for rp in rect_pos])
    max_size = 30  # s.max()  # 30

    rect_pos, rect_size = randrect.rect_lists2arr(rect_pos=rect_pos, rect_size=rect_size, n_rectangles_max=max_size)

    world_imgs = img_basic.compressed2img(img_cmp=world_db.obstacle_img_cmp.values, n_voxels=n_voxels[0], n_dim=n_dim)
    new_world_imgs = randrect.rectangles2image(n_voxels=n_voxels, rect_pos=rect_pos, rect_size=rect_size,
                                               n_samples=n_worlds)

    np.allclose(world_imgs, new_world_imgs)


def sort_database(by=None, directory=None):
    # if by is None:
    by = ['i_world', 'i_sample']

    # directory = '3D/SR/3dof/'
    # directory = '2D/SR/2dof_eval10_D/'  # -> Fine, semi
    # directory = '2D/FB/2dof_eval10/'    # -> Fine
    # directory = '2D/FB/3dof_eval10'     # -> Fine
    # directory = '3D/SR/3dof_eval10'     # -> Fine
    # directory = '3D/FB/7dof_eval10'     # -> Fine
    # directory = '3D/FB/7dof_Justin_0'
    # directory = '2D/FB/3dof/'
    # directory = '2D/FA/3dof/'
    # directory = '3D/FB/7dof_Justin/'
    directory = get_sample_dir(directory=directory)
    file = directory + PATH_DB
    df = get_values_sql(file=file)
    df = df.iloc[21 * 10000:]

    df.sort_values(by, inplace=True)

    con = sql.connect(file)
    df.to_sql(name='db', con=con, if_exists='replace', idx=False)
    con.close()

    a = df.i_world.values
    print(len(a))
    import matplotlib.pyplot as plt
    plt.plot(a)

    # x = np.zeros(5000)
    # for o in range(5000):
    #     x[o] = np.sum(a == o)
    # plt.hist(x, bins=1000)


def drop_till(i_world=-1, n=1000, directrory=None):
    """
    Small helper to ensure all worlds have exactly 'n' paths.
    Not of big importance, but makes the whole procedure clearer and cleaner.
    """
    n = 5
    directory = '2D/2dof/'
    # directory = '2D/FB/3dof/'
    # directory = '2D/FB/3dof_5/'
    # directory = '2D/FB3/2dof_final1/'
    directory = get_sample_dir(directory=directory)

    file = directory + PATH_DB

    i_world_list = arg_wrapper__i_world(i_world, directory=directory)

    counts = get_n_samples(file=file, i_worlds=i_world_list)

    df = get_values_sql(file=file)
    by = ['i_world', 'i_sample']
    df.sort_values(by, inplace=True)

    keep_idx = np.ones(len(df), dtype=bool)

    for i, iw in enumerate(i_world_list):
        if counts[i] <= n:
            continue

        b = df.i_world.values == iw
        bb = np.nonzero(b)[0]
        bb = bb[n:]
        keep_idx[bb] = False

    df = df[keep_idx]
    with open_db_connection(file=file) as con:
        df.to_sql(name='db', con=con, if_exists='replace', idx=False)


def combine_df():
    directory_a = '3D/FB/7dof_Justin_0123/'
    directory_b = '3D/FB/7dof_Justin_456789/'
    # from Util.Loading.dlr import copy_userstore2homelocal
    # copy_userstore2local(directory_a)
    # copy_userstore2local(directory_b)

    directory_ab = '3D/FB/7dof_Justin/'

    file_a = directory_a + PATH_DB
    file_b = directory_b + PATH_DB
    file_ab = directory_ab + PATH_DB

    world_df_a = load_pd.load_world_df(file=directory_a)
    world_df_b = load_pd.load_world_df(file=directory_b)

    world_df_abc = world_df_a.append(world_df_b, ignore_idx=True)
    world_df_abc.to_pickle(path=get_sample_dir(directory_ab) + WORLD_DB)

    path_df_a = get_values_sql(file=file_a)
    path_df_b = get_values_sql(file=file_b)
    # import shutil
    # shutil.rmtree(d.arg_wrapper__sample_dir(directory_a))
    # shutil.rmtree(d.arg_wrapper__sample_dir(directory_b))

    n_worlds_a = path_df_a.i_world.values.max() + 1
    n_worlds_b = path_df_b.i_world.values.max() + 1

    path_df_b.i_world = path_df_b.i_world + n_worlds_a

    path_df_a = path_df_a.append(path_df_b)
    del path_df_b

    print('Total Number of RandomRectangles: {}'.format(n_worlds_a + n_worlds_b))
    with open_db_connection(file=file_ab) as con:
        path_df_a.to_sql(name='db', con=con, if_exists='replace', idx=False)


def split_df():
    directory_abc = '3D/FB/7dof_Justin_ABC/'
    directory_i = "3D/FB/Justin_{}/"

    file_abc = directory_abc + PATH_DB
    file_i = directory_i + PATH_DB

    path_df_abc = get_values_sql(file=file_abc)
    world_df_abc = load_pd.load_world_df(file=directory_abc)

    n_splits = 10
    i_worlds = arg_wrapper__i_world(-1, directory=directory_abc)
    n_worlds = len(i_worlds)

    i_splits = np.arange(start=0, stop=n_worlds + 1, step=n_worlds // n_splits)

    for i in range(n_splits):
        bool_i_lower = path_df_abc.i_world >= i_splits[i]
        bool_i_upper = path_df_abc.i_world < i_splits[i + 1]
        bool_i = np.logical_and(bool_i_lower, bool_i_upper)

        path_df_i = path_df_abc.loc[bool_i, :]
        path_df_i.i_world = path_df_i.i_world - i_splits[i]

        world_df_i = world_df_abc.iloc[i_splits[i]:i_splits[i + 1]]
        world_df_i.reset_idx(drop=True, inplace=True)

        with open_db_connection(file=file_i.format(i)) as con:
            path_df_i.to_sql(name='db', con=con, if_exists='replace', idx=False)

        world_df_i.to_pickle(path=get_sample_dir(directory_i.format(i)) + WORLD_DB)

    # import matplotlib.pyplot as plt
    # plt.plot(iwi)


def change2new():
    from Util.Loading.load_sql import get_values_sql, df2sql, get_table_name, rename_table
    file_world = 'RobotPathData/world.db'
    file_path = 'RobotPathData/path.db'
    world = get_values_sql(file=file_world)
    # path = get_values_sql(file=file_path)

    world.drop(inplace=True, columns=['obst_img_latent', 'edt_img_cmp'])
    world.rename(columns={'rectangle_position': 'rectangle_pos'}, inplace=True)
    world.head()

    rename_table(file=file_path, tables='paths')
    get_table_name(file=file_path)
    df2sql(df=world, file=file_path, table_name='worlds', if_exists='replace')

    path = get_values_sql(file=file_path, table='paths', rows=np.arange(10))


def change2new_path_db(file):
    # Different ordering of x_flat
    # Remove base for fixed robots
    # change limits from [0. 2*np.pi] to [-np.pi, np.pi]
    # rename table

    ###
    file = DLR_HOMELOCAL_DATA_SAMPLES + '2D/SA/3dof/path.db'
    n_waypoints = 22
    n_dim = 2
    n_dof = 3
    ###

    rename_table(file=file, new_name='db')

    path_db = get_values_sql(file=file)
    q_path = object2numeric_array(path_db.x_path.values)
    q_start = object2numeric_array(path_db.x_start.values)
    q_end = object2numeric_array(path_db.x_end.values)

    # Remove fixed_base
    q_start = q_start[:, n_dim:]
    q_end = q_end[:, n_dim:]

    # Change x_flat
    q_path = q_path.reshape(-1, n_dim + n_dof, n_waypoints).transpose(0, 2, 1)
    q_path = q_path[:, :, n_dim:]
    q_path = path.x2x_flat(x=q_path)

    # Check infinity joints:
    # n_test = 100000
    # q_path = path.x_flat2x(x_flat=q_path, n_wp=n_waypoints, n_dof=n_dof)
    # idx = np.nonzero(np.abs(np.diff(q_path[:n_test], axis=-2)) > 4)
    # print(idx)
    # q_path[idx[0][0]]

    # Change joint limits
    q_start = path.inf_joint_wrapper(q_start[..., np.newaxis], inf_bool=np.ones(1, dtype=bool))[..., 0]
    q_end = path.inf_joint_wrapper(q_end[..., np.newaxis], inf_bool=np.ones(1, dtype=bool))[..., 0]
    q_path = path.inf_joint_wrapper(q_path[..., np.newaxis], inf_bool=np.ones(1, dtype=bool))[..., 0]

    # Change 100 -> 10
    # q_start /= 10
    # q_end /= 10
    # q_path /= 10

    # Drop unnecessary columns
    path_db.drop(columns=['r_sphere', 'n_waypoints', 'objective'],  # 'x_pred', 'objective_pred'],
                 axis=1, inplace=True)

    # Rename columns
    path_db.rename(columns={'x_start': START_Q,
                            'x_end': END_Q,
                            'x_path': PATH_Q}, inplace=True)

    path_db[START_Q] = numeric2object_array(q_start)
    path_db[END_Q] = numeric2object_array(q_end)
    path_db[PATH_Q] = numeric2object_array(q_path)

    os.remove(file)
    with open_db_connection(file=file) as con:
        path_db.to_sql(name='db', con=con, if_exists='replace', index=False)


def change2world_size(old_s=100, new_s=10):
    file = DLR_HOMELOCAL_DATA_SAMPLES + '2D/SR/2dof/path.db'
    path_db = get_values_sql(file=file)
    x_start = object2numeric_array(path_db.x_start.values)
    x_end = object2numeric_array(path_db.x_end.values)
    x_path = object2numeric_array(path_db.x_path.values)
    x_start *= new_s / old_s
    x_end *= new_s / old_s
    x_path *= new_s / old_s

    path_db[START_Q] = numeric2object_array(x_start)
    path_db[END_Q] = numeric2object_array(x_end)
    path_db[PATH_Q] = numeric2object_array(x_path)
    with open_db_connection(file=file) as con:
        path_db.to_sql(name='db', con=con, if_exists='replace', idx=False)


def world_pickle2df(directory):
    # 2D/SR/2dof/', 100, 64, 22, 3
    # 2D/SA/2dof/', 100, 64, 22, 3 world_df['fixed_base'] (array([32, 32]), 21)
    #  lll: [array([5.19615242, 5.19615242, 5.19615242, 5.19615242]),
    #        array([5.19615242, 5.19615242, 5.19615242, 5.19615242])]
    # 2D/SA/3dof/', shape: 100, n_voxels: 64, radius: 3 , min_max: [3, 7], world_df['fixed_base'] (array([32, 32]), 21)
    # lll: [array([5.19615242, 5.19615242, 5.19615242]),
    #       array([5.19615242, 5.19615242, 5.19615242]),
    #       array([5.19615242, 5.19615242, 5.19615242])]

    directory = DLR_HOMELOCAL_DATA_SAMPLES + '2D/SA/3dof/'
    print('Changed x_flat representation!')
    world_df = load_pd.load_world_df(file=directory + WORLD_PKL_OLD, decompress_edt=False)
    world_df = load_pd.add_obstacle_img_column(world_df, values_only=False, verbose=0)

    world_df.rectangle_pos = [np.array(v, dtype=int).flatten() for v in world_df.rectangle_pos.values]
    world_df.rectangle_size = [np.array(v, dtype=int).flatten() for v in world_df.rectangle_size.values]

    world_df.rename(columns={'edt_img': 'edt_img_cmp'}, inplace=True)
    world_df.rename(columns={'obstacle_img': 'obstacle_img_cmp'}, inplace=True)

    world_dict = {'shape': world_df['world_size'].values[0],
                  'n_voxels': world_df['n_voxels'].values[0],
                  'min_max_obstacle_size': world_df['min_max_obstacle_size'].values[0]}
    if world_df['fixed_base'].values[0] == (None, None):
        world_dict['opening_pos'] = []
        world_dict['opening_size'] = []

    else:
        raise NotImplementedError

    save_pickle(obj=world_dict, file=directory + 'world_info')

    world_df.drop(columns=['world_size', 'n_voxels', 'min_max_obstacle_size', 'r_agent_max', 'lll', 'fixed_base'],
                  axis=1, inplace=True)

    world_df[obstacle_img_CMP] = np.array([img_basic.img2compressed(img=img) for
                                       img in world_df[obstacle_img_CMP].values], dtype=object)

    df2sql(df=world_df, file=directory + 'world2.db', if_exists='append')


def initialize_x_pred(file):
    if 'x_pred' in get_values_sql(file=file, rows=0).columns:
        return

    print('Initialize x_pred')
    path_df = get_values_sql(file=file)
    path_df.loc[:, 'x_pred'] = path_df.loc[:, PATH_Q].values

    with open_db_connection(file=file, close=True, lock=None) as con:
        path_df.to_sql(name='db', con=con, if_exists='replace', idx=False)


def add_latent_obstacle_representation(file, net):
    world_df = get_values_sql(file=file)

    obstacle_img = img_basic.compressed2img(img_cmp=world_df[obstacle_img_CMP].values, n_voxels=64, n_dim=2,
                                            n_channels=1, dtype=bool)

    latent_var = net.predict(obstacle_img).astype(float)
    latent_var = numeric2object_array(latent_var)
    world_df[obstacle_img_LATENT] = latent_var

    with open_db_connection(file=file) as con:
        world_df.to_sql(name='db', con=con, if_exists='replace', idx=False)


def add_relative_path_length(file):
    path_df = get_values_sql(file=file)

    x_path = object2numeric_array(path_df.x_path.values)
    x_path = x_path.reshape(-1, 22, 2)

    path_length = path.path_length(x=x_path)
    direct_path_length = path.linear_distance(x=x_path)
    relative_path_length = path_length / direct_path_length

    from wzk import new_fig
    fig, ax = new_fig()
    ax.hist(relative_path_length, bins=200)
    path_df['relative_path_length'] = relative_path_length

    with open_db_connection(file=file) as con:
        path_df.to_sql(name='db', con=con, if_exists='replace', idx=False)


def add_x_path_equidistant():
    from scipy.interpolate import splprep, splev
    import Util.Visualization.plotting_2 as plt2

    verbose = 0
    file = DLR_HOMELOCAL_DATA_SAMPLES + '2D/SR/2dof/path.db'
    path_db = get_values_sql(file=file, rows=np.arange(20000))

    n_samples = len(path_db)
    x_path = object2numeric_array(path_db.x_path.values)

    xx = path.x_flat2x(x_flat=x_path[123][None], n_dof=2)[0]

    tck, u = splprep(xx.T, s=0, k=3)
    u = np.linspace(0.0, 1.0, 1000)
    xx2 = np.column_stack(splev(x=u, tck=tck))
    fig, ax = plt2.new_world_fig(limits=10, title='Interpolation')
    plt2.plot_x_path(x=xx, ax=ax, c='r')
    plt2.plot_x_path(x=xx2[::20], ax=ax, c='k', marker='x')

    path.path_length(xx)
    path.path_length(xx2)

    # path length
    path_length = path.path_length(x_path)
    path_length.min()
    path_length.max()
    path_length.mean()
    x_forwardew = np.zeros((n_samples, 51, 2))

    fixed_step_length = 0.3
    for i in range(n_samples):
        print_progress(iteration=i, total=n_samples)
        tck, u = splprep(x_path[i].T, s=0, k=3)
        u = np.linspace(0.0, 1.0, 1000)
        x_path2 = np.column_stack(splev(x=u, tck=tck))
        mean_step_length = path_length[i] / 1000
        n_small_steps = int(np.round(fixed_step_length / mean_step_length))
        x_path3 = x_path2[::n_small_steps]

        x_forwardew[i, :x_path3.shape[-2], :] = x_path3
        if verbose >= 1:
            _, ax = plt2.new_world_fig(world_size=10, title='Interpolation')
            plt2.plot_x_path(ax=ax, x=x_path[i], c='r')
            plt2.plot_x_path(ax=ax, x=x_path2, c='b')
            plt2.plot_x_path(ax=ax, x=x_path3, c='k')

    path_db['x_path_eq'] = numeric2object_array(x_forwardew)

    with open_db_connection(file=file) as con:
        path_db.to_sql(name='db', con=con, if_exists='replace', idx=False)
