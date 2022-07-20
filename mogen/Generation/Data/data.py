import os
import numpy as np
import pandas as pd

from wzk.image import compressed2img, img2compressed
from wzk.numpy2 import squeeze_all
from wzk import sql2, safe_unify
from wzk.training import n2train_test, train_test_split  # noqa
from wzk.dlr import LOCATION
from wzk.gcp.gcloud2 import gsutil_cp

from mogen.Generation.Data.tables import T_PATHS, T_WORLDS, T_IKS, T_INFO

meta_df_columns = np.array(['par', 'gd'])


n_samples_per_world = 1000  # TODO get rid of this, its just unnecessary to adhere to this restriction


img_cmp0 = [img2compressed(img=np.zeros((64,), dtype=bool), n_dim=1),
            img2compressed(img=np.zeros((64, 64), dtype=bool), n_dim=2),
            img2compressed(img=np.zeros((64, 64, 64), dtype=bool), n_dim=3)]


def get_file_ik(robot_id, copy=True):
    __file_stub_ik_dlr = '/home_local/tenh_jo/ik_{}.db'
    __file_stub_ik_mac = '/Users/jote/Documents/DLR/Data/mogen/ik_{}/ik_{}.db'
    __file_stub_ik_gcp = '/home/johannes_tenhumberg_gmail_com/sdb/ik_{}.db'

    __file_stub_ik_dict = dict(dlr=__file_stub_ik_dlr, mac=__file_stub_ik_mac, gcp=__file_stub_ik_gcp)
    file_stub_ik = __file_stub_ik_dict[LOCATION]

    file = file_stub_ik.format(*[robot_id]*2)

    if copy and not os.path.exists(file):
        gsutil_cp(src=f'gs://tenh_jo/{robot_id}/ik_{robot_id}.db', dst=file)

    return file


def get_file(robot_id):
    __file_stub_dlr = '/home_local/tenh_jo/{}.db'
    __file_stub_mac = '/Users/jote/Documents/DLR/Data/mogen/{}/{}.db'
    __file_stub_gcp = '/home/johannes_tenhumberg_gmail_com/sdb/{}.db'

    __file_stub_dict = dict(dlr=__file_stub_dlr, mac=__file_stub_mac, gcp=__file_stub_gcp)
    __file_stub = __file_stub_dict[LOCATION]

    file = __file_stub.format(*[robot_id]*2)

    if not os.path.exists(file):
        gsutil_cp(src=f'gs://tenh_jo/{robot_id}/{robot_id}.db', dst=file)

    return file


def arg_wrapper__i_world(i_worlds, file=None):
    """
    Return a numpy array of world indices.
    - if i_worlds is an integer
    Helper function to handle the argument i_world.
    If i_world is an integer, wrap it in a list.
    If i_world is -1, use all available worlds.
    If i_world is already a list, leave it untouched.
    """

    if isinstance(i_worlds, (int, np.int32, np.int64)):
        i_worlds = int(i_worlds)
        if i_worlds == -1:
            n_worlds = sql2.get_n_rows(file=file, table='worlds')
            i_worlds = list(range(n_worlds))
        else:
            i_worlds = [i_worlds]

    return np.array(i_worlds)


def get_sample_indices(i_worlds=-1, file=None, validation_split=0.2, validation_split_random=True):
    """
    Calculates the indices for training and testing.
    Splits the indices for whole worlds, so the test set is more difficult/ closer to reality.
    """

    i_world_list = arg_wrapper__i_world(i_worlds, file=file)

    # Split the samples for training and validation, separate the two sets by worlds -> test set contains unseen worlds
    if validation_split_random:
        np.random.shuffle(i_world_list)
    sample_indices = np.array([get_i_samples_world(iw, file=file) for iw in i_world_list]).flatten()

    # Divide in trainings and test Measurements
    n_worlds = len(i_world_list)
    n_samples = n_worlds * n_samples_per_world

    if validation_split is not None and validation_split > 0.0:
        idx_split = int(validation_split * n_samples)
        idx_split -= idx_split % n_samples_per_world
        sample_indices_training = sample_indices[:n_samples - idx_split]
        sample_indices_test = sample_indices[n_samples - idx_split:]

        return sample_indices_training, sample_indices_test

    else:
        return sample_indices, 0


def sort_sample_indices(sample_indices):
    """
    Make a sorted copy of the sample indices, sort with respect to the world

    5000, 5001, 5002, ..., 5999
    0000, 0001, 0002, ..., 0999
    3000, 3001, 3002, ..., 3999

    ->

    0000, 0001, 0002, ..., 0999
    1000, 1001, 1002, ..., 1999
    2000, 2001, 2002, ..., 2999
    ...
    """

    n_worlds = sample_indices.size // n_samples_per_world
    sample_indices0 = sample_indices.reshape((n_worlds, n_samples_per_world)).copy()
    idx_worlds0 = get_i_world(sample_indices0[:, 0])
    sample_indices0 = sample_indices0[np.argsort(idx_worlds0), :]

    return sample_indices0


# i_worlds <-> i_samples
def get_i_world(i_sample_global):
    """
    Get the world-indices corresponding to the global samples.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    if isinstance(i_sample_global, (list, tuple)):
        i_sample_global = np.array(i_sample_global)

    return i_sample_global // n_samples_per_world


def get_i_samples_world(i_worlds, file):
    """
    Get the global sample-indices corresponding to the world numbers.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    i_world_list = arg_wrapper__i_world(i_worlds=i_worlds, file=file)

    if np.size(i_world_list) == 1:
        return np.arange(n_samples_per_world * i_world_list[0], n_samples_per_world * (i_world_list[0] + 1))
    else:
        return np.array([get_i_samples_world(iw, file=file) for iw in i_world_list])


def get_i_samples_global(i_worlds, i_samples_local):
    """
    Get the global sample-indices corresponding to the world numbers.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    if i_samples_local is None or i_worlds is None:
        return None
    if np.size(i_samples_local) / np.size(i_worlds) % n_samples_per_world == 0:
        i_worlds = np.repeat(i_worlds, n_samples_per_world)

    return i_samples_local + i_worlds * n_samples_per_world


def get_i_samples_local(i_sample_global):
    return i_sample_global % n_samples_per_world


# Wrapper and Helper
#  Switching dimensions between net-arguments and mpl
def df_set_column_lists(df, column, ll):
    n_samples = len(df)
    df.loc[:, column] = 0
    df.loc[:, column] = df.loc[:, column].astype(object)
    for i in range(n_samples):
        df.at[i, column] = ll[i]

    return df


def initialize_df():
    return pd.DataFrame(data=None)


def create_world_df(i_world: np.ndarray, img_cmp: np.ndarray):
    data = {key: value for key, value in zip(T_WORLDS.names(), [i_world, img_cmp])}
    print(data)
    return pd.DataFrame(data)


def create_path_df(i_world, i_sample, q, objective, feasible) -> pd.DataFrame:
    data = {key: value for key, value in zip(T_PATHS.names(), [i_world, i_sample, q, objective, feasible])}
    print(data)
    return pd.DataFrame(data)


def create_ik_df(i_world, i_sample, q, f, objective, feasible) -> pd.DataFrame:
    data = {key: value for key, value in zip(T_IKS.names(), [i_world, i_sample, q, f, objective, feasible])}
    print(data)
    return pd.DataFrame(data)


def rename_old_columns(file):
    # TODO not forever needed
    # columns = dict(i_world='world_i64',
    #                i_sample='sample_i64',
    #                q0='q0_f64',
    #                q='q_f64',
    #                objective='objective_f64',
    #                feasible='feasible_b')

    # columns_paths = dict(world_i32='world_i32',
    #                      sample_i32='sample_i32',
    #                      q_f32='q_f32',
    #                      objective_f32='objective_f32',
    #                      feasible_b='feasible_b')

    columns_paths = dict(world_i='world_i32',
                         sample_i='sample_i32',
                         q_f32='q_f32',
                         objective_f='objective_f32',
                         feasible_i='feasible_b')

    columns_worlds = dict(world_i='world_i32',
                          img_cmp='img_cmp')

    sql2.rename_columns(file=file, table='paths', columns=columns_paths)
    sql2.rename_columns(file=file, table='worlds', columns=columns_worlds)
    print('tables renamed!')


def iw2is_wrapper(iw, iw_all):
    i = np.nonzero(iw_all == iw)[0]
    return i


def get_samples_for_world(file, par, i=None, i_w=None):
    if i_w is not None:
        i_w, i_w_all = i_w
        i = iw2is_wrapper(iw=i_w, iw_all=i_w_all)

    i_w, i_s, q, o, f = get_paths(file, i)
    q = q.reshape((len(i), par.n_wp, par.robot.n_dof))
    i_w = safe_unify(i_w)
    img = get_worlds(file=file, i_w=i_w, img_shape=par.world.shape)

    q_start, q_end = q[..., 0, :], q[..., -1, :]
    par.q_start, par.q_end = q_start, q_end
    par.update_oc(img=img)
    return i, q, img


def get_paths(file, i):
    i_w, i_s, q, o, f = sql2.get_values_sql(file=file, table=T_PATHS(), rows=i,
                                            columns=[c.name for c in T_PATHS.cols])
    i_w, i_s, o, f = squeeze_all(i_w, i_s, o, f)
    return i_w, i_s, q, o, f


def get_worlds(file, i_w, img_shape):
    img_cmp = sql2.get_values_sql(file=file, table=T_WORLDS(), rows=i_w, columns=T_WORLDS.C_IMG_CMP())
    img = compressed2img(img_cmp=img_cmp, shape=img_shape, dtype=bool)
    return img


def get_samples(file, i, img_shape):
    i_w, i_s, q, o, f = get_paths(file, i)
    img = get_worlds(file=file, i_w=i_w, img_shape=img_shape)
    return i_w, i_s, q, img


def combine_df_list(df_list):
    try:
        df = df_list[0]
    except IndexError:
        return None

    for df_i in df_list[1:]:
        df = pd.concat([df, df_i])
    return df


def create_info_table(file):
    robot_id = os.path.splitext(os.path.split(file)[1])[0]

    i_w0 = sql2.get_values_sql(file=file, table=T_WORLDS(), rows=-1, columns=[T_WORLDS.C_WORLD_I()])
    i_w, i_s = sql2.get_values_sql(file=file, table=T_PATHS(), rows=-1, columns=[T_PATHS.C_WORLD_I(), T_PATHS.C_SAMPLE_I()])
    n_samples = len(i_s)
    n_worlds = len(i_w0)

    n_samples_per_world = np.zeros(n_worlds, dtype=int)
    for iwi in range(n_worlds):
        n_samples_per_world[iwi] = np.sum(i_w == iwi)

    assert np.allclose(i_w0, np.arange(n_worlds))
    assert n_samples == np.sum(n_samples_per_world)

    data = {key: value for key, value in zip(T_INFO.cols.names(),
                                             [[robot_id], [n_worlds], [n_samples], [n_samples_per_world]])}
    data = pd.DataFrame(data)
    sql2.df2sql(df=data, file=file, table=T_INFO(), dtype=T_INFO.types_sql(), if_exists='replace')
    print('Created Info Table')


def get_info_table(file):
    robot_id, n_worlds, n_samples, n_samples_per_world = \
        sql2.get_values_sql(file=file, table=T_INFO(), squeeze_row=True,
                            columns=[T_INFO.C_ROBOT_ID(), T_INFO.C_N_WORLDS(), T_INFO.C_N_SAMPLES(), T_INFO.C_N_SAMPLES_PER_WORLD()])

    return robot_id, n_worlds, n_samples, n_samples_per_world


if __name__ == '__main__':
    _file = '/Users/jote/Documents/DLR/Data/mogen/SingleSphere02/SingleSphere02.db'
    # _file = '/home_local/tenh_jo/SingleSphere02.db'

    # rename_old_columns(_file)
    create_info_table(_file)
