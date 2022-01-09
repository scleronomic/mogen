import numpy as np
import pandas as pd

from wzk.dtypes import str2np
from wzk.image import compressed2img

from wzk.sql2 import get_n_rows, rename_columns, get_values_sql, df2sql  # noqa
from wzk.training import n2train_test, train_test_split  # noqa

meta_df_columns = np.array(['par', 'gd'])
world_df_columns = np.array(['world_i32', 'img_cmp'])
path_df_columns = np.array(['world_i32', 'sample_i32', 'q0_f32', 'q_f32', 'objective_f32', 'feasible_b'])


n_samples_per_world = 1000


def arg_wrapper__i_world(i_worlds, file=None):
    """
    Return an numpy array of world indices.
    - if i_worlds is an integer
    Helper function to handle the argument i_world.
    If i_world is an integer, wrap it in a list.
    If i_world is -1, use all available worlds.
    If i_world is already a list, leave it untouched.
    """

    if isinstance(i_worlds, (int, np.int32, np.int64)):
        i_worlds = int(i_worlds)
        if i_worlds == -1:
            n_worlds = get_n_rows(file=file, table='worlds')
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
    Make a sorted copy of the the sample indices, sort with respect to the world

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
    data = {key: value for key, value in zip(world_df_columns, [i_world, img_cmp])}
    data = prepare_data(data)
    return pd.DataFrame(data)


def create_path_df(i_world: np.ndarray, i_sample: np.ndarray,
                   q0: np.ndarray, q: np.ndarray,
                   objective: np.ndarray, feasible: np.ndarray) -> pd.DataFrame:
    data = {key: value for key, value in zip(path_df_columns, [i_world, i_sample, q0, q, objective, feasible])}
    data = prepare_data(data=data)
    return pd.DataFrame(data)


def prepare_data(data: dict) -> dict:
    for key in data:
        data[key] = data[key].astype(str2np(key))

        if np.size(data[key][0]) > 0 and not isinstance(data[key][0], bytes):
            data[key] = [xx.tobytes() for xx in data[key]]

    return data


def rename_old_columns(file):
    # TODO not for ever needed
    columns = dict(i_world='world_i64',
                   i_sample='sample_i64',
                   q0='q0_f64',
                   q='q_f64',
                   objective='objective_f64',
                   feasible='feasible_b')

    rename_columns(file=file, table='paths', columns=columns)


def get_sample(file, i_s, img_shape):
    i_w, i_s, q = get_values_sql(file=file, table='paths',
                                 rows=i_s, columns=['world_i32', 'sample_i32', 'q_f32'],
                                 values_only=True)
    i_w = int(np.squeeze(i_w))
    i_s = int(np.squeeze(i_s))
    img_cmp = get_values_sql(file=file, rows=i_w, table='worlds', columns='img_cmp', values_only=True)
    img = compressed2img(img_cmp=img_cmp, shape=img_shape, dtype=bool)

    return i_w, i_s, q, img
