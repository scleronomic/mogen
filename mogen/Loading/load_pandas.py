import numpy as np
import pandas as pd

from wzk.dtypes import str2np

meta_df_columns = np.array(['par', 'gd'])
world_df_columns = np.array(['world', 'img_cmp'])
path_df_columns = np.array(['world_i32', 'sample_i32', 'q0_f32', 'q_f32', 'objective_f32', 'feasible_b'])


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
