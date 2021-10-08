import numpy as np
import pandas as pd

meta_df_columns = np.array(['par', 'gd'])
world_df_columns = np.array(['i_world', 'img_cmp'])
path_df_columns = np.array(['i_world', 'i_sample', 'q0', 'q', 'objective', 'feasible'])


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


def create_world_df(i_world, img_cmp):
    data = {key: value for key, value in zip(world_df_columns, [i_world, img_cmp])}
    return pd.DataFrame(data)


def create_path_df(i_world, i_sample, q0, q, objective, feasible):
    q0 = [qq0.tobytes() for qq0 in q0]
    q = [qq.tobytes() for qq in q]

    data = {key: value for key, value in zip(path_df_columns, [i_world, i_sample, q0, q, objective, feasible])}
    return pd.DataFrame(data)
