import os.path

import numpy as np
import pandas.io.sql

from wzk.sql2 import get_values_sql, set_values_sql, df2sql, vacuum, get_n_rows, delete_rows
from shutil import copy

from wzk import compressed2img, find_largest_consecutives, object2numeric_array, squeeze, tictoc
from wzk.mpl import new_fig

from rokin.Robots import *

# from mopla.main import objective_feasibility
from mopla.Optimizer import choose_optimum
from mopla.Optimizer.gradient_descent import gd_chomp


from mogen.Generation.parameter import init_par
from mogen.Loading.load import create_path_df, create_world_df, get_samples, get_paths


def check_consistency(robot,
                      db_file=None,
                      q=None, f=None, o=None, img=None):

    gen = init_par(robot_id=robot.id)

    if q is None:
        i_w, i_s, q0, q, o, f = get_values_sql(file=db_file, table='paths',
                                               rows=i,
                                               columns=['world_i32', 'sample_i32', 'q0_f32', 'q_f32',
                                                        'objective_f32', 'feasible_f32'],
                                               values_only=True)

    if img is None:
        img_cmp = get_values_sql(file=db_file, rows=np.arange(300), table='worlds', columns='img_cmp', values_only=True)
        img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)

    q = q.reshape(-1, gen.par.n_waypoints, robot.n_dof).copy()

    o_label, f_label = objective_feasibility(q=q, imgs=img, par=gen.par, iw=i_w)
    print(f.sum(), f_label.sum(), (f == f_label).mean())


def check_iw_is(i_w, i_s, m):
    i_w = i_w.copy()
    i_w = i_w.reshape(-1, m)
    bw = np.unique(i_w, axis=1)
    assert bw.shape[1] == 1

    i_s = i_s.copy()
    i_s = i_s.reshape(-1, m)
    bs = np.unique(i_s, axis=1)
    assert bs.shape[1] == 1


def plot(file, i):
    # def plot_o_distributions(o):
    o, f = get_values_sql(file=file, table='paths', rows=i,
                          columns=['objective', 'feasible'], values_only=True)

    o = o.reshape(-1, 50)
    f = f.reshape(-1, 50)

    o_mean = np.mean(o, axis=0)
    o_med = np.median(o, axis=0)
    fig, ax = new_fig()
    ax.plot(np.arange(50), o_med, color='blue')
    ax.fill_between(np.arange(50), np.percentile(o, q=20, axis=0), np.percentile(o, q=80, axis=0), color='blue', alpha=0.5)

    print('mean feasibility', f.mean())
    fig, ax = new_fig()
    ax.plot(np.arange(50), f.mean(axis=0), color='blue')

    j = np.argmin(o, axis=-1)

    ju, jc = np.unique(j, return_counts=True)

    fig, ax = new_fig()
    ax.plot(ju, jc, color='blue')

    fig, ax = new_fig()
    ax.plot(ju, np.cumsum(jc)/jc.sum(), color='blue')

    # FINDING, the worlds for Justin where to easy, 80% of the time it possible to converge from a direct connection


def get_df_subset(file: str,
                   i: np.ndarray, i0: int = 0, n: int = 1,
                   file_new='', if_exists='append'):

    i_w, i_s, q0, q, o, f = get_paths(file=file, i=i)
    iw, i_s, q0, q, o, f = i_w[i], i_s[i], q0[i], q[i], o[i], f[i]

    i_s = i0 + np.arange(len(i_s)//n).repeat(n)

    q0 = object2numeric_array(q0)
    if q0.dtype == object:  # TODO do not let this happen by adjusting the generation
        q0 = np.full(len(i_s), -1)

    iw, i_s, q0, q, o, f = squeeze(iw, i_s, q0, q, o, f)

    if len(iw) == 0:
        return None

    df_i = create_path_df(i_world=iw, i_sample=i_s, q0=q0, q=q, objective=o, feasible=f)

    if file_new:
        assert isinstance(file_new, str)
        df2sql(df=df_i, file=file_new, table='paths', if_exists=if_exists)

    # print(f"{iw_i} | easy: {len(df_easy)} | hard: {len(df_hard)}")
    # df2sql(df=df_hard, file=file_hard, table='paths', if_exists=ra)
    #


    return df_i


def main_choose_best(robot_id: str):
    # gen = init_par(robot_id=robot_id)
    file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_sc_hard'
    # file = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot_id}_sc2_hard'

    file2 = f"{file}2.db"
    file = f"{file}.db"

    print(f"choose best multi start for {file}")
    copy(file, file2)

    m = 11  # TODO automate
    n = get_n_rows(file=file, table='paths')
    print('N_rows original:', n)
    i = np.arange(0, (n // m) * m)

    i_w, i_s, o, f = get_values_sql(file=file, table='paths', rows=i, columns=['world_i32', 'sample_i32',
                                                                               'objective_f32', 'feasible_b'],
                                    values_only=True)
    check_iw_is(i_w=i_w, i_s=i_s, m=m)

    o_max = o.max()
    o2 = o.copy()

    o2[f != 1] += o_max
    o2 = o2.reshape(-1, m)
    j = np.argmin(o2, axis=-1) + np.arange(len(o2))*m

    print('max j', j.max())

    f = np.squeeze(f)
    j = j[f[j] == 1]
    df = get_df_subset(file=file, i=j, n=1)

    print('N_rows new:', len(df))

    df2sql(df=df, file=file2, table='paths', if_exists='replace')
    vacuum(file2)


def separate_easy_hard(file, i):

    i_s = get_values_sql(file=file, table='paths', rows=i, columns=['sample_i32'], values_only=True)

    n, i_hard = find_largest_consecutives(x=i_s)
    if n == 1:
        i_hard = np.array([], dtype=int)
    else:
        i_hard = i_hard[:, np.newaxis] + np.arange(n)[np.newaxis, :]
        i_hard = i_hard.ravel()

    i_easy = np.arange(len(i_s))
    i_easy = np.delete(i_easy, i_hard)

    i_s_easy = np.arange(len(i_easy))
    i_s_hard = np.arange(len(i_hard) // n).repeat(n)

    return (i_easy, i_hard), (i_s_easy, i_s_hard)

    # i_easy = i0_easy + i_easy
    # i_hard = i0_hard + i_hard
    # df_easy = get_df_subset(file=file, file_new=file_easy, i=i_easy, n=1, i0=i0_easy)
    # df_hard = get_df_subset(file=file, file_new=file_hard, i=i_hard, n=n, i0=i0_hard)
    # return df_easy, df_hard


# def batch_loop:


def main_separate_easy_hard(file: str):

    file, _ = os.path.splitext(file)

    # file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}'
    # file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_sc'
    # file = f'/Users/jote/Documents/DLR/Data/mogen/{robot_id}_sc'

    file_easy = f"{file}_easy.db"
    file_hard = f"{file}_hard.db"
    file = f"{file}.db"
    table = 'paths'

    print(f"Separate {file} into easy and hard")
    print('copy initial file -> file_easy')
    copy(file, file_easy)

    n = get_n_rows(file=file, table=table)
    print(f"total: {n}")

    iw_all = get_values_sql(file=file, table='paths', rows=np.arange(n), columns=['world_i32'], values_only=True)
    iw_all = iw_all.astype(np.int32)
    i_s = np.full(n, -1)
    b_easy = np.zeros(n, dtype=bool)
    b_hard = np.zeros(n, dtype=bool)

    print(f"Separate indices")
    for iw_i in np.unique(iw_all):
        j = np.nonzero(iw_all == iw_i)[0]
        (i_easy, i_hard), (i_s_easy, i_s_hard) = separate_easy_hard(file=file, i=j)  # TODO how to combine with without high memory consumption

        j_easy = j[i_easy]
        j_hard = j[i_hard]
        i_s[j_easy] = i_s_easy
        i_s[j_hard] = i_s_hard
        b_easy[j_easy] = True
        b_hard[j_hard] = True
        print(f"World:{iw_i} | total: {j.size} | easy: {j_easy.size} | hard: {j_hard.size} ")

    assert not np.any(i_s == -1)
    assert np.allclose(b_easy, ~b_hard)

    set_values_sql(file=file_easy, table=table,
                   values=(i_s.astype(np.int32).tolist(),), columns='sample_i32')

    print('copy file_easy -> file_hard')
    copy(file_easy, file_hard)

    print('delete respective complementing rows in file_easy and file_hard')
    delete_rows(file=file_easy, table=table, rows=b_hard)
    delete_rows(file=file_hard, table=table, rows=b_easy)

    # set_values_sql(file=file_hard, table=table,
    #                values=(np.full(b_hard.sum(), -1, dtype=np.float32).tolist(),), columns='q0_f32')  #  )

    print(get_values_sql(file=file_hard, table=table))
    n_easy = get_n_rows(file=file_easy, table=table)
    n_hard = get_n_rows(file=file_hard, table=table)
    print(f"total: {n} | easy: {n_easy} | hard: {n_hard}")


def test_separate_easy_hard():
    import pandas as pd
    columns = ['world_i32', 'sample_i32', 'x']
    data = [[0, 0, 0],
            [0, 1, 1],
            [0, 2, 'a'],
            [0, 2, 2],
            [0, 2, 2],
            [0, 3, 3],
            [0, 4, 4],
            [0, 5, 'a'],
            [0, 5, 5],
            [0, 5, 5],
            [1, 0, 1],
            [1, 1, 2],
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [2, 0, 'a'],
            [2, 0, 2],
            [2, 0, 2],
            [2, 1, 'a'],
            [2, 1, 3],
            [2, 1, 3],
            [2, 2, 'a'],
            [2, 2, 4],
            [2, 2, 4]]

    df = pd.DataFrame(columns=columns, data=data, index=None)

    file = '/Users/jote/Documents/Code/Python/DLR/wzk/wzk/tests/dummy.db'
    table = 'paths'
    df2sql(df=df, file=file, table=table, if_exists='replace')

    main_separate_easy_hard(file=file)


if __name__ == '__main__':
    test_separate_easy_hard()
    # main_separate_easy_hard(robot_id='Justin19')
    # main_choose_best(robot_id='SingleSphere02')
    # main_choose_best(robot_id='Justin19')


