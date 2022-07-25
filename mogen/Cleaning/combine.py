import os.path
from shutil import copy

import numpy as np
import fire

from wzk import sql2
from wzk import find_largest_consecutives, squeeze_all
from wzk.gcp import gcloud2
from wzk.mpl import new_fig

from mogen.Generation import data
from mogen.Cleaning import clean


def check_iw_is(i_w, i_s, m):
    print('m', m)
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
    o, f = sql2.get_values_sql(file=file, table=data.T_PATHS(), rows=i,
                               columns=[data.T_PATHS.C_OBJECTIVE_F(), data.T_PATHS.C_FEASIBLE_I()])

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


def combine_files(old_files, new_file, clean_s0, table):
    new_file_dir = os.path.split(new_file)[0]
    
    for i, f in enumerate(old_files):
        print(i)
        if f.startswith('gs://'):
            f2 = f"{new_file_dir}/{os.path.split(f)[1]}"
            gcloud2.gsutil_cp(src=f, dst=f2)
            f = f2

        if clean_s0:
            delete_not_s0(file=f)

        if i == 0:
            os.rename(f, new_file)

        else:
            if os.path.exists(f):
                print(f"Concatenate {f}")
                sql2.concatenate_tables(file=new_file, table=table, file2=f, table2=table)
                print(f"Remove {f}")
                os.remove(f)
            else:
                print(f"File {f} does not exist -> skip")
        n = sql2.get_n_rows(file=new_file, table=table)
        print(f"Total number of rows: {n}")

    if old_files[0].startswith('gs://'):
        dst = f"{os.path.split(old_files[0])[0]}/{os.path.split(new_file[1])[0]}"
        gcloud2.gsutil_cp(src=new_file, dst=dst)


def separate_easy_hard(file, i):

    i_s, q = sql2.get_values_sql(file=file, table=data.T_PATHS.names(), rows=i,
                                 columns=[data.T_PATHS.C_SAMPLE_I(), data.T_PATHS.C_Q_F()])
    q0 = q[:, 0]
    xu = q0 + i_s * np.random.random()
    n, i_hard = find_largest_consecutives(x=xu)
    print('n_consecutives', n)

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


def delete_not_s0(file):
    table = data.T_PATHS()
    w, s = sql2.get_values_sql(file=file, table=table, rows=-1,
                               columns=[data.T_PATHS.C_WORLD_I(), data.T_PATHS.C_SAMPLE_I()])

    s_not0 = np.nonzero(s != 0)[0]

    if np.size(s_not0) > 0:
        i = s_not0[-1]
        assert i < 10000
        sql2.delete_rows(file=file, table=table,  rows=np.arange(i+1))


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
    sql2.df2sql(df=df, file=file, table=table, if_exists='replace')

    main_separate_easy_hard(file=file)


def main_choose_best(file):

    file = os.path.splitext(file)[0]
    file2 = f"{file}2.db"
    file = f"{file}.db"
    table = 'paths'

    print(f"Choose best multi start for {file}")

    print('Copy initial file -> file2')
    copy(file, file2)

    print('Load data')
    i_w, i_s, o, f = sql2.get_values_sql(file=file2, table=table, rows=-1,
                                         columns=['world_i32', 'sample_i32', 'objective_f32', 'feasible_b'],
                                         values_only=True)
    i_w, i_s, o, f = squeeze_all(i_w, i_s, o, f)

    m, _ = find_largest_consecutives(i_s)
    m = 11  # TODO
    check_iw_is(i_w=i_w, i_s=i_s, m=m)

    o_max = o.max()
    o2 = o.copy()

    o2[f != 1] += o_max
    o2 = o2.reshape(-1, m)
    j = np.argmin(o2, axis=-1) + np.arange(len(o2))*m

    j = j[f[j] == 1]

    n_old = len(i_w)
    n_new = len(j)

    print('Delete worst & infeasible paths')
    j_delete = np.delete(np.arange(n_old), j)
    sql2.delete_rows(file=file2, table=table, rows=j_delete)

    clean.reset_sample_i32(file2)
    print(f"old {n_old} | tries per sample {m} -> old {n_old//m} | new {n_new}")


def main_separate_easy_hard(file: str):

    file, _ = os.path.splitext(file)

    file_easy = f"{file}_easy.db"
    file_hard = f"{file}_hard.db"
    file = f"{file}.db"
    table = data.T_PATHS()

    print(f"Separate {file} into easy and hard")
    print('Copy initial file -> file_easy')
    copy(file, file_easy)

    n = sql2.get_n_rows(file=file_easy, table=table)
    print(f"Total: {n}")

    print(f"Load all world indices")
    iw_all = sql2.get_values_sql(file=file_easy, table=data.T_PATHS(), rows=-1, columns=[data.T_PATHS.C_WORLD_I()],
                                 values_only=True)

    iw_all = iw_all.astype(np.int32)
    i_s = np.full(n, -1)
    b_easy = np.zeros(n, dtype=bool)
    b_hard = np.zeros(n, dtype=bool)

    print(f"Separate indices")
    for iw_i in np.unique(iw_all):
        j = np.nonzero(iw_all == iw_i)[0]
        (i_easy, i_hard), (i_s_easy, i_s_hard) = separate_easy_hard(file=file_easy, i=j)

        j_easy = j[i_easy]
        j_hard = j[i_hard]
        i_s[j_easy] = i_s_easy
        i_s[j_hard] = i_s_hard
        b_easy[j_easy] = True
        b_hard[j_hard] = True
        print(f"World:{iw_i} | total: {j.size} | easy: {j_easy.size} | hard: {j_hard.size} ")
    #
    assert not np.any(i_s == -1)
    assert np.allclose(b_easy, ~b_hard)

    print('Set new indices')
    sql2.set_values_sql(file=file_easy, table=table, values=(i_s.astype(np.int32).tolist(),),
                        columns=data.T_PATHS.C_SAMPLE_I())
    print('Copy file_easy -> file_hard')
    copy(file_easy, file_hard)

    print('Delete respective complementing rows in file_easy and file_hard')
    sql2.delete_rows(file=file_easy, table=table, rows=b_hard)
    sql2.delete_rows(file=file_hard, table=table, rows=b_easy)

    n_easy = sql2.get_n_rows(file=file_easy, table=table)
    n_hard = sql2.get_n_rows(file=file_hard, table=table)
    print(f"total: {n} | easy: {n_easy} | hard: {n_hard}")
    clean.reset_sample_i32(file=file_easy)


def main_combine_files(robot_id, i, table=data.T_PATHS(), prefix=''):
    prefix = f'{prefix}_' if prefix else ''
    if isinstance(i, str):
        i = eval(i)

    i = list(i)

    print(i)

    old_files = [f"gs://tenh_jo/{prefix}{robot_id}/{prefix}{robot_id}_{ii}.db" for ii in i]
    new_file = f"{prefix}{robot_id}_combined_{i[0]}-{i[-1]+1}.db"
    new_file = f"/home/johannes_tenhumberg_gmail_com/sdb/{new_file}"
    combine_files(old_files=old_files, new_file=new_file, clean_s0=False, table=table)


def main_combine_files_hard2():
    old_files = ["gs://tenh_jo/Justin19_combined_0-20_hard2.db",
                 "gs://tenh_jo/Justin19_combined_20-40_hard2.db",
                 "gs://tenh_jo/Justin19_combined_40-60_hard2.db",
                 "gs://tenh_jo/Justin19_combined_60-80_hard2.db"]
    new_file = f"/home/johannes_tenhumberg_gmail_com/sdb/Justin19_combined_0-80_hard2.db"
    tabel = 'paths'
    combine_files(old_files=old_files, new_file=new_file, clean_s0=False, table=tabel)


def split_df(file):
    file = '/home/johannes_tenhumberg_gmail_com/sdb/Justin19_combined_0-40.db'
    i_s = sql2.get_values_sql(file=file, table=data.T_PATHS(), rows=-1,
                              columns=data.T_PATHS.C_SAMPLE_I(), values_only=True)

    i_s0 = np.nonzero(i_s == 0)[0]
    i_s00 = np.nonzero(i_s0[1:] != i_s0[:-1] + 1)[0] + 1
    i_center = i_s00[len(i_s00)//2]

    i_center = i_s0[i_center]
    print(i_center)


def delete_half():
    file = '/home/johannes_tenhumberg_gmail_com/sdb/Justin19_combined_20-40.db'
    i = 33897660
    n = 66006680
    # sql2.delete_rows(file=file, table='paths', rows=np.arange(0, i))
    # sql2.delete_rows(file=file, table='paths', rows=np.arange(i, n))


if __name__ == '__main__':
    # fire.Fire({
    #     'combine': main_combine_files,
    #     'separate': main_separate_easy_hard,
    #     'choose_best': main_choose_best,
    # })

    _file = "/home/johannes_tenhumberg_gmail_com/sdb/JustinArm07.db"
    main_separate_easy_hard(file=_file)

    # sql2.sort_table(file=_file_hard2, table='paths', order_by=['world_i32', 'sample_i32', 'ROWID'])
    # reset_sample_i32(file=_file_hard2)
    # gcloud2.gsutil_cp(src=f"gs://tenh_jo/{os.path.basename(_file_hard2)}.db", dst=f"{_file_hard2}.db")


    # robot_id = 'SingleSphere02'
    # # i = np.arange(60, 80)
    # # i = np.delete(i, 3)
    # main_combine_files(robot_id=robot_id, i=i)
    #
    # tic()
    # _file0 = f"{robot_id}"
    # _file_bucket = f"gs://tenh_jo/{_file0}"
    # _file = f"/home/johannes_tenhumberg_gmail_com/sdb/{_file0}"
    # _file = "/Users/jote/Documents/DLR/Data/mogen/SingleSphere02/SingleSphere02"
    # #
    # _file_easy = _file + '_easy'
    # _file_hard = _file + '_hard'
    # _file_hard2 = _file + '_hard2'
    # gcloud2.gsutil_cp(src=f"gs://tenh_jo/{os.path.basename(_file)}.db", dst=f"{_file}.db")
    #
    # _file_hard2 = "/home/johannes_tenhumberg_gmail_com/sdb/Justin19_combined_0-80_hard2"


    #
    # #
    # print('sort easy')
    # sql2.sort_table(file=_file_easy, table='paths', order_by=['world_i32', 'sample_i32', 'ROWID'])
    # print('sort hard')
    # sql2.sort_table(file=_file_hard, table='paths', order_by=['world_i32', 'sample_i32', 'ROWID'])

    #
    # print('upload easy and hard')
    # gcloud2.gsutil_cp(src=f"{_file_easy}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_easy)}.db")
    # gcloud2.gsutil_cp(src=f"{_file_hard}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_hard)}.db")
    #
    # main_choose_best(file=_file_hard)
    #
    # print('upload hard2')
    # gcloud2.gsutil_cp(src=f"{_file_hard2}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_hard2)}.db")
    # toc()

    # reset_sample_i32_0(file=_file)
    # sql2.copy_table(file=_file, table_src='paths', table_dst='paths2',
    #                 columns=['world_i32', 'sample_i32', 'q_f32', 'objective_f32', 'feasible_b'],
    #                 dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_REAL, sql2.TYPE_INTEGER])

    # sql2.vacuum(file=_file)
    # _file = '/Users/jote/Documents/DLR/Data/mogen/ik_Justin19/ik_Justin19.db'
    # sql2.alter_table(_file, table='paths', columns=['world_i32', 'sample_i32', 'q_f32', 'objective_f32', 'feasible_b'],
    #                  dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_REAL, sql2.TYPE_INTEGER])
    # sql2.squeeze_table(file=_file, table='paths')

    #