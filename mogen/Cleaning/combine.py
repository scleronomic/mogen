import os.path
import subprocess
from shutil import copy

import numpy as np

from wzk import sql2
from wzk import tic, toc, find_largest_consecutives, squeeze
from wzk.gcp import gcloud2
from wzk.mpl import new_fig


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
    o, f = sql2.get_values_sql(file=file, table='paths', rows=i,
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


def reset_sample_i32(file):
    print('Reset indices')
    table = 'paths'
    iw_all = sql2.get_values_sql(file=file, table=table, rows=-1, columns=['world_i32'], values_only=True)
    iw_all = np.squeeze(iw_all).astype(np.int32)
    n = len(iw_all)
    i_s = np.full(n, -1, dtype=np.int32)

    for iw_i in np.unique(iw_all):
        j = np.nonzero(iw_all == iw_i)[0]
        i_s[j] = np.arange(len(j))

    sql2.set_values_sql(file=file, table=table, values=(i_s.astype(np.int32).tolist(),), columns='sample_i32')


def reset_sample_i32_0(file):
    print(f"Reset sample_i32 0: {file}")
    table = 'paths'

    print('Load indices')
    w, s = sql2.get_values_sql(file=file, table=table, rows=-1, columns=['world_i32', 'sample_i32'], values_only=True)
    w100, s100 = sql2.get_values_sql(file=file, table=table, rows=np.arange(100), columns=['world_i32', 'sample_i32'], values_only=True)
    w = np.squeeze(w).astype(np.int32)
    s = np.squeeze(s).astype(np.int32)
    assert np.all(s == 0)

    print('Update indices')
    w0 = np.nonzero(w == 0)[0]
    b0 = w0[1:] != w0[:-1] + 1
    wb0 = w0[1:][b0]

    for wb0_i in wb0:
        s[wb0_i:] += 1

    print('Set indices')
    sql2.set_values_sql(file=file, table=table, values=(s.astype(np.int32).tolist(),), columns='sample_i32')
    sql2.set_values_sql(file=file, table=table, values=(s[:100].astype(np.int16).tolist(),), columns='sample_i32')


def combine_files(old_files, new_file, clean_s0):
    table = 'paths'
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
            print(f'concatenate')
            sql2.concatenate_tables(file=new_file, table=table, file2=f, table2=table)
            print(f'remove')
            os.remove(f)

        n = sql2.get_n_rows(file=new_file, table=table)
        print(f"Total number of rows: {n}")

    if old_files[0].startswith('gs://'):
        gcloud2.gsutil_cp(src=new_file, dst=f"{os.path.split(old_files[0])[0]}/{os.path.split(new_file[1])[0]}")


def main_choose_best(file):

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
    i_w, i_s, o, f = squeeze(i_w, i_s, o, f)

    m, _ = find_largest_consecutives(i_s)
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

    reset_sample_i32(file2)
    print(f"old {n_old} | tries per sample {m} -> old {n_old//m} | new {n_new}")


def separate_easy_hard(file, i):

    i_s, q = sql2.get_values_sql(file=file, table='paths', rows=i, columns=['sample_i32', 'q_f32'], values_only=True)
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
    table = 'paths'
    w, s = sql2.get_values_sql(file=file, table=table, rows=-1, columns=['world_i32', 'sample_i32'], values_only=True)

    s_not0 = np.nonzero(s != 0)[0]

    if np.size(s_not0) > 0:
        i = s_not0[-1]
        assert i < 10000
        sql2.delete_rows(file=file, table=table,  rows=np.arange(i+1))


def main_separate_easy_hard(file: str):

    file, _ = os.path.splitext(file)

    file_easy = f"{file}_easy.db"
    file_hard = f"{file}_hard.db"
    file = f"{file}.db"
    table = 'paths'

    print(f"Separate {file} into easy and hard")
    print('Copy initial file -> file_easy')
    copy(file, file_easy)

    n = sql2.get_n_rows(file=file_easy, table=table)
    print(f"Total: {n}")

    print(f"Load all world indices")
    iw_all = sql2.get_values_sql(file=file_easy, table='paths', rows=-1, columns=['world_i32'], values_only=True)

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
    sql2.set_values_sql(file=file_easy, table=table, values=(i_s.astype(np.int32).tolist(),), columns='sample_i32')
    print('Copy file_easy -> file_hard')
    copy(file_easy, file_hard)

    print('Delete respective complementing rows in file_easy and file_hard')
    sql2.delete_rows(file=file_easy, table=table, rows=b_hard)
    sql2.delete_rows(file=file_hard, table=table, rows=b_easy)

    n_easy = sql2.get_n_rows(file=file_easy, table=table)
    n_hard = sql2.get_n_rows(file=file_hard, table=table)
    print(f"total: {n} | easy: {n_easy} | hard: {n_hard}")
    reset_sample_i32(file=file_easy)


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


def main_combine_files(robot_id, i):
    old_files = [f"gs://tenh_jo/{robot_id}_{ii}.db" for ii in i]
    new_file = f"/home/johannes_tenhumberg_gmail_com/sdb/{robot_id}_combined_{i[0]}-{i[-1]+1}.db"
    combine_files(old_files=old_files, new_file=new_file, clean_s0=False)


def split_df(file):
    file = '/home/johannes_tenhumberg_gmail_com/sdb/Justin19_combined_0-40.db'
    i_s = sql2.get_values_sql(file=file, table='paths', rows=-1, columns='sample_i32', values_only=True)

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

    robot_id = 'Justin19'
    # i = np.arange(60, 80)
    # i = np.delete(i, 3)
    # main_combine_files(robot_id=robot_id, i=i)

    tic()
    _file0 = f"{robot_id}_combined_0-20"
    _file_bucket = f"gs://tenh_jo/{_file0}"
    _file = f"/home/johannes_tenhumberg_gmail_com/sdb/{_file0}"

    _file_easy = _file + '_easy'
    _file_hard = _file + '_hard'
    _file_hard2 = _file + '_hard2'
    gcloud2.gsutil_cp(src=f"gs://tenh_jo/{os.path.basename(_file)}.db", dst=f"{_file}.db")

    main_separate_easy_hard(file=_file)

    #
    print('sort easy')
    sql2.sort_table(file=_file_easy, table='paths', order_by=['world_i32', 'sample_i32', 'ROWID'])
    print('sort hard')
    sql2.sort_table(file=_file_hard, table='paths', order_by=['world_i32', 'sample_i32', 'ROWID'])

    print('upload easy and hard')
    gcloud2.gsutil_cp(src=f"{_file_easy}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_easy)}.db")
    gcloud2.gsutil_cp(src=f"{_file_hard}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_hard)}.db")

    main_choose_best(file=_file_hard)

    print('upload hard2')
    gcloud2.gsutil_cp(src=f"{_file_hard2}.db", dst=f"gs://tenh_jo/{os.path.basename(_file_hard2)}.db")
    toc()

    # reset_sample_i32_0(file=_file)
    # sql2.copy_table(file=_file, table_src='paths', table_dst='paths2',
    #                 columns=['world_i32', 'sample_i32', 'q_f32', 'objective_f32', 'feasible_b'],
    #                 dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_REAL, sql2.TYPE_INTEGER])

    # sql2.vacuum(file=_file)
    # sql2.alter_table(_file_easy, table='paths', columns=['world_i32', 'sample_i32', 'q_f32', 'objective_f32', 'feasible_b'],
    #                  dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_TEXT, sql2.TYPE_REAL, sql2.TYPE_INTEGER])
    # sql2.squeeze_table(file=_file, table='paths')

    #