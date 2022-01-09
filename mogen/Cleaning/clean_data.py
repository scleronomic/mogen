import numpy as np
from wzk.sql2 import get_values_sql, set_values_sql, df2sql, vacuum, get_n_rows
from shutil import copy

from wzk import compressed2img, find_consecutives
from wzk.mpl import new_fig

from rokin.Robots import *
from mopla.main import objective_feasibility
from mopla.Optimizer import choose_optimum
from mopla.Optimizer.gradient_descent import gd_chomp


from mogen.Generation.parameter import init_par
from mogen.Loading.load import create_path_df, create_world_df, get_sample


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


def main_choose_best(robot_id: str):
    # gen = init_par(robot_id=robot_id)
    # file_org = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_hard.db'
    # file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_hard2.db'
    file_org = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot_id}_hard.db'
    file = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot_id}_hard2.db'
    # copy(file_org, file)

    m = 50
    n = get_n_rows(file=file_org, table='paths')
    print('N_rows original:', n)
    i = np.arange(0, (n // m) * m)
    # i = np.arange(+15994950
    #               -10000000, 15994950)
    i_w, i_s, q0, q, o, f = get_values_sql(file=file_org, table='paths', rows=i,
                                           columns=['world_i32', 'sample_i32',
                                                    'q0_f32', 'q_f32', 'objective_f32', 'feasible_b'],
                                           values_only=True)

    check_iw_is(i_w=i_w, i_s=i_s, m=m)

    # check_consistency(q=q, f=f, i_w=i_w)
    o_max = o.max()
    o2 = o.copy()

    o2[f != 1] += o_max
    o2 = o2.reshape(-1, m)
    j = np.argmin(o2, axis=-1) + np.arange(len(o2))*m

    print('max j', j.max())

    f = np.squeeze(f)
    j = j[f[j] == 1]
    df = get_df_subset(i_w=i_w, i_s=i_s, q0=q0, q=q, o=o, f=f, i=j, n=1)

    print('N_rows new:', len(df))

    df2sql(df=df, file=file, table='paths', if_exists='replace')
    vacuum(file)


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


def get_df_subset(i_w, i_s, q0, q, o, f,
                  i: np.ndarray, n: int = 1):
    iw_i, is_i, q0_i, q_i, o_i, f_i = i_w[i], i_s[i], q0[i], q[i], o[i], f[i]
    is_i = np.arange(len(is_i)//n).repeat(n)
    iw_i = np.squeeze(iw_i)
    print(iw_i)
    print(is_i)
    if len(iw_i) > 0:
        df_i = create_path_df(i_world=iw_i, i_sample=is_i,
                              q0=q0_i, q=q_i, objective=o_i, feasible=f_i)
    else:
        df_i = None

    return df_i


def separate_easy_hard(file, iw_i, iw_all):
    b_iwi = iw_all == iw_i
    j_iwi = np.nonzero(b_iwi)[0]

    i_w, i_s, q0, q, o, f = get_values_sql(file=file, table='paths',
                                           rows=j_iwi, columns=['world_i64', 'sample_i64',
                                                                'q0_f64', 'q_f64',
                                                                'objective_f64', 'feasible_b'],
                                           values_only=True)

    n = 31  # Justin19
    n = 50  # SingleSphere02

    i_hard = find_consecutives(x=i_s, n=n)

    i_hard = i_hard[:, np.newaxis] + np.arange(n)[np.newaxis, :]
    i_hard = i_hard.ravel()

    i_easy = np.arange(len(i_s))
    i_easy = np.delete(i_easy, i_hard)

    df_easy = get_df_subset(i_w=i_w, i_s=i_s, q0=q0, q=q, o=o, f=f, i=i_easy, n=1)
    df_hard = get_df_subset(i_w=i_w, i_s=i_s, q0=q0, q=q, o=o, f=f, i=i_hard, n=n)

    return df_easy, df_hard


def main_separate_easy_hard(robot_id: str):

    file = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}.db'
    file_hard = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_hard.db'
    file_easy = f'/net/rmc-lx0062/home_local/tenh_jo/{robot_id}_easy.db'

    img_cmp = get_values_sql(file=file, table='worlds',
                             rows=-1, columns=['img_cmp'],
                             values_only=True)
    i_w = np.arange(len(img_cmp))
    df = create_world_df(i_world=i_w, img_cmp=img_cmp)
    print(len(img_cmp))
    df2sql(df=df, file=file_easy, table='worlds', if_exists='replace')
    df2sql(df=df, file=file_hard, table='worlds', if_exists='replace')

    # return
    n = get_n_rows(file=file, table='paths')
    print('n', n)
    iw_all = get_values_sql(file=file, table='paths',
                            rows=-1, columns=['world_i64'], values_only=True)
    iw_all = iw_all.astype(np.int32)

    for iw_i in range(0, 10000):
        ra = 'replace' if iw_i == 0 else 'append'
        df_easy, df_hard = separate_easy_hard(file=file, iw_i=iw_i, iw_all=iw_all)

        print(f"{iw_i} | easy: {len(df_easy)} | hard: {len(df_hard)}")
        df2sql(df=df_easy, file=file_easy, table='paths', if_exists=ra)
        df2sql(df=df_hard, file=file_hard, table='paths', if_exists=ra)

        if iw_i == 0:
            vacuum(file_easy)
            vacuum(file_hard)


if __name__ == '__main__':
    # main_separate_easy_hard(robot_id='SingleSphere02')
    main_choose_best(robot_id='SingleSphere02')


