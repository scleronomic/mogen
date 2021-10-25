import numpy as np
from wzk.sql2 import get_values_sql, df2sql, vacuum, get_n_rows
from shutil import copy

from wzk import compressed2img

from rokin.Robots import *
from mopla.main import objective_feasibility
from mogen.Loading.load_pandas import create_path_df


n_waypoints = 20


def check_consistency(robot,
                      db_file=None,
                      q=None, f=None, o=None, img=None):
    from mopla import parameter

    par = parameter.Parameter(robot=robot)
    par.check.self_collision = False
    par.sc.n_substeps_check = 3
    par.oc.n_substeps_check = 3

    if q is None:
        i_w, i_s, q0, q, o, f = get_values_sql(file=db_file, table='paths',
                                               rows=i,
                                               columns=['i_world', 'i_sample', 'q0', 'q', 'objective', 'feasible'],
                                               values_only=True)

    if img is None:
        img_cmp = get_values_sql(file=db_file, rows=np.arange(300), table='worlds', columns='img_cmp', values_only=True)
        img = compressed2img(img_cmp=img_cmp, n_voxels=par.world.n_voxels, dtype=bool)

    q = q.reshape(-1, n_waypoints, robot.n_dof).copy()

    o_label, f_label = objective_feasibility(q=q, imgs=img, par=par, iw=i_w)
    print(f.sum(), f_label.sum(), (f == f_label).mean())


def check_iw_is(i_w, i_s):
    m = 50

    i_w = i_w.copy()
    i_w = i_w.reshape(-1, m)
    bw = np.unique(i_w, axis=1)
    assert bw.shape[1] == 1

    i_s = i_s.copy()
    i_s = i_s.reshape(-1, m)
    bs = np.unique(i_s, axis=1)
    assert bs.shape[1] == 1


def main():
    robot = JustinArm07()
    # robot = SingleSphere02(radius=0.25)
    # robot = StaticArm(n_dof=4, limb_lengths=0.5, limits=np.deg2rad([-170, +170]))

    db_file_org = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot.id}.db'
    db_file = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot.id}_global.db'
    copy(db_file_org, db_file)

    # db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/JustinArm07_global2.db'
    # db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/JustinArm07.db'
    # db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/Justin19_global2.db'
    # db_file = '/Users/jote/Documents/Code/Python/DLR/mogen/Justin19.db'
    # db_file = '/Users/jote/Documents/Code/Python/DLR/mogen/StaticArm04.db'

    # i = np.arange(0, 200*100*50)
    # i = np.arange(0, 5000)
    # i = np.arange(0, 300*100*50)

    n = get_n_rows(file=db_file, table='paths')
    print('N_rows original:', n)
    i = np.arange(0, (n // 50) * 50)



    i_w, i_s, q0, q, o, f = get_values_sql(file=db_file, table='paths',
                                           rows=i, columns=['i_world', 'i_sample', 'q0', 'q', 'objective', 'feasible'],
                                           values_only=True)

    # check_consistency(q=q, f=f, i_w=i_w)
    # return
    o_max = o.max()
    o2 = o.copy()

    # o2[f != 1] += o_max
    # o2 = o2.reshape(-1, 50)
    # j = np.argmin(o2, axis=-1) + np.arange(len(o2))*50
    check_iw_is(i_w=i_w, i_s=i_s)
    print(f.dtype)
    print(f.shape)
    f = np.squeeze(f)

    m = 50
    qs = q.reshape(len(i), 20, -1).copy()
    qs = ((qs[:, 1:, :] - qs[:, :-1, :])**2).sum(axis=(-1, -2))
    qs[f != 1] += o_max
    qs = qs.reshape(-1, m)
    j = np.argmin(qs, axis=-1) + np.arange(len(qs))*m

    print('max j', j.max())
    # print((j != j2).mean())
    j = j[f[j] == 1]

    i_wi = i_w[j]
    i_si = i_s[j]
    q0i = q0[j]
    qi = q[j]
    oi = o[j]
    fi = f[j]

    df = create_path_df(i_world=i_wi, i_sample=i_si,
                        q0=q0i, q=qi, objective=oi, feasible=fi)
    print(df)
    print('N_rows new:', len(df))

    df2sql(df=df, file=db_file, table='paths', if_exists='replace')
    vacuum(db_file)


def plot():
    from wzk.mpl import new_fig, error_area
    # def plot_o_distributions(o):
    o, f = get_values_sql(file=db_file, table='paths', rows=i,
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


main()


# TODO write rename column function

n_batch = 10000

i = 0
while True:
    idx = np.arange(i, i+n_batch)
    i_worlds, i_samples =

