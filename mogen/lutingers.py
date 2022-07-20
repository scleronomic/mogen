import numpy as np

from wzk import sql2, spatial, tictoc, print_progress, grid_x2i
from wzk.mpl import plot_projections_2d

from mopla.Automatica.lut import IKTableFull, IKShelfFull, lut_cube, IK_LUT_CubesTable
from rokin.Robots.Justin19 import Justin19, justin19_primitives
from rokin.Vis import robot_3d

from mogen.Generation.Data.data import T_PATHS

from mopla.Automatica import planner, scenes
from mopla.Parameter import adapt_ik_par_justin19, get_par_justin19
from mopla.main import choose_optimum
from mopla.Optimizer.length import len_close2q_cost


par, gd, staircase = get_par_justin19()
adapt_ik_par_justin19(par=par, mode=('table', 'right'))

par.qc.q = justin19_primitives.justin19_primitives(justin='getready_right_fly')


def create_lut():
    file = '/Users/jote/Documents/DLR/Data/mogen/Automatica2022/shelf_left_lut.db'
    robot = Justin19()
    f_idx = 13

    q, o = sql2.get_values_sql(file=file, table=T_PATHS(), rows=-1, columns=[T_PATHS.C_Q_F(), T_PATHS.C_OBJECTIVE_F()])

    f = robot.get_frames(q)[:, f_idx, :, :]

    x, rv = spatial.frame2trans_rotvec(f=f)
    xrv = np.concatenate((x, rv), axis=-1)
    plot_projections_2d(x=xrv, ls='', marker='o', markersize=1, alpha=0.1, color='k')
    euler = spatial.frame2euler(f=f)
    plot_projections_2d(x=euler, ls='', marker='o', markersize=1, alpha=0.1, color='k')

    # lut = IKShelfFull()
    #
    # with tictoc('fill_lut') as _:
    #     lut.fill_lut(x=xrv, y=q, o=o, verbose=True)


def animate_ik_lut(lut, robot, f_idx, n=1000, ):

    j = np.random.choice(np.arange(len(lut.lut_y)), n, replace=False)

    q = lut.lut_y[j]
    f = lut.y2X(q, robot, f_idx)

    x, rv = spatial.frame2trans_rotvec(f=f)
    xrv = np.concatenate((x, rv), axis=-1)
    plot_projections_2d(x=xrv, ls='', marker='o', markersize=1, alpha=0.1, color='k')

    # scene = scenes.CubeScene()
    # robot_3d.animate_path(q=q, robot=robot,
    #                       kwargs_frames=dict(f_fix=f, f_idx_robot=[f_idx], scale=0.05),
    #                       kwargs_world=dict(img=scene.img, limits=scene.limits))


def redo_filled_cells(lut, robot, n=3):

    res = []
    i_filled = np.array(np.nonzero(lut.lut_i != -1)).T
    for count, i in enumerate(i_filled):
        print_progress(i=count, n=len(i_filled))
        i = lut.lut_i[tuple(i)]
        q0 = lut.lut_y[i]
        f = lut.y2X(q0, robot, np.squeeze(par.xc.f_idx))

        q, status = planner.solve_ik_lut(par=par, gd=gd, f=f, lut=lut, n=n)
        q, status, mce, cost = choose_optimum.get_feasible_optimum(q=q[:, np.newaxis, :], par=par, status=status,
                                                                   verbose=0)
        res.append(mce > 0)
        if mce > 0:
            o0 = lut.y2o(q0, par.qc.q, par.weighting.joint_motion)
            o = lut.y2o(q, par.qc.q, par.weighting.joint_motion)
            if o < o0:
                lut.lut_y[i] = q
                # print(count, o-o0)
            else:
                pass


def add_empty_cells(lut, n=10):
    i_empty = np.array(np.nonzero(lut.lut_i == -1)).T
    x_samples = lut.sample_bin_centers()

    res = []
    for count, i in enumerate(i_empty):
        print_progress(i=count, n=len(i_empty), suffix=f"{np.sum(res)}/{len(i_empty)}")
        x = x_samples[tuple(i) + (slice(None),)]
        f = lut.x2X(x)
        q, status = planner.solve_ik_lut(par=par, gd=gd, f=f, lut=lut, n=n)
        q, status, mce, cost = choose_optimum.get_feasible_optimum(q=q[:, np.newaxis, :], par=par, status=status,
                                                                   verbose=0)

        res.append(mce > 0)
        if mce > 0:
            lut.lut_y = np.concatenate((lut.lut_y, q[0]), axis=0)
            lut.lut_i[tuple(i)] = len(lut.lut_y) - 1

    print(np.sum(res))


def test_ik_lut():
    res = []
    for i in range(2000):
        # f = spatial.sample_frames(x_low=lut.limits[:3, 0], x_high=lut.limits[:3, 1], shape=100)
        q0 = lut.lut_y[np.random.choice(range(len(lut.lut_y)), size=100, replace=False)]
        f = robot.get_frames(q0)[:, f_idx, :, :]

        with tictoc() as _:
            q, status = planner.solve_ik_lut(par=par, gd=gd, f=f[0], lut=lut)
            u, c = np.unique(status, return_counts=True)
            try:
                res.append(c[u == 1][0])
            except:
                res.append(0)

    # from wzk.mpl import new_fig
    # fig, ax = new_fig()
    # ax.hist(res, bins=20)


def transfer_luts():
    robot = Justin19()
    f_idx = 13
    lut_new = IK_LUT_CubesTable()
    lut_old = lut_cube

    lut_old.y2X = lambda y: robot.get_frames(y)[..., f_idx, :, :]
    lut_old.y2o = lambda y: len_close2q_cost(q=y, qclose=par.qc.q, joint_weighting=par.weighting.joint_motion, is_periodic=None)

    y = lut_old.lut_y
    x = lut_old.X2x(lut_old.y2X(y=y))
    o = lut_old.y2o(y=y)
    lut_new.fill_lut(x=x, y=y, o=o)
    lut_new.save_lut('/Users/jote/Documents/DLR/Data/mogen/Automatica2022/Cubes/ik_lut_cubes_cleaned3.npy')


# TODO add heuristic for the free joints
def main():

    # create_lut()
    # transfer_luts()
    robot = Justin19()
    f_idx = 13
    lut = lut_cube
    # print(lut.limits)
    animate_ik_lut(lut=lut, robot=robot, f_idx=f_idx, n=10000)
    #
    # lut.y2X = lambda y: robot.get_frames(y)[..., f_idx, :, :]
    # lut.y2o = lambda y: len_close2q_cost(q=y, qclose=par.qc.q, joint_weighting=par.weighting.joint_motion, is_periodic=None)
    #
    # lut.sanity_check()

    #
    # x_samples = lut.sample_bin_centers()
    # i = lut.get_i(x_samples)
    # mi = x_samples.min(axis=(0, 1, 2, 3))
    # ma = x_samples.max(axis=(0, 1, 2, 3))
    # print(lut.limits)
    # print(np.vstack((mi, ma)).T)
    # print(lut.n)
    # print(i.max(axis=(0, 1, 2, 3)))

    # add_empty_cells(lut=lut, n=10)
    # add_empty_cells(lut=lut, n=10)
    # add_empty_cells(lut=lut, n=10)

    # for kk in range(10):
    # redo_filled_cells(lut, robot=Justin19(), n=5)
    # lut.save_lut('/Users/jote/Documents/DLR/Data/mogen/Automatica2022/Cubes/ik_lut_cubes_fly.npy')

    # animate_ik_lut(lut=lut, robot=Justin19(), f_idx=13, n=100)

    # test_ik_lut()


if __name__ == '__main__':
    main()
