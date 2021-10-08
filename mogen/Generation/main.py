import numpy as np

from wzk.ray2 import ray
from wzk import tic, toc
from wzk.trajectory import inner2full
from wzk.gd.Optimizer import Naive
from wzk.image import compressed2img
from wzk.dicts_lists_tuples import change_tuple_order

from rokin.Robots import *
from rokin.Vis.robot_3d import robot_path_interactive
from mopla import parameter
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql, get_values_sql, get_n_rows
from mogen.Generation.sample_start_end import sample_q_start_end

# ray.init(address='auto')


class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')

robot0 = SingleSphere02(radius=0.25)
# db_file = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot0.id}.db'
db_file = f'/Users/jote/Documents/Code/Python/DLR/mogen/{robot0.id}.db'
# np_result_file = f'/volume/USERSTORE/tenh_jo/0_Data/Samples/{robot0.id}.npy'

print(db_file)

def set_sc_on(par):
    par.check.self_collision = True
    par.planning.self_collision = True
    par.sc.n_substeps = 3
    par.sc.n_substeps_check = 3


def init_par():
    robot = SingleSphere02(radius=0.25)
    # robot = JustinArm07()
    # robot = StaticArm(n_dof=4, limb_lengths=0.5, limits=np.deg2rad([-170, +170]))
    # robot = Justin19()
    bee_rate = 0.0
    n_multi_start = [[0, 1, 2, 3], [1, 17, 16, 16]]

    par = parameter.Parameter(robot=robot, obstacle_img=None)
    par.n_waypoints = 20

    par.check.obstacle_collision = True
    par.planning.obstacle_collision = True
    par.oc.n_substeps = 3  # was 3 for justin
    par.oc.n_substeps_check = 3

    # set_sc_on(par)

    gd = parameter.GradientDescent()
    gd.opt = Naive(ss=1)
    gd.n_processes = 1
    gd.n_steps = 1000  # was 750, was 1000 for StaticArm / SingleSphere02

    gd.return_x_list = False
    n0, n1 = gd.n_steps//3, gd.n_steps//3
    n2 = gd.n_steps - (n0 + n1)
    gd.clipping = np.concatenate([np.ones(n0)*np.deg2rad(1), np.ones(n1)*np.deg2rad(0.1), np.ones(n2)*np.deg2rad(0.01)])

    # gd.clipping = np.ones(gd.n_steps) * np.deg2rad(3)
    # gd.clipping = 0.1

    gen = Generation()
    gen.par = par
    gen.gd = gd
    gen.n_multi_start = n_multi_start
    gen.bee_rate = bee_rate

    return gen


def __chomp(q0, q_start, q_end,
            gd, par,
            i_world, i_sample):

    q, o = gradient_descent.gd_chomp(q0=q0.copy(), q_start=q_start, q_end=q_end, gd=gd, par=par)

    q0 = inner2full(inner=q0, start=q_start, end=q_end)
    q = inner2full(inner=q, start=q_start, end=q_end)
    f = feasibility_check(q=q, par=par) == 1

    n = len(q)
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample

    return create_path_df(i_world=i_world, i_sample=i_sample,
                          q0=q0, q=q, objective=o, feasible=f)


def sample_path(gen, i_world, i_sample, img_cmp, verbose=0):
    np.random.seed()

    par = gen.par
    gd = gen.gd
    assert gen.n_multi_start[0][0] == 0

    obstacle_img = compressed2img(img_cmp=img_cmp, n_voxels=par.world.n_voxels, dtype=bool)
    parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=obstacle_img)
    q_start, q_end = sample_q_start_end(robot=par.robot, feasibility_check=lambda qq: feasibility_check(q=qq, par=par),
                                        acceptance_rate=gen.bee_rate)

    get_q0 = InitialGuess.path.q0_random_wrapper(robot=par.robot, n_multi_start=gen.n_multi_start,
                                                 n_waypoints=par.n_waypoints, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)

    q00 = q0[:1]
    q0 = q0[1:]

    df0 = __chomp(q0=q00, q_start=q_start, q_end=q_end, gd=gd, par=par, i_world=i_world, i_sample=i_sample)
    print(i_sample)
    print(df0.feasible[0])
    if df0.feasible[0]:
        return df0

    df = __chomp(q0=q0, q_start=q_start, q_end=q_end, gd=gd, par=par, i_world=i_world, i_sample=i_sample)

    if verbose > 0:
        j = np.argmin(o + (df.feasible == -1)*df.objective.max())
        robot_path_interactive(q=df.q[j], robot=par.robot,
                               kwargs_world=dict(img=obstacle_img, limits=par.world.limits),
                               kwargs_robot=dict(mode='sphere'))

    df = df0.append(df)
    return df


def test_samples():
    pass


def main(iw_list=None):
    n_samples_per_world = 10
    worlds = get_values_sql(file=db_file, table='worlds', columns='img_cmp', values_only=True)

    # gen = init_par()
    # df = sample_path(gen=gen, i_world=0, i_sample=0, img_cmp=worlds[0], verbose=1)

    # @ray.remote
    def sample_ray(_i_w, _i_s):
        gen = init_par()
        return sample_path(gen=gen, i_world=_i_w, i_sample=_i_s, img_cmp=worlds[_i_w])

    futures = []
    for i_w in iw_list:
        for i_s in range(n_samples_per_world):
            # futures.append(sample_ray(i_w, i_s))
            futures.append(sample_ray.remote(i_w, i_s))

    df_list = ray.get(futures)
    # df_list = futures

    df = df_list[0]
    for df_i in df_list[1:]:
        df = df.append(df_i)

    df2sql(df=df, file=db_file, table='paths', if_exists='replace')
    print(df)
    return df


def meta_main():
    worlds = np.arange(0, 200)
    for iw in np.array_split(worlds, 40):
        main(iw)


if __name__ == '__main__':
    from wzk import tic, toc
    tic()
    # meta_main()
    df = main(iw_list=np.arange(100))
    toc()
    # print('New DB:', get_n_rows(file=db_file, table='paths'))


#