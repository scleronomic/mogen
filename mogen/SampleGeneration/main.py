import numpy as np

from wzk.trajectory import inner2full
from wzk.gd.Optimizer import Naive

from rokin.Robots import StaticArm
from mopla import parameter
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql

from mogen.SampleGeneration.sample_start_end import sample_q_start_end


class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')


def init_par():
    robot = StaticArm(n_dof=4, limb_lengths=0.5, limits=np.deg2rad([-170, +170]))

    bee_rate = 0.05
    n_multi_start = [[0, 1, 2, 3], [1, 17, 16, 16]]

    par = parameter.Parameter(robot=robot, obstacle_img='perlin')
    par.n_waypoints = 20

    par.check.obstacle_collision = True
    par.planning.obstacle_collision = True
    par.oc.n_substeps = 3
    par.oc.n_substeps_check = 5

    gd = parameter.GradientDescent()
    gd.opt = Naive(ss=1)
    gd.n_processes = 1
    gd.n_steps = 100

    gd.clipping = np.concatenate([np.ones(gd.n_steps//2)*np.deg2rad(1),
                                  np.ones(gd.n_steps//3)*np.deg2rad(0.1),
                                  np.ones(gd.n_steps-(gd.n_steps//2 - gd.n_steps//3))*np.deg2rad(0.01),
                                  ])

    gen = Generation()
    gen.par = par
    gen.gd = gd
    gen.n_multi_start = n_multi_start
    gen.bee_rate = bee_rate

    return gen


def sample_path(gen, i_world, i_sample):
    print(i_world, i_sample)
    np.random.seed()

    par = gen.par
    gd = gen.gd

    parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img='perlin')  # TODO

    q_start, q_end = sample_q_start_end(robot=par.robot, feasibility_check=lambda qq: feasibility_check(q=qq, par=par),
                                        acceptance_rate=gen.bee_rate)

    get_q0 = InitialGuess.path.q0_random_wrapper(robot=par.robot, n_multi_start=gen.n_multi_start,
                                                 n_waypoints=par.n_waypoints, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)

    from wzk import tic, toc

    tic()
    q, o = gradient_descent.gd_chomp(q0=q0.copy(), q_start=q_start, q_end=q_end, gd=gd, par=par, verbose=1)
    toc()
    f = feasibility_check(q=q, par=par) == 1

    q0 = inner2full(inner=q0, start=q_start, end=q_end)
    q = inner2full(inner=q, start=q_start, end=q_end)
    print(q[:2])
    n = len(q)
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample

    q0 = [qq0.tobytes() for qq0 in q0]
    q = [qq.tobytes() for qq in q]
    print(q[:2])

    return create_path_df(i_world=i_world, i_sample=i_sample,
                          q0=q0, q=q, objective=o, feasible=f)


def main():
    n_worlds = 2
    n_samples_per_world = 10
    from wzk.ray2 import ray
    ray.init(address='auto')

    @ray.remote
    def sample_ray(_i_w, _i_s):
        gen = init_par()
        return sample_path(gen=gen, i_world=_i_w, i_sample=_i_s)

    futures = []
    for i_w in range(n_worlds):
        for i_s in range(n_samples_per_world):
            futures.append(sample_ray.remote(i_w, i_s))

    df = ray.get(futures)

    dff = df[0]
    for dfff in df[1:]:
        dff = dff.append(dfff)

    print(dff.shape)
    print(dff)
    df2sql(df=dff, file='datadata.db', table_name='path', if_exists='replace')
    return df


if __name__ == '__main__':
    pass
    # df = main()
    # print(len(df))

# df = create_path_df(i_world=np.zeros(10), i_sample=np.ones(10),
#                     q0=np.ones((10, 100, 20)).tolist(), q=np.ones((10, 100, 20)).tolist(),
#                     feasible=np.zeros(10), objective=np.ones(10),)
# df2sql(df=df, file='datadata.db', table_name='path', if_exists='replace')

# ~1s per path per core
# 3600*24*60 ~ 5 Million samples in one day
