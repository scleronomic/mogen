import numpy as np

from wzk.trajectory import inner2full
from wzk.gd.Optimizer import Naive
from wzk.image import compressed2img

from rokin.Robots import StaticArm, Justin19
from mopla import parameter
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql, get_values_sql
from mogen.SampleGeneration.sample_start_end import sample_q_start_end


class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')


db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/Justin19.db'


def init_robot():
    robot = StaticArm(n_dof=4, limb_lengths=0.5, limits=np.deg2rad([-170, +170]))


def set_sc_on(par):
    par.check.self_collision = True
    par.planning.self_collision = True
    par.sc.n_substeps_check = 3
    par.sc.n_substeps = 3


def init_par():
    # robot = StaticArm(n_dof=4, limb_lengths=0.5, limits=np.deg2rad([-170, +170]))
    robot = Justin19()

    bee_rate = 0.05
    n_multi_start = [[0, 1, 2, 3], [1, 17, 16, 16]]

    par = parameter.Parameter(robot=robot, obstacle_img='perlin')
    par.n_waypoints = 20

    par.check.obstacle_collision = True
    par.planning.obstacle_collision = True
    par.oc.n_substeps = 3
    par.oc.n_substeps_check = 5

    set_sc_on(par)

    gd = parameter.GradientDescent()
    gd.opt = Naive(ss=1)
    gd.n_processes = 1
    gd.n_steps = 1000

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


def sample_path(gen, i_world, i_sample, img_cmp):
    np.random.seed()

    par = gen.par
    gd = gen.gd

    obstacle_img = compressed2img(img_cmp=img_cmp, n_voxels=par.world.n_voxels, dtype=bool)
    parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=obstacle_img)

    q_start, q_end = sample_q_start_end(robot=par.robot, feasibility_check=lambda qq: feasibility_check(q=qq, par=par),
                                        acceptance_rate=gen.bee_rate)

    get_q0 = InitialGuess.path.q0_random_wrapper(robot=par.robot, n_multi_start=gen.n_multi_start,
                                                 n_waypoints=par.n_waypoints, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)

    from wzk import tic, toc

    tic()
    q, o = gradient_descent.gd_chomp(q0=q0.copy(), q_start=q_start, q_end=q_end, gd=gd, par=par, verbose=1)
    toc(name=f"{i_world}, {i_sample}")
    f = feasibility_check(q=q, par=par) == 1

    q0 = inner2full(inner=q0, start=q_start, end=q_end)
    q = inner2full(inner=q, start=q_start, end=q_end)
    n = len(q)
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample

    q0 = [qq0.tobytes() for qq0 in q0]
    q = [qq.tobytes() for qq in q]

    return create_path_df(i_world=i_world, i_sample=i_sample,
                          q0=q0, q=q, objective=o, feasible=f)


def main():
    n_worlds = 5
    n_samples_per_world = 50
    from wzk.ray2 import ray
    ray.init(address='auto')

    worlds = get_values_sql(file=db_file, table='worlds', columns='img_cmp', values_only=True)

    @ray.remote
    def sample_ray(_i_w, _i_s):
        gen = init_par()
        return sample_path(gen=gen, i_world=_i_w, i_sample=_i_s, img_cmp=worlds[_i_w])

    futures = []
    for i_w in range(n_worlds):
        for i_s in range(n_samples_per_world):
            futures.append(sample_ray.remote(i_w, i_s))

    df_list = ray.get(futures)

    df = df_list[0]
    for df_i in df_list[1:]:
        df = df.append(df_i)

    df2sql(df=df, file=db_file, table_name='paths', if_exists='replace')
    print(df)
    return df


if __name__ == '__main__':
    from wzk import tic, toc
    tic()
    main()
    toc()

