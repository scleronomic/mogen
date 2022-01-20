import numpy as np

from wzk import tic, toc, tictoc
from wzk.dlr import LOCATION
from wzk.ray2 import ray, ray_init
from wzk.trajectory import inner2full
from wzk.image import compressed2img, img2compressed
from wzk.sql2 import df2sql, get_values_sql, vacuum

from rokin.Vis.robot_3d import robot_path_interactive
from mopla.main import chomp_mp
from mopla.Parameter.parameter import initialize_oc
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load import create_path_df
from mogen.Generation.parameter import init_par
from mogen.Generation.starts_ends import sample_q_start_end


__file_stub_dlr = '/net/rmc-lx0062/home_local/tenh_jo/{}_sc.db'
__file_stub_mac = '/Users/jote/Documents/DLR/Data/mogen/{}_sc.db'
__file_stub_gc = '/home/johannes_tenhumberg/Data/{}_sc.db'

file_stub_dict = dict(dlr=__file_stub_dlr, mac=__file_stub_mac, gc=__file_stub_gc)
file_stub = file_stub_dict[LOCATION]


img_cmp0 = [img2compressed(img=np.zeros((64,), dtype=bool), n_dim=1),
            img2compressed(img=np.zeros((64, 64), dtype=bool), n_dim=2),
            img2compressed(img=np.zeros((64, 64, 64), dtype=bool), n_dim=3)]


def __chomp(q0, q_start, q_end,
            gen,
            i_world, i_sample):

    gen.par.q_start, gen.par.q_end = q_start, q_end

    q, o = gradient_descent.gd_chomp(q0=q0.copy(), gd=gen.gd, par=gen.par)

    q = inner2full(inner=q, start=gen.par.q_start, end=gen.par.q_end)
    f = feasibility_check(q=q, par=gen.par) == 1

    n = len(q)
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample

    return create_path_df(i_world=i_world, i_sample=i_sample,
                          q=q, objective=o, feasible=f)


def __chomp2(q_start, q_end,
             gen,
             i_world, i_sample):

    gen.par.q_start, gen.par.q_end = q_start, q_end

    q, o = chomp_mp(gd=gen.gd, par=gen.par, staircase=gen.staircase)
    n = len(q)
    f = feasibility_check(q=q, par=gen.par) == 1
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample
    return create_path_df(i_world=i_world, i_sample=i_sample, q=q, objective=o, feasible=f)


def sample_path(gen, i_world, i_sample, img_cmp, verbose=0):
    np.random.seed(None)

    obstacle_img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    initialize_oc(par=gen.par, obstacle_img=obstacle_img)
    q_start, q_end = sample_q_start_end(robot=gen.par.robot,
                                        feasibility_check=lambda qq: feasibility_check(q=qq, par=gen.par),
                                        acceptance_rate=gen.bee_rate)

    get_q0 = InitialGuess.path.q0s_random_wrapper(robot=gen.par.robot, n_multi_start=[[0], [1]],
                                                  n_waypoints=gen.par.n_waypoints, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)[:1]

    if verbose > 0:
        tic()

    df0 = __chomp(q0=q0, q_start=q_start, q_end=q_end, gen=gen, i_world=i_world, i_sample=i_sample)
    if np.all(np.frombuffer(df0.feasible_b[0], dtype=bool)):
        if verbose > 0:
            toc(text=f"{gen.par.robot.id}, {i_world}, {i_sample}")
        return df0

    df = __chomp2(q_start=q_start, q_end=q_end, gen=gen, i_world=i_world, i_sample=i_sample)

    if verbose > 2:
        j = np.argmin(df.objective + (df.feasible == -1)*df.objective.max())
        robot_path_interactive(q=df.q[j], robot=gen.par.robot,
                               kwargs_world=dict(img=obstacle_img, limits=gen.par.world.limits))

    df = df0.append(df)
    if verbose > 0:
        toc(text=f"{gen.par.robot.id}, {i_world}, {i_sample}")
    return df


def main(robot_id: str, iw_list=None, n_samples_per_world=1000, ra='append'):
    file = file_stub.format(robot_id)

    @ray.remote
    def sample_ray(_i_w: int, _i_s: int):
        gen = init_par(robot_id=robot_id)
        _i_w = int(_i_w)
        if _i_w == -1:
            img_cmp = img_cmp0[gen.par.robot.n_dim-1]
        else:
            img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return sample_path(gen=gen, i_world=_i_w, i_sample=_i_s, img_cmp=img_cmp, verbose=0)

    futures = []
    for i_w in iw_list:
        for i_s in range(n_samples_per_world):
            futures.append(sample_ray.remote(i_w, i_s))
            # futures.append(sample_ray(i_w, i_s))

    df_list = ray.get(futures)
    # df_list = futures

    df = df_list[0]
    for df_i in df_list[1:]:
        df = df.append(df_i)

    with tictoc(text=f"Saving {len(df)} new samples") as _:
        df2sql(df=df, file=file, table='paths', if_exists=ra)
        if ra == 'replace':
            vacuum(file=file)
    return df


# def main_loop(robot_id):
    # main(robot_id=_robot_id, iw_list=[0], ra='replace')

#     for i in range(10):
#         worlds = np.arange(10000)
#         for iw in np.array_split(worlds, len(worlds)//10):
#             iw = iw.astype(int)
#             print(f"{i}:  {min(iw)} - {max(iw)}", end="  |  ")
#             with tictoc() as _:
#                 main(robot_id=robot_id, iw_list=iw, ra='append')


def main_loop_sc(robot_id):
    worlds = [-1]
    main(robot_id=robot_id, iw_list=worlds, ra='replace', n_samples_per_world=100)
    for i in range(10000):
        worlds = [-1]
        with tictoc(f'loop {i}') as _:
            main(robot_id=robot_id, iw_list=worlds, ra='append', n_samples_per_world=1000)


if __name__ == '__main__':

    ray_init(perc=100)
    _robot_id = 'Justin19'

    # import os
    # print(os.environ['PYTHONPATH'])
    # @ray.remote
    # def dummy():
    #     from mopla import Parameter
    #
    #
    # futures = []
    # for i_w in range(10):
    #     futures.append(dummy.remote())
    #
    # df_list = ray.get(futures)

    with tictoc('total time') as _:
        main_loop_sc(_robot_id)
