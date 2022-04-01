import os
import numpy as np

from wzk import tic, toc, tictoc
from wzk.dlr import LOCATION
from wzk.ray2 import ray, ray_init
from wzk.trajectory import inner2full
from wzk.image import compressed2img
from wzk.sql2 import df2sql, get_values_sql, vacuum
from wzk.gcp import gcloud2
from wzk.subprocess2 import call2

from rokin.Vis.robot_3d import robot_path_interactive
from mopla.main import chomp_mp
from mopla.Parameter.parameter import initialize_oc
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Generation import data, parameter, starts_ends

__file_stub_dlr = '/home_local/tenh_jo/{}.db'
__file_stub_mac = '/Users/jote/Documents/DLR/Data/mogen/{}_sc.db'
__file_stub_gcp = '/home/johannes_tenhumberg_gmail_com/sdb/{}.db'

file_stub_dict = dict(dlr=__file_stub_dlr, mac=__file_stub_mac, gcp=__file_stub_gcp)
file_stub = file_stub_dict[LOCATION]


def copy_init_world(robot_id):
    file_bucket_stub = 'gs://tenh_jo/{}_worlds0.db'
    call2(cmd=f"sudo chmod 777 -R {os.path.split(file_stub.format(robot_id))[0]}")
    if LOCATION == 'gcp':
        gcloud2.gsutil_cp(src=file_bucket_stub.format(robot_id), dst=file_stub.format(robot_id))
    else:
        pass


def __chomp0(q0, q_start, q_end,
             gen,
             i_world, i_sample):

    gen.par.q_start, gen.par.q_end = q_start, q_end

    q, o = gradient_descent.gd_chomp(q0=q0.copy(), gd=gen.gd, par=gen.par)

    q = inner2full(inner=q, start=gen.par.q_start, end=gen.par.q_end)
    f = feasibility_check(q=q, par=gen.par) == 1

    n = len(q)
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample

    return data.create_path_df(i_world=i_world, i_sample=i_sample, q=q, objective=o, feasible=f)


def __chomp_staircase(q_start, q_end,
                      gen,
                      i_world, i_sample):

    gen.par.q_start, gen.par.q_end = q_start, q_end

    q, o = chomp_mp(gd=gen.gd, par=gen.par, staircase=gen.staircase)
    n = len(q)
    f = feasibility_check(q=q, par=gen.par) == 1
    i_world = np.ones(n, dtype=int) * i_world
    i_sample = np.ones(n, dtype=int) * i_sample
    return data.create_path_df(i_world=i_world, i_sample=i_sample, q=q, objective=o, feasible=f)


def generate_path(gen, i_world, i_sample, img_cmp, verbose=0):
    np.random.seed(None)

    obstacle_img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    initialize_oc(par=gen.par, obstacle_img=obstacle_img)
    try:
        q_start, q_end = starts_ends.sample_q_start_end(robot=gen.par.robot,
                                                        feasibility_check=lambda qq: feasibility_check(qq, par=gen.par),
                                                        acceptance_rate=gen.bee_rate)
    except RuntimeError:
        df = data.create_path_df(i_world=np.ones(1)*i_world, i_sample=np.ones(1)*i_sample,
                                 q=np.zeros((1, gen.par.n_wp, gen.par.robot.n_dof)),
                                 objective=np.ones(1)*-1, feasible=np.zeros(1, dtype=bool))
        return df

    get_q0 = InitialGuess.path.q0s_random_wrapper(robot=gen.par.robot, n_multi_start=[[0], [1]],
                                                  n_wp=gen.par.n_wp, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)[:1]

    if verbose > 0:
        tic()

    df0 = __chomp0(q0=q0, q_start=q_start, q_end=q_end, gen=gen, i_world=i_world, i_sample=i_sample)
    if np.all(np.frombuffer(df0.feasible_b[0], dtype=bool)):
        if verbose > 0:
            toc(text=f"{gen.par.robot.id}, {i_world}, {i_sample}")
        return df0

    df = __chomp_staircase(q_start=q_start, q_end=q_end, gen=gen, i_world=i_world, i_sample=i_sample)

    if verbose > 2:
        j = np.argmin(df.objective + (df.feasible == -1)*df.objective.max())
        robot_path_interactive(q=df.q[j], robot=gen.par.robot,
                               kwargs_world=dict(img=obstacle_img, limits=gen.par.world.limits))

    df = df0.append(df)
    if verbose > 0:
        toc(text=f"{gen.par.robot.id}, {i_world}, {i_sample}")
    return df


def main(robot_id: str, iw_list=None, n_samples_per_world=1000, s0=0, ra='append'):
    file = file_stub.format(robot_id)

    @ray.remote
    def generate_ray(_i_w: int, _i_s: int):
        gen = parameter.init_par(robot_id=robot_id)
        _i_w = int(_i_w)
        if _i_w == -1:
            img_cmp = data.img_cmp0[gen.par.robot.n_dim-1]
        else:
            img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return generate_path(gen=gen, i_world=_i_w, i_sample=_i_s, img_cmp=img_cmp, verbose=0)

    with tictoc(text=f"Generating Samples for world {iw_list[0]}-{iw_list[-1]} with {n_samples_per_world} samples") as _:
        futures = []
        for i_w in iw_list:
            for i_s in range(n_samples_per_world):
                futures.append(generate_ray.remote(i_w, s0+i_s))

        df_list = ray.get(futures)
        df = data.combine_df_list(df_list)

    with tictoc(text=f"Saving {len(df)} new samples", verbose=(1, 2)) as _:
        df2sql(df=df, file=file, table='paths', dtype=data.path_df_dtypes, if_exists=ra)
        if ra == 'replace':
            vacuum(file=file)
    return df


def main_loop(robot_id):
    copy_init_world(robot_id)
    worlds = np.arange(10000)

    for i in range(100000):
        print(i)
        with tictoc() as _:
            main(robot_id=robot_id, iw_list=worlds, ra='append', s0=i, n_samples_per_world=1)


def main_loop_sc(robot_id):
    worlds = [-1]
    main(robot_id=robot_id, iw_list=worlds, ra='replace', n_samples_per_world=100)
    for i in range(10000):
        worlds = [-1]
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=1000, ra='append',)


if __name__ == '__main__':

    ray_init(perc=100)
    _robot_id = 'SingleSphere02'

    with tictoc('total time') as _:
        main_loop(_robot_id)
