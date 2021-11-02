import numpy as np

from wzk.ray2 import ray
from wzk.trajectory import inner2full
from wzk.image import compressed2img
from wzk.sql2 import df2sql, get_values_sql, vacuum

from rokin.Vis.robot_3d import robot_path_interactive
from mopla.parameter import initialize_oc
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load import create_path_df
from mogen.Generation.parameter import init_par
from mogen.Generation.starts_ends import sample_q_start_end

ray.init(address='auto', log_to_driver=False)

file_stub = '/net/rmc-lx0062/home_local/tenh_jo/{}.db'


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

    obstacle_img = compressed2img(img_cmp=img_cmp, shape=par.world.shape, dtype=bool)
    initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=obstacle_img)
    q_start, q_end = sample_q_start_end(robot=par.robot, feasibility_check=lambda qq: feasibility_check(q=qq, par=par),
                                        acceptance_rate=gen.bee_rate)

    get_q0 = InitialGuess.path.q0_random_wrapper(robot=par.robot, n_multi_start=gen.n_multi_start,
                                                 n_waypoints=par.n_waypoints, order_random=True, mode='inner')
    q0 = get_q0(start=q_start, end=q_end)

    q00 = q0[:1]
    q0 = q0[1:]

    if verbose > 0:
        tic()

    df0 = __chomp(q0=q00, q_start=q_start, q_end=q_end, gd=gd, par=par, i_world=i_world, i_sample=i_sample)
    if df0.feasible_b[0]:
        if verbose > 0:
            toc(name=f"{par.robot.id}, {i_world}, {i_sample}")
        return df0

    df = __chomp(q0=q0, q_start=q_start, q_end=q_end, gd=gd, par=par, i_world=i_world, i_sample=i_sample)

    if verbose > 2:
        j = np.argmin(df.objective + (df.feasible == -1)*df.objective.max())
        robot_path_interactive(q=df.q[j], robot=par.robot,
                               kwargs_world=dict(img=obstacle_img, limits=par.world.limits))

    df = df0.append(df)
    if verbose > 0:
        toc(name=f"{par.robot.id}, {i_world}, {i_sample}")
    return df


def main(robot_id: str, iw_list=None, ra='append'):
    file = file_stub.format(robot_id)
    n_samples_per_world = 1000
    # worlds = get_values_sql(file=file, rows=np.arange(1000), table='worlds', columns='img_cmp', values_only=True)
    # print("# Worlds", len(worlds))
    # gen = init_par()
    # df = sample_path(gen=gen, i_world=0, i_sample=0, img_cmp=worlds[0], verbose=1)

    @ray.remote
    def sample_ray(_i_w: int, _i_s: int):
        _i_w = int(_i_w)
        img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)
        gen = init_par(robot_id=robot_id)
        return sample_path(gen=gen, i_world=_i_w, i_sample=_i_s, img_cmp=img_cmp, verbose=0)

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

    # tic()
    df2sql(df=df, file=file, table='paths', if_exists=ra)
    if ra == 'replace':
        vacuum(file=file)
    # toc(f'Time for appending {len(df)} rows')
    return df


def main_loop(robot_id):
    for i in range(10):
        worlds = np.arange(10000)
        for iw in np.array_split(worlds, len(worlds)//10):
            iw = iw.astype(int)
            print(f"{i}:  {min(iw)} - {max(iw)}", end="  |  ")
            tic()
            main(robot_id=robot_id, iw_list=iw, ra='append')
            toc()


if __name__ == '__main__':

    robot_id = 'JustinArm07'
    from wzk import tic, toc

    main(robot_id=robot_id, iw_list=[0], ra='replace')

    tic()
    main_loop(robot_id)
    toc()
