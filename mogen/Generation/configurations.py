import numpy as np

from wzk import tictoc
from wzk.math2 import angle2minuspi_pluspi
from wzk.ray2 import ray, ray_init
from wzk.image import compressed2img
from wzk.sql2 import df2sql, get_values_sql, vacuum
from wzk.spatial import frame2trans_rotvec

from mopla.main import ik_mp, set_free_joints2close
from mopla.Optimizer import feasibility_check, choose_optimum
from mopla.World import automatica2022

from mogen.Generation import data, parameter


def sample_f(robot, f_idx, n=None, mode='q'):

    if mode == 'q':
        q = robot.sample_q(shape=n)
        f = robot.get_frames(q)
        f = f[..., f_idx, :, :]

    elif mode == 'automatica_cube4':
        assert n % 4 == 0
        n = n // 4

        scene = automatica2022.setup_table_scene(shape=(256, 256, 256))
        cubes = automatica2022.Cubes(n=n*4)
        x = np.random.uniform(low=scene.table[0, 0], high=scene.table[0, 1], size=n)
        y = np.random.uniform(low=scene.table[1, 0], high=scene.table[1, 1], size=n)
        z = np.random.uniform(low=scene.table[2, 1]+0.04, high=scene.table[2, 1]+0.54, size=n)
        x = np.vstack(([x, y, z])).T
        x = np.repeat(x, repeats=4, axis=0)

        a = np.random.uniform(0, 2*np.pi, size=n)
        a = np.repeat(a, repeats=4, axis=0) + np.tile([0, np.pi/2, np.pi, 3*np.pi/2], reps=n)
        a = angle2minuspi_pluspi(a)

        cubes.x = x
        cubes.a = a
        f = automatica2022.xa_to_f(cubes)
        f = f @ automatica2022.F_CUBE_HAND
    else:
        raise ValueError

    return f


def generate_ik(gen, img_cmp, i_world, n_samples, sample_mode):
    np.random.seed(None)
    img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    par = gen.par
    par.update_oc(img=img)

    f = sample_f(robot=par.robot, f_idx=par.xc.f_idx, n=n_samples, mode=sample_mode)
    df_list = []
    for i in range(n_samples):
        m = 1000
        par.xc.frame = f[i]
        q = ik_mp(par=par, q_close=par.qc.q, n=m, n_processes=1, mode=None)

        q = np.tile(q, reps=(2, 1))  # TODO do this everywhere, can be done more efficient with feasibility check
        set_free_joints2close(q[:m], par=par, q_close=par.qc.q)

        status = feasibility_check(q=q[:, np.newaxis, :], par=par, verbose=0)

        q_opt, _, mce, cost = choose_optimum.get_feasible_optimum(q=q[:, np.newaxis, :], status=status,
                                                                  par=par, q_close=par.qc.q, mode='min')

        print((status == 1).sum())
        if np.any(status == 1):
            xrv = np.array(frame2trans_rotvec(f=f[i]))
            df = data.create_ik_df(i_world=np.array([i_world]), i_sample=np.array([i]),
                                   q=[q_opt.ravel()], f=[xrv.ravel()],
                                   objective=np.array(([cost.min()])), feasible=np.ones(1, dtype=bool))
            df_list.append(df)

    return data.combine_df_list(df_list)


def main(robot_id, iw_list, sample_mode, n_samples_per_world=1000, ra='append'):
    file = data.get_file_ik(robot_id=robot_id, copy=False)

    @ray.remote
    def generate_ray(_i_w: int):
        gen = parameter.init_par(robot_id=robot_id)
        parameter.adapt_ik_par(par=gen.par, mode='main_loop_automatica_sc')

        _i_w = int(_i_w)
        if _i_w == -1:
            img_cmp = data.img_cmp0[gen.par.robot.n_dim-1]
        else:
            img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return generate_ik(gen=gen, i_world=_i_w, img_cmp=img_cmp, n_samples=n_samples_per_world, sample_mode=sample_mode)

    with tictoc(text=f"Generating samples for world {iw_list[0]}-{iw_list[-1]} with {n_samples_per_world} samples") as _:
        # generate_ray(-1)
        futures = []
        for i_w in iw_list:
            futures.append(generate_ray.remote(i_w))

        df_list = ray.get(futures)
        df = data.combine_df_list(df_list)

    with tictoc(text=f"Saving {len(df)} new samples", verbose=(1, 2)) as _:
        df2sql(df=df, file=file, table=data.T_IKS.table, dtype=data.T_IKS.types_dict_sql(), if_exists=ra)
        if ra == 'replace':
            vacuum(file=file)
    return df


def main_loop_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 600
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=200, ra='append', sample_mode='q')


def main_loop_automatica_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 20
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=40, ra='append',
                 sample_mode='automatica_cube4')


if __name__ == '__main__':
    ray_init(perc=50)
    _robot_id = 'Justin19'

    with tictoc() as _:
        main_loop_automatica_sc(robot_id=_robot_id)
