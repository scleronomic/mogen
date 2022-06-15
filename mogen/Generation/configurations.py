import numpy as np

from wzk import spatial, sql2, image, tictoc
from wzk.ray2 import ray, ray_init

from mopla.main import ik_mp, set_free_joints2close
from mopla.Optimizer import feasibility_check, choose_optimum
from mopla.Automatica import scenes, lut

from mogen.Generation import data, parameter


def sample_f(robot, f_idx, n=None, mode='q', i_world=None):

    if mode == 'q':
        q = robot.sample_q(shape=n)
        f = robot.get_frames(q)
        f = f[..., f_idx, :, :]

    elif mode == 'automatica_cube4':  # TODO write general with limits in xyz, alphabetagamma and use automatica as a special case
        assert n % 4 == 0
        n = n // 4

        scene = scenes.CubeScene(shape=(128, 128, 128), n_cubes=n * 4)
        x, a = scene.sample_cube_frames(n=n)
        x = np.repeat(x, repeats=4, axis=0)
        a = np.repeat(a, repeats=4, axis=0) + np.tile([0, np.pi/2, np.pi, 3*np.pi/2], reps=n)
        a = spatial.angle2minuspi_pluspi(a)
        xa = np.concatenate((x, a), axis=-1)
        f = scene.xa_cube2f_tcp(xa)

    elif mode == 'automatica_lut':
        _lut = lut.IKTableFull(_lut=None)
        x = _lut.sample_bin_centers()
        x = x.reshape((-1,) + x.shape[4:])
        x = x[i_world].reshape(-1, x.shape[-1])
        # x = x[np.random.choice(np.arange(len(x)), size=3, replace=False)]
        f = spatial.trans_rotvec2frame(trans=x[:, :3], rotvec=x[:, 3:])

    else:
        raise ValueError

    return f


def generate_ik(gen, img_cmp, i_world, n_samples, sample_mode):
    np.random.seed(None)
    img = image.compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    par = gen.par
    par.update_oc(img=img)

    f = sample_f(robot=par.robot, f_idx=par.xc.f_idx, n=n_samples, mode=sample_mode, i_world=i_world)

    df_list = []
    for i, fi in enumerate(f):
        m = 1000
        par.xc.frame = fi
        q = ik_mp(par=par, qclose=par.qc.q, n_samples=m, n_iter=13, n_processes=1, mode=None)

        q = np.tile(q, reps=(2, 1))  # TODO do this everywhere, can be done more efficient with feasibility check
        set_free_joints2close(q[:m], par=par, qclose=par.qc.q)

        status = feasibility_check(q=q[:, np.newaxis, :], par=par, verbose=0)

        q_opt, _, mce, cost = choose_optimum.get_feasible_optimum(q=q[:, np.newaxis, :], status=status,
                                                                  par=par, qclose=par.qc.q, mode='min')

        print((status == 1).sum())
        if np.any(status == 1):
            df = data.create_path_df(i_world=np.array([i_world]), i_sample=np.array([i]),
                                     q=[q_opt.ravel()], objective=np.array(([cost.min()])), feasible=np.ones(1, dtype=bool))
            df_list.append(df)

    return data.combine_df_list(df_list)


def main(robot_id, iw_list, sample_mode, par_mode, n_samples_per_world=1000, ra='append'):
    file = data.get_file_ik(robot_id=robot_id, copy=False)

    @ray.remote
    def generate_ray(_i_w: int):
        gen = parameter.init_par(robot_id=robot_id)
        parameter.adapt_ik_par_justin19(par=gen.par, mode=par_mode)

        _i_w = int(_i_w)
        # if _i_w == -1:
        img_cmp = data.img_cmp0[gen.par.robot.n_dim-1]
        # else:
        #     img_cmp = sql2.get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return generate_ik(gen=gen, i_world=_i_w, img_cmp=img_cmp, n_samples=n_samples_per_world, sample_mode=sample_mode)

    with tictoc(text=f"Generating samples for world {iw_list[0]}-{iw_list[-1]} with {n_samples_per_world} samples") as _:
        # generate_ray(-1)
        futures = []
        for i_w in iw_list:
            futures.append(generate_ray.remote(i_w))

        df_list = ray.get(futures)
        df = data.combine_df_list(df_list)

    if df is None:
        print('No Samples Generated')
        return None

    with tictoc(text=f"Saving {len(df)} new samples", verbose=(1, 2)) as _:
        sql2.df2sql(df=df, file=file, table=data.T_PATHS(), dtype=data.T_IKS.types_dict_sql(), if_exists=ra)
        if ra == 'replace':
            sql2.vacuum(file=file)

    return df


def main_loop_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 600
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=200, ra='append',
                 sample_mode='q', par_mode=None)


def main_loop_automatica_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 20
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=200, ra='append',
                 sample_mode='automatica_cube4', par_mode='table')


def main_loop_table_lut(robot_id):
    _lut = lut.IKTableFull(_lut=None)
    print(_lut.n)
    n_xyz = np.prod(_lut.n[:4])
    worlds = np.arange(n_xyz)
    worlds = np.array_split(worlds, 168)
    print(n_xyz, len(worlds), len(worlds[0]))

    for i, w in enumerate(worlds):
        main(robot_id=robot_id, iw_list=w, n_samples_per_world=1000, ra='append',
             sample_mode='automatica_lut', par_mode=('table', 'left'))


def test_sample_f():
    from rokin.Robots import Justin19
    robot = Justin19
    f = sample_f(robot=robot, f_idx=13, n=100, mode='automatica_table_right', i_world=0)
    print(f.shape)


if __name__ == '__main__':
    ray_init(perc=100)
    _robot_id = 'Justin19'
    main_loop_table_lut(robot_id=_robot_id)

    # with tictoc() as _:
    #     main_loop_automatica_sc(robot_id=_robot_id)
