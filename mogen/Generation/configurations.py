import numpy as np

from wzk import tictoc
from wzk.ray2 import ray, ray_init
from wzk.image import compressed2img
from wzk.sql2 import df2sql, get_values_sql, vacuum

from mopla.main import ik_mp
from mopla.Optimizer import feasibility_check, choose_optimum

from mogen.Generation import data, parameter


def sample_f(robot, f_idx, n=None, mode='q'):

    if mode == 'q':
        q = robot.sample_q(shape=n)
        f = robot.get_frames(q)
        f = f[..., f_idx, :, :]
    else:
        raise ValueError

    return f


def redo():
    pass
    # print((status == 1).sum())
    # q2 = q.copy()
    # for j in range(10):
    #     j = objectives.ik_grad(par=par, q=q2, q_close=q_close, jac_only=True)
    #     q2 = nullspace.nullspace_projection2(robot=par.robot, q=q2, u=-j,
    #                                          f0=par.xc.frame, f_idx=par.xc.f_idx, mode='f')
    # status = feasibility_check(q=q2[:, np.newaxis, :], par=par, verbose=0)


def generate_ik(gen, img_cmp, i_world, n_samples):
    np.random.seed(None)
    img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    par = gen.par
    par.update_oc(img=img)

    f = sample_f(robot=par.robot, f_idx=par.xc.f_idx, n=n_samples, mode='q')
    df_list = []
    for i in range(n_samples):
        par.xc.frame = f[i]
        q = ik_mp(par=par, q_close=par.qc.q, n=1000, n_processes=1, mode=None)

        status = feasibility_check(q=q[:, np.newaxis, :], par=par, verbose=0)
        q_opt, _, mce, cost = choose_optimum.get_feasible_optimum(q=q[:, np.newaxis, :], status=status,
                                                                  par=par, q_close=par.qc.q, mode='min')

        print((status == 1).sum())
        if np.any(status == 1):
            df = data.create_path_df(i_world=np.array([i_world]), i_sample=np.array([i]), q=[q_opt.ravel()],
                                     objective=np.array(([cost.min()])), feasible=np.ones(1, dtype=bool))
            df_list.append(df)

    return data.combine_df_list(df_list)


def main(robot_id, iw_list, n_samples_per_world=1000, ra='append'):
    file = data.get_file_ik(robot_id=robot_id, copy=False)

    @ray.remote
    def generate_ray(_i_w: int):
        gen = parameter.init_par(robot_id=robot_id)
        parameter.adapt_ik_par(par=gen.par)
        _i_w = int(_i_w)
        if _i_w == -1:
            img_cmp = data.img_cmp0[gen.par.robot.n_dim-1]
        else:
            img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return generate_ik(gen=gen, i_world=_i_w, img_cmp=img_cmp, n_samples=n_samples_per_world)

    with tictoc(text=f"Generating samples for world {iw_list[0]}-{iw_list[-1]} with {n_samples_per_world} samples") as _:
        # generate_ray(-1)
        futures = []
        for i_w in iw_list:
            futures.append(generate_ray.remote(i_w))

        df_list = ray.get(futures)
        df = data.combine_df_list(df_list)

    with tictoc(text=f"Saving {len(df)} new samples", verbose=(1, 2)) as _:
        df2sql(df=df, file=file, table='paths', if_exists=ra)
        if ra == 'replace':
            vacuum(file=file)
    return df


def main_loop_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 600
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=200, ra='append',)


if __name__ == '__main__':
    ray_init(perc=50)

    _robot_id = 'Justin19'

    with tictoc() as _:
        main_loop_sc(robot_id=_robot_id)
