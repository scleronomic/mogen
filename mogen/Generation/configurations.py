import numpy as np

from wzk import tictoc
from wzk.dlr import LOCATION
from wzk.ray2 import ray, ray_init
from wzk.image import compressed2img
from wzk.sql2 import df2sql, get_values_sql, vacuum
from rokin.Robots.Justin19.justin19_primitives import justin_primitives

from mopla.main import ik_mp
from mopla.Parameter.parameter import initialize_oc
from mopla.Optimizer import feasibility_check, choose_optimum

from mogen.Generation import load, parameter

__file_stub_dlr = '/home_local/tenh_jo/ik_{}.db'
__file_stub_mac = '/Users/jote/Documents/DLR/Data/mogen/ik_{}.db'
__file_stub_gcp = '/home/johannes_tenhumberg_gmail_com/sdb/ik_{}.db'

file_stub_dict = dict(dlr=__file_stub_dlr, mac=__file_stub_mac, gcp=__file_stub_gcp)
file_stub = file_stub_dict[LOCATION]


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

#     robot_path_interactive(q=q[status == 1], robot=par.robot,
#                            kwargs_frames=dict(f_fix=par.xc.frame, f_idx_robot=par.xc.f_idx))


def adapt_par(par):
    par.xc.f_idx = 13
    par.check.obstacle_collision = False
    par.check.self_collision = True
    par.check.x_close = True
    par.check.center_of_mass = False

    par.plan.obstacle_collision = False
    par.plan.self_collision = True
    par.plan.x_close = False
    par.plan.center_of_mass = True

    par.qc.q = justin_primitives(justin='getready')
    par.weighting.joint_motion = np.array([200, 100, 100,
                                           20, 20, 10, 10, 1, 1, 1,
                                           20, 20, 10, 10, 1, 1, 1,
                                           5, 5], dtype=float)


def generate_ik(gen, img_cmp, i_world, n_samples):
    np.random.seed(None)
    obstacle_img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
    initialize_oc(par=gen.par, obstacle_img=obstacle_img)
    par = gen.par

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
            df = load.create_path_df(i_world=np.array([i_world]), i_sample=np.array([i]), q=q_opt,
                                     objective=np.array(([cost.min()])), feasible=np.ones(1, dtype=bool))
            df_list.append(df)

    return load.combine_df_list(df_list)


def main(robot_id, iw_list, n_samples_per_world=1000, ra='append'):
    file = file_stub.format(robot_id)

    @ray.remote
    def generate_ray(_i_w: int):
        gen = parameter.init_par(robot_id=robot_id)
        adapt_par(par=gen.par)
        _i_w = int(_i_w)
        if _i_w == -1:
            img_cmp = load.img_cmp0[gen.par.robot.n_dim-1]
        else:
            img_cmp = get_values_sql(file=file, rows=_i_w, table='worlds', columns='img_cmp', values_only=True)

        return generate_ik(gen=gen, i_world=_i_w, img_cmp=img_cmp, n_samples=n_samples_per_world)

    with tictoc(text=f"Generating samples for world {iw_list[0]}-{iw_list[-1]} with {n_samples_per_world} samples") as _:
        futures = []
        for i_w in iw_list:
            futures.append(generate_ray.remote(i_w))

        df_list = ray.get(futures)
        df = load.combine_df_list(df_list)

    with tictoc(text=f"Saving {len(df)} new samples", verbose=(1, 2)) as _:
        df2sql(df=df, file=file, table='paths', if_exists=ra)
        if ra == 'replace':
            vacuum(file=file)
    return df


def main_loop_sc(robot_id):
    for i in range(10000):
        worlds = [-1] * 60
        with tictoc(f"loop {i}") as _:
            main(robot_id=robot_id, iw_list=worlds, n_samples_per_world=10, ra='append',)


if __name__ == '__main__':
    ray_init(perc=100)

    _robot_id = 'Justin19'

    with tictoc() as _:
        main_loop_sc(robot_id=_robot_id)

    # TODO add good weighting for q close
