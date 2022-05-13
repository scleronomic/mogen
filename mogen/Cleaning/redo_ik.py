import numpy as np
# import fire

from wzk import sql2, tictoc
from wzk.ray2 import ray, ray_init

from mopla.Optimizer import feasibility_check, objectives
from mopla.main import ik_w_projection

from mogen.Generation import data, parameter
from mogen.Cleaning import redo

from rokin.Robots.Justin19.justin19_primitives import justin_primitives


def recalculate_objective(file, par, i, mode):
    i, q, img = data.get_samples_for_world(file=file, par=par, i=i)
    q = q[..., 0, :]
    o = objectives.o_len.len_close2q_cost(q=q, q_close=par.qc.q, is_periodic=par.robot.is_periodic,
                                          joint_weighting=par.weighting.joint_motion)

    sql2.set_values_sql(file=file, table=data.T_PATHS, rows=i, values=(o,),
                        columns=[data.C_OBJECTIVE_F])


def refine_omp(file, par, gd,
               q_fun=None, i=None,
               verbose=0, mode=None):
    i, q0, img = data.get_samples_for_world(file=file, par=par, i=i)
    q0 = q0[..., 0, :]
    n = len(i)

    if q_fun is not None:
        q_pred = np.squeeze(q_fun(i=i))

    else:
        q_pred = q0

    frames0 = par.robot.get_frames(q0)[..., par.xc.f_idx, :, :]
    f0 = np.zeros(n, dtype=bool)
    f1 = np.zeros(n, dtype=bool)
    q1 = q_pred.copy()

    for j in range(n):
        par.xc.frame = frames0[j]
        q1[j] = ik_w_projection(q=q1[j:j + 1, np.newaxis, :], par=par, gd=gd)
        f1[j] = feasibility_check(q=q1[j:j + 1, np.newaxis, :], par=par)
        f0[j] = feasibility_check(q=q0[j:j + 1, np.newaxis, :], par=par)

    o0 = objectives.o_len.len_close2q_cost(q=q0, q_close=par.qc.q, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)
    o1 = objectives.o_len.len_close2q_cost(q=q1, q_close=par.qc.q, is_periodic=par.robot.is_periodic, joint_weighting=par.weighting.joint_motion)

    b_fb, b_nfb, b_rest = redo.get_b_improvements(o0=o0, o1=o1, f0=f0, f1=f1)

    redo.print_improvements(o0=o0, o1=o1, f0=f0, f1=f1, b_fb=b_fb, verbose=verbose)
    if verbose > 10:
        pass  # TODO plot

    q1[b_rest] = q0[b_rest]
    o1[b_rest] = o0[b_rest]
    f1[b_rest] = f0[b_rest]

    par.check.obstacle_collision = False
    par.check.self_collision = False
    par.check.center_of_mass = False
    par.check.x_close = False
    par.check.limits = True
    f = feasibility_check(q=q1[:, np.newaxis], par=par,)
    print('limits', (f == -3).mean())


    if mode is None:
        print('no set_values')

    elif mode == 'set_sql':
        sql2.set_values_sql(file=file, table=data.T_PATHS, rows=i, values=(q1, o1, f1),
                            columns=[data.C_Q_F32, data.C_OBJECTIVE_F, data.C_FEASIBLE_I])

    elif mode == 'save_numpy':
        directory_np = redo.file2numpy_directory(file=file)
        file2 = f"{directory_np}/s_{min(i)}-{max(i)}"
        np.savez(file2, i=i, q=q1, o=o1, f=f1)

    else:
        raise ValueError


def main_refine_chomp(robot_id, q_fun=None, ray_perc=100, mode=None):
    ray_init(perc=ray_perc)

    file = data.get_file_ik(robot_id=robot_id)
    n = sql2.get_n_rows(file=file, table=data.T_PATHS)

    redo.create_numpy_directory(file=file, replace=True)
    q_fun_ray0 = ray.put(q_fun)

    @ray.remote
    def refine_ray(q_fun_ray, i):
        gen = parameter.init_par(robot_id)
        par, gd = gen.par, gen.gd
        parameter.adapt_ik_par(par=par)
        adapt_gd(gd=gd)

        with tictoc(text=f'World {min(i)}') as _:
            refine_omp(file=file, par=par, gd=gd, q_fun=q_fun_ray, i=i, verbose=1, mode=mode)

        return 1

    futures = []
    for ii in np.array_split(np.arange(n), n // 500):
        futures.append(refine_ray.remote(q_fun_ray0, ii))

    res = ray.get(futures)
    print(f"{np.sum(res)} / {np.size(res)}")


def adapt_gd(gd):
    gd.n_steps = 20
    gd.stepsize = 1/50
    gd.clipping = 0.3


def main(robot_id):

    file = data.get_file_ik(robot_id=robot_id)

    gen = parameter.init_par(robot_id)
    par, gd = gen.par, gen.gd

    with tictoc() as _:
        i = np.arange(1000)
        for j in np.array_split(i, len(i) // 200):
            refine_omp(file=file, par=par, gd=gd, q_fun=None, i=j, verbose=2, mode=None)


if __name__ == '__main__':
    _robot_id = 'Justin19'
    main_refine_chomp(robot_id=_robot_id, q_fun=None, ray_perc=100, mode='save_numpy')
    redo.tmp_numpy2sql(file=data.get_file_ik(robot_id=_robot_id))


