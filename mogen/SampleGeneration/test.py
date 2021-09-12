import numpy as np

from wzk.trajectory import inner2full
from wzk.gd.Optimizer import Naive
from wzk.image import compressed2img

from mopla import parameter
from rokin.Robots import StaticArm, Justin19
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql, get_values_sql
from mogen.SampleGeneration.sample_start_end import sample_q_start_end


# db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/Justin19.db'
db_file = '/Users/jote/Documents/Code/Python/DLR/mogen/Justin19.db'



class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')


def set_sc_on(par):
    par.check.self_collision = True
    par.planning.self_collision = True
    par.sc.n_substeps_check = 5
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
    toc(name=f"{par.robot.id}, {i_world}, {i_sample}")
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


par = init_par().par

worlds = get_values_sql(rows=np.arange(100), file=db_file, table='worlds', columns='img_cmp', values_only=True)

i_w, i_s, q0, q, o, f = get_values_sql(file=db_file, table='paths',
                                       rows=np.arange(1000),
                                       columns=['i_world', 'i_sample', 'q0', 'q', 'objective', 'feasible'],
                                       values_only=True)

q0 = q0.reshape((-1, 20, 19))
q = q.reshape((-1, 20, 19))

i = 77
from rokin.Vis import robot_3d
obstacle_img = compressed2img(img_cmp=worlds[i_w[i]], n_voxels=par.world.n_voxels, dtype=bool)
robot_3d.robot_path_interactive(q=q[i], robot=par.robot, mode='mesh',
                                img_mode='mesh',
                                obstacle_img=obstacle_img,
                                voxel_size=par.world.voxel_size, lower_left=par.world.limits[:, 0],
                                additional_frames=np.eye(4)[np.newaxis, :, :])

from rokin.Vis.configuration_space import q_path
q_path(q[i])

parameter.initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=obstacle_img)
feasibility_check(q=q[50:51], par=par)

i_w, i_s, q0, q, o, f = get_values_sql(file=db_file, table='paths',
                                       rows=np.arange(1000),
                                       columns=-1,
                                       values_only=True)

df2sql(df=df, file=db_file, table='paths', if_exists='append')