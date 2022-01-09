import numpy as np

# from rokin.Robots import *
# from mopla.Parameter.Justin19 import get_par_justin19
# from mopla.Parameter.JustinArm07 import get_par_justinarm07

# import numpy as np
from wzk.gd.Optimizer import Naive
from rokin.Robots.Justin19 import Justin19
from mopla.Optimizer.length import get_len_q_cost_Pinv_dict
from mopla.Parameter import parameter


def get_par_justin19():
    robot = Justin19()

    par = parameter.Parameter(robot=robot, obstacle_img=None)

    par.n_waypoints = 20
    par.plan.obstacle_collision = True
    par.plan.length = True
    par.plan.self_collision = True
    par.plan.center_of_mass = False
    par.plan.x_close = False

    par.check.obstacle_collision = True
    par.check.self_collision = True
    par.check.x_close = False
    par.check.center_of_mass = False

    par.weighting.collision = 10
    par.weighting.length = 1

    par.oc.n_substeps = 3
    par.sc.n_substeps = 3
    par.oc.n_substeps_check = 3
    par.sc.n_substeps_check = 3

    # par.sc.dist_threshold = -0.015
    # par.oc.dist_threshold = -0.015

    parameter.initialize_sc(par=par)

    gd = parameter.GradientDescent()
    gd.opt = Naive(ss=1)
    gd.n_steps = 10
    gd.stepsize = 1
    gd.clipping = np.linspace(np.deg2rad(10), np.deg2rad(10), gd.n_steps)
    gd.n_processes = 10

    staircase = parameter.GDStaircase()
    staircase.n_multi_start = [[0, 1, 2, 3], [1, 11, 10, 10]]
    staircase.n_steps = [10, 10, 10, 10, 10, 20]
    staircase.clipping = np.deg2rad([10, 5, 5, 3, 1, 0.5])
    staircase.n_wp = [5, 5, 5, 10, 15, 20]
    staircase.P_inv_dict = get_len_q_cost_Pinv_dict(par=par, n=[5, 10, 15, 20])
    staircase.substeps_dict = {5: (3, 1),
                               10: (2, 1),
                               15: (2, 1),
                               20: (2, 1)}

    return par, gd, staircase


class Generation:
    __slots__ = ('par',
                 'gd',
                 'staircase',
                 'bee_rate')


def init_par(robot_id: str):

    if robot_id == 'SingleSphere02':
        robot = SingleSphere02(radius=0.25)
        raise NotImplementedError

    elif robot_id == 'StaticArm04':
        robot = StaticArm(n_dof=4, lengths=0.5, limits=np.deg2rad([-170, +170]))
        raise NotImplementedError

    elif robot_id == 'JustinArm07':
        pass
        # par, gd, staircase = get_par_justinarm07()

    elif robot_id == 'Justin19':
        par, gd, staircase = get_par_justin19()

    else:
        raise ValueError

    bee_rate = 0.0

    gen = Generation()
    gen.par = par
    gen.gd = gd
    gen.gd.use_loop_instead_of_processes = True
    gen.staircase = staircase
    gen.bee_rate = bee_rate
    return gen
