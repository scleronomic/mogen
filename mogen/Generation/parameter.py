import numpy as np

from rokin.Robots import *
from mopla.parameter import Parameter, GradientDescent

from wzk.gd.Optimizer import Naive


class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')


def init_par(robot_id: str):

    if robot_id == 'SingleSphere02':
        robot = SingleSphere02(radius=0.25)

    elif robot_id == 'StaticArm04':
        robot = StaticArm(n_dof=4, lengths=0.5, limits=np.deg2rad([-170, +170]))

    elif robot_id == 'JustinArm07':
        robot = JustinArm07()

    elif robot_id == 'Justin19':
        robot = Justin19()

    else:
        raise ValueError

    gen = __init_par(robot=robot)
    return gen


def __init_par(robot):

    bee_rate = 0.0
    n_multi_start = [[0, 1, 2, 3], [1, 10, 10, 10]]

    par = Parameter(robot=robot, obstacle_img=None)
    par.n_waypoints = 20

    par.check.obstacle_collision = True
    par.planning.obstacle_collision = True
    par.oc.n_substeps = 3
    par.oc.n_substeps_check = 3

    if isinstance(robot, Justin19):
        set_sc_on(par)

    gd = GradientDescent()
    gd.opt = Naive(ss=1)
    gd.n_processes = 1
    gd.n_steps = 100

    gd.return_x_list = False
    n0, n1 = gd.n_steps//2, gd.n_steps//3
    n2 = gd.n_steps - (n0 + n1)
    gd.clipping = np.concatenate([np.ones(n0)*np.deg2rad(3), np.ones(n1)*np.deg2rad(1), np.ones(n2)*np.deg2rad(0.1)])

    gen = Generation()
    gen.par = par
    gen.gd = gd
    gen.n_multi_start = n_multi_start
    gen.bee_rate = bee_rate

    return gen


def set_sc_on(par):
    par.check.self_collision = True
    par.planning.self_collision = True
    par.sc.n_substeps = 3
    par.sc.n_substeps_check = 3

