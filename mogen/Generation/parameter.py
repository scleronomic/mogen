import numpy as np

from rokin.Robots import *
from mopla.Parameter import get_par_justin19, get_par_justinarm07


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
        par, gd, staircase = get_par_justinarm07()

    elif robot_id == 'Justin19':
        par, gd, staircase = get_par_justin19()

    else:
        raise ValueError

    bee_rate = 0.0

    # gen = __init_par(robot=robot)
    gen = Generation()
    gen.par = par
    gen.gd = gd
    gen.gd.use_loop_instead_of_processes = True
    gen.staircase = staircase
    gen.bee_rate = bee_rate

    return gen
