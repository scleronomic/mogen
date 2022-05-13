import numpy as np

from wzk.strings import find_one_of_n

from rokin.Robots.Justin19.justin19_primitives import justin_primitives
from mopla.Parameter import get_par_justin19, get_par_justinarm07, get_par_staticarm, get_par_singlesphere02


class Generation:
    __slots__ = ('par',
                 'gd',
                 'staircase',
                 'bee_rate')


__robots = ['SingleSphere02', 'StaticArm04', 'JustinArm07', 'Justin19']


def get_robot_str(s):
    return find_one_of_n(s=s, n=__robots)


def init_par(robot_id: str):

    if robot_id == 'SingleSphere02':
        par, gd, staircase = get_par_singlesphere02(radius=0.25)

    elif robot_id == 'StaticArm04':
        par, gd, staircase = get_par_staticarm(n_dof=4, lengths=0.25, widths=0.1)

    elif robot_id == 'JustinArm07':
        pass
        par, gd, staircase = get_par_justinarm07()

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

    gen.gd.hesse_inv = gen.staircase.P_inv_dict[par.n_wp]

    return gen


def adapt_ik_par(par):
    if par.robot.id == 'Justin19':

        par.check.x_close = True
        par.check.obstacle_collision = False
        par.check.self_collision = True
        par.check.center_of_mass = False
        par.check.limits = True

        par.plan.x_close = False
        par.plan.obstacle_collision = False
        par.plan.self_collision = True
        par.plan.center_of_mass = True

        par.xc.f_idx = 13

        par.qc.q = justin_primitives(justin='getready')

        par.weighting.joint_motion = np.array([200, 100, 100,
                                               20, 20, 10, 10, 1, 1, 1,
                                               20, 20, 10, 10, 1, 1, 1,
                                               5, 5], dtype=float)
