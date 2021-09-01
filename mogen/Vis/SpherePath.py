import numpy as np

from wzk.trajectory import get_substeps
import rokin.Vis.robot_2d as plt2
from rokin.forward import get_frames_x

import mopla.Optimizer.InitialGuess.path as path_i


def get_path_sample(robot, directory=None, i_sample_global=0):
    n_waypoints = 10

    if directory is None:
        q_start, q_end = robot.sample_q(2)
        q_path = path_i.q0_random(start=q_start, end=q_end, n_waypoints=n_waypoints, n_random_points=0,
                                  robot=robot, order_random=True)
    else:
        raise NotImplementedError
        # from mogen.definitions import START_Q, END_Q, PATH_Q, PATH_DB
        # q_start, q_end, q_path, = \
        #     ld_sql.get_values_sql(file=directory + PATH_DB,
        #                           columns=[START_Q, END_Q, PATH_Q], rows=i_sample_global, values_only=True)

    return q_start, q_end, q_path


class SpherePath:

    def __init__(self, *, i_sample=0, ax, file=None,
                 exp, gd, par):

        # Sample
        self.i_sample_global = i_sample

        # Optimization
        self.gd = gd
        self.par = par
        self.exp = exp

        # GridWorld obstacles
        self.directory = None  # get_sample_dir(directory=directory)

        self.q = None

        # Path of the whole arm
        self.x_spheres = None

        # Initialize plot
        if ax is None:
            self.fig, self.ax = plt2.new_world_fig(limits=self.par.world.limits)
        else:
            self.fig, self.ax = ax.get_figure(), ax

        self.change_sample(i_sample=self.i_sample_global)

        self.h_path = None
        self.q2x_spheres()

    def update_path(self, q_start, q_end):
        q_start = q_start.reshape((1, -1))
        q_end = q_end.reshape((1, -1))

        # linear
        print(q_start.shape)
        print(q_end.shape)
        self.q = get_substeps(x=np.concatenate([q_start, q_end], axis=0), n=self.par.n_waypoints - 1,
                              is_periodic=self.par.robot.infinity_joints, include_start=True)

        # random
        # optimizer
        # net

        self.q2x_spheres()

    def change_sample(self, i_sample):

        self.i_sample_global = i_sample
        q_start, q_end, self.q = get_path_sample(robot=self.par.robot, directory=self.directory,
                                                 i_sample_global=self.i_sample_global)
        return q_start, q_end

    def q2x_spheres(self):
        self.x_spheres = get_frames_x(q=self.q, robot=self.par.robot)

        if self.h_path is None:
            self.h_path = plt2.plot_x_spheres(x_spheres=self.x_spheres, ax=self.ax)
        else:
            plt2.plot_x_spheres_update(h=self.h_path, x_spheres=self.x_spheres)


def test():
    from mopla.parameter import Parameter
    par = Parameter(robot='StaticArm04')
    sp = SpherePath(par=par, exp=None, gd=None, ax=None)


if __name__ == '__main__':
    test()
