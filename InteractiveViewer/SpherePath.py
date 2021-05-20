import numpy as np

import Kinematic.forward as forward

import Optimizer.InitialGuess.path as path_i
import Optimizer.path as path
import Visualization.plotting_2 as plt2


from definitions import START_Q, END_Q, PATH_Q, PATH_DB


def get_path_sample(robot, directory=None, i_sample_global=0):
    n_waypoints = 10

    if directory is None:
        q_start, q_end = robot.sample_q(2)
        q_path = path_i.q0_random(start=q_start, end=q_end, n_waypoints=n_waypoints, n_random_points=0,
                                  robot=robot, order_random=True)
    else:
        raise NotImplementedError
        # q_start, q_end, q_path, = \
        #     ld_sql.get_values_sql(file=directory + PATH_DB,
        #                           columns=[START_Q, END_Q, PATH_Q], rows=i_sample_global, values_only=True)

    return q_start, q_end, q_path


class SpherePath:

    def __init__(self, *, i_sample=0, ax, directory=None,
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
        q_start = q_start.reshape((1, 1, -1))
        q_end = q_end.reshape((1, 1, -1))
        pass
        # linear
        # random
        # optimizer
        # net

        # linear
        self.q = path.get_substeps(q=np.concatenate([q_start, q_end], axis=1), n=self.par.size.n_waypoints - 1,
                                   infinity_joints=self.par.robot.infinity_joints, include_start=True)

        self.q2x_spheres()

    def change_sample(self, i_sample):

        self.i_sample_global = i_sample
        q_start, q_end, self.q = get_path_sample(robot=self.par.robot, directory=self.directory,
                                                 i_sample_global=self.i_sample_global)
        return q_start, q_end

    def q2x_spheres(self):
        self.x_spheres = forward.get_frames_x(q=self.q, robot=self.par.robot)[0]

        print(self.x_spheres.shape)
        if self.h_path is None:
            self.h_path = plt2.plot_x_spheres(x_spheres=self.x_spheres, ax=self.ax)
        else:
            plt2.plot_x_spheres_update(h=self.h_path, x_spheres=self.x_spheres)


def test():
    from parameter import Parameter
    par = Parameter(robot='SingleSphere02')

    sp = SpherePath(par=par, exp=None, gd=None, ax=None)


if __name__ == '__main__':
    test()
