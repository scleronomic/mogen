import numpy as np

from rokin.Vis import robot_2d, robot_3d
from rokin.forward import get_frames_x

import mopla.Optimizer.InitialGuess.path as path_i

from mogen.Generation import data


def get_path_sample(par, file=None, i=0):
    if file is None:
        q_start, q_end = par.robot.sample_q(2)
        q = path_i.q0_linear_uniform(start=q_start, end=q_end, n_wp=par.n_wp, n_random_points=0,
                                     robot=par.robot, order_random=True)
    else:
        i_w, i_s, q, o, f = data.get_paths(file=file, i=i)
        q = np.reshape(q, (par.n_wp, par.robot.n_dof))

    return q


class PathViewer:

    def __init__(self, *, i_sample=0, ax=None, file=None,
                 par, gd, **kwargs):

        self.file = file
        self.i_sample = i_sample

        self.par = par
        self.gd = gd

        self.kwargs = kwargs
        self.q = None
        self.x = None

        # Initialize plot
        if ax is None:
            self.fig, self.ax = robot_2d.new_world_fig(limits=self.par.world.limits)
        else:
            self.fig, self.ax = ax.get_figure(), ax

        self.h_path = None
        self.change_sample(i_sample=self.i_sample)

    def update_path(self, q_start, q_end, q):
        q_start = q_start
        q_end = q_end

        if q is None:
            self.q = path_i.q0_linear_uniform(start=q_start, end=q_end, n_wp=self.par.n_wp, n_random_points=0,
                                              robot=self.par.robot, order_random=True)

            # random  # TODO add different modes
            # optimizer
            # net

        else:
            self.q = q

            assert np.allclose(q[0], q_start), f"{q[0]}{q_start}"
            assert np.allclose(q[-1], q_end), f"{q[-1]}{q_end}"

        self.plot()

    def change_sample(self, i_sample):
        self.i_sample = i_sample
        self.q = get_path_sample(par=self.par, file=self.file, i=self.i_sample)
        self.plot()

    def plot(self):
        self.x = get_frames_x(q=self.q, robot=self.par.robot)
        self.h_path = robot_2d.plot_path(q=self.q, par=self.par, ax=self.ax, h=self.h_path, **self.kwargs)[1]


def test():
    from mopla.Parameter.parameter import Parameter
    par = Parameter(robot='StaticArm04')
    pv = PathViewer(par=par, exp=None, gd=None, ax=None)


if __name__ == '__main__':
    test()
