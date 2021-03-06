import numpy as np

from wzk.mpl import DraggableCircleList
from wzk.trajectory import periodic_dof_wrapper

import rokin.Vis.robot_2d as plt2


def abs2rel_angles(q):
    q %= 2 * np.pi
    q[..., 1:] = q[..., 1:] - q[..., :-1]
    periodic_dof_wrapper(x=q, is_periodic=np.ones(q.shape[-1], dtype=bool))
    return q


def get_x_frames_distances(robot):
    x = robot.get_frames(q=np.zeros(robot.n_dof))[..., :-1, -1]
    x = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    return x


def spheres2joints(x_spheres, robot):
    lengths = get_x_frames_distances(robot=robot)

    n_dim = 2
    n_dof = robot.n_dof
    n_spheres = x_spheres.size // n_dim
    x_spheres = x_spheres.reshape((n_spheres, n_dim))

    q = np.zeros(n_dim + n_dof)
    q[:n_dim] = x_spheres[0]

    for i in range(n_spheres-1):
        # Ensure that points satisfy the constraints of the arm
        step = x_spheres[i + 1, :] - x_spheres[robot.prev_f_idx[i + 1], :]
        dist = np.linalg.norm(step)
        r = lengths[i]
        x_spheres[i + 1, :] -= step * (dist - r) / dist

        # Get the angles from the configuration
        q[n_dim + i] = np.arctan2(step[1], step[0])

    q[n_dim:] = abs2rel_angles(q=q[n_dim:])

    if robot.id == 'SingleSphere02':
        q = q[:n_dim]
    elif 'StaticArm' in robot.id:
        q = q[n_dim:]
    else:
        raise ValueError

    return q, x_spheres


class DraggableSphereRobot:
    def __init__(self, q, ax, robot, **kwargs):

        self.robot = robot
        self.q = np.squeeze(q)
        self.x = None
        self.ax = ax

        self.kwargs = kwargs
        self.callback = kwargs.pop('callback', None)
        self.h = None

        self.update_q2x()
        self.drag_circles = DraggableCircleList(ax=self.ax, xy=self.x,
                                                radius=self.robot.spheres_rad.mean(),
                                                callback=self.callback, **self.kwargs)
        if 'StaticArm' in robot.id:
            self.drag_circles.dp_list[0].vary_xy = (False, False)

        self.drag_circles.add_callback_drag(callback=self.update_x2q_plot)
        self.update_x2q_plot()

    def update_val2plot(self):
        self.h = plt2.plot_path(q=self.q[np.newaxis, :], par=self.robot, ax=self.ax, h=self.h, **self.kwargs)[1]
        self.drag_circles.set_xy(x=self.x[..., 0].flatten(), y=self.x[..., 1].flatten())

    def update_q2x(self):
        self.x = self.robot.get_frames(q=self.q.flatten())[..., :-1, -1]

    def update_x2q(self):
        self.x = self.drag_circles.get_xy()
        self.q, self.x = spheres2joints(x_spheres=self.x, robot=self.robot)

    def update_x2q_plot(self, *args):
        self.update_x2q()
        self.update_val2plot()

    def update_q2spheres_plot(self):
        self.update_q2x()
        self.update_val2plot()

    def set_q(self, q):
        self.q = np.squeeze(q)
        self.update_q2spheres_plot()

    def get_q(self):
        self.update_x2q_plot()
        return self.q

    def set_callback_drag(self, callback):
        self.drag_circles.set_callback_drag(callback=callback)

    def add_callback_drag(self, callback):
        self.drag_circles.add_callback_drag(callback=callback)

    def toggle_visibility(self, value=None):
        v = self.drag_circles.toggle_visibility(value=value)
        for hh in self.h[0]:
            hh.set_visible(v)
        for hh in self.h[1]:
            hh.set_visible(v)

        # self.set_visible(not self.get_visible())


def test():
    from wzk.mpl import new_fig
    from rokin.Robots import StaticArm

    robot = StaticArm(n_dof=4)
    fig, ax = new_fig(aspect=1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    dsr = DraggableSphereRobot(q=robot.sample_q(), ax=ax, robot=robot)

    def cb(*args):  # noqa
        pass
        # print(dsr.get_q())

    dsr.add_callback_drag(cb)


if __name__ == '__main__':
    test()
