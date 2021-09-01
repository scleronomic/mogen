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
    limb_lengths = get_x_frames_distances(robot=robot)

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
        r = limb_lengths[i]
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
    def __init__(self, q, ax, robot, style_path, style_arm):

        self.q = np.squeeze(q)
        self.ax = ax
        self.robot = robot
        self.callback = None
        self.x_spheres = None

        self.update_q2spheres()
        self.drag_circles = DraggableCircleList(ax=self.ax, xy=self.x_spheres,
                                                radius=self.robot.spheres_rad.mean(),
                                                callback=self.callback, **style_path)
        self.drag_circles.set_callback(callback=self.update_spheres2q_plot)
        self.x_spheres_plot_h = plt2.plot_x_spheres(x_spheres=self.x_spheres, ax=self.ax, style_path=style_path,
                                                    style_arm=style_arm)

    def update_val2plot(self):
        plt2.plot_x_spheres_update(h=self.x_spheres_plot_h, x_spheres=self.x_spheres)
        self.drag_circles.set_xy(x=self.x_spheres[..., 0].flatten(), y=self.x_spheres[..., 1].flatten())

    def update_q2spheres(self):
        self.x_spheres = self.robot.get_frames(q=self.q.flatten())[..., :-1, -1]

    def update_spheres2q(self):
        self.x_spheres = self.drag_circles.get_xy()
        self.q, self.x_spheres = spheres2joints(x_spheres=self.x_spheres, robot=self.robot)

    def update_spheres2q_plot(self):
        self.update_spheres2q()
        self.update_val2plot()

    def update_q2spheres_plot(self):
        self.update_q2spheres()
        self.update_val2plot()

    def set_q(self, q):
        self.q = np.squeeze(q)
        self.update_q2spheres_plot()

    def get_q(self):
        self.update_spheres2q_plot()
        return self.q

    def set_callback(self, callback):
        self.drag_circles.set_callback(callback=callback)

    def add_callback(self, callback):
        self.drag_circles.add_callback(callback=callback)


def test():
    from wzk.mpl import new_fig
    from rokin.Robots import StaticArm

    robot = StaticArm(n_dof=4)
    fig, ax = new_fig(aspect=1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    dsr = DraggableSphereRobot(q=robot.sample_q(), ax=ax, robot=robot, style_arm=dict(color='k', lw=3), style_path={})

    def cb(*args):
        print(dsr.get_q())

    dsr.add_callback(cb)


if __name__ == '__main__':
    test()
