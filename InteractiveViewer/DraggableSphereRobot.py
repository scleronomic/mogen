import numpy as np
from wzk.mpl import DraggableCircleList

import Kinematic.forward as forward
import Optimizer.path as path
import Visualization.plotting_2 as plt2


def abs2rel_angles(q):
    q %= 2 * np.pi
    q[..., 1:] = q[..., 1:] - q[..., :-1]
    path.inf_joint_wrapper(x=q, inf_bool=np.ones(q.shape[-1], dtype=bool))
    return q


def get_x_frames_distances(robot):
    x = forward.get_frames_x(q=np.zeros((1, 1, robot.n_dof)), robot=robot)
    x = np.squeeze(x)
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
        # Ensure that points ensure the constraints of the arm
        step = x_spheres[i + 1, :] - x_spheres[robot.prev_frame_idx[i + 1], :]
        dist = np.linalg.norm(step)
        r = limb_lengths[i]
        x_spheres[i + 1, :] -= step * (dist - r) / dist

        # Get the angles from the configuration
        q[n_dim + i] = np.arctan2(step[1], step[0])

    q[n_dim:] = abs2rel_angles(q=q[n_dim:])

    if robot.id == 'Single_Sphere_02':
        q = q[:n_dim]
    elif 'Stat_Arm_' in robot.id:
        q = q[n_dim:]

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
                                                radius=self.robot.SPHERES_RAD.mean(),
                                                callback=self.callback, **style_path)
        self.drag_circles.set_callback(callback=self.update_spheres2q_plot)
        self.x_spheres_plot_h = plt2.plot_x_spheres(x_spheres=self.x_spheres, ax=self.ax, style_path=style_path,
                                                    style_arm=style_arm)

        # if self.g.fixed_base:
        #     self.base_circle = Circle(ax=self.ax, xy=self.xs_start[0, :, 0],
        #                               radius=radius=self.par.robot.spheres_rad[0],
        #                               fc='xkcd:dark grey', hatch='XXXX', alpha=1, zorder=200)

    def update_val2plot(self):
        plt2.plot_x_spheres_update(h=self.x_spheres_plot_h, x_spheres=self.x_spheres)
        self.drag_circles.set_xy(x=self.x_spheres[..., 0].flatten(), y=self.x_spheres[..., 1].flatten())

    def update_q2spheres(self):
        self.x_spheres = forward.get_frames_x(q=self.q.reshape((1, 1, self.robot.n_dof)), robot=self.robot)[0]
        self.x_spheres = self.x_spheres.reshape(-1, 2)

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
