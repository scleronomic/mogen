import numpy as np
from wzk.mpl import create_button

from wzk import trajectory

from rokin.Vis import robot_2d

from mopla.Optimizer.gradient_descent import gd_chomp
from mopla.Optimizer import feasibility_check, length

from mogen.Vis import WorldViewer, PathViewer, DraggableSphereRobot, style
from mogen.Generation import parameter, data


class InteractiveSampleViewer:
    def __init__(self, *, i_world=0, i_sample=0, file=None,
                 par, gd=None,
                 get_prediction=None,
                 show_buttons=True):

        self.file = file
        self.i_world = i_world
        self.i_sample = i_sample
        self.n_worlds, self.n_samples, self.n_samples_per_world, self.n_samples_per_world_cs = None, None, None, None
        self.init_indices()

        self.gd = gd
        self.par = par

        self.get_prediction = get_prediction

        # GUI options
        self.vis_count = 0
        self.show_buttons = show_buttons

        # Initialize plot
        self.obstacle_colors = np.array(['k'])
        self.fig, self.ax = robot_2d.new_world_fig(limits=self.par.world.limits,
                                                   title=f"world={self.i_world} | sample={self.i_sample}")

        self.world = WorldViewer(world=self.par.world, i_world=self.i_world, file=self.file, ax=self.ax)

        self.path = PathViewer(i_sample=self.i_sample, file=self.file, ax=self.ax, gd=gd, par=par, **style.style_middle)

        self.path_o = PathViewer(i_sample=self.i_sample, file=self.file, ax=self.ax, gd=gd, par=par, **style.style_optimize)
        if self.get_prediction is not None:
            self.path_p = PathViewer(i_sample=self.i_sample, file=self.file, ax=self.ax, gd=gd, par=par, **style.style_predict)

        # self.drag_config = DraggableConfigSpace(q=self.q, limits=self.par.robot.limits, color='k')
        self.drag_start = DraggableSphereRobot(q=self.path.q[0, :], ax=self.ax, robot=self.par.robot,
                                               **style.style_start, callback=self.on_drag)
        self.drag_end = DraggableSphereRobot(q=self.path.q[-1, :], ax=self.ax, robot=self.par.robot,
                                             **style.style_end,
                                             callback=self.on_drag)
        if self.par.n_wp == 1:
            self.drag_end.toggle_visibility(value=False)
            self.path.toggle_visibility(value=False)

        self.fig.canvas.draw()

        # GUI
        self.b_predict = None
        self.b_initial_guess = None
        self.b_initial_guess_random = None
        self.b_optimize = None
        # self.init_buttons()

        # Connect events
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def init_indices(self):
        if self.file is None:
            self.n_worlds, self.n_samples, self.n_samples_per_world = 100, 10, np.full(100, 10)

        else:
            _, self.n_worlds, self.n_samples, self.n_samples_per_world = data.get_info_table(file=self.file)

        self.n_samples_per_world_cs = np.cumsum(self.n_samples_per_world)
        self.n_samples_per_world_cs[1:] = self.n_samples_per_world_cs[:-1]
        self.n_samples_per_world_cs[0] = 0

        # print(self.n_worlds, self.n_samples, self.n_samples_per_world, self.n_samples_per_world_cs)

    def init_exp(self):
        pass

    def init_plot(self):
        pass

    # GUI and events
    def init_buttons(self):
        if self.show_buttons:
            self.b_predict = create_button(fig=self.fig, axes=[0.8, 0.1, 0.08, 0.06], name='Predict',
                                           listener_fun=self.plot_predict)

            self.b_initial_guess = create_button(fig=self.fig, axes=[0.8, 0.2, 0.08, 0.06], name='Initial Guess',
                                                 listener_fun=self.plot_initial_guess)

            self.b_optimize = create_button(fig=self.fig, axes=[0.8, 0.4, 0.08, 0.06], name='Optimize',
                                            listener_fun=self.plot_optimize)

    def on_key(self, event=None):
        print(event.key)
        # Update plot

        if event.key in ['I', 'i']:
            self.plot_predict()

        if event.key in ['X', 'x']:
            self.plot_optimize()

        # Turn GUI features on/off
        if event.key in ['A', 'a']:
            self.world.toggle_activity()

        if event.key in ['C', 'c']:
            self.obstacle_colors = np.hstack((self.obstacle_colors[1:], self.obstacle_colors[0]))  # Cycle through col
            print('Obstacle color: ', self.obstacle_colors[0])

        # Visualize activation of dilated convolution
        if event.key in ['V', 'v']:
            pass
            # self.visualize_layer_activation()

        # Move between worlds and samples
        if event.key == 'right':
            if self.file is not None:
                self.i_sample = (self.i_sample + 1) % self.n_samples
                self.i_world = np.nonzero(self.i_sample >= self.n_samples_per_world_cs)[0][-1]
            self.change_sample()

        if event.key == 'left':
            if self.file is not None:
                self.i_sample = (self.i_sample - 1) % self.n_samples
                self.i_world = np.nonzero(self.i_sample >= self.n_samples_per_world_cs)[0][-1]
            self.change_sample()

        if event.key == 'up':
            self.i_world = (self.i_world + 1) % self.n_worlds
            self.i_sample = self.n_samples_per_world_cs[self.i_world]
            self.change_sample()

        if event.key == 'down':
            self.i_world = (self.i_world - 1) % self.n_worlds
            self.i_sample = self.n_samples_per_world_cs[self.i_world]
            self.change_sample()

        # # Toggle visibility of different parts of the plot
        if event.key in ['f1']:
            self.drag_start.drag_circles.toggle_visibility()
            self.drag_end.drag_circles.toggle_visibility()

        # if event.key in ['f2']:
        #     plt2.toggle_visibility(h=self.x_path_plot_h)
        #
        # if event.key in ['f3']:
        #     plt2.toggle_visibility(h=self.x_pred_plot_h)
        #
        # if event.key in ['f4']:
        #     plt2.toggle_visibility(h=self.x_opt_plot_h)

    def on_drag(self, *args):
        self.path.update_path(q_start=self.drag_start.get_q(), q_end=self.drag_end.get_q(), q=None)
        self.plot_predict()

    def change_sample(self):

        i_sample_local = self.i_sample - self.n_samples_per_world_cs[self.i_world]
        print(self.i_world, self.i_sample)
        self.fig.suptitle(f"world={self.i_world} | sample={i_sample_local}")

        self.world.change_sample(i_world=self.i_world)
        self.path.change_sample(i_sample=self.i_sample)

        self.plot_predict()
        self.drag_start.set_q(self.path.q[0])
        self.drag_end.set_q(self.path.q[-1])

    def plot_initial_guess(self):
        pass

    def plot_optimize(self):
        self.plot_predict()
        q_start, q_end = trajectory.full2start_end(self.path_p.q, mode='20')

        self.par.q_start, self.par.q_end = q_start, q_end
        self.par.update_oc(img=self.world.img)

        q0 = self.path_p.q[..., 1:-1, :]
        q0 = np.reshape(q0, (1, self.par.n_wp-2, self.par.robot.n_dof))
        print(q0.shape)

        q_o, _ = gd_chomp(q0=q0, par=self.par, gd=self.gd)
        q_o = trajectory.inner2full(inner=q_o, start=q_start, end=q_end)
        q_o = trajectory.get_path_adjusted(q_o)
        self.path_o.q = np.reshape(q_o, (self.par.n_wp, self.par.robot.n_dof))
        self.path_o.plot()

        print('feasibility: path_o', feasibility_check(q=self.path_o.q[np.newaxis, :, :], par=self.par))
        len_cost = length.len_q_cost(q=self.path.q, is_periodic=None, joint_weighting=1)
        len_cost_o = length.len_q_cost(q=self.path_o.q, is_periodic=None, joint_weighting=1)
        len_cost_car = length.len_q_cost_cartesian(q=self.path.q, is_periodic=None).astype(float),
        len_cost_car_o = length.len_q_cost_cartesian(q=self.path_o.q, is_periodic=None)

        print(f"len: label {np.round(len_cost, 3)  } | {np.round(len_cost_car, 3)}")
        print(f"len: optim {np.round(len_cost_o, 3)} | {np.round(len_cost_car_o, 3)}")

    def plot_predict(self):

        if self.get_prediction is None:
            print("no 'get_prediction(img, q_start, q_end)' given")
            return

        else:
            q_start, q_end = self.path.q[[0, -1]].copy()

            # self.path_p.q = trajectory.fromto_spline(x=self.path_p.q, n_c=4, start_end0=True)

            q_pred = self.get_prediction(img=self.world.img, q_start=q_start, q_end=q_end)
            self.path_p.q = np.reshape(q_pred, (self.par.n_wp, self.par.robot.n_dof))
            self.path_p.plot()


if __name__ == '__main__':
    robot_id = 'StaticArm04'
    _par = parameter.init_par(robot_id=robot_id).par
    _par.n_wp = 1
    _file = data.get_file(robot_id=robot_id)
    isv = InteractiveSampleViewer(par=_par, file=None)
