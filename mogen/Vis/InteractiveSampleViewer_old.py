import os

import matplotlib.pyplot as plt
import matplotlib.widgets as wdgs
import numpy as np
from matplotlib.patches import Circle
from wzk.DraggableCircle import DraggableCircle, DraggableCircleList

import World.world2grid
import Kinematic.forward as forward
import Optimizer.InitialGuess.path as path_i
import Optimizer.gradient_descent as opt2
import Optimizer.objectives as cost
import Optimizer.path as path
import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import Visualization.plotting_2 as plt2
import Visualization.visualize_net as visn
import World.obstacle_distance as cost_f
import World.swept_volume
import World.swept_volume as w2i
import X_Tests.parameter_old as par
import definitions as d


class InteractiveSampleViewer(object):
    def __init__(self, i_world=0, i_sample=0, net_type='2x_inner', show_buttons=True, net=None, directory=None,
                 use_path_points=False, use_path_img_ip=False, use_x_start_end_ip=False,
                 use_z_decision=False, use_edt=False):

        self.directory = d.arg_wrapper__sample_dir(directory)
        self.g = par.Geometry(self.directory)

        # Sample
        self.i_world = i_world
        self.i_sample_local = i_sample
        self.i_sample_global = ld.get_i_samples_global(i_worlds=self.i_world, i_samples_local=self.i_sample_local)

        self.net_type = net_type
        self.use_path_img_ip = use_path_img_ip
        self.use_x_start_end_ip = use_x_start_end_ip
        self.use_z_decision = use_z_decision
        self.use_edt = use_edt

        # Optimization
        self.gamma_cost = 1
        self.gamma_grad = 0.1
        self.gamma_grad = np.linspace(start=0.01, stop=1, num=100).tolist()
        self.eps_obstacle_cost = 2
        self.gd_step_number = 100
        self.gd_step_size = 0.001
        self.adjust_gd_step_size = True

        # World obstacles
        world_df = ld.load_world_df(directory=directory)
        self.n_worlds = len(world_df)
        self.obstacle_img_list = ld.add_obstacle_img_column(world_df, values_only=True, verbose=0)
        self.obstacle_img_cur = None
        self.obstacle_cost_fun = None

        self.x_start = None
        self.x_end = None
        self.x_path = None
        self.path_img = None

        # Path of the whole arm
        self.x_path_warm = None
        self.x_pred_warm = None
        self.x_opt_warm = None
        self.x_start_warm = None
        self.x_end_warm = None

        # Net
        self.net = net

        # Path prediction, initialization, optimization,
        self.x_opt = None
        self.x_pred = None
        self.rrt = None
        self.img_pred = None

        self.obj_path = None
        self.obj_pred = None
        self.obj_opt = None

        # Plot handles
        self.x_path_plot_h = None
        self.x_pred_plot_h = None
        self.x_opt_plot_h = None
        self.img_pred_plot_h = None
        self.img_path_plot_h = None

        self.use_path_points = use_path_points

        self.update_x_world()

        if self.use_path_points:
            self.x_path = path.x_flat2x(x_flat=self.x_path, n_dof=self.g.n_dim)
            self.path_points = DraggableCircleList(x=self.x_path[:, 0], y=self.x_path[:, 1], r=self.g.r_sphere)
        else:
            if self.g.lll is None:
                self.start_point = DraggableCircle(x=self.x_start[0], y=self.x_start[1], r=self.g.r_sphere)
                self.end_point = DraggableCircle(x=self.x_end[0], y=self.x_end[1], r=self.g.r_sphere)
            else:
                self.start_point = DraggableCircle(x=self.x_start_warm[..., 0], y=self.x_start_warm[..., 1],
                                                   r=self.g.r_sphere)
                self.end_point = DraggableCircle(x=self.x_end_warm[..., 0], y=self.x_end_warm[..., 1],
                                                 r=self.g.r_sphere)
                if self.g.fixed_base:
                    self.base_point = DraggableCircle(x=self.x_start_warm[0, :, 0], y=self.x_start_warm[0, :, 1],
                                                      r=self.g.r_sphere)

        # GUI options
        self.obstacle_colors = np.array(['k', 'r'])
        self.vis_count = 0
        self.show_buttons = show_buttons

        # Initialize plot
        self.fig, self.ax = plt2.new_world_fig(limits=self.g.world_size)
        self.subtitle = plt.suptitle('world={} | sample={}'.format(self.i_world, self.i_sample_local))

        # Obstacles
        self.obstacle_pixel_grid = plt2.initialize_pixel_grid(img=self.obstacle_img_cur,
                                                              ax=self.ax, color='black', hatch='x')
        # Start-, end-points; path-points
        if use_path_points:
            self.path_points.draw(ax=self.ax, fc='g', hatch='////', alpha=0.5, zorder=100)
        else:
            self.start_point.draw(ax=self.ax, fc='g', hatch='////', alpha=0.7, zorder=100)
            self.end_point.draw(ax=self.ax, fc='r', hatch='\\\\\\\\', alpha=0.7, zorder=100)
            if self.g.fixed_base:
                self.base_point.draw(ax=self.ax, fc='xkcd:dark grey', hatch='XXXX', alpha=1, zorder=200)

        self.fig.canvas.draw()

        # Path
        if self.g.lll is None:
            self.x_path_plot_h = plt2.plot_x_path(x=self.x_path, ax=self.ax, marker='o', color='b', alpha=0.8, lw=3,
                                                  label='Optimizer, Cost: {}'.format(self.obj_path))
        else:
            print(self.x_path_warm.shape)
            self.x_path_plot_h = plt2.plot_x_spheres(x_spheres=self.x_path_warm, ax=self.ax)
            self.x_start_plot_h = plt2.plot_x_spheres(x_spheres=self.x_start_warm, ax=self.ax, c='g', lw=4)
            self.x_end_plot_h = plt2.plot_x_spheres(x_spheres=self.x_end_warm, ax=self.ax, c='r', lw=4)

        # Prediction
        if self.net is not None:
            self.predict()
            self.init_plot_prediction()

        # GUI
        self.init_buttons()

        # Connect events
        # self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click_obst)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.rectangle_selector = wdgs.RectangleSelector(self.ax, onselect=self.on_select_obst)
        self.rectangle_selector.set_active(False)

        # # TODO Debugging
        # self.update_x_path()
        # self.get_prediction_input()
        # self.optimize()

    def init_plot_prediction(self):

        if self.net_type == '2path_img' or self.net_type == '22path_img_x':
            if self.g.lll is None:
                self.img_pred_plot_h = plt2.plot_prediction_world(img=self.img_pred, world_size=self.g.world_size,
                                                                  ax=self.ax, zorder=-100, alpha=0.99)
            else:
                self.img_pred_plot_h = plt2.imshow(img=self.img_pred, limits=self.g.world_size, ax=self.ax,
                                                   zorder=0, alpha=0.99)
                # print(self.img_pred.mean())

        if self.g.lll is None:
            self.x_pred_plot_h = plt2.plot_x_path(x=self.x_pred, ax=self.ax, marker='o', color='r',
                                                  label='Prediction, Cost: {}'.format(self.obj_pred))
        else:
            self.x_pred_plot_h = plt2.plot_x_spheres(x_spheres=self.x_pred_warm, ax=self.ax)

    def init_plot(self):
        pass

    # GUI and events
    def init_buttons(self):
        # Buttons
        if self.show_buttons:
            self.bax_predict = plt.axes([0.8, 0.1, 0.08, 0.06])
            self.b_predict = wdgs.Button(self.bax_predict, 'Predict')
            self.b_predict.on_clicked(self.plot_prediction)

            self.bax_initial_guess = plt.axes([0.8, 0.2, 0.08, 0.06])
            self.b_initial_guess = wdgs.Button(self.bax_initial_guess, 'Initial Guess')
            self.b_initial_guess.on_clicked(self.initial_guess_plot)

            self.bax_initial_guess_random = plt.axes([0.8, 0.3, 0.08, 0.06])
            self.b_initial_guess_random = wdgs.Button(self.bax_initial_guess_random, 'Initial Guess Random')
            self.b_initial_guess_random.on_clicked(self.initial_guess_plot)

            self.bax_optimize = plt.axes([0.8, 0.4, 0.08, 0.06])
            self.b_optimize = wdgs.Button(self.bax_optimize, 'Optimize')
            self.b_optimize.on_clicked(self.optimize)

    def on_click_obst(self, event):
        x = event.xdata
        y = event.ydata
        i, j = World.world2grid.grid_x2i([x, y])
        self.change_obstacle_pixel(i, j)
        plt.draw()

    def on_select_obst(self, e_click, e_release):

        xy_click = np.array([e_clickb.xdata, e_clickb.ydata])
        xy_release = np.array([e_release.xdata, e_release.ydata])

        ij_click = World.world2grid.grid_x2i(xy_click)
        ij_release = World.world2grid.grid_x2i(xy_release)

        i_low = min(ij_click[0], ij_release[0])
        i_high = max(ij_click[0], ij_release[0])
        j_low = min(ij_click[1], ij_release[1])
        j_high = max(ij_click[1], ij_release[1])

        self.set_obstacle_img_cur(i_low=i_low, i_high=i_high, j_low=j_low, j_high=j_high)

    def on_key(self, event=None):
        # print(event.key)
        # TODO add text boxes to change the world / sample via text input

        # Update plot
        if event.key in ['U', 'u']:
            # ip = self.get_prediction_input()
            # start_img, end_img = np.split(ip[..., 1:], 2, axis=-1)
            #
            # gen = path_i.stepwise_obstruction(net=self.net,
            #                                   obstacle_img=self.obstacle_img_cur.copy(),
            #                                   start_img=start_img, end_img=end_img, world_size=self.g.world_size,
            #                                   tries=5, obstruction_size=7, n_dim=self.g.n_dim)

            # VAE latent decision variable
            # for o in np.linspace(-2, 2, 5):
            #     print(o)
            #     self.z = np.array([o])[:, np.newaxis]
            #     self.x_pred = gen.__next__()
            #     self.plot_prediction()
            #     # os.system("pause")
            #     time.sleep(0.01)
            #     self.fig.canvas.draw()

            # Heuristic
            # for o, x_pred in enumerate(gen):
            #     print(o)
            #     self.x_pred = x_pred
            #     self.plot_prediction()
            #
            #     time.sleep(0.01)
            #     self.fig.canvas.draw()

            self.plot_prediction()

        if event.key == ['I', 'o']:
            self.initial_guess_plot()

        if event.key == ['X', 'x']:
            self.optimize()

        if event.key in ['W', 'w']:
            self.x_path_plot_h.set_visible(not self.x_path_plot_h.get_visible())

        # Turn GUI features on/off
        if event.key in ['A', 'a']:
            self.rectangle_selector.set_active(not self.rectangle_selector.active)
            print('RectangleSelector: ', self.rectangle_selector.active)

        if event.key in ['C', 'c']:
            self.obstacle_colors = np.hstack((self.obstacle_colors[1:], self.obstacle_colors[0]))  # Cycle trough colors
            print('Obstacle color: ', self.obstacle_colors[0])

        # Visualize activation of dilated convolution
        if event.key in ['V', 'v']:
            self.visualize_layer_activation()

        # Move between worlds and samples
        if event.key == 'right':
            self.i_sample_local = (self.i_sample_local + 1) % d.n_samples_per_world
            self.update_world()

        if event.key == 'left':
            self.i_sample_local = (self.i_sample_local - 1) % d.n_samples_per_world
            self.update_world()

        if event.key == 'up':
            self.i_world = (self.i_world + 1) % self.n_worlds
            self.update_world()

        if event.key == 'down':
            self.i_world = (self.i_world - 1) % self.n_worlds
            self.update_world()

        if event.key in ['0']:
            self.start_point.toggle_visibility()
            self.end_point.toggle_visibility()

        if event.key in ['1']:
            self.toggle_plot_visibility(plot_h=self.x_path_plot_h)

        if event.key in ['2']:
            self.toggle_plot_visibility(plot_h=self.x_pred_plot_h)

        if event.key in ['3']:
            self.toggle_plot_visibility(plot_h=self.x_opt_plot_h)

    def toggle_plot_visibility(self, plot_h):
        if self.g.lll is None:
            plot_h.set_visible(not plot_h.get_visible())
        else:
            for h in plot_h:
                for hh in h:
                    hh.set_visible(not hh.get_visible())

    # Update obstacle image
    def change_obstacle_pixel(self, i, j, value=None, color=None):
        if value is None:
            value = not self.obstacle_img_cur[i, j]
        if color is None:
            color = self.obstacle_colors[0]

        self.obstacle_img_cur[i, j] = value
        self.obstacle_pixel_grid[i, j].set_visible(value)
        self.obstacle_pixel_grid[i, j].set_color(color)

    def update_obstacle_img(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.change_obstacle_pixel(i=i, j=j, value=img[i, j])

    # Update i_world, i_sample
    def update_world(self):
        self.update_x_world()
        self.update_plot_world()

    def update_x_world(self):
        self.i_sample_global = ld.get_i_samples_global(i_worlds=self.i_world, i_samples_local=self.i_sample_local)
        self.obstacle_img_cur = self.obstacle_img_list[self.i_world].copy()
        self.x_start, self.x_end, self.x_path, = \
            ld_sql.get_values_sql(columns=[START_Q, END_Q, PATH_Q], rows=self.i_sample_global,
                                  values_only=True, directory=self.directory)

        # Update Objective
        self.obj_path = self.update_objective(x=self.x_path)

        if self.g.lll is not None:
            self.x_start_warm = forward.xa2x_warm_2d(xa=self.x_start, lll=self.g.lll, n_dim=self.g.n_dim,
                                                     n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                     links_only=True, with_base=True)
            self.x_end_warm = forward.xa2x_warm_2d(xa=self.x_end, lll=self.g.lll, n_dim=self.g.n_dim,
                                                   n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                   links_only=True, with_base=True)

            self.x_path_warm = forward.xa2x_warm_2d(xa=self.x_path, lll=self.g.lll, n_dim=self.g.n_dim,
                                                    n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                    links_only=True, with_base=True)

    def update_plot_world(self):
        self.subtitle.set_text(f"world={self.i_world} | sample={self.i_sample_local}")
        self.update_obstacle_img(img=self.obstacle_img_cur)
        self.update_points_path()
        if self.g.lll is None:
            self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)
            self.x_path_plot_h.set_label(f"Label, Cost: {self.obj_path}")
        else:
            self.plot_x_path(x=self.x_path_warm, h=self.x_path_plot_h)
            self.x_path_plot_h[0][0].set_label(f"Label, Cost: {self.obj_path}")

        self.plot_prediction()

    def set_x_start_end(self, x_start=None, x_end=None):
        if x_start is not None:
            self.x_start = x_start

        if x_end is not None:
            self.x_end = x_end

        if self.g.lll is not None:
            self.x_start_warm = forward.xa2x_warm_2d(xa=self.x_start, lll=self.g.lll, n_dim=self.g.n_dim,
                                                     n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                     links_only=True, with_base=True)
            self.x_end_warm = forward.xa2x_warm_2d(xa=self.x_end, lll=self.g.lll, n_dim=self.g.n_dim,
                                                   n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                   links_only=True, with_base=True)

        self.update_plot_world()

    def set_obstacle_img_cur(self, i_low, i_high, j_low, j_high):
        mean_obstacle_occurrence = np.mean(self.obstacle_img_cur[i_low: i_high, j_low: j_high])

        new_value = not np.round(mean_obstacle_occurrence)
        for i in range(i_low, i_high):
            for j in range(j_low, j_high):
                self.change_obstacle_pixel(i, j, new_value)
        plt.draw()

    # x_path - points
    def update_x_path(self):

        if self.use_path_points:
            self.x_path = self.path_points.get_xy()
            self.x_start = self.x_path[0, :]
            self.x_end = self.x_path[-1, :]

            # self.plot_img_path()
            self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)

        else:
            if self.g.lll is None:
                self.x_start = self.start_point.get_xy()[0]
                self.x_end = self.end_point.get_xy()[0]
            else:
                x_start_warm = self.start_point.get_xy().transpose((0, 2, 1))
                x_end_warm = self.end_point.get_xy().transpose((0, 2, 1))

                xa_start = np.zeros(self.g.n_dim + self.g.n_joints)
                xa_end = np.zeros(self.g.n_dim + self.g.n_joints)
                xa_start[:self.g.n_dim] = x_start_warm[0]
                xa_end[:self.g.n_dim] = x_end_warm[0]

                def img2xa(x_pos_warm):
                    xa_pos = np.zeros(self.g.n_dim + self.g.n_joints)
                    xa_pos[:self.g.n_dim] = x_pos_warm[0]
                    for l in range(self.g.n_joints):
                        # Ensure that points ensure the constraints of the arm
                        step = x_pos_warm[l + 1, 0, :] - x_pos_warm[l, 0, :]
                        dist = np.linalg.norm(step)
                        r = np.sum(self.g.lll[l])
                        x_pos_warm[l + 1, 0, :] -= step * (dist - r) / dist
                        # Get the angles from the configuration
                        if step[0] > 0:
                            xa_pos[self.g.n_dim + l] = np.arctan(step[1] / step[0]) % (2 * np.pi)

                        else:
                            xa_pos[self.g.n_dim + l] = np.pi - np.arctan(-step[1] / step[0]) % (2 * np.pi)

                    xa_pos[self.g.n_dim:] = forward.abs2rel_angles(q=xa_pos[self.g.n_dim:], n_joints=self.g.n_joints)
                    # xa_pos[self.g.n_dim:] %= 2*np.pi  # TODO np.pi
                    return xa_pos, x_pos_warm

                self.x_start, self.x_start_warm = img2xa(x_start_warm)
                self.x_end, self.x_end_warm = img2xa(x_end_warm)

                self.update_points_path()

    def update_points_path(self):

        if self.use_path_points:
            self.x_path = path.x_flat2x(x_flat=self.x_path, n_dof=self.g.n_dim)
            self.path_points.set_xy(x=self.x_path[:, 0], y=self.x_path[:, 1], idx=-1)
            # self.plot_img_path()
            self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)

        else:
            if self.g.lll is None:
                self.start_point.set_xy_draggable(xy=self.x_start)
                self.end_point.set_xy_draggable(xy=self.x_end)
            else:
                self.start_point.set_xy_draggable(x=self.x_start_warm[..., 0].flatten(),
                                                  y=self.x_start_warm[..., 1].flatten())
                self.end_point.set_xy_draggable(x=self.x_end_warm[..., 0].flatten(),
                                                y=self.x_end_warm[..., 1].flatten())

                self.plot_x_path(x=self.x_start_warm, h=self.x_start_plot_h)
                self.plot_x_path(x=self.x_end_warm, h=self.x_end_plot_h)

    def plot_x_path(self, x, h):
        if self.g.lll is None:
            _x = path.x_flat2x(x, n_dof=self.g.n_dim)
            h.set_xdata(_x[:, 0])
            h.set_ydata(_x[:, 1])
        else:
            plt2.plot_x_spheres_update(h_paths=h[0], h_arms=h[1], x_spheres=x)

    def plot_img_path(self):
        self.path_img = w2i.sphere2grid_path(x=self.x_path, r_sphere=self.g.r_sphere)
        if self.img_path_plot_h is None:
            self.img_path_plot_h = plt2.plot_prediction_world(self.path_img, world_size=self.g.world_size, ax=self.ax,
                                                              zorder=-100, alpha=0.5)
        else:
            self.img_path_plot_h.set_array(self.path_img.T)

    # Prediction
    def get_prediction_input(self):
        self.update_x_path()
        if self.g.lll is None:
            start_img = w2i.sphere2grid_whole(x=self.x_start, r_sphere=self.g.r_sphere)
            end_img = w2i.sphere2grid_whole(x=self.x_end, r_sphere=self.g.r_sphere)
        else:
            x_start_warm = forward.xa2x_warm_2d(lll=self.g.lll, xa=self.x_start, n_dim=self.g.n_dim,
                                                n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot)
            x_end_warm = forward.xa2x_warm_2d(lll=self.g.lll, xa=self.x_end, n_dim=self.g.n_dim,
                                              n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot)

            start_img = w2i.sphere2grid_whole(x=x_start_warm, r_sphere=self.g.r_sphere,
                                              n_samples=self.g.n_spheres_tot + 1)
            end_img = w2i.sphere2grid_whole(x=x_end_warm, r_sphere=self.g.r_sphere,
                                            n_samples=self.g.n_spheres_tot + 1)

            if self.use_path_img_ip:
                path_img = ld_sql.get_values_sql(columns=PATH_IMG_CMP, rows=self.i_sample_global,
                                                 values_only=True, directory=self.directory)
                path_img = compressed2img(img_cmp=path_img, n_voxels=self.g.n_voxels, n_dim=self.g.n_dim,
                                          n_channels=self.g.n_spheres_tot)

            if self.g.fixed_base:
                start_img = start_img[..., 1:]
                end_img = end_img[..., 1:]

        if self.use_edt:
            obstacle_img = cost_f.obstacle_img2dist_img(img=self.obstacle_img_cur, world_size=self.g.world_size,
                                                    add_boundary=True)
        else:
            obstacle_img = self.obstacle_img_cur

        if self.net_type == '2path_img':  # TODO smarter
            ip = ld.imgs2x(img_list=[obstacle_img, np.logical_or(start_img, end_img)], n_dim=self.g.n_dim)
        else:
            ip = ld.imgs2x(img_list=[obstacle_img, start_img, end_img], n_dim=self.g.n_dim)

            if self.use_path_img_ip:
                ip = ld.imgs2x(img_list=[path_img, start_img, end_img], n_dim=self.g.n_dim)

        if self.use_x_start_end_ip:
            x_start = self.x_start.copy()
            x_end = self.x_end.copy()

            x_start[..., :self.g.n_dim] = path.norm_x(x_start[..., :self.g.n_dim], world_size=self.g.world_size)
            x_end[..., :self.g.n_dim] = path.norm_x(x_end[..., :self.g.n_dim], world_size=self.g.world_size)
            x_start[..., self.g.n_dim:] = forward.norm_a(x_start[..., self.g.n_dim:], zero_centered=False)
            x_end[..., self.g.n_dim:] = forward.norm_a(x_end[..., self.g.n_dim:], zero_centered=False)

            if self.g.fixed_base:
                x_start = path.q2x_q(x_start, n_dim=self.g.n_dim, n_joints=self.g.n_joints, n_samples=1)[1]
                x_end = path.q2x_q(x_end, n_dim=self.g.n_dim, n_joints=self.g.n_joints, n_samples=1)[1]
            else:
                x_start = path.x_flat2x(x_start, n_dof=self.g.n_dim, n_samples=1)
                x_end = path.x_flat2x(x_end, n_dof=self.g.n_dim, n_samples=1)
            ip = [ip, x_start, x_end]  # TODO check if ip is already a list
        return ip

    def predict(self, z=None):
        ip = self.get_prediction_input()

        if self.net is None:
            return

        # Get prediction
        # self.use_z_decision = True

        if self.use_z_decision:
            # Test whole net work, including the backward decision loop -> results in nan
            # x_path = path.x2x_inner(x=self.x_path, n_dof=self.g.n_dim+self.g.n_joints, return_flat=False)
            # x_path = path.norm_x(x=x_path, world_size=self.g.world_size)
            # x_path = x_path[np.newaxis, :]
            # x_pred = self.net.predict([ip, x_path])
            # print(x_pred)

            # plt2.world_imshow(img=ip[0], world_size=self.g.world_size, ax=self.ax)
            # print(ip[..., 0].min(), ip[..., 0].max(), ip[..., 0].mean())

            if z is None:
                z = np.random.standard_normal((1, 1)) * 1  # TODO CHEAT
                # z = -1 + 2 * np.random.randint(0, 2, shape=(1, 1))
            else:
                z = np.reshape(z, (1, 1))
            if self.use_x_start_end_ip:
                a_pred = self.net[0].predict(ip[0], batch_size=1)
                x_pred = self.net[1].predict([a_pred, ip[-2], ip[-1]], z)

            else:
                a_pred = self.net[0].predict(ip, batch_size=1)
                try:
                    x_pred = self.net[1].predict([a_pred, z])
                except ValueError:

                    x_pred = self.net[1].predict(np.concatenate([a_pred, z], axis=-1))

            # a_pred /= a_pred.mean()
            # TODO a lot of zeros in bottleneck is not a good sign

            # a_pred = np.concatenate([a_pred, z], axis=1)
            # print(x_pred)
        else:
            x_pred = self.net.predict(ip)

        if isinstance(x_pred, list):
            x_pred_img = x_pred[0]
            x_pred = x_pred[-1]

        # Update prediction
        if self.net_type == '22path_img_x':
            self.img_pred = ld.reshape_img(img=x_pred_img, n_dim=self.g.n_dim, n_samples=1, sample_dim=False,
                                           channel_dim=False)

        if self.net_type == '2path_img':
            self.img_pred = ld.reshape_img(img=x_pred, n_dim=self.g.n_dim, n_samples=1, sample_dim=False,
                                           channel_dim=False)
            # self.x_pred = path_i.initialize_x0_rrt(x_start=self.x_start, x_end=self.x_end,
            #                                        n_waypoints=self.g.n_waypoints, verbose=1,
            #                                        p=lambda:
            #                                        path_i.sample_from_distribution(prob_img=self.img_pred,
            #                                                                        world_size=self.g.world_size,
            #                                                                        shape=1)[0])

            self.x_pred = self.x_path.copy()

        if self.net_type == '2x':
            self.x_pred = x_pred

        if self.net_type == '2x_delta':
            x_pred = path.delta_x2x(x_delta=x_pred, x_start=self.x_start, x_end=self.x_end, denormalize=True,
                                    n_dim=self.g.n_dim)
            self.x_pred = x_pred

        if self.net_type == '2x_inner' or self.net_type == '22path_img_x':
            if self.g.lll is None:
                x_pred = path.denorm_x(x=x_pred, world_size=self.g.world_size)
            else:
                if not self.g.fixed_base:
                    x_pred[..., :self.g.n_dim] = path.denorm_x(x=x_pred[..., :self.g.n_dim], world_size=self.g.world_size)
                else:
                    x_base = path.q2x_q(xq=self.x_path, n_dim=self.g.n_dim, n_joints=self.g.n_joints)[0]
                    x_pred = np.concatenate((x_base[..., 1:-1, :self.g.n_dim], x_pred[0]), axis=-1)
                x_pred[..., self.g.n_dim:] = forward.denorm_a(a=x_pred[..., self.g.n_dim:], zero_centered=False)

            x_pred = path.x_inner2x(inner=x_pred, start=self.x_start, end=self.x_end,
                                    n_dof=self.g.n_dim + self.g.n_joints)
            self.x_pred = x_pred

        # else:
        #     raise ValueError('Incompatible net_type')

        if self.g.lll is not None:
            self.x_pred_warm = forward.xa2x_warm_2d(xa=self.x_pred, lll=self.g.lll, n_dim=self.g.n_dim,
                                                    n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                    links_only=True, with_base=True)
        # Update Objective
        self.obj_pred = self.update_objective(x=self.x_pred)

        return self.x_pred

    def plot_prediction(self, event=None):
        self.predict()
        if self.net is not None:
            if self.net_type == '2path_img' or self.net_type == '22path_img_x':
                if self.g.lll is None:
                    self.img_pred_plot_h.set_array(self.img_pred.T)
                else:
                    for i, h in enumerate(self.img_pred_plot_h):
                        h.set_array(self.img_pred[..., i].T)

            if self.g.lll is None:
                self.plot_x_path(x=self.x_pred, h=self.x_pred_plot_h)
                self.x_pred_plot_h.set_label('Prediction, Cost: {}'.format(self.obj_pred))
            else:
                self.plot_x_path(x=self.x_pred_warm, h=self.x_pred_plot_h)
                self.x_pred_plot_h[0][0].set_label('Prediction, Cost: {}'.format(self.obj_pred))

        self.ax.legend(loc='upper right')
        plt.draw()

    def update_objective(self, x):

        obstacle_cost_fun = \
            cost_f.obstacle_img2cost_fun(obstacle_img=self.obstacle_img_cur, r=self.g.r_sphere,
                                         interp_order=0, eps=self.eps_obstacle_cost,
                                         world_size=self.g.world_size)

        if self.g.lll is None:
            length_norm = path.get_start_end_normalization(start=self.x_start, end=self.x_end,
                                                           n=self.g.n_waypoints)
        else:
            length_norm = forward.get_beeline_normalization(q_start=self.x_start, q_end=self.x_end,
                                                            n_wp=self.g.n_waypoints, n_joints=self.g.n_joints,
                                                            n_spheres_tot=self.g.n_spheres_tot,
                                                            lll=self.g.lll, fixed_base=self.g.fixed_base,
                                                            infinity_joints=self.g.infinity_joints)

        obj = cost.chomp_cost(x_inner=path.x2x_inner(x=x, n_dof=self.g.n_dim + self.g.n_joints), n_dim=self.g.n_dim,
                              robot_id=self.g.lll, x_start=self.x_start, x_end=self.x_end, gamma_len=self.gamma_cost,
                              n_substeps=5, obst_cost_fun=obstacle_cost_fun, length_norm=length_norm,
                              fixed_base=self.g.fixed_base, return_separate=False,
                              infinity_joints=self.g.infinity_joints,
                              n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot, n_wp=self.g.n_waypoints)

        return obj

    # RRT, Optimizer
    def initial_guess_plot(self, event=None, random=False):

        if random:
            prediction_img = np.ones_like(self.img_pred)
            print('random')
        else:
            prediction_img = self.img_pred

        # _x_path = opt2.prediction_2_initial_path(xy_start=self.xy_start, xy_end=self.xy_end,
        #                                          prediction=self.prediction_img, world_size=world_size, n_waypoints=9)
        # TODO other methods for coming up with an initial guess
        tree, _x_path = path_i.initialize_path_rrt(x_start=self.x_start, x_end=self.x_end,
                                                   n_waypoints=self.g.n_waypoints,
                                                   verbose=2, return_tree=True,
                                                   p=lambda: path_i.sample_from_image(prob_img=prediction_img,
                                                                                      world_size=self.g.world_size,
                                                                                      size=1)[0])

        if self.x_pred_plot_h is None:
            self.x_pred_plot_h = self.ax.plot(_x_path[:, 0], _x_path[:, 1])[0]
        else:
            self.x_pred_plot_h.set_xdata(_x_path[:, 0])
            self.x_pred_plot_h.set_ydata(_x_path[:, 1])

    def optimize(self, event=None):
        self.update_x_path()

        # Obstacles in cost function
        obstacle_cost_fun = \
            cost_f.obstacle_img2cost_fun(obstacle_img=self.obstacle_img_cur, r=self.g.r_sphere,
                                         interp_order=0, eps=self.eps_obstacle_cost,
                                         world_size=self.g.world_size)

        obstacle_cost_fun_grad = \
            cost_f.obstacle_img2cost_fun_grad(obstacle_img=self.obstacle_img_cur, r=self.g.r_sphere,
                                              eps=self.eps_obstacle_cost, world_size=self.g.world_size, interp_order=1)

        x0 = path.x2x_inner(x=self.x_pred, n_dof=self.g.n_dim + self.g.n_joints).copy()
        x_opt, obj_opt = \
            opt2.gradient_descent_path_cost(x_inner=x0, n_dim=self.g.n_dim,
                                            x_start=self.x_start, x_end=self.x_end,
                                            gamma_grad=self.gamma_grad.copy(), gamma_cost=self.gamma_cost,
                                            n_ss_obst_cost=5, obst_cost_fun=obstacle_cost_fun,
                                            obst_cost_fun_grad=obstacle_cost_fun_grad, fixed_base=self.g.fixed_base,
                                            gd_step_number=self.gd_step_number, gd_step_size=self.gd_step_size,
                                            adjust_gd_step_size=self.adjust_gd_step_size, length_norm=True,
                                            lll=self.g.lll,
                                            return_separate=False,
                                            infinity_joints=self.g.infinity_joints,
                                            n_wp=self.g.n_waypoints,
                                            constraints_x=self.g.constraints_x, constraints_q=self.g.constraints_q)

        self.obj_opt = obj_opt

        self.x_opt = path.x_inner2x(inner=x_opt, start=self.x_start, end=self.x_end,
                                    n_dof=self.g.n_dim + self.g.n_joints)
        if self.g.lll is not None:
            self.x_opt_warm = forward.xa2x_warm_2d(xa=self.x_opt, lll=self.g.lll, n_dim=self.g.n_dim,
                                                   n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
                                                   links_only=True, with_base=True)

        # Plot
        if self.x_opt_plot_h is None:
            if self.g.lll is None:
                self.x_opt_plot_h = plt2.plot_x_path(x=self.x_opt, marker='o', color='c', ax=self.ax,
                                                     label='Optimizer, Cost: {}'.format(self.obj_opt))
            else:
                self.x_opt_plot_h = plt2.plot_x_spheres(x_spheres=self.x_opt_warm, ax=self.ax)

        else:
            if self.g.lll is None:
                self.plot_x_path(x=self.x_opt, h=self.x_opt_plot_h)
                self.x_opt_plot_h.set_label('Optimizer, Cost: {}'.format(self.obj_opt))
            else:
                self.plot_x_path(x=self.x_opt_warm, h=self.x_opt_plot_h)
                self.x_opt_plot_h[0][0].set_label('Optimizer, Cost: {}'.format(self.obj_opt))
        self.ax.legend()

    # Visualize layers of the net
    def visualize_layer_activation(self):
        self.update_x_path()

        x = self.get_prediction_input()

        visn.plot_conv_activation(model=self.net, x=x, layers_idx='conv', plot_prediction=False,
                                  figure_name=str(self.vis_count) + '_')
        self.vis_count += 1

    # Helper
    def set_obstacle_img_ref(self):
        self.obstacle_img_cur = self.obstacle_img_cur[self.i_world].copy()

    def plot_cost_img(self):
        cost_img = cost_f.obstacle_img2cost_img(self.obstacle_img_cur, world_size=self.g.world_size,
                                                r_spheres=self.g.r_sphere,
                                                eps=self.eps_obstacle_cost)
        plt2.imshow(img=cost_img, limits=self.g.world_size, ax=self.ax, alpha=0.6)

    def save_fig(self, filename='test',
                 plot_path_label=True, plot_x_pred=True, plot_pred_img=True,
                 cmap='Blues', plot_new_obstacles=False,
                 save=True):

        if '/' in filename:
            filename_split = filename.split('/')
            filename = filename_split[-1]

            if len(filename_split) > 2:
                img_dir = '/'.join(filename_split[:-1])
            else:
                img_dir = filename_split[0]
            img_dir += '/'
        else:
            sample_dir = d.arg_wrapper__sample_dir(self.directory, full=False)
            img_dir = d.PROJECT_DATA_IMAGES + 'IA/' + sample_dir
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
        filename = img_dir + 'w{}p{}_{}'.format(self.i_world, self.i_sample_local, filename)

        fig, ax = plt2.new_world_fig(limits=self.g.world_size, scale=2)

        # Obstacles
        plt2.plot_img_outlines(img=self.obstacle_img_cur, ax=ax, world_size=self.g.world_size,
                               color='k', ls='-', lw=2)
        plt2.plot_img_patch(img=self.obstacle_img_cur, ax=ax, hatch='xxx', color='k',
                            lw=0, alpha=0.9)

        if plot_new_obstacles:
            obstacle_img_new = self.obstacle_img_cur.astype(int) - self.obstacle_img_list[self.i_world].astype(int)
            obstacle_img_new[obstacle_img_new < 0] = 0
            obstacle_img_new = obstacle_img_new.astype(bool)
            if obstacle_img_new.sum() > 0:
                plt2.plot_img_patch(img=obstacle_img_new, ax=ax, hatch='xxx',
                                    color=c_tum['orange'], lw=0, alpha=0.9)
                plt2.plot_img_outlines(img=obstacle_img_new, ax=ax, world_size=self.g.world_size,
                                       color=c_tum['orange'], ls='-', lw=2)

        # Start, End
        if self.g.lll is None:

            # ax.set_ylim([20, 80])  # TODO  for FINAL/SingleSphereRobots/global_pos_switch_2D_SR_2dof.py
            # y = np.arange(40, 61, 2)
            # for yy in y:
            #     circle = Circle(xy=np.array([90.0, yy]), radius=self.g.r_sphere, fc='r', hatch='\\\\\\\\', alpha=0.2)
            #     ax.add_patch(circle)

            circle_a = Circle(xy=self.x_start, radius=self.g.r_sphere, fc='g', hatch='////', alpha=1, zorder=100)
            circle_e = Circle(xy=self.x_end, radius=self.g.r_sphere, fc='r', hatch='\\\\\\\\', alpha=1, zorder=100)
            ax.add_patch(circle_a)
            ax.add_patch(circle_e)

        else:
            for xy in self.x_start_warm:
                circle_a = Circle(xy=xy.flatten(), radius=self.g.r_sphere, fc='g', hatch='////', alpha=1, zorder=100)
                ax.add_patch(circle_a)

            for xy in self.x_end_warm:
                circle_a = Circle(xy=xy.flatten(), radius=self.g.r_sphere, fc='r', hatch='\\\\\\\\', alpha=1, zorder=100)
                ax.add_patch(circle_a)

        if self.g.fixed_base:
            circle_b = Circle(xy=self.x_start_warm[0, :].flatten(), radius=self.g.r_sphere,
                              fc='xkcd:dark gray', hatch='XXXX', alpha=1, zorder=200)
            ax.add_patch(circle_b)

        if self.g.lll is not None:
            plt2.plot_x_spheres(x_spheres=self.x_start_warm, ax=ax, c='g', lw=5)
            plt2.plot_x_spheres(x_spheres=self.x_end_warm, ax=ax, c='r', lw=5)
            if plot_path_label:
                plt2.plot_x_spheres(x_spheres=self.x_path_warm, ax=ax)

        # Prediction Image
        if plot_pred_img:
            if self.g.lll is None:
                plt2.plot_prediction_world(self.img_pred, world_size=self.g.world_size, ax=ax, zorder=-1,
                                           cmap=cmap)
            else:
                # plt2.world_imshow(img=self.img_pred, world_size=self.g.world_size, ax=ax, zorder=50, alpha=1)
                plt2.plot_prediction_world(self.img_pred[..., 0], world_size=self.g.world_size, ax=ax, zorder=3,
                                           cmap='Blues', alpha=0.7, cmap_clip=[-2, -3])
                plt2.plot_prediction_world(self.img_pred[..., 1], world_size=self.g.world_size, ax=ax, zorder=2,
                                           cmap='Greens', alpha=0.7, vmax=1.4, cmap_clip=[-2, -3])
                plt2.plot_prediction_world(self.img_pred[..., 2], world_size=self.g.world_size, ax=ax, zorder=1,
                                           cmap='Blues', alpha=0.7, cmap_clip=[-2, -3])
        # Prediction Path
        if plot_x_pred:
            if self.g.lll is None:
                plt2.plot_x_path(x=self.x_pred, n_dim=2, ax=ax, marker='o', c=c_tum['blue_2'], lw=3, alpha=1,
                                 markersize=10, markeredgecolor='k')
            else:
                plt2.plot_x_spheres(x_spheres=self.x_pred_warm, n_dim=self.g.n_dim, ax=ax)

        # Plot different decisions
        # if self.use_z_decision:
        #     for o in np.linspace(-3, 3, 10):
        #             print(o)
        #             z = np.array([o])[:, np.newaxis]
        #             x_pred = self.predict(z=z)
        #             plt2.plot_path(x=x_pred, n_dim=2, ax=ax, marker='o', c=str(o/10), lw=3)

        # z = np.linspace(-3, 0, 10)
        # for o in range(10):
        #     x_pred = self.predict(z=z[o])
        #     plt2.plot_path(x=x_pred, n_dim=2, ax=ax, marker='o', c='k', lw=3, alpha=0.5)
        #     plt2.plot_path(x=x_pred, n_dim=2, ax=ax, marker='o', c=str((10-o)/20), lw=3, alpha=0.9, markersize=10,
        #                    markeredgecolor='k')

        save_fig(filename, fig=fig, save=save)
        return fig, ax


class ISCPathImg2x(InteractiveSampleViewer):
    def __init__(self, i_world=0, i_sample=0, net_type='2x_inner', show_buttons=True, directory=None, net=None):
        super(ISCPathImg2x, self).__init__(i_world=i_world, i_sample=i_sample,
                                           net_type=net_type, show_buttons=show_buttons,
                                           directory=directory, net=net, use_path_points=True)

        self.update_obstacle_img(img=np.zeros_like(self.obstacle_img_cur, dtype=bool))

    def get_prediction_input(self):
        self.update_x_path()
        self.plot_img_path()
        start_img = w2i.sphere2grid_whole(x=self.x_start, r_sphere=self.g.r_sphere)
        end_img = w2i.sphere2grid_whole(x=self.x_end, r_sphere=self.g.r_sphere)
        x = ld.imgs2x(img_list=[self.path_img, start_img, end_img], n_dim=self.g.n_dim)
        return x

    def update_plot_world(self):
        self.subtitle.set_text('world={} | sample={}'.format(self.i_world, self.i_sample_local))
        self.update_points_path()
        self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)
        self.plot_img_path()
        self.plot_prediction()


class ISCOptimizer(InteractiveSampleViewer):
    def __init__(self, i_world=0, i_sample=0, net_type='2x_inner', show_buttons=True, directory=None, net=None,
                 gamma=0.1, eps_obstacle_cost=0.01, opt_maxiter=100, opt_method='constraints'):
        super(ISCOptimizer, self).__init__(i_world=i_world, i_sample=i_sample, net_type=net_type,
                                           show_buttons=show_buttons, directory=directory, net=net,
                                           use_path_points=True)
        self.gamma = gamma
        self.eps_obstacle_cost = eps_obstacle_cost
        self.opt_maxiter = opt_maxiter
        self.opt_method = opt_method

    def update_plot_world(self):
        self.subtitle.set_text('world={} | sample={}'.format(self.i_world, self.i_sample_local))
        self.update_points_path()
        self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)
        self.update_obstacle_img(img=self.obstacle_img_cur)
        self.update_x_path()
        self.plot_x_path(x=self.x_path, h=self.x_path_plot_h)
