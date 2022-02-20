from rokin.Vis.robot_2d import *
from mopla.Parameter import parameter

from wzk import sql2
from mogen.Vis import WorldViewer, DraggableSphereRobot, SpherePath


n_samples_per_world = 100

style_dict_start = {'style_path': {'facecolor': 'g',
                                   'hatch': '////',
                                   'alpha': 0.7,
                                   'zorder': 100},
                    'style_arm': {'c': 'g',
                                  'lw': 4}}
style_dict_end = {'style_path': {'facecolor': 'r',
                                 'hatch': '\\\\\\\\',
                                 'alpha': 0.7,
                                 'zorder': 100},
                  'style_arm': {'c': 'r',
                                'lw': 4}}


class InteractiveSampleViewer:
    def __init__(self, *, i_world=0, i_sample=0, file=None,
                 exp_par, gd_par, par,
                 show_buttons=True):

        # Sample
        self.i_world = i_world
        self.i_sample_local = i_sample
        self.i_sample_global = i_sample
        # self.i_sample_global = get_i_samples_global(i_worlds=self.i_world, i_samples_local=self.i_sample_local)

        # Optimization
        self.gd = gd_par
        self.par = par
        self.exp = exp_par

        # World obstacles

        self.file = file

        if self.file is None:
            self.n_worlds = 0
        else:
            self.n_worlds = sql2.get_n_rows(file=self.file, table='worlds')

        self.q_start = None
        self.q_end = None
        self.q = None

        # Net
        # self.net = net

        # GUI options
        self.vis_count = 0
        self.show_buttons = show_buttons

        # Plot handles
        self.x_pred_plot_h = None
        self.x_opt_plot_h = None

        # Initialize plot
        self.obstacle_colors = np.array(['k'])
        self.fig, self.ax = new_world_fig(limits=self.par.world.limits,
                                          title=f"world={self.i_world} | sample={self.i_sample_local}")

        self.world_viewer = WorldViewer(world=self.par.world, ax=self.ax, i_world=self.i_world,
                                        file=self.file)
        self.sphere_path = SpherePath(i_sample=self.i_sample_global, ax=self.ax, file=self.file,
                                      exp=exp_par, gd=gd_par, par=par)

        self.q = np.squeeze(self.sphere_path.q)
        # self.drag_config = DraggableConfigSpace(q=self.q, limits=self.par.robot.limits, color='k')
        self.drag_pos_start = DraggableSphereRobot(q=self.q[0, :], ax=self.ax, robot=self.par.robot, **style_dict_start)
        self.drag_pos_end = DraggableSphereRobot(q=self.q[-1, :], ax=self.ax, robot=self.par.robot, **style_dict_end)

        def cb_start_end2path(*args):
            self.sphere_path.update_path(q_start=self.drag_pos_start.get_q(), q_end=self.drag_pos_end.get_q())

        self.drag_pos_start.add_callback(cb_start_end2path)
        self.drag_pos_end.add_callback(cb_start_end2path)

        self.fig.canvas.draw()

        # GUI
        self.b_predict = None
        self.b_initial_guess = None
        self.b_initial_guess_random = None
        self.b_optimize = None
        # self.init_buttons()

        # Connect events
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def init_plot(self):
        pass

    # GUI and events
    def init_buttons(self):
        if self.show_buttons:
            self.b_predict = create_button(axes=[0.8, 0.1, 0.08, 0.06], name='Predict',
                                           listener_fun=self.plot_prediction)

            self.b_initial_guess = create_button(axes=[0.8, 0.2, 0.08, 0.06], name='Initial Guess',
                                                 listener_fun=self.initial_guess_plot)

            self.b_initial_guess_random = create_button(axes=[0.8, 0.3, 0.08, 0.06], name='Initial Guess Random',
                                                        listener_fun=self.initial_guess_plot)

            self.b_optimize = create_button(axes=[0.8, 0.4, 0.08, 0.06], name='Optimize',
                                            listener_fun=self.optimize)

    def on_key(self, event=None):
        print(event.key)
        # Update plot

        if event.key == ['I', 'o']:
            pass
            # self.initial_guess_plot()

        if event.key == ['X', 'x']:
            pass
            # self.optimize()

        # Turn GUI features on/off
        if event.key in ['A', 'a']:
            self.world_viewer.toggle_activity()

        if event.key in ['C', 'c']:
            self.obstacle_colors = np.hstack((self.obstacle_colors[1:], self.obstacle_colors[0]))  # Cycle trough colors
            print('Obstacle color: ', self.obstacle_colors[0])

        # Visualize activation of dilated convolution
        if event.key in ['V', 'v']:
            pass
            # self.visualize_layer_activation()

        # Move between worlds and samples
        if event.key == 'right':
            if self.file is not None:
                self.i_sample_local = (self.i_sample_local + 1) % n_samples_per_world
            self.change_sample()

        if event.key == 'left':
            if self.file is not None:
                self.i_sample_local = (self.i_sample_local - 1) % n_samples_per_world
            self.change_sample()

        if event.key == 'up':
            if self.file is not None:
                self.i_world = (self.i_world + 1) % self.n_worlds
            self.change_sample()

        if event.key == 'down':
            if self.file is not None:
                self.i_world = (self.i_world - 1) % self.n_worlds
            self.change_sample()

        # # Toggle visibility of different parts of the plot
        if event.key in ['f1']:
            self.drag_pos_start.drag_circles.toggle_visibility()
            self.drag_pos_end.drag_circles.toggle_visibility()

        # if event.key in ['f2']:
        #     plt2.toggle_visibility(h=self.x_path_plot_h)
        #
        # if event.key in ['f3']:
        #     plt2.toggle_visibility(h=self.x_pred_plot_h)
        #
        # if event.key in ['f4']:
        #     plt2.toggle_visibility(h=self.x_opt_plot_h)

    def change_sample(self):
        self.fig.suptitle(f"world={self.i_world} | sample={self.i_sample_local}")

        self.i_sample_global = self.i_sample_local
        #get_i_samples_global(i_worlds=self.i_world, i_samples_local=self.i_sample_local)

        self.world_viewer.change_sample(i_world=self.i_world)

        q_start, q_end = self.sphere_path.change_sample(i_sample=self.i_sample_global)

        self.drag_pos_start.set_q(q_start)
        self.drag_pos_end.set_q(q_end)


if __name__ == '__main__':
    par = parameter.Parameter(robot='StaticArm03')

    par.n_wp = 10
    print('World Limits:')
    print(par.world.limits)
    print('Robot Limits:')
    print(par.robot.limits)
    a = InteractiveSampleViewer(par=par, exp_par=None, gd_par=None)


# TODO write before optimizer call the right location for this function
# parameter.initialize_oc_par(oc=self.par.oc, world=self.par.world, obstacle_img=self.obstacle_img)

    #         if event.key in ['U', 'u']:
    #             # ip = self.get_prediction_input()
    #             # start_img, end_img = np.split(ip[..., 1:], 2, axis=-1)
    #             #
    #             # gen = path_i.stepwise_obstruction(net=self.net,
    #             #                                   obstacle_img=self.obstacle_img_cur.copy(),
    #             #                                   start_img=start_img, end_img=end_img, world_size=self.g.world_size,
    #             #                                   tries=5, obstruction_size=7, n_dim=self.g.n_dim)
    #
    #             # VAE latent decision variable
    #             # for o in np.linspace(-2, 2, 5):
    #             #     print(o)
    #             #     self.z = np.array([o])[:, np.newaxis]
    #             #     self.x_pred = gen.__next__()
    #             #     self.plot_prediction()
    #             #     # os.system("pause")
    #             #     time.sleep(0.01)
    #             #     self.fig.canvas.draw()
    #
    #             # Heuristic
    #             # for o, x_pred in enumerate(gen):
    #             #     print(o)
    #             #     self.x_pred = x_pred
    #             #     self.plot_prediction()
    #             #
    #             #     time.sleep(0.01)
    #             #     self.fig.canvas.draw()
    #
    #             self.plot_prediction()

    # def plot_img_path(self):
    #     self.path_img = w2i.path2image(x=self.q, r=self.g.r)
    #     if self.img_path_plot_h is None:
    #         self.img_path_plot_h = plt2.plot_prediction_world(self.path_img, world_size=self.g.world_size, ax=self.ax,
    #                                                           zorder=-100, alpha=0.5)
    #     else:
    #         self.img_path_plot_h.set_array(self.path_img.T)
    #
    # # Prediction
    # def init_plot_prediction(self):
    #
    #     if self.net_type == '2path_img' or self.net_type == '22path_img_x':
    #         if self.g.lll is None:
    #             self.img_pred_plot_h = plt2.plot_prediction_world(img=self.img_pred, world_size=self.g.world_size,
    #                                                               ax=self.ax, zorder=-100, alpha=0.99)
    #         else:
    #             self.img_pred_plot_h = plt2.world_imshow(img=self.img_pred, limits=self.g.world_size, ax=self.ax,
    #                                                      zorder=0, alpha=0.99)
    #             # print(self.img_pred.mean())
    #
    #     if self.g.lll is None:
    #         self.x_pred_plot_h = plt2.plot_x_path(x=self.q_pred, ax=self.ax, marker='o', color='r',
    #                                               label='Prediction, Cost: {}'.format(self.obj_pred))
    #     else:
    #         self.x_pred_plot_h = plt2.plot_x_spheres(x_spheres=self.xs_pred, ax=self.ax)
    #
    # def get_prediction_input(self):
    #     self.update_x_start_end_path()
    #     if self.g.lll is None:
    #         start_img = w2i.position2image(x_pos=self.q_start, r=self.g.r)
    #         end_img = w2i.position2image(x_pos=self.q_end, r=self.g.r)
    #     else:
    #         x_start_warm = forward.xa2x_warm_2d(lll=self.g.lll, xa=self.q_start, n_dim=self.g.n_dim,
    #                                             n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot)
    #         x_end_warm = forward.xa2x_warm_2d(lll=self.g.lll, xa=self.q_end, n_dim=self.g.n_dim,
    #                                           n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot)
    #
    #         start_img = w2i.position2image(x_pos=x_start_warm, r=self.g.r,
    #                                        n_samples=self.g.n_spheres_tot + 1)
    #         end_img = w2i.position2image(x_pos=x_end_warm, r=self.g.r,
    #                                      n_samples=self.g.n_spheres_tot + 1)
    #
    #         if self.use_path_img_ip:
    #             path_img = ld_sql.get_values_sql(columns=PATH_IMG_CMP, rows=self.i_sample_global,
    #                                              values_only=True, directory=self.directory)
    #             path_img = compressed2img(img_cmp=path_img, shape=self.g.shape, n_dim=self.g.n_dim,
    #                                       n_channels=self.g.n_spheres_tot)
    #
    #         if self.g.fixed_base:
    #             start_img = start_img[..., 1:]
    #             end_img = end_img[..., 1:]
    #
    #     if self.use_edt:
    #         obstacle_img = cost_f.obstacle_img2dist_img(img=self.obstacle_img_cur, world_size=self.g.world_size,
    #                                                 add_boundary=True)
    #     else:
    #         obstacle_img = self.obstacle_img_cur
    #
    #     if self.net_type == '2path_img':  # TODO smarter
    #         ip = ld.imgs2x(img_list=[obstacle_img, np.logical_or(start_img, end_img)], n_dim=self.g.n_dim)
    #     else:
    #         ip = ld.imgs2x(img_list=[obstacle_img, start_img, end_img], n_dim=self.g.n_dim)
    #
    #         if self.use_path_img_ip:
    #             ip = ld.imgs2x(img_list=[path_img, start_img, end_img], n_dim=self.g.n_dim)
    #
    #     if self.use_x_start_end_ip:
    #         x_start = self.q_start.copy()
    #         x_end = self.q_end.copy()
    #
    #         x_start[..., :self.g.n_dim] = path.norm_x(x_start[..., :self.g.n_dim], world_size=self.g.world_size)
    #         x_end[..., :self.g.n_dim] = path.norm_x(x_end[..., :self.g.n_dim], world_size=self.g.world_size)
    #         x_start[..., self.g.n_dim:] = forward.norm_a(x_start[..., self.g.n_dim:], zero_centered=False)
    #         x_end[..., self.g.n_dim:] = forward.norm_a(x_end[..., self.g.n_dim:], zero_centered=False)
    #
    #         if self.g.fixed_base:
    #             x_start = path.q2x_q(x_start, n_dim=self.g.n_dim, n_joints=self.g.n_joints, n_samples=1)[1]
    #             x_end = path.q2x_q(x_end, n_dim=self.g.n_dim, n_joints=self.g.n_joints, n_samples=1)[1]
    #         else:
    #             x_start = path.x_flat2x(x_start, n_dof=self.g.n_dim, n_samples=1)
    #             x_end = path.x_flat2x(x_end, n_dof=self.g.n_dim, n_samples=1)
    #         ip = [ip, x_start, x_end]  # TODO check if ip is already a list
    #     return ip
    #
    # def predict(self, z=None):
    #     ip = self.get_prediction_input()
    #
    #     if self.net is None:
    #         return
    #
    #     # Get prediction
    #     # self.use_z_decision = True
    #
    #     if self.use_z_decision:
    #         # Test whole net work, including the backward decision loop -> results in nan
    #         # x_path = path.x2x_inner(x=self.x_path, n_dof=self.g.n_dim+self.g.n_joints, return_flat=False)
    #         # x_path = path.norm_x(x=x_path, world_size=self.g.world_size)
    #         # x_path = x_path[np.newaxis, :]
    #         # x_pred = self.net.predict([ip, x_path])
    #         # print(x_pred)
    #
    #         # plt2.world_imshow(img=ip[0], world_size=self.g.world_size, ax=self.ax)
    #         # print(ip[..., 0].min(), ip[..., 0].max(), ip[..., 0].mean())
    #
    #         if z is None:
    #             z = np.random.standard_normal((1, 1)) * 1  # TODO CHEAT
    #             # z = -1 + 2 * np.random.randint(0, 2, shape=(1, 1))
    #         else:
    #             z = np.reshape(z, (1, 1))
    #         if self.use_x_start_end_ip:
    #             a_pred = self.net[0].predict(ip[0], batch_size=1)
    #             x_pred = self.net[1].predict([a_pred, ip[-2], ip[-1]], z)
    #
    #         else:
    #             a_pred = self.net[0].predict(ip, batch_size=1)
    #             try:
    #                 x_pred = self.net[1].predict([a_pred, z])
    #             except ValueError:
    #
    #                 x_pred = self.net[1].predict(np.concatenate([a_pred, z], axis=-1))
    #
    #         # a_pred /= a_pred.mean()
    #         # TODO a lot of zeros in bottleneck is not a good sign
    #
    #         # a_pred = np.concatenate([a_pred, z], axis=1)
    #         # print(x_pred)
    #     else:
    #         x_pred = self.net.predict(ip)
    #
    #     if isinstance(x_pred, list):
    #         x_pred_img = x_pred[0]
    #         x_pred = x_pred[-1]
    #
    #     # Update prediction
    #     if self.net_type == '22path_img_x':
    #         self.img_pred = ld.reshape_img(img=x_pred_img, n_dim=self.g.n_dim, n_samples=1, sample_dim=False,
    #                                        channel_dim=False)
    #
    #     if self.net_type == '2path_img':
    #         self.img_pred = ld.reshape_img(img=x_pred, n_dim=self.g.n_dim, n_samples=1, sample_dim=False,
    #                                        channel_dim=False)
    #         # self.x_pred = path_i.initialize_x0_rrt(x_start=self.x_start, x_end=self.x_end,
    #         #                                        n_wp=self.g.n_wp, verbose=1,
    #         #                                        p=lambda:
    #         #                                        path_i.sample_from_distribution(prob_img=self.img_pred,
    #         #                                                                        world_size=self.g.world_size,
    #         #                                                                        shape=1)[0])
    #
    #         self.q_pred = self.q.copy()
    #
    #     if self.net_type == '2x':
    #         self.q_pred = x_pred
    #
    #     if self.net_type == '2x_delta':
    #         x_pred = path.delta_x2x(x_delta=x_pred, x_start=self.q_start, x_end=self.q_end, denormalize=True,
    #                                 n_dim=self.g.n_dim)
    #         self.q_pred = x_pred
    #
    #     if self.net_type == '2x_inner' or self.net_type == '22path_img_x':
    #         if self.g.lll is None:
    #             x_pred = path.denorm_x(x=x_pred, world_size=self.g.world_size)
    #         else:
    #             if not self.g.fixed_base:
    #                 x_pred[..., :self.g.n_dim] = path.denorm_x(x=x_pred[..., :self.g.n_dim], world_size=self.g.world_size)
    #             else:
    #                 x_base = path.q2x_q(xq=self.q, n_dim=self.g.n_dim, n_joints=self.g.n_joints)[0]
    #                 x_pred = np.concatenate((x_base[..., 1:-1, :self.g.n_dim], x_pred[0]), axis=-1)
    #             x_pred[..., self.g.n_dim:] = forward.denorm_a(a=x_pred[..., self.g.n_dim:], zero_centered=False)
    #
    #         x_pred = path.x_inner2x(x_inner=x_pred, x_start=self.q_start, x_end=self.q_end,
    #                                 n_dof=self.g.n_dim + self.g.n_joints)
    #         self.q_pred = x_pred
    #
    #     # else:
    #     #     raise ValueError('Incompatible net_type')
    #
    #     if self.g.lll is not None:
    #         self.xs_pred = forward.xa2x_warm_2d(xa=self.q_pred, lll=self.g.lll, n_dim=self.g.n_dim,
    #                                             n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
    #                                             links_only=True, with_base=True)
    #     # Update Objective
    #     self.obj_pred = self.update_objective(x=self.q_pred)
    #
    #     return self.q_pred
    #
    # def plot_prediction(self, event=None):
    #     self.predict()
    #     if self.net is not None:
    #         if self.net_type == '2path_img' or self.net_type == '22path_img_x':
    #             if self.g.lll is None:
    #                 self.img_pred_plot_h.set_array(self.img_pred.T)
    #             else:
    #                 for o, h in enumerate(self.img_pred_plot_h):
    #                     h.set_array(self.img_pred[..., o].T)
    #
    #         if self.g.lll is None:
    #             self.plot_x_path(x=self.q_pred, h=self.x_pred_plot_h)
    #             self.x_pred_plot_h.set_label('Prediction, Cost: {}'.format(self.obj_pred))
    #         else:
    #             self.plot_x_path(x=self.xs_pred, h=self.x_pred_plot_h)
    #             self.x_pred_plot_h[0][0].set_label('Prediction, Cost: {}'.format(self.obj_pred))
    #
    #     self.ax.legend(loc='upper right')
    #     plt.draw()
    #
    # def update_objective(self, x):
    #
    #     obstacle_cost_fun = \
    #         cost_f.obstacle_img2cost_fun(obstacle_img=self.obstacle_img_cur, r_spheres=self.g.r,
    #                                      interp_order=0, eps=self.eps_obstacle_cost,
    #                                      world_size=self.g.world_size)
    #
    #     if self.g.lll is None:
    #         length_norm = path.get_start_end_normalization(q_start=self.q_start, q_end=self.q_end,
    #                                                        n_wp=self.g.n_wp)
    #     else:
    #         length_norm = forward.get_beeline_normalization(q_start=self.q_start, q_end=self.q_end,
    #                                                         n_wp=self.g.n_wp, n_joints=self.g.n_joints,
    #                                                         n_spheres_tot=self.g.n_spheres_tot,
    #                                                         lll=self.g.lll, fixed_base=self.g.fixed_base,
    #                                                         is_periodic=self.g.is_periodic)
    #
    #     obj = cost.chomp_cost(x_inner=path.x2x_inner(x=x, n_dof=self.g.n_dim + self.g.n_joints), n_dim=self.g.n_dim,
    #                           robot_id=self.g.lll, x_start=self.q_start, x_end=self.q_end, gamma_len=self.gamma_cost,
    #                           n_substeps=5, obst_cost_fun=obstacle_cost_fun, length_norm=length_norm,
    #                           fixed_base=self.g.fixed_base, return_separate=False,
    #                           is_periodic=self.g.is_periodic,
    #                           n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot, n_wp=self.g.n_wp)
    #
    #     return obj
    #
    # # RRT, Optimizer
    # def initial_guess_plot(self, event=None, random=False):
    #
    #     if random:
    #         prediction_img = np.ones_like(self.img_pred)
    #         print('random')
    #     else:
    #         prediction_img = self.img_pred
    #
    #     # _x_path = opt2.prediction_2_initial_path(xy_start=self.xy_start, xy_end=self.xy_end,
    #     #                                          prediction=self.prediction_img, world_size=world_size, n_wp=9)
    #     # TODO other methods for coming up with an initial guess
    #     tree, _x_path = path_i.initialize_path_rrt(x_start=self.q_start, x_end=self.q_end,
    #                                                n_wp=self.g.n_wp,
    #                                                verbose=2, return_tree=True,
    #                                                p=lambda: path_i.sample_from_image(prob_img=prediction_img,
    #                                                                                   world_size=self.g.world_size,
    #                                                                                   shape=1)[0])
    #
    #     if self.x_pred_plot_h is None:
    #         self.x_pred_plot_h = self.ax.plot(_x_path[:, 0], _x_path[:, 1])[0]
    #     else:
    #         self.x_pred_plot_h.set_xdata(_x_path[:, 0])
    #         self.x_pred_plot_h.set_ydata(_x_path[:, 1])
    #
    # def optimize(self, event=None):
    #     self.update_x_start_end_path()
    #
    #     # Obstacles in cost function
    #     obstacle_cost_fun = \
    #         cost_f.obstacle_img2cost_fun(obstacle_img=self.obstacle_img_cur, r_spheres=self.g.r,
    #                                      interp_order=0, eps=self.eps_obstacle_cost,
    #                                      world_size=self.g.world_size)
    #
    #     obstacle_cost_fun_grad = \
    #         cost_f.obstacle_img2cost_fun_grad(obstacle_img=self.obstacle_img_cur, r_spheres=self.g.r,
    #                                           eps=self.eps_obstacle_cost, world_size=self.g.world_size, interp_order=1)
    #
    #     x0 = path.x2x_inner(x=self.q_pred, n_dof=self.g.n_dim + self.g.n_joints).copy()
    #     x_opt, obj_opt = \
    #         opt2.gradient_descent_path_cost(x_inner=x0, n_dim=self.g.n_dim,
    #                                         x_start=self.q_start, x_end=self.q_end,
    #                                         gamma_grad=self.gamma_grad.copy(), gamma_cost=self.gamma_cost,
    #                                         n_ss_obst_cost=5, obst_cost_fun=obstacle_cost_fun,
    #                                         obst_cost_fun_grad=obstacle_cost_fun_grad, fixed_base=self.g.fixed_base,
    #                                         gd_step_number=self.gd_step_number, gd_step_size=self.gd_step_size,
    #                                         adjust_gd_step_size=self.adjust_gd_step_size, length_norm=True,
    #                                         lll=self.g.lll,
    #                                         return_separate=False,
    #                                         is_periodic=self.g.is_periodic,
    #                                         n_wp=self.g.n_wp,
    #                                         constraints_x=self.g.constraints_x, constraints_q=self.g.constraints_q)
    #
    #     self.obj_pred_opt = obj_opt
    #
    #     self.q_opt = path.x_inner2x(x_inner=x_opt, x_start=self.q_start, x_end=self.q_end,
    #                                 n_dof=self.g.n_dim + self.g.n_joints)
    #     if self.g.lll is not None:
    #         self.xs_pred_opt = forward.xa2x_warm_2d(xa=self.q_opt, lll=self.g.lll, n_dim=self.g.n_dim,
    #                                                 n_joints=self.g.n_joints, n_spheres_tot=self.g.n_spheres_tot,
    #                                                 links_only=True, with_base=True)
    #
    #     # Plot
    #     if self.x_opt_plot_h is None:
    #         if self.g.lll is None:
    #             self.x_opt_plot_h = plt2.plot_x_path(x=self.q_opt, marker='o', color='c', ax=self.ax,
    #                                                  label='Optimizer, Cost: {}'.format(self.obj_pred_opt))
    #         else:
    #             self.x_opt_plot_h = plt2.plot_x_spheres(x_spheres=self.xs_pred_opt, ax=self.ax)
    #
    #     else:
    #         if self.g.lll is None:
    #             self.plot_x_path(x=self.q_opt, h=self.x_opt_plot_h)
    #             self.x_opt_plot_h.set_label('Optimizer, Cost: {}'.format(self.obj_pred_opt))
    #         else:
    #             self.plot_x_path(x=self.xs_pred_opt, h=self.x_opt_plot_h)
    #             self.x_opt_plot_h[0][0].set_label('Optimizer, Cost: {}'.format(self.obj_pred_opt))
    #     self.ax.legend()
    #
    # # Visualize layers of the net
    # def visualize_layer_activation(self):
    #     self.update_x_start_end_path()
    #
    #     x = self.get_prediction_input()
    #
    #     visn.plot_layer_activation(model=self.net, x=x, layers='conv', plot_prediction=False,
    #                                figure_name=str(self.vis_count) + '_')
    #     self.vis_count += 1
    #
    # # Helper
    # def set_obstacle_img_ref(self):
    #     self.obstacle_img_cur = self.obstacle_img_cur[self.i_world].copy()
    #
    # def plot_cost_img(self):
    #     cost_img = cost_f.obstacle_img2cost_img(self.obstacle_img_cur, world_size=self.g.world_size,
    #                                             r_spheres=self.g.r,
    #                                             eps=self.eps_obstacle_cost)
    #     plt2.world_imshow(img=cost_img, limits=self.g.world_size, ax=self.ax, alpha=0.6)
    #
    # def save_fig(self, file='test',
    #              plot_path_label=True, plot_x_pred=True, plot_pred_img=True,
    #              cmap='Blues', plot_new_obstacles=False,
    #              save=True):
    #
    #     if '/' in file:
    #         file_split = file.split('/')
    #         file = file_split[-1]
    #
    #         if len(file_split) > 2:
    #             img_dir = '/'.join(file_split[:-1])
    #         else:
    #             img_dir = file_split[0]
    #         img_dir += '/'
    #     else:
    #         sample_dir = d.arg_wrapper__sample_dir(self.directory, full=False)
    #         img_dir = d.PROJECT_DATA_IMAGES + 'IA/' + sample_dir
    #         if not os.path.exists(img_dir):
    #             os.makedirs(img_dir)
    #     file = img_dir + 'w{}p{}_{}'.format(self.i_world, self.i_sample_local, file)
    #
    #     fig, ax = plt2.new_world_fig(limits=self.g.world_size, scale=2)
    #
    #     # Obstacles
    #     plt2.plot_img_outlines(img=self.obstacle_img_cur, ax=ax, world_size=self.g.world_size,
    #                            color='k', ls='-', lw=2)
    #     plt2.plot_img_patch(img=self.obstacle_img_cur, ax=ax, hatch='xxx', color='k',
    #                         lw=0, alpha=0.9)
    #
    #     if plot_new_obstacles:
    #         obstacle_img_new = self.obstacle_img_cur.astype(int) - self.obstacle_img_list[self.i_world].astype(int)
    #         obstacle_img_new[obstacle_img_new < 0] = 0
    #         obstacle_img_new = obstacle_img_new.astype(bool)
    #         if obstacle_img_new.sum() > 0:
    #             plt2.plot_img_patch(img=obstacle_img_new, ax=ax, hatch='xxx',
    #                                 color=c_tum['orange'], lw=0, alpha=0.9)
    #             plt2.plot_img_outlines(img=obstacle_img_new, ax=ax, world_size=self.g.world_size,
    #                                    color=c_tum['orange'], ls='-', lw=2)
    #
    #     # Start, End
    #     if self.g.lll is None:
    #
    #         # ax.set_ylim([20, 80])  # TODO  for FINAL/SingleSphereRobots/global_pos_switch_2D_SR_2dof.py
    #         # y = np.arange(40, 61, 2)
    #         # for yy in y:
    #         #     circle = Circle(xy=np.array([90.0, yy]), radius=self.g.r, fc='r', hatch='\\\\\\\\', alpha=0.2)
    #         #     ax.add_patch(circle)
    #
    #         circle_a = Circle(xy=self.q_start, radius=self.g.r, fc='g', hatch='////', alpha=1, zorder=100)
    #         circle_e = Circle(xy=self.q_end, radius=self.g.r, fc='r', hatch='\\\\\\\\', alpha=1, zorder=100)
    #         ax.add_patch(circle_a)
    #         ax.add_patch(circle_e)
    #
    #     else:
    #         for xy in self.xs_start:
    #             circle_a = Circle(xy=xy.flatten(), radius=self.g.r, fc='g', hatch='////', alpha=1, zorder=100)
    #             ax.add_patch(circle_a)
    #
    #         for xy in self.xs_end:
    #             circle_a = Circle(xy=xy.flatten(), radius=self.g.r, fc='r', hatch='\\\\\\\\', alpha=1, zorder=100)
    #             ax.add_patch(circle_a)
    #
    #     if self.g.fixed_base:
    #         circle_b = Circle(xy=self.xs_start[0, :].flatten(), radius=self.g.r,
    #                           fc='xkcd:dark gray', hatch='XXXX', alpha=1, zorder=200)
    #         ax.add_patch(circle_b)
    #         ax.add_patch(circle_b)
    #
    #     if self.g.lll is not None:
    #         plt2.plot_x_spheres(x_spheres=self.xs_start, ax=ax, c='g', lw=5)
    #         plt2.plot_x_spheres(x_spheres=self.xs_end, ax=ax, c='r', lw=5)
    #         if plot_path_label:
    #             plt2.plot_x_spheres(x_spheres=self.xs_path, ax=ax)
