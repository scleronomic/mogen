import numpy as np

import Optimizer.InitialGuess.path as path_i
import Optimizer.gradient_descent as gd_ms
import Optimizer.choose_optimum as choose_optimum
import Optimizer.path as path
import mogen.SampleGeneration.sample_start_end as se_sampling
import Util.Visualization.plotting_2 as plt2
import World.random_obstacles as randrect
import World.swept_volume as w2i
import parameter

robot_id = 'Single_Sphere_02'
par = parameter.initialize_par(robot_id=robot_id)

# Parameters
par.size.n_waypoints = 20
n_obstacles = 10
min_max_obstacle_size_voxel = [4, 8]

par.tcp.frame = None  # np.array([[[5, 7]]])

par.oc.n_substeps = 2
par.oc.n_substeps = 2

par.planning.include_end = False
par.planning.length = False
par.pbp.wp_idx = np.array([], dtype=int)
# par.pbp.q = par.head.frame
par.pbp.joint_weighting = np.array([np.ones(par.robot.n_dim)], dtype=float)

# Optimization
# Gradient Descent
gd = parameter.GradientDescent()
gd.n_steps = 100
gd.step_size = 0.01
gd.adjust_step_size = True
gd.staircase_iteration = False
gd.grad_tol = np.full(gd.n_steps, 1)
gd.n_processes = 6
gd.prune_limits = limits.prune_limits_wrapper(robot=par.robot)
gd.callback = None

weighting = parameter.Weighting()
weighting.joint_motion = np.full(par.robot.n_dof, 1 / par.robot.n_dof)
weighting.length = np.linspace(start=1, stop=1, num=gd.n_steps)
weighting.collision = np.linspace(start=100, stop=1000, num=gd.n_steps)
weighting.rotation = 1
par.weighting = weighting

# Number of multi starts
n_multi_start_rp = [[0, 1, 2], [1, 4 * gd.n_processes, 4 * gd.n_processes - 1]]
# n_multi_start_rp = [[1], [1]]

obstacle_img = randrect.create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel,
                                               n_voxels=par.world.n_voxels)

if par.tcp.frame is not None:
    obstacle_img = w2i.sphere2grid_whole(x=par.tcp.frame[0], r_sphere=0.5, voxel_size=par.world.voxel_size,
                                         img=obstacle_img)

parameter.initialize_oc(oc=par.oc, world=par.world, obstacle_img=obstacle_img)

# Generator of multi starts
get_x0 = path_i.q0_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
                                  n_waypoints=par.size.n_waypoints, order_random=True)

q_start, q_end = se_sampling.sample_q_start_end(par=par, n_samples=1)
# Horizontal
# q_start = np.array([[[1, 5]]])
# q_end = np.array([[[9, 5]]])

# Vertical
# xq_start = np.array([[5, 1]])
# xq_end = np.array([[5, 9]])

# Diagonal
# q_start = np.array([[[2, 2]]])
# q_end = np.array([[[8, 8]]])

# a = x0_generator_rp(xa_start, xa_end)
# for c in a:
#     fig, ax = new_fig()
#     ax.plot(c)


###
# x0 = init_guess_tcp.sample_end_points_base(n_ms=n_ms_target, robot=par.robot, head=par.head.frame, q_close=q_start)
# q_end, all_x = gd_ms.gd_wrapper_tcp_goal_base(x0=x0, q_close=q_start, par=par, gd=gd_end_pos)
#
# print('TCP Base')
# ax = plt2.plot_obstacle_img(world_size=par.world.shape, voxel_size=par.world.voxel_size, img=obstacle_img)
# for o in range(all_x.shape[0]):
#     ax.plot(all_x[o, 0], all_x[o, 1], marker='o', color='b')
#
# ax.plot(par.head.frame[0, 0, 0], par.head.frame[0, 0, 1], marker='x', color='g', markersize=20)
# ax.plot(q_end[0, 0, 0], q_end[0, 0, 1], marker='x', color='r', markersize=20)

par.weighting = weighting

x0 = get_x0(x_start=q_start, x_end=q_end)

x_initial, q_ms, objective = gd_ms.gd_chomp(q0=x0, q_start=q_start, q_end=q_end,
                                            gd=gd, par=par, return_all=True, verbose=1)
q_opt, _ = choose_optimum.get_feasible_optimum(q=q_ms, par=par, verbose=0)
q_opt = path.x_inner2x(inner=q_opt, start=q_start, end=q_end)

# return all
fig, ax = plt2.new_world_fig(limits=par.world.limits)
plt2.imshow(img=obstacle_img, limits=par.world.limits, ax=ax, vmax=0.9, vmin=0.5, cmap='Greys', zorder=-10)
markersize = size_units2points(size=2 * np.max(par.oc.r_spheres), ax=ax, reference='y')
plt2.plot_x_path(x=q_opt, ax=ax, marker='o', markersize=markersize, color='b')
for q_temp in q_ms:
    plt2.plot_x_path(x=q_temp, ax=ax, marker='o', markersize=markersize / 4, color='k', alpha=0.5)
