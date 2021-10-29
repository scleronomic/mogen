import matplotlib as mpl
import numpy as np
from wzk import new_fig, save_fig, get_points_inbetween

import mogen.Loading.load_pandas as ld
import wzk.sql2 as ld_sql
import Util.Visualization.plotting_2 as plt2

directory = '2D/SR/2dof/'  # '2D/FB/3dof/'

save_img = False
i_world = list(range(0, 5000))
plot_prediction = True
plot_worlds = True

# Load Data
i_world_list = ld.arg_wrapper__i_world(i_worlds=i_world, directory=directory)
n_worlds = len(i_world_list)
world_df = ld.load_world_df(directory=directory)
obstacle_img = ld.add_obstacle_img_column(world_df=world_df, values_only=True)
x, objective = ld_sql.get_values_sql(columns=[PATH_Q, 'objective'],
                                     i_worlds=i_world, directory=directory, values_only=True)
img_dir = PROJECT_DATA_IMAGES + 'World_Statistics/' + directory
n_samples = x.shape[0]
objective = objective.reshape(n_worlds, n_samples_per_world)

x_inner = path.x2x_inner(x=x)

# OBJECTIVE
# Histogram of path costs over worlds
hist_list = []
hist_range = (1, 5)
bins = 50
base = (hist_range[1] - hist_range[0]) / bins
x_hist = hist_range[0] + base / 2 + np.arange(bins) * base
for iw in range(n_worlds):
    hist_list.append(np.histogram(objective[iw], bins=bins, range=hist_range)[0])

fig, ax = new_fig()
ax.set_title('Distribution of the Objective Costs per World')
ax.set_xlabel('Objective Cost')
ax.set_ylabel('Number of RandomRectangles')

for iw in range(n_worlds):
    ax.plot(x_hist, hist_list[iw], c='k', alpha=0.05)

hist_total = np.histogram(objective.flatten(), bins=bins, range=hist_range)[0] / n_worlds
ax.plot(x_hist, hist_total, c='r', alpha=1)
ax.set_xlim(hist_range[0], hist_range[1])
ax.set_ylim(0.05, 300)
save_fig(img_dir + 'hist__Objective_Costs_per_World', fig=fig, save=save_img)

fig, ax = new_fig()
ax.set_title('Distribution of the Objective Costs')
ax.set_xlabel('Objective Cost')
ax.set_ylabel('Number of Paths')
ax.hist(objective.flatten(), bins=bins, range=(1, 10))

# Histogram for mean world objective
objective_pw = objective.mean(axis=1)

##
# objective_ss = objective.reshape((n_worlds, d.n_samples_per_world)).copy()
# objective_ss_idx = np.arange(n_worlds*d.n_samples_per_world).reshape((n_worlds, d.n_samples_per_world))
#
# objective_pw_sorted_idx = np.argsort(objective_pw)
# objective_ss = objective_ss[objective_pw_sorted_idx, :]
# objective_ss_idx = objective_ss_idx[objective_pw_sorted_idx, :]
#
# for o in range(n_worlds):
#     objective_i_sorted_idx = np.argsort(objective_ss[o])
#     objective_ss[o, :] = objective_ss[o, objective_i_sorted_idx]
#     objective_ss_idx[o, :] = objective_ss_idx[o, objective_i_sorted_idx]
#
# sample_indices = sample_indices.reshape((n_batches, self.batch_size))
# sample_indices = sample_indices[np.random.permutation(n_batches), :]
# sample_indices = sample_indices.flatten()
#
# n_squares_pw = 10
#
# squares = np.zeros((n_worlds, n_squares_pw))
# square_size = d.n_samples_per_world // n_squares_pw
# for w in range(n_worlds):
#     for s in range(n_squares_pw):
#         squares[w, s] = objective_ss[w, s*square_size:(s+1)*square_size].mean()
#
# fig, ax = new_fig()
# ax.imshow(squares.T, aspect='auto')
#
#
# fig, ax = new_fig()
# ax.hist(squares.flatten(), bins=20)
#
# # remove successive starting from the easiest square, keep every 10th square, to 75 % , every

##

fig, ax = new_fig()
ax.set_title('Distribution of the Mean Objective Costs per World')
ax.set_xlabel('Mean Objective Cost')
ax.set_ylabel('Number of RandomRectangles')
hist_range = (1, 3)
bins = 50
base = (hist_range[1] - hist_range[0]) / bins
x_hist = hist_range[0] + base / 2 + np.arange(bins) * base
hist_worlds = np.histogram(objective_pw, bins=bins, range=hist_range)[0]
ax.plot(x_hist, hist_worlds, c='r', alpha=1)
save_fig(img_dir + 'hist__Mean_Objective_Costs_per_World', fig=fig, save=save_img)

# Obstacle Coverage
obst_coverage = obstacle_img.mean(axis=(1, 2))
obst_coverage_argsort = np.argsort(obst_coverage)

# Obstacle Coverage vs. Mean World Objective
fig, ax = new_fig()
ax.set_title('Obstacle Coverage vs. Mean World Objective')
ax.set_xlabel('Obstacle Coverage')
ax.set_ylabel('Mean World Objective')
img = ax.hist2d(obst_coverage, objective_pw, norm=mpl.colors.LogNorm(), bins=100)[-1]
fig.colorbar(img, ax=ax)
save_fig(img_dir + 'hist2D__Obstacle_Coverage_vs_Mean_World_Objective', fig=fig, save=save_img)


# SPECIAL WORLDS
if plot_worlds:
    # Hardest (most expensive) World
    iw = np.argmax(objective_pw)
    fig, ax = plt2.new_world_fig(limits=g.world_size)
    ax.set_title('World: {}, Coverage: {}, Mean Objective: {} - most expensive'.
                 format(iw, obst_coverage[iw], objective_pw[iw]))
    plt2.plot_img_patch_w_outlines(limits=g.world_size, shape=g.shape, img=obstacle_img[iw], ax=ax, n_dim=g.n_dim)
    save_fig(img_dir + 'world__most_expensive', fig=fig, save=save_img)

    # Easiest (Cheapest) World
    iw = np.argmin(objective_pw)
    fig, ax = plt2.new_world_fig(limits=g.world_size)
    ax.set_title('World: {}, Coverage: {}, Mean Objective: {} - cheapest'.
                 format(iw, obst_coverage[iw], objective_pw[iw]))
    plt2.plot_img_patch_w_outlines(limits=g.world_size, shape=g.shape, img=obstacle_img[iw], ax=ax, n_dim=n_dim)
    save_fig(img_dir + 'world__cheapest', fig=fig, save=save_img)

    # Densest World
    iw = obst_coverage_argsort[-1]
    fig, ax = plt2.new_world_fig(limits=g.world_size)
    ax.set_title(
        'World: {}, Coverage: {}, Mean Objective: {} - densest'.format(iw, obst_coverage[iw], objective_pw[iw]))
    plt2.plot_img_patch_w_outlines(limits=g.world_size, shape=g.shape, img=obstacle_img[iw], ax=ax, n_dim=n_dim)
    save_fig(img_dir + 'world__densest', fig=fig, save=save_img)

    # Second densest
    iw = obst_coverage_argsort[-2]
    fig, ax = plt2.new_world_fig(limits=g.world_size)
    ax.set_title('World: {}, Coverage: {}, Mean Objective: {} - second densest'.
                 format(iw, obst_coverage[iw], objective_pw[iw]))
    plt2.plot_img_patch_w_outlines(limits=g.world_size, shape=g.shape, img=obstacle_img[iw], ax=ax, n_dim=g.n_dim)
    save_fig(img_dir + 'world__densest2', fig=fig, save=save_img)
    # TODO way cheaper because of advantageous map layout, edges ar blocked and only the center is free

# PATH LENGTH
path_length = path.path_length(x, n_dim=n_dim, n_samples=n_samples)
direct_path_length = path.linear_distance(x=x, n_samples=n_samples)
relative_path_length = path_length / direct_path_length
# x_norm = path.get_start_end_normalization()


# Paths
# Histogram
fig, ax = new_fig()
ax.set_title('Distribution of the Path Length')
ax.set_xlabel('Path Length, mean={}'.format(path_length.mean()))
ax.set_ylabel('Number of Paths')
ax.hist(path_length, bins=50, range=[0, 230])
save_fig(img_dir + 'hist__path_length', fig=fig, save=save_img)

fig, ax = new_fig()
ax.set_title('Distribution of the Direct Path Length')
ax.set_xlabel('Linear distance between start and end point, mean={}'.format(direct_path_length.mean()))
ax.set_ylabel('Number of Paths')
ax.hist(direct_path_length, bins=50, range=[0, 130])
save_fig(img_dir + 'hist__direct_path_length', fig=fig, save=save_img)

fig, ax = new_fig()
ax.set_title('Distribution of the Relative Path Length')
ax.set_xlabel('Relative Path Length, mean={}'.format(relative_path_length.mean()))
ax.set_ylabel('Number of Paths')
(y_hist, x_hist, _) = ax.hist(relative_path_length, bins=1000, range=[0.999, 2])
save_fig(img_dir + 'hist__relative_path_length', fig=fig, save=save_img)
# TODO almost 50% of all paths are direct, test what happens if you use only a small fraction of such examples

fig, ax = new_fig()
ax.plot(get_points_inbetween(x_hist), y_hist.cumsum())

# Relative Path Length vs. Objective
fig, ax = new_fig()
ax.set_xlabel('Relative Path Length')
ax.set_ylabel('Objective')
img = ax.hist2d(relative_path_length, objective.flatten(), norm=mpl.colors.LogNorm(), bins=100,
                range=[[1, 3.5], [1, 15]])[-1]
fig.colorbar(img, ax=ax)
ax.set_xlim([1, 3.5])
ax.set_ylim([1, 15])
save_fig(img_dir + 'hist2D__Relative_Path_Length_vs_Objective', fig=fig, save=save_img)

# Direct Path Length vs. Objective
fig, ax = new_fig()
ax.set_xlabel('Direct Path Length')
ax.set_ylabel('Objective')
img = ax.hist2d(direct_path_length, objective.flatten(), norm=mpl.colors.LogNorm(), bins=100,
                range=[[1, 130], [1, 15]])[-1]
fig.colorbar(img, ax=ax)
ax.set_xlim([1, 130])
ax.set_ylim([1, 15])
save_fig(img_dir + 'hist2D__Direct_Path_Length_vs_Objective', fig=fig, save=save_img)

fig, ax = new_fig()

path_length_pw = path_length.reshape(n_worlds, dfn.n_samples_per_world).mean(axis=1)
direct_path_length_pw = direct_path_length.reshape(n_worlds, dfn.n_samples_per_world).mean(axis=1)
relative_path_length_pw = relative_path_length.reshape(n_worlds, dfn.n_samples_per_world).mean(axis=1)

# Mean World Relative Path Length vs. Mean World Objective
fig, ax = new_fig()
ax.set_xlabel('Mean World Relative Path Length')
ax.set_ylabel('Mean World Objective')
# ax.scatter(relative_path_length_pw, objective_pw, alpha=0.5)
img = ax.hist2d(relative_path_length_pw, objective_pw, norm=mpl.colors.LogNorm(), bins=100)[-1]
fig.colorbar(img, ax=ax)
save_fig(img_dir + 'scatter__MW_Relative_Path_Length_vs_MW_Objective', fig=fig, save=save_img)

# Mean World Direct Path Length vs. Mean World Objective
fig, ax = new_fig()
ax.set_xlabel('Mean World Direct Path Length')
ax.set_ylabel('Mean World Objective')
# ax.scatter(direct_path_length_pw, objective_pw, alpha=0.5)  # TODO switch between scatter and hist2d
img = ax.hist2d(direct_path_length_pw, objective_pw, norm=mpl.colors.LogNorm(), bins=100)[-1]
fig.colorbar(img, ax=ax)
save_fig(img_dir + 'scatter__MW_Direct_Path_Length_vs_MW_Objective', fig=fig, save=save_img)

# NET LOSS
if plot_prediction:
    x_pred, objective_pred = ld_sql.get_values_sql(columns=['x_pred', 'objective_pred'],
                                                   i_worlds=i_world, directory=directory, values_only=True)
    x_pred_inner = path.x2x_inner(x=x_pred, n_dof=n_dim, n_samples=n_samples)

    loss = c_loss.euclidean_loss_np(y_true=x_inner, y_pred=x_pred_inner, n_dim=n_dim, batch_size=n_samples,
                                    world_size=1)
    loss_pw = loss.reshape(n_worlds, dfn.n_samples_per_world).mean(axis=1)

    fig, ax = new_fig()
    ax.set_xlabel('Loss of Neural Network, mean={}'.format(loss.mean()))
    ax.hist(loss, bins=50, range=[0, 10])
    save_fig(img_dir + 'hist__Network_Loss', fig=fig, save=save_img)

    # Mean World Objective vs. Mean Network Loss
    fig, ax = new_fig()
    ax.set_xlabel('Mean World Objective')
    ax.set_ylabel('Mean World Network Loss')
    # ax.scatter(objective_pw, loss_pw, alpha=0.5)
    img = ax.hist2d(objective_pw, loss_pw, norm=mpl.colors.LogNorm(), bins=100)[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__MW_Objective_vs_MW_Network_Loss', fig=fig, save=save_img)
    # TODO look at the outliers, why is a expensive worlds easy for the network and the other way round

    fig, ax = new_fig()
    ax.set_xlabel('Objective')
    ax.set_ylabel('Network Loss')
    # ax.scatter(objective_pw, loss_pw, alpha=0.5)
    img = ax.hist2d(objective.flatten(), loss, norm=mpl.colors.LogNorm(), bins=100, range=[[1, 10], [0, 40]])[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__Objective_vs_Network_Loss', fig=fig, save=save_img)

    # Objective vs Prediction Objective
    fig, ax = new_fig()
    ax.set_xlabel('Objective')
    ax.set_ylabel('Objective Prediction')
    # ax.scatter(objective_pw, loss_pw, alpha=0.5)
    img = ax.hist2d(objective.flatten(), objective_pred, norm=mpl.colors.LogNorm(),
                    bins=100, range=[[1, 10], [0, 200]])[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__Objective_vs_Objective_pred', fig=fig, save=save_img)

    # Coverage_vs_Network_Loss
    fig, ax = new_fig()
    ax.set_xlabel('Covered Area of Obstacles in Percent')
    ax.set_ylabel('Network Loss')
    # ax.scatter(world_df.n_obstacles, obst_coverage)
    img = ax.hist2d(obst_coverage, loss_pw, norm=mpl.colors.LogNorm(), bins=30)[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2d__Coverage_vs_Network_Loss', fig=fig, save=save_img)

    np.roll

    # Length - Loss
    # Mean World Direct Path Length vs. Mean Network Loss
    fig, ax = new_fig()
    ax.set_xlabel('Mean Direct Path Length')
    ax.set_ylabel('Mean World Network Loss')
    # ax.scatter(direct_path_length_pw, loss_pw, alpha=0.5)
    img = ax.hist2d(direct_path_length_pw, loss_pw, norm=mpl.colors.LogNorm(), bins=100)[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__MW_Direct_Path_Length_vs_MW_Network_Loss', fig=fig, save=save_img)

    fig, ax = new_fig()
    ax.set_xlabel('Path Length')
    ax.set_ylabel('Network Loss')
    # ax.scatter(path_length, loss, s=20, alpha=0.002, edgecolor='none')
    img = ax.hist2d(path_length, loss, norm=mpl.colors.LogNorm(), bins=100, range=[[0, 150], [0, 40]])[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'scatter__Path_Length_vs_MW_Network_Loss', fig=fig, save=save_img)

    fig, ax = new_fig()
    ax.set_xlabel('Relative Path Length')
    ax.set_ylabel('Network Loss')
    img = ax.hist2d(relative_path_length, loss, norm=mpl.colors.LogNorm(), bins=100, range=[[1, 3.5], [0, 40]])[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__Relative_Path_Length_vs_MW_Network_Loss', fig=fig, save=save_img)

    fig, ax = new_fig()
    ax.set_xlabel('Direct Path Length')
    ax.set_ylabel('Network Loss')
    img = ax.hist2d(direct_path_length, loss, norm=mpl.colors.LogNorm(), bins=100)[-1]
    fig.colorbar(img, ax=ax)
    save_fig(img_dir + 'hist2D__Direct_Path_Length_vs_MW_Network_Loss', fig=fig, save=save_img)

    if plot_worlds:
        # Plot path with worst prediction/ largest loss
        for i in range(1, 11):
            idx_sorted_loss = np.argsort(loss)
            ax = plt2.plot_sample(i_sample_global=idx_sorted_loss[-i], directory=directory,
                                  title='{}. worst prediction'.format(i))
            plt2.plot_x_path(x=x_pred[idx_sorted_loss[-i]], n_dim=g.n_dim, ax=ax, marker='o')
            save_fig(img_dir + 'world__Highest_Network_Loss_{}'.format(i), fig=fig, save=save_img)

        for i in range(1, 11):
            idx_sorted_loss = np.argsort(path_length)
            ax = plt2.plot_sample(i_sample_global=idx_sorted_loss[-i], directory=directory,
                                  title='{}. longest path'.format(i))
            plt2.plot_x_path(x=x_pred[idx_sorted_loss[-i]], n_dim=g.n_dim, ax=ax, marker='o')
            save_fig(img_dir + 'world__Longest_Path_{}'.format(i), fig=fig, save=save_img)

        for i in range(1, 11):
            idx_sorted_loss = np.argsort(direct_path_length)
            ax = plt2.plot_sample(i_sample_global=idx_sorted_loss[-i], directory=directory,
                                  title='{}. longest direct path'.format(i))
            plt2.plot_x_path(x=x_pred[idx_sorted_loss[-i]], n_dim=g.n_dim, ax=ax, marker='o')
            save_fig(img_dir + 'world__Longest_Direct_Path_{}'.format(i), fig=fig, save=save_img)

        for i in range(1, 11):
            idx_sorted_loss = np.argsort(relative_path_length)
            ax = plt2.plot_sample(i_sample_global=idx_sorted_loss[-i], directory=directory,
                                  title='{}. longest relative path'.format(i))
            plt2.plot_x_path(x=x_pred[idx_sorted_loss[-i]], n_dim=g.n_dim, ax=ax, marker='o')
            save_fig(img_dir + 'world__Longest_Relative_Path_{}'.format(i), fig=fig, save=save_img)

            # TODO Try to update paths with net + heuristics

fig, ax = new_fig()
ax.set_xlabel('Blocked ratio')
ax.hist(obst_coverage, bins=50)
save_fig(img_dir + 'hist__obstacle_coverage', fig=fig, save=save_img)

fig, ax = new_fig()
ax.set_xlabel('Number of Obstacles')
ax.hist(world_df.n_obstacles, bins=30, range=(0, 30))
save_fig(img_dir + 'hist__n_obstacles', fig=fig, save=save_img)

fig, ax = new_fig()
ax.set_xlabel('Number of Obstacles')
ax.set_ylabel('Covered Area  of Obstacles in Percent')
# ax.scatter(world_df.n_obstacles, obst_coverage)
img = ax.hist2d(world_df.n_obstacles, obst_coverage, norm=mpl.colors.LogNorm(), bins=30,
                range=[[0, 30], [0, 0.175]])[-1]
fig.colorbar(img, ax=ax)
save_fig(img_dir + 'hist2d__n_obstacles_vs_coverage', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Loop in path')
# ax.hist(path_df.loop_in_path, bins=3, range=[0, 2])
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist__loop_in_path', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Sum of path angles, mean={}'.format(path_df.path_angle.mean()))
# ax.hist(path_df.path_angle, bins=100, range=[0, 360])
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist__path_angle', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Tries, mean={}'.format(path_df.tries.mean()))
# ax.hist(path_df.tries, bins=40, range=[1, 41])
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist__tries', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Net Redo')
# ax.hist(path_df.net_redo, bins=4, range=[-1, 2])
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist__net_redo', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Objective')
# ax.hist(path_df.objective.values.astype(float), bins=50, range=[0, 300])
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist__objective', fig=fig, save=save_img)

# fig, ax = new_fig()
# ax.set_xlabel('Path length')
# ax.set_ylabel('Path angle')
# matrix = ax.hist2d(x=path_df.path_length, y=path_df.path_angle, bins=[100, 100], cmin=10, range=[[0, 200], [0, 360]])[-1]
# fig.colorbar(matrix)
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__hist2d__length_angle', fig=fig, save=save_img)
#


# fig, ax = new_fig()
# ax.set_xlabel('Path Length')
# ax.set_ylabel('Tries')
# ax.set_xlim([0, 200])
# ax.set_ylim([1, 41])
# ax.scatter(path_df.path_length.values, path_df.tries.values, s=100, alpha=0.005, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__length_tries', fig=fig, save=save_img)
#
# # fig, ax = new_fig()
# # ax.set_xlabel('Path length')
# # ax.set_ylabel('Path Tries')
# # matrix = ax.hist2d(x=path_df.path_length, y=path_df.tries, bins=[100, 40], cmin=10, range=[[0, 150], [1, 41]])[-1]
# # fig.colorbar(matrix)
#
#
# fig, ax = new_fig()
# ax.set_xlabel('Relative Path Length')
# ax.set_ylabel('Tries')
# ax.set_xlim([1, 3])
# ax.set_ylim([0, 42])
# ax.scatter(path_df.relative_path_length.values, path_df.tries.values, s=100, alpha=0.005, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__relative_length_tries', fig=fig, png_only=png_only,
#              save=save_img)
#
# fig, ax = new_fig()
# ax.set_xlabel('Path angle')
# ax.set_ylabel('Tries')
# ax.set_xlim([0, 360])
# ax.set_ylim([0, 42])
# ax.scatter(path_df.path_angle.values, path_df.tries.values, s=100, alpha=0.005, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__angles_tries', fig=fig, save=save_img)
#
# fig, ax = new_fig()
# ax.set_xlabel('Relative Path Length')
# ax.set_ylabel('Path angle')
# ax.set_xlim([1, 3])
# ax.set_ylim([0, 360])
# ax.scatter(path_df.relative_path_length.values, path_df.path_angle.values, s=100, alpha=0.002, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__relative_length_angle', fig=fig, save=save_img)
#
# fig, ax = new_fig()
# ax.set_xlabel('Path Length')
# ax.set_ylabel('Path angle')
# ax.set_xlim([0, 200])
# ax.set_ylim([0, 360])
# ax.scatter(path_df.path_length.values, path_df.path_angle.values, s=20, alpha=0.002, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__length_angle', fig=fig, save=save_img)
#
#
# fig, ax = new_fig()
# ax.set_xlabel('Tries')
# ax.set_ylabel('Loss')
# ax.set_xlim([1, 41])
# ax.set_ylim([0, 0.3])
# ax.scatter(path_df.tries.values, loss, s=20, alpha=0.002, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__tries_loss', fig=fig, save=save_img)
#
# fig, ax = new_fig()
# ax.set_xlabel('Path_angle')
# ax.set_ylabel('Loss')
# ax.set_xlim([0, 300])
# ax.set_ylim([0, 0.3])
# ax.scatter(path_df.path_angle.values, loss, s=20, alpha=0.002, edgecolor='none')
# save_fig(d.PROJECT_DATA_IMAGES + 'path_stat__scatter__angle_loss', fig=fig, save=save_img)
#
