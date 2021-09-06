import World.random_obstacles as randrect
import Util.Loading.load

import Util.Loading.load_sql as load_sql
from Util.Loading.dlr import get_sample_dir
import Util.Visualization.plotting_2 as plt2
from wzk import compressed2img, reduce_n_voxels, save_fig, concatenate_images
from definitions import *


def plot_sample(directory,
                i_world=None, i_sample_local=None,
                i_sample_global=None,
                model=None, ax=None, title=None,
                save=False,
                reduce_img=1):

    if i_sample_global is None:
        i_sample_global = Util.Loading.load.get_i_samples_global(i_worlds=i_world, i_samples_local=i_sample_local)
    else:
        i_world = Util.Loading.load.get_i_world(i_sample_global=i_sample_global)
        i_sample_local = Util.Loading.load.get_i_samples_local(i_sample_global=i_sample_global)

    n_dim = 2
    limits = None
    world_size, n_voxels, rect_pos, rect_size, lll, fixed_base = \
        load_sql.get_values_sql(file=directory + WORLD_DB, rows=i_world,
                                columns=['world_size', 'n_voxels', 'rectangle_pos',
                                         'rectangle_size', 'lll', 'fixed_base'])

    obstacle_img = randrect.rectangles2image(n_voxels=n_voxels, rect_pos=rect_pos, rect_size=rect_size)
                                             # safety_idx=fixed_base[0], safety_margin=fixed_base[1])

    x_start, x_path, path_img, start_img, end_img, objective = load_sql.get_values_sql(
        columns=[START_Q, PATH_Q, PATH_IMG_CMP, START_IMG_CMP, END_IMG_CMP, 'objective'],
        file=directory, values_only=True, rows=int(i_sample_global))

    # n_voxels = 16
    # dtype = float
    n_channels=None
    dtype = bool
    start_img = compressed2img(img_cmp=start_img, n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels, dtype=dtype)
    end_img = compressed2img(img_cmp=end_img, n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels, dtype=dtype)
    path_img = compressed2img(img_cmp=path_img, n_voxels=n_voxels, n_dim=n_dim, n_channels=n_channels, dtype=dtype)

    # Check overlap
    # overlap_img = w2i.check_overlap(b=obstacle_img, a=path_img, return_arr=True)
    # print('collision: ', overlap_img.any())

    # Reduce grid resolution
    if reduce_img > 1:
        obstacle_img = reduce_n_voxels(img=obstacle_img, n_voxels=n_voxels, n_dim=n_dim, n_channels=None,
                                       kernel=reduce_img, pooling_type='average', n_samples=None,
                                       sample_dim=False, channel_dim=False)
        # start_img = make wrapper function
        # end_img =
        # path_img =
        # overlap_img =
        n_voxels /= reduce_n_voxels

    # Create figure
    if ax is None:
        fig, ax = plt2.new_world_fig(limits=limits, n_dim=n_dim)
        if title is None:
            fig.suptitle('World: {}, Path: {} | Objective: {:.4}'.format(i_world, i_sample_local, objective))
        else:
            fig.suptitle('World: {}, Path: {} | Objective: {:.4} | {}'.
                         format(i_world, i_sample_local, objective, title))

    else:
        fig = ax.get_figure()

    if n_channels == 1:
        path_img = path_img[..., 0]
    if n_dim == 3:
        if lll is None:
            obstacle_img = (rect_pos, rect_size)  # TODO they aren't cut to the safety margin -> alter
            # In the other case with Justin/ FixedBase, the safety margin around the base needs to be found
    plt2.plot_img_patch_w_outlines(ax=ax, img=obstacle_img, limits=limits)

    # Plot path image
    if True:
        pass
        # plot_path_img(img=overlap_img, n_voxels=n_voxels, limits=limits, ax=ax, n_dim=n_dim)
        # plot_path_img(img=path_img, n_voxels=n_voxels, v=limits, ax=ax, n_dim=n_dim)
        # plot_path_img(img=end_img, n_voxels=n_voxels, limits=limits, ax=ax, n_dim=n_dim)
        # plot_path_img(img=start_img, n_voxels=n_voxels, limits=limits, ax=ax, n_dim=n_dim)

    else:
        pass
        # plt2.plot_path_img(ax=ax, img=path_img, limits=limits)
        # path_img = path_img.sum(axis=-1)
        # start_img = start_img.sum(axis=-1)
        # end_img = end_img.sum(axis=-1)

    plt2.plot_x_path(x=x_path, ax=ax, marker='o')

    if n_dim == 2:
        plt2.imshow(img=start_img, limits=limits, ax=ax, alpha=1)
        plt2.imshow(img=end_img, limits=limits, ax=ax, alpha=1)
        plt2.imshow(img=path_img, limits=limits, ax=ax, alpha=0.5)
    #
    if model is not None:  # Old, only for img2img
        x_path = concatenate_images(obstacle_img, start_img, end_img, axis=-1)
        prediction_img = model.predict(x_path)
        plt2.imshow(img=prediction_img, limits=limits, ax=ax)

    if save:
        sample_dir = get_sample_dir(directory=directory, full_path=False)
        if not os.path.exists(PROJECT_DATA_IMAGES + sample_dir):
            os.makedirs(PROJECT_DATA_IMAGES + sample_dir)

        save_fig(PROJECT_DATA_IMAGES + sample_dir + 'w{}p{}'.format(i_world, i_sample_local), fig=fig)

    return ax
