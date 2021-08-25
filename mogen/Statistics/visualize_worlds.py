from wzk import new_fig

import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import Util.Visualization.plotting_2 as plt2
import definitions as d

save = True


def plot_world_paths(directory, iw, i_link=None, n_channels=9,
                     print_starts_ends=False):
    image_dir = d.PROJECT_DATA_IMAGES + d.arg_wrapper__sample_dir(directory=directory, full=False)
    d.safe_create_dir(image_dir)

    world_df = ld.load_world_df(directory=directory)
    world_size, n_voxels, lll = world_df.loc[iw, ['world_size', 'n_voxels', 'lll']]

    obstacle_img = ld.add_obstacle_img_column(world_df=world_df.loc[iw], values_only=True)[0]
    start_img_cmp, end_img_cmp, path_img_cmp = \
        ld_sql.get_values_sql(directory=directory, columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP],
                              i_worlds=iw, values_only=True)
    paths_img = compressed2img(path_img_cmp, n_voxels=n_voxels, n_dim=2, n_channels=n_channels).sum(axis=0)
    starts_img = compressed2img(start_img_cmp, n_voxels=n_voxels, n_dim=2, n_channels=n_channels).sum(axis=0)
    ends_img = compressed2img(end_img_cmp, n_voxels=n_voxels, n_dim=2, n_channels=n_channels).sum(axis=0)
    starts_ends_img = starts_img + ends_img
    if i_link is not None:
        paths_img = paths_img[..., i_link]
        starts_ends_img = starts_ends_img[..., i_link]

    fig, ax = plt2.new_world_fig(limits=world_size)
    plt2.plot_obstacle_path_world(world_size=world_size, n_voxels=n_voxels, obstacle_img=obstacle_img, ax=ax)
    plt2.plot_prediction_world(img=paths_img, world_size=world_size, ax=ax)
    save_fig(image_dir + 'w{}_l{}_paths_img_combined'.format(iw, i_link), fig=fig, save=save)

    if print_starts_ends:
        fig, ax = plt2.new_world_fig(limits=world_size)
        plt2.plot_obstacle_path_world(world_size=world_size, n_voxels=n_voxels, obstacle_img=obstacle_img, ax=ax)
        plt2.plot_prediction_world(img=starts_ends_img, world_size=world_size, ax=ax)
        save_fig(image_dir + 'w{}_l{}_start_end_img_combined'.format(iw, i_link), fig=fig, save=save)


# for o in range(10):
#     plot_world_paths(directory='2D/2dof', iw=np.random.randint(5000), i_link=None)


# for o in range(10):
#     plot_world_paths(directory='2D/FB/2dof', iw=np.random.randint(5000), i_link=4)
#
#
# for o in range(8):
#     plot_world_paths(directory='2D/FB/2dof', iw=0, i_link=o)'

for i in range(100, 5000):
    print(i)
    plot_world_paths(directory='2D/SR/2dof', iw=i, i_link=None, n_channels=None)

# for o in range(9):
#     plot_world_paths(directory='2D/FB/3dof', iw=33, i_link=o)
