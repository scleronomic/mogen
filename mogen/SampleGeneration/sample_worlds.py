import numpy as np
from mogen.Loading.load_pandas import create_world_df, initialize_df

from wzk.image import img2compressed

from rokin.sample_configurations import sample_q
from mopla.GridWorld import create_perlin_image, create_rectangles
from mopla.Optimizer import feasibility_check
from mopla import parameter


def sample_worlds(robot, n, n_voxels, mode='perlin', **kwargs):

    img_list = []
    if mode == 'perlin':
        while len(img_list) < n:
            img = create_perlin_image(n_voxels=n_voxels, **kwargs)

            par = parameter.Parameter(robot=robot, obstacle_img=img)

            try:
                sample_q(robot, shape=10, feasibility_check=lambda q: feasibility_check(q=q, par=par))
                img_list.append(img)

            except RuntimeError:
                pass

    else:
        raise NotImplementedError

    img_list = img2compressed(img=np.array(img_list, dtype=bool), n_dim=len(n_voxels))
    world_df = create_world_df(i_world=np.arange(n).tolist(), img_cmp=img_list)
    return world_df


def test():
    from mogen.Loading.load_sql import df2sql
    from rokin.Robots import StaticArm
    robot = StaticArm(n_dof=4)
    par = parameter.Parameter(robot=robot, obstacle_img=None)
    df = sample_worlds(robot=robot, n=10000, n_voxels=(64, 64))

    df2sql(df=df, file=robot.id, table_name='worlds')

