import numpy as np
from wzk import img2compressed

import Optimizer.path as path
import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import GridWorld.swept_volume as sv

import Kinematic.forward as forward
import parameter


# parameter.initialize_par(robot_id='Stat_Arm_04')
#
#
# def generate_samples(n_samples, directory, lock=None, verbose=1):
#
#     q =
#     x_spheres =
#     # Create DataFrames
#     # world_df_new = ld.create_world_dataframe_path_img(world_size=world_size, n_voxels=n_voxels, r_sphere=r_sphere)
#     path_df_new = ld.initialize_dataframe()
#
#     for o in range(n_samples):
#
#
#         pose_img = sv.sphere2grid_whole(x=x_warm[o, 1:], r_sphere=r_sphere)
#         pose_img_cmp = img2compressed(img=pose_img)
#
#         path_df_new = path_df_new.append(
#             ld.create_path_dataframe_path_img(i_world=0, i_sample=o,
#                                               x_path=xa[o], x_warm=path.x2x_flat(x_warm[o, :, 0, :]),
#                                               path_img_cmp=pose_img_cmp))
#
#     # Save to SQL
#     ld_sql.df2sql(df=path_df_new, file=file, if_exists='append', lock=lock)
#
#     # toc()
#
#
# generate_samples(n_samples=10, directory='PoseImg/3D/FB/TEST')
# df = ld_sql.get_values_sql(file='PoseImg/3D/FB/TEST')
