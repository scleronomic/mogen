import numpy as np
from wzk import sql2

from mogen.Generation import data, parameter

T_PATHS = data.T_PATHS
T_WORLDS = data.T_WORLDS


def sort(file):
    print(f'Sort: {file}')

    sql2.sort_table(file=file, table=T_PATHS(),
                    order_by=[T_PATHS.C_WORLD_I(), T_PATHS.C_SAMPLE_I(), 'ROWID'])


def reset_sample_i32(file):
    print(f'Reset indices: {file}')
    iw_all = sql2.get_values_sql(file=file, table=T_PATHS(), rows=-1, columns=[T_PATHS.C_WORLD_I()])
    iw_all = np.squeeze(iw_all)
    n = len(iw_all)
    i_s = np.full(n, -1, dtype=int)

    for iw_i in np.unique(iw_all):
        j = np.nonzero(iw_all == iw_i)[0]
        i_s[j] = np.arange(len(j))

    sql2.set_values_sql(file=file, table=T_PATHS(), values=(i_s,), columns=T_PATHS.C_SAMPLE_I())


def reset_sample_i32_0(file):
    print(f"Reset sample_i32 0: {file}")
    table = T_PATHS()

    print('Load indices')
    w, s = sql2.get_values_sql(file=file, table=table, rows=-1,
                               columns=[T_PATHS.C_WORLD_I(), T_PATHS.C_SAMPLE_I()])
    w = np.squeeze(w).astype(np.int32)
    s = np.squeeze(s).astype(np.int32)
    assert np.all(s == 0)

    print('Update indices')
    w0 = np.nonzero(w == 0)[0]
    b0 = w0[1:] != w0[:-1] + 1
    wb0 = w0[1:][b0]

    for wb0_i in wb0:
        s[wb0_i:] += 1

    print('Set indices')
    sql2.set_values_sql(file=file, table=table, values=(s,), columns=T_PATHS.C_SAMPLE_I())


def remove_infeasible(file):
    f = sql2.get_values_sql(file=file, table=T_PATHS(), rows=-1, columns=[T_PATHS.C_FEASIBLE_I()])
    print(np.unique(f, return_counts=True))
    feasible = f == +1
    print(f"feasible {feasible.sum()}/{feasible.size} ~ {np.round(feasible.mean(), 3)}")
    if not np.all(feasible):
        sql2.delete_rows(file=file, table=T_PATHS(), rows=~f)
        reset_sample_i32(file=file)


def update_cast_joint_errors(q, limits, eps=1e-6):
    below_lower = q < limits[:, 0]
    above_upper = q > limits[:, 1]

    q[below_lower] += eps
    q[above_upper] -= eps
    return q


def set_dtypes(file):
    print(f'set dtypes {file}')
    # paths
    columns_paths_old = sql2.get_columns(file=file, table=T_PATHS())
    columns_paths_new = [T_PATHS.C_WORLD_I(), T_PATHS.C_SAMPLE_I(),
                         T_PATHS.C_Q_F(), T_PATHS.C_OBJECTIVE_F(), T_PATHS.C_FEASIBLE_I()]
    dtypes_paths_new = [sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_REAL, sql2.TYPE_INTEGER]
    assert np.all(columns_paths_old.name.values == columns_paths_new), f"{columns_paths_old.name.values} \n {columns_paths_new}"
    sql2.alter_table(file, table='paths', columns=columns_paths_new, dtypes=dtypes_paths_new)

    # worlds
    columns_worlds_old = sql2.get_columns(file=file, table=data.T_WORLDS())
    columns_worlds_new = [data.T_WORLDS.C_WORLD_I(), data.T_WORLDS.C_IMG_CMP()]
    dtypes_worlds_new = [sql2.TYPE_INTEGER, sql2.TYPE_BLOB]
    assert np.all(columns_worlds_old.name.values == columns_worlds_new)
    sql2.alter_table(file, table='worlds', columns=columns_worlds_new, dtypes=dtypes_worlds_new)


def check_consistency(robot,
                      db_file=None,
                      q=None, f=None, o=None, img=None):

    gen = parameter.init_par(robot_id=robot.id)
    raise NotImplementedError


def add_dist_img_column(file, robot_id):
    from mopla.World.obstacle_distance import img2dist_img
    from wzk import image, limits2cell_size, print_progress, tic, toc

    gen = parameter.init_par(robot_id=robot_id)

    limits = gen.par.world.limits
    shape = gen.par.world.shape
    voxel_size = limits2cell_size(shape=shape, limits=limits)

    dimg = []
    img_cmp = data.sql2.get_values_sql(file=file, rows=np.arange(10000), table=T_WORLDS(), columns=T_WORLDS.C_IMG_CMP())
    
    for i, cimg_i in enumerate(img_cmp):
        print_progress(i=i, n=len(img_cmp))
        bimg_i = image.compressed2img(img_cmp=cimg_i, shape=shape, dtype='bool')
        dimg_i = img2dist_img(img=bimg_i, voxel_size=voxel_size, add_boundary=False)
        cimg_i = image.img2compressed(img=dimg_i)
        dimg.append(cimg_i)

    sql2.add_column(file=file, table=T_WORLDS(), column='dimg_cmp', dtype=sql2.TYPE_BLOB)
    sql2.set_values_sql(file, table=T_WORLDS(), columns='dimg_cmp', values=(dimg,), rows=-1, lock=None)

    # test
    tic()
    dimg_cmp = data.sql2.get_values_sql(file=file, rows=np.arange(10000), table=T_WORLDS(), columns='dimg_cmp')
    dimg = []
    for i, cimg_i in enumerate(dimg_cmp):
        print_progress(i=i, n=len(dimg_cmp))
        dimg_i = image.compressed2img(img_cmp=cimg_i, shape=shape, dtype='float')
        dimg.append(dimg_i)
    toc()





if __name__ == '__main__':

    _robot_id = 'JustinArm07'
    _file = data.get_file(robot_id=_robot_id)
    _file = f"/Users/jote/Documents/DLR/Data/mogen/{_robot_id}/{_robot_id}_worlds0.db"
    # _file = '/home/johannes_tenhumberg_gmail_com/sdb/JustinArm07_hard2.db'
    add_dist_img_column(file=_file, robot_id=_robot_id)

    # set_dtypes(_file)
    # sort(_file)

    # remove_infeasible(file=_file)
    # reset_sample_i32(file=_file)

    # SingleSphere02
    # feasible 1354396/1354713 ~ 1.0 v0
    # feasible 1354713/1354713 ~ 1.0 v1



