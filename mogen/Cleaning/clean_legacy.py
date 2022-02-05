import numpy as np

from wzk import sql2


def f32_to_f64(file: str, column: str):
    table = 'paths'
    assert column.endswith('_f32')
    column_new = column.replace('_f32', '_f64')
    x_f32 = sql2.get_values_sql(file=file, table=table, columns=column, values_only=True)
    x_f64 = np.array(x_f32, dtype=np.float64)
    sql2.rename_columns(file=file, table=table, columns={column: column_new})
    sql2.set_values_sql(file=file, table=table, values=(x_f64,), columns=column_new)


def add_start_end_column(file, n_wp, n_dof):
    table = 'paths'
    q = sql2.get_values_sql(file=file, table=table, columns='q_f64', values_only=True)
    q = q.reshape((-1, n_wp, n_dof))
    q_start = q[:, 0, :]
    q_end = q[:, -1, :]

    sql2.add_column(file=file, table=table, column='q_start_f64', dtype=sql2.TYPE_BLOB)
    sql2.add_column(file=file, table=table, column='q_end_f64', dtype=sql2.TYPE_BLOB)

    sql2.set_values_sql(file=file, table=table, values=(q_start, q_end), columns=['q_start_f64', 'q_end_f64'])


def rename2old(file):
    sql2.rename_columns(file=file, table='paths', columns={'q_f64': 'q_path',
                                                           'q_start_f64': 'q_start',
                                                           'q_end_f64': 'q_end',
                                                           'world_i32': 'i_world',
                                                           'sample_i32': 'i_sample',
                                                           })
    sql2.rename_columns(file=file, table='worlds', columns={'img_cmp': 'obst_img_cmp',
                                                            'world_i32': 'i_world',
                                                            })


def main():
    file = '/Users/jote/Documents/Code/Python/RobotPathData/StaticArm04.db'
    # f32_to_f64(file=file, column='q_f32')
    # add_start_end_column(file=file, n_wp=20, n_dof=4)
    # rename2old(file=file)
    sql2.vacuum(file)


if __name__ == '__main__':
    main()
