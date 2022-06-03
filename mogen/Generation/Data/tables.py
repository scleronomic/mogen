import numpy as np
from wzk import sql2, dtypes


class COL:
    __slots__ = ('name',
                 'type_sql',
                 'type_np',
                 'shape')

    def __init__(self, name, type_sql, type_np, shape):
        self.name = name
        self.type_sql = type_sql
        self.type_np = type_np
        self.shape = shape

    def __call__(self):
        return self.name

    def __repr__(self):
        return f"SQL Column ({self.name} | {self.type_sql} | {self.type_np} | {self.shape})"


class SQLTABLE:
    __slots__ = ('table',
                 'cols')

    def __call__(self):
        return self.table

    def __getitem__(self, item):
        self.cols.__getitem__(item)

    def __len__(self):
        return len(self.cols)

    def __iter__(self):
        return self.cols.__iter__()

    def names(self):
        return [c.name for c in self.cols]

    def types_sql(self):
        return [c.type_sql for c in self.cols]

    def types_dict_sql(self):
        return {c.name: c.type_sql for c in self.cols}

    def types_np(self):
        return [c.type_np for c in self.cols]


C_WORLD_I = COL(name='world_i32', type_sql=sql2.TYPE_INTEGER, type_np=np.int32, shape=None)
C_SAMPLE_I = COL(name='sample_i32', type_sql=sql2.TYPE_INTEGER, type_np=np.int32, shape=None)
C_Q_F = COL(name='q_f32', type_sql=sql2.TYPE_BLOB, type_np=np.float32, shape=(20, 3))  # TODO
C_OBJECTIVE_F = COL(name='objective_f32', type_sql=sql2.TYPE_NUMERIC, type_np=np.float32, shape=None)
C_FEASIBLE_I = COL(name='feasible_b', type_sql=sql2.TYPE_INTEGER, type_np=np.float32, shape=None)


class SQL_WORLDS(SQLTABLE):
    table = 'worlds'
    C_WORLD_I = C_WORLD_I
    C_IMG_CMP = COL(name='img_cmp', type_sql=sql2.TYPE_BLOB, type_np=bool, shape=None)

    cols = [C_WORLD_I, C_IMG_CMP]


class SQL_PATHS(SQLTABLE):
    table = 'paths'
    C_WORLD_I = C_WORLD_I
    C_SAMPLE_I = C_SAMPLE_I
    C_Q_F = C_Q_F
    C_OBJECTIVE_F = C_OBJECTIVE_F
    C_FEASIBLE_I = C_FEASIBLE_I

    cols = [C_WORLD_I, C_SAMPLE_I, C_Q_F, C_OBJECTIVE_F, C_FEASIBLE_I]


class SQL_IKS(SQLTABLE):
    table = 'iks'
    C_WORLD_I = C_WORLD_I
    C_SAMPLE_I = C_SAMPLE_I
    C_Q_F = C_Q_F
    C_FRAME_F = COL(name='frame_f', type_sql=sql2.TYPE_BLOB, type_np=np.float32, shape=(4, 4))
    # C_FRAME_F = COL(name='frame_f32', type_sql=sql2.TYPE_BLOB, type_np=np.float32, shape=(4, 4))
    C_OBJECTIVE_F = C_OBJECTIVE_F
    C_FEASIBLE_I = C_FEASIBLE_I

    cols = [C_WORLD_I, C_SAMPLE_I, C_Q_F, C_FRAME_F, C_OBJECTIVE_F, C_FEASIBLE_I]


class SQL_INFO(SQLTABLE):
    table = 'info'
    C_ROBOT_ID = COL(name='robot_id_txt', type_sql=sql2.TYPE_TEXT, type_np=str, shape=None)
    C_N_WORLDS = COL(name='worlds_i', type_sql=sql2.TYPE_INTEGER, type_np=str, shape=None)
    C_N_SAMPLES = COL(name='samples_i', type_sql=sql2.TYPE_INTEGER, type_np=str, shape=None)
    C_N_SAMPLES_PER_WORLD = COL(name='samples_per_world_i', type_sql=sql2.TYPE_INTEGER, type_np=str, shape=None)

    cols = [C_ROBOT_ID, C_N_WORLDS, C_N_SAMPLES, C_N_SAMPLES_PER_WORLD]


n_samples_per_world = 1000  # TODO get rid of this, its just unnecessary to adhere to this restriction

T_WORLDS = SQL_WORLDS()
T_PATHS = SQL_PATHS()
T_IKS = SQL_IKS()
T_INFO = SQL_INFO()


if __name__ == '__main__':
    paths = SQL_PATHS()
    paths2 = SQL_PATHS()
    for cc in paths:
        print(cc)

    # it's all the same, be aware of mutations
    paths.C_Q_F.name = 'why'
    print(paths.C_Q_F)
    print(paths2.C_Q_F)
    print(C_Q_F)
