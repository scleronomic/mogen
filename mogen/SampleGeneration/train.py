import numpy as np

from wzk.trajectory import inner2full
from wzk.gd.Optimizer import Naive
from wzk.image import compressed2img

from rokin.Robots import StaticArm, Justin19
from mopla import parameter
from mopla.Optimizer import InitialGuess, feasibility_check, gradient_descent

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql, get_values_sql
from mogen.SampleGeneration.sample_start_end import sample_q_start_end


class Generation:
    __slots__ = ('par',
                 'gd',
                 'bee_rate',
                 'n_multi_start')


db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/StaticArm04.db'

worlds = get_values_sql(file=db_file, table='worlds', columns='img_cmp', values_only=True)
worlds = np.array([compressed2img(img_cmp=img_cmp, n_voxels=(64, 64)) for img_cmp in worlds[100:]])
q0, q, o, f = get_values_sql(file=db_file, table='paths',
                             rows=np.arange(10000),
                             columns=['q0', 'q', 'objective', 'feasible'], values_only=True)
