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
from torch import nn


class OMPNet(nn.Module):
    def __init__(self, robot, n_fb, n_wp):
        super(OMPNet, self).__init__()

        s = 100
        n_o = n_wp * robot.n_dof
        n_i = n_fb + n_o

        self.base = nn.Sequential(
            nn.Linear(n_i, n_i * 2),
            nn.ReLU(),
            nn.Linear(n_i * 2, n_i * 2),
            nn.ReLU(),
            nn.Linear(n_i * 2, n_i),
        )

        self.head_q = nn.Sequential(
            nn.Linear(s, n_o * 2),
            nn.ReLU(),
            nn.Linear(n_o * 2, n_o * 2),
            nn.ReLU(),
            nn.Linear(n_o * 2, n_o),
        )

        self.head_fo = nn.Sequential(
            nn.Linear(s, n_o // 2),
            nn.ReLU(),
            nn.Linear(n_o // 2, n_o // 2),
            nn.ReLU(),
            nn.Linear(n_o // 2, 2)
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits

db_file = '/volume/USERSTORE/tenh_jo/0_Data/Samples/StaticArm04.db'

worlds = get_values_sql(file=db_file, table='worlds', columns='img_cmp', values_only=True)
worlds = np.array([compressed2img(img_cmp=img_cmp, n_voxels=(64, 64)) for img_cmp in worlds[100:]])
q0, q, o, f = get_values_sql(file=db_file, table='paths',
                             rows=np.arange(10000),
                             columns=['q0', 'q', 'objective', 'feasible'], values_only=True)
