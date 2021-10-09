import numpy as np
from wzk import tic, toc

from mogen.Loading.load_pandas import create_path_df
from mogen.Loading.load_sql import df2sql

fileA = f'/volume/USERSTORE/tenh_jo/A.db'
fileB = '/rmc-lx0062/home_local/tenh_jo/B.db'


n = 10000
i_world = np.arange(n)
i_sample = np.arange(n)
f = np.ones(n,dtype=bool)
o = np.random.random(n)
q = np.random.random((n, 20, 10))
q0 = np.random.random((n, 20, 10))
df = create_path_df(i_world=i_world, i_sample=i_sample, q0=q0, q=q, objective=o, feasible=f)

print(n)
tic()
df2sql(df=df, file=fileA, table='paths')
toc('userstore')

tic()
df2sql(df=df, file=fileB, table='paths')
toc('home_local')
