import numpy as np
from wzk import tic, toc

from mogen.loading.load import create_path_df
from wzk.sql2 import df2sql

fileA = '/volume/USERSTORE/tenh_jo/A.db'
fileB = '/net/rmc-lx0062/home_local/tenh_jo/B.db'
fileC = '/home_local/tenh_jo/C.db'


n = 1000000
i_world = np.arange(n)
i_sample = np.arange(n)
f = np.ones(n,dtype=bool)
o = np.random.random(n)
q = np.random.random((n, 20, 10))
q0 = np.random.random((n, 20, 10))
df = create_path_df(i_world=i_world, i_sample=i_sample, q0=q0, q=q, objective=o, feasible=f)


if_exists = 'replace'
print(n)
# tic()
# df2sql(df=df, file=fileA, table='paths', if_exists=if_exists)
# toc('userstore')

tic()
df2sql(df=df, file=fileB, table='paths', if_exists=if_exists)
toc('home_local')

# tic()
# df2sql(df=df, file=fileC, table='paths', if_exists=if_exists)
# toc('home_local')
