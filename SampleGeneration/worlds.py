import numpy as np

import Util.Loading.load_pandas as ld
import Util.Loading.load_sql as ld_sql
import GridWorld.random_obstacles as randrect
from wzk.image import img2compressed
from definitions import *


from wzk import ObjectDict


world = ObjectDict()

world.n_dim = 2
world.limits = np.array([[0, 10],
                         [0, 10]])

world.n_voxels = (64, 64)

random_rectangles = ObjectDict()
random_rectangles.n_max = 30
random_rectangles.limits = np.array([[0, 1]])
random_rectangles.pos = 1
random_rectangles.size = 1
world.random_rectangles = random_rectangles
special_dim = None

"""
### general
n_dim
limits
n_voxels
voxel_size

### safe space
rect_pos
rect_size

### random rectangles
n_rectangles_max
rectangle_sizes
special_dim

# 
rect_pos
rect_size


# safe space
rect_pos
rect_size

# cellular automata
# -> http://www.roguebasin.com/index.php?title=Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels

# img
# perlin / simplex noise
# -> from noise import pnoise, snoise 
#
"""

n_obstacles = np.random.randint(low=0, high=30, size=n_samples)


def aa(n_obstacles, ):
    verbose = 0
    save = False
    n_samples = int(1e1)

    obstacle_img_cmp = np.zeros(n_samples, dtype=object)
    world_df_new = ld.initialize_dataframe()

    obstacle_img, (rect_pos, rect_size) = \
        randrect.create_rectangle_image(n=n_obstacles, size_limits=1, n_voxels=2,
                                        special_dim=special_dim, return_rectangles=True)

    obstacle_img_cmp = img2compressed(img=obstacle_img)

    world_db = pd.DataFrame()
    world_db.loc[:, obstacle_img_CMP] = obstacle_img_cmp

    file = DLR_HOMELOCAL_DATA_SAMPLES + '2D/RandomRectangles/worlds_10M.db'
    ld_sql.df2sql()
    if save:
        with ld_sql.open_db_connection(file=file) as con:
            world_db.to_sql(name='db', con=con, if_exists='replace', index=False)
