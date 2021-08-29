import os
import platform

#######################################################################################################################
# Metric are always SI, if not stated otherwise
# Second s, Meter m, Kilogram kg

# Notation and Abbreviations
# --- Variable Names
# sm - safety margin
# rp - random point
# cs - cumsum
# iw - world idx
# pw - per world
# wp - way_points
# n_ - number of
# to work with 2D and 3d simultaneously view x and i as collection of xyz and ijk
# xy/xyz -> x instead of differentiating between 2 and 3 dimension use just x and check the dimension by shape / shape
# ij/ijk -> i


# --- Concepts
# world            consists of a number of obstacles and is the environment for multiple paths (motions from A to B)
# path             evolution of configuration over time 'motion problem'
#                  sequence of the intermediate way points between start and end configuration
# way points       the discrete points along the path (via points)
#                      x - c - c - c - c - c - x  |  Start and end are fixed
#                      x - c - c - c - c - c - >  |  End is not fixed but constrained through the head
#                      number of waypoints, include start and endpoint,
#                     most of the time there are only n_wp-2 true variables
#                     X - c - c - X - c - c - X - c - c - X - c - c - X
#                     0           1           2           3           4
#                     1   2   3   4   5   6   7   8   9  10  11  12  13
# limb             body part between two joints (ie. upper arm [between shoulder joint and elbow joint])
#                  consists of one or multiple frames
# frame            matrix describing the position and rotation of a body part

# --- Dimensions of common variables
# joints  / q      ->  n_samples x n_waypoints x n_dof
# joints  / q_ss   ->  n_samples x n_waypoints x n_substeps x n_dof

# spheres / x      ->  n_samples x n_waypoints x n_spheres x n_dim
# spheres / dx_dq  ->  n_samples x n_waypoints x n_spheres x n_dim x n_dof

# frames / f       ->  n_samples x n_waypoints x n_frames x n_dim x n_dim
# frames / df_dq   ->  n_samples x n_waypoints x n_frames x n_dim x n_dim x n_dof

# wheels ->         n_waypoints x n_wheels x n_dof

# --- Comments
# If something is marked as '# Correct', I wondered myself is this passage is correct and checked it thoroughly,
# so it 'should' be correct and does not need to be checked again

# Project directory names
DATA = '/0_Data'
IMAGES = '/Images'
NETS = '/Network_Models'
UTIL = '/Util'
SAMPLES = '/Samples'
POSES = '/Poses'
REAL_SCENES = '/Real_Scenes'
CALIBRATION = '/Calibration'
TMP = '/mopla_tmp'

# USERNAME = os.path.expanduser("~").split(sep='/')[-1]
USERNAME = 'tenh_jo'

# DLR/net
JUSTIN_VISION_TMP = '/net/kauket/home_local/baeuml/tmp'  # directory for Voxel Model Vicon Measurements etc


# Alternative storage places for the samples
DLR_USERSTORE = f"/volume/USERSTORE/{USERNAME}"  # Daily Back-up, relies on connection -> not for large Measurements
DLR_HOMELOCAL = f"/home_local/{USERNAME}"       # No Back-up, but fastest drive -> use for calculation

# Project structure
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_DATA = PROJECT_ROOT + DATA
PROJECT_DATA_NETS = PROJECT_DATA + NETS
PROJECT_DATA_IMAGES = PROJECT_DATA + IMAGES
PROJECT_DATA_SAMPLES = PROJECT_DATA + SAMPLES
PROJECT_DATA_UTIL = PROJECT_DATA + UTIL

DLR_USERSTORE_DATA = DLR_USERSTORE + DATA
DLR_USERSTORE_DATA_NETS = DLR_USERSTORE_DATA + NETS
DLR_USERSTORE_DATA_IMAGES = DLR_USERSTORE_DATA + IMAGES
DLR_USERSTORE_DATA_SAMPLES = DLR_USERSTORE_DATA + SAMPLES
DLR_USERSTORE_DATA_REAL_SCENES = DLR_USERSTORE_DATA + REAL_SCENES
DLR_USERSTORE_DATA_TMP = DLR_USERSTORE_DATA + TMP

DLR_USERSTORE_DATA_CALIBRATION = DLR_USERSTORE_DATA + CALIBRATION
DLR_USERSTORE_PAPER_20CAL = DLR_USERSTORE_DATA + CALIBRATION + '/Results/Paper'

DLR_HOMELOCAL_DATA = DLR_HOMELOCAL + DATA
DLR_HOMELOCAL_DATA_SAMPLES = DLR_HOMELOCAL_DATA + SAMPLES
DLR_HOMELOCAL_TMP = DLR_HOMELOCAL + TMP


# Planner temp files
PLANNER_VIEWER_dir = DLR_USERSTORE + 'TEMP_Planner'
PLANNER_VIEWER_DATA = 'viewer_data.npz'
PLANNER_VIEWER_DATA_CHECK = 'viewer_data_check.txt'


# bullet_lib = ''
# bullet_include = '/usr/local/Cellar/bullet/3.08_2/include/bullet/'
bullet_include = '/volume/USERSTORE/tenh_jo/Software/bullet3/src/'

# Use this locations as standard if working in the office at DLR
PLATFORM_IS_LINUX = platform.system() == 'Linux'

if PLATFORM_IS_LINUX:
    PROJECT_DATA_SAMPLES = DLR_HOMELOCAL_DATA_SAMPLES
    PROJECT_DATA_IMAGES = DLR_USERSTORE_DATA_IMAGES
    PROJECT_DATA_NETS = DLR_USERSTORE_DATA_NETS

else:
    DLR_USERSTORE_PAPER_20CAL = PROJECT_ROOT + '/A_Data/Calibration'
    PLANNER_VIEWER_dir = PROJECT_ROOT + '/A_Data/TEMP_Planner'

#######################################################################################################################
# Names of Measurements files
WORLD_DB = 'world.db'
PATH_DB = 'path.db'

# Column Names
# --- Configurations
START_Q = 'q_start'
END_Q = 'q_end'
PATH_Q = 'q_path'

# --- Images (save always compressed)
START_IMG_CMP = 'start_img_cmp'
END_IMG_CMP = 'end_img_cmp'
PATH_IMG_CMP = 'path_img_cmp'
obstacle_img_CMP = 'obstacle_img_cmp'
EDT_IMG_CMP = 'edt_img_cmp'
obstacle_img_LATENT = 'obstacle_img_latent'

PATH_QEQ = 'x_path_eq'
Z_LATENT = 'z_latent'
X_WARM = 'x_warm'

# --- Image types
IMG_BOOL = 'bool'
IMG_EDT = 'edt'
IMG_LATENT = 'latent'
IMG_FB = 'fixed-basis'

IMG_TYPE_TYPE_DICT = {IMG_BOOL: bool,
                      IMG_EDT: float,
                      IMG_LATENT: float,
                      IMG_FB: float}
