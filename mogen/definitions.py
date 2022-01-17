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
OBSTACLE_IMG_CMP = 'obstacle_img_cmp'
EDT_IMG_CMP = 'edt_img_cmp'
OBSTACLE_IMG_LATENT = 'obstacle_img_latent'

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
