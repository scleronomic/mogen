import numpy as np
from tensorflow.keras.utils import Sequence

from Util.Loading.load import arg_wrapper__i_world, sort_sample_indices, get_i_world
from wzk import normalize_11, denormalize_11, remove_nones, atleast_list, image as img_basic

import Optimizer.path as path
import Util.Loading.load_sql as load_sql
import GridWorld.obstacle_distance as obst_dist
from definitions import *


# The number of paths per world is fixed to 1000 right now, not sure if that is good or not,
# Might be better if each world is only used for one path -> more random/general settings for the problem
# For the Measurements structure right now it is easier to handle fixed amounts of paths per world, and o think I keep it that
# way, it allows for statistics based on different worlds. There is one category more to order the problem. And if
# the problem is complex it sounds like a good idea to use this, rather than go a step further
# -> For the Measurements structure, 1 path per world would make the separation between world and path files obsolete
#    and each sample should incorporate the whole information


# Alternatives to SQL
# The samples are split in different directories (2 million files in one directory is a bad idea
# -> https://askubuntu.com/questions/584315/placing-many-10-million-files-in-one-directory )
# Till sample set 010203 there are exactly 1000 paths per world, so the separation is also between the different
# worlds, but that doesn't have to be true in general.
# In each sample file the real world number is saved, which is more reliable.
# -> Use SQL


class DataGeneratorLoad(Sequence):
    """
    Generates Measurements for Model
    """

    def __init__(self, *,
                 file, world_img_dict=None,
                 sample_idx,
                 par,
                 return_sample_idx=False):

        self.file = file

        self.net_type = par.net_type

        self.shuffle = par.shuffle
        self.batch_size = par.batch_size
        self.normalize_inputs = par.normalize_inputs
        self.use_one_world_per_batch = par.use_one_world_per_batch
        if self.use_one_world_per_batch:
            assert n_samples_per_world % self.batch_size == 0
        self.n_dim = par.robot.n_dim
        self.n_dof = par.robot.n_dof
        self.robot_limits = par.robot.limits
        self.infinity_joints = par.robot.infinity_joints
        self.n_voxels = par.n_voxels

        self.active_spheres_start_end = par.active_spheres_start_end
        self.active_spheres_path = par.active_spheres_path

        self.world_img_dict = world_img_dict
        self.img_dtype = bool  # TODO Only breaks if you want to load edt/latent path/world images from db
        self.cast2float32 = True
        self.__use_q_start_end = par.use_q_start_end
        self.use_z_decision = par.use_z_decision
        self.use_start_end_symmetry = par.use_start_end_symmetry
        self.combine_swept_volume_path = par.combine_swept_volume_path
        self.combine_swept_volume_start_end = par.combine_swept_volume_start_end

        self.return_sample_idx = return_sample_idx

        # Individual channel for each sphere
        self.n_channels_1img_saved = len(self.active_spheres_path)
        self.n_channels_1img = np.sum(self.active_spheres_path)

        # TODO more separate keywords
        # use q_start
        # use start_img

        self.sample_idx = sample_idx
        # Get ordered indices
        self.sample_indices0 = sort_sample_indices(self.sample_idx)

        self.curriculum = par.curriculum
        if self.curriculum is None:
            self.curriculum_fcn = None
        else:
            self.ignore_prc = self.curriculum[0]
            self.keep_prc = self.curriculum[1]
            self.curriculum_fcn = create_curriculum(directory=self.file, sample_indices=self.sample_idx,
                                                    n_units_pw=10, use_one_world_per_batch=self.use_one_world_per_batch)

        self.n_samples0 = self.sample_indices0.size
        self.n_worlds0 = self.n_samples0 // n_samples_per_world
        self.n_batches0 = self.n_samples0 // self.batch_size
        self.n_batches = self.n_batches0
        self.epoch = -1

    def on_epoch_end(self):
        self.epoch += 1

        # Update the sample indices for the next epoch
        if self.curriculum is not None:
            ig_prc, kp_prc = get_save_ignore_keep_prc(ignore_prc=self.ignore_prc,
                                                      keep_prc=self.keep_prc, epoch=self.epoch)

            self.sample_idx = self.curriculum_fcn(ignore_prc=ig_prc, keep_prc=kp_prc)

        if self.shuffle:
            if self.use_one_world_per_batch:

                if self.curriculum is None:
                    self.sample_idx = self.sample_indices0.copy()

                for w in range(self.n_worlds0):
                    # TODO ???
                    # Check for -1 just necessary for curriculum
                    idx_not_ignored = self.sample_idx[w, self.sample_idx[w, :] != -1]
                    n_not_ignored = np.size(idx_not_ignored)
                    idx_not_ignored = idx_not_ignored[np.random.permutation(n_not_ignored)]
                    self.sample_idx[w, :n_not_ignored] = idx_not_ignored
                    self.sample_idx[w, n_not_ignored:] = -1

                self.sample_idx = self.sample_idx.reshape((self.n_batches0, self.batch_size))
                self.sample_idx = self.sample_idx[np.random.permutation(self.n_batches0), :]
                self.sample_idx = self.sample_idx.flatten()

            else:
                np.random.shuffle(self.sample_idx)
                # Check for -1 just necessary for curriculum
                self.sample_idx = self.sample_idx[self.sample_idx != -1]

        self.n_batches = len(self.sample_idx) // self.batch_size

        assert self.n_batches * self.batch_size == len(self.sample_idx), str(len(self.sample_idx))  # FIXME

    def __get_sib(self, batch_idx):
        # Get list of sample indices for batch
        sample_indices_batch = self.sample_idx[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        # SQLite returns the results of the query in ascending order, to make sure all indices match it's
        # necessary to sort the batch indices
        sample_indices_batch.sort()
        return sample_indices_batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.n_batches

    def __getitem__(self, batch_idx):
        return self.get_samples(sample_idx=self.__get_sib(batch_idx=batch_idx))

    def get_samples(self, sample_idx):

        # Img 2 Q
        if self.net_type == W_IMG2P_Q:
            x, y = self.__dg_world_img2x(sib=sample_idx)

        elif self.net_type == W_IMG2V:
            x, y = self.__dg_world_img2v(sib=sample_idx)

        elif self.net_type == W_IMG2P_QEQ:
            x, y = self.__dg_world_img2x_eq(sib=sample_idx)

        elif self.net_type == W_IMG2P_IMG2P_Q:
            x, y = self.__dg_world_img2path_img2q(sib=sample_idx)
        elif self.net_type == P_IMG2Q:
            x, y = self.__dg_path_img2x(sib=sample_idx)

        elif self.net_type == W_IMG_P_IMG2P_Q:
            x, y = self.__dg_world_path_img2x(sib=sample_idx)
        elif self.net_type == W_IMG2Z:
            x, y = self.__dg_world_img2z(sib=sample_idx)
        elif self.net_type == W_IMG2P_Q_reinf:
            x, y = self.__dg_world_img2x_reinf(sib=sample_idx)

        # Img 2 Img
        elif self.net_type == W_IMG2P_IMG:
            x, y = self.__dg_world_img2path_img(sib=sample_idx)

        # Test forward fun
        elif self.net_type == P_Q2IMG:
            x, y = self.__dg_x2path_img(sib=sample_idx)
        elif self.net_type == Q2X_SPHERES:
            x, y = self.__dg_q2x_spheres(sib=sample_idx)

        # VAE
        elif self.net_type == W_IMG2W_IMG:
            x, y = self.__dg_vae_world_img2world_img(sib=sample_idx)
        elif self.net_type == P_IMG2P_IMG:
            x, y = self.__dg_vae_path_img2path_img(sib=sample_idx)
        elif self.net_type == P_Q2P_Q:
            x, y = self.__dg_vae_q2q(sib=sample_idx)

        else:
            raise ValueError(f"No correct sample_type given: {self.net_type}")

        if self.cast2float32:
            x, y = atleast_list(x, y)
            for i, xx in enumerate(x):
                x[i] = xx.astype(np.float32)

            for i, yy in enumerate(y):
                y[i] = yy.astype(np.float32)

        if self.return_sample_idx:
            return sample_idx, (x, y)
        else:
            return x, y

    def __get_path(self, q_path):
        q_path = path.x_flat2x(x_flat=q_path, n_dof=self.n_dof)

        if self.normalize_inputs:
            q_path = normalize_11(x=q_path, low=self.robot_limits[:, 0], high=self.robot_limits[:, 1])

        if self.__use_q_start_end:
            q_start = q_path[:, :1, :]
            q_end = q_path[:, -1:, :]
        else:
            q_start = q_end = None

        q_path = path.x2x_inner(x=q_path)

        return q_start, q_end, q_path

    def __get_path_step(self, q_path):
        q_path = denormalize_11(x=q_path, low=self.robot_limits[:, 0], high=self.robot_limits[:, 1])

        batch_size, n_waypoints, n_dof = q_path.shape
        i_current = np.random.randint(low=0, high=n_waypoints - 1, size=batch_size)
        q_current = q_path[np.arange(batch_size), i_current, :]
        q_next = q_path[np.arange(batch_size), i_current + 1, :]

        q_step = q_next - q_current
        q_step = path.inf_joint_wrapper(x=q_step, inf_bool=self.infinity_joints)

        q_step_norm = np.linalg.norm(q_step, axis=-1, keepdims=True)
        q_step[q_step_norm.ravel() != 0] /= q_step_norm[q_step_norm.ravel() != 0]
        # TODO this should not be necessary, such paths should not exist but they do

        q_current = normalize_11(x=q_current, low=self.robot_limits[:, 0], high=self.robot_limits[:, 1])
        return q_current, q_step

    def __get_obstacle_img(self, sib):
        i_worlds = get_i_world(i_sample_global=sib)
        return np.array(list(map(lambda iw: self.world_img_dict[iw], i_worlds)))

    def __decompress_img(self, img_cmp, n_channels=None):
        if n_channels is None:
            n_channels = self.n_channels_1img
        return img_basic.compressed2img(img_cmp=img_cmp, n_voxels=self.n_voxels, n_dim=self.n_dim,
                                        n_channels=n_channels, dtype=self.img_dtype)

    def __start_end_img_symmetry(self, start_img, end_img):
        if self.use_start_end_symmetry:
            start_img = np.logical_or(start_img, end_img)
            end_img = None
        return start_img, end_img

    def __use_q_start_end_path(self, x, q_start=None, q_end=None, q_path=None):
        if not isinstance(x, list):
            x = [x]

        if self.use_z_decision:
            x += [q_start, q_end]

        if self.use_z_decision:
            x += [q_path]

        # if len(x) == 1:
        #     x = x[0]

        return x

    def __concatenate_images(self, img_list):

        img_list = remove_nones(img_list)

        if len(img_list) == 1:
            return img_list

        n_channels = [img.shape[-1] for img in img_list]
        n_channels_cs = np.cumsum([0] + n_channels)
        img_ip = img_basic.initialize_image_array(n_samples=self.batch_size, n_voxels=self.n_voxels,
                                                  n_dim=self.n_dim, n_channels=sum(n_channels))

        for idx, img in enumerate(img_list):
            img_ip[..., n_channels_cs[idx]:n_channels_cs[idx + 1]] = img

        return img_ip

    def __get_image_ip(self, obstacle_img=None, start_img_cmp=None, end_img_cmp=None, path_img_cmp=None):

        start_img = end_img = path_img = None

        if start_img_cmp is not None:
            start_img = self.__decompress_img(img_cmp=start_img_cmp)[..., self.active_spheres_start_end]
            if self.combine_swept_volume_start_end:
                start_img = start_img.sum(axis=-1, keepdims=True) > 0

        if end_img_cmp is not None:
            end_img = self.__decompress_img(img_cmp=end_img_cmp)[..., self.active_spheres_start_end]
            if self.combine_swept_volume_start_end:
                end_img = end_img.sum(axis=-1, keepdims=True) > 0

        if path_img_cmp is not None:
            path_img = self.__decompress_img(img_cmp=end_img_cmp)[..., self.active_spheres_path]
            if self.combine_swept_volume_path:
                path_img = path_img.sum(axis=-1, keepdims=True) > 0

        start_img, end_img = self.use_start_end_symmetry(start_img=start_img, end_img=end_img)

        x = self.__concatenate_images(img_list=[obstacle_img, start_img, end_img, path_img])

        return [x]

    # Functions for each model type
    def __dg_vae_q2q(self, sib):
        q_path = load_sql.get_values_sql(file=self.file, rows=sib, values_only=True, columns=PATH_Q)

        _, _, q_path = self.__get_path(q_path=q_path)
        return [q_path], [q_path]

    def __dg_vae_path_img2path_img(self, sib):
        path_img = load_sql.get_values_sql(file=self.file, rows=sib, values_only=True, squeeze_row=False,
                                           columns=[PATH_IMG_CMP])

        path_img = self.__get_image_ip(path_img_cmp=path_img)
        return [path_img], [path_img]

    def __dg_vae_world_img2world_img(self, sib):
        obstacle_img_cmp = load_sql.get_values_sql(file=self.file, rows=sib, values_only=True,
                                               columns=obstacle_img_CMP, squeeze_row=False)

        obstacle_img = self.__decompress_img(img_cmp=obstacle_img_cmp, n_channels=1)
        return [obstacle_img], [obstacle_img]

    def __dg_path_img2x(self, sib):
        start_img_cmp, end_img_cmp, path_img_cmp, q_path = \
            load_sql.get_values_sql(file=self.file, rows=sib, values_only=True,
                                    columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP, PATH_Q])

        q_start, q_end, q_path = self.__get_path(q_path=q_path)
        x = self.__get_image_ip(start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp, path_img_cmp=path_img_cmp)
        x = self.__use_q_start_end_path(x=x, q_end=q_end, q_start=q_start, q_path=q_path)

        return x, q_path

    def __dg_world_img2x(self, sib):

        start_img_cmp, end_img_cmp, q_path = \
            load_sql.get_values_sql(file=self.file, rows=sib, squeeze_row=False, values_only=True,
                                    columns=[START_IMG_CMP, END_IMG_CMP, PATH_Q])

        obstacle_img = self.__get_obstacle_img(sib=sib)
        x = self.__get_image_ip(obstacle_img=obstacle_img, start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp)

        q_start, q_end, q_path = self.__get_path(q_path=q_path)
        x = self.__use_q_start_end_path(x=x, q_end=q_end, q_start=q_start, q_path=q_path)

        return x, q_path

    def __dg_world_img2v(self, sib):
        q_path = load_sql.get_values_sql(file=self.file, rows=sib, squeeze_row=False, values_only=True,
                                         columns=[PATH_Q])

        q_start, q_end, q_path = self.__get_path(q_path=q_path)
        q_current, q_step = self.__get_path_step(q_path)

        obstacle = self.__get_obstacle_img(sib=sib)

        ip = [obstacle, q_current, q_end[:, 0, :]]

        return ip, q_step

    def __dg_world_img2x_eq(self, sib):

        q_path = load_sql.get_values_sql(file=self.file, rows=sib, squeeze_row=False, values_only=True,
                                         columns=[PATH_QEQ])

        q_start, q_end, q_path = self.__get_path(q_path=q_path)

        zero_entry_idx = np.argwhere(q_path == -1)
        q_path[zero_entry_idx] = 0
        _, bi = np.unique(zero_entry_idx[:, 0], return_idx=True)
        n_steps = zero_entry_idx[bi, 1] - 1
        q_start = q_path[:, 0, :]
        q_goal = q_path[np.arange(self.batch_size), n_steps, :]

        # Get obstacle images
        obstacle = self.__get_obstacle_img(sib=sib)
        ip = [obstacle, q_start, q_goal]

        return ip, q_path

    def __dg_world_img2path_img2q(self, sib):

        # Is it necessary to open a new connection every iteration?
        # Not better: https://stackoverflow.com/questions/46913748/keras-deal-with-threads-and-large-datasets
        start_img_cmp, end_img_cmp, path_img_cmp, q_path = \
            load_sql.get_values_sql(file=self.file, rows=sib, squeeze_row=False, values_only=True,
                                    columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP, PATH_Q])

        obstacle_img = self.__get_obstacle_img(sib=sib)
        x = self.__get_image_ip(obstacle_img=obstacle_img, start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp)
        img_path = self.__get_image_ip(path_img_cmp=path_img_cmp)

        q_start, q_end, q_path = self.__get_path(q_path=q_path)
        x = self.__use_q_start_end_path(x=x, q_end=q_end, q_start=q_start, q_path=q_path)

        # path_img = np.round(path_img + 0.1)
        return x, [img_path, q_path]

    def __dg_world_path_img2x(self, sib):

        # Load relevant Measurements from sql
        start_img_cmp, end_img_cmp, path_img_cmp, q_path = \
            load_sql.get_values_sql(file=self.file, rows=sib, values_only=True, squeeze_row=False,
                                    columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP, PATH_Q])

        obstacle_img = self.__get_obstacle_img(sib=sib)
        x = self.__get_image_ip(obstacle_img=obstacle_img,
                                start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp, path_img_cmp=path_img_cmp)

        q_start, q_end, q_path = self.__get_path(q_path=q_path)
        x = self.__use_q_start_end_path(x=x, q_end=q_end, q_start=q_start, q_path=q_path)

        return x, q_path

    def __dg_world_img2z(self, sib):  # TODO Combine z and x as representation in one function

        start_img_cmp, end_img_cmp, z_latent = \
            load_sql.get_values_sql(file=self.file, rows=sib, values_only=True, squeeze_row=False,
                                    columns=[START_IMG_CMP, END_IMG_CMP, Z_LATENT])

        obstacle_img = self.__get_obstacle_img(sib=sib)
        x = self.__get_image_ip(obstacle_img=obstacle_img, start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp)

        return x, z_latent

    def __dg_world_img2x_reinf(self, sib):

        # Load relevant Measurements from sql
        q_path = load_sql.get_values_sql(file=self.file, rows=sib, columns=[PATH_Q], values_only=True)
        q_start, q_end, q_path = self.__get_path(q_path=q_path)

        # Get obstacle images
        i_worlds = get_i_world(i_sample_global=sib)
        obst_cost_fun = np.array(list(map(lambda iw: self.world_img_dict[iw][1], i_worlds)))  # for reinforcement
        obst_cost_fun_grad = np.array(list(map(lambda iw: self.world_img_dict[iw][2], i_worlds)))

        x = [obst_cost_fun, obst_cost_fun_grad, q_start, q_end, q_path]

        return x, None

    def __dg_world_img2path_img(self, sib):

        start_img_cmp, end_img_cmp, path_img_cmp = \
            load_sql.get_values_sql(file=self.file, rows=sib, values_only=True,
                                    columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])

        obstacle_img = self.__get_obstacle_img(sib=sib)
        x = self.__get_image_ip(obstacle_img=obstacle_img,
                                start_img_cmp=start_img_cmp, end_img_cmp=end_img_cmp, path_img_cmp=path_img_cmp)
        path_img = self.__get_image_ip(path_img_cmp=path_img_cmp)

        return x, path_img

    def __dg_x2path_img(self, sib):

        q_path, pose_img_cmp = load_sql.get_values_sql(file=self.file, rows=sib, values_only=True,
                                                       columns=[PATH_Q, PATH_IMG_CMP])

        q_start, _, _ = self.__get_path(q_path=q_path)

        pose_img = self.__decompress_img(img_cmp=pose_img_cmp)

        if q_path.shape[1] == 1:
            q_path = q_path[..., 0, :]  # pose
        return q_path, pose_img

    def __dg_q2x_spheres(self, sib):  # Forward ForwardKinematic

        # Load relevant Measurements from sql
        q, x_spheres = load_sql.get_values_sql(file=self.file, rows=sib, values_only=True,
                                               columns=[PATH_Q, X_WARM])

        x_spheres = x_spheres.reshape((self.batch_size, -1, self.n_spheres, self.n_dim))

        _, _, q = self.__get_path(q_path=q)

        if self.normalize_inputs:
            x_spheres = normalize_11(x=x_spheres, low=self.robot_limits[:self.n_dim, 0],
                                     high=self.robot_limits[:self.n_dim, 1])

        if q.shape[1] == 1:
            q = q[..., 0, :]  # pose

        return q, x_spheres


def get_ordered_objective(sample_indices, file):
    n_worlds = sample_indices.size // n_samples_per_world
    sample_indices0 = sort_sample_indices(sample_indices=sample_indices).flatten()
    objective0 = load_sql.get_values_sql(file=file, columns='objective', rows=sample_indices0, values_only=True)
    objective0 = objective0.reshape((n_worlds, n_samples_per_world))

    return objective0


def get_save_ignore_keep_prc(ignore_prc, keep_prc, epoch):
    try:
        i = ignore_prc[epoch]
    except IndexError:
        i = ignore_prc[-1]

    try:
        k = keep_prc[epoch]
    except IndexError:
        k = keep_prc[-1]

    return i, k


def create_curriculum(directory, sample_indices, n_units_pw=10,
                      use_one_world_per_batch=False):
    # Ensure the worlds are ordered to begin with
    n_samples = np.size(sample_indices)
    n_worlds = n_samples // n_samples_per_world

    sample_indices0 = sort_sample_indices(sample_indices=sample_indices)
    objective0 = get_ordered_objective(sample_indices=sample_indices, file=directory)

    if use_one_world_per_batch:
        for i in range(n_worlds):
            obj_sorted_idx0 = np.argsort(objective0[i])
            objective0[i, :] = objective0[i, obj_sorted_idx0]
            sample_indices0[i, :] = sample_indices0[i, obj_sorted_idx0]

        # Learning unit
        l_unit_size = n_samples_per_world // n_units_pw
        l_unit = np.zeros((n_worlds, n_units_pw))
        l_unit_idx = np.zeros((n_worlds, n_units_pw, l_unit_size))
        for w in range(n_worlds):
            for u in range(n_units_pw):
                l_unit[w, u] = objective0[w, u * l_unit_size:(u + 1) * l_unit_size].mean()
                l_unit_idx[w, u, :] = sample_indices0[w, u * l_unit_size:(u + 1) * l_unit_size]

        l_unit_sorted_idx = np.argsort(l_unit.flatten())

        n_units = n_worlds * n_units_pw
        # Ordered list (of lists), starting from the easiest

        curriculum_idx_look_up = l_unit_idx.reshape((n_units, l_unit_size))[l_unit_sorted_idx]

    else:
        obj_sorted_idx0 = np.argsort(objective0.flatten())
        # Ordered list starting from the easiest
        curriculum_idx_look_up = sample_indices0.flatten()[obj_sorted_idx0]
        n_units = n_samples  # Here effective every sample is a single learning unit

    def curriculum(ignore_prc, keep_prc=.1):

        res = curriculum_idx_look_up.copy()

        # Ignore the easiest 'ignore_prc' percent
        n_ignore = int(n_units * ignore_prc)
        # While keeping 'keep_prc' percent of these cases
        n_keep = int(n_ignore * keep_prc)

        if n_ignore > 0 and n_ignore - n_keep > 0:
            idx_ignore = np.random.choice(np.arange(n_ignore), size=n_ignore - n_keep, replace=False)
            res[idx_ignore, ...] = -1

        if use_one_world_per_batch:
            res = res.reshape((n_worlds, n_samples_per_world))

        return res.astype(int)

    # del sample_indices0, objective0, obj_sorted_idx0

    return curriculum


def get_curriculum_schedule(n_full=1, ignore_prc_max=.8, step_size=.05):
    assert n_full > 0
    c = [.0] * n_full
    while c[-1] < ignore_prc_max:
        c.append(c[-1] + step_size)

    # make sure the values are hole percent points
    c = np.array(c)
    c = np.round(c * 100) / 100
    c = np.abs(c)

    return c.tolist()


def get_n_channels(*, obstacle_img=True,
                   start_end_img=True, start_end_symmetry=False, path_img=False, n_spheres=1):
    """
    Figure out how many channels are necessary for an input/output image depending on different options.
    """

    if obstacle_img:
        n_obst = 1
    else:
        n_obst = 0

    if start_end_img:
        if start_end_symmetry:
            n_start_end = 1  # Put start and end configuration in one image to enforce symmetry
        else:
            n_start_end = 2
    else:
        n_start_end = 0

    if path_img:
        n_path = 1
    else:
        n_path = 0

    n_channels = n_obst + (n_start_end + n_path) * n_spheres

    return n_channels


def create_obstacle_dict(directory, i_worlds=-1, obstacle_type=IMG_BOOL,
                         n_voxels=64, n_dim=2):
    from wzk.image import compressed2img
    i_world_list = arg_wrapper__i_world(i_worlds, directory=directory)

    # TODO better naming use obstacle type to access the columns directly
    # TODO smaller grid
    # world_img_type = f"obstacle_img_{smaller_grid}"
    # n_dim = np.shape(world_df.loc[0, 'rectangle_pos'][0])
    # obstacle_img_small = world_df.loc[:, world_img_type].values
    # obstacle_img_small = compressed2img(img_cmp=obstacle_img_small, n_voxels=smaller_grid, n_dim=n_dim, dtype=float)
    # world_df.loc[:, world_img_type] = numeric2object_array(obstacle_img_small)

    file = directory + WORLD_DB
    dtype = IMG_TYPE_TYPE_DICT[obstacle_type]

    if obstacle_type == IMG_BOOL:
        obstacle_img_cmp = load_sql.get_values_sql(file=file, columns=obstacle_img_CMP, values_only=True)
        obst = compressed2img(img_cmp=obstacle_img_cmp, n_voxels=n_voxels, n_dim=n_dim, n_channels=1, dtype=dtype)

    elif obstacle_type == IMG_EDT:
        edt_img_cmp = load_sql.get_values_sql(file=file, columns=EDT_IMG_CMP, values_only=True)
        obst = compressed2img(img_cmp=edt_img_cmp, n_voxels=n_voxels, n_dim=2, n_channels=1, dtype=dtype)

    elif obstacle_type == IMG_LATENT:
        obst = load_sql.get_values_sql(file=file, columns=obstacle_img_LATENT, values_only=True)

    else:
        raise ValueError(f"Unknown obstacle_type {obstacle_type}")

    obst_dict = {}
    for iw in i_world_list:
        obst_dict[iw] = obst[iw]

    # if edt_fun:
    #     obst_dict[iw] = [obst_dict[iw], ]
    #
    # if edt_grad:
    #     obst_dict[iw] = [obst_dict[iw], ]

    return obst_dict


def create_edt_fun_dict(*, directory=None, world_img_dict=None,  # Or
                        voxel_size, lower_left, spheres_rad,
                        eps=0.005, order=1):  # Optionally

    if world_img_dict is None:
        world_img_dict = create_obstacle_dict(i_worlds=-1, directory=directory, obstacle_type='bool_img')

    edt_fun_grad_dict = {}
    for key in world_img_dict.keys():
        edt_fun_grad_dict[key] = obst_dist.obstacle_img2cost_fun_grad(obstacle_img=world_img_dict[key][..., 0],
                                                                      eps=eps, add_boundary=True,
                                                                      voxel_size=voxel_size, r=spheres_rad,
                                                                      interp_order=order, lower_left=lower_left)
    return edt_fun_grad_dict
