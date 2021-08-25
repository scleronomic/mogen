import numpy as np

from mogen.Loading import load_sql as load_sql
from wzk.training import n2train_test, train_test_split


def arg_wrapper__i_world(i_worlds, directory=None):
    """
    Return an numpy array of world indices.
    - if i_worlds is an integer
    Helper function to handle the argument i_world.
    If i_world is an integer, wrap it in a list.
    If i_world is -1, use all available worlds.
    If i_world is already a list, leave it untouched.
    """

    if isinstance(i_worlds, (int, np.int32, np.int64)):
        i_worlds = int(i_worlds)
        if i_worlds == -1:
            n_worlds = load_sql.get_n_rows(file=directory + WORLD_DB)
            i_worlds = list(range(n_worlds))
        else:
            i_worlds = [i_worlds]

    return np.array(i_worlds)


def get_sample_indices(i_worlds=-1, directory=None, validation_split=0.2, validation_split_random=True):
    """
    Calculates the indices for training and testing.
    Splits the indices for whole worlds, so the test set is more difficult/ closer to reality.
    """

    i_world_list = arg_wrapper__i_world(i_worlds, directory=directory)

    # Split the samples for training and validation, separate the two sets by worlds -> test set contains unseen worlds
    if validation_split_random:
        np.random.shuffle(i_world_list)
    sample_indices = np.array([get_i_samples_world(iw, directory=directory) for iw in i_world_list]).flatten()

    # Divide in trainings and test Measurements
    n_worlds = len(i_world_list)
    n_samples = n_worlds * n_samples_per_world

    if validation_split is not None and validation_split > 0.0:
        idx_split = int(validation_split * n_samples)
        idx_split -= idx_split % n_samples_per_world
        sample_indices_training = sample_indices[:n_samples - idx_split]
        sample_indices_test = sample_indices[n_samples - idx_split:]

        return sample_indices_training, sample_indices_test

    else:
        return sample_indices, 0


def sort_sample_indices(sample_indices):
    """
    Make a sorted copy of the the sample indices, sort with respect to the world

    5000, 5001, 5002, ..., 5999
    0000, 0001, 0002, ..., 0999
    3000, 3001, 3002, ..., 3999

    ->

    0000, 0001, 0002, ..., 0999
    1000, 1001, 1002, ..., 1999
    2000, 2001, 2002, ..., 2999
    ...
    """

    n_worlds = sample_indices.size // n_samples_per_world
    sample_indices0 = sample_indices.reshape((n_worlds, n_samples_per_world)).copy()
    idx_worlds0 = get_i_world(sample_indices0[:, 0])
    sample_indices0 = sample_indices0[np.argsort(idx_worlds0), :]

    return sample_indices0


# i_worlds <-> i_samples
def get_i_world(i_sample_global):
    """
    Get the world-indices corresponding to the global samples.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    if isinstance(i_sample_global, (list, tuple)):
        i_sample_global = np.array(i_sample_global)

    return i_sample_global // n_samples_per_world


def get_i_samples_world(i_worlds, directory):
    """
    Get the global sample-indices corresponding to the world numbers.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    i_world_list = arg_wrapper__i_world(i_worlds=i_worlds, directory=directory)

    if np.size(i_world_list) == 1:
        return np.arange(n_samples_per_world * i_world_list[0], n_samples_per_world * (i_world_list[0] + 1))
    else:
        return np.array([get_i_samples_world(iw, directory=directory) for iw in i_world_list])


def get_i_samples_global(i_worlds, i_samples_local):
    """
    Get the global sample-indices corresponding to the world numbers.
    Assumes that the number of samples per world (ie 1000) is correct -> see definitions
    """

    if i_samples_local is None or i_worlds is None:
        return None
    if np.size(i_samples_local) / np.size(i_worlds) % n_samples_per_world == 0:
        i_worlds = np.repeat(i_worlds, n_samples_per_world)

    return i_samples_local + i_worlds * n_samples_per_world


def get_i_samples_local(i_sample_global):
    return i_sample_global % n_samples_per_world
