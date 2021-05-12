from wzk import *

import InverseKinematic.util as inv_util
import Kinematic.frames as fr
from Kinematic.sample_configurations import sample_q, sample_q_frames_mp
import definitions as dfn


# Loading / Generating Data
def generate_pose_samples(*, n_samples, par, valid=True, mode_2d='vector',
                          n_processes=1, save=False):
    """
    it takes  1s to generate 1e4 samples with 10 cores
    it takes 20s to generate 1e6 samples with 10 cores
    """

    q, frames = sample_q_frames_mp(n_processes=n_processes, shape=n_samples, valid=valid, par=par,
                                   frames_idx=par.tcp.frame_idx)

    if save:
        pos, rot = fr.frame2trans_rot(frame=frames, mode='vector')
        np.savez(f"{dfn.DLR_USERSTORE_DATA}Inv_Data/{par.robot.id}/{get_timestamp()}.npz",
                 pos=pos, rot=rot, q=q)

    return q, frames


def load_data(n_samples, directory, verbose=1):
    n_samples_per_file = int(1e6)
    file_name_stump = 'inv_data_1M_{:0>2}.npz'
    assert n_samples % n_samples_per_file == 0

    n_files_to_load = n_samples // n_samples_per_file
    file_list = [file_name_stump.format(i) for i in range(1, n_files_to_load + 1)]
    res = combine_npz_files(directory=directory, file_list=file_list, save=False, verbose=verbose)
    return res['pos_r'], res['quat_r'], res['pos_l'], res['quat_l'], res['q']


def generate_pose_samples_net(*, n_samples, par,
                              return_q_close=False,
                              angles_vectors='angles',
                              n_processes=10):
    q, frames = generate_pose_samples(n_samples=n_samples, n_processes=n_processes, par=par, save=False)
    if return_q_close:
        q_close = sample_q(n_samples=n_samples, robot=par.robot, valid=False)
        x = inv_util.encode_configurations_frames(robot=par.robot, frame_idx=par.tcp.frame_idx, q=q_close,
                                                  frames=frames)
    else:
        x = inv_util.encode_frames(frames=frames)

    y = inv_util.encode_configurations(q=q, robot=par.robot, frame_idx=par.tcp.frame_idx, encoding=angles_vectors)
    return x, y


def __test1__():
    import parameter as para
    n_samples = int(1e6)
    n_processes = 10
    valid = True
    par = para.initialize_par('Stat_Arm_08')
    # par = para.initialize_par('Justin19')
    par.tcp.frame_idx = np.array([4, 5, 6])
    par.tcp.frame_idx = np.array([4, 5, 6])

    tic()
    a, b, c = generate_pose_samples(n_samples=n_samples, par=par, n_processes=n_processes, valid=valid, save=False)
    toc()

    tic()
    q = sample_config.sample_q_mp(n_processes=n_processes, n_samples=n_samples, valid=valid, par=par)
    toc()


def __test2__():
    import Nets.Util.tf_settings as l_util
    l_util.set_cuda_visible_devices(1)

    import tensorflow as tf
    net = tf.keras.models.load_model(dfn.DLR_USERSTORE_DATA + 'InverseKinematic/nets/InvNet_100k_nn0.hdf5')
    # generate_samples_right_distance(n_samples=20, net=net)


