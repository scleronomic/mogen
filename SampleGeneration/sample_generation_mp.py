import multiprocessing as mp
import time

import numpy as np

import SampleGeneration.sample_generation as sg
import SampleGeneration.sample_generation_path_img as sg_path
import SampleGeneration.sample_generation_pose_img as sg_pose
import Util.Loading.load_pandas as ld
import X_Tests.parameter_old as par
import definitions as d

# sample_dir = '3D/SR/3dof'
# sample_dir = '2D/FB/2dof'
# sample_dir = '3D/FB/7dof_Justin'

# sample_dir = 'PoseImg/3D/FB/Justin_0'
# sample_dir = 'PathImg/2D/FB/6dof_direct'


# Evaluation
# sample_dir = '2D/FB/2dof_eval10_TEST'
# sample_dir = '2D/2dof_eval/'
# sample_dir = '3D/3dof_eval10/'

sample_dir = '3D/FB/7dof_eval10'
sample_dir = '2D/SR/2dof_eval10_B'
sample_dir = '2D/SR/2dof_eval10_D'

g = par.Geometry(sample_dir)
o = par.Optimizer(lll=g.lll, fixed_base=g.fixed_base, evaluation_samples=10, gd_step_number=50)

directory = d.arg_wrapper__sample_dir(sample_dir)

# Save USERSTORE to Local
# d.copy_userstore2local(sample_dir)
print(sample_dir)


# Functions which should be parallelized
def new_world_samples_mp(pid, flag_arr, iw, lock,
                         n_new_world, n_samples_per_world, evaluation_samples):
    """
    Function called by the multiprocessing_wrapper, to perform sg2d.add_path_samples() simultaneously.
    """

    # Update the seed, because the sate for a new child process is always the same -> samples dependent on .random()
    # end always in the same place
    np.random.seed(int(time.time()))

    # Parallel function call
    sg.new_world_samples(n_new_world=n_new_world, n_samples_per_world=n_samples_per_world,
                         directory=directory, evaluation_samples=evaluation_samples, lock=lock, g=g, o=o)

    flag_arr[pid] = 0  # Set flag to 'finished'


def save_add_path_samples_mp(pid, flag_arr, iw, lock,
                             fill, samples_per_world, evaluation_samples):
    """
    Function called by the multiprocessing_wrapper, to perform sg2d.add_path_samples() simultaneously.
    """

    # Update the seed, because the sate for a new child process is always the same -> samples dependent on .random()
    # end always in the same place
    np.random.seed(int(time.time()))

    # Parallel function call
    sg.add_path_samples(i_worlds=iw, n_samples_miss=1, fill=fill, verbose=1, directory=directory, lock=lock,
                        n_samples_cur=samples_per_world[iw], evaluation_samples=evaluation_samples, g=g, o=o)

    flag_arr[pid] = 0  # Set flag to 'finished'


def test_random_seed_mp(pid, flag_arr, iw, lock):
    """
    Function called by the multiprocessing_wrapper, to perform sg2d.add_path_samples() simultaneously.
    """

    # Update the seed, because the sate for a new child process is always the same -> samples dependent on .random()
    # end always in the same place
    np.random.seed(int(time.time()) % 1000)

    print(np.random.random(5))

    flag_arr[pid] = 0  # Set flag to 'finished'


def improve_path_samples_mp(pid, flag_arr, iw, lock,
                            use_x_pred):
    sg.improve_path_samples(i_worlds=iw, directory=directory, use_x_pred=use_x_pred, lock=lock, verbose=1, g=g, o=o)
    flag_arr[pid] = 0  # Set flag to 'finished'


def redo_path_samples_mp(pid, flag_arr, iw, lock,
                         redo_cost_threshold):
    np.random.seed(int(time.time()) % 1000)

    sg.redo_path_samples(i_worlds=iw, directory=directory, lock=lock, verbose=1,
                         redo_cost_threshold=redo_cost_threshold, g=g, o=o)
    flag_arr[pid] = 0  # Set flag to 'finished'


def update_objective_mp(pid, flag_arr, iw, lock,
                        use_x_pred):
    sg.update_objective(i_worlds=iw, directory=directory, lock=lock, use_x_pred=use_x_pred, verbose=1, g=g, o=o)
    flag_arr[pid] = 0  # Set flag to 'finished'


def update_path_img_mp(pid, flag_arr, iw, lock):
    sg.update_path_img(i_worlds=iw, directory=directory, lock=lock, verbose=1, g=g)
    flag_arr[pid] = 0  # Set flag to 'finished'


def repair_path_img_mp(pid, flag_arr, iw, lock):
    sg.repair_path_img(i_worlds=iw, directory=directory, lock=lock, verbose=1, g=g, o=o)
    flag_arr[pid] = 0  # Set flag to 'finished'


def reduce_n_voxels_mp(pid, flag_arr, iw, lock,
                       kernel, kernel_old):
    sg.reduce_n_voxels_df(g=g, directory=directory, lock=lock, kernel=kernel, kernel_old=kernel_old, i_worlds=iw)
    flag_arr[pid] = 0  # Set flag to 'finished'


# Pose Image
def pose_img_generation_mp(pid, flag_arr, iw, lock,
                           n_samples):
    sg_pose.generate_samples(n_samples=n_samples, directory=directory, lock=lock, verbose=0)
    flag_arr[pid] = 0  # Set flag to 'finished'


def path_img_generation_mp(pid, flag_arr, iw, lock,
                           n_samples):
    sg_path.generate_samples(n_samples=n_samples, directory=directory, lock=lock, verbose=0)
    flag_arr[pid] = 0  # Set flag to 'finished'


# Wrapper for Parallelization
def mutiprocessing_wrapper(fun_parallel, kwargs_fun={}, i_worlds=-1, n_processes=2, sleep_time=10.0):
    """
    Wrapper function that takes a function handle and additional arguments and runs it in parallel.
    The parallelization is done for the different worlds. Each processor handles one world at a time and then moves on
    to the next 'free' world.
    Fixed arguments for the function are the processor ID 'pid', the world number 'iw' and an array to indicate the
    state of each processor 'flag_arr' and a lock variable (for example for sql queries)
    """

    i_world_list = ld.arg_wrapper__i_world(i_worlds)

    n_processes = min(n_processes, len(i_world_list))

    processes = []
    processes_cur_world = []
    flag_arr = mp.Array('o', range(n_processes))
    lock = mp.Lock()
    for i in range(n_processes):
        # m_pipe, worker_pipe = mp.Pipe(duplex=False)
        cur_world = i_world_list.pop()
        flag_arr[i] = 1  # Running
        kwargs_process = {'pid': i, 'flag_arr': flag_arr, 'iw': cur_world, 'lock': lock}
        p = mp.Process(target=fun_parallel, name=str(i), kwargs={**kwargs_process, **kwargs_fun})
        p.start()

        processes.append(p)
        processes_cur_world.append(cur_world)

    # Loop forever over all process to keep everything running
    while i_world_list or mp.active_children():
        print('Number of worlds remaining', len(i_world_list))
        print(processes_cur_world)
        # print(mp.active_children())

        time.sleep(sleep_time)
        for i, p in enumerate(processes):
            if not p.is_alive():
                if flag_arr[i] == 1:  # Not Properly finished
                    i_world_list.append(processes_cur_world[i])

                if i_world_list:
                    cur_world = i_world_list.pop()
                    flag_arr[i] = 1  # Running
                    kwargs_process = {'pid': i, 'flag_arr': flag_arr, 'iw': cur_world, 'lock': lock}
                    p_new = mp.Process(target=fun_parallel, name=str(i),
                                       kwargs={**kwargs_process, **kwargs_fun})
                    p_new.start()

                    processes[i] = p_new
                    processes_cur_world[i] = cur_world


# MAIN
try:
    # n_prcs = 24
    # fill_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] + [1000] * 10000
    #
    # for o in range(len(fill_list)):
    #
    #     i_worlds = ld.arg_wrapper__i_world(-1, directory=directory)
    #     i_worlds_pre = i_worlds
    #     #
    #     samples_per_world_ = ld_sql.get_n_samples(i_worlds=i_worlds, directory=directory)
    #     i_worlds = np.array(i_worlds)[samples_per_world_ < fill_list[o]].tolist()
    #     print(len(i_worlds_pre), len(i_worlds))
    #
    #     if len(i_worlds) > 0:
    #         mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=n_prcs, sleep_time=1,
    #                                fun_parallel=save_add_path_samples_mp,
    #                                kwargs_fun={'fill': fill_list[o],
    #                                            'samples_per_world': samples_per_world_,
    #                                            'evaluation_samples': False})
    #
    #     elif fill_list[o] == fill_list[-1]:
    #         break

    # for o in range(10):
    #
    #     i_worlds = list(range(1))
    #     print(len(i_worlds))
    #     #
    #     mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=6, sleep_time=1,
    #                            fun_parallel=test_random_seed_mp)
    #     time.sleep(1)

    # for o in range(10):
    #     cost_threshold = 3
    #     obj = ld_sql.get_values_sql(columns='objective', values_only=True, directory=directory)
    #     idx = np.nonzero(obj > cost_threshold)[0]
    #     i_worlds = np.unique(ld.get_i_world(idx)).tolist()
    #
    #     mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=16, fun_parallel=redo_path_samples_mp, sleep_time=3,
    #                            kwargs_fun={'redo_cost_threshold': cost_threshold})

    # i_worlds = ld.arg_wrapper__i_world(-1, directory=directory)
    # mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=12,
    #                        fun_parallel=reduce_n_voxels_mp, kwargs_fun={'kernel': 2,
    #                                                                      'kernel_old': 2}, sleep_time=1)

    # mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=12,
    #                        fun_parallel=improve_path_samples_mp, kwargs_fun={'use_x_pred': False}, sleep_time=5)
    # mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=10,
    #                        fun_parallel=update_objective_mp, kwargs_fun={'use_x_pred': False}, sleep_time=0.1)
    # mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=10,
    #                        fun_parallel=repair_path_img_mp, sleep_time=0.1)

    # Evaluation Data
    n_prcs = 12  # TODO think of system to incorporate the world_number
    i_worlds = list(range(10000))
    mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=n_prcs, sleep_time=1,
                           fun_parallel=save_add_path_samples_mp,
                           kwargs_fun={'fill': 2,
                                       'samples_per_world': [1] * len(i_worlds),
                                       'evaluation_samples': True})

    # Pose Image PathImg
    # n_samples_ = 1000000
    # n_processes = 12
    # n_samples_per_process = 1000
    # i_worlds = list(range(int(n_samples_ / n_samples_per_process)))
    # mutiprocessing_wrapper(i_worlds=i_worlds, n_processes=n_processes,
    #                        fun_parallel=path_img_generation_mp, sleep_time=1,
    #                        kwargs_fun={'n_samples': n_samples_per_process})

finally:
    print(sample_dir)
    pass
    # Save Local to USERSTORE
    d.copy_homelocal2userstore(sample_dir)
