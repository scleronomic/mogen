import numpy as np

from Kinematic.sample_configurations import sample_q


def __arg_wrapper_acceptance_rate(fun=None):
    if fun is None:
        def acceptance_rate(dist):
            return 1

        return acceptance_rate
    else:
        return fun


def __sample_acceptance_dist(start, end, acceptance_rate=np.inf, joint_weighting=1):
    distance = np.linalg.norm((start - end) * joint_weighting)
    if np.random.random() <= acceptance_rate(distance):
        return True
    else:
        return False


def sample_q_start_end(*, par, feasibility_check=False, acceptance_rate_q=None):
    # TODO idea: if n_samples > 1 use a distance matrix to find nice problems

    acceptance_rate_q = __arg_wrapper_acceptance_rate(acceptance_rate_q)

    q_start = None
    q_end = None

    max_iter = 1000
    count = 0

    for j in range(max_iter):
        q_start, q_end = sample_q(robot=par.robot, shape=2, feasibility_check=feasibility_check)

        if __sample_acceptance_dist(start=q_start, end=q_end, acceptance_rate=acceptance_rate_q):
            break

        count += 1

        # print('Count_dist', count)

    return q_start, q_end
