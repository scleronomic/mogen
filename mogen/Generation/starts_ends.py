import numpy as np

from rokin.sample_configurations import sample_q
from mopla.Optimizer import InitialGuess


# FINDING:
#   The length between start and end configuration is not really a good measure for telling hoq difficult a path is
#   A better one might be to check if thee straight line connection between A and B is feasible:
#   if it is, the path is easy - if not, the path is hard
#   The disadvantage of this method is that it is just boolean and no continuous metric.
#   All the metrics I can think of tell you only how hard thee problem is after the fact.
#   If you tried 100 multistarts and none succeeded, the problem was very hard, or probably insolvable...
#   If 100 / 100 succeed, than the problem was easy
#   But if only 10 / 100 succeed, can the problem itself still be easy, solvable through a straight line
#   but all the multistarts get stuck in thee difficult environment
# Idea: if n_samples > 1 use a distance matrix to find nice problems


def __arg_wrapper_acceptance_rate(fun=None):
    if fun is None:
        def acceptance_rate(x):  # noqa
            return 1

        return acceptance_rate

    if isinstance(fun, float):
        def acceptance_rate(x):  # noqa
            return fun

        return acceptance_rate

    else:
        return fun


def __sample_acceptance_bee(start, end, robot, feasibility_check):
    n = 100
    q = InitialGuess.path.q0s_random(start=start, end=end, n_waypoints=n, n_multi_start=[[0], [1]],
                                     robot=robot, order_random=True)
    return feasibility_check(q) == 1


def __sample_acceptance_dist(start, end, rate=np.inf, joint_weighting=1):
    distance = np.linalg.norm((start - end) * joint_weighting)
    if np.random.random() <= rate(distance):
        return True
    else:
        return False


def sample_q_start_end(robot, feasibility_check=None, acceptance_rate=None,
                       verbose=0):

    acceptance_rate = __arg_wrapper_acceptance_rate(acceptance_rate)

    q_start = None
    q_end = None

    max_iter = 1000
    count = 0

    p = np.random.random() < acceptance_rate(None)

    for j in range(max_iter):
        q_start, q_end = sample_q(robot=robot, shape=2, feasibility_check=feasibility_check, verbose=verbose)

        pp = __sample_acceptance_bee(start=q_start, end=q_end,
                                     robot=robot, feasibility_check=feasibility_check)
        if pp == p:
            break

        count += 1

    return q_start, q_end


def test_start_end_sampling(par, robot, feasibility_check):

    n = 10000
    d = np.empty(n)
    f = np.empty(n)

    for i in range(n):
        q_start, q_end = sample_q_start_end(robot=robot, feasibility_check=lambda qq: feasibility_check(q=qq, par=par),
                                            acceptance_rate=0.95)

        q = InitialGuess.path.q0s_random(start=q_start, end=q_end, n_waypoints=100, n_multi_start=[[0], [1]],
                                         robot=robot, order_random=True)
        f[i] = feasibility_check(q, par=par)
        d[i] = np.linalg.norm(q_end-q_start)

    from wzk.mpl import new_fig, plt

    print((f == 1).mean())
    fig, ax = new_fig()
    ax.hist(d, bins=50)
    fig, ax = new_fig()
    ax.hist(f)

    plt.show()
# df = sample_generation_gd(par=par, gd=gd)
