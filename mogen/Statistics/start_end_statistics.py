import numpy as np
import scipy.interpolate as interp
from wzk import new_fig, print_progress, get_points_inbetween, binomial, grid_x2i

from definitions import PROJECT_DATA_UTIL


def get_distance_distribution_file(n_dim, mode):
    return PROJECT_DATA_UTIL + f"distance_distribution_{mode}_{n_dim}dim.npy"


def get_distance_distribution(n_samples=1e8, bins=100, verbose=0, save=False, n_dim=2, mode='l2'):
    """
    The distance between two points, which are both sampled from an uniform distribution [0, 1], is not distributed
    uniformly. Moderate distances are far more likely than the very large and the very small distances.
    To describe the resulting distribution a histogram is used.
    The two points are sampled on (U[0, 1])**2 in 2D therefore the minimal distance is 0 and the maximal distance is
    sqrt(2).
    """

    n_samples = int(n_samples)
    x_start = np.random.random((n_samples, n_dim))
    x_end = np.random.random((n_samples, n_dim))
    if mode == 'l1':
        dist = np.abs(x_end - x_start).mean(axis=-1)
    elif mode == 'l2':
        dist = np.linalg.norm(x_end - x_start, axis=1)
    else:
        raise ValueError(f"Unknown mode {mode}")

    y, x_b = np.histogram(dist, range=(0, np.sqrt(n_dim)), bins=bins)

    x_c = get_points_inbetween(x_b)

    # Normalize probability distribution to 1
    x_base = x_b[1:] - x_b[:-1]
    y = y / np.sum(y * x_base)

    # Include origin
    y = np.hstack(([0], y))
    x_c = np.hstack(([0], x_c))

    if verbose >= 1:
        fig, ax = new_fig()
        ax.plot(x_c, y)

    if save:
        np.save(get_distance_distribution_file(n_dim=n_dim, mode=mode), [x_c, y])

    return x_c, y


def aim_distribution(y, x_c, x_drop_start=0.70, x_drop_threshold=5e-5, alpha=1, n_dim=2, verbose=0):
    """
    Define a desired distribution for the direct path length of the samples.
    Set all values after the peak of the distribution to the value of the peakb.
    Model the tail of the distribution via the parameters 'x_drop_start' and 'x_drop_end'.
    These to percent numbers define the region in which the distribution goes down to zero.
    """

    voxel_size = np.diff(x_c).mean()
    y_aim = np.copy(y)

    if n_dim == 1:
        template_length = len(x_c) * 18 // 100

        template = np.cumsum(irwin_hall_distribution(x_c[:template_length], n=2))
        y_aim[:template_length] = template
        i_ymax = np.argmax(y_aim)
        y_aim[i_ymax:len(x_c) - template_length] = y[i_ymax]
        y_aim[len(x_c) - template_length:] = template[::-1]
    else:

        i_ymax = np.argmax(y)

        i_xr1 = grid_x2i(x=x_drop_start * np.sqrt(n_dim), cell_size=voxel_size, lower_left=0)
        xc_tail = np.linspace(-5, 5, len(x_c[i_xr1:]))

        y_aim[i_ymax:] = y[i_ymax]
        y_aim[i_xr1:] = -0.5 * y[i_ymax] * (1 + np.tanh(alpha * xc_tail)) + y[i_ymax]
        y_aim[y < x_drop_threshold] = y[y < x_drop_threshold]

    # Normalize probability distribution to 1
    y_aim = y_aim / np.sum(y_aim * voxel_size)

    if verbose >= 1:
        fig, ax = new_fig()
        ax.plot(x_c, y_aim)

    return y_aim


def get_acceptance_rate(x_c, y, y_aim, verbose=0, return_full=False):
    """
    Maximal meaningful value for the acceptance probability is 1, because you can't do more than accept a sample.
    If the probability of acceptance for a given x1 is 5 and for x2 it is 3, there is no difference
    both samples are accepted 100% of the time they occur. To account for the different weighting
    between the two, the whole distribution is normalized, so that the maximal prop. of acceptance
    is 1.
    """

    y_a_y = y_aim / y
    y_a_y[np.isposinf(y_a_y)] = 0  # Make sure all nan's and inf's which may occur are set to 0
    y_a_y = np.nan_to_num(y_a_y)
    y_a_y[0] = y_a_y[1]  # Enforce continuity / standard behavior for small distances

    # Normalize probability distribution to 1
    # x_base = x_b[1:] - x_b[:-1]
    # y_a_y = y_a_y / np.sum(y_a_y * x_base)

    y_a_y /= np.max(y_a_y)

    y_a_y_interp = interp.interp1d(x_c, y_a_y)

    if verbose >= 1:
        fig, ax = new_fig()
        ax.plot(x_c, y, c='b', label='distance distribution')
        ax.plot(x_c, y_aim, c='g', label='desired distribution')
        ax.plot(x_c, y_a_y, c='r', label='conversion factor')
        ax.legend()

    if return_full:
        return y_a_y_interp, y_a_y
    else:
        return y_a_y_interp


def get_acceptance_rate_distance_wrapper(world_size=1, verbose=0, n_dim=2, mode='l2',
                                         load=True):
    """
    Return a function which calculates the acceptance rate for a given distance.
    Calculate the distribution which the samples would follow normally and the distribution which they should follow.
    The acceptance rate is defined as the relation of these two probability distributions.
    Try to load distribution from file to avoid new sampling of the normal behavior.
    """

    if load:
        try:
            x_c, y = np.load(get_distance_distribution_file(n_dim=n_dim, mode=mode))
        except FileNotFoundError:
            x_c, y = get_distance_distribution(n_dim=n_dim, mode=mode, verbose=verbose)
    else:
        x_c, y = get_distance_distribution(n_dim=n_dim, mode=mode, verbose=verbose)

    y_aim = aim_distribution(y=y, x_c=x_c, n_dim=n_dim, verbose=verbose)
    y_a_y = get_acceptance_rate(x_c=x_c, y=y, y_aim=y_aim, verbose=verbose)
    factor = world_size  # sqrt(ws^2 + ws^2) / sqrt(2) = sqrt(ws^2) * sqrt(2) / sqrt(2) = ws

    def ar(distance):
        try:
            return y_a_y(distance / factor)
        except ValueError:
            # print(distance)
            return 0

    if verbose >= 2:
        test_distribution(acceptance_rate=lambda distance: ar(factor * distance), n_dim=n_dim,
                          verbose=verbose - 2)

    return ar


def test_distribution(acceptance_rate, n_samples=1000, n_dim=2, verbose=0):
    xy_dist = np.zeros(n_samples)
    count = 0
    for i in range(n_samples):
        if verbose >= 1:
            # print(o)
            pass
        while True:
            count += 1
            x_start = np.random.random(n_dim)
            x_end = np.random.random(n_dim)
            dist = np.linalg.norm(x_end - x_start)  # x
            if np.random.random() <= acceptance_rate(dist):
                break

        xy_dist[i] = dist

    fig, ax = new_fig()
    ax.hist(xy_dist, bins=100)
    print('Tries per sample: ', count / n_samples)


def test_distribution_a(n_samples=10000, n_dof=3, mode='l2'):
    n_samples = int(n_samples)
    acceptance_rate = get_acceptance_rate_distance_wrapper(world_size=1, verbose=0, n_dim=1, mode=mode, load=True)

    dist = np.zeros(n_samples)

    if np.size(acceptance_rate) == 1:
        acceptance_rate = [acceptance_rate for _ in range(n_dof)]

    count = 0
    for i in range(n_samples):
        print_progress(i, n_samples)
        while True:
            count += 1
            a_start = np.random.random(n_dof)
            a_end = np.random.random(n_dof)

            acceptance_temp = np.array([acceptance_rate[ii](np.abs(e - a))
                                        for ii, (a, e) in enumerate(zip(a_start, a_end))])

            # acceptance_temp = acceptance_temp.mean() ** 2  # TODO magic, plot the results
            acceptance_temp = acceptance_temp.mean() ** 2  # TODO magic, plot the results

            if np.random.random() <= acceptance_temp:
                break

        dist[i] = np.abs(a_end - a_start).mean()

    y, x_b = np.histogram(dist, range=(0, 1), bins=100)
    x_c = get_points_inbetween(x_b)
    fig, ax = new_fig()
    ax.plot(x_c, y)

    np.save(PROJECT_DATA_UTIL + f"mad_real_{mode}_{n_dof}.npy", [x_c, y])

    print('Tries per sample: ', count / n_samples)


def irwin_hall_distribution(x, n=2):
    """
    https://en.wikipedia.org/wiki/Irwin-Hall_distribution
    """

    pre_factor = 1 / 2 / np.math.factorial(n - 1)

    f_xn = 0
    for k in range(n + 1):
        f_xn += (-1) ** k * binomial(n, k) * (x - k) ** (n - 1) * np.sign(x - k)

    return pre_factor * f_xn
