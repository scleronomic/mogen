import numpy as np
from wzk.mpl import DraggableEllipseList, new_fig, add_safety_limits
from wzk import get_mean_divisor_pair


def draggable_configuration_trajectories(q, limits, circle_ratio=1/3, **kwargs):
    q = np.squeeze(q)
    n_waypoints, n_dof = q.shape

    if n_dof > 5:
        n_cols, n_rows = get_mean_divisor_pair(n=n_dof)
    else:
        n_cols, n_rows = 1, n_dof

    fig, axes = new_fig(n_rows=n_rows, n_cols=n_cols, share_x=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.2)

    axes.flatten()[-1].set_xlim([-1, n_waypoints])
    axes.flatten()[-1].set_xticks(np.arange(n_waypoints))
    axes.flatten()[-1].set_xticklabels([str(i) if i % 2 == 0 else '' for i in range(n_waypoints)])

    for ax, limits_i in zip(axes.flatten(), limits):
        limits_i_larger = add_safety_limits(limits=limits_i, factor=0.05)

        y_ticks = np.linspace(limits_i[0], limits_i[1], 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(["{:.2f}".format(v) for v in y_ticks])
        ax.set_ylim(limits_i_larger)

    x_temp = np.arange(n_waypoints)

    h_lines = [ax.plot(x_temp, q_i, **kwargs)[0] for ax, q_i in zip(axes.flatten(), q.T)]

    def update_wrapper(i):
        def __update():
            y_i = dgel_list[i].get_xy()[:, 1]
            h_lines[i].set_ydata(y_i)

        return __update

    dgel_list = [DraggableEllipseList(ax=ax, vary_xy=(False, True),
                                      xy=np.vstack([x_temp, q_i]).T,
                                      width=circle_ratio, height=None,
                                      callback=update_wrapper(i),
                                      **kwargs)
                 for i, (ax, q_i, limits_i) in enumerate(zip(axes.flatten(), q.T, limits))]

    return fig, axes, dgel_list, h_lines


class DraggableConfigSpace:

    def __init__(self, q, limits, circle_ratio=1/3, **kwargs):
        self.q = np.squeeze(q)
        self.n_wp, self.n_dof = self.q.shape

        self.fig, self.axes, self.dgel_list, self.h_lines = \
            draggable_configuration_trajectories(q=q, limits=limits, circle_ratio=circle_ratio, **kwargs)

    def set_callback(self, callback):
        for dgel in self.dgel_list:
            dgel.set_callback(callback=callback)

    def add_callback(self, callback):
        for dgel in self.dgel_list:
            dgel.add_callback(callback=callback)

    def get_q(self):
        return np.array([dgel.get_xy()[:, 1] for dgel in self.dgel_list]).T

    def set_q(self, q):
        self.q = q
        for dgel, h_line_i, q_i in zip(self.dgel_list, self.h_lines, self.q.T):
            dgel.set_xy(y=q_i)
            h_line_i.set_ydata(q_i)


def test():
    q = np.random.random((20, 3))
    limits = np.zeros((q.shape[-1], 2))
    limits[:, 1] = 1
    dcs = DraggableConfigSpace(q=q, limits=limits, circle_ratio=1 / 4, color='k')

    print(dcs.get_q().shape)

    dcs.set_q(q=np.zeros((20, 3)))
    input()
    dcs.set_q(q=np.ones((20, 3)))


if __name__ == '__main__':
    test()
