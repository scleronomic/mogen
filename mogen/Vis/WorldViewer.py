import numpy as np
from matplotlib import widgets, patches, collections
from wzk.numpy2 import grid_i2x, grid_x2i

import rokin.Vis.robot_2d as plt2
from mopla.GridWorld import templates, random_obstacles


class TextBoxSafe(widgets.TextBox):
    def __init__(self, ax, label, initial='', color='.95', hovercolor='1', label_pad=.01):
        super().__init__(ax=ax, label=label, initial=initial, color=color, hovercolor=hovercolor, label_pad=label_pad)
        self.text_cur = None

    def on_submit(self, func):
        def func_safe(text):
            if text == self.text_cur:
                return
            else:
                self.text_cur = text
                func(text)
        super().on_submit(func_safe)


def get_world_sample(n_voxels,
                     mode='rectangles',
                     file=None, i_world=None):
    if file is None:
        if mode == 'rectangles':
            obstacle_img = random_obstacles.create_rectangle_image(n_voxels=n_voxels, n=10)
        elif mode == 'perlin':
            obstacle_img = random_obstacles.create_perlin_image(n_voxels=n_voxels)
        else:
            raise ValueError
    else:
        # i_world += 1
        raise NotImplementedError
        # obstacle_img_cmp = ld_sql.get_values_sql(file=directory + WORLD_DB, rows=i_world, columns=obstacle_img_CMP,
        #                                          values_only=True)
        # obstacle_img = compressed2img(obstacle_img_cmp, n_voxels=n_voxels)

    return obstacle_img


def initialize_pixel_grid(img, voxel_size, lower_left, ax, **kwargs):

    ij = np.array(list(np.ndindex(img.shape)))
    xy = grid_i2x(i=ij, cell_size=voxel_size, lower_left=lower_left, mode='b')

    pixel_grid = collections.PatchCollection([patches.Rectangle(xy=(x, y), width=voxel_size,
                                                                height=voxel_size, snap=True)
                                              for (x, y) in xy], **kwargs)
    ax.add_collection(pixel_grid)

    return pixel_grid


def set_pixel_grid(bool_img, pixel_grid, **kwargs):
    val_none_dict = {'color': 'None',
                     'edgecolor': 'None',
                     'facecolor': 'None',
                     'linewidth': 0,
                     'linestyle': 'None'}

    def value_wrapper(k):
        v = kwargs[k]

        if isinstance(v, (tuple, list)) and len(v) == 2:
            v, v_other = v

        else:
            v_other = val_none_dict[k]

        def ravel(val):
            if np.size(val) == np.size(bool_img):
                val = np.ravel(val)
            return val

        v = ravel(v)
        v_other = ravel(v_other)

        return np.where(np.ravel(bool_img), v, v_other).tolist()

    for kw in kwargs:
        set_fun = getattr(pixel_grid, f"set_{kw}")
        set_fun(value_wrapper(kw))


def switch_img_values(bool_img, i, j, value=None):
    i = [i] if isinstance(i, int) else i
    j = [j] if isinstance(j, int) else j

    if value is None:
        mean_obstacle_occurrence = np.mean(bool_img[i, j])
        value = not np.round(mean_obstacle_occurrence)

    bool_img[i, j] = value


def get_selected_rectangle(e_click, e_release):
    """
    Order the location of the click and release event.
    return (x_low, y_low), (x_high, y_high)

    If voxel_size is given, convert the Measurements location to pixel location
    """

    x_poss = [e_click.xdata, e_release.xdata]
    y_poss = [e_click.ydata, e_release.ydata]

    x_poss.sort()
    y_poss.sort()

    return np.array((x_poss[0], y_poss[0])), np.array((x_poss[1], y_poss[1]))


class WorldViewer:

    def __init__(self, world, ax=None, i_world=0, file=None):
        self.world = world

        self.file = file
        self.i_world = i_world

        self.face_color = '0.3'
        self.face_color_new = 'orange'
        self.edge_color = 'k'

        if ax is None:
            self.fig, self.ax = plt2.new_world_fig(limits=self.world.limits,
                                                   title=f"world={self.i_world} | sample=?")
        else:
            self.fig, self.ax = ax.get_figure(), ax
        pass

        self.obstacle_img = np.zeros(self.world.n_voxels, dtype=bool)
        self.obstacle_pixel_grid = initialize_pixel_grid(ax=self.ax, img=self.obstacle_img,
                                                         voxel_size=self.world.voxel_size,
                                                         lower_left=self.world.limits[:, 0])
        self.change_sample(i_world=self.i_world)

        # self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click_obstacle)
        # self.cid_click.set_active(False)

        self.rectangle_selector = widgets.RectangleSelector(self.ax, onselect=self.on_select_obstacle)
        self.rectangle_selector.set_active(True)

        self.text_box = TextBoxSafe(plt2.plt.axes([0.85, 0.1, 0.1, 0.05]), 'GridWorld', initial='')
        self.text_box.on_submit(self.__submit)

    def on_click_obstacle(self, event):
        i, j = grid_x2i(x=[event.xdata, event.ydata],
                        cell_size=self.world.voxel_size, lower_left=self.world.limits[:, 0])
        switch_img_values(bool_img=self.obstacle_img, i=i, j=j, value=None)
        self.update_obstacle_image()

    def on_select_obstacle(self, e_click, e_release):
        (x_ll), (x_ur) = get_selected_rectangle(e_click=e_click, e_release=e_release)

        i_ll, i_ur = grid_x2i(x=np.array([x_ll, x_ur]), cell_size=self.world.voxel_size,
                              lower_left=self.world.limits[:, 0])

        switch_img_values(bool_img=self.obstacle_img, value=None,
                          i=slice(i_ll[0], i_ur[0] + 1), j=slice(i_ll[1], i_ur[1] + 1))

        self.update_obstacle_image()

    def update_obstacle_image(self):
        set_pixel_grid(bool_img=self.obstacle_img, pixel_grid=self.obstacle_pixel_grid,
                       edgecolor=self.edge_color, facecolor=self.face_color)

    def change_sample(self, i_world):
        self.i_world = i_world
        self.obstacle_img = get_world_sample(file=self.file, i_world=self.i_world,
                                             n_voxels=self.world.n_voxels)
        self.update_obstacle_image()

    def toggle_activity(self):
        self.rectangle_selector.set_active(not self.rectangle_selector.active)
        print('GridWorld Viewer - RectangleSelector: ', self.rectangle_selector.active)

    def __submit(self, text):
        try:
            self.change_sample(i_world=int(text))

        except ValueError:
            text = [t.strip() for t in text.split(',')]
            self.obstacle_img = templates.create_template_2d(n_voxels=self.world.n_voxels, world=text)
            self.update_obstacle_image()


def test():
    from mopla.parameter import Parameter

    par = Parameter(robot='SingleSphere02', obstacle_img=None)

    wv = WorldViewer(world=par.world, ax=None)
    # set_pixel_grid(bool_img=wv.obstacle_img, pixel_grid=wv.obstacle_pixel_grid, linestyle=('-', '--'))
    # set_pixel_grid(bool_img=wv.obstacle_img, pixel_grid=wv.obstacle_pixel_grid, linewidth=(1, 2))
    # wv.obstacle_pixel_grid.set_hatch('x')
    return wv


if __name__ == '__main__':
    test()
