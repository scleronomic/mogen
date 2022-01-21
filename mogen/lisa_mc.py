import numpy as np

import pandas as pd
from wzk.mpl import (Rectangle, make_every_box_fancy, new_fig, golden_ratio, set_style, set_borders,
                     set_ticks_and_labels, set_ticks_position, save_fig)

from wzk.mpl.colors import blues248_9


directory = '/Users/jote/Documents/'
file_csv = directory + 'Miro_Results_heart.csv'
df = pd.read_csv(file_csv, sep=';')
columns = df.columns.values


newline_size = 13
plot_size_big = np.array((15, 15*1/golden_ratio)) / 2.54
set_style('ieee')
ordered_colors5 = blues248_9[::2]


def plot_grid(ax, data, width, height, colors, alpha, text):

    n = len(colors)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            importance = data[x, y]
            if importance != 0:
                rec = Rectangle((y * width, x * height), width=width/n*importance, height=height, fill=True,
                                color=colors[importance - 1], alpha=alpha)
                ax.add_patch(rec)
                if text:
                    ax.text((y + 0.1) * width, (x + 0.5) * height, str(importance),  va='center', ha='center')


def plot_a():
    c = np.arange(15, 24)
    data = df.iloc[:, c].values
    labels_x = columns[c].tolist()
    labels_y = df["Expert field"].values.tolist()[:-1][::-1]
    data = data[:-1][::-1]

    set_borders(left=0.17, right=0.99, top=0.88, bottom=0.18)
    fig, ax = new_fig(width=plot_size_big[0], height=0.9*plot_size_big[1])
    fig.text(0.02, 0.88, 'Expert\nrepresenting...', va='bottom', ha='left')

    axT = ax.twiny()

    width = 3
    height = 1
    ax.set_xlim(0, data.shape[1] * width)
    axT.set_xlim(0, data.shape[1] * width)
    ax.set_ylim(0, data.shape[0] * height)

    colors = ['green', 'orange', 'red']
    plot_grid(ax=ax, data=data, width=width, height=height, colors=colors, alpha=0.5, text=True)
    make_every_box_fancy(ax, shrink_x=0, shrink_y=height*0.1, pad=0.05*max(width, height))

    set_ticks_and_labels(ax=ax, axis='y', labels=labels_y, ticks=np.arange(data.shape[0]) * height + height / 2)

    labels_x_bottom = [lx if i % 2 == 1 else '' for i, lx in enumerate(labels_x)]
    labels_x_top = [lx if i % 2 == 0 else '' for i, lx in enumerate(labels_x)]
    set_ticks_and_labels(ax=ax, axis='x', labels=labels_x_bottom,
                         ticks=np.arange(data.shape[1]) * width + width / 2)
    set_ticks_and_labels(ax=axT, axis='x', labels=labels_x_top,
                         ticks=np.arange(data.shape[1]) * width + width / 2)

    ax.set_xlabel("Perceived criticality for acceptance (3/red: very critical, 2/yellow: somewhat critical, 1/green: not critical)")
    save_fig(file=f"{directory}/plot_a", fig=fig, bbox=None, formats=('png', 'pdf'))


def plot_b():
    c = np.arange(2, 10)
    data = df.iloc[-1, c]
    labels_x = columns[c]

    color = blues248_9[5]
    set_borders(left=0.1, right=0.99, top=0.99, bottom=0.17)
    fig, ax = new_fig(width=plot_size_big[0], height=plot_size_big[1]*0.55)
    ax.bar(np.arange(len(data)), data, color=color)
    make_every_box_fancy(ax, shrink_x=0, shrink_y=0, pad=0.2)
    ax.bar(np.arange(len(data)), data-10, color=color)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Importance of parameter\n(accumulated of all expert rankings)")
    set_ticks_and_labels(ax=ax, axis='x', labels=labels_x, ticks=np.arange(len(data)))
    save_fig(file=f"{directory}/plot_b", fig=fig, bbox=None, formats=('png', 'pdf'))


def plot_c():
    c = np.arange(10, 15)
    data = df.iloc[:, c].values
    labels_x = columns[c].tolist()
    labels_y = df["Expert field"].values.tolist()[:-1][::-1]
    data = data[:-1][::-1]

    colors = ordered_colors5[::-1]

    set_borders(left=0.17, right=0.99, top=0.88, bottom=0.01)
    fig, ax = new_fig(width=plot_size_big[0], height=plot_size_big[1]*0.75)
    fig.text(0.02, 0.88, 'Expert\nrepresenting...', va='bottom', ha='left')

    width = 4
    height = 1
    ax.set_xlim(0, data.shape[1] * width)
    ax.set_ylim(0, data.shape[0] * height)
    plot_grid(ax=ax, data=data, width=width, height=height, colors=colors, alpha=0.9, text=True)
    make_every_box_fancy(ax, shrink_x=width*0.1, shrink_y=height*0.1, pad=0.05*max(width, height))

    set_ticks_and_labels(ax=ax, axis='y', labels=labels_y, ticks=np.arange(data.shape[0]) * height + height / 2)
    set_ticks_and_labels(ax=ax, axis='x', labels=labels_x, ticks=np.arange(data.shape[1]) * width + width / 2)

    set_ticks_position(ax=ax, position='top')
    save_fig(file=f"{directory}/plot_c", fig=fig, bbox=None, formats=('png', 'pdf'))


def plot_d():
    c = np.arange(2, 10)
    data = df.iloc[:, c].values
    labels_x = columns[c].tolist()
    labels_y = df["Expert field"].values.tolist()[:-1][::-1]
    data = data[:-1][::-1]

    colors = blues248_9[::-1]

    set_borders(left=0.17, right=0.99, top=0.88, bottom=0.01)
    fig, ax = new_fig(width=plot_size_big[0], height=plot_size_big[1]*0.75)
    fig.text(0.02, 0.88, 'Expert\nrepresenting...', va='bottom', ha='left')
    ax.text(4*4, 1.5, '---- uniform budget ----', va='center', ha='center')
    ax.text(4*4, 12.5, '---- uniform budget ----', va='center', ha='center')

    width = 4
    height = 1
    ax.set_xlim(0, data.shape[1] * width)
    ax.set_ylim(0, data.shape[0] * height)
    plot_grid(ax=ax, data=data, width=width, height=height, colors=colors, alpha=0.9, text=True)
    make_every_box_fancy(ax, shrink_x=width*0.1, shrink_y=height*0.1, pad=0.05*max(width, height))

    set_ticks_and_labels(ax=ax, axis='y', labels=labels_y, ticks=np.arange(data.shape[0]) * height + height / 2)
    set_ticks_and_labels(ax=ax, axis='x', labels=labels_x, ticks=np.arange(data.shape[1]) * width + width / 2)
    set_ticks_position(ax=ax, position='top')
    save_fig(file=f"{directory}/plot_d", fig=fig, bbox=None, formats=('png', 'pdf'))


plot_a()
plot_b()
plot_c()
plot_d()

