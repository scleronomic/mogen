import numpy as np

import pandas as pd
from wzk.mpl import Rectangle, make_every_box_fancy, new_fig, golden_ratio, set_style, set_borders
from wzk.mpl.colors import *
from wzk import change_tuple_order

# Export as csv from QuestionPro
#   - Single Header Row
#   - show answer values
# Remove the first tow lines from the csv file

directory = '/Users/jote/Downloads/'
report_csv = directory + 'SurveyReport.csv'
fig_directory = directory + 'Plots/'

# df = pd.read_csv('/Users/jote/Documents/Miro_Results.csv', sep=';')
#
# print(df)
# print(df.columns)
# print(df.Bereich)

newline_size = 13
plot_size_small = np.array((8, 8*1/golden_ratio)) / 2.54
plot_size_square = np.array((9, 8)) / 2.54
plot_size_big = np.array((15, 15*1/golden_ratio)) / 2.54
plot_size = np.array(plot_size_small)

save_bbox = None
fontweight_label = 'normal'

formats = 'png'
set_style('ieee')
set_borders(left=0.125, right=0.875, top=0.975, bottom=0.25)


ordered_colors5 = blues248_9[::2]
ordered_colors4 = ordered_colors5[:-1]
ordered_colors3 = ordered_colors5[::2]
ordered_colors2 = ordered_colors5[[0, 2]]

mixed_colors6 = np.array([reds842_9[[3, 6]], blues248_9[[3, 6]], greens284_9[[3, 6]]]).ravel()
mixed_colors5 = np.array([blues248_9[3], greens284_9[3], reds842_9[3], blues248_9[5], reds842_9[5]])
mixed_colors4 = np.array([blues248_9[3], greens284_9[3], reds842_9[3], blues248_9[5]])
mixed_colors3 = np.array([blues248_9[3], greens284_9[3], reds842_9[3]])
wtp_charge_colors7 = blues248_9[:7]


data = np.random.randint(low=0, high=3, size=(13, 9))
color = ['red']
fig, ax = new_fig(aspect=1, width=13, height=59)
ax.set_xlim(0, data.shape[0])
ax.set_ylim(0, data.shape[1])
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        sq = Rectangle((x, y), 1, 1, fill=True, color=ordered_colors3[data[x, y]])
        ax.add_patch(sq)

make_every_box_fancy(ax, shrink_x=0.1, shrink_y=0.1, pad=0.2)
#
# df = pd.read_csv(report_csv, sep=',', error_bad_lines=False)
#
#
# # Clean and Translate data
# del_columns = list(set(range(df.shape[1])).difference(set(en_dict_columns.keys())))
# en_dict_columns = {df.columns[key]: en_dict_columns[key] for key in en_dict_columns}
#
# df.replace(en_dict_values, inplace=True)
# df.rename(columns=en_dict_columns, inplace=True)
#
# df.drop(df.columns[del_columns], axis=1, inplace=True)
# df.drop(0, axis=0, inplace=True)
#
# df.replace({'rural area': 'village'}, inplace=True)
#
#
