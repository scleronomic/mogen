import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile as shp  # Requires the pyshp package: pip install pyshp


file = '/Users/jote/Documents/Lisa_GPS/onlygeocode.csv'
map_size = 'l'

shapefile_dir = '/Users/jote/Documents/Lisa_GPS/osm_ger_muc'


def get_extent(size='l'):
    """
    l: large
    m: medium
    ms: medium-small
    s: small
    c: center
    ap: with airport

    """
    ll_lon = {'l': 11.35,  # 29.5km x 24.4km  |  9 x 5, 36 x 40, 288 x 144  |  large
              'm': 11.42578125,  # 22.9km x 14.7km  |  7 x 3, 28 x 24, 224 x 96   |  medium
              'ms': 11.46972657,  # 16.4km x 14.7km  |  5 x 3, 20 x 24, 160 x 96   |  medium-small
              's': 11.51367188,  # 9.8km  x 4.9km   |  3 x 1, 12 x 8,   96 x 32   |  small
              'c': 11.55761719,  # 3.3km  x 4.9km   |  1 x 1,  4 x 8,   32 x 32   |  center
              'ap': 11.375}  # with airport
    ll_lat = {'l': 48.05,
              'm': 48.07617188,
              'ms': 48.07617188,
              's': 48.12011719,
              'c': 48.12011719,
              'ap': 48.04}
    ur_lon = {'l': 11.75,
              'm': 11.73339843,
              'ms': 11.68945312,
              's': 11.64550781,
              'c': 11.6015625,
              'ap': 11.75}
    ur_lat = {'l': 48.25195312,
              'm': 48.20800781,
              'ms': 48.20800781,
              's': 48.1640620,
              'c': 48.1640620,
              'ap': 48.29}

    """Return tuple for the map boundaries"""
    return ll_lon[size], ur_lon[size], ll_lat[size], ur_lat[size]


def plot_shapefile(ax, sf, **kwargs):

    if isinstance(sf, (tuple, list)):
        sf_dir, sf = sf

        sf2 = f"{sf_dir}/{sf}/{sf}.shp"

    sf2 = shp.Reader(sf2)

    for shape in sf2.shapeRecords():
        if sf != 'districts' or 'bezirk' in shape.record.name.lower():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            ax.plot(x, y, **kwargs)


extent = get_extent(size=map_size)


df = pd.read_csv(file, names=['Lon', 'Lat'], sep=";")
points = np.array((df["Lon"].values, df["Lat"].values)).T

fig, ax = plt.subplots()
ax.set_aspect(1.5)
ax.set_xlim(extent[0:2])
ax.set_ylim(extent[2:4])
ax.plot(*points.T, color="blue", marker='o', markersize=3, alpha=0.5, ls='', zorder=100)

plot_shapefile(ax=ax, sf=(shapefile_dir, 'big_roads'), color='black', lw=1)
plot_shapefile(ax=ax, sf=(shapefile_dir, 'districts'), color='xkcd:light blue', lw=3)

fig.savefig('/Users/jote/Documents/Lisa_GPS/munich.pdf')
plt.show()

