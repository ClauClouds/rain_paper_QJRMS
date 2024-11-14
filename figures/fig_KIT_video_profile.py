'''Code to create a gif of radar reflectivity profile 
'''


from readers.cloudtypes import read_cloud_class, read_rain_ground, read_cloud_base, read_in_clouds_radar_moments
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_paper_plots
from readers.lcl import read_lcl, read_diurnal_cycle_lcl
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import xarray as xr
#from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import metpy.calc as mpcalc
from metpy.units import units
import pdb
from dask.diagnostics import ProgressBar
ProgressBar().register()

# define path for plots
from readers.radar import read_radar_single_file
def main():

    path_paper_plots = "/net/ostro/plots_rain_paper/ZE_gif/"

    # read radar reflectivity
    ds = read_radar_single_file('2020', '02', '19')    
    # select a rainy day
    print(ds)

    # setting cmap edges
    cmap_min = -4
    cmap_max = 4

    # loop on time 
    for i_t, time_val in enumerate(ds.time.values):
        print(time_val)

        # convert time_val to string yymmdd_hhmmss
        time_val_str = pd.to_datetime(time_val).strftime("%y%m%d_%H%M%S")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        line = plot_colored_line(ax, 
                      ds.sel(time=time_val).radar_reflectivity, 
                      ds.height, 
                      ds.sel(time=time_val).height,
                      cmap_min,
                      cmap_max)
        .plot(ax=ax, cmap=CMAP, vmin=-20, vmax=20)
        ax.set_title(time_val)
        plt.savefig(f"{path_paper_plots}/{time_val_str}_radar_reflectivity.png")
        plt.close()

    
    # create gif
    os.system(f"convert -delay 100 -loop 0 {path_paper_plots}/*_radar_reflectivity.png {path_paper_plots}/radar_reflectivity.gif")
    

def plot_colored_line(ax, x, y, z, cmap_name, label, cmap_min, cmap_max):
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
        
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(cmap_min, cmap_max)


    # set line style based on class
    if label == 'congestus non prec':
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('--')
        lc.set_linewidth(4)
    elif label == 'congestus prec':
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('solid')
        lc.set_linewidth(6)
    else:
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('--')
        lc.set_linewidth(2)
    ax.legend(frameon=False)

    return(line)


if __name__ == "__main__":
    main()
