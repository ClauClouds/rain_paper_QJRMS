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
from figures.mpl_style import cmap_gif, cmap_gif2
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
import os 
import pdb
import imageio.v3 as iio

from PIL import Image
import glob

# define path for plots
from readers.radar import read_radar_single_file


def main():

    # set directory where to store plots and gif
    directory = path_paper_plots+'/Ze_gif/'
    
    # read radar reflectivity
    ds = read_radar_single_file('2020', '02', '19')    
    # select some hours for plotting
    ds_sel = ds.sel(time=slice('2020-02-19T00:20:00', '2020-02-19T00:40:00'))
    print(ds_sel)

    # plot radar reflectivity profiles
    plot_profiles(ds_sel, directory)

    # list filenames 
    create_gif(directory)


def create_gif(directory):
    """
    # Create a list to store the images
    """
    frames = []
    imgs = sorted(glob.glob(directory+"/*.png"))

    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(path_paper_plots+'/png_to_gif.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)
    print(f"GIF created at {directory}")

    return(None)


    
def plot_profiles(ds_sel, directory):
    """
    script to produce plots of radar reflectivity profiles

    Parameters
    ----------
    ds_sel : xarray dataset
        dataset with radar reflectivity and mean doppler velocity
    directory : str
        directory where to store plots and gif
    """
    # set all fonts size
    plt.rcParams.update({'font.size': 20})

    # setting cmap edges
    cmap_min = -2
    cmap_max = 2

    # loop on time 
    for i_t, time_val in enumerate(ds_sel.time.values):
        print(time_val)

        # convert time_val to string yymmdd_hhmmss
        time_val_str = pd.to_datetime(time_val).strftime("%y%m%d_%H%M%S")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(ds_sel.sel(time=time_val).radar_reflectivity, 
                ds_sel.height, 
                label='radar reflectivity', 
                color='black')
        line = plot_colored_line(ax, 
                          ds_sel.sel(time=time_val).radar_reflectivity, 
                          ds_sel.height, 
                          ds_sel.sel(time=time_val).mean_doppler_velocity, 
                          cmap_gif, 
                          cmap_min, 
                          cmap_max)
        

        # add colorbar
        cbar = plt.colorbar(line, ax=ax)
        #add cbar Label
        cbar.set_label('Mean Doppler Velocity (m/s)')

        ax.set_xlim(-45, 20)
        ax.set_ylim(0, 4000)
        ax.set_ylabel('Height (m)')
        ax.set_xlabel('Radar Reflectivity (dBZ)')
        plt.savefig(f"{directory}/{time_val_str}_radar_reflectivity.png", 
                    transparent=True)
        plt.close()
    return(None)


def plot_colored_line(ax, x, y, z, cmap_name, cmap_min, cmap_max):
    
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

    lc = LineCollection(segments, cmap=cmap_name, norm=norm)
    # Set the values used for colormapping
    lc.set_array(z)
    line = ax.add_collection(lc)
    lc.set_linewidth(4)

    

    return(line)


if __name__ == "__main__":
    main()
