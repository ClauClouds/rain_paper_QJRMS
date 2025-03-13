
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figures.mpl_style import CMAP_HF_ALL, COLOR_SHALLOW, COLOR_CONGESTUS, COLOR_N, COLOR_S, COLOR_T


from readers.lidars import read_and_map



def prepare_arthus_data_lat_mean(data, var_name):
    
    # classifying regions based on latitude
    
    #southern region
    ind_southern = np.where(data.Latitude < 10.5)[0]
    # transition region
    ind_transition = np.where( (data.Latitude <= 12.5) & (data.Latitude >= 10.5))[0]
    # northern region
    ind_northern = np.where(data.Latitude > 12.5)[0]
    
    # selecting the data
    ds_s = data.isel(Time=ind_southern)
    ds_t = data.isel(Time=ind_transition)
    ds_n = data.isel(Time=ind_northern) 
    
    var_mean_s = np.nanmean(ds_s[var_name].values, axis=0)
    var_mean_t = np.nanmean(ds_t[var_name].values, axis=0)
    var_mean_n = np.nanmean(ds_n[var_name].values, axis=0)
    var_mean_all_domain = np.nanmean(data[var_name].values, axis=0)
    
    return(var_mean_s, var_mean_t, var_mean_n, var_mean_all_domain, data.Height.values)

def plot_profiles_arthus(ax, 
                         t_s, 
                         t_t, 
                         t_n, 
                         t_mean, 
                         height):
    
    

    
    
    # fill area between the region and all
    kwargs = dict(
        x1=t_mean, y=height * 1e-3, linewidth=0, alpha=0.5, zorder=0
    )
    ax.fill_betweenx(x2=t_s, color=COLOR_S, **kwargs)
    ax.fill_betweenx(x2=t_t, color=COLOR_T, **kwargs)
    ax.fill_betweenx(x2=t_n, color=COLOR_N, **kwargs)

    # plot mean profile southern
    ax.plot(
        t_s,
        height * 1e-3,
        color=COLOR_S,
        zorder=1,
    )
    
    # plot mean profile northern
    ax.plot(
        t_n,
        height * 1e-3,
        color=COLOR_N,
        zorder=1,
    )
   
    # plot mean profile northern
    ax.plot(
        t_t,
        height * 1e-3,
        color=COLOR_T,
        zorder=1,
    )
   
    # plot mean profile of all
    ax.plot(
        t_mean,        
        height* 1e-3,
        color='black',
        label="All observations",
        linewidth=4,
        zorder=1,
    ) 
    
    ax.grid(True, linestyle='--', alpha=0.5)
    # set fontsize in ax axis ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    
        
    # Create custom legend handles
    legend_handles = [
        Patch(facecolor=COLOR_S, 
              edgecolor='none', 
              alpha=0.5, 
              label='Southern'),
        Patch(facecolor=COLOR_T, 
              edgecolor='none', 
              alpha=0.5, 
              label='Transition'),
        Patch(facecolor=COLOR_N, 
              edgecolor='none', 
              alpha=0.5, 
              label='Northern'), 
        Line2D([0], [0], 
               color='black', 
               linewidth=4, 
               label='All observations'),      
    ]

    # Add the custom legend handles to the legend
    ax.legend(handles=legend_handles, loc="upper right", fontsize=16)

    
    return(ax)




font_size=25
font_titles=25

# read lidar data and lat/lon from ship data and assign lat /lon to arthus
T = read_and_map('T')
MR = read_and_map('MR')

# prepare data for the figure (mean T over the trajectory, mean over southern, transition, and northern domain)
t_s, t_t, t_n, t_mean, height = prepare_arthus_data_lat_mean(T,'Product')
mr_s, mr_t, mr_n, mr_mean, height = prepare_arthus_data_lat_mean(MR,'Product')


# plot of the profiles of T for the trajectory regions
fig, ax = plt.subplots(figsize=(8, 12))

ax.set_title("c) Mean temperature", 
                loc='left', 
            fontsize=font_titles,
            fontweight='black')
ax.set_xlim([290, 299])
ax.set_yticks(np.arange(0.25, 1, 0.1))
ax.set_ylim([0.250, 1])
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xlabel('Temperature [K]', fontsize=font_size)
ax.set_ylabel('Height [km]', fontsize=font_size)
plot_profiles_arthus(ax, t_s, t_t, t_n, t_mean, height)

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_c.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

print('figure fig_1_c.png saved')


fig, ax = plt.subplots(figsize=(8, 12))

ax.set_title("d) Mean mixing ratio", 
                loc='left', 
            fontsize=font_titles,
            fontweight='black')
ax.set_xlim([12, 17])
ax.set_yticks(np.arange(0.25, 1, 0.1))
ax.set_ylim([0.250, 1])
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xlabel('Mixing ratio  [gkg$^{-1}$]', fontsize=font_size)
ax.set_ylabel('Height [km]', fontsize=font_size)
plot_profiles_arthus(ax, mr_s, mr_t, mr_n, mr_mean, height)

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_d.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

print('figure fig_1_c.png saved')

