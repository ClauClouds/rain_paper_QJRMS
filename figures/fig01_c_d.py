
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
from figures.fig04_anomalies_subcloud_radar_incloud import calc_lcl_grid_no_dc

from readers.lidars import read_and_map
from readers.lcl import read_lcl
from readers.cloudtypes import read_cloud_class, read_rain_flags


def prepare_arthus_data_lat_mean(data, var_name):
    
    # classifying regions based on latitude
    
    #southern region
    ind_southern = np.where(data.ship_latitude < 10.5)[0]
    # transition region
    ind_transition = np.where( (data.ship_latitude <= 12.5) & (data.ship_latitude >= 10.5))[0]
    # northern region
    ind_northern = np.where(data.ship_latitude > 12.5)[0]
    
    # selecting the data
    ds_s = data.isel(time=ind_southern)
    ds_t = data.isel(time=ind_transition)
    ds_n = data.isel(time=ind_northern) 
    
    var_mean_s = np.nanmean(ds_s[var_name].values, axis=0)
    var_mean_t = np.nanmean(ds_t[var_name].values, axis=0)
    var_mean_n = np.nanmean(ds_n[var_name].values, axis=0)
    var_mean_all_domain = np.nanmean(data[var_name].values, axis=0)
    
    return(var_mean_s, var_mean_t, var_mean_n, var_mean_all_domain, data.height.values)

def plot_profiles_arthus(ax, 
                         t_s, 
                         t_t, 
                         t_n, 
                         t_mean, 
                         height,
                         group='all'):
    
    

    
    
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
   
    if group == 'all':
        # plot mean profile of all
        ax.plot(
            t_mean,        
            height* 1e-3,
            color='black',
            label="All observations",
            linewidth=4,
            zorder=1,
        ) 
        label_legend = 'All observations'
        color_legend = 'black'

    elif group == 'shallow':
        ax.plot(
            t_mean,        
            height* 1e-3,
            color=COLOR_SHALLOW,
            label="Shallow",
            linewidth=4,
            zorder=1,
        )   
        label_legend = 'Shallow clouds'
        color_legend = COLOR_SHALLOW
        
    elif group == 'congestus':
        ax.plot(
                t_mean,        
                height* 1e-3,
                color=COLOR_CONGESTUS,
                label="Congestus",
                linewidth=4,
                zorder=1,
            )     
        label_legend = 'Congestus clouds'
        color_legend = COLOR_CONGESTUS
        
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
               color=color_legend, 
               linewidth=4, 
               label=label_legend),      
    ]

    # Add the custom legend handles to the legend
    ax.legend(handles=legend_handles, loc="upper right", fontsize=16)

    
    return(ax)



font_size=25
font_titles=25


# read lidar data and lat/lon from ship data and assign lat /lon to arthus
T = read_and_map('T')
MR = read_and_map('MR')

# rename time and height variables
T = T.rename({'Time': 'time', 'Height': 'height'})
MR = MR.rename({'Time': 'time', 'Height': 'height'})

# read lcl and calculate its diurnal cycle at 15 mins and at 30 mins (for fluxes)
ds_lcl = read_lcl()

# align time stamps for ds_lcl and ds_arthus
ds_lcl_interp_T, ds_T_interp = xr.align(ds_lcl, T, join="inner")
ds_lcl_interp_MR, ds_MR_interp = xr.align(ds_lcl, MR, join="inner")


# regrid temperature and mixing ratio on height with respect to lcl
T_lcl = calc_lcl_grid_no_dc(ds_T_interp, ds_lcl_interp_T, 'height', 'time', 'Product')
MR_lcl = calc_lcl_grid_no_dc(ds_MR_interp, ds_lcl_interp_MR, 'height', 'time', 'Product')

# read cloud classification to add profiles for congestus and shallow clouds
# read cloud classification
cloudclassdata = read_cloud_class()

# read rain flags
rainflags = read_rain_flags()

# create new xarray dataset and add variable field, ship_latitude and longitude to it
ds_T = xr.Dataset()
# add variable field to the dataset
ds_T['T'] = (['time', 'height'], T_lcl.values)
# add ship latitude and longitude to the dataset
ds_T['ship_latitude'] = (['time'], ds_T_interp.ship_latitude.values)
ds_T['ship_longitude'] = (['time'], ds_T_interp.ship_longitude.values)
# add time coordinate to the dataset
ds_T['time'] = ds_T_interp.time
# add height coordinate to the dataset
ds_T['height'] = T_lcl.height

ds_MR = xr.Dataset()
# add variable field to the dataset
ds_MR['MR'] = (['time', 'height'], MR_lcl.values)
# add ship latitude and longitude to the dataset
ds_MR['ship_latitude'] = (['time'], ds_MR_interp.ship_latitude.values)
ds_MR['ship_longitude'] = (['time'], ds_MR_interp.ship_longitude.values)
# add time coordinate to the dataset
ds_MR['time'] = ds_MR_interp.time
# add height coordinate to the dataset
ds_MR['height'] = MR_lcl.height 

# align all datasets
cloudclassdata, rainflags, ds_T, ds_MR, ds_lcl = xr.align(cloudclassdata, rainflags, ds_T, ds_MR, ds_lcl, join="inner")   

# select cloud columns
cloudyflag = ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain

# interpolate classification on LWP time stamps and derive flags for shallow/congestus data
cloudclassdataip = cloudclassdata.interp(time=ds_T.time)
shallow = (cloudclassdataip.shape.values == 0) & (cloudyflag == 1)
congestus = (cloudclassdataip.shape.values == 1) &(cloudyflag == 1)

# define ds_T_sl and ds_T_cg for shallow and congestus clouds
ds_T_sl = ds_T.sel(time=shallow, drop=True)
ds_T_cg = ds_T.sel(time=congestus, drop=True)
ds_MR_sl = ds_MR.sel(time=shallow, drop=True)
ds_MR_cg = ds_MR.sel(time=congestus, drop=True)
ds_lcl_sl = ds_lcl.sel(time=shallow, drop=True)
ds_lcl_cg = ds_lcl.sel(time=congestus, drop=True)

# calculate mean LCL height for shallow and congestus clouds
lcl_mean_all = np.nanmean(ds_lcl.lcl.values, axis=0)
lcl_sl_mean = lcl_mean_all - np.nanmean(ds_lcl_sl.lcl.values, axis=0)
lcl_cg_mean = lcl_mean_all - np.nanmean(ds_lcl_cg.lcl.values, axis=0)


# derive mean profiles for shallow and congestus clouds 
# (mean T over the trajectory, mean over southern, transition, and northern domain)
t_sl_s, t_sl_t, t_sl_n, t_sl_mean, height_sl = prepare_arthus_data_lat_mean(ds_T_sl,'T')
t_cg_s, t_cg_t, t_cg_n, t_cg_mean, height_cg = prepare_arthus_data_lat_mean(ds_T_cg,'T')
mr_sl_s, mr_sl_t, mr_sl_n, mr_sl_mean, height_sl = prepare_arthus_data_lat_mean(ds_MR_sl,'MR')
mr_cg_s, mr_cg_t, mr_cg_n, mr_cg_mean, height_cg = prepare_arthus_data_lat_mean(ds_MR_cg,'MR')


# prepare data for the figure (mean T over the trajectory, mean over southern, transition, and northern domain)
t_s, t_t, t_n, t_mean, height = prepare_arthus_data_lat_mean(ds_T,'T')
mr_s, mr_t, mr_n, mr_mean, height = prepare_arthus_data_lat_mean(ds_MR,'MR')

# plot shallow and congestus profiles in a separate figure 
fig, ax = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
# Color the area below the LCL height in grey
lcl_height = 0  # Assuming LCL height is at 0 km
ax[0].fill_between([190, 299],
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)
# plot temperature profile for all louds
#ax[0].set_title("c) Mean temperature", 
#                loc='left', 
#            fontsize=font_titles,
#            fontweight='black')
ax[0].set_xlim([290, 299])
ax[0].set_ylim([-0.5, 1.])
ax[0].set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax[0].set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax[0].set_ylabel("Height above LCL [km]",fontsize=font_size)
ax[0].tick_params(axis='both', which='major', labelsize=font_size)
ax[0].set_xlabel('Temperature [K]', fontsize=font_size)
plot_profiles_arthus(ax[0], t_s, t_t, t_n, t_mean, height, "all")
ax[1].fill_between([290, 299], 
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)

#ax[1].set_title("c) Mean temperature", 
#                loc='left', 
#            fontsize=font_titles,
#            fontweight='black')
ax[1].set_xlim([290, 299])
ax[1].set_ylim([-0.5, 1.])
ax[1].set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax[1].set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax[1].tick_params(axis='both', which='major', labelsize=font_size)
ax[1].set_xlabel('Temperature [K]', fontsize=font_size)
plot_profiles_arthus(ax[1], t_sl_s, t_sl_s, t_sl_n, t_sl_mean, height_sl, 'shallow')
plot_profiles_arthus(ax[1], t_cg_s, t_cg_t, t_cg_n, t_cg_mean, height_cg, 'congestus')

# plot horizontal dashed lines for LCL height in shallow and congestus clouds
ax[1].axhline(y=lcl_sl_mean * 1e-3, 
           color=COLOR_SHALLOW, 
           linestyle='--', 
           linewidth=2, 
           label='LCL Shallow')

ax[1].axhline(y=lcl_cg_mean * 1e-3,
              color=COLOR_CONGESTUS,
                linestyle='--',
                linewidth=2,
                label='LCL Congestus')

# add legend for LCL lines
ax[1].legend(loc='upper right', fontsize=16, frameon=False) 

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_c_shallow_congestus_rev.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

print('figure fig_1_c.png saved')

# plot of the profiles of T for the trajectory regions
fig, ax = plt.subplots(figsize=(8, 12))

# Color the area below the LCL height in grey
lcl_height = 0  # Assuming LCL height is at 0 km
ax.fill_between([190, 299], 
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)


ax.set_title("c) Mean temperature", 
                loc='left', 
            fontsize=font_titles,
            fontweight='black')
ax.set_xlim([290, 299])
ax.set_ylim([-0.8, 0.6])
ax.set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax.set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax.set_ylabel("Height above LCL [km]",fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xlabel('Temperature [K]', fontsize=font_size)
plot_profiles_arthus(ax, t_s, t_t, t_n, t_mean, height, "all")

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_c.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

print('figure fig_1_c.png saved')


# plot now the profiles of MR for shallow and congestus clouds
fig, ax = plt.subplots(1, 2, figsize=(16, 12), sharey=True)

# Color the area below the LCL height in grey
lcl_height = 0  # Assuming LCL height is at 0 km
ax[0].fill_between([12, 17], 
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)



ax[0].set_xlim([12, 17])
ax[0].set_ylim([-0.5, 1.])
ax[0].set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax[0].set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax[0].set_ylabel("Height above LCL [km]",fontsize=font_size)
ax[0].tick_params(axis='both', which='major', labelsize=font_size)
ax[0].set_xlabel('Mixing ratio  [gkg$^{-1}$]', fontsize=font_size)
plot_profiles_arthus(ax[0], mr_s, mr_t, mr_n, mr_mean, height, 'all')


# second subplot
ax[1].fill_between([12, 17], 
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)
ax[1].set_xlim([12, 17])
ax[1].set_ylim([-0.5, 1.])
ax[1].set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax[1].set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax[1].tick_params(axis='both', which='major', labelsize=font_size)
ax[1].set_xlabel('Mixing ratio  [gkg$^{-1}$]', fontsize=font_size)
plot_profiles_arthus(ax[1], mr_sl_s, mr_sl_s, mr_sl_n, mr_sl_mean, height_sl, 'shallow')
plot_profiles_arthus(ax[1], mr_cg_s, mr_cg_t, mr_cg_n, mr_cg_mean, height_cg, 'congestus')
# plot horizontal dashed lines for LCL height in shallow and congestus clouds
ax[1].axhline(y=lcl_sl_mean * 1e-3, 
           color=COLOR_SHALLOW, 
           linestyle='--', 
           linewidth=2, 
           label='LCL Shallow')

ax[1].axhline(y=lcl_cg_mean * 1e-3,
              color=COLOR_CONGESTUS,
                linestyle='--',
                linewidth=2,
                label='LCL Congestus')

ax[1].legend(loc='upper right', fontsize=16, frameon=False) 

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_d_shallow_congestus_rev.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)



# plot of the profiles of MR for the trajectory regions
fig, ax = plt.subplots(figsize=(8, 12))

# Color the area below the LCL height in grey
lcl_height = 0  # Assuming LCL height is at 0 km
ax.fill_between([12, 17], 
                -1, 
                lcl_height, 
                color='grey', 
                alpha=0.2)


ax.set_title("d) Mean mixing ratio", 
                loc='left', 
            fontsize=font_titles,
            fontweight='black')
ax.set_xlim([12, 17])
ax.set_ylim([-0.8, 0.6])
ax.set_yticks([-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
ax.set_yticklabels(["-0.5","-0.25", "LCL", "0.25", "0.5", "0.75", "1."])
ax.set_ylabel("Height above LCL [km]",fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xlabel('Mixing ratio  [gkg$^{-1}$]', fontsize=font_size)
plot_profiles_arthus(ax, mr_s, mr_t, mr_n, mr_mean, height, 'all')

# save figure
fig.savefig('/work/plots_rain_paper/fig_1_d.png',
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

print('figure fig_1_c.png saved')

print('mean values of LCL height for shallow clouds: ', lcl_sl_mean)
print('mean values of LCL height for congestus clouds: ', lcl_cg_mean)

