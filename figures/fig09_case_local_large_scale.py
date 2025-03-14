'''
code to plot case study of the 12 february 2020 where a intense precipitation from a flower caused a change in the mesoscale organization 

'''
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import pdb
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
import matplotlib as mpl
import os.path
import itertools
import os.path
from matplotlib import pyplot as plt, dates
from figures.mpl_style import CMAP, CMAP_an
from PIL import Image, ImageDraw
from matplotlib.gridspec import GridSpec
from mpl_style import CMAP, CMAP_an, COLOR_CONGESTUS, COLOR_SHALLOW

from readers.ship import read_ship
from readers.lidars import read_mrr_given_day, read_Hwind_speeds, extract_halo_cell_profiles
from readers.radar import read_lwp, read_radar_single_file
from readers.sat_data import read_satellite_classification
from readers.lidars import read_anomalies


def main():
    
    
    # setting case study time information
    date = '20200212'
    yy = '2020'
    mm = '02'
    dd = '12'
    #time_start = datetime(2020, 2, 12, 16, 15, 0)
    #time_end = datetime(2020, 2, 12, 16, 40, 0)
    time_start = datetime(2020, 2, 12, 11, 0, 1)
    time_end = datetime(2020, 2, 12, 18, 59, 59)


    # read input data
    ship_data = read_ship()
    LWP_IWV_data = read_lwp()
    radarData = read_radar_single_file(yy,mm,dd)
    MRR_data = read_mrr_given_day(yy,mm,dd)
    H_wind_speed_data = read_Hwind_speeds()
    
    # read mixing ratio anomalies
    MR_anomaly = read_anomalies('MR')

    sat_data = read_satellite_classification()

    
    # selecting data for the specified transition time interval
    LWP_transition = LWP_IWV_data.sel(time=slice(time_start, time_end))
    radar_transition = radarData.sel(time=slice(time_start, time_end))
    #flags_transition = flag_data.sel(time=slice(time_start, time_end))   
    ship_transition = ship_data.sel(time=slice(time_start, time_end))   
    MRR_transition = MRR_data.sel(time=slice(time_start, time_end))   
    H_s_transition = H_wind_speed_data.sel(time=slice(time_start, time_end))   
    MR_transition = MR_anomaly.sel(Time=slice(time_start, time_end))  
    sat_transition = sat_data.sel(time=slice(time_start, time_end))  
    
    # find transition times
    class_2 = np.where(sat_transition.classification.values == 2)[0]
    time_start_transition = sat_transition.time.values[class_2[0]]
    time_end_transition = sat_transition.time.values[class_2[-1]]
    
    
    # visualize transition in a plot: surface obs and satellite measurements
    plot_figure9(ship_transition, radar_transition, H_s_transition, LWP_transition, MRR_transition, MR_transition, time_start_transition, time_end_transition)
    visualize_satellite_composite(sat_data)
    

    
    

    
    
def plot_figure9(ship_transition, radar_transition, H_s_transition, LWP_transition, MRR_transition, MR_transition, time_start_transition, time_end_transition):
    
    
    # set all fonts to 20
    rcParams.update({'font.size': 25})
    rcParams.update({'axes.labelsize': 25})
    rcParams.update({'xtick.labelsize': 20})
    rcParams.update({'ytick.labelsize': 20})
    rcParams.update({'legend.fontsize': 20})

    # set y axis labels size
    # define timestamps useful for plotting
    time_5 = np.datetime64('2020-02-12T11:40:00')
    time_2 = np.datetime64('2020-02-12T16:40:00')
    time_transition = np.datetime64('2020-02-12T15:00:00')
    time_start = np.datetime64('2020-02-12T11:00:01')
    time_end = np.datetime64('2020-02-12T18:59:59')
    color_2 = '#f8c78d'
    color_5 = '#4892b2'

    # Convert timedelta to seconds
    time_interval_5 = (time_transition - time_start).astype('timedelta64[s]').astype(int)
    time_interval_2 = (time_end_transition - time_transition).astype('timedelta64[s]').astype(int)

    
    # selecting height for mrr
    # select one height 
    h_sel = 500.
    MRR_time_serie = MRR_transition.sel(height=h_sel, method='nearest')
    
    fig = plt.figure(figsize=(25,28), layout="constrained")
    # set that x axis is shared
    gs = GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], figure=fig)

    # first suplot: LWP and rain rates
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(LWP_transition.time.values, 
                        LWP_transition.lwp.values, 
                        linewidth=5,
                        label='LWP',
                        color='black')
    ax0.set_ylim(0., 1000.)
    ax0.vlines(time_5, 0, 1000., color=color_5, linewidth=8, linestyle=':')
    ax0.vlines(time_2, 0, 1000., color=color_2, linewidth=8, linestyle=':')
    ax0.grid(False)

    # add second y-axis for plotting rain rate
    ax00 = ax0.twinx()
    ax00.plot(MRR_time_serie.time.values, 
                     MRR_time_serie.rain_rate.values, 
                     linewidth=5, 
                     color='blue',
                     label='RR at 500 m')
    
    # set tick labels font size on ax00
    ax00.tick_params(axis='y', labelsize=20)
    ax00.grid(False)
    ax00.vlines(time_5, 0, 10., color=color_5, linewidth=8, linestyle=':')
    ax00.vlines(time_2, 0, 10., color=color_2, linewidth=8, linestyle=':')
    ax00.set_xlabel('Time [hh:mm]', fontsize=25)
    
    # add legend getting labels from ax0 and ax00
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax00.get_legend_handles_labels()
    ax00.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
    ax0.set_ylabel('LWP [g m$^{-2}$]', fontsize=18) 
    ax00.set_ylabel('Rain rate at 500 m [mmh$^{-1}$]', fontsize=18)
    
    
    # second subplot: mixing ratio anomaly
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    mesh00 = ax1.pcolormesh(MR_transition.Time.values, 
                        MR_transition.Height.values, 
                        MR_transition.anomaly.values.T,
                        linewidth=7, 
                        cmap=CMAP_an, 
                        vmin=-2, vmax=2)
 
    ax1.set_ylim(250., 700.)
    cbar = fig.colorbar(mesh00, ax=ax1, orientation='vertical', aspect=10, pad=0.005)
    cbar.set_label('Anomaly \n Mixing Ratio [g kg$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25)   
    ax1.set_ylabel('Height [m]', fontsize=18)  
    ax1.vlines(time_5, 50, 600., color=color_5, linewidth=8, linestyle=':')
    ax1.vlines(time_2, 50, 600., color=color_2, linewidth=8, linestyle=':')
    
    # third subplot: horizontal wind speed
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    mesh22 = ax2.pcolormesh(H_s_transition.time.values, 
                              H_s_transition.height.values,
                             H_s_transition.H_wind_speed.values.T, 
                            cmap=CMAP, 
                            vmin=7, vmax=15.)

    cbar = fig.colorbar(mesh22, ax=ax2, orientation='vertical', aspect=10, pad=0.005)
    cbar.set_label('Horizontal \n wind speed [m s$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25)    
    ax2.set_ylim(250., 700.)  
    ax2.set_ylabel('Height [m]', fontsize=18)  
    ax2.vlines(time_5, 50, 600., color=color_5, linewidth=8, linestyle=':')
    ax2.vlines(time_2, 50, 600., color=color_2, linewidth=8, linestyle=':')
    
    
    # fourth subplot: surface relative humidity 
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    mesh2 = ax3.plot(ship_transition.time.values, 
                        ship_transition.RH.values*100, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='ship data')
     
    mesh2 = ax3.plot(radar_transition.time.values, 
                        radar_transition.relative_humidity.values*100, 
                        linewidth=7,
                        color='black', 
                        label='radar weather station')   
    ax3.set_ylim(50., 90.)  
    ax3.legend(frameon=False, loc='lower right', fontsize=18)
    ax3.vlines(time_5, 50, 90., color=color_5, linewidth=8, linestyle=':')
    ax3.vlines(time_2, 50, 90., color=color_2, linewidth=8, linestyle=':')

    # fifth subplot: surface air temperature
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    mesh2 = ax4.plot(ship_transition.time.values, 
                        ship_transition.T.values, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='ship data')

    mesh2 = ax4.plot(radar_transition.time.values, 
                        radar_transition.air_temperature.values, 
                        linewidth=7,
                        color='black', 
                        label='radar weather station') 
    ax4.legend(frameon=False, loc='lower right', fontsize=18)
    ax4.vlines(time_5, 297, 302., color=color_5, linewidth=8, linestyle=':')
    ax4.vlines(time_2,297, 302., color=color_2, linewidth=8, linestyle=':')
    ax4.set_ylim(297., 302.)  

    
    
    # add subplot for satellite images classification

    # create an array of the same lenfht as time array 
    ax5 = fig.add_subplot(gs[5, 0], sharex=ax0)

    ax5.barh(5, 
        time_interval_5, 
        height=1, 
        left=time_start, 
        color=color_5, 
        label='Class 5',
        edgecolor='black', 
        linewidth=1.5)

    # plot horizontal bar from time_transition to time_end
    ax5.barh(2, 
             time_interval_2, 
            height=1, 
            left=time_transition, 
            color=color_2, 
            label='Class 2',
            edgecolor='black', 
            linewidth=1.5)
    
    ax5.set_ylim(0, 7)
    # define y ticks
    ax5.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax5.set_yticklabels(['cl 1', 'cl 2', 'cl 3', 'cl 4', 'cl 5', 'cl 6', 'cl 7'], fontsize=20)
    ax5.set_xlabel('Time [hh:mm]', fontsize=25)  


    for ax, l in zip([ax0,ax1,ax2,ax3,ax4,ax5], ['a) LWP [g m$^{-2}$] and Rain rate at 500 m [mmh$^{-1}$]', 'b) Mixing ratio anomaly ',  'c) Horizontal wind speed [m s$^{-1}$]', 'd) Surface relative humidity [%]','e) Surface air temperature [C]', 'f) Satellite class.']):
                                                   
        ax.text(-0.05, 1.07, l,  fontweight='black', fontsize=25, transform=ax.transAxes)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(3)
        ax.spines["left"].set_linewidth(3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.tick_params(which='minor', length=5, width=2, labelsize = 5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.tick_params(axis='both', labelsize=20)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlim(time_start, time_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        
        
    # adjust subplots
    #fig.subplots_adjust(hspace=1., wspace=1.)
    #fig.tight_layout()

    fig.savefig('/work/plots_rain_paper/figure09_time_series.png', transparent=True)



def visualize_satellite_composite(sat_data):
    """
    figure with 2 satellite images next to each other, one from class 5 and one from class 2.
    Args:
        ds (_type_): _description_
    """
    # path to images
    path_images = '/work/plots_rain_paper/transition_plot/feb122020/043/'
    
    # creating images with crosses in the mittle
    im_new_list = []
    for i_time in range(len(sat_data.time.values)-1):
        
        # editing the image
        imNew = Image.open(path_images+sat_data.file_names.values[i_time]+'_ship.tif')
        draw = ImageDraw.Draw(imNew)
        #font = ImageFont.truetype("sans-serif.ttf", 16)
        draw.line(((256/2-10, 256/2-10), (256/2+10, 256/2+10)), fill=(255, 255, 255))
        draw.line(((256/2+10, 256/2-10), (256/2-10, 256/2+10)), fill=(255, 255, 255))
        im_new_list.append(imNew)

        if i_time == 2 or i_time == 21:
            print(sat_data.time.values[i_time])

            
    fig, axs = plt.subplots(1,2, figsize=(25,10), sharex=True, constrained_layout=True)
    
    axs[0].imshow(im_new_list[2])
    axs[1].imshow(im_new_list[21])
    axs[0].axis('off')
    axs[1].axis('off')
    
    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure09_class5_class2.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    

    plt.close()









    
if __name__ == "__main__":
    main()