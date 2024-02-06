'''
code to plot case study of the 12 february 2020 where a intense precipitation from a flower caused a change in the mesoscale organization 

'''
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import glob
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


from readers.ship import read_ship
from readers.lidars import read_mrr_given_day, read_Hwind_speeds, extract_halo_cell_profiles
from readers.wband import read_lwp, read_radar_single_file
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
    time_start = datetime(2020, 2, 12, 11, 0, 0)
    time_end = datetime(2020, 2, 12, 19, 0, 0)


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
    MR_transition = MR_anomaly.sel(time=slice(time_start, time_end))  
    sat_transition = sat_data.sel(time=slice(time_start, time_end))  
    
    # find transition times
    class_2 = np.where(sat_transition.classification.values == 2)[0]
    time_start_transition = sat_transition.time.values[class_2[0]]
    time_end_transition = sat_transition.time.values[class_2[-1]]
    
    
    # visualize transition in a plot: surface obs and satellite measurements
    plot_figure9(ship_transition, radar_transition, H_s_transition, LWP_transition, MRR_transition, MR_transition, time_start_transition, time_end_transition)
    visualize_satellite_composite(sat_data)
    

    
    

    
    
def plot_figure9(ship_transition, radar_transition, H_s_transition, LWP_transition, MRR_transition, MR_transition, time_start_transition, time_end_transition):
    
    time_5 = datetime(2020,2,12,11,40,0)
    time_2 = datetime(2020,2,12,16,40,0)
    
    color_2 = '#f8c78d'
    color_5 = '#4892b2'
    
    # selecting height for mrr
    # select one height 
    h_sel = 500.
    MRR_time_serie = MRR_transition.sel(height=h_sel, method='nearest')
    
    fig, axs = plt.subplots(6,1, 
                            figsize=(25,20), 
                            sharex=True, 
                            constrained_layout=True)
    

    mesh0 = axs[0].plot(LWP_transition.time.values, 
                        LWP_transition.lwp.values, 
                        linewidth=7, 
                        color='black')
    axs[0].set_ylim(0., 1000.)
    axs[0].vlines(time_5, 0, 1000., color=color_5, linewidth=8)
    axs[0].vlines(time_2, 0, 1000., color=color_2, linewidth=8)

    
    mesh00 = axs[1].pcolormesh(MR_transition.time.values, 
                        MR_transition.height.values, 
                        MR_transition.MR_anomaly.values.T,
                        linewidth=7, 
                        cmap='binary', 
                        vmin=-2, vmax=2)
 
    axs[1].set_ylim(50., 600.)
    cbar = fig.colorbar(mesh00, ax=axs[1], orientation='vertical')
    cbar.set_label('Anomaly \n Mixing Ratio [g kg$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25)   
    axs[1].set_ylabel('Height [m]', fontsize=18)  
    axs[1].vlines(time_5, 50, 600., color=color_5, linewidth=8)
    axs[1].vlines(time_2, 50, 600., color=color_2, linewidth=8)
      
    mesh2 = axs[2].plot(ship_transition.time.values, 
                        ship_transition.RH.values*100, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='ship data')
     
    mesh2 = axs[2].plot(radar_transition.time.values, 
                        radar_transition.relative_humidity.values*100, 
                        linewidth=7,
                        color='black', 
                        label='radar weather station')   
    axs[2].set_ylim(50., 90.)  
    axs[2].legend(frameon=False, loc='lower right', fontsize=18)
    axs[2].vlines(time_5, 50, 90., color=color_5, linewidth=8)
    axs[2].vlines(time_2, 50, 90., color=color_2, linewidth=8)

    mesh2 = axs[3].plot(ship_transition.time.values, 
                        ship_transition.T.values, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='ship data')

    mesh2 = axs[3].plot(radar_transition.time.values, 
                        radar_transition.air_temperature.values, 
                        linewidth=7,
                        color='black', 
                        label='radar weather station') 
    axs[3].legend(frameon=False, loc='lower right', fontsize=18)
    axs[3].vlines(time_5, 297, 302., color=color_5, linewidth=8)
    axs[3].vlines(time_2,297, 302., color=color_2, linewidth=8)
    axs[3].set_ylim(297., 302.)  
    
    mesh22 = axs[4].pcolormesh(H_s_transition.time.values, 
                              H_s_transition.height.values,
                             H_s_transition.H_wind_speed.values.T, 
                            cmap='binary', 
                            vmin=7, vmax=15.)

    cbar = fig.colorbar(mesh22, ax=axs[4], orientation='vertical')
    cbar.set_label('Horizontal wind speed [m s$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25)    
    axs[4].set_ylim(50., 600.)  
    axs[4].set_ylabel('Height [m]', fontsize=18)  
    axs[4].vlines(time_5, 50, 600., color=color_5, linewidth=8)
    axs[4].vlines(time_2, 50, 600., color=color_2, linewidth=8)    
    
    mesh3 = axs[5].scatter(MRR_time_serie.time.values, 
                           MRR_time_serie.rain_rate.values,
                           linewidth=7,
                            color='black')  
    axs[5].vlines(time_5, 0, 10., color=color_5, linewidth=8)
    axs[5].vlines(time_2, 0, 10., color=color_2, linewidth=8)
    axs[5].set_xlabel('Time [hh:mm]', fontsize=25)
    for ax, l in zip(axs[:].flatten(), ['a) LWP [g m$^{-2}$]', 'b) Mixing ratio anomaly ','c) Surface relative humidity [%]','d) Surface air temperature [C]', 'e) Horizontal wind speed [m s$^{-1}$]',  'f) Rain rate at 500 m [mmh$^{-1}$]']):
                                                   
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
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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