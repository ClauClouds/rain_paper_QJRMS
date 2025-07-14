'''
code to produce a plot of specific humidity on y axis and equivalent potential temperature on 
x axis. 

This is figure 11 in the paper

'''
from readers.soundings import read_merian_soundings
from datetime import datetime, timedelta
import numpy as np
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
from glob import glob
import metpy.calc as mpcalc 
from metpy.calc import equivalent_potential_temperature
from metpy.calc import virtual_potential_temperature
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import pdb
from figures.mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from readers.lcl import read_lcl
from readers.cloudtypes import read_cloud_top_base

def main():
    
    # read radiosounding data
    ds = read_merian_soundings()
    
    # set date for searching for radiosondes
    date = '20200212'
    sound1 = 'MS-Merian__ascent__13.83_-57.20__202002121415'
    sound2 = 'MS-Merian__ascent__13.52_-57.40__202002121855'
    
    # selecting data from the two soundings
    ds_s1 = ds.sel(sounding=sound1)
    ds_s2 = ds.sel(sounding=sound2)
    
    theta_e_1, q_1, theta_v_1, z_1 = calc_paluch_quantities(ds_s1, '202002121415')
    theta_e_2, q_2, theta_v_2, z_2 = calc_paluch_quantities(ds_s2, '202002121855')
 
    
    # read time stamps for the soundings
    time_s1 = ds_s1.launch_time.values #datetime(2020,2,12, 14, 15) # 14:15 LT
    time_s2 = ds_s2.launch_time.values #datetime(2020,2,12, 18, 55) # 18:55 LT
    
    print(f'Time of sounding 1: {time_s1}'
          f'\nTime of sounding 2: {time_s2}')
    
    # read lcl data
    ds_lcl = read_lcl()
    
    # select lcl data for the two soundings
    lcl_s1 = ds_lcl.sel(time=time_s1, method='nearest')
    lcl_s2 = ds_lcl.sel(time=time_s2, method='nearest')
    
    print(f'LCL for sounding 1: {lcl_s1.lcl.values} m')
    print(f'LCL for sounding 2: {lcl_s2.lcl.values} m')
    
    # read cloud properties
    cb_ct = read_cloud_top_base()
    
    # find cloud base and cloud top for the sounding s1
    cb_ct_s1 = cb_ct.sel(time=time_s1, method='nearest')
    cb_s1 = cb_ct_s1.cloud_base.values
    ct_s1 = cb_ct_s1.cloud_top_height.values
    print(f'Cloud base for sounding 1: {cb_s1} m')
    print(f'Cloud top for sounding 1: {ct_s1} m')
    
    # find closest heights in radiosondes to mean cb and ct 
    idx_cb_s1 = np.abs(ds_s1['alt'].values - cb_s1).argmin()
    idx_ct_s1 = np.abs(ds_s1['alt'].values - ct_s1).argmin()   
    # get the values of q and theta_e at the cloud base and cloud top for both soundings
    q_cb_s1 = q_1[idx_cb_s1]
    q_ct_s1 = q_1[idx_ct_s1]
    theta_e_cb_s1 = theta_e_1[idx_cb_s1]
    theta_e_ct_s1 = theta_e_1[idx_ct_s1]

    # time difference between sounding and selected time selection
    time_diff_s1 = (time_s1 - cb_ct_s1.time.values) / np.timedelta64(1, 'm')
    print(f'Time difference for sounding 1: {time_diff_s1:.2f} minutes')
    
    # find cloud base and cloud top for the sounding s2
    # selecting only time stamps without nans and with cb >0 
    cb = cb_ct.where(~np.isnan(cb_ct.cloud_base), drop=True)
    ct = cb_ct.where(~np.isnan(cb_ct.cloud_top_height), drop=True)
    cb = cb.where(cb.cloud_base > 0, drop=True)
    ct = ct.where(ct.cloud_top_height > 0, drop=True)
    
    # reading cloud base and cloud tops for sounding s2
    cb_s2_selected = cb.sel(time=time_s2, method='nearest')
    cb_s2 = cb_s2_selected.cloud_base.values
    ct_s2_selected = ct.sel(time=time_s2, method='nearest')
    ct_s2 = ct_s2_selected.cloud_top_height.values
    print(f'Cloud base for sounding 2: {cb_s2} m')
    print(f'Cloud top for sounding 2: {ct_s2} m')
    
    # find time distance of selected times for cb and ct to s2
    time_diff_cb_s2 = (time_s2 - cb_s2_selected.time.values) / np.timedelta64(1, 'm')
    time_diff_ct_s2 = (time_s2 - ct_s2_selected.time.values) / np.timedelta64(1, 'm')
    print(f'Time difference for cloud base in sounding 2: {time_diff_cb_s2:.2f} minutes')
    print(f'Time difference for cloud top in sounding 2: {time_diff_ct_s2:.2f} minutes')
    
    # find q and theta_v for S2 cloud base and cloud top
    q_cb_s2 = q_2[np.abs(ds_s2['alt'].values - cb_s2).argmin()]
    q_ct_s2 = q_2[np.abs(ds_s2['alt'].values - ct_s2).argmin()]
    theta_e_cb_s2 = theta_e_2[np.abs(ds_s2['alt'].values - cb_s2).argmin()]
    theta_e_ct_s2 = theta_e_2[np.abs(ds_s2['alt'].values - ct_s2).argmin()]
    print(f'q at cloud base for sounding 2: {q_cb_s2:.2f} g/kg')
    print(f'q at cloud top for sounding 2: {q_ct_s2:.2f} g/kg')
    print(f'theta_e at cloud base for sounding 2: {theta_e_cb_s2:.2f} K')
    print(f'theta_e at cloud top for sounding 2: {theta_e_ct_s2:.2f} K')
    
    
    # find pair of values of q and theta_e that are closest to the LCL for both soundings
    # find the index of the closest value to the LCL for sounding 1
    idx_s1 = np.abs(ds_s1['alt'].values - lcl_s1.lcl.values).argmin()
    idx_s2 = np.abs(ds_s2['alt'].values - lcl_s2.lcl.values).argmin()
    
    # get the values of q and theta_e at the LCL for both soundings
    q_lcl_s1 = q_1[idx_s1]
    q_lcl_s2 = q_2[idx_s2]
    theta_e_lcl_s1 = theta_e_1[idx_s1]
    theta_e_lcl_s2 = theta_e_2[idx_s2]

    # print the values of q and theta_e at the LCL for both soundings
    print(f'q at LCL for sounding 1: {q_lcl_s1:.2f} g/kg')
    print(f'q at LCL for sounding 2: {q_lcl_s2:.2f} g/kg')
    print(f'theta_e at LCL for sounding 1: {theta_e_lcl_s1:.2f} K')
    print(f'theta_e at LCL for sounding 2: {theta_e_lcl_s2:.2f} K')
    

    # define colorbars for sounding 1 and sounding 2
    cmap_s1 = CMAP
    norm_s1 = mpl.colors.Normalize(vmin=0, vmax=4000.)
    cmap_s1.set_under('white')
    
    # plot profiles of theta_v and theta_e in 2 different subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 15), sharey=True)
    # set font sizes of all the plot
    fontsize = 30
    rcParams.update({'font.size': fontsize})
    marker_size = 300
    marker_other = 500
    
    axs[0].plot(theta_e_1, z_1, color='black', label='14:15 LT', linewidth=3)
    axs[0].plot(theta_e_2, z_2, color='red', label='18:55 LT', linestyle='--', linewidth=3)
    
    axs[0].set_xlabel('$\Theta_e$ [K]', fontsize=fontsize)
    axs[0].set_ylabel('Height [m]', fontsize=fontsize)
    
    axs[1].plot(theta_v_1, z_1, color='black', label='14:15 LT', linewidth=3)
    axs[1].plot(theta_v_2, z_2, color='red', label='18:55 LT', linestyle='--', linewidth=3)
    axs[1].set_xlabel('$\Theta_v$ [K]', fontsize=fontsize)
    axs[0].set_ylim(0, 5000)
    axs[1].set_ylim(0, 5000)
    axs[0].set_xlim(310, 350)
    axs[1].set_xlim(300, 320)
    
    # set fontsize of the x and y ticks
    axs[0].tick_params(axis='both', labelsize=fontsize)
    axs[1].tick_params(axis='both', labelsize=fontsize)

    fig.tight_layout()
    axs[0].legend(fontsize=fontsize)
    axs[1].legend(fontsize=fontsize)
    fig.savefig(f'/net/ostro/plots_rain_paper/fig_paquita_theta_e_theta_v_{date}.png')
    
    
    # plot figure Paluch plot
    fig, ax = plt.subplots(figsize=(14, 14))

    # set font sizes of all the plot
    fontsize = 30
    rcParams.update({'font.size': fontsize})
    marker_size = 300
    
    # Define the color boundaries and create a BoundaryNorm object
    boundaries = np.linspace(0, 4000, 250)
    norm = BoundaryNorm(boundaries, ncolors=256)
    
    ax.invert_yaxis()  # Invert y-axis to decrease from top to bottom
    ax.scatter(theta_e_1, 
               q_1,
               c=z_1, 
               marker='o',
               cmap=CMAP,
               norm=norm_s1, 
               edgecolors='black',
               label='14:15 LT', 
               s=marker_size)
    ax.scatter(theta_e_2, 
               q_2, 
               c=z_2, 
               marker='^',
               edgecolors='red',
               cmap=CMAP, 
               norm=norm_s1, 
               label='18:55 LT', 
               s=marker_size)
    
    # plot colorbars
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_s1, cmap=cmap_s1), 
                        ax=ax,  # Extend colorbar for the entire height of the figure
                        orientation='vertical', 
                        label='Altitude [m]')
    # set fontsize of the colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    
    ax.set_xlabel('Equivalent potential temperature [K]', fontsize=fontsize)
    ax.set_ylabel('Specific humidity [gkg$^{-1}$]', fontsize=fontsize)
    
    ax.legend()
    ax.set_xlim(310, 350)
    ax.set_ylim(16,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2, labelsize = 5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=24)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    fig.savefig('/net/ostro/plots_rain_paper/fig10_paluch_'+date+'.png')


    # PALUCH PLOT REVIEWERS: 
    # plot separated profiles of theta_e and q as asked by the reviewers
    fig, axs = plt.subplots(2,1, figsize=(15, 18), sharex=True)
    # plot figure Paluch plot at the two time stamps

    # set font sizes of all the plot
    fontsize = 30
    rcParams.update({'font.size': fontsize})
    marker_size = 300
    marker_other = 500
    
    # Define the color boundaries and create a BoundaryNorm object
    boundaries = np.linspace(0, 4000, 250)
    norm = BoundaryNorm(boundaries, ncolors=256)
    
    axs[0].invert_yaxis()  # Invert y-axis to decrease from top to bottom
    axs[0].scatter(theta_e_1, 
               q_1,
               c=z_1, 
               marker='o',
               cmap=CMAP,
               norm=norm_s1, 
               edgecolors='black',
               s=marker_size)

    # add lcl position at 14:15 LT
    axs[0].scatter(theta_e_lcl_s1, 
                   q_lcl_s1, 
                   marker='*',
                     color='white',
                     edgecolors='black',
                     s=marker_other,
                     label='LCL 14:15 LT')

    
    # cloud base closest to sounding S1
    axs[0].scatter(theta_e_cb_s1, 
                   q_cb_s1, 
                   marker='X',
                     color='orange',
                     edgecolors='black',
                     s=marker_other,
                     label='Cloud base closest to 14:15 LT')
    
    # cloud top closest to sounding S1
    axs[0].scatter(theta_e_ct_s1, 
                   q_ct_s1, 
                   marker='X',
                     color='red',
                     edgecolors='black',
                     s=marker_other,
                     label='Cloud top closest to 14:15 LT')
    
         
    axs[0].legend(frameon=False, fontsize=18, loc='lower left')
    axs[0].set_title('Paluch plot at 14:15 LT', fontsize=fontsize)
    
    # plot colorbars
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_s1, cmap=cmap_s1), 
                        ax=axs[0],  # Extend colorbar for the entire height of the figure
                        orientation='vertical', 
                        label='Altitude [m]')
    cbar.ax.tick_params(labelsize=fontsize)

    # secon subplot for the second time stamp
    axs[1].scatter(theta_e_2, 
               q_2, 
               c=z_2, 
               marker='o',
               edgecolors='black',
               cmap=cmap_s1, 
               norm=norm_s1,  
               s=marker_size)
    
    # add lcl position at 18:55 LT
    axs[1].scatter(theta_e_lcl_s2, 
                   q_lcl_s2, 
                   marker='*',
                     color='white',
                     edgecolors='black',
                     s=marker_other,
                     label='LCL 18:55 LT')
    

    # add cloud base closest to sounding S2
    axs[1].scatter(theta_e_cb_s2, 
                   q_cb_s2, 
                   marker='X',
                     color='orange',
                     edgecolors='black',
                     s=marker_other,
                     label='Cloud base closest to 18:55 LT')    

    # add cloud top closest to sounding S2
    axs[1].scatter(theta_e_ct_s2,
                     q_ct_s2, 
                     marker='X',
                        color='red',
                        edgecolors='black',
                        s=marker_other,
                        label='Cloud top closest to 18:55 LT')
    
    axs[1].legend(frameon=False, fontsize=18, loc='lower left')
    axs[1].set_title('Paluch plot at 18:55 LT', fontsize=fontsize)

    # plot colorbars
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_s1, cmap=cmap_s1), 
                        ax=axs[1],  # Extend colorbar for the entire height of the figure
                        orientation='vertical', 
                        label='Altitude [m]')
    cbar.ax.tick_params(labelsize=fontsize)

    # set fontsize of the colorbar
    # loop on axs to set the fontsize of the colorbar
    for axi in axs:
        axi.set_xlabel('Equivalent potential temperature [K]', fontsize=fontsize)
        axi.set_ylabel('Specific humidity [gkg$^{-1}$]', fontsize=fontsize)
        axi.set_xlim(310, 350)
        axi.set_ylim(16,1)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['bottom'].set_linewidth(2)
        axi.spines['left'].set_linewidth(2)
        axi.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        axi.tick_params(which='minor', length=5, width=2, labelsize = 5)
        axi.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        axi.tick_params(axis='both', labelsize=24)
        axi.get_xaxis().tick_bottom()
        axi.get_yaxis().tick_left()
        
    fig.savefig('/net/ostro/plots_rain_paper/fig10_paluch_'+date+'_reviewer.png')


    fig, axs = plt.subplots(2,1, figsize=(15, 18))
    # plot profiles of q and theta_e for the two time stamps 
    
    # select points for z_1 and z_2 <4000m
    ind_1 = z_1 < 4000
    ind_2 = z_2 < 4000
    z_1 = z_1[ind_1]
    z_2 = z_2[ind_2]
    theta_e_1 = theta_e_1[ind_1]
    theta_e_2 = theta_e_2[ind_2]
    q_1 = q_1[ind_1]
    q_2 = q_2[ind_2]
    
    # plot specific humidity for the two time stamps
    axs[0].plot(q_1, 
                z_1,                
                color='black', 
                label='14:15 LT',
                linewidth=4)
    axs[0].plot(q_2,
                z_2,
                color='red',
                label='18:55 LT',
                linewidth=4)
    # plot horizontal line for LCL 
    axs[0].axhline(lcl_s1.lcl.values,
                   color='black',
                   linestyle='--',
                   linewidth=2,
                   label='LCL 14:15 LT')
    axs[0].axhline(lcl_s2.lcl.values,
                   color='red',
                   linestyle='--',
                   linewidth=2,
                   label='LCL 18:55 LT')
    # plot cloud top for the two time stamps
    axs[0].axhline(ct_s1,
                     color='black',
                     linestyle=':',
                     linewidth=2,
                     label='Cloud top 14:15 LT')
    axs[0].axhline(ct_s2,
                        color='red',
                        linestyle=':',
                         linewidth=2,
                        label='Cloud top 18:55 LT')
    
    # make axis thicker
    for ax in axs:
        ax.set_ylim(0, 4000)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.tick_params(which='minor', length=5, width=2, labelsize=20)
        ax.tick_params(which='major', length=7, width=3, labelsize=20)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    
    # plot theta_e and q for the two time stamps
    # select only the heights where the difference is not nan
    ind_good_1 = ~np.isnan(theta_e_1)
    theta_e_1 = theta_e_1[ind_good_1]
    ind_good_2 = ~np.isnan(theta_e_2)

    theta_e_2 = theta_e_2[ind_good_2]
    z_1 = z_1[ind_good_1]
    z_2 = z_2[ind_good_2]

    axs[1].plot(theta_e_1.magnitude, 
                z_1,                
                color='black', 
                label='14:15 LT',
                linewidth=4)
    
    axs[1].plot(theta_e_2.magnitude,
                z_2,                
                color='red', 
                label='18:55 LT',
                linewidth=4)
       # plot horizontal line for LCL 
    axs[1].axhline(lcl_s1.lcl.values,
                   color='black',
                   linestyle='--',
                    linewidth=2,
                   label='LCL 14:15 LT')
    axs[1].axhline(lcl_s2.lcl.values,
                   color='red',
                   linestyle='--',
                   linewidth=2,
                   label='LCL 18:55 LT')
    # plot cloud top for the two time stamps
    axs[1].axhline(ct_s1,
                     color='black',
                     linestyle=':',
                     linewidth=2,
                     label='Cloud top 14:15 LT')
    axs[1].axhline(ct_s2,
                        color='red',
                        linestyle=':',
                        linewidth=2,
                        label='Cloud top 18:55 LT')
    
    axs[0].legend(loc='upper right', fontsize=18, frameon=False)
    axs[0].set_ylabel('Height [m]', fontsize=fontsize)
    axs[0].set_xlabel('Specific humidity [gkg$^{-1}$]', fontsize=fontsize)
    axs[1].set_xlabel('$\Theta_e$ [K]', fontsize=fontsize)
    axs[0].set_xlim(0, 20)
    axs[1].set_xlim(310, 350)
    axs[1].set_ylabel('Height [m]', fontsize=fontsize)

    # save plot
    fig.savefig('/net/ostro/plots_rain_paper/fig10_q_thetav_diff_'+date+'.png')
    # first subplot: 
def calc_paluch_quantities(ds, date=202002121415):
    """
    function to calculate the quantities needed for the Paluch plot
    

    Args:
        ds (xarray dataset): dataset from one radiosonde

    Returns:
        theta_e: array
        q: array
        
    """
    
    p = ds['p'].values # in Pa
    T = ds['ta'].values # in K
    Td = ds['dp'].values # in K
    z = ds['alt'].values # in m
    
    # convert in the right units
    # convert p from Pa to hpa
    p = p/100
    # convert T from K to C
    T = T - 273.15
    # convert Td from K to C
    Td = Td - 273.15

    # calculate equivalent potential temperature
    theta_e = equivalent_potential_temperature(p * units.hPa, T * units.degC, Td * units.degC)
    
    # calculate specific humidity
    q = specific_humidity_from_dewpoint(p * units.hPa, Td * units.degC).to('g/kg')
    
    # calculate virtual potential temperature
    # select values where q is not nan
    i_good = ~np.isnan(q)
    p = p[i_good]
    T = T[i_good]
    q = q[i_good]
    z = z[i_good]
    theta_e = theta_e[i_good]
    
    theta_v = virtual_potential_temperature(p * units.hPa, T * units.degC, q)  
    
    # plot the input of the function
    ##fig, ax = plt.subplots()
    #ax.plot(T, p, label='Temperature')
    #ax.plot(Td, p, label='Dewpoint')
    #ax.set_yscale('log')
    #ax.invert_yaxis()
    #ax.set_xlabel('Temperature (C)')
    #ax.set_ylabel('Pressure (hPa)')
    #ax.legend()
    #fig.savefig('/net/ostro/plots_rain_paper/profiles_rs_'+date+'.png')
    
    
    return theta_e, q, theta_v, z
    




if __name__ == "__main__":
    main()
