"""
Code to plot radiosonde profiles of dry intrusions and corresponding ERA5 data from different levels to explain the
origin of the air masses

"""
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
from figures.mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW, cmap_wind_era5
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from readers.lcl import read_lcl
from readers.cloudtypes import read_cloud_top_base
from readers.ship import read_ship
from metpy.calc import pressure_to_height_std
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.mpl.ticker as cticker
import iris
import iris.plot as iplt
import iris.quickplot as qplt



def main():
    
    # read radiosonde data for the two selected launches
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
    
    
    # read era5 
    ds_era5 = xr.open_dataset('/work/plots_rain_paper/era5_20200212/589d85aa1852a07ab4af850847f42b71.nc')

    # read ship trajectory
    ds_ship = read_ship()
    
    # select ship trajectory and era5 data at 19:00
    era5_time = pd.to_datetime(datetime(2020, 2, 12, 19, 0)) 
    timestamp_end = era5_time + pd.Timedelta(hours=1)

    # selecting ship data in the hour to plot ship trajectory for the hour
    ds_ship_hour = ds_ship.sel(time=slice(era5_time, timestamp_end))
    
    ds_era5_sel = ds_era5.sel(valid_time='2020-02-12T19:00:00', method='nearest')
        
    # Convert to height (in meters)
    pressure_levels = ds_era5_sel.pressure_level.values # in hPa
    heights = pressure_to_height_std(pressure_levels * units("hPa"))

    # remove units from heights for further processing
    heights = 1e3 * heights.magnitude  # convert to numpy array without units

    # select heights at which we want the era5 maps
    height_1 = 2000 # in m
    height_2 = 3000 # in m
    
    # find closest pressure levels for the selected heights
    idx_height_1 = np.abs(heights - height_1).argmin()
    idx_height_2 = np.abs(heights - height_2).argmin()
    p_1 = pressure_levels[idx_height_1]  # in hPa
    p_2 = pressure_levels[idx_height_2]  # in hPa
    
    # selecting variable fields to plot
    u_1 = ds_era5_sel.sel(pressure_level=p_1)['u'].squeeze()
    v_1 = ds_era5_sel.sel(pressure_level=p_1)['v'].squeeze()
    u_2 = ds_era5_sel.sel(pressure_level=p_2)['u'].squeeze()
    v_2 = ds_era5_sel.sel(pressure_level=p_2)['v'].squeeze()
    w_1 = ds_era5_sel.sel(pressure_level=p_1)['w'].squeeze()
    w_2 = ds_era5_sel.sel(pressure_level=p_2)['w'].squeeze()
    q_spec_1 = ds_era5_sel.sel(pressure_level=p_1)['q'].squeeze()
    q_spec_2 = ds_era5_sel.sel(pressure_level=p_2)['q'].squeeze()
    
    # convert to numpy arrays
    u_1 = u_1.values
    v_1 = v_1.values
    u_2 = u_2.values
    v_2 = v_2.values
    w_1 = w_1.values
    w_2 = w_2.values
    q_spec_1 = q_spec_1.values
    q_spec_2 = q_spec_2.values
    
    # calculating horizontal wind speed
    speed_1 = np.sqrt(u_1**2 + v_1**2)
    speed_2 = np.sqrt(u_2**2 + v_2**2)
    # calculating wind direction
    direction_1 = np.arctan2(v_1, u_1) * 180 / np.pi  # in degrees
    direction_2 = np.arctan2(v_2, u_2) * 180 / np.pi  # in degrees  
    # ensure direction is in the range [0, 360]
    direction_1 = np.mod(direction_1, 360)
    direction_2 = np.mod(direction_2, 360)  
    
    # create dataset for the ERA5 data
    ds_era5_plot = xr.Dataset(
        {
            'u_2000': (['lat', 'lon'], u_1),
            'v_2000': (['lat', 'lon'], v_1),
            'u_3000': (['lat', 'lon'], u_2),
            'v_3000': (['lat', 'lon'], v_2),
            'w_2000': (['lat', 'lon'], w_1),
            'w_3000': (['lat', 'lon'], w_2),
            'q_2000': (['lat', 'lon'], q_spec_1),
            'q_3000': (['lat', 'lon'], q_spec_2),
            'speed_2000': (['lat', 'lon'], speed_1),
            'speed_3000': (['lat', 'lon'], speed_2),

        },
        coords={
            'lat': (['lat'], ds_era5_sel.latitude.values),
            'lon': (['lon'], ds_era5_sel.longitude.values),
            'pressure_level': (['pressure_level'], pressure_levels),
        },
        attrs={
            'description': 'ERA5 data at 2000 m and 3000 m',
            'pressure_levels': pressure_levels,
            'heights': heights, # in meters
        }
    )   
    
    
    # plot figure for the paper
    plot_figure_11_rs_era5(theta_e_1, q_1, z_1,
                           theta_e_2, q_2, z_2,
                           lcl_s1.lcl.values, lcl_s2.lcl.values,
                           cb_s1, ct_s1, cb_s2, ct_s2,
                           ds_era5_plot, ds_ship_hour)
                           
    
def plot_figure_11_rs_era5(theta_e_1, 
                           q_1, 
                           z_1,
                           theta_e_2, 
                           q_2, 
                           z_2,
                           lcl_s1, 
                           lcl_s2,
                           cb_s1, 
                           ct_s1, 
                           cb_s2, 
                           ct_s2,
                           ds_era5, 
                           ds_ship_hour):
    """
    function to plot 4 subpanels: 
     - profile of theta_3 for sounding 1 and sounding 2
     - profile of q for sounding 1 and sounding 2
     - ERA5 vertical wind speed and direction at 2000 m and 3000 m
     - ERA5 specific humidity at 2000 m and 3000 m
    inputs: 
        theta_e_1: array of theta_e for sounding 1
        q_1: array of q for sounding 1
        z_1: array of heights for sounding 1
        theta_e_2: array of theta_e for sounding 2
        q_2: array of q for sounding 2
        z_2: array of heights for sounding 2
        lcl_s1: LCL for sounding 1
        lcl_s2: LCL for sounding 2
        cb_s1: cloud base for sounding 1
        ct_s1: cloud top for sounding 1
        cb_s2: cloud base for sounding 2
        ct_s2: cloud top for sounding 2
        ds_era5: xarray dataset with ERA5 data at 2000 m and 3000 m
        ds_ship_hour: xarray dataset with ship trajectory data
   
    """  
    from figures.mpl_style import cmap_era5

    
    # set max and min range for vertical wind speed
    w_min = -0.5
    w_max = 0.5  # adjust as needed for wind speed
    w1 = ds_era5['w_2000'].values  # vertical wind speed at 2000 m
    w2 = ds_era5['w_3000'].values  # vertical wind speed at 3000 m
    cbar_label = 'Vertical velocity (m/s)'
    
    # set max and min range for specific humidity and convert to g/kg
    q_min = 0
    q_max = 10  # adjust as needed for specific humidity
    q_spec_1 = ds_era5['q_2000'].values * 1000  # convert from kg/kg to g/kg
    q_spec_2 = ds_era5['q_3000'].values * 1000  # convert from kg/kg to g/kg
    
    #normalize u and v components
    u_norm_1 = ds_era5['u_2000'] / ds_era5['speed_2000']
    v_norm_1 = ds_era5['v_2000'] / ds_era5['speed_2000']  
    u_norm_2 = ds_era5['u_3000'] / ds_era5['speed_3000']
    v_norm_2 = ds_era5['v_3000'] / ds_era5['speed_3000']
    
    # read lat and lon from the dataset
    lat = ds_era5.lat.values
    lon = ds_era5.lon.values
    
    
    # Create a meshgrid for the lat/lon coordinates
    # Thin out the data for easier plotting
    skip = 2  # more dense than 5
    u_plot_1 = u_norm_1[::skip, ::skip]
    v_plot_1 = v_norm_1[::skip, ::skip]
    u_plot_2 = u_norm_2[::skip, ::skip]
    v_plot_2 = v_norm_2[::skip, ::skip]
    lat_plot = lat[::skip]
    lon_plot = lon[::skip]
    lon2d, lat2d = np.meshgrid(lon_plot, lat_plot)
    
    # set up the figure
    fig, axs = plt.subplots(2, 1, 
                            figsize=(8, 12))
    
    # Create subplots with mixed projections
    ax1 = plt.subplot(2, 1, 1)  # Regular subplot
    ax2 = plt.subplot(2, 1, 2)  # Regular subplot
    

    # FIRST COLUMN
    # first subplot: theta_e profiles
    #ax1 = axs[0, 0]
    ax1.plot(theta_e_1, z_1, label='14:15', color='black', linewidth=3)
    ax1.plot(theta_e_2, z_2, label='18:55', color='red', linestyle='--', linewidth=3)
    ax1.axhline(lcl_s1, color='black', linestyle='--', label='LCL 14:15', linewidth=2)
    ax1.axhline(lcl_s2, color='red', linestyle='--', label='LCL 18:55', linewidth=2)
    ax1.axhline(cb_s1, color='black', linestyle=':', label='CB 14:15', linewidth=2)
    ax1.axhline(ct_s1, color='black', linestyle='-.', label='CT 14:15', linewidth=2)
    ax1.axhline(cb_s2, color='red', linestyle=':', label='CB 18:55', linewidth=2)
    ax1.axhline(ct_s2, color='red', linestyle='-.', label='CT 18:55', linewidth=2)
    
    ax1.set_xlabel('$\Theta_e$ [K]', fontsize=20)
    ax1.set_ylabel('Height [m]', fontsize=20)
    ax1.set_ylim(0, 4000)  # set y limits to focus on lower troposphere
    ax1.set_xlim(310, 350)  # set x limits to focus on the range of theta_e
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
    # set fontsize of ticks
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # second subplot: q profiles
    #ax2 = axs[1, 0]
    ax2.plot(q_1, z_1, label='14:15', color='black', linewidth=3)
    ax2.plot(q_2, z_2, label='18:55', color='red', linestyle='--', linewidth=3)
    ax2.axhline(lcl_s1, color='black', linestyle='--', label='LCL - 14:15', linewidth=2)
    ax2.axhline(lcl_s2, color='red', linestyle='--', label='LCL - 18:55', linewidth=2)
    ax2.axhline(cb_s1, color='black', linestyle=':', label='CB - 14:15', linewidth=2)
    ax2.axhline(ct_s1, color='black', linestyle='-.', label='CT - 14:15', linewidth=2)
    ax2.axhline(cb_s2, color='red', linestyle=':', label='CB - 18:55', linewidth=2)
    ax2.axhline(ct_s2, color='red', linestyle='-.', label='CT - 18:55', linewidth=2)  
    
    ax2.set_xlabel('Specific Humidity [g/kg]', fontsize=20)
    ax2.set_ylabel('Height [m]', fontsize=20)
    # set fontsize of ticks
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # add legend
    #ax2.legend(fontsize=16, loc='upper right')
    ax2.set_ylim(0, 4000)  # set y limits to focus on lower troposphere
    ax2.set_xlim(0, 20)  # set x limits to focus on the range of specific humidity
    ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
    ax2.set_ylabel('Height [m]', fontsize=20)
    
    # make axis thicker
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    
    #ax1.set_title('a) Eq. pot. temp. profiles',
    #            fontweight='bold', 
    #            fontsize=16, 
    #            loc='left')
    
    #ax2.set_title('b) Spec. hum. profiles',
    ##        fontweight='bold', 
     #       fontsize=16, 
     #       loc='left')
    # save figure
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.savefig('/work/plots_rain_paper/fig11_part_a_rs_dry_intr.png',
                dpi=300, bbox_inches='tight')   
    
    
    # set up the figure
    fig2, axs = plt.subplots(nrows=2,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,8.5))

    axs = axs.flatten()
    
    # SECOND COLUMN: W at 2000 and w at 3000m
    # third subplot: ERA5 vertical wind speed and direction at 2000 m
    #ax3 = axs[0, 1]
    #ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax3 = axs[0]
    ax3.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())

    # Add features
    ax3.coastlines(resolution='110m')
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax3.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax3.add_feature(cfeature.RIVERS)

    # plot contour with wind speed
    sc = ax3.pcolormesh(lon2d,
                          lat2d,
                          w1[::skip, ::skip],
                          cmap=cmap_era5,
                          vmin=w_min,
                          vmax=w_max,
                          shading='auto',
                          transform=ccrs.PlateCarree())
    #cbar = plt.colorbar(sc, ax=ax3, orientation='horizontal', pad=0.05, aspect=50)
    ##cbar.set_label(cbar_label, fontsize=14)
    #cbar.ax.tick_params(labelsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.set_xlabel('Longitude', fontsize=12)
    # Plot wind vectors 
     # scale values: Smaller = longer arrows; try 100, 50, 20, etc.
    ax3.quiver(lon2d, 
              lat2d, 
              u_plot_1, 
              v_plot_1, 
              transform=ccrs.PlateCarree(), 
              color='black', 
              scale=20) 

    # Add ship trajectory
    ax3.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    

    # Add gridlines
    gl = ax3.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    
    #ax33 = axs[1, 1]
    #ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax33 = axs[2]
    ax33.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())

    # Add features
    ax33.coastlines(resolution='110m')
    ax33.add_feature(cfeature.BORDERS, linestyle=':')
    ax33.add_feature(cfeature.LAND, facecolor='lightgray')
    ax33.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax33.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax33.add_feature(cfeature.RIVERS)

    # plot contour with wind speed
    sc_1 = ax33.pcolormesh(lon2d,
                          lat2d,
                          w2[::skip, ::skip],
                          cmap=cmap_era5,
                          vmin=w_min,
                          vmax=w_max,
                          shading='auto',
                          transform=ccrs.PlateCarree())
    #cbar = plt.colorbar(sc, ax=ax33, orientation='horizontal', pad=0.05, aspect=50)
    ##cbar.set_label(cbar_label, fontsize=14)
    #cbar.ax.tick_params(labelsize=12)
    
    # Plot wind vectors 
     # scale values: Smaller = longer arrows; try 100, 50, 20, etc.
    ax33.quiver(lon2d, 
              lat2d, 
              u_plot_2, 
              v_plot_2, 
              transform=ccrs.PlateCarree(), 
              color='black', 
              scale=20) 

    # Add ship trajectory
    ax33.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    
    # Add gridlines
    gl = ax33.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # THIRD COLUMN: q at 2000 and q at 3000m
    # fourth subplot: ERA5 specific humidity at 2000 m and 3000 m
    #ax4 = axs[0, 2]
    #ax4 = plt.axes(projection=ccrs.PlateCarree(), ax=ax4)
    ax4 = axs[1]
    ax4.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())
    # Add features
    ax4.coastlines(resolution='110m')
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    ax4.add_feature(cfeature.LAND, facecolor='lightgray')
    ax4.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax4.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax4.add_feature(cfeature.RIVERS)
    
    # plot contour with specific humidity
    sc = ax4.pcolormesh(lon2d,
                        lat2d,          
                        q_spec_1[::skip, ::skip],
                        cmap=cmap_wind_era5,
                        vmin=q_min,
                        vmax=q_max,  # adjust as needed for specific humidity
                        shading='auto',
                        transform=ccrs.PlateCarree())
    ###cbar = plt.colorbar(sc, ax=ax4, orientation='horizontal', pad=0.05, aspect=50)
    ##cbar.set_label('Specific Humidity [kg/kg]', fontsize=14)
    #cbar.ax.tick_params(labelsize=12)


    # Add ship trajectory
    ax4.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax4.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    
    # sixth subplot: ERA5 specific humidity at 2000 m and 3000 m
    #ax44 = axs[1, 2]
    #ax44 = plt.axes(projection=ccrs.PlateCarree())
    ax44 = axs[3]
    ax44.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())
    # Add features
    ax44.coastlines(resolution='110m')
    ax44.add_feature(cfeature.BORDERS, linestyle=':')
    ax44.add_feature(cfeature.LAND, facecolor='lightgray')
    ax44.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax44.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax44.add_feature(cfeature.RIVERS)
    
    # plot contour with specific humidity
    sc = ax44.pcolormesh(lon2d,
                        lat2d,          
                        q_spec_2[::skip, ::skip],
                        cmap=cmap_wind_era5,
                        vmin=q_min,
                        vmax=q_max,  # adjust as needed for specific humidity
                        shading='auto',
                        transform=ccrs.PlateCarree())



    # Add ship trajectory
    ax44.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax44.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # set names to axes
    titles = ['b) w at 2000 m',
              'c) q at 2000 m',
              'e) w at 3000 m',
              'f) q at 3000 m']    
    #for ax in axs:
        # align the title to the left  in bold fonts
         
        
    #    ax.set_title(titles[axs.tolist().index(ax)],
    #                 fontweight='bold', 
    #                 fontsize=16, 
    #                 loc='left')
        # set the title fontsize


    

    # Adjust the location of the subplots on the page to make room for the colorbar
    #fig2.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95,
    #                    wspace=0.01, hspace=0.2)
    
    # adjust location of subplots to make room for the colorbar
    fig2.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.75,
                        wspace=0.2, hspace=0.15)    
    
    # Add a colorbar axis at the bottom of the graph
    #cbar_ax_1 = fig2.add_axes([0.2, 0.2, 0.6, 0.02]) # left, bottom, width, height
    #cbar_ax_2 = fig2.add_axes([0.2, 0.1, 0.6, 0.02])
    
    # add the colorbars at the right side of the figure
    cbar_ax_1 = fig2.add_axes([0.8, 0.25, 0.01, 0.5]) # left, bottom, width,
    # height
    cbar_ax_2 = fig2.add_axes([0.9, 0.25, 0.01, 0.5])   
    # Draw the colorbar
    cbar=fig2.colorbar(sc_1, cax=cbar_ax_1,orientation='vertical')
    cbar.set_label('vertical velocity [ms$^{-1}$]', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    cbar2=fig2.colorbar(sc, cax=cbar_ax_2,orientation='vertical')
    cbar2.set_label('Specific Humidity [gkg$^{-1}$]', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    
    # save figure
    #fig2.subplots_adjust(hspace=0.3, wspace=0.3)
    fig2.savefig('/work/plots_rain_paper/fig11_part_b_era5_dry_intr.png',
                dpi=300, bbox_inches='tight')   
    
    plt.close(fig)
    return(fig)    
    
    
      
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
 
    
if __name__ == '__main__':
    main()
    