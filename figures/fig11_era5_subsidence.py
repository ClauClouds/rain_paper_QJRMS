"""
script to plot subsidence from ERA5 data at the selected heights of
500 Hpa, cloud top height around 2000 m, and LCL height around 650 m
together with COD from satellite. Data are every hour, from 14 to 18
"""


import pdb

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from readers.lcl import read_lcl
from readers.ship import read_ship
from glob import glob
from datetime import datetime
import pandas as pd
from readers.lcl import read_lcl
from metpy.calc import pressure_to_height_std
from metpy.units import units
from figures.mpl_style import CMAP, cmap_era5, cmap_wind_era5

def main():
    

    # read era5 
    ds_era5 = xr.open_dataset('/work/plots_rain_paper/era5_20200212/589d85aa1852a07ab4af850847f42b71.nc')

    # read ship trajectory
    ds_ship = read_ship()
    
    # read satellite data
    ds_sat = read_satellite_data()
    
    # select era5 between 14 and 18 hours
    ds_era5_sel = ds_era5.sel(valid_time=slice('2020-02-12T14:00:00',
                                               '2020-02-12T18:00:00'))
    # read satellite dates
    satdates, satpaths = read_satellite_data()
    
    # read lcl 
    ds_lcl = read_lcl()

    # Convert to height (in meters) 
    pressure_levels = ds_era5.pressure_level.values # in hPa
    heights = pressure_to_height_std(pressure_levels * units("hPa"))

    # remove units from heights for further processing
    heights = 1e3 * heights.magnitude  # convert to numpy array without units

    w_ds = []
    ship_ds = []
    sat_paths = []
    
    # loop on ds_era5 time steps to extract data at ship location
    for i_time, timestamp in enumerate(ds_era5_sel.valid_time.values):
        
        # find closest satdate datetime to the timestamp
        timestamp = pd.to_datetime(timestamp)
        closest_satdate = min(satdates, key=lambda x: abs(x - timestamp))
        sat_paths.append(satpaths[satdates.index(closest_satdate)])
        

        #print era5 time stamp
        print(f"Processing time: {timestamp}")
        
        # find all time stamps from the ship data that are within 1 hour of the current era5 time stamp
        era5_time = pd.to_datetime(timestamp) 
        timestamp_end = era5_time + pd.Timedelta(hours=1)

        # selecting ship data in the hour to plot ship trajectory for the hour
        ds_ship_hour = ds_ship.sel(time=slice(era5_time, timestamp_end))
        
        # select heights at which we want to look at the variables
        
        # 1) 500 hPa
        pressure_500hPa = 500  # hPa
        # find closest height to 500 hPa
        pressure_500hPa_index = np.abs(pressure_levels - pressure_500hPa).argmin()
        closest_500hPa_height = heights[pressure_500hPa_index]
        closest_500hPa_pressure = pressure_levels[pressure_500hPa_index]
        
        # 2) LCL
        # select and calculate mean lcl in the selected hour of timestamp
        lcl_height = 650  # meters
        lcl_closest_height = np.abs(heights - lcl_height).argmin()
        closest_lcl_pressure = pressure_levels[lcl_closest_height]

        
        # 3) 2000 m (corresponding to mean cloud top height in the area)
        cloud_top_height = 2000  # meters
        # find closest height to cloud top height
        cloud_top_height_index = np.abs(heights - cloud_top_height).argmin()
        closest_cloud_top_height = heights[cloud_top_height_index]
        closest_cloud_top_pressure = pressure_levels[cloud_top_height_index]
            
        # create array of selected pressures at which we want to look at the variables
        p_selected_arr = np.array([closest_500hPa_pressure,
                           closest_cloud_top_pressure, 
                           closest_lcl_pressure])
        
        p_string_arr = ['500 hPa', 'cloud top - 2000 m', 'LCL - ' + str(int(lcl_height)) + ' m']
        
        # loop on pressure to extract w and cod
        w_list = []
        
        for p_level in p_selected_arr:
            # extract w at the selected pressure level
            w = ds_era5_sel.w.sel(pressure_level=p_level, valid_time=timestamp)
            w_list.append(w)
            
        # extract cod from satellite data
        
        # convert lists to xarray DataArrays
        w_da = xr.concat(w_list, dim='pressure_level')
        print(f"Shape of w_da: {w_da.shape}")
        w_ds.append(w_da)
        ship_ds.append(ds_ship_hour)
        
    # create a new dataset with the extracted data
    # merge w_ds to a single dataset
    ds_w = xr.concat(w_ds[:], dim='valid_time')
    
    # plot figure
    plot_figure11_era5_subsidence(ds_w, ship_ds, p_selected_arr, p_string_arr, sat_paths)




 
    
def read_satellite_data():
    """    Reads satellite data 
    """
   
    sat_path = '/data/sat/goes-r-abi/ATLANTIC/2020/043/'
    
    # list all files in the directory
    satfiles = sorted(glob(sat_path +'clavrx*.nc'))

    # store date, day of year, year, hour, minute
    satdates = []
    satpaths = []    
    for ind, file in enumerate(satfiles):
        
        satdate,doy,yyyy,hh,mm = fday_of_year(file)
        satdates.append(satdate) 
        satpaths.append(file)
    

    return satdates, satpaths

        
def fday_of_year(image_files1):
    '''
       returns details of the date from file address of GOES-r-ABi files
       image_files1: file address of GOES-r-ABi files
       returns: date, day of year, year, hour, minute
    '''
    date = image_files1.split('_s')[1].split('_')[0]   # may have to be changed
    date = datetime.strptime(date, '%Y%j%H%M%S%f')
    day_of_year = date.timetuple().tm_yday
    year = date.timetuple().tm_year
    hour = date.timetuple().tm_hour
    minute = date.timetuple().tm_min
    
    return date, day_of_year, year, hour, minute


def plot_figure11_era5_subsidence(ds_w, ship_ds, pressure_levels, p_string_arr, sat_paths):
    """
    Function to plot subsidence from ERA5 data at the selected heights
    of 500 Hpa, cloud top height around 2000 m, and LCL height around 650 m
    together with COD from satellite.
    """
    start_pos_legend = [0.7, 0.5, 0.3] # left, bottom, width, height
    # reading the number of pressure levels.
    n_levels = len(pressure_levels)
    n_times = len(ds_w.valid_time)
    print(f"Number of pressure levels: {n_levels}, Number of time steps: {n_times}")
    
    # create a figure with 4 rows and 5 columns
    fig, ax = plt.subplots(4,5,
                           sharex=True, 
                           sharey=True,
                           dpi=100, 
                           figsize=(15,10))
    
    # adjust location of subplots to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.75,
                        wspace=0.2, hspace=0.2) 
    
    
    # writing the title by looping on times and writing on first row
    for i in range(n_times):
        
        ax[0,i].set_title('%i'%(i+14))
        ax[0,i].set_ylim(12,15)
        ax[0,i].set_xlim(-60, -55)
        
        # read satthpath for the hour
        sat_hour_path = sat_paths[i]
        
        # read satellite data for the hour
        ds_sat_hour = xr.open_dataset(sat_hour_path)
        cod = ds_sat_hour.cld_opd_dcomp


        #last row: satellite cod:
        for j, p_level in enumerate(pressure_levels):
            
            print(f"Plotting for pressure level: {p_level} at time {i+14}")
            print('index of pressure level:', j)
            print('index for time:', i)
            
            #plot ERA-5 subsidence:
            #at 500hPa
            ja=ax[j,i].pcolormesh(ds_w.longitude, 
                                ds_w.latitude, 
                                ds_w[i,j,:,:],
                                vmin=-0.5,
                                vmax=0.5,
                                cmap=cmap_era5)
            
            # plot ship trajectory
            jja = ax[j,i].scatter(ship_ds[i].lon,
                             ship_ds[i].lat,                 
                             color='white',
                             marker='o',
                             s=50,
                             label='Ship Trajectory')
            
            if i == n_times-1:
                cbar_ax_1 = fig.add_axes([0.8, start_pos_legend[j], 0.01, 0.15]) # left, bottom, width, height
                plt.colorbar(ja, 
                             cax=cbar_ax_1,
                             orientation='vertical',
                             label='w at '+p_string_arr[j], 
                             ax=ax[j,i])
    
    
        #plot satellite data
        ja = ax[-1,i].pcolormesh(cod.longitude,
                                 cod.latitude, 
                                 cod.values, 
                                 vmin=0, 
                                 vmax=20)
        
        # plot ship trajectory
        jjja = ax[-1,i].scatter(ship_ds[i].lon,
                             ship_ds[i].lat,                 
                             color='white',
                             marker='o',
                             s=50,
                             label='Ship Trajectory')
        ax[-1,i].set_ylim(12,15)
        ax[-1,i].set_xlim(-60, -55)
        
        if i == n_times-1:
            cbar_ax_2 = fig.add_axes([0.8, 0.1, 0.01, 0.15]) # left, bottom, width, height
            plt.colorbar(ja, 
                         cax=cbar_ax_2, 
                         ax=ax[-1,i], 
                         label='COD')


        
        #if i == n_times-1:
        #    plt.legend([jja], ['Ship Trajectory'], loc='upper right')
    
    

    # save figure
    #plt.tight_layout()
    plt.savefig(f'/work/plots_rain_paper/fig11_era5_subsidence.png', dpi=300)

    plt.close()
    return(fig, ax )    
        
if __name__ == "__main__":
    main()