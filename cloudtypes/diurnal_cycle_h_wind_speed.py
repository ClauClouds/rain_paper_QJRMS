"""
Calculate diurnal cycle of raman and doppler lidar variables 

Procedure:
- postprocess the lidar variables: removing double time stamps, applying mask to filter noise in the data, 
- calculate diurnal cycle
- store diurnal cycle in ncdf
"""

from readers.lidars import read_diurnal_cycle, f_call_postprocessing_arthus, f_read_variable_dictionary, read_h_wind
from path_folders import path_diurnal_cycle_arthus, path_anomaly_arthus, path_diurnal_cycle_plots
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def main():


    # call postprocessing routine for arthus data
    data = read_h_wind()
    
    diurnal_cycle = calc_h_speed_diurnal_cycle(data, path_diurnal_cycle_plots, path_diurnal_cycle_arthus)
    
    # calculate anomaly for the selected variable        
    calculate_anomaly(diurnal_cycle, data, path_anomaly_arthus)
    
    
    
def calc_h_speed_diurnal_cycle(data, path_plot, path_out):
    '''
    date: 20.08.2024
    author: Claudia Acquistapace
    input:
        data: xarray dataset containing the data over which to calculate diurnal cycle
        path_out: output path for the ncdf file containing the diurnal cycle
    goal: calculate the diurnal cycle of the input variable and save it in ncdf.
    '''
    
    
    
    # calculating the mean of the variable over the time interval requested
    data = data.resample(time='15T').mean()
    
    # re-writing time array as hh:mm for then being able to group
    data['time'] = pd.to_datetime(data.time.values).strftime("%H:%M")
    
    # grouping and calculating mean of the profiles
    grouped_mean = data.groupby('time').mean()
    
    # plot diurnal cycle
    fig2, axs = plt.subplots(1,1, figsize=(16,7), constrained_layout=True)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_linewidth(3)
    axs.spines["left"].set_linewidth(3)
    mesh1 = axs.pcolormesh(pd.to_datetime(grouped_mean['time'].values), 
                           grouped_mean['height'].values,
                           grouped_mean['H_wind_speed'].values.T,
                           vmin=0,
                           vmax=15, 
                           cmap='viridis', 
                           rasterized=True)
    cbar = fig2.colorbar(mesh1, 
                         ax=axs, 
                         label='Horizontal wind speed [ms$^{-1}$]',
                         location='right', aspect=20)
    axs.set_xlabel('Time UTC [HH:MM]')
    axs.set_ylabel('Height [m]')
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs.text(0, 1.02, 'Diurnal cycle of horizontal wind speed averaged over 15 min', \
                fontweight='black', transform=axs.transAxes)
    fig2.savefig(
    os.path.join(path_plot, 'H_wind_speed_diurnal_cycle.png'),
    dpi=300,
    bbox_inches="tight",
    transparent=True,   
    )
    plt.close()
    
    # saving diurnal cycle in ncdf file
    dims             = ['Time','Height']
    coords           = {"Time":pd.to_datetime(grouped_mean['time'].values), "Height":grouped_mean['height'].values}
    diurnal_cycle       = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['H_wind_speed'].values,\
                         attrs={'long_name':'diurnal cycle over 15 min for horizontal wind speed',
                                'units':'ms-1'})
    global_attributes = {'CREATED_BY'       : 'Claudia Acquistapace',
                        'CREATED_ON'       :  str(datetime.now()),
                        'FILL_VALUE'       :  'NaN',
                        'AUTHOR_NAME'          : 'Claudia Acquistapace',
                        'AUTHOR_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'AUTHOR_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'AUTHOR_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'diurnal cycle of the variable h wind speed calculated over 15 minutes',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Lidar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_SOURCE'      : 'arthus data',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/SST-impact/',
                        'INSTRUMENT_MODEL' : 'arthus raman lidar system',
                         'COMMENT'         : 'original data postprocessed by Diego Lange' }
    dataset    = xr.Dataset(data_vars = {'diurnal_cycle':diurnal_cycle},
                                      coords = coords,
                                       attrs = global_attributes)
    
    dataset.to_netcdf(path_out+'H_wind_speed_diurnal_cycle.nc')  
    
    return(dataset)
    



def calculate_anomaly(diurnal_cycle, data, path_out):
    
    """
    function to calculate anomalies of the selected variable with respect to the calculated diurnal cycle.
    
    Arguments:
        diurnal_cycle: xarray dataset containing the diurnal cycle for the selected variable
        data: xarray dataset containing the variable as a function of time and height
        var_name: string for the variable to be used to calculate diurnal cycle [ options are T, MR, WV]

    Returns:
        data: xarray dataset containing the original data and the added anomaly variable as a function of time and height
    """
    # calculate anomaly of temperature 
    anomaly = np.empty_like(data['H_wind_speed'].values)
    anomaly.fill(np.nan)


    for i_time, time_stamp in enumerate(data.time.values):
        
        # reading hour and minute to select from the time stamp of the actual time
        HH = pd.to_datetime(time_stamp).hour
        MM = pd.to_datetime(time_stamp).minute
        # reading year of the diurnal cycle file
        year =  pd.to_datetime(diurnal_cycle.Time.values[0]).year
        mm = pd.to_datetime(diurnal_cycle.Time.values[0]).month
        day = pd.to_datetime(diurnal_cycle.Time.values[0]).day
        
        # building time stamp to select from the diurnal cycle matrix
        var_profile = diurnal_cycle.sel(Time=datetime(year, mm, day, HH,MM,0), method='nearest')
        
        # subtracting the diurnal cycle from the time selected column of data
        anomaly[i_time, :] = data['H_wind_speed'].values[i_time,:] - var_profile.diurnal_cycle.values
        
    # adding anomaly variable to the dataset
    data = data.assign(anomaly=(['time','height'], anomaly))

    print(data)
    

    data.to_netcdf(path_out+'H_wind_speed_arthus_anomaly.nc',
                   encoding={'H_wind_speed':{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})

    return data
    
 
    
if __name__ == "__main__":
    main()
    
    
