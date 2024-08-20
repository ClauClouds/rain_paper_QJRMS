"""
Calculate diurnal cycle of raman and doppler lidar variables 

Procedure:
- postprocess the lidar variables: removing double time stamps, applying mask to filter noise in the data, 
- calculate diurnal cycle
- store diurnal cycle in ncdf
"""

from readers.lidars import read_diurnal_cycle, f_call_postprocessing_arthus, f_read_variable_dictionary
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_anomaly_arthus, path_diurnal_cycle_plots
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    
    # string names of the variable to process for calculation of diurnal cycle
    var_string_arr = ['T', 'MR', 'VW']
    
    # loop on variables 
    for ind, var_string in enumerate(var_string_arr):

        print(var_string)
        
        # read dictionary associated to the variable
        dict_var = f_read_variable_dictionary(var_string)

        # call postprocessing routine for arthus data
        data = f_call_postprocessing_arthus(var_string)
        
        # call function to calculate diurnal cycle if diurnal cycle file does not exist otherwise read it
        if os.path.isfile(path_diurnal_cycle_arthus+var_string+'_diurnal_cycle.ncdf'):
            ds = read_diurnal_cycle(var_string)
            print('calculating diurnal cycle')
        else:
            ds = f_calc_diurnal_cycle(data, dict_var, path_diurnal_cycle_arthus, path_diurnal_cycle_plots)
        
        
        print('calculating anomalies')   
         # calculate anomaly for the selected variable        
        calculate_anomaly(ds, data, var_string, path_anomaly_arthus)
    
        
    
def f_calc_diurnal_cycle(arthus_data_interp, dict_var, path_out, path_plot):
    '''
    date: 24/11/2021
    author: Claudia Acquistapace
    input:
        arthus_data_interp: xarray dataset containing the data over which to calculate diurnal cycle
        dict_var : dictionary of settings specific of the variable of interest
        path_out: output path for the ncdf file containing the diurnal cycle
    goal: calculate the diurnal cycle of the input variable and save it in ncdf. The time resolution for the diurnal
    cycle is stored in dict_var['avg_time'] and depends on the variable to process.
    '''
    # calculating the mean of the variable over the time interval requested
    arthus_data_interp = arthus_data_interp.resample(Time=dict_var['avg_time']+'T').mean()
    
    # re-writing time array as hh:mm for then being able to group
    arthus_data_interp['Time'] = pd.to_datetime(arthus_data_interp.Time.values).strftime("%H:%M")
    
    # grouping and calculating mean of the profiles
    grouped_mean = arthus_data_interp.groupby('Time').mean()
    
    # plot diurnal cycle
    fig2, axs = plt.subplots(1,1, figsize=(16,7), constrained_layout=True)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_linewidth(3)
    axs.spines["left"].set_linewidth(3)
    mesh1 = axs.pcolormesh(pd.to_datetime(grouped_mean['Time'].values), 
                           grouped_mean['Height'].values,
                           grouped_mean['Product'].values.T,
                           vmin=dict_var['var_min'],
                           vmax=dict_var['var_max'], 
                           cmap=dict_var['cmap'], 
                           rasterized=True)
    cbar = fig2.colorbar(mesh1, 
                         ax=axs, 
                         label=dict_var['var_string']+dict_var['var_units'],
                         location='right', aspect=20)
    axs.set_xlabel('Time UTC [HH:MM]')
    axs.set_ylabel('Height [m]')
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs.text(0, 1.02, 'Diurnal cycle of '+dict_var['var_string']+' averaged over '+dict_var['avg_time']+' min', \
                fontweight='black', transform=axs.transAxes)
    fig2.savefig(
    os.path.join(path_plot, dict_var['var_string']+'_diurnal_cycle.png'),
    dpi=300,
    bbox_inches="tight",
    transparent=True,   
    )
    plt.close()
    
    # saving diurnal cycle in ncdf file
    dims             = ['Time','Height']
    coords           = {"Time":pd.to_datetime(grouped_mean['Time'].values), "Height":grouped_mean['Height'].values}
    diurnal_cycle       = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['Product'].values,\
                         attrs={'long_name':'diurnal cycle over '+dict_var['avg_time']+'min for '+dict_var['var_string'],\
                                'units':dict_var['var_units']})
    global_attributes = {'CREATED_BY'       : 'Claudia Acquistapace',
                        'CREATED_ON'       :  str(datetime.now()),
                        'FILL_VALUE'       :  'NaN',
                        'AUTHOR_NAME'          : 'Claudia Acquistapace',
                        'AUTHOR_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'AUTHOR_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'AUTHOR_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'diurnal cycle of the variable '+dict_var['var_string']+'calculated over '+dict_var['avg_time']+'minutes',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Lidar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_SOURCE'      : 'arthus data',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/SST-impact/',
                        'INSTRUMENT_MODEL' : 'arthus raman lidar system',
                         'COMMENT'         : 'original data postprocessed by Diego Lange' }
    dataset    = xr.Dataset(data_vars = {'diurnal_cycle':diurnal_cycle},
                                      coords = coords,
                                       attrs = global_attributes)
    
    dataset.to_netcdf(path_out+dict_var['var_name']+'_diurnal_cycle.nc')  
    
    return(dataset)
    



def calculate_anomaly(diurnal_cycle, data, var_name, path_out):
    
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
    anomaly = np.empty_like(data['Product'].values)
    anomaly.fill(np.nan)


    for i_time, time_stamp in enumerate(data.Time.values):
        
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
        anomaly[i_time, :] = data['Product'].values[i_time,:] - var_profile.diurnal_cycle.values
        
    # adding anomaly variable to the dataset
    data = data.assign(anomaly=(['Time','Height'], anomaly))
    
    
    
    
    # storing data to ncdf
    if var_name == 'T':
        data.to_netcdf(path_out+var_name+'_arthus_anomaly.nc',
                   encoding={'Product':{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "Time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    elif var_name == 'MR':
        data.to_netcdf(path_out+var_name+'_arthus_anomaly.nc',
                   encoding={'Product':{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "Time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    elif var_name == 'VW':
        data.to_netcdf(path_out+var_name+'_arthus_anomaly.nc',
                   encoding={'Product':{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "Time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    return data
    
 
    
if __name__ == "__main__":
    main()
    
    
