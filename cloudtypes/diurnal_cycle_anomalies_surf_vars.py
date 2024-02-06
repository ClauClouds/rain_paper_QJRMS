
from readers.wband import read_radar_multiple, read_weather_data
from readers.cloudtypes import read_rain_flags, read_cloud_class
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from readers.mrr import read_rain_cell_mrr_filelist, read_mrr_dsd
from figures.mpl_style import CMAP
from readers.lidars import f_read_variable_dictionary
import pandas as pd
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os.path

def main():

    # setting output path
    path_out = '/work/4sibylle/surface_diurnal_cycles/'
    filename= path_out+'_surface_diurnal_cycle.nc'
    # reading weather station on the radar
    surface_station_data = read_weather_data()    
    
    
    # calculate diurnal cycle of surface variables
    if os.path.isfile(filename): 
        surf_diurnal_cycle = xr.open_dataset(filename)
    else:
        surf_diurnal_cycle = calc_diurnal_cycle(surface_station_data, path_out)
    
    
    # plot diurnal cycles of the variables
    #plot_diurnal_surface_vars(surf_diurnal_cycle)
    
    # read mrr file list for eac
    #mrr_data_all = read_mrr_dsd()
    #print('mrr data read in')
    
    # calculate anomalies for all variables
    #T_anomaly = calculate_anomaly(surf_diurnal_cycle, surface_station_data, 'air_temperature', path_out)
    #RH_anomaly = calculate_anomaly(surf_diurnal_cycle, surface_station_data, 'relative_humidity', path_out)
    HWIND_anomaly = calculate_anomaly(surf_diurnal_cycle, surface_station_data, 'wind_speed',path_out)

    print(T_anomaly)
    # read cloud classification
    #all_flags = read_rain_flags()

    # read shape from classification original file
    #cloud_properties = read_cloud_class()  
    ##cloud_properties['flag_rain_ground'] = (('time'), all_flags.flag_rain_ground.values)
    #cloud_properties['flag_rain'] = (('time'), all_flags.flag_rain.values)


    # selecting virga data (rain not reaching the ground) and rain data (reaching the ground)
    #ind_prec = np.where(np.logical_or(cloud_properties.flag_rain_ground.values == 1, cloud_properties.flag_rain.values == 1))[0]
    #prec = cloud_properties.isel(time=ind_prec)


def calc_diurnal_cycle(surface_station_data, path_out, avg_time='15'):
    '''
    function to calculate diurnal cycle for surface data

    '''
    # calculating the mean of the variable over the time interval requested
    surface_station_data = surface_station_data.resample(time=avg_time+'T').mean()
    # re-writing time array as hh:mm for then being able to group
    surface_station_data['time'] = pd.to_datetime(surface_station_data.time.values).strftime("%H:%M")
    # grouping and calculating mean of the profiles
    grouped_mean = surface_station_data.groupby('time').mean()
    
    
    # storing diurnal cycle in ncdf file
    dims             = ['time']
    coords           = {"time":pd.to_datetime(grouped_mean['time'].values)}
    diurnal_cycle_T  = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['air_temperature'].values,\
                         attrs={'long_name':'diurnal cycle over '+avg_time+'min for '+'surface air temperature',\
                                'units':'K'})
    diurnal_cycle_RH  = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['relative_humidity'].values,\
                         attrs={'long_name':'diurnal cycle over '+avg_time+'min for '+'surface relative humidity',\
                                'units':'%'})   
    diurnal_cycle_Hwind  = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['wind_speed'].values,\
                         attrs={'long_name':'diurnal cycle over '+avg_time+'min for '+'surface wind speed',\
                                'units':'ms-1'})     
    global_attributes = {'CREATED_BY'       : 'Claudia Acquistapace',
                        'CREATED_ON'       :  str(datetime.now()),
                        'FILL_VALUE'       :  'NaN',
                        'AUTHOR_NAME'          : 'Claudia Acquistapace',
                        'AUTHOR_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'AUTHOR_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'AUTHOR_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'diurnal cycle of the surface weather station variables calculated on '+avg_time+'minutes',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - weather station on wband radar',
                        'DATA_GROUP'       : 'Experimental;Moving',
                        'DATA_SOURCE'      : 'wband radar data',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/rain_paper_QJRSM',
                        'INSTRUMENT_MODEL' : '',
                         'COMMENT'         : 'original data postprocessed by Claudia Acquistapace' }
    dataset    = xr.Dataset(data_vars = {'air_temperature_dc':diurnal_cycle_T, 
                                         'relative_humidity_dc':diurnal_cycle_RH, 
                                         'wind_speed_dc': diurnal_cycle_Hwind},
                                       coords = coords,
                                       attrs = global_attributes)
    # storing data to ncdf
    dataset.to_netcdf(path_out+'_surface_diurnal_cycle.nc',
                   encoding={'air_temperature_dc':{"zlib":True, "complevel":9},\
                    "relative_humidity_dc": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "wind_speed_dc": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    
    return dataset





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
    anomaly = np.empty_like(data[var_name].values)
    anomaly.fill(np.nan)


    for i_time, time_stamp in enumerate(data.time.values):
        
        # reading hour and minute to select from the time stamp of the actual time
        HH = pd.to_datetime(time_stamp).hour
        MM = pd.to_datetime(time_stamp).minute
        # reading year of the diurnal cycle file
        year =  pd.to_datetime(diurnal_cycle.time.values[0]).year
        mm = pd.to_datetime(diurnal_cycle.time.values[0]).month
        day = pd.to_datetime(diurnal_cycle.time.values[0]).day
        
        # building time stamp to select from the diurnal cycle matrix
        var_profile = diurnal_cycle.sel(time=datetime(year, mm, day, HH,MM,0), method='nearest')
                
        # subtracting the diurnal cycle from the time selected column of data
        anomaly[i_time] = data[var_name].values[i_time] - var_profile[var_name+'_dc'].values
        
    # adding anomaly variable to the dataset
    data = data.assign(anomaly=(['time'],anomaly))
    
    # storing data to ncdf
    data.to_netcdf(path_out+var_name+'_surface_anomaly.nc',
                   encoding={var_name:{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    
    return data



def  plot_diurnal_surface_vars(surf_diurnal_cycle):
    '''
    plotting diurnal cycles'''

    fig, axs = plt.subplots(3,1, 
                            figsize=(15,10), 
                            sharex=True, 
                            constrained_layout=True)
    

    mesh0 = axs[0].plot(surf_diurnal_cycle.time.values, 
                        surf_diurnal_cycle.air_temperature_dc.values, 
                        linewidth=7, 
                        color='black')
    axs[0].set_ylim(297., 302.)  

    mesh2 = axs[1].plot(surf_diurnal_cycle.time.values, 
                        surf_diurnal_cycle.relative_humidity_dc.values*100, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='RH')
    axs[1].set_ylim(50., 90.)  

    mesh2 = axs[2].plot(surf_diurnal_cycle.time.values, 
                        surf_diurnal_cycle.wind_speed_dc.values*100, 
                        linewidth=7,
                        linestyle='--',
                        color='grey',
                        label='RH')
    
    axs[2].legend(frameon=False, loc='lower right', fontsize=18)

    for ax, l in zip(axs[:].flatten(), ['a) ', 'b)  ','c)']):
                                                   
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

    fig.savefig('/work/plots_rain_paper/surface_diurnal_cycles.png', transparent=True)
    plt.close()
    
    
    
if __name__ == "__main__":
    main()