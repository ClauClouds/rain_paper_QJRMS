"""
Calculate diurnal cycle of raman and doppler lidar variables 

Procedure:
- postprocess the lidar variables: removing double time stamps, applying mask to filter noise in the data, 
- calculate diurnal cycle
- store diurnal cycle in ncdf
"""

from readers.lidars import f_call_postprocessing_arthus, f_read_variable_dictionary

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
import custom_color_palette as ccp
import matplotlib as mpl 
import os.path
import itertools    
import os.path


def main():
    
    # string names of the variable to process for calculation of diurnal cycle
    var_string_arr = ['T', 'MR', 'VW']
    
    # loop on variables 
    for var_string in enumerate(var_string_arr):
        
        # read dictionary associated to the variable
        dict_var = f_read_variable_dictionary(var_string)

        # call postprocessing routine for arthus data
        data = f_call_postprocessing_arthus(var_string)

        # read diurnal cycle file
        diurnal_cycle = read_diurnal_cycle(var_string)
    
        # calculate anomaly for the selected variable
        calculate_anomaly(diurnal_cycle, data, var_string)
    
    



def calculate_anomaly(diurnal_cycle, data, var_name):
    
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
        year =  pd.to_datetime(diurnal_cycle.Time.values[0]).year
        mm = pd.to_datetime(diurnal_cycle.Time.values[0]).month
        day = pd.to_datetime(diurnal_cycle.Time.values[0]).day
        
        # building time stamp to select from the diurnal cycle matrix
        var_profile = diurnal_cycle.sel(Time=datetime(year, mm, day, HH,MM,0), method='nearest')
        
        # subtracting the diurnal cycle from the time selected column of data
        anomaly[i_time, :] = data[var_name].values[i_time,:] - var_profile.diurnal_cycle.values
        
    # adding anomaly variable to the dataset
    data = data.assign(anomaly=(['time','height'],anomaly))
    
    # storing data to ncdf
    data.to_netcdf(path_out+var_string+'_arthus_anomaly.nc',
                   encoding={var_string:{"zlib":True, "complevel":9},\
                    "anomaly": {"dtype": "f4", "zlib": True, "complevel":9}, \
                    "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    
    return data

 
    
if __name__ == "__main__":
    main()
    
    