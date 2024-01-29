'''
reader for arthus raman lidar and doppler lidar data for eurec4a msm ship
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


def extract_halo_cell_profiles(file_string, time_start, time_end, var_string, var_unit):
    '''function to read Arthus and doppler lidar data and extract the mean profiles over time on each of the rain cells 
    input: file_string: string containing file name list 
            time_start: time start of the rain cell
            time_end: time end of the rain cell
    output: data_point: xarray dataset containing the mean profile

    '''
    
    
    # variable list
    var_list =['T','MR','W']

    T_dict = {
         'var_name'  : 'T',
         'var_string': 'Temperature',
         'var_units' : ' $^{\circ}$K',
         'var_min'   : 290.,
         'var_max'   : 310.,
         'thr_min'   : 280.,
         'thr_max'   : 330.,
         'avg_time'  : '15',
         'cmap'      : 'jet',
         'title'     : 'Air temperature: 28/01-04/02'}

    MR_dict = {
         'var_name'  : 'MR',
         'var_string': 'Water vapor mixing ratio',
         'var_units' : ' g kg$^{-1}$',
         'var_min'   : 0.,
         'var_max'   : 30.,
         'thr_min'   : 0.,
         'thr_max'   : 30.,
         'avg_time'  : '15',
         'cmap'      : 'jet',
         'title'     : 'Water vapor mixing ratio: 28/01-04/02'}

    W_dict = {
         'var_name'  : 'W',
         'var_string': 'Vertical velocity',
         'var_units' : ' ms$^{-1}$',
         'var_min'   : -2.,
         'var_max'   : 2.,
         'thr_min'   : -5.,
         'thr_max'   : 5.,
         'avg_time'  : '15',
         'cmap'      : 'seismic',
         'title'     : 'Vertical velocity: 28/01-04/02'}
    dict_list = [T_dict, MR_dict, W_dict]
    
    data = xr.open_dataset(np.sort(glob.glob(file_string))[0])
    # selecting cells for the data
    data_cell = data.sel(Time=slice(time_start, time_end))
    
    # filtering values out of the given range for the selected variable
    if var_string == 'MR':
        dict_var = MR_dict
    elif var_string == 'T':
        dict_var = T_dict
    else:
        dict_var = W_dict
    
    thr_min = dict_var['thr_min'] 
    thr_max = dict_var['thr_max'] 
    
    data_cell["nans"] = xr.full_like(data_cell.Product, fill_value=np.nan)
    data_cell['Product'] = xr.where(((data_cell.Product.values > thr_min) * (data_cell.Product.values < thr_max)), data_cell['Product'], data_cell["nans"])
    
    # calculating mean over time
    data_mean = data_cell.mean(dim='Time', skipna=True)
    data_std = data_cell.std(dim='Time', skipna=True)
    
    # creating output dataset
    # define data with variable attributes
    data_vars = {var_string:(['height_a'], data_mean.Product.values, 
                             {'units': var_unit, 
                              'long_name':var_string}), 
                 var_string+'_std':(['height_a'], data_std.Product.values, 
                             {'units': var_unit})}

    # define coordinates
    coords = {'height_a': (['height_a'], data_mean.Height.values)}

    # define global attributes
    attrs = {'creation_date':str(datetime.now()), 
             'author':'Claudia Acquistapace', 
             'email':'cacquist@uni-koeln.de'}

    # create dataset
    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)


    
    return(ds, data_cell)



def read_mrr_given_day(yy,mm,dd):
    """
    function to read MRR radar data
    Input
    yy, mm, dd: strings of year, month and day
    
    Returns
    -----
    xarray dataset of MRR data 
    
    """


    # reading H wind speed
    MRR_data = xr.open_mfdataset(np.sort(glob.glob('/data/obs/campaigns/eurec4a/msm/mrr/new_postprocessing/'+yy+mm+dd+'_*')))
    
    return MRR_data


def read_Hwind_speeds():
    """
    function to read horizontal wind speeds from the wind lidar data

    Returns:
    ------
        H_wind_speed_data: xarray dataset 
    """
    H_wind_speed_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/h_wind_speed/horizontal_wind_direction_msm_eurec4a_campaign.nc')

    return H_wind_speed_data