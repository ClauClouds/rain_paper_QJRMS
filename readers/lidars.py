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


def read_h_wind():
    """
    function to read horizontal wind speed 
    
    """
    data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/h_wind_speed/horizontal_wind_direction_msm_eurec4a_campaign.nc')
    return(data)


def read_diurnal_cycle(var_string):
    """function to read diurnal cycle of the given input var

    Args:
        var_string (string): input var
    """
    from cloudtypes.path_folders import path_diurnal_cycle_arthus
    data = xr.open_dataset(path_diurnal_cycle_arthus+var_string+'_diurnal_cycle.ncdf')
    return(data)

    


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


def f_read_variable_dictionary(var_str):

    '''
    function to reach the dictionary associated to an Arthus variable containing the max/min temperature and the units
    possible variable inputs: var_list =['T','MR','VW','LHF','SHF']
    
    Arguments:
    var_str [string] : string identifying variable to be processed 
    
    '''    
    print('processing '+var_str)
    
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

    WVMR_dict = {
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

    VW_dict = {
         'var_name'  : 'VW',
         'var_string': 'Vertical velocity',
         'var_units' : ' ms$^{-1}$',
         'var_min'   : -2.,
         'var_max'   : 2.,
         'thr_min'   : -5.,
         'thr_max'   : 5.,
         'avg_time'  : '15',
         'cmap'      : 'seismic',
         'title'     : 'Vertical velocity: 28/01-04/02'}
         
         
    VW_dl_dict = {
         'var_name'  : 'VW_dl',
         'var_string': 'Vertical velocity dl',
         'var_units' : ' ms$^{-1}$',
         'var_min'   : -2.,
         'var_max'   : 2.,
         'thr_min'   : -5.,
         'thr_max'   : 5.,
         'avg_time'  : '15',
         'cmap'      : 'seismic',
         'title'     : 'Vertical velocity: 28/01-04/02'}     

    LHF_dict = {
         'var_name'  : 'LHF',    
         'var_string': 'Latent heat flux',
         'var_units' : ' W m$^{-2}$',
         'var_min'   : -250.,
         'var_max'   : 250.,
         'thr_min'   : -250.,
         'thr_max'   : 250.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Latent heat flux: 28/01-04/02'}

    SHF_dict = {
         'var_name'  : 'SHF',
         'var_string': 'Sensible heat flux',
         'var_units' : ' W m$^{-2}$',
         'var_min'   : -100.,
         'var_max'   : 100.,
         'thr_min'   : -100.,
         'thr_max'   : 100.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Sensible heat flux: 28/01-04/02'}

    Ze_dict = {
         'var_name'  : 'Ze',
         'var_string': 'Radar reflectivity',
         'var_units' : 'dB',
         'var_min'   : -50.,
         'var_max'   : 10.,
         'thr_min'   : -50.,
         'thr_max'   : 10.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Radar reflectivity'}    
    
    Vd_dict = {
         'var_name'  : 'Vd',
         'var_string': 'Mean Doppler velocity',
         'var_units' : 'ms$^{-1}$',
         'var_min'   : -10.,
         'var_max'   : 10.,
         'thr_min'   : -10.,
         'thr_max'   : 10.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Mean Doppler velocity'}  
    
    Sw_dict = {
         'var_name'  : 'Sw',
         'var_string': 'Spectral width',
         'var_units' : 'ms$^{-1}$',
         'var_min'   : -10.,
         'var_max'   : 10.,
         'thr_min'   : -10.,
         'thr_max'   : 10.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Spectral width'} 
    
    H_speed_dict = {
         'var_name'  : 'H_wind_speed',    
         'var_string': 'Horizontal wind speed',
         'var_units' : ' m s$^{-1}$',
         'var_min'   : 0.,
         'var_max'   : 20.,
         'thr_min'   : 0.,
         'thr_max'   : 20.,
         'avg_time'  : '30',
         'cmap'      : 'jet',
         'title'     : 'Horizontal wind speed: 28/01-04/02'}
    
    
    if var_str == 'radar_reflectivity':  
        dict_out = Ze_dict
    if var_str == 'MR':  
        dict_out = WVMR_dict
    if var_str == 'VW':  
        dict_out = VW_dict
    if var_str == 'VW_dl': 
        print('im here') 
        dict_out = VW_dl_dict
    if var_str == 'LHF':  
        dict_out = LHF_dict
    if var_str == 'SHF':  
        dict_out = SHF_dict
    if var_str == 'T':
        dict_out = T_dict
    if var_str == 'mean_doppler_velocity':
        dict_out = Vd_dict
    if var_str == 'spectral_width':
        dict_out = Sw_dict
    if var_str == 'H_wind_speed':
        dict_out = H_speed_dict
               
    return(dict_out)
    

def f_closest(array,value):
    '''
    # closest function
    #---------------------------------------------------------------------------------
    # date :  16.10.2017
    # author: Claudia Acquistapace
    # goal: return the index of the element of the input array that in closest to the value provided to the function
    '''
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx  



def f_clean_lidar_signal_from_noise_v1(data_path, dict_var, noise_mask_file='/net/ostro/4sibylle/diurnal_cycle_arthus/noise_mask.nc'):
    
    '''
    date: 24/11/2021
    author: Claudia Acquistapace
    goal: apply to the data the following filters:
    - filter to nans the values out of the threshold values (max/min values for the input variable)
    - remove signal above cloud base (because there the lidar cannot see)
    input: 
        data_path: string containing the path to the ncdf files of the arthus lidar system
        dict_var : dictionary of settings specific of the variable of interest 
        noise_mask_file: filename including the path of the noise_mask file containing the cloud base height used for the filtering
    output: arthus_data_interp. Xarray dataset containing the variables without the noise
    note: this version of the function (v1) reads the entire dataset at once.
    For a function working on a single day, check V2
    '''
        
    global arthus_files
    if (dict_var['var_name'] == 'SHF'):
        arthus_files = np.sort(glob.glob(data_path+'*.cdf'))
    if (dict_var['var_name'] == 'LHF'):
        arthus_files = np.sort(glob.glob(data_path+'*.cdf'))
    if (dict_var['var_name'] == 'VW'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))
    if (dict_var['var_name'] == 'VW_dl'):
        arthus_files = np.sort(glob.glob(data_path+'*_HALO_dt_ds_av.cdf'))
    if (dict_var['var_name'] == 'T'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))
    if (dict_var['var_name'] == 'MR'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))

    # merging data from each day in a single xarray dataset
    #print('arthus file ist:', arthus_files)
    arthus_data = xr.open_mfdataset(arthus_files, combine='nested', concat_dim='Time')

    # removing time duplicates for LHF and SHF file
    #if (dict_var['var_name'] == 'LHF') | (dict_var['var_name'] == 'SHF'):
    _, index = np.unique(arthus_data['Time'], return_index=True)
    arthus_data = arthus_data.isel(Time=index)
        
        
    # set to nan the values out of the thresholds for the selected variable
    mask = (arthus_data["Product"].values > dict_var['thr_min']) & (arthus_data["Product"].values < dict_var['thr_max'])
    arthus_data["nans"] = xr.full_like(arthus_data.Product, fill_value=np.nan)
    arthus_data['Product'] = xr.where(mask, arthus_data['Product'], arthus_data["nans"])
    
    
    # opening noise mask file to read cloud base
    noise_mask_file = '/net/ostro/4sibylle/diurnal_cycle_arthus/noise_mask.nc'
    noise_mask = xr.open_dataset(noise_mask_file)
    cloud_base = noise_mask.cloud_base_height.values


    # interpolating time of the arthus product (T,WVMR, LHF, SHF) on the time of the noise mask (BR ratio time res)
    arthus_data_interp = arthus_data.interp(Time=noise_mask['Time'].values)

    # find closest height of arthus data to the threshold height for every time stamp
    arthus_height_thr = []
    for ind_t in range(len(arthus_data_interp.Time.values)):
        arthus_height_thr.append(arthus_data_interp['Height'].values[f_closest(arthus_data_interp['Height'].values, cloud_base[ind_t])])    


    # building noise mask for arthus_data
    mask = np.ones((len(pd.to_datetime(arthus_data_interp['Time'].values)),len(arthus_data_interp['Height'].values)))
    for ind in range(len(pd.to_datetime(arthus_data_interp['Time'].values))):    
        ind_zero = np.where(arthus_data_interp['Height'].values > arthus_height_thr[ind])
        mask[ind,ind_zero[0]] = 0.
        

    # applying the mask to the product variable
    arthus_data_interp["nans"] = xr.full_like(arthus_data_interp.Product, fill_value=np.nan)
    arthus_data_interp['Product'] = xr.where(mask, arthus_data_interp['Product'], arthus_data_interp["nans"])
    
    return(arthus_data)
    
    
    

def f_call_postprocessing_arthus(var_string):
    '''
    function to call: 
    - creation of a dictionary of parameters for the selected variable to read (f_read_variable_dictionary)
    - clean the arthus data using f_clean_lidar_signal_from_noise_v1
    and returns the arthus dataset with time and height with small capital letters and the product specified as variable
    '''   
    
    # calling function to create dictionary for the selected variable
    dict_var = f_read_variable_dictionary(var_string)
    print(dict_var)
    
    # calling function to remove noise from Arthus radar data
    data_path = '/data/obs/campaigns/eurec4a/msm/arthus_dl/'+var_string+'/'
    data = f_clean_lidar_signal_from_noise_v1(data_path, dict_var)
    
    # rename time and height with small letters
    #data_new = data.rename({'Time': 'time','Height':'height','Product':var_string})

    return(data)




 
def f_read_merian_data(input_data_string, var_string_arthus='T'):
    
    
    # path to arthus data
    data_path = '/data/obs/campaigns/eurec4a/msm/arthus_dl/'+var_string_arthus+'/'
    if (var_string_arthus == 'LHF'):
        arthus_files = np.sort(glob.glob(data_path+'*.cdf'))
    if (var_string_arthus == 'VW'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))
    if (var_string_arthus == 'T'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))
    if (var_string_arthus == 'MR'):
        arthus_files = np.sort(glob.glob(data_path+'*_gr_10s_50m.cdf'))
        
        
    #print(arthus_files)
    # finding on which dates we have arthus data
    dates_list = []
    len_path = len(data_path)
    for ind_list in range(len(arthus_files)):
        dates_list.append(arthus_files[ind_list][len_path:len_path+8])
    
    if input_data_string == 'ARTHUS':
        if (var_string_arthus == 'MR'):
            data = f_call_postprocessing_arthus(var_string='MR')
        if (var_string_arthus == 'T'):
            data = f_call_postprocessing_arthus('T')
        if (var_string_arthus == 'VW'):
            data = f_call_postprocessing_arthus('VW')
        if (var_string_arthus == 'LHF'):
            data = f_call_postprocessing_arthus('LHF')


    
    
    return(data, dates_list)       
       
def read_anomalies_and_rename(measurement):
        
    """
    function to read arthus anomalies files
    """
    data = xr.open_dataset('/net/ostro/4sibylle/diurnal_cycle_arthus/5_mins/anomalies/'+measurement + '_arthus_anomaly.nc')    

    # rename time and height with small letters and anomaly with the variable name
    data_new = data.rename({'Time': 'time','Height':'height','anomaly':measurement+'_anomaly'})

    return data_new

    
def read_anomalies(measurement):
    
    """
    function to read arthus anomalies files
    """
    data = xr.open_dataset('/net/ostro/4sibylle/diurnal_cycle_arthus/5_mins/anomalies/'+measurement + '_arthus_anomaly.nc')    
    
    return data

    

def convert_local_time_and_reorder_for_time_hh_mm(data):
    """
    function to convert time array of diurnal cycle data from UTC to local time and reorder 
    data with respect to hh:mm, displaying diurnal cycle correctly.
    

    Args:
        data (xarray dataset): data to reorder
    Returns:
        data_sorted
    """
    data['Time'] = pd.to_datetime(data.Time.values) - timedelta(hours=4)
    times = pd.to_datetime(data.Time.values)
    times_hh = times.hour
    times_mm = times.minute
    num_times= np.arange(0,len(times_mm))
    times_new = ([datetime(2000,1,1,times_hh[i], times_mm[i]) for i in num_times])
    data['Time'] = times_new
    data_sorted = data.sortby('Time')
    
    return(data_sorted)

def read_all_lidar_diurnal_cycle_files(path_to_file):
    """function to read all lidar diurnal cycle files and 
    convert them to local time 
    
    arguments:
    path_to_file: string, path to the file directory
    dependencies:
    
    convert_local_time_and_reorder_for_time_hh_mm
    
    returns (list of xarray dataset)
    """
    T_data = xr.open_dataset(path_to_file+'T_diurnal_cycle.nc')
    MR_data = xr.open_dataset(path_to_file+'MR_diurnal_cycle.nc')
    VW_data = xr.open_dataset(path_to_file+'VW_diurnal_cycle.nc')

    data_array = [T_data, MR_data, VW_data]
    var_list = ['T','MR','VW']
    # convert time to local time (UTC - 4)
    data_lt = []
    
    for i, data in enumerate(data_array):
        var_name = var_list[i]
        data_new = convert_local_time_and_reorder_for_time_hh_mm(data)
        data_new = data_new.rename({'diurnal_cycle':var_name})
        data_lt.append(data_new)
        
    data_out = xr.merge(data_lt)
    return(data_out)


def read_h_wind(path_to_file):
    """function to read all lidar diurnal cycle files and 
    convert them to local time 
    
    arguments:
    path_to_file: string, path to the file directory
    dependencies:
    
    convert_local_time_and_reorder_for_time_hh_mm
    
    returns (list of xarray dataset)
    """
    H_wind_data = xr.open_dataset(path_to_file+'H_wind_speed_diurnal_cycle.nc')


    data_array = [ H_wind_data]
    var_list = [ 'HW']
    # convert time to local time (UTC - 4)
    data_lt = []
    
    for i, data in enumerate(data_array):
        var_name = var_list[i]
        data_new = convert_local_time_and_reorder_for_time_hh_mm(data)
        data_new = data_new.rename({'diurnal_cycle':var_name, 'Time':'time_h', 'Height':'height_h'})
        
        data_lt.append(data_new)

    data_out = xr.merge(data_lt, join='inner' )
    return(data_out)


def read_fluxes(path_to_file):
    """function to read all lidar diurnal cycle files and 
    convert them to local time 
    
    arguments:
    path_to_file: string, path to the file directory
    dependencies:
    
    convert_local_time_and_reorder_for_time_hh_mm
    
    returns (list of xarray dataset)
    """
    LHF_data = xr.open_dataset(path_to_file+'LHF_diurnal_cycle.nc')
    SHF_data = xr.open_dataset(path_to_file+'SHF_diurnal_cycle.nc')

    data_array = [ LHF_data, SHF_data]
    var_list = [ 'LHF', 'SHF']
    # convert time to local time (UTC - 4)
    data_lt = []
    
    for i, data in enumerate(data_array):
        var_name = var_list[i]
        data_new = convert_local_time_and_reorder_for_time_hh_mm(data)
        data_new = data_new.rename({'diurnal_cycle':var_name, 'Time':'time_coarse', 'Height':'height_coarse'})
        
        data_lt.append(data_new)

    data_out = xr.merge(data_lt, join='inner', )
    return(data_out)