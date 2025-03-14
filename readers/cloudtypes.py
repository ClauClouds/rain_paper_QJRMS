"""
Reader for cloud types
"""

import os

import pandas as pd
import xarray as xr

def read_cloud_thickness_top_base():
    ''''
    read cloud thickness, cloud base and cloud top
    and returns ds for the entire campaign
    '''
    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_class_prop_flags.nc"),
            drop_variables = ['flag_cloud_base_source',
                            'flag_rain',
                            'virga_depth',
                            'precip',
                            'shape']
        )
    
    return ds

def read_cloud_top_base():
    """
    read cloud top and returns ds for the entire campaign
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_class_prop_flags.nc"),
            drop_variables = ['flag_cloud_base_source',
                            'flag_rain',
                            'flag_rain_ground',
                            'virga_depth',
                            'cloud_geometrical_thickness',
                            'precip',
                            'shape']
        )
    
    return ds

def read_cloud_base():
    """
    read cloud base and returns ds for the entire campaign
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_class_prop_flags.nc"), 
            drop_variables = ['flag_cloud_base_source', 
                         'flag_rain', 
                         'flag_rain_ground',
                         'virga_depth',
                         'cloud_top_height',
                         'cloud_geometrical_thickness', 
                         'precip', 
                         'shape']

    )

    return ds

def read_rain_flags():
    """
    read rain flags and returns ds for the entire campaign
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_class_prop_flags.nc",
        )
    )

    return ds


def read_cloud_class():
    """
    read cloud classification and returns ds for the entire campaign
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_lcl_classification_v3.nc",
        )
    )

    return ds


def read_merian_classification():
    """
    function to read merian mesoscale classification from Isabel
    returns:
    ds: xarray dataset with classification data
    """

    ds = xr.open_dataset(
        "/data/obs/campaigns/eurec4a/msm/EUREC4A_ATOMIC_C3ONTEXTClassifications_CollocatedTo_MS-Merian_ManualInstant_IR_FreqLim_0.5.nc",
        decode_times=False,
    )
    ds["time"] = pd.to_datetime("2020-1-1") + pd.to_timedelta(
        ds.day_frac.values, unit="days"
    )  # converting time stamps from fractional time of the year to datetime

    return ds



def read_in_clouds_radar_moments():
    '''
    read file containing radar moments selected above cloud base
    
    '''
    
    path_in = '/work/plots_rain_paper/'
    data = xr.open_dataset(path_in+'ze_sk_vd_above_cloud_base.nc')
    
    return data




def read_diurnal_cycle(var_string):
    """
    function to read the diurnal cycle from the ncdf file where it is stored
    arguments:
        var_string: string indicating the variable name to be read
     
    output:
        diurnalcycle dataset (xarray dataset)
    """
    diurnal_cycle = xr.open_dataset('/work/plots_rain_paper/diurnal_cycle_arthus/5_mins/'+var_string+'_diurnal_cycle.nc')

    return diurnal_cycle


def read_rain_ground():
    """
    read file with flag for rain at the ground
    """
    ds_r = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/rain_ground_classification_v3.nc')
    return(ds_r)


def read_cloudtypes():

    ds = xr.open_dataset(os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_lcl_classification_v3.nc",
        ))
    
    return(ds)