"""
Reader for cloud types
"""

import os

import xarray as xr
from datetime import timedelta, datetime

def read_rain_flags():
    '''
    read rain flags and returns ds for the entire campaign
    '''
    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_class_prop_flags.nc",
        )
    )
    
    return ds


def read_cloudtypes():
    """
    Read cloud types

    Returns
    -------
    ds: cloud types for entire campaign
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_lcl_classification_v2.nc",
        )
    )

    return ds


def read_cloud_class():
    '''
    read cloud classification and returns ds for the entire campaign
    ''' 
    ds = xr.open_dataset( 
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_lcl_classification_v3.nc",
        )
    )

    return ds


def read_merian_classification():
    """_summary_
    function to read merian mesoscale classification from Isabel 
    returns:
    ds: xarray dataset with classification data
    
    """
    ds = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/EUREC4A_ATOMIC_C3ONTEXTClassifications_CollocatedTo_MS-Merian_ManualInstant_IR_FreqLim_0.5.nc', decode_times=False)
    ds['time'] = pd.to_datetime('2020-1-1') \
                + pd.to_timedelta(ds.day_frac.values, unit='days') # converting time stamps from fractional time of the year to datetime
    
    
    return ds


