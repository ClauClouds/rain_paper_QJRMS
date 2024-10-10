"""
reader for ship data

"""

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr



def read_ship():
    """
    Returns xarray dataset containing ship data with removed time duplicates
    -------
    """
    
    # reading ship data for t, p, Rh, SST, lat , lon
    ship_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_allvariables.nc')

    # removing duplicated times
    _, index = np.unique(ship_data['time'], return_index=True)
    ship_data = ship_data.isel(time=index)
    
    
    return ship_data


def read_and_save_P_file():
    
    vars_to_drop = ['T', 'RH', 'SST', 'lat', 'lon']
    # reading ship data for t, p, Rh, SST, lat , lon
    ship_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_allvariables.nc',
                                drop_variables=vars_to_drop)

    # removing duplicated times
    _, index = np.unique(ship_data['time'], return_index=True)
    ship_data = ship_data.isel(time=index)
    ship_data.to_netcdf('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_P.nc')
    return ship_data

def read_ship_pressure():
    """
    Returns xarray dataset containing ship data with removed time duplicates
    -------
    """
    
    vars_to_drop = ['T', 'RH', 'SST', 'lat', 'lon']
    # reading ship data for t, p, Rh, SST, lat , lon
    ship_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_allvariables.nc',
                                drop_variables=vars_to_drop)

    # removing duplicated times
    _, index = np.unique(ship_data['time'], return_index=True)
    ship_data = ship_data.isel(time=index)
    
    
    return ship_data
