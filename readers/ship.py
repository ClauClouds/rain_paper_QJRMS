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
    ship_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/ship data/ship_dataset_allvariables.nc')
    ship_data

    # removing duplicated times
    _, index = np.unique(ship_data['time'], return_index=True)
    ship_data = ship_data.isel(time=index)
    
    
    return ship_data
