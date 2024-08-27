"""
Reader for LCL data
"""

import os

import xarray as xr


def read_lcl():
    """
    Read LCL from RV Maria S Merian

    Returns
    -------
    ds: lifting condensation level dataset
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/LCL_dataset.nc",
        )
    )

    return ds

def read_diurnal_cycle_lcl(path_to_file):
    
    """
    Read diurnal cycle of lcl file

    Returns
    -------
    ds: lifting condensation level diurnal cycle dataset
    """

    ds = xr.open_dataset(
        os.path.join(
            path_to_file, 
            "_lcl_diurnal_cycle.nc",
        )
    )

    return ds