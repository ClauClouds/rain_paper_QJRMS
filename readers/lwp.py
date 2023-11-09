"""
Reader for LWP data
"""

import os

import xarray as xr


def read_lwp():
    """
    Read LWP from RV Maria S Merian

    Returns
    -------
    ds: liquid water path dataset with a temporal resolution of 3 seconds
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/mwr_dataset/version2.0.0/Level3/"
            "EUREC4A_Merian_MWR_Level3_v2.0.0.nc",
        )
    )

    return ds
