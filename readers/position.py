"""
Reader MSM ship position
"""

import os

import xarray as xr


def read_position():
    """
    Read position of RV Maria S Merian

    Returns
    -------
    ds : xarray.Dataset
        Ship position dataset.
    """

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/ship_data/",
            "shipData_all2.nc",
        )
    )

    return ds
