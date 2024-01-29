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
