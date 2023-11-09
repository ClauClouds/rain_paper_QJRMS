"""
Reader for cloud types
"""

import os

import xarray as xr


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
