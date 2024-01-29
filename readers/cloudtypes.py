"""
Reader for cloud types
"""

import os

import pandas as pd
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
            "cloud_lcl_classification_v3.nc",
        )
    )

    return ds


def read_merian_classification():
    """
    function to read merian mesoscale classification from Isabel Mc Coy
    returns:
    ds: xarray dataset with classification data

    """

    ds = xr.open_dataset(
        "/data/obs/campaigns/eurec4a/msm/EUREC4A_ATOMIC_C3ONTEXTClassifications_CollocatedTo_MS-Merian_ManualInstant_IR_FreqLim_0.5.nc",
        decode_times=False,
    )

    ds_new = preprocess_class(ds)

    return ds_new


def preprocess_class(ds):
    """
    preprocess time stamps converting them from fractional day of the year to datetime
    Args:
        ds (xarray dataset): dataset with mesoscale classification data and additional datetime coordinate
    """

    ds["datetime"] = pd.to_datetime(
        ds.day_frac.values, unit="D", origin=pd.Timestamp("01-01-2020")
    )

    return ds
