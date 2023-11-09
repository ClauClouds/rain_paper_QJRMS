"""
Divides clouds into different classes based on the lifting condensation level.

Procedure:
- apply height-dependent Ze sensitivity threshold onto radar observations
- 
"""

import os
import sys

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../")

from readers.radar import read_radar_multiple

ProgressBar().register()


def main():
    """
    Cloud classification
    """

    # read radar reflectivity and lifting condensation level
    ds_ze, ds_lcl = prepare_data()

    # create cloud mask
    da_cm = cloud_mask(ds_ze)

    # create cloud types
    ds_ct = cloud_typing(da_cm=da_cm, da_lcl=ds_lcl.lcl)

    # compute result
    ds_ct = ds_ct.compute()

    # write cloud types to netcdf file
    ds_ct.to_netcdf(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "cloud_lcl_classification_v2.nc",
        )
    )


def ze_threshold(z):
    """
    Height-dependent Ze threshold based on Equation 2 in Dias et al. (2019)

    Parameters
    ----------
    z : np.array
        Height in meters.

    Returns
    -------
    ze_db : np.array
        Threshold radar reflectivity in dBZ.
    """

    a_below3km, b_below3km = 8.35e-10, 1.75
    a_above3km, b_above3km = 8.35e-10, 1.63

    ze_lin = np.full(shape=z.shape, fill_value=np.nan)

    ze_lin[z <= 3000] = exp_fun(z[z <= 3000], a_below3km, b_below3km)
    ze_lin[z > 3000] = exp_fun(z[z > 3000], a_above3km, b_above3km)

    ze_db = 10 * np.log10(ze_lin)

    return ze_db


def exp_fun(z, a, b):
    """Exponential function"""
    return a * z**b


def prepare_data():
    """
    Prepares input data for cloud type classification
    """

    # lifting condensation level
    ds_lcl = xr.open_dataset("/data/obs/campaigns/eurec4a/msm/LCL_dataset.nc")

    # radar reflectivity
    ds_ze = read_radar_multiple()

    # interpolate lifting condensation level onto radar time steps
    ds_lcl = ds_lcl.interp(time=ds_ze.time)

    # align both time series
    ds_ze, ds_lcl = xr.align(ds_ze, ds_lcl)

    assert len(ds_lcl.time) == len(ds_ze.time)

    return ds_ze, ds_lcl


def cloud_mask(ds):
    """
    Creates cloud mask from radar reflectivity

    Parameters
    ----------
    ds : xarray.Dataset
        Radar reflectivity dataset.

    Returns
    -------
    da_cm : xarray.DataArray
        Cloud mask dataset.
    """

    da_cm = ds.radar_reflectivity > ze_threshold(ds.height)
    da_cm = da_cm.astype("uint8")
    da_cm = da_cm.rename("cloud_mask")

    return da_cm


def cloud_typing(da_cm, da_lcl):
    """
    Typing of clouds based on cloud mask and lifting condensation level

    Procedure:
    - check if any hydrometeor is above/below lcl

    Parameters
    ----------
    da_cm : xarray.DataArray
        Cloud mask.
    da_lcl : xarray.DataArray
        Lifting condensation level.
    """

    # 1. check if cloud is precipitating
    # get index of range gates above LCL and below or equal to LCL
    ix_above_lcl = da_cm.height > da_lcl
    ix_below_lcl = da_cm.height <= da_lcl
    cloud_above_lcl = da_cm.where(ix_above_lcl).sum("height").values > 0
    cloud_below_lcl = da_cm.where(ix_below_lcl).sum("height").values > 0

    # 2. check cloud height with respect to LCL
    # check if cloud is within 600 m above of lcl or above
    ix_within_lcl_600 = (da_cm.height > da_lcl) & (
        da_cm.height <= da_lcl + 600
    )
    ix_above_lcl_600 = (da_cm.height > da_lcl + 600) & (da_cm.height <= da_lcl)
    ix_above_4000 = da_cm.height > 4000
    cloud_within_lcl_600 = (
        da_cm.where(ix_within_lcl_600).sum("height").values > 0
    )
    cloud_above_lcl_600 = da_cm.where(ix_above_lcl_600).sum("height").values > 0
    cloud_above_4000 = (
        da_cm.isel({"height": ix_above_4000}).sum("height").values > 0
    )
    cloud_somewhere = da_cm.sum("height").values > 0

    # array to save time-dependend classes
    p_arr = np.full(shape=len(da_cm.time), fill_value=-3, dtype="int8")
    q_arr = np.full(shape=len(da_cm.time), fill_value=-3, dtype="int8")

    # precipitation classes
    p_arr[np.isnan(da_lcl).values] = -2  # no lcl data available
    p_arr[~cloud_somewhere] = -1  # no cloud observed
    p_arr[~cloud_below_lcl & cloud_above_lcl] = 0  # non-precipitating cloud
    p_arr[cloud_below_lcl & cloud_above_lcl] = 1  # precipitating cloud
    p_arr[cloud_below_lcl & ~cloud_above_lcl] = 2  # only below lcl

    # height classes
    q_arr[np.isnan(da_lcl).values] = -2  # no LCL
    q_arr[~cloud_somewhere] = -1  # no cloud observed
    q_arr[
        cloud_within_lcl_600 & ~cloud_above_lcl_600 & ~cloud_above_4000
    ] = 0  # shallow
    q_arr[cloud_above_lcl_600 & ~cloud_above_4000] = 1  # stratiform
    q_arr[cloud_below_lcl & ~cloud_above_lcl] = 2  # very low cloud
    q_arr[
        ~cloud_below_lcl
        & ~cloud_within_lcl_600
        & ~cloud_above_lcl_600
        & cloud_above_4000
    ] = 3  # very high cloud, pixels above 4 km height
    q_arr[
        (cloud_below_lcl | cloud_within_lcl_600 | cloud_above_lcl_600)
        & cloud_above_4000
    ] = 4  # very high clouds, also somewhere else

    # write to xarray dataset
    ds_ct = xr.Dataset()
    ds_ct.coords["time"] = da_cm.time.values
    ds_ct["precip"] = ("time", p_arr)
    ds_ct["shape"] = ("time", q_arr)

    ds_ct["precip"].attrs = dict(
        comment=(
            "-3: unclassified, "
            "-2: no LCL data available, "
            "-1: no cloud observed, "
            "0: non-precipitating cloud (no pixels below LCL, but above LCL), "
            "1: precipitating cloud (pixels occur below and above LCL), "
            "2: cloud only below LCL"
        )
    )

    ds_ct["shape"].attrs = dict(
        comment=(
            "-3: unclassified, "
            "-2: no LCL data available, "
            "-1: no cloud observed, "
            "0: shallow cloud = cloud top between LCL and LCL + 600 m), "
            "1: stratiform cloud = cloud top between LCL + 600 m and 4 km height, "
            "2: cloud only below LCL, "
            "3: cloud only above 4 km height, "
            "4: cloud above and below 4 km height"
        )
    )

    return ds_ct


if __name__ == "__main__":
    main()
