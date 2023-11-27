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
from readers.lcl import read_lcl

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
            "cloud_lcl_classification_v3.nc",
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
    ds_lcl = read_lcl()

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

    Precipitation definitions:
    - no cloud observed
    - non-precipitating: no pixel below lcl, but above lcl
    - precipitating: pixels occur below and above lcl
    - other (only below lcl)

    Height definitions:
    - shallow cloud = cloud top between LCL and LCL + 600 m
    - congestus cloud = cloud top between LCL + 600 m and 4 km height
    - cloud only below LCL
    - cloud only above 4 km height
    - cloud above and below 4 km height

    Parameters
    ----------
    da_cm : xarray.DataArray
        Cloud mask.
    da_lcl : xarray.DataArray
        Lifting condensation level.
    """

    # check inputs
    assert np.sum(np.isnan(da_lcl)) == 0

    # array to save time-dependend classes
    p_arr = np.full(shape=len(da_cm.time), fill_value=-2, dtype="int8")
    q_arr = np.full(shape=len(da_cm.time), fill_value=-2, dtype="int8")

    # to select only height bins above lcl
    ix_above_lcl = da_cm.height > da_lcl
    ix_below_lcl = da_cm.height <= da_lcl
    ix_below_lclplus600 = da_cm.height <= (da_lcl + 600)
    ix_above_lclplus600 = da_cm.height > (da_lcl + 600)
    ix_within_lcl_lclplus600 = ix_below_lclplus600 & ix_above_lcl
    ix_above_4000 = da_cm.height > 4000

    # non-height dependent binary flags
    # specific height range
    is_not_any = da_cm.sum("height").values == 0

    # hydrometeor presence
    is_any_above_lcl = da_cm.where(ix_above_lcl).sum("height").values > 0
    is_any_below_lcl = da_cm.where(ix_below_lcl).sum("height").values > 0
    is_any_within_lcl_lclplus600 = (
        da_cm.where(ix_within_lcl_lclplus600).sum("height").values > 0
    )
    is_any_above_lcl_600 = (
        da_cm.where(ix_above_lclplus600).sum("height").values > 0
    )
    is_any_above_4000 = da_cm.where(ix_above_4000).sum("height").values > 0

    # general definitions
    is_clear = is_not_any

    # precipitation definitions
    is_precipitating = is_any_below_lcl & is_any_above_lcl
    is_not_precipitating = ~is_any_below_lcl & is_any_above_lcl
    is_below_lcl_and_not_above_lcl = is_any_below_lcl & ~is_any_above_lcl

    assert len(is_clear) == (
        is_clear.sum()
        + is_precipitating.sum()
        + is_not_precipitating.sum()
        + is_below_lcl_and_not_above_lcl.sum()
    )

    # type definitions
    is_shallow = is_any_within_lcl_lclplus600 & ~is_any_above_lcl_600
    is_congestus = is_any_above_lcl_600 & ~is_any_above_4000

    assert is_shallow.sum() > 0
    assert is_congestus.sum() > 0
    assert (is_shallow & is_congestus).sum() == 0

    # precipitation classes
    p_arr[is_clear] = -1
    p_arr[is_not_precipitating] = 0
    p_arr[is_precipitating] = 1
    p_arr[is_below_lcl_and_not_above_lcl] = 2

    # height classes
    q_arr[is_clear] = -1
    q_arr[is_shallow] = 0
    q_arr[is_congestus] = 1

    # write to xarray dataset
    ds_ct = xr.Dataset()
    ds_ct.coords["time"] = da_cm.time.values
    ds_ct["precip"] = xr.DataArray(
        data=p_arr,
        dims=("time"),
        coords={"time": da_cm.time.values},
        attrs=dict(
            comment=(
                "-1: no hydrometeors present, "
                "0: non-precipitating (pixels are above LCL, but not below), "
                "1: precipitating (pixels are below and above LCL), "
                "2: other (pixels are below LCL, but not above)"
            )
        ),
    )

    ds_ct["shape"] = xr.DataArray(
        data=q_arr,
        dims=("time"),
        coords={"time": da_cm.time.values},
        attrs=dict(
            comment=(
                "-2: unclassified (neither shallow nor congestus), "
                "-1: no hydrometeors present, "
                "0: shallow (cloud top between LCL and LCL + 600 m), "
                "1: congestus (cloud top between LCL + 600 m and 4 km), "
            )
        ),
    )

    return ds_ct


if __name__ == "__main__":
    main()
