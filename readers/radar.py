"""
Reader for radar data.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

DROP_VARIABLES = [
    "rain_rate",
    "relative_humidity",
    "air_temperature",
    "air_pressure",
    "wind_speed",
    "wind_direction",
    "liquid_water_path",
    "brightness_temperature",
    "mean_doppler_velocity",
    "spectral_width",
    "skewness",
    "instrument",
]


def read_radar_multiple():
    """
    Read all radar files at once. This uses the preprocessor to interpolate the
    radar data onto a common height grid.

    Returns
    -------
    ds: xarray dataset with radar reflectivity
    """

    files = get_radar_files()

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    return ds


def preprocess(ds):
    """
    Preprocessor to open all radar files at once with dask
    """

    ds = ze_interp(ds)

    ds = ds["radar_reflectivity"]

    return ds


def read_radar(date=None, file=None, height_interp=True):
    """
    Read radar data from RV Maria S Merian either from specific date or from
    specific file. One can interpolate the radar data onto a common height
    grid.

    Parameters
    ----------
    date: date of radar file
    file: filename of radar file
    height_interp: whether to interpolate radar reflectivity onto a common
    height grid

    Returns
    -------
    ds: xarray Dataset of radar data
    """

    if date is None:
        ds = xr.open_dataset(file, drop_variables=DROP_VARIABLES)

    else:
        file = get_radar_file(date)
        ds = xr.open_dataset(file, drop_variables=DROP_VARIABLES)

    if height_interp:
        ds = ze_interp(ds)

    return ds


def ze_interp(ds):
    """
    Interpolates radar reflectivity onto common height grid

    Parameters
    ----------
    ds: dataset with radar reflectivity as a function of time and height

    Returns
    -------
    ze: radar reflectivity interpolated onto new height grid
    """

    # convert radar reflectivity to linear units
    ds["radar_reflectivity"] = 10 ** (0.1 * ds["radar_reflectivity"])

    height = read_reference_height()
    ds = ds.interp(coords={"height": height}, method="linear")

    # convert back to dBZ
    ds["radar_reflectivity"] = 10 * np.log10(ds["radar_reflectivity"])

    return ds


def read_reference_height():
    """
    Read reference height on which radar reflectivity is interpolated for
    the entire campaign.
    """

    height = read_radar(date="20200120", height_interp=False).height

    return height


def get_radar_file(date):
    """
    Returns list of all available radar files from RV Maria S Merian

    Parameters
    ----------
    date: date of radar file

    Returns
    -------
    file: filename of radar file
    """

    date = pd.Timestamp(date).strftime("%Y%m%d")

    file = os.path.join(
        "/data/obs/campaigns/eurec4a/msm/wband_radar/ncdf/",
        "published_data_v2/",
        f"{date}_wband_radar_msm_eurec4a_intake.nc",
    )

    return file


def get_radar_files():
    """
    Returns list of all available radar files from RV Maria S Merian

    Returns
    -------
    files: list of all radar files
    """

    files = sorted(
        glob(
            os.path.join(
                "/data/obs/campaigns/eurec4a/msm/wband_radar/ncdf/",
                "published_data_v2/",
                "2020*_wband_radar_msm_eurec4a_intake.nc",
            )
        )
    )

    return files


def read_lwp():
    """
    read LWP/IWV data obtained from the single channel retrieval at 89 GHz from the passive channel on the wband radar data
    function modified to include post processing provided by Sabrina on the 29.01.2024 for flagging non valid LWP data
    Returns
    -----
    xarray dataset with LWP data
    
    """
    
    LWP_IWV_data = xr.open_dataset('/data/obs/campaigns/eurec4a/mwr_dataset/version2.0.0/Level2/EUREC4A_Merian_MWR-MSMRAD_Level2_v2.0.0.nc')
    # post process LWP data to read only data without precipitation and where obs flags from MWR are true
    LWP_good = apply_filters(LWP_IWV_data,site='site')
    
    
    return LWP_good



def read_radar_single_file(yy,mm,dd):
    """
    Read radar data corresponding to the selected day

    Returns
    -------
    ds: xarray dataset with radar data
    """
    radarFile = '/data/obs/campaigns/eurec4a/msm/wband_radar/ncdf/published_data_v2/'+yy+mm+dd+'_wband_radar_msm_eurec4a_intake.nc'
    radarData =  xr.open_dataset(radarFile)
    
    return radarData


def apply_filters(xrin,site='merian'):
    '''
    applies precip_mask, quality_masks to mwr xr datasets by setting nans where flags are True
    INPUT: 
    - xrin: xr Dataset
    - site: string with site name ('merian' for our ship)
    OUTPUT:
    - xr Dataset with 'faulty'=filtered data replaced by nans
    '''
    data = xrin.copy()
    
    # now do Wband =============
    #general wband quality
    for k in ['iwv', 'lwp']:
        data[k].values[data.quality_mask[:,0] == 1.] = np.nan
        #data[k].values[data.precip_mask == 1.] = np.nan #to ensure that Clau and I are using the same precip masks(?)
    #iwv quality:
    data['iwv'].values[data.quality_mask[:,1] == 1.] = np.nan
    #lwp quality:
    data['lwp'].values[data.quality_mask[:,2] == 1.] = np.nan

    if site == 'merian':
        data['tb'].values[data.quality_mask[:,0] == 1.] = np.nan 
        #data['tb'].values[data.precip_mask == 1.] = np.nan 
    
    return data

