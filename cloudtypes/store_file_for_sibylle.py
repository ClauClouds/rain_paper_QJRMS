""" code to produce an array with time, 
max_ze, rain_rate, flag from cloud radar data"""

from readers.soundings import read_merian_soundings

import numpy as np
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
from glob import glob
import metpy.calc as mpcalc 
from metpy.calc import equivalent_potential_temperature
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import pdb
#from figures.fig_sup1_eval_lidar_rain import read_cloud_radar

def main():
    

    # read cloud radar data to extract maxZe, rain rate and rain/no rain flag
    ds_cloud_radar = read_cloud_radar()
        
    # store cloud radar data
    ds_cloud_radar.to_netcdf('/net/ostro/ancillary_data_rain_paper/cloud_radar_data_for_sibylle.nc')



def read_cloud_radar():
    """
    Read all radar files at once. This uses the preprocessor to interpolate the
    radar data onto a common height grid.

    Returns
    -------
    ds: xarray dataset with radar reflectivity
    """
    from readers.radar import get_radar_files
    from readers.cloudtypes import read_rain_flags
    files = get_radar_files()

    ds = xr.open_mfdataset(files, drop_variables=['relative_humidity',
                                                  'skewness', 
                                                  'air_pressure',
                                                  'instrument',
                                                  'latitude',
                                                  'longitude',
                                                  'wind_speed', 
                                                  'liquid_water_path',
                                                  'air_temperature',
                                                  'brightness_temperature',
                                                  'wind_direction',
                                                  'mean_doppler_velocity',
                                                  'spectral_width'])
    ds = ds.reset_coords(drop=True)

    # find max radar_reflectivity for each time stamp
    ds['max_radar_reflectivity'] = ds.radar_reflectivity.max(dim='height')

    # drop radar_reflectivity variable from ds
    ds = ds.drop('radar_reflectivity')

    # read flags
    ds_flags = read_rain_flags()

    # define rain and no rain conditions
    is_no_rain = ds_flags.flag_rain == 0 
    is_no_rain_surface = ds_flags.flag_rain_ground == 0
    is_no_rain = is_no_rain & is_no_rain_surface
    is_rain = ~is_no_rain

    # add flag rain no rain to ds
    ds['flag_rain'] = np.repeat(-1, len(ds.time))

    # set flag to 0 for no rain
    ds['flag_rain'].values[is_no_rain] = 0

    # set flag to 1 for rain
    ds['flag_rain'].values[is_rain] = 1
    
    values_flag = ds['flag_rain'].values
    rain_rate = ds['rain_rate'].values
    launch_time = ds.time.values
    # define flag_rain as dataarray 
    da_flag_rain = xr.DataArray(values_flag, dims='launch_time')
    da_max_ze = xr.DataArray(ds['max_radar_reflectivity'].values, dims='launch_time')
    da_rain_rate = xr.DataArray(rain_rate, dims='launch_time')
    
    # create output dataset adding dataarray flag_rain and mazx radar reflectivity
    ds_output = xr.Dataset({'flag_rain': da_flag_rain, 
                            'rain_rate': da_rain_rate, 
                            'max_radar_reflectivity': da_max_ze},
                           coords={'time': ds.time})
    #rename time coordinate as launch_time
    return ds_output



if __name__ == '__main__':
    main()
    