"""
Created on Jan 11 2024
code to produce stats of the shallow vs cumulus clouds with the latest version of the code (ready for publication) 
and store temporary files for producing figure 2b
- 
@author: cacquist
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime


# path for output plot
path_out = '/work/plots_rain_paper/'


# reading cloud classification based on lcl and ct distance from it and cloud flags
cloud_prop_flags =  xr.open_mfdataset(np.sort(glob.glob('/data/obs/campaigns/eurec4a/msm/flags/new_version/*.nc')))
classification_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/cloud_lcl_classification_v3.nc')

# reading radar data files
radar_file_list = np.sort(glob.glob('/data/obs/campaigns/eurec4a/msm/wband_radar/ncdf/published_data_v2/*.nc'))
radar_data = xr.open_mfdataset(radar_file_list)

# merging all data together
data_all = xr.merge([cloud_prop_flags, radar_data, classification_data])

# selecting shallow and deep clouds
ind_shallow = np.where((data_all.shape.values == 0))[0]
ind_congestus = np.where((data_all.shape.values == 1))[0]
shallow_clouds = data_all.isel(time = ind_shallow)
congestus_clouds = data_all.isel(time = ind_congestus)

print('dim shallow', len(shallow_clouds.time.values))
print('dim congestus', len(congestus_clouds.time.values))

# defining function that reads data above cloud base
def f_build_masks(data_sel):
    """ function to extract from the input dataset the data that lie above the cloud bas
    we build a mask of 1 where we need to save values and of zeros where we need to drop the values

    input: dataset containing cloud base and data_sel
    output: dataset_above
    """
    # reading the matrix of reflectivity values data
    ze = data_sel.radar_reflectivity.values
    # defining the mask that will be used to remove the values below cloud base
    mask_matrix = np.zeros_like(ze)
    # filling the values with nans
    mask_matrix.fill(np.nan)
    
    height = data_sel.height.values
    ind_cb_arr = []
    # loop on time stamps to read the cloud base value
    for ind_time in range(len(data_sel.time.values)):
        # reading column associated with the given time stamp
        data_profile = data_sel.isel(time=ind_time)
        # selecting heights above or equal to cloud base height for that time stamp
        ind_ones = np.where(data_profile.height.values >= data_profile.cloud_base.values)[0]
        mask_matrix[ind_time, ind_ones] = 1

    return(mask_matrix)

print('processing shallow clouds')
# selecting shallow and deep data that are above cloud base
mask_shallow = f_build_masks(shallow_clouds)

ze_shallow = np.multiply(shallow_clouds.radar_reflectivity.values, mask_shallow)
vd_shallow = np.multiply(shallow_clouds.mean_doppler_velocity.values, mask_shallow)
sk_shallow = np.multiply(shallow_clouds.skewness.values, mask_shallow)

print('processing congestus clouds')

mask_congestus = f_build_masks(congestus_clouds)

ze_congestus = np.multiply(congestus_clouds.radar_reflectivity.values, mask_congestus)
vd_congestus = np.multiply(congestus_clouds.mean_doppler_velocity.values, mask_congestus)
sk_congestus = np.multiply(congestus_clouds.skewness.values, mask_congestus)

print('storing data in ncdf')

# store matrices in a ncdf array
 # saving the data in CF compliant conventions
data_out = xr.Dataset(
    data_vars={
        "ze_congestus": (('time_c','height_c'), ze_congestus, {'long_name': 'radar reflectivity above cloud base', 'units':'dBz'}),
        'vd_congestus': (('time_c','height_c'), vd_congestus, {'long_name': 'mean Doppler velocity above cloud base', 'units':'m s-1'}),
        'sk_congestus':(('time_c','height_c'), sk_congestus, {'long_name': 'skewness above cloud base', 'units':''}),
        'ze_shallow':(('time_s','height_s'), ze_shallow, {'long_name': 'radar reflectivity above cloud base', 'units':'dBz'}),
        'vd_shallow':(('time_s','height_s'), vd_shallow, {'long_name': 'mean Doppler velocity above cloud base', 'units':'m s-1'}),
        'sk_shallow':(('time_s','height_s'), sk_shallow, {'long_name': 'skewness above cloud base', 'units':''}),
    },  
    coords={
        "time_c": (('time_c',),  congestus_clouds.time.values, {"axis": "T","standard_name": "time"}),
        "height_c": (('height_c',), congestus_clouds.height.values, {"axis": "Z","positive": "up","units": "m", "long_name":'radar range height'}),
        "time_s": (('time_s',), shallow_clouds.time.values, {"axis": "T","standard_name": "time"}),
        "height_s": (('height_s',), shallow_clouds.height.values, {"axis": "Z","positive": "up","units": "m", "long_name":'radar range height'}),
    },  
    attrs={'CREATED_BY'     : 'Claudia Acquistapace and Albert Garcia Benadi',
                        'ORCID-AUTHORS'   : "Claudia Acquistapace: 0000-0002-1144-4753, Albert Garcia Benadi : 0000-0002-5560-4392", 
                    'CREATED_ON'       : str(datetime.now()),
                    'FILL_VALUE'       : 'NaN',
                    'PI_NAME'          : 'Claudia Acquistapace',
                    'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                    'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                    'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DO_NAME'          : 'University of Cologne - Germany',
                    'DO_AFFILIATION'   : 'University of Cologne - Germany',
                    'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                    'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DS_NAME'          : 'University of Cologne - Germany',
                    'DS_AFFILIATION'   : 'University of Cologne - Germany',
                    'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                    'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DATA_DESCRIPTION' : 'wband radar data collected on Maria S. Merian (msm) ship during EUREC4A campaign',
                    'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                    'DATA_GROUP'       : 'Experimental;Profile;Moving',
                    'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                    'DATA_SOURCE'      : 'wband radar data postprocessed',
                    'DATA_PROCESSING'  : 'ship motion correction and filtering of interference the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                    'INSTRUMENT_MODEL' : 'Wband radar data',
                    'COMMENT'          : 'data above cloud base' }
)

# assign istrument id
instrument_id = xr.DataArray("wband-radar",dims=(),attrs={"cf_role": "trajectory_id"},)
data_out = data_out.assign({"instrument": instrument_id,})

# assign additional attributes following CF convention
data_out = data_out.assign_attrs({
        "Conventions": "CF-1.8",
        "title": data_out.attrs["DATA_DESCRIPTION"],
        "institution": data_out.attrs["DS_AFFILIATION"],
        "history": "".join([
            "source: " + data_out.attrs["DATA_SOURCE"] + "\n",
            "processing: " + data_out.attrs["DATA_PROCESSING"] + "\n", 
            "adapted to enhance CF compatibility\n",
        ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
        "featureType": "trajectoryProfile",
    })

# storing ncdf data
data_out.to_netcdf(path_out+'ze_sk_vd_above_cloud_base.nc', encoding={"ze_congestus":{"zlib":True, "complevel":9},\
                                                                                    "vd_congestus": {"dtype": "f4", "zlib": True, "complevel":9}, \
                                                                                    "sk_congestus": {"zlib": True, "complevel":9}, \
                                                                                    "ze_shallow": {"zlib": True, "complevel":9}, \
                                                                                    "vd_shallow": {"zlib": True, "complevel":9}, \
                                                                                    "sk_shallow": {"zlib": True, "complevel":9}, \
                                                                                    "time_c": {"units": "seconds since 2020-01-01","dtype": "i4"}, \
                                                                                    "time_s": {"units": "seconds since 2020-01-01", "dtype": "i4"}})









