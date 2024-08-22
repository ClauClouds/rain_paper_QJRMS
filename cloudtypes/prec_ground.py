"""
produce a flag for precipitation reaching the ground based on the 
flags

"""


import os
import sys

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import glob


ProgressBar().register()

from path_folders import path_flag_files

def main():
    
    """
    classification rain at the ground
    """
    ds_r = read_rain(path_flag_files)
    print(ds_r)
    
    
    # write cloud types to netcdf file
    ds_r.to_netcdf(
        os.path.join(
            "/data/obs/campaigns/eurec4a/msm/",
            "rain_ground_classification_v3.nc",
        )
    )

    
def read_rain(path_flag_files):
    """
    list files of flags from the folder and eextract 
    the flag rain_ground and its attributes
    = 0 no rain at the ground
    = 1 rain at the ground
    Args:
        path_flag_files (string): path to the folder
    """
    vars_to_drop = ['cloud_base',
                    'cloud_geometrical_thickness', 
                    'cloud_top_height', 
                    'virga_depth', 
                    'flag_rain', 
                    'flag_cloud_base_source']
    
    file_list = sorted(glob.glob(path_flag_files+'*.nc'))
    data = xr.open_mfdataset(file_list, drop_variables=vars_to_drop)
    
    return(data)


if __name__ == "__main__":
    main()
