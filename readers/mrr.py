

import xarray as xr
import numpy as np
from glob import glob
import os


DROP_VARIABLES = [
    "Doppler_spectra_att_corr",
    "drop_size_distribution",
    "Doppler_spectra",
    "rain_rate",
    "liquid_water_content",
    "Ze",
    "attenuation",
    "Zea",
    "instrument",
    "lat",
    "lon", 
    "Doppler_velocity"
]

def get_mrr_files():
    """
    Returns list of all available radar files from RV Maria S Merian

    Returns
    -------
    files: list of all radar files
    """

    files = sorted(
        glob(
            os.path.join(
                "/data/obs/campaigns/eurec4a/msm/mrr/new_postprocessing/",
                "2020*_MRR_PRO_msm_eurec4a.nc",
            )
        )
    )
    return files


def read_mrr_dsd():
    
    """
    read mrr dsd data from ncdffiles
    """
    file_list = get_mrr_files()
    data = xr.open_mfdataset(file_list, drop_variables=DROP_VARIABLES)
    
    return data


def check_string_format(string):
        
        if len(string) <2:
            string_new = '0'+string
        else:
            string_new = string
            
        return string_new


def read_rain_cell_mrr_filelist(time_start, time_end, date):
    
    """"
    returns mrr_data for the selected cell
    """
    print(time_start.hour)
    dd = check_string_format(str(date.day))
    mm = check_string_format(str(date.month))
    yy = check_string_format(str(date.year))
    print(yy,mm,dd)
    
    # start and end time in the same hour: read only one file otherwise read 2 files
    if time_start.hour == time_end.hour:
        hh = check_string_format(str(time_start.hour))
        mrr_data_all = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/mrr/new_postprocessing/'+yy+mm+dd+'_'+hh+'_MRR_PRO_msm_eurec4a.nc', drop_variables=DROP_VARIABLES)
        
    else:
        
        hh = check_string_format(str(time_start.hour))
        hh_end = check_string_format(str(int(hh)+1))
        file_list = ['/data/obs/campaigns/eurec4a/msm/mrr/new_postprocessing/'+yy+mm+dd+'_'+hh+'_MRR_PRO_msm_eurec4a.nc', \
            '/data/obs/campaigns/eurec4a/msm/mrr/new_postprocessing/'+yy+mm+dd+'_'+hh_end+'_MRR_PRO_msm_eurec4a.nc']
        mrr_data_all= xr.open_mfdataset(file_list, drop_variables=DROP_VARIABLES)
    
    
    rain_cell = mrr_data_all.sel(time=slice(time_start, time_end))
    rain_cell['time']= np.array(rain_cell.time.values)
    
    return rain_cell