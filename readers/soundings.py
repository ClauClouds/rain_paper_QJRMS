'''
scripts to read radiosoundings data

'''
import xarray as xr

def read_merian_soundings():
    '''
    read merian radiosoundings data
    return: xarray dataset with radiosoundings data
    
    '''
    dir_files = '/data/obs/campaigns/eurec4a/soundings/radiosondes_v3/level2/'
    file_name = 'EUREC4A_MS-Merian_Vaisala-RS_L2_v3.0.0.nc'
    
    ds = xr.open_dataset(dir_files + file_name)
    return ds

