'''
code to compare T/MR profiles from raman lidar/wind lidar with RS in prec/non prec cond
the goal is to establish until which rain amount we can trust lidar obs in rain

1) we first read the radiosonde data 
2) we process the radar data ( no data gaps) to extract the needed time series
3) interpolate radar data on rs time stamps
4) interpolate RS on heigth grid of lidar

'''
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
from figures.fig09bis_rs_q_thetae_classes import assign_cloud_type, group_soundings_per_class
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW, cmap_rs_shallow, cmap_rs_congestus

def main():
    
    # check if file exists otherwise create it
    try:
        ds_rs_T_MR = xr.open_dataset('/net/ostro/ancillary_data_rain_paper/T_rs_rain_rate_maxZe.nc')
    except:
        print('file does not exist, creating it')

        # read radiosounding data
        ds = read_merian_soundings()
        # set as time coordinate the variable launch_time
        ds = ds.set_index(sounding='launch_time')
        
        # read cloud radar data to extract maxZe, rain rate and rain/no rain flag
        ds_cloud_radar = read_cloud_radar()
        
        # find closest time stamps of ds_cloud_radar to ds_sounding values
        ds_DCR = ds_cloud_radar.interp(launch_time=ds.sounding)
        
        # merge ds_DCR with ds
        ds_merged = xr.merge([ds, ds_DCR])
        
        # read lidar data
        T_lidar = read_lidar_data(var='T')
        MR_lidar = read_lidar_data(var='MR')
        
   
        # interpolate rs on lidar height grid
        ds_merged = ds_merged.interp(alt=T_lidar.height)
        print(ds_merged)

        # read exact lidar profile for each radiosonde profile
        ds_rs_T = read_exact_lidar_to_rs(ds_merged, T_lidar, 'T_lidar', 'T')
        ds_rs_T_MR = read_exact_lidar_to_rs(ds_rs_T, MR_lidar, 'MR_lidar', 'MR')

        # store ds_rs as ncdf
        ds_rs_T_MR.to_netcdf('/net/ostro/ancillary_data_rain_paper/T_rs_rain_rate_maxZe.nc')
        pass
    
    
    # plot scatter plot
    plot_scatter_lidar_rs(ds_rs_T_MR, 'T', 'rain_rate', 'ta', 'T_lidar')
    plot_scatter_lidar_rs(ds_rs_T_MR, 'T', 'max_ze', 'ta', 'T_lidar')
    plot_scatter_lidar_rs(ds_rs_T_MR, 'T', 'height', 'ta', 'T_lidar')

    # plot profile differences
    plot_profile_diff(ds_rs_T_MR, 'T', 'ta', 'T_lidar')



def plot_profile_diff(ds_rs, var, var_rs, var_arthus):
    """
    function to plot scatter plot of lidar vs rs data
    input:
        - ds_rs: xarray dataset with radiosonde data
        - var: string with variable to plot from rs/lidar
        - var_rs: var name for rs data
        - var_arthus: var name for arthus data
    
    """
    
    # selecting rainy soundings
    is_rain = ds_rs['flag_rain'] == 1
    is_no_rain = ds_rs['flag_rain'] == 0
    ds_rain = ds_rs.sel(sounding=is_rain)
    ds_no_rain = ds_rs.sel(sounding=is_no_rain)

    profile_rs_rain = ds_rain[var_rs].values
    profile_arthus_rain = ds_rain[var_arthus].values
    prof_diff_rain = profile_rs_rain - profile_arthus_rain
    
    profile_rs_norain = ds_no_rain[var_rs].values
    profile_arthus_norain = ds_no_rain[var_arthus].values
    prof_diff_norain = profile_rs_norain - profile_arthus_norain
    
    # Define figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    #for i_prof in range(len(ds_rain.sounding.values)):
    #    ax[0].plot(prof_diff_rain[i_prof, :],
    ##               ds_rain.height, 
    #               linewidth=1,
    #               color='blue', 
    #               alpha=0.2)
    #for i_prof in range(len(ds_no_rain.sounding.values)):
    #    ax[1].plot(prof_diff_norain[i_prof, :],
    #               ds_no_rain.height, 
    #               linewidth=1,
    #               color='blue', 
    #               alpha=0.2)
        
    # plot mean profiles
    ax[0].plot(np.nanmean(prof_diff_rain, axis=0), ds_rain.height, color='black', linewidth=3)
    ax[1].plot(np.nanmean(prof_diff_norain, axis=0), ds_no_rain.height, color='black', linewidth=3)

    # plot percentiles of difference 
    ax[0].fill_betweenx(ds_rain.height,
                        np.nanpercentile(prof_diff_rain, 25, axis=0),
                        np.nanpercentile(prof_diff_rain, 75, axis=0),
                        color='blue',
                        alpha=0.2)
    
    ax[1].fill_betweenx(ds_no_rain.height,
                        np.nanpercentile(prof_diff_norain, 25, axis=0),
                        np.nanpercentile(prof_diff_norain, 75, axis=0),
                        color='blue',
                        alpha=0.2)
    
    ax[0].set_title('Rain')
    ax[1].set_title('No rain')
    if var == 'T':
        ax[0].set_xlabel('RS temperature - lidar temperature (K)')
        ax[1].set_xlabel('RS temperature - lidar temperature (K)')
        ax[0].set_ylabel('Height (m)')
        ax[1].set_ylabel('Height (m)')
        ax[0].set_xlim(-5, 5)
        ax[0].set_ylim(0, 4000)
        ax[1].set_xlim(-5, 5)
        ax[1].set_ylim(0, 4000)
        var_string='T'
    elif var == 'MR':
        ax[0].set_xlabel('RS mixing ratio - lidar mixing ratio (g/kg)')
        ax[1].set_xlabel('RS mixing ratio - lidar mixing ratio (g/kg)')
        ax[0].set_ylabel('Height (m)')
        ax[1].set_ylabel('Height (m)')
        ax[0].set_xlim(-5, 5)
        ax[0].set_ylim(0, 4000)
        ax[1].set_xlim(-5, 5)
        ax[1].set_ylim(0, 4000)
        var_string='MR'
    ax[0].grid()
    ax[1].grid()
    
    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    
    # save figure
    fig.savefig('/net/ostro/plots_rain_paper/profile_diff_lidar_RS_'+var+'.png')

def plot_scatter_lidar_rs(ds_rs, var, var_c, var_rs, var_arthus):
    """
    function to plot scatter plot of lidar vs rs data
    input:
        - ds_rs: xarray dataset with radiosonde data
        - var: string with variable to plot from rs/lidar
        - var_c: string with variable to plot for colorbar
        - var_rs: var name for rs data
        - var_arthus: var name for arthus data
    
    """
    if var_c == 'max_ze':
        CMAP = cmap_rs_congestus
        symbol_size = 100
        cbar_string = 'max Ze (dBZ)'
        cmap_min = 0.
        cmap_max = 10.
        
    elif var_c == 'rain_rate':
        CMAP = cmap_rs_congestus
        cbar_string = 'rain rate (mm/h)'
        symbol_size = 100
        cmap_min = 0.
        cmap_max = 5.

    elif var_c == 'height':
        CMAP = cmap_rs_congestus
        cbar_string = 'height (m)'
        symbol_size = 100
        cmap_min = 0.
        cmap_max = 4000.

    # selecting rainy soundings
    is_rain = ds_rs['flag_rain'] == 1
    is_no_rain = ds_rs['flag_rain'] == 0
    ds_rain = ds_rs.sel(sounding=is_rain)
    ds_no_rain = ds_rs.sel(sounding=is_no_rain)

    # read variables for plot
    profile_rs_rain = ds_rain[var_rs].values.flatten()
    profile_arthus_rain = ds_rain[var_arthus].values.flatten()

    profile_rs_norain = ds_no_rain[var_rs].values.flatten()
    profile_arthus_norain = ds_no_rain[var_arthus].values.flatten()
    
    print('number of rainy samples', len(ds_rain.sounding.values))
    print('number of non rainy samples', len(ds_no_rain.sounding.values))
    pdb.set_trace() 
    

    
    
    # Define figure with 2 subplots 
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # setting colorbar
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)

    
    plot_scatter(fig, ax[0], ds_rain, CMAP, symbol_size, cbar_string, norm, profile_rs_rain, profile_arthus_rain, var_c)
    plot_scatter(fig, ax[1], ds_no_rain, CMAP, symbol_size, cbar_string, norm, profile_rs_norain, profile_arthus_norain, var_c)

    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    ax[0].set_title('Rain')
    ax[1].set_title('No rain')
    if var == 'T':
        ax[0].set_xlabel('RS temperature (K)')
        ax[1].set_xlabel('RS temperature (K)')
        ax[0].set_ylabel('Lidar temperature (K)')
        ax[1].set_ylabel('Lidar temperature (K)')
        ax[0].set_xlim(280, 300)
        ax[0].set_ylim(275, 310)
        ax[1].set_xlim(280, 300)
        ax[1].set_ylim(275, 310)
        var_string='T'
    elif var == 'MR':
        ax[0].set_xlabel('RS mixing ratio (g/kg)')
        ax[1].set_xlabel('RS mixing ratio (g/kg)')
        ax[0].set_ylabel('Lidar mixing ratio (g/kg)')
        ax[1].set_ylabel('Lidar mixing ratio (g/kg)')
        ax[0].set_xlim(0, 20)
        ax[0].set_ylim(0, 20)
        ax[1].set_xlim(0, 20)
        ax[1].set_ylim(0, 20)
        var_string='MR'
    ax[0].grid()
    ax[1].grid()
    
    # save figure
    fig.savefig('/net/ostro/plots_rain_paper/scatter_lidar_RS_'+var_string+'_'+var_c+'.png')

    # find 


def plot_scatter(fig, ax, ds_rs, CMAP, symbol_size, cbar_string, norm, profile_rs, profile_arthus, var_c):
    
    if var_c == 'height':
        h_matrix = np.tile(ds_rs['alt'].values, (len(ds_rs['sounding'].values), 1))
        c_data = h_matrix.flatten()
    elif var_c == 'max_ze':
        ze_matrix = np.tile(ds_rs['max_radar_reflectivity'].values, (len(ds_rs['height'].values), 1))
        c_data = ze_matrix.flatten()
    elif var_c == 'rain_rate':
        rain_rate_matrix = np.tile(ds_rs['rain_rate'].values, (len(ds_rs['height'].values), 1))
        c_data = rain_rate_matrix.flatten()
    
    # remove values with nan
    rs = profile_rs.flatten()
    arthus = profile_arthus.flatten()
    
    ind_good = np.where(~np.isnan(rs) | ~np.isnan(arthus) | ~np.isnan(c_data))
    c_data = c_data[ind_good]
    rs = rs[ind_good]
    arthus = arthus[ind_good]
    
    # sele
    # plot scatter with selected variables
    ax.scatter(rs, 
               arthus, 
               c=c_data, 
               cmap=CMAP, 
               marker='o',
               s=symbol_size, 
               edgecolors='black',
               norm=norm)
    ax.plot(profile_rs, profile_rs, linestyle='--', color='black')

    # add colorbar 
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP),
                                                ax=ax,
                        orientation='vertical',
                        label=cbar_string)

    return(ax)

def read_exact_lidar_to_rs(ds, ds_lidar, var_name, var):
   
    # define empty variable to fill with lidar data
    var_lidar = np.zeros((len(ds.sounding.values), len(ds.height.values)))

    for i_t, time_rs in enumerate(ds.sounding.values):
        
        time_rs = pd.to_datetime(time_rs)
                
        # find closest time stamp in ds_cp to the sounding datetime string
        time_diff_lidar = np.abs(pd.to_datetime(ds_lidar.time.values) - time_rs)
        ind_min_lidar = np.where(time_diff_lidar == np.min(time_diff_lidar))[0][0]
        print(' selected time lidar ', ds_lidar.time.values[ind_min_lidar])
        print('*********************************************************')
        
        # print('time difference in seconds', time_diff[ind_min].seconds)
        if time_diff_lidar[ind_min_lidar].seconds > 30:
            #print('time difference is too large, skipping')
            var_lidar[i_t, :] = np.repeat(np.nan, len(ds_lidar.height))  
        else:
            #assign cloud type classification to sounding
            var_lidar[i_t, :] = ds_lidar[var].values[ind_min_lidar, :]
        

    # add vars to dataset
    ds[var_name] = (('sounding', 'height'), var_lidar) #xr.DataArray(T_lidar, dims=('launch_time', 'height')) 

    return ds

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
    ds_output = ds_output.rename({'time': 'launch_time'})
    return ds_output

def read_lidar_data(var='T'):
    '''
    read lidar data
    var: string for variable to read
    dependencies:
    f_read_variable_dictionary
    f_call_postprocessing_arthus(var_string)

    '''
    from readers.lidars import f_call_postprocessing_arthus, f_read_variable_dictionary

     # read dictionary associated to the variable
    dict_var = f_read_variable_dictionary(var)

    # call postprocessing routine for arthus data
    data = f_call_postprocessing_arthus(var)

    # rename Time to time in data 
    data = data.rename({'Time': 'time'})

    # rename Height to height in data
    data = data.rename({'Height': 'height'})

    # rename Product to var in data
    data = data.rename({'Product': var})

    # drop variable Emission_Wavelength
    data = data.drop('Emission_Wavelength')
    data = data.drop('nans')
    data = data.drop('Elevation')
    data = data.drop('Azimuth')
    # remove chunksizes in xarray merged dataset
    data = data.chunk(chunks=None)

    return data



if __name__ == '__main__':
    main() 


