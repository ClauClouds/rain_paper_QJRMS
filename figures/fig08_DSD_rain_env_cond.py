'''
code to plot the figure 08 of the paper about precipitation
'''

from readers.wband import read_radar_multiple
from readers.cloudtypes import read_rain_flags, read_cloud_class
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from readers.mrr import read_rain_cell_mrr_filelist, read_mrr_dsd
from figures.mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
import datetime 
import pandas as pd
import itertools
from readers.lidars import read_anomalies


def main():
    
    # derive aligned anomalies with MRR full dataset
    anomalies, diameters, height_dsd = derive_aligned_anomalies()
    print('anomalies aligned')
    print(anomalies)
    
    # define filenames of prec mrr for shallow and congestus
    shallow_file = '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_shallow.nc'
    congestus_file = '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_congestus.nc'

    # read mrr data selected for precipitation
    if os.path.isfile(shallow_file) * os.path.isfile(congestus_file): 
        print('reading shallow and congestus mrr datasets')
        mrr_shallow = xr.open_dataset(shallow_file)
        mrr_congestus = xr.open_dataset(congestus_file)

    else:
        print(' no file for shallow nor congestus mrr data found, producing it again')
        mrr_shallow, mrr_congestus = produce_mrr_shallow_congestus()

    print('read dsd shallow/congestus')
    
    
    # read surface variable anomalies
    surface_anomalies = prepare_surf_anomalies_data()
    
    print('surf anomalies read')
    
    # align surf data and mrr data for shallow and congestus
    mrr_shallow, surf_shallow = xr.align(mrr_shallow, surface_anomalies)
    mrr_congestus, surf_congestus = xr.align(mrr_congestus, surface_anomalies)
    
    print('aligned mrr and surface obs')
    
    # identify days and hours for each mrr selection of time stamps but removing mrr time stamps for reading arthus anomalies
    MR_non_prec_anomaly_shallow = find_days_hours(mrr_shallow, anomalies)
    MR_non_prec_anomaly_congestus = find_days_hours(mrr_congestus, anomalies)
    T_non_prec_anomaly_shallow = find_days_hours(mrr_shallow, anomalies)
    T_non_prec_anomaly_congestus = find_days_hours(mrr_congestus, anomalies)
    


    print('anomalies profiles extracted')
    
    # visualize plot
    visualize_figure_rain(mrr_shallow, mrr_congestus, surf_shallow, surf_congestus, MR_non_prec_anomaly_shallow, MR_non_prec_anomaly_congestus, T_non_prec_anomaly_shallow, T_non_prec_anomaly_congestus, diameters, height_dsd)
        
    
def visualize_figure_rain(mrr_shallow, mrr_congestus, surf_shallow, surf_congestus, MR_non_prec_anomaly_shallow, MR_non_prec_anomaly_congestus, T_non_prec_anomaly_shallow, T_non_prec_anomaly_congestus, diam_grid, height_dsd):
    """
    function to plot figure 9 of the paper

    Args:
        mrr_shallow (xarray dataset): dsd rain shallow
        mrr_congestus (xarray dataset): dsd rain congestus
        surf_shallow (xarray dataset): surf anomalies shallow
        surf_congestus (xarray dataset): surf anomalies congestus
        MR_non_prec_anomaly_shallow (xarray dataset): mr anomalies shallow
        MR_non_prec_anomaly_congestus (xarray dataset): mr anomalies congestus
        T_non_prec_anomaly_shallow (xarray dataset): t anomalies shallow
        T_non_prec_anomaly_congestus (xarray dataset): t anomalies congestus
        diameters: diameters of the drops
    """
    # merging datasets for profiles
    MR_p_an = xr.merge([MR_non_prec_anomaly_shallow, MR_non_prec_anomaly_congestus])
    T_p_an = xr.merge([T_non_prec_anomaly_shallow, T_non_prec_anomaly_congestus])
    
    # dropping common time stamps
    common_times_MR, boh1 = xr.align(MR_non_prec_anomaly_shallow, MR_non_prec_anomaly_congestus, join='inner')
    common_times_T, boh2 = xr.align(T_non_prec_anomaly_shallow, T_non_prec_anomaly_congestus, join='inner')

    MR_non_prec_anomaly_shallow = MR_non_prec_anomaly_shallow.drop_sel({'time':common_times_MR.time.values})
    MR_non_prec_anomaly_congestus = MR_non_prec_anomaly_congestus.drop_sel({'time':common_times_MR.time.values})
    T_non_prec_anomaly_shallow = T_non_prec_anomaly_shallow.drop_sel({'time':common_times_T.time.values})
    T_non_prec_anomaly_congestus = T_non_prec_anomaly_congestus.drop_sel({'time':common_times_T.time.values})
    
    print('number of shallow profiles', len(MR_non_prec_anomaly_shallow.time.values), len(T_non_prec_anomaly_shallow.time.values))
    print('number of congestus profiles', len(MR_non_prec_anomaly_congestus.time.values), len(T_non_prec_anomaly_congestus.time.values))

    MR_mean = MR_p_an.MR_anomaly.mean(axis=0, skipna=True)
    MR_std = MR_p_an.MR_anomaly.std(axis=0, skipna=True)
    T_mean = T_p_an.T_anomaly.mean(axis=0, skipna=True)
    T_std = T_p_an.T_anomaly.std(axis=0, skipna=True)
    
    # deriving mean anomalies profiles for MR and T for shallow and congestus clouds
    height = MR_non_prec_anomaly_shallow.height.values
    MR_shallow = MR_non_prec_anomaly_shallow.MR_anomaly.mean(axis=0, skipna=True)
    MR_congestus = MR_non_prec_anomaly_congestus.MR_anomaly.mean(axis=0, skipna=True)
    T_shallow = T_non_prec_anomaly_shallow.T_anomaly.mean(axis=0, skipna=True)
    T_congestus = T_non_prec_anomaly_congestus.T_anomaly.mean(axis=0, skipna=True)
    

    
    # deriving mean dsd in time (function of height and diameter)    
    dsd_shallow = mrr_shallow.drop_size_distribution_att_corr.mean(axis=0, skipna='True')
    dsd_deep = mrr_congestus.drop_size_distribution_att_corr.mean(axis=0, skipna='True')
    print(np.shape(dsd_deep))
    
    # calculate mean dsd over height for both shallow and deep
    dsd_shallow_mean_h = dsd_shallow.mean(axis=1, skipna='True')
    dsd_shallow_std_h = dsd_shallow.std(axis=1, skipna='True')
    dsd_deep_mean_h = dsd_deep.mean(axis=1, skipna='True')
    dsd_deep_std_h = dsd_deep.std(axis=1, skipna='True')
   
    print(np.shape(dsd_shallow_mean_h))
    print('--------------------')
    
    # plot dsd profiles, mean dsd over height, and mean anomalies profiles for the same indeces
    fig, axs = plt.subplots(4,2, figsize=(20,25))
    tick_levels = [3., 5., 7., 9.]
    # dsd shallow colors 
    mesh0 = axs[0,0].pcolormesh(diam_grid,
                    height_dsd, 
                    np.log10(dsd_shallow.values).T,
                    cmap=CMAP, 
                    vmin=2., 
                    vmax=10.)
    
    # contours of dsd shallow overplotted to the colormesh 
    cs0 = axs[0,0].contour(diam_grid,
                    height_dsd, 
                    np.log10(dsd_shallow.values).T,
                    levels=tick_levels, 
                    colors=['white', 'white', 'white', 'white'], 
                    )   
    axs[0,0].clabel(cs0, inline=True, fontsize=14)
    axs[0,0].set_ylabel('Height [m]', fontsize=25)
    axs[0,0].set_xlabel('Diameters [mm]', fontsize=25)
    axs[0,0].set_xlim(0., 6.)
    cbar0 = plt.colorbar(mesh0, ax=axs[0,0])
    cbar0.set_label('Log(Drop size distribution [m$^{-4}$])', fontsize=25)

    # dsd congestus clouds 
    mesh1 = axs[0,1].pcolormesh(diam_grid,
                    height_dsd, 
                    np.log10(dsd_deep.values).T,
                    cmap=CMAP, 
                    vmin=2., 
                    vmax=10.)
    axs[0,1].set_xlabel('Diameters [mm]', fontsize=25)
    axs[0,1].set_xlim(0., 6.)
    axs[0,1].set_ylabel('Height [m]', fontsize=25)
    cbar = plt.colorbar(mesh1, ax=axs[0,1])
    cbar.set_label('Log(Drop size distribution [m$^{-4}$])', fontsize=25)

    # contour overplotted to dsd congestus
    cs1 = axs[0,1].contour(diam_grid,
                    height_dsd, 
                    np.log10(dsd_deep.values).T,
                    levels=tick_levels,
                    colors=['white', 'white', 'white', 'white'])
    axs[0,1].clabel(cs1, inline=True, fontsize=14)

    # second line of plots: dsd shallow averaged over heights
    # for shallow
    axs[1,0].plot(diam_grid, 
                    np.log10(dsd_shallow_mean_h.values), 
                    color=COLOR_SHALLOW, 
                    linestyle='-', 
                    linewidth=6)
    #axs[1,0].fill_between(diam_grid, 
    #                 np.log10(dsd_shallow_mean_h.values + dsd_shallow_std_h.values), 
    #                 np.log10(dsd_shallow_mean_h.values - dsd_shallow_std_h.values), 
    #                 alpha=0.2, 
    #                 color=COLOR_SHALLOW) 
    axs[1,0].set_xlabel('Diameters [mm]', fontsize=25)
    axs[1,0].set_xlim(0., 6.)
    axs[1,0].set_ylabel('Log(Drop size distribution [m$^{-4}$])', fontsize=25)
    # for congestus    
    axs[1,1].plot(diam_grid, 
                np.log10(dsd_deep_mean_h.values), 
                color=COLOR_CONGESTUS, 
                linestyle='-', 
                linewidth=6)
    #axs[1,1].fill_between(diam_grid, 
    #                 np.log10(dsd_deep_mean_h.values + dsd_deep_std_h.values), 
    #                 np.log10(dsd_deep_mean_h.values - dsd_deep_std_h.values), 
    #                 alpha=0.2, 
    #                 color=COLOR_CONGESTUS)  
    axs[1,1].set_xlabel('Diameters [mm]', fontsize=25)
    axs[1,1].set_xlim(0., 6.)
    axs[1,1].set_ylabel('Log(Drop size distribution [m$^{-4}$])', fontsize=25)   
    
    # third line of plots: distributions of T surface anomaly and RH surface anomaly
    axs[2,0].hist(surf_shallow.air_temperature_anomaly.values, 
                bins=10, 
                color=COLOR_SHALLOW, 
                linestyle='-', 
                density='True', 
                histtype='step', 
                linewidth=6)

    axs[2,0].hist(surf_congestus.air_temperature_anomaly.values, 
                bins=10, 
                color=COLOR_CONGESTUS, 
                linestyle='-', 
                histtype='step',
                density='True', 
                linewidth=6)

    axs[2,0].set_ylabel('Normalized occ [%]', fontsize=25)
    axs[2,0].set_xlabel('surface air T anomaly [K]', fontsize=25)

    axs[2,1].hist(surf_shallow.relative_humidity_anomaly.values*100, 
                bins=10, 
                color=COLOR_SHALLOW, 
                linestyle='-', 
                histtype='step',
                density='True', 
                linewidth=6)
    axs[2,1].hist(surf_congestus.relative_humidity_anomaly.values*100, 
                bins=10, 
                color=COLOR_CONGESTUS, 
                linestyle='-',
                histtype='step',
                density='True', 
                linewidth=6)
    axs[2,1].set_ylabel('Normalized occ [%]', fontsize=25)
    axs[2,1].set_xlabel('surface air RH anomaly [%]', fontsize=25)
   
    # fourth line of plots: profiles anomalies of T and MR for shallow/congestus clouds
    axs[3,1].plot(MR_shallow.data, 
                   MR_shallow.height, 
                   color=COLOR_SHALLOW, 
                   linewidth=6)
    axs[3,1].plot(MR_congestus.values, 
                   height, 
                   color=COLOR_CONGESTUS, 
                   linewidth=6)        
    

    axs[3,1].plot(MR_mean.data, 
                  height, 
                  'black', 
                  linewidth=6)

    axs[3,1].fill_betweenx(height, 
                          MR_mean.data - MR_std.data, 
                          MR_mean.data + MR_std.data,  
                          facecolor='black', 
                          alpha = 0.1)

    axs[3,1].set_ylim(200., 1200.)
    axs[3,1].set_ylabel('Height [m]', fontsize=25)
    axs[3,1].set_xlabel('WVMR anomaly [K]', fontsize=25)
    axs[3,1].set_xlim(-5., 5.)

    axs[3,0].plot(T_shallow.values, 
                   height, 
                   color=COLOR_SHALLOW, 
                   linewidth=6)
    axs[3,0].plot(T_congestus.values, 
                   height, 
                   color=COLOR_CONGESTUS, 
                   linewidth=6) 
    
    axs[3,0].plot(T_mean.data, 
                  height, 
                  'black', 
                  linewidth=6)

    axs[3,0].fill_betweenx(height, 
                          T_mean.data - T_std.data, 
                          T_mean.data + T_std.data,  
                          facecolor='black', 
                          alpha = 0.1)
    axs[3,0].set_ylim(200., 1200.)
    axs[3,0].set_xlim(-5., 5.)

    axs[3,0].set_ylabel('Height [m]', fontsize=25)
    axs[3,0].set_xlabel('T anomaly [K]', fontsize=25)

     
    for ax, l in zip(axs[:].flatten(), ['a) Shallow',  'b) Congestus', 'c) Mean shallow dsd', 'd) Mean congestus dsd', 'e) Surface air temperature anomaly [K]', 'f) Surface relative humidity anomaly [%]', 'h) Air temperature anomaly profile [K] ', 'i) WVMR anomaly profile [kg kg-1]']):
        ax.text(-0.05, 1.1, l,  fontweight='black', fontsize=25, transform=ax.transAxes)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(3)
        ax.spines["left"].set_linewidth(3)
        ax.tick_params(which='minor', length=5, width=2, labelsize=25)
        ax.tick_params(which='major', length=7, width=3, labelsize=25)

    fig.tight_layout()

    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure09.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

    
def derive_aligned_anomalies():
    """
    function to align mrr data with arthus anomalies data

    Returns:
       anomalies xarray datasets aligned with mrr observations
    """
    # read mrr data
    mrr_dsd = read_mrr_dsd()
    
    # read anomalies for T, MR, VW
    T_anomaly  = read_anomalies('T')
    MR_anomaly = read_anomalies('MR')
    
    T_aligned, MR_aligned = xr.align(T_anomaly, MR_anomaly, join='inner')
    anomalies = xr.merge([T_aligned, MR_aligned])
    
    # selecting profiles closest to the mrr observations
    anomalies = anomalies.interp(time=mrr_dsd.time.values, method='nearest')
    print(len(anomalies.time.values), len(mrr_dsd.time.values))
    return anomalies, mrr_dsd.diameters.values, mrr_dsd.height.values
    
def read_surf_anomalies(var_name):
    '''
    read surface anomalies from ncdf files
    
    '''
    ds = xr.open_dataset('/net/ostro/4sibylle/surface_diurnal_cycles/'+var_name+'_surface_anomaly.nc')
    return ds


def prepare_surf_anomalies_data():
    '''
    read surface anomalies and merge them in a unique dataset
    
    dependencies:
    read_surf_anomalies
    '''
    # reading anomalies data
    t_surf = read_surf_anomalies('air_temperature')
    rh_surf = read_surf_anomalies('relative_humidity')
    #hw_surf = read_surf_anomalies('wind_speed')
    
    # storing only anomalies variables in a ncdf dataset
    air_temperature_anomaly = t_surf.anomaly.values
    relative_humidity_anomaly = rh_surf.anomaly.values
    #wind_speed_anomaly = hw_surf.anomaly.values
    
    surf_data = xr.Dataset(
                    data_vars={
                    "air_temperature_anomaly": (('time',), 
                                                air_temperature_anomaly,
                                                {'long_name': 'surface air temperature anomaly ', 
                                                'units':'K'}),
                    "relative_humidity_anomaly": (('time',), 
                                                relative_humidity_anomaly,
                                                {'long_name': 'surface relative humidity anomaly ', 
                                                'units':'K'})}, 
                    coords={
                    "time": (('time',), 
                             t_surf.time.values, 
                             {"axis": "T",
                              "standard_name": "time"})}
                   )
    return surf_data
    
def find_days_hours(prec_data, anomaly):
    """
    function that reads time stamps of input dataset and returns a time array containing beginning and ending 
    of the hours in which the time stamp is in, but removing the time stamps of the input array. They correspond
    to when mrr profiles of precipitation are collected, and we don't trust the raman lidar obs in rainy conditions. 
    The time stamp in output represents surrounding time stamps around a prec event.

    Args:
        dataset (xarray dataset): mrr shallow or congestus input dataset
    """
    
    '''
    function to select time stamps of the hours in which prec occurred and removing time stamps when 
    mrr data are collected.
    Args:
        dataset : prec_dataset from shallow or congestus clouds
        data    :  MR or T anomalies dataset aligned with mrr data all
    
    '''
    
    # reading time stamps without precipitation
    non_prec = anomaly.drop_sel({'time':prec_data.time.values})
    non_prec_time = non_prec.time.values
    
    # define hourly time grid for 
    time = anomaly.time.values
    time_grid = pd.date_range(time[0], time[-1], freq='15T') # T for minutes, H for hours
    
    # empty list to fill
    time_stamps_lidar = []
    
    # loop on time grid hours
    for ind_hours, hour_start in enumerate(time_grid[:-1]):
    
        hour_end = time_grid[ind_hours+1]
    
        # find indeces in the grid
        ind_found = np.where((non_prec_time > hour_start) * (non_prec_time <= hour_end))[0]
        if len(ind_found) > 0:

            # selecting anomalies in the hour
            time_stamps_lidar.append(non_prec_time[ind_found])


    time_anomaly = pd.to_datetime(list(itertools.chain.from_iterable(time_stamps_lidar)))
    non_prec_anomaly = anomaly.sel({'time':time_anomaly})
    
    return non_prec_anomaly
    

def produce_mrr_shallow_congestus():
    '''
    function to produce ncdf files containing mrr data for 
    shallow and mrr data for congestus clouds in precipitation conditions
    data are stored as
    '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_shallow.nc'
    '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_congestus.nc', 
    
    dependencies:
    read_mrr_dsd
    read_prec_indeces
    '''
    # read mrr file list for eac
    mrr_data_all = read_mrr_dsd()
    print('mrr data read in')
    
    # read indices of precipitation
    shallow_prec, deep_prec = read_prec_indeces()
    print('number of precipitation shallow ',len(shallow_prec.time.values))
    print('number of precipitation deep ', len(deep_prec.time.values))

    # align shallow and deep mrr datasets
    mrr_shallow, shallow_prec = xr.align(mrr_data_all, shallow_prec, join='inner')
    print('number of mrr shallow samples ', len(mrr_shallow.time.values))
    
    mrr_deep, deep_prec = xr.align(mrr_data_all, deep_prec, join='inner')
    print('number of mrr shallow samples ', len(mrr_deep.time.values))
    
    # storing mrr_deep and mrr_shallow to ncdf files for further analysis
    mrr_deep.to_netcdf(
        '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_congestus.nc', 
        encoding={"drop_size_distribution_att_corr":{"zlib":True, "complevel":9},\
        "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})

    mrr_shallow.to_netcdf(
        '/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/mrr_shallow.nc', 
        encoding={"drop_size_distribution_att_corr":{"zlib":True, "complevel":9},\
        "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    return mrr_shallow, mrr_deep
    
        
def read_prec_indeces():
    '''
    function to read cloud properties and flags and return indeces for shallow and congestus clouds that 
    correspond to precipitation, either in the air or on the ground. 
    '''
    
    # read cloud classification
    all_flags = read_rain_flags()

    # read shape from classification original file
    cloud_properties = read_cloud_class()  
    cloud_properties['flag_rain_ground'] = (('time'), all_flags.flag_rain_ground.values)
    cloud_properties['flag_rain'] = (('time'), all_flags.flag_rain.values)


    # selecting virga data (rain not reaching the ground) and rain data (reaching the ground)
    ind_prec = np.where(np.logical_or(cloud_properties.flag_rain_ground.values == 1, cloud_properties.flag_rain.values == 1))[0]
    prec = cloud_properties.isel(time=ind_prec)

    # selecting shallow and congestus datasets
    ind_shallow_prec = np.where((prec.shape.values == 0))[0]
    ind_deep_prec    = np.where((prec.shape.values == 1))[0]
    
    # selecting datasets for shallow/congestus
    shallow_prec = prec.isel(time=ind_shallow_prec)
    deep_prec = prec.isel(time = ind_deep_prec)
    
    return shallow_prec, deep_prec
    

    
if __name__ == "__main__":
    main()