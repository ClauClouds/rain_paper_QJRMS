


'''
This script is used to plot the statistics of cold pools from the Arthus dataset.
It is designed based on fig09_cold_pool_stat.ipynb, and it is made to answer the request
from the reviewers to add wiskers plot 

'''

from matplotlib.ticker import AutoMinorLocator

import sys
import os
sys.path.append(os.path.abspath('../readers'))
from readers.lidars import read_anomalies
from readers.cloudtypes import read_rain_flags
from readers.ship import read_ship
from readers.radar import read_W_band
from readers.surf_anomalies import read_surf_anomaly
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import xarray as xr
import pandas as pd
import pdb
import seaborn as sns

# definition of list of variables to drop in the Wband radar data
# (to reduce the size of the dataset)
# (the variables are not needed for the cold pool statistics)
DROP_VARIABLES = [
    'air_temperature',
    "liquid_water_path",
    "brightness_temperature",
    "spectral_width",
    "skewness",
    "instrument",
    'relative_humidity',
    'air_pressure',
    'wind_speed',
    'wind_direction',

    ]


def main():


    # Define start and end dates for the analysis
    start_date = datetime(2020, 1, 25)
    end_date = datetime(2020, 2, 19)

    date_list = create_date_list(start_date, end_date)    

    Wband_all = []

    for date in date_list:
        #convert date to string
        date_str = date.strftime("%Y%m%d")
        #read Wband data
        Wband = read_W_band(date_str)
        Wband_all.append(Wband)

    # Concatenate the datasets along the 'time' dimension
    Wband = xr.concat(Wband_all, dim="time")

    # read anomalies of T and RH surface
    TSurf_anomaly = read_surf_anomaly('air_temperature', start_date, end_date)
    RHSurf_anomaly = read_surf_anomaly('relative_humidity', start_date, end_date)
        
    # read T, MR, VW anomalies
    MR_anomaly = read_anomalies('MR')
    T_anomaly = read_anomalies('T')
    VW_anomaly = read_anomalies('VW')
    
    # read ship data and selected it in the time range
    ds_ship = read_ship()
    ds = ds_ship.sel(time=slice(start_date, end_date))

    # read rain flag data
    rain_flag = read_rain_flags()
    ds_rain_flag = rain_flag.sel(time=slice(start_date, end_date))
        
    
    # identify cold pool candidates
    index_coldpool_candidates, ds_T = identify_coldpool_candidates(ds)

    # identify the onset of the coldpool candidates
    ds , tmax_indices_1 = identify_onset_of_coldpool_candidates(ds_T, index_coldpool_candidates)
    # identify the minimum of a coldpool
    ds, tmin_indices, tmax_indices = find_minimum_of_coldpool(ds, ds_T, tmax_indices_1)
    # identify the end of a coldpool 
    ds, tend_indices = coldpool_end(ds, tmax_indices, tmin_indices) 

    print("Number of cold pool candidates: ", len(index_coldpool_candidates))
    
    #select height for all datasets
    Wband_sel_h =  Wband.sel(height=slice(0,1000))
    T_anomaly_sel_h = T_anomaly.sel(Height=slice(0,1000))
    MR_anomaly_sel_h = MR_anomaly.sel(Height=slice(0,1000))
    VW_anomaly_sel_h = VW_anomaly.sel(Height=slice(0,1000))

    #put all variables into one dataset
    all_data = T_anomaly_sel_h
    all_data = all_data.rename({'anomaly': 'T_anomaly'})
    all_data = all_data.drop_vars(['Latitude', 
                                   'Longitude', 
                                   'ZSL', 
                                   'Emission_Wavelength', 
                                   'Range_Resolution',
                                    'Elevation', 
                                    'Elevation_Resolution',
                                    'Azimuth', 
                                    'Azimuth_Resolution',
                                    'nans', 
                                    'Product'])
    
    all_data['MR_anomaly'] = MR_anomaly_sel_h['anomaly']
    all_data['VW_anomaly'] = VW_anomaly_sel_h['anomaly']

    # rename coordinates for Wband radar data to homogeneize with the other datasets
    Wband_sel_h = Wband_sel_h.rename({"time": "Time"})
    Wband_sel_h = Wband_sel_h.rename({"height": "Height"})
    interpolated_rr = Wband_sel_h["radar_reflectivity"].interp(Height=all_data["Height"], Time=all_data["Time"])
    ds_rain_flag = ds_rain_flag.rename({"time": "Time"})
    interpolated_rain_flag = ds_rain_flag['flag_rain'].interp(Time=all_data['Time'])  
    ds_ship_1 = ds_ship.sel(time=slice(start_date, end_date))
    ds_ship_1 = ds_ship_1.rename({"time": "Time"})
    
    # interpolate T and RH surface variables to the time of the cold pool candidates
    # (to have the same time coordinates as the other variables)
    
    # call function to calculate diurnal cycle of surface variables and their anomalies
    interpolated_T_ds = TSurf_anomaly.interp(Time= all_data["Time"])
    interpolated_RH_ds = RHSurf_anomaly.interp(Time= all_data["Time"])
    
    # reading the variable containing the anomalies
    interpolated_T_surface = interpolated_T_ds['anomaly']
    interpolated_RH_surface = interpolated_RH_ds['anomaly']

    # Add the interpolated variable to the target dataset
    all_data = all_data.assign({"radar_reflectivity": interpolated_rr})
    all_data = all_data.assign({"T_surface": interpolated_T_surface })
    all_data = all_data.assign({"RH_surface": interpolated_RH_surface })
    all_data = all_data.assign({'rain_flag': interpolated_rain_flag})

    Wband_4h = Wband
    Wband_4h = Wband_4h.rename({"time": "Time"})
    Wband_4h = Wband_4h.rename({"height": "Height"})
    Wband_4000 = Wband_4h.interp(Time=all_data["Time"])

    datasets_start, datasets_end = sel_start_end_dataset(tmax_indices, tmin_indices, all_data, ds)
    W_start, W_end = sel_start_end_dataset(tmax_indices, tmin_indices, Wband_4000, ds)
    
    # Aligning the datasets 
    desired_start_time_start = "2000-01-01T00:00:00"
    datasets_start = shift_time_of_dataset(datasets_start, desired_start_time_start)

    #datasets_start = set_rain_to_nan(datasets_start)

    desired_start_time_end = "2000-01-01T01:20:00"
    datasets_end = shift_time_of_dataset(datasets_end, desired_start_time_end)
    #datasets_end = set_rain_to_nan(datasets_end)

    datasets_between = list_and_resample_and_reassign_between_tmax_tmin(ds, tmax_indices, tmin_indices, all_data)

    #create dataset to safe all mean variables in
    ds_mean_start = datasets_start[0].drop_vars(list(datasets_start[0].data_vars))
    ds_mean_between = datasets_between[0].drop_vars(list(datasets_between[0].data_vars))
    ds_mean_end = datasets_end[0].drop_vars(list(datasets_end[0].data_vars))
        
    
    for var in ['T_surface' ,'T_anomaly', 'RH_surface' , 'MR_anomaly', 'VW_anomaly', 'radar_reflectivity']:
        # calculate mean of each variable in the datasets
        ds_mean_start[var], ds_mean_start[var+'number_of_values_the_mean_was_calculated'] = take_mean(datasets_start, var)
        ds_mean_between[var], ds_mean_between[var+'number_of_values_the_mean_was_calculated'] = take_mean(datasets_between, var)
        ds_mean_end[var], ds_mean_end[var+'number_of_values_the_mean_was_calculated'] = take_mean(datasets_end, var)


    # calculate percentiles of T surf and RH surf variables in the dataset
    # calculate anomalies for T surf and RH surf
    # (anomalies are calculated as the difference to the mean of the whole dataset)
    percentiles_arr = [0.1, 0.25, 0.5, 0.75, 0.9]  # Define the percentiles to calculate
    ds_perc_start_T = take_percentiles(datasets_start, 'T_surface', percentiles_arr )
    ds_perc_between_T = take_percentiles(datasets_between, 'T_surface', percentiles_arr )
    ds_perc_end_T = take_percentiles(datasets_end, 'T_surface', percentiles_arr )
    
    ds_perc_start_RH = take_percentiles(datasets_start, 'RH_surface', percentiles_arr )
    ds_perc_between_RH = take_percentiles(datasets_between, 'RH_surface', percentiles_arr )
    ds_perc_end_RH = take_percentiles(datasets_end, 'RH_surface', percentiles_arr )
    
    # combine all percentiles in one dataset for each variable 
    ds_perc_T_surf = xr.concat([ds_perc_start_T, ds_perc_between_T, ds_perc_end_T], dim="Time")
    ds_perc_RH_surf = xr.concat([ds_perc_start_RH, ds_perc_between_RH, ds_perc_end_RH], dim="Time")
    
    # combine all in one dataset
    ds_mean_all = ds_mean_start.merge(ds_mean_between)
    ds_mean_all = ds_mean_all.merge(ds_mean_end)


    # same process for Wband (higher height):
    W_start = shift_time_of_dataset(W_start, desired_start_time_start)
    W_end = shift_time_of_dataset(W_end, desired_start_time_end)
    W_between = list_and_resample_and_reassign_between_tmax_tmin(ds, tmax_indices, tmin_indices, Wband_4000)

    W_mean_start, a = take_mean(W_start, 'radar_reflectivity')
    W_mean_between, a = take_mean(W_between, 'radar_reflectivity')
    W_mean_end, a = take_mean(W_end, 'radar_reflectivity')

    W_mean_all = xr.concat([W_mean_start, W_mean_between, W_mean_end], dim="Time")
    
    # Ensure time remains sorted if needed
    W_mean_all = W_mean_all.sortby("Time")
    
    # converting RH in percent 
    ds_mean_all['RH_surface'] = ds_mean_all['RH_surface']*100
    ds_perc_RH_surf = ds_perc_RH_surf * 100

    print('ds perc T surf: ', ds_perc_RH_surf)
    #pdb.set_trace()
    # Plotting the results
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25

    fig, axs = plt.subplots(6, figsize=(18,18), constrained_layout=True, sharex=True)
    fig.suptitle('Mean of '+str(len(ds_mean_all))+' coldpool cases', fontsize =27)#, fontweight='bold')

    # Define custom x-axis labels
    custom_labels = [-60, -40, -20, 'tmax', 'tmin', 20, 40]

    # Select every 20-minute interval from the dataset for labeling
    time_ticks = pd.date_range(start=ds_mean_start.Time[0].values, end=ds_mean_end.Time[-1].values, freq="20min")
    CMAP_rr = cmc.batlow
    CMAP = cmc.vik
    i = 0
    fs_axes = 20 #fontsize of axes
    vmin = [-2.5, -6, -10, -0.5, -2, -1]
    vmax = [ 1,  13,  10,  0.5,  2,  1]
    variables = ['T_surface', 'RH_surface' ,'radar_reflectivity', 'VW_anomaly', 'T_anomaly',  'MR_anomaly']
    colorbar_label = ['Surface Temperature \n Anomaly [K]','Surface Relative Humidity Anomaly', 'Ze [dBZ]', 'Vertical Wind \n Anomaly [ms$^{-1}$]', 'Temperature \n Anomaly [K]', 'Mixing Ratio \n Anomaly [gkg$^{-1}$]']
    titles = ['a) Mean Surface Temperature Anomaly', 'b) Mean Surface Relative Humidity Anomaly', 'c) Mean Radar Reflectivity', 'd) Mean Vertical Wind Speed Anomaly', 'e) Mean Temperature Anomaly','f) Mean Mixing Ratio Anomaly']
    for var in variables:
        
        if i == 0 or i ==1:
            
            honset = axs[i].vlines(ds_mean_start.Time.values[-1], vmin[i], vmax[i], color = 'black', linewidth=2, label = 'Onset of cp')
            hminimum = axs[i].vlines(ds_mean_between.Time.values[-1], vmin[i], vmax[i], color = 'black', linewidth=2,  linestyle='--', label = 'Minimum of cp')
            
            # construct the label h_mean only for the first subplot
            if i == 0:
                h_mean = axs[i].plot(ds_mean_all.Time.values, ds_mean_all[var].values, color='black', linewidth=4, label='Mean')[0]
            else:
                axs[i].plot(ds_mean_all.Time.values, ds_mean_all[var].values, color='black', linewidth=4, label='Mean')
            axs[i].set_ylim(vmin[i], vmax[i])
            

            
            # add percentiles to the plot and collecting labels
            h10, h25, h75, h90 = plot_percentiles(ds_perc_T_surf, 'T_surface', axs[0], color_shading='blue')
            hh10, hh25, hh75, hh90 = plot_percentiles(ds_perc_RH_surf, 'RH_surface', axs[1], color_shading='green')


            #axes[4,0].legend(handles, labels, loc="upper left", frameon=False, ncol=3, fontsize=16)


        elif i == 2:
            im0 = axs[i].pcolormesh(ds_mean_all.Time.values, ds_mean_all.Height.values, ds_mean_all[var].values.T,
                                        vmin = vmin[i],
                                        vmax = vmax[i], 
                                        cmap = CMAP_rr)
            colorbar = fig.colorbar(im0, ax=axs[i], orientation='vertical', 
                       fraction=0.046, aspect=4)
            colorbar.set_label(colorbar_label[i], fontsize=20)

            axs[i].vlines(ds_mean_start.Time.values[-1], 200, 1000, color = 'black', label = 'Onset of cp')
            axs[i].vlines(ds_mean_between.Time.values[-1], 200, 1000, color = 'black', linestyle = '--', label = 'Minimum of cp')

            
        
        else:
            im0 = axs[i].pcolormesh(ds_mean_all.Time.values, ds_mean_all.Height.values, ds_mean_all[var].values.T,
                                        vmin = vmin[i],
                                        vmax = vmax[i], 
                                        cmap = CMAP)
            colorbar = fig.colorbar(im0, ax=axs[i], orientation='vertical', 
                       fraction=0.046, aspect=4)
            colorbar.set_label(colorbar_label[i], fontsize=20)

            axs[i].vlines(ds_mean_start.Time.values[-1], 200, 1000, color = 'black', label = 'Onset of cp')
            axs[i].vlines(ds_mean_between.Time.values[-1], 200, 1000, color = 'black', linestyle = '--', label = 'Minimum of cp')
            
            
        
        axs[i].set_title(titles[i], loc='left', fontsize=25, fontweight='bold')
        i+=1

    # Set the x-axis ticks to the 20-minute intervals
    axs[0].set_xticks(time_ticks)
    axs[0].tick_params(axis='x', labelsize=25)    
    axs[0].set_xticklabels(custom_labels)
    axs[0].set_ylabel("Temperature [K]", fontsize=fs_axes)
    axs[1].set_ylabel("Relative \n Humidity [%]", fontsize=fs_axes)
    # plot legend to the figure outside of the first and second subplots
    handles = [h10, h25, h_mean, h75, h90, honset, hminimum]
    labels = ['10th perc', '25th perc', 'mean', '75th perc', '90th perc', 'onset cp', 'minimum cp']
    
    axs[0].legend(handles=handles, 
                labels=labels, 
                loc='upper left',
                bbox_to_anchor=(1.15, 0.8),  # Outside the figure area
                ncol=1,  # Single column
                fontsize=15)
    # Set minor ticks for first subplot (Temperature)
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(n=5))  # n=5 means 4 minor ticks between major ticks
    axs[0].tick_params(which='minor', axis='y', length=4, width=1)

    # Set minor ticks for second subplot (Relative Humidity)
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(n=5))
    axs[1].tick_params(which='minor', axis='y', length=4, width=1)

    #ax[0].set_xlabel('')
    axs[5].set_xlabel('Time [minutes]', fontsize = fs_axes)
    axs[-1].set_ylim(240,700)
    
    # Set axis labels if needed
    for ind_ax in np.arange(2,5):
        axs[ind_ax].set_ylabel("Height (m)", fontsize=fs_axes)
        axs[ind_ax].set_ylim(240., 700.)



    
    plt.savefig('/work/plots_rain_paper/fig09_Mean_of_86_coldpool_cases.png', bbox_inches="tight")
    plt.close()



def plot_percentiles(ds, var, ax, color_shading):
    
    """
    Plots the percentiles for a given variable in the dataset.
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable.
    var : str
        The name of the variable to plot.
    i : int
        The index of the subplot.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    color_shading : color for the shadowing
    """
    # plot percentiles for temperature
    h10 = ax.plot(ds.Time.values,
                ds.values[0,:],
                color='black', 
                linewidth=2,
                linestyle=':',
                label='10th percentile'
                )[0]
    h25 = ax.plot(ds.Time.values,
                ds.values[1,:],
                color='black', 
                linewidth=2,
                linestyle='--',
                label='25th percentile'
                )[0]
    
    h75 = ax.plot(ds.Time.values, 
                ds.values[3,:], 
                color='black',  # dotted linestyle 
                linewidth=2,
                linestyle='--',
                label='75th percentile')[0]
    h90 = ax.plot(ds.Time.values,
                ds.values[4,:],
                color='black', 
                linewidth=2,
                linestyle=':',
                label='90th percentile'
                )[0]
    # color the area between the 25th and the 75th percentiles
    ax.fill_between(ds.Time.values,
                        ds.values[1,:],
                        ds.values[3,:],
                        color=color_shading, alpha=0.1)
    
    return h10, h25, h75, h90
            

def shift_time_of_dataset(datasets, desired_start_time):
    
    # Step 2: Shift the selected times from all the datasets to the same time
    #Shift the time of the '1hour before' datasets to 2000-01-01T00:00:00 - 2000-01-01T01:00:00 
    
    # Define the start time and time interval
    time_interval = "10S"  # 10 seconds

    # Loop over each dataset
    for i, dss in enumerate(datasets):
        
        # Get the length of the current dataset's time coordinate
        original_time_length = len(dss.Time)
        
        # Generate the new time range with the same number of steps
        new_time = pd.date_range(start=desired_start_time, periods=original_time_length, freq=time_interval)
        
        # Check if the dataset has the expected length
        if original_time_length != len(new_time):
            raise ValueError(f"Dataset {i} has a different time length than expected.")
    
        # Replace the time coordinate with the new time range
        dss = dss.assign_coords(Time=new_time)
    
        # Save the modified dataset back to the list
        datasets[i] = dss

    return datasets


def take_percentiles(datasets, variable, percentiles=[0.25, 0.5, 0.75]):
    '''
    takes the percentiles of a list of datasets where all datasets have the same timestamps
    input:
         datasets: <list>
             list of datasets with the same time coordinates
         variable: <str>
             name of the variable to calculate percentiles for
         percentiles: <list>
             list of percentiles to calculate (default is [0.25, 0.5, 0.75])
    output:
        Array: <xarray.core.dataarray.DataArray>
            Array with the Percentiles dependent on (Time, Height)
    
    '''
    import xarray as xr
    # Step 3: Extract the variable from each dataset
    anomaly_list = [dss[variable] for dss in datasets]
    
    # Step 4: Concatenate along a new dimension called 'dataset'
    anomaly_combined = xr.concat(anomaly_list, dim='dataset')
    
    # Step 5: Calculate the percentiles along the 'dataset' dimension
    anomaly_percentiles = anomaly_combined.quantile(percentiles, dim='dataset')  

    return anomaly_percentiles


def take_mean(datasets, variable):
    '''
    takes the mean of a list of datasets where all datasets have the same timestamps
    input:
         datasets: <list>
             list of datasets with the same time coordinates
    output:
        Array: <xarray.core.dataarray.DataArray>
            Array with the Mean dependent on (Time, Height)
    
    '''
    import xarray as xr
    # 1 hour before start
    # Step 3: Extract the variable from each dataset
    anomaly_list = [dss[variable] for dss in datasets]
    
    # Step 4: Concatenate along a new dimension called 'dataset'
    anomaly_combined = xr.concat(anomaly_list, dim='dataset')
    
    # Step 5: Calculate the mean along the 'dataset' dimension
    anomaly_mean = anomaly_combined.mean(dim='dataset')  

    # Calculate the count of non-NaN values along the 'dataset' dimension
    non_nan_count = anomaly_combined.count(dim='dataset')



    return anomaly_mean, non_nan_count


def list_and_resample_and_reassign_between_tmax_tmin(ds, tmax_indices, tmin_indices, all_data):


    import xarray as xr
    import pandas as pd
    
    ## Step 1: select for each coldpool datasets containing the time between the start and minimum of cp and put them in a list 
    datasets_between = []
    for i in range(0,len(tmax_indices)-12):
        ds_sel_between = ds.isel(time=slice(tmax_indices[i], tmin_indices[i]))
        all_data_between = all_data.sel(Time=slice(ds_sel_between.time.values[0], ds_sel_between.time.values[-1]))
        datasets_between.append(all_data_between)
    
    
    ## Step 2: resample datasets to 120 timesteps
    
    # Define the target number of timesteps
    target_timesteps = 120
    
    # Loop through each dataset and resample to 120 timesteps
    for i, dss in enumerate(datasets_between):
        start_time = pd.to_datetime(dss.Time.values[0])
        end_time = pd.to_datetime(dss.Time.values[-1])
        
        # Calculate the interval in nanoseconds, then convert to seconds
        time_interval_ns = (end_time - start_time) / (target_timesteps - 1)
        time_interval_s = time_interval_ns / np.timedelta64(1, 's')
        
        # Generate the new time range with 120 points, each separated by the calculated interval
        new_time = [start_time + pd.Timedelta(seconds=j * time_interval_s) for j in range(target_timesteps)]
        
        # Interpolate dataset to the new time range
        ds_interp = dss.interp(Time=new_time)
        
        # Replace the original dataset with the resampled dataset
        datasets_between[i] = ds_interp

    
    ## Step3: reassingn the coordinates to the same timegrid e.g. 23:03, 23:05, 23:07 --> 01:01:10, 01:01:20, 01:01:30, 15:23, 15:33, 15:43 --> 01:01:10, 01:01:20, 01:01:30, ..... 
 
    # Define the target time grid (same length as the datasets' time dimension)
    start_time = "2000-01-01T01:00:00"
    target_timesteps = 120  # Number of time points to align to
    time_interval = "10S"   # Desired interval between time points
    
    # Generate the target time grid
    target_time_grid = pd.date_range(start=start_time, periods=target_timesteps, freq=time_interval)
    
    # Loop through each dataset and replace the time coordinates
    aligned_datasets = []
    for i, dss in enumerate(datasets_between):
        # Ensure the dataset has the same number of timesteps as the target time grid
        if dss.dims['Time'] != target_timesteps:
            raise ValueError(f"Dataset {i} has a different number of timesteps than expected.")
        
        # Reassign the time coordinates directly
        ds_aligned = dss.assign_coords(Time=target_time_grid)
        
        # Store the aligned dataset
        aligned_datasets.append(ds_aligned)

    return aligned_datasets


def sel_start_end_dataset(tmax_indices, tmin_indices, dataset_sel_h, ds):
    '''
    input: 
    tmax_indices: <list>
        - list with the indices of the maxima of the coldpool. The indices correspond to the time indices of ds_ship (ds)

    tmin_indices: <list>
        - list with the indices of the minima of the coldpool. The indices correspond to the time indices of ds_ship (ds)

    dataset_sel_h: <dataset>

    ds: <dataset>
    
        
    
    selcts for each coldpool the part of the dataset of 
    - 1 hour before the max of the coldpool and puts all the selected datasets in a list (datasets_start)
    - 1 hour after the min of the coldpool and puts all the selected datasets in a list (datasets_end)
    '''
    
    datasets_start = []
    datasets_end = []
    datasets_T_surface = []
    for i in range(1,len(tmax_indices)-12):
        ds_sel_start = ds.isel(time=slice(tmax_indices[i] - 61, tmax_indices[i])) # select one hour before max from ship dataset (indices correspond to ship dataset)
        dataset_sel_start = dataset_sel_h.sel(Time=slice(ds_sel_start.time.values[0], ds_sel_start.time.values[-1])) #selct the time 
        datasets_start.append(dataset_sel_start)
    
        ds_sel_end = ds.isel(time=slice(tmin_indices[i], tmin_indices[i] + 61)) # select one hour after tmin
        dataset_sel_end = dataset_sel_h.sel(Time=slice(ds_sel_end.time.values[0], ds_sel_end.time.values[-1]))
        datasets_end.append(dataset_sel_end)


    return datasets_start, datasets_end

    
def identify_coldpool_candidates(ds, threshold=-0.05):
    """
    Identify cold pool candidates based on temperature anomalies.

    Parameters:
    - ds_T : xarray.Dataset
        Dataset containing temperature data from ship pbservations.
    - threshold : float
        Temperature drop threshold to identify cold pool candidates.

    Returns:
    - index_coldpool_candidates : list of int
        Indices of cold pool candidates in the dataset.
    """

    #prepare data and search for coldpool candidates
    #select temperature
    T = ds.T

    # 0.1) resample data to one min resolution
    T_min = T.resample(time="1min").mean()


    # 0.2) 11 min running average
    T_min_float64 = T_min.astype(np.float64)
    T_fil = T_min_float64.rolling(time=11, center=True).mean().dropna("time")


    # 0.3) classify coldpool candidates: dT = T_fil(t) - T_fil(t-1) < -0.05 K
    #create dataset to save all variables in
    ds_T = T_fil.to_dataset()
    ds_T['T_fil'] = ds_T['T']
    ds_T = ds_T.drop_vars(['T'])

    # initialize a coldpoolflag. Put all entries to zero (False). -> no coldpool
    ds_T = ds_T.assign(coldpool_flag=(['time'],np.zeros(len(ds_T.time.values))))
    ds_T = ds_T.assign(dT=(['time'],np.zeros(len(ds_T.time.values))))

    T_fil_precision = T_fil.astype(np.float64) # double precision, otherwise 0.05 is too small

    index_coldpool_candidates = []
    for i in range(1,len(T_fil.time.values)-2):
        dT = T_fil_precision[i] -T_fil_precision[i-1]
        ds_T['dT'][i] = dT
        if dT < threshold:
            ds_T['coldpool_flag'][i] = 1
            index_coldpool_candidates.append(i)
            
    return index_coldpool_candidates, ds_T


def create_date_list(start_date, end_date):
    '''
    Create a list of dates from start_date to end_date, inclusive.
    Parameters:
    - start_date : datetime
        The starting date of the range.
    - end_date : datetime
        The ending date of the range.
    Returns:
    - date_list : list of datetime  
        A list of datetime objects representing each day in the range.
    '''
    
    from datetime import datetime, timedelta
     
    # Initialize an empty list
    date_list = []
     
    # Loop through the range of dates and append to the list
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += timedelta(days=1)

    return date_list

def identify_onset_of_coldpool_candidates(ds_T, index_coldpool_candidates):

    """
    Identifies the onset of cold pool events in a time-series dataset and marks them.

    The onset of the cold-pool front tmax is defined as: 
    - the last instance of δT > 0 K within 20 min before the initial abrupt temperature drop with δT < −0.05 K (coldpool_candidate). 
    - If the temperature is falling continuously in this period:
        --> tmax is choosen as the time of the maximum temperature 
    (that is, 20 min before the abrupt temperature drop). We refer to the smoothed temperature at tmax as Tmax.
    (Step 2. in the paper of Vogel et al 2021)
    
    Parameters:
    -----------
    ds_T : xarray.Dataset
        A dataset containing time-series data with temperature (`T_fil`) and temperature
        difference (`dT`) variables, along with a time coordinate. The dataset also contains
        a list of cold pool candidate indices (`index_coldpool_candidates`).

    Returns:
    --------
    ds : xarray.Dataset
        The updated dataset with a new variable `onset_cp` marking the cold pool onset times. 
        Values are set to 1 at the onset times and 0 otherwise.
    
    tmax_indexes : list of int
        A list of indices corresponding to the times of cold pool onsets in the dataset.
    """


    ds_T = ds_T.assign(onset_cp=(['time'],np.zeros(len(ds_T.time.values))))
    
    ds = ds_T
    tmax_indexes = []
    for i in index_coldpool_candidates:

        # Get the time of the abrupt drop
        abrupt_drop_time = ds['time'][i]
    
        # Create a window for 20 minutes before the abrupt drop
        start_time = abrupt_drop_time - np.timedelta64(20, 'm')
        time_window = ds.sel(time=slice(start_time, abrupt_drop_time))

        
        # 1. Find tmax (the onset of the cold pool)
        # Find the last instance of δT > 0 K within the 20-minute window
        tmax_condition = time_window['dT'] > 0
        tmax_times = time_window['time'].where(tmax_condition, drop=True)
        
        if len(tmax_times) > 0: #if tmax_times exists 
            # tmax is the last time where δT > 0 K
            tmax = tmax_times[-1]
            tmax = tmax.values

        
        # 2. If temperature is falling continuously, tmax is the time of the maximum temperature
        else:
            tmax_index_time_window = time_window['T_fil'].argmax(dim='time')
            tmax = time_window.time.values[tmax_index_time_window]
        
        # Only append the index of the onset if the index does not already exist 
        # (Different coldpool candidates have the same onset, because the temperature normally drops longer than just one timestamp)
        tmax_index = np.where(ds['time'] == tmax)[0][0]  
        if tmax_index not in tmax_indexes:
            tmax_indexes.append(tmax_index)
            
        # save the time of the onset
        ds['onset_cp'].loc[{'time': tmax}] = 1

    return ds, tmax_indexes



def find_minimum_of_coldpool(ds, ds_T, tmax_indices):
    '''
    
    Identify the minimum Temperature of T_fil (Tmin) of a coldpool (tmin), after tmax (the onset of the coldpool has already been identified)
    (Relates to the second step of identifying coldpools from the paper Vogel et. al. 2021)

    tmin is identified as:
    - The minimum of contiguous temperature minima in the dataset after each tmax time.
    - If there is a subsequent candidate cold pool within 20 minutes of a previous minimum, the two cold pools are 
      combined if the temperature in between does not rise by more than 0.5 K above the previous minimum.
    
    Parameters:
    - ds : xarray.Dataset
        The dataset containing temperature data and cold pool candidate indicators.
    - tmax_indices : list of int
        List of indices representing the onset times (tmax) of cold pool events.
    
    Returns:
    - ds : xarray.Dataset
        The updated dataset with an added 'minimum_of_cp' variable indicating Tmin for each cold pool event.
    - tmin_indices : list of int
        List of indices representing the minimum times (tmin) of each cold pool.
    
    Notes:
    - The function first finds Tmin after each tmax by identifying when `T_fil` starts rising after reaching a minimum.
    - If a subsequent cold pool candidate is detected (δT < -0.05 K) within 20 minutes of tmin, the two events are combined 
      if the temperature between the two events does not exceed Tmin by more than 0.5 K.
    - If cold pool events are combined, the later tmin is used as the effective minimum time, and the earlier tmax 
      is retained as the cold pool onset time.
    

    '''  
    ds = ds.assign(minimum_of_cp=(['time'],np.zeros(len(ds_T.time.values))))
    tmin_indices = []
    combine_cold_pool = 0
    i = 0
    pop = 0
    while i <= (len(tmax_indices)-1):
    
        tmax_index = tmax_indices[i]

        ### 1. search for minimum of contiguous temperature minima. 
        # Select the data after the tmax time and look when dT > 0 (T_fil rises)
        tmax_time = ds.time.values[tmax_index]
        data_after_tmax = ds.sel(time=slice(tmax_time+1, tmax_time + np.timedelta64(500, 'm')))  # 500 min after onset of coldpool should be more than enough (normal duration of cp 2-3h)
        condition = data_after_tmax['dT'] >= 0
        indices_of_rise = np.nonzero(condition.values)[0]
        index_first_rise = indices_of_rise[0] # index of data_after_max
        
        # time of minimum
        tmin_time = data_after_tmax.time.values[index_first_rise]

        # value of minimum
        tmin_value = data_after_tmax.T_fil.values[index_first_rise]
        tmin_index = ds.get_index('time').get_loc(tmin_time)

        # only save, if the minimum does not already exist
        if tmin_index not in tmin_indices:
            tmin_indices.append(tmin_index)  
            ds['minimum_of_cp'].loc[{'time': tmin_time}] = 1 # mark it in the big dataset
        #if minimum exists already, purge the onset of the coldpool from the onset list. 
        # --> then the earliest onset point will be taken (20min before abrupt temperature drop)
        else:
            tmax_indices.pop(i)
            ds['onset_cp'].loc[{'time': tmax_time}] = 0
            i = i -1
    
        #if previous coldpool shall be combined with actual one, 
        # tmax(onset) stays the same (from first coldpool) --> delete the second tmax
        # tmin is the second tmin (delete the first tmin)
        if combine_cold_pool == 1:
            ds['minimum_of_cp'].loc[{'time': tmin_time_previous}] = 0
            ds['onset_cp'].loc[{'time': tmax_time}] = 0
            tmax_indices.pop(i)
            tmin_indices.pop(-2) # the indices not from this time, but before can be deleted
            i = i - 1

        
        ### 2. Subsequent candidate cold pools with δT < −0.05 K occurring within 20 min of the previous minimum are combined 
        # if the temperature does not rise by more than 0.5 K above the previous minimum in between
        #search for other coldpool event within 20 min of minima
        combine_cold_pool = 0 # variable that indicates if we combine two cp events in thze next loop or not
        # Select data starting from tmax
        data_after_tmin = ds.sel(time=slice(tmin_time, (tmin_time + np.timedelta64(20, 'm')) ))
        # Find where 'coldpoolflag' equals 1
        cp_condition = data_after_tmin['coldpool_flag'] == 1
        # Get the indices where the condition is met (True)
        indices_of_cp = cp_condition.where(cp_condition, drop=True)

    
        # Step 3: Get the index of the first occurrence where the flag is 1
        if len(indices_of_cp['time']) > 0:
            # Get the time of the first occurrence
            next_cp_timestamp = indices_of_cp['time'][0]
        
            #check if T rises over 0.5K in between minimum_time and coldpool_candidate_time
            # --> search for maximum between minimum_time and coldpool_candidate_time and check if Tmax - Tmin > 0.5
            #  seach for maxima
            # Select the data between time of minimum and the next coldpool candidate
            data_between_tmin_coldpool_candidate = ds.sel(time=slice(tmin_time, next_cp_timestamp))  
            # Find the index of the maximum value after tmax
            interval_tmax_index = data_between_tmin_coldpool_candidate['T_fil'].argmax(dim='time')
            # Get the corresponding time for the minimum
            interval_tmax_time = data_between_tmin_coldpool_candidate['time'][interval_tmax_index]
            # Get the maximum value itself 
            interval_tmax_value = data_between_tmin_coldpool_candidate['T_fil'].isel(time=interval_tmax_index)
 
            if interval_tmax_value.values - tmin_value <= 0.5:
                pop +=1
                #combine current coldpool with next coldpool candidate
                combine_cold_pool = 1
                tmin_time_previous = tmin_time    

        i += 1

    
    return ds, tmin_indices, tmax_indices


def coldpool_end(ds, tmax_indices, tmin_indices):
    '''
    Identify the end time (`tend`) of each cold pool in the dataset, based on the filtered temperature series(T_fil).
    
    tend is determined using three conditions, in sequence:
    
    1. Condition (a): `tend` is defined as the time when the temperature first exceeds `Tmin + ΔT / e`, where:
       - ΔT is the difference between the maximum (`Tmax`) and minimum (`Tmin`) temperatures within the cold pool.
       - e is Euler's number.
    2. Condition (b): If another cold pool begins before condition (a) is met, `tend` is set to the onset of that next cold pool.
    3. Condition (c): If any temperature between `tmin` and the current `tend` is below `Tmin - 0.15 K`, `tend` is redefined as the 
       time when temperature first decreases after a period of increase following `tmin`.
    
    Cold pools that end based on condition (a) are labeled as "recovered" in the dataset.

    Parameters:
    - ds : xarray.Dataset
        The dataset containing temperature data, cold pool candidates, and identified `tmax` and `tmin` values.
    - tmax_indices : list of int
        List of indices representing the onset times (tmax) for each cold pool.
    - tmin_indices : list of int
        List of indices representing the times of minimum temperature (tmin) for each cold pool.
    
    Returns:
    - ds : xarray.Dataset
        The updated dataset with two additional variables:
        - `end_of_cp`: Indicates the end time of each cold pool (1 at tend, 0 elsewhere).
        - `recovered_cp`: Indicates whether the cold pool has "recovered" (1 if true, 0 otherwise).
    - tend_indices : list of int
        List of indices representing the end times (tend) for each cold pool.
    
    Steps:
    - For each cold pool, calculate the total temperature drop (`ΔT`) and determine `tend` using conditions (a) and (b).
    - If `tend` defined by either condition (a) or (b) results in temperatures below `Tmin - 0.15 K`, use condition (c) to redefine `tend`.
    - Label the cold pool as recovered if `tend` was determined by condition (a).
    '''



    ds = ds.assign(end_of_cp=(['time'],np.zeros(len(ds.time.values))))
    ds = ds.assign(recovered_cp=(['time'],np.zeros(len(ds.time.values))))
    tend_indices = []
    for i in range(0,len(tmax_indices)-1):
        
        tmax_index = tmax_indices[i] 
        tmin_index = tmin_indices[i] 
        Tmax = ds.T_fil.values[tmax_index]
        Tmin = ds.T_fil.values[tmin_index]
        

        # Condition (a): Time when temperature exceeds Tmin + ΔT / e
        # Calculate ΔT = Tmax - Tmin
        delta_T = Tmax - Tmin
        T_threshold = Tmin + delta_T / np.e
        data_after_tmin = ds.isel(time=slice(tmin_index, None))
        exceeds_threshold = data_after_tmin['T_fil'] > T_threshold

        if exceeds_threshold.any():
            tend_a = data_after_tmin['time'].where(exceeds_threshold, drop=True).isel(time=0)
            tend_a = tend_a.values #to get only timestamp and not array
            tend = tend_a
            end = 'a'
        else:
            tend_a = None
            
        
        # Condition (b): Onset of the next cold pool (if provided)
        # only if onset is behind minimum. If onset of next coldpool is between current onset and minimum 
        begin_next_onset =  ds.time.values[tmax_indices[i+1]] 
        l = i
        while begin_next_onset < ds.time.values[tmin_indices[i]]:
            if l+1 == len(tmax_indices):
                begin_next_onset = None
                break    

            begin_next_onset = ds.time.values[tmax_indices[l+1]] 
            l = l+1
            
        tend_b = begin_next_onset
        
       
        # First, use either condition (a) or (b) to define tend, if both exist
        if tend_a and tend_b:
            tend = min(tend_a, tend_b)  # Use the earlier of the two
        elif tend_a:
            tend = tend_a
        elif tend_b:
            tend = tend_b
            end = 'b'
        else:
            print("Neither condition (a) nor (b) could define tend")
            tend_indices.append(np.nan)
                  
        
        # Condition (c): Check if any temperature between tmin and tend is smaller than Tmin - 0.15 K
        Tmin_minus_015 = Tmin - 0.15
        tend_index = ds.get_index('time').get_loc(tend)
        data_between_tmin_tend = ds.isel(time=slice(tmin_index, tend_index))
        below_threshold = data_between_tmin_tend['T_fil'] < Tmin_minus_015
        
        if below_threshold.any():
            # Condition (c) defines tend: the time when temperature first decreases after increasing
            increasing = data_between_tmin_tend['T_fil'].diff(dim='time') > 0  # Find where temperature is increasing
            decrease_after_increase = (~increasing).cumsum(dim='time') == 1  # Find the first decrease after increase
            tend_c = data_between_tmin_tend['time'].where(decrease_after_increase, drop=True).isel(time=0)
            tend = tend_c.values  # Redefine tend according to condition (c)
            tend_index = ds.get_index('time').get_loc(tend)
            end = 'c'

    
        #save tend in dataset

        ds.end_of_cp.values[tend_index] = 1
        tend_indices.append(tend_index)
        
        # Determine if the cold pool is recovered (if tend was defined by condition (a))
        if tend == tend_a:
            ds.recovered_cp[tend_index].values = 1


    return ds, tend_indices




if __name__ == "__main__":
    main()
   