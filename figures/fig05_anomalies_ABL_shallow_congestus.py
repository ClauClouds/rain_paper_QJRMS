"""
plot profiles of anomalies for T, MR, VW for shallow and congestus clouds in precipitating and  non precipitating conditions
Procedure:
- read anomaly data
- read flags data to select rain and no rain
- plot mean +- std profiles of variables
"""

from readers.cloudtypes import read_rain_flags, read_cloud_class
from readers.lidars import f_read_merian_data, read_anomalies
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def main():
        
    # read cloud classification
    all_flags = read_rain_flags()
    
    # read shape from classification original file
    cloud_properties = read_cloud_class()  
    cloud_properties['flag_rain_ground'] = (('time'), all_flags.flag_rain_ground.values)
    
    
    # prepare data (calculate mean and std of the profiles for shallow/congestus in prec and non prec)
    T_anomalies = prepare_anomaly_profiles(cloud_properties, "T")
    MR_anomalies = prepare_anomaly_profiles(cloud_properties, "MR")
    VW_anomalies = prepare_anomaly_profiles(cloud_properties, "VW")

    
    # visualize anomalies
    visualize_anomalies(T_anomalies, MR_anomalies, VW_anomalies)
    
    
    
    
    
    
def prepare_anomaly_profiles(cloud_properties, var_string):
    """
    function to calculate mean and std of shallow and congestus profiles for prec and non prec conditiosn

    Args:
        cloud_properties (xarray dataset): contains cloud classification
        rain_flags (xarray dataset): contains rainy flags and 
    """
    # build the name of the variables
    anomaly_var = var_string+'_anomaly'
    
    
    # reading arthus data anomalies
    dataset_arthus_anomaly = read_anomalies(var_string)

    # interpolate classification of clouds over anomalies time stamps
    class_interp = cloud_properties.interp(time=dataset_arthus_anomaly.time.values, method='nearest')
                                                                     
                                                                    
    # merging datasets obtained (classification and anomalies)
    merian_dataset = xr.merge([class_interp, dataset_arthus_anomaly])  

    # selecting rain data
    ind_rain = np.where((class_interp.flag_rain_ground.values == 1.))[0]
    data_rain = merian_dataset.isel(time=ind_rain)
    
    # segregating with respect to shallow and deep clouds
    ind_rain_shallow = np.where((data_rain.shape.values == 0))[0]
    data_rain_shallow = data_rain.isel(time=ind_rain_shallow)
    ind_rain_deep = np.where((data_rain.shape.values == 1))[0]
    data_rain_deep = data_rain.isel(time=ind_rain_deep)

    #calculate mean for rainy data for shallow and congestus
    mean_r_s = data_rain_shallow.mean(dim='time')
    std_r_s = data_rain_shallow.std(dim = 'time')
    mean_r_d = data_rain_deep.mean(dim='time')
    std_r_d = data_rain_deep.std(dim = 'time')
    
    # selecting rain free data
    ind_no_rain = np.where((merian_dataset.flag_rain_ground.values == 0))[0]
    data_no_rain = merian_dataset.isel(time=ind_no_rain)
    
    
    # segregating with respect to shallow and deep clouds
    ind_no_rain_shallow = np.where((data_no_rain.shape.values == 0))[0]
    data_no_rain_shallow = data_no_rain.isel(time=ind_no_rain_shallow)
    ind_no_rain_deep = np.where((data_no_rain.shape.values == 1))[0]
    data_no_rain_deep = data_no_rain.isel(time=ind_no_rain_deep)
    
    #calculate mean
    mean_nr_s = data_no_rain_shallow.mean(dim='time')
    std_nr_s = data_no_rain_shallow.std(dim = 'time')
    mean_nr_d = data_no_rain_deep.mean(dim='time')
    std_nr_d = data_no_rain_deep.std(dim = 'time')
        
    # create dataset for output
    output_dataset = xr.Dataset(
        data_vars = {
            "mean_r_s" :(('height'), mean_r_s[anomaly_var].values),
            "std_r_s"  :(('height'), std_r_s[anomaly_var].values),
            "mean_r_d" :(('height'), mean_r_d[anomaly_var].values),
            "std_r_d"  :(('height'), std_r_d[anomaly_var].values),
            "mean_nr_s":(('height'), mean_nr_s[anomaly_var].values),
            "std_nr_s" :(('height'), std_nr_s[anomaly_var].values),
            "mean_nr_d":(('height'), mean_nr_d[anomaly_var].values),
            "std_nr_d" :(('height'), std_nr_d[anomaly_var].values),
        }, 
        coords = {
            "range":(('height',), data_no_rain.height.values)
        },
        attrs={'CREATED_BY'     : 'Claudia Acquistapace',
                         'ORCID-AUTHORS'   : '0000-0002-1144-4753', 
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
                        'DATA_DESCRIPTION' : 'Anomalies of '+var_string+' with respect to diurnal cycle calculated over entire campaign',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing lidar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                        'DATA_SOURCE'      : 'arthus raman lidar data postprocessed',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/rain_paper_QJRMS',
                        'INSTRUMENT_MODEL' : 'Arthus Raman Lidar',
                        'COMMENT'          : 'Arthus Raman lidar mentors: Diego Lange, Arthus Raman lidar owned by Uni Hohenheim' }
    )
    return output_dataset
    
def visualize_anomalies(T_anomalies, MR_anomalies, VW_anomalies):
    """
    function to plot anomalies profile in a composite plot for the publication figure number 5

    Args:
        T_anomalies (xarray dataset): anomaly mean and std profiles of T observations for rainy and non rainy shallow and congestus
        MR_anomalies (xarray dataset): anomaly mean and std profiles of MR observations for rainy and non rainy shallow and congestus
        VW_anomalies (xarray dataset): anomaly mean and std profiles of VW observations for rainy and non rainy shallow and congestus
    """
    
    fig, axs = plt.subplots(2,3, figsize=(10,7), sharey=True)

    # plot settings
    linewidth = 3
    color_shallow = '#ff9500'
    color_congestus = '#008080'   
    
    
    # loop on first line of subplots
    for ax, var2plot in zip(axs[0,:].flatten(), ['T', 'MR', 'VW']):
        
        # call function to derive settings and dataset to plot
        label, xmin, xmax, prof_an = settings_var2plot(var2plot, T_anomalies, MR_anomalies, VW_anomalies)
      
        # plot mean profiles and uncertainties
        ax.plot(prof_an.mean_nr_s.values, prof_an.range.values, color_shallow, label = 'shallow', linewidth=linewidth)
        ax.plot(prof_an.mean_nr_d.values, prof_an.range.values, color_congestus, label = 'congestus', linewidth=linewidth)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_nr_s.values - prof_an.std_nr_s.values, prof_an.mean_nr_s.values,  facecolor=color_shallow, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_nr_s.values + prof_an.std_nr_s.values, prof_an.mean_nr_s.values,  facecolor=color_shallow, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_nr_d.values - prof_an.std_nr_d.values, prof_an.mean_nr_d.values,  facecolor=color_congestus, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_nr_d.values + prof_an.std_nr_d.values, prof_an.mean_nr_d.values,  facecolor=color_congestus, alpha = 0.3)
        ax.set_xlim(xmin,xmax)
        ax.set_xlabel(label, fontsize=10)              
        
    for ax, var2plot in zip(axs[1,:].flatten(), ['T', 'MR', 'VW']):
            
        # call function to derive settings and dataset to plot
        label, xmin, xmax, prof_an = settings_var2plot(var2plot, T_anomalies, MR_anomalies, VW_anomalies)
        
        # plot mean profiles and uncertainties
        ax.plot(prof_an.mean_r_s.values, prof_an.range.values, color_shallow, label = 'shallow', linewidth=linewidth)
        ax.plot(prof_an.mean_r_d.values, prof_an.range.values, color_congestus, label = 'congestus', linewidth=linewidth)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_r_s.values + prof_an.std_r_s.values, prof_an.mean_r_s.values,  facecolor=color_shallow, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_r_s.values - prof_an.std_r_s.values, prof_an.mean_r_s.values,  facecolor=color_shallow, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_r_d.values - prof_an.std_r_d.values, prof_an.mean_r_d.values,  facecolor=color_congestus, alpha = 0.3)
        ax.fill_betweenx(prof_an.range.values, prof_an.mean_r_d.values + prof_an.std_r_d.values, prof_an.mean_r_d.values,  facecolor=color_congestus, alpha = 0.3)
        ax.set_xlim(xmin,xmax)
        ax.set_xlabel(label, fontsize=10)   
        
    # loop on all subplots to fix axes limits and line shapes
    for ax, l in zip(axs.flatten(), ['Non precipitating', ' ', ' ', 'Precipitating', '', '']):
        ax.text(-0.05, 1.07, l,  fontweight='black', fontsize=10, transform=ax.transAxes)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.tick_params(which='minor', length=5, width=1, labelsize = 10)
        ax.tick_params(axis='both', labelsize=10)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_ylim(200.,1000.)
        ax.grid(color='grey', alpha=0.3)

    axs[0,0].set_ylabel('Height [m]', fontsize=10)     
    axs[1,0].set_ylabel('Height [m]', fontsize=10)     

    fig.tight_layout()
    
    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "fig05_ABL_anomalies.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

        
        
        
def settings_var2plot(var2plot, T_anomalies, MR_anomalies, VW_anomalies):
    '''
    define settings for plot for the specific variable
    
    returns:
    - label for x axis
    - xrange min
    - xrange max
    - dataset of the selected variable 

    '''
    
    # selecting variables to plot and corresponding xarray dataset
    if var2plot == 'T':
        label = 'Anomaly Temperature [K]'
        xmin = -10
        xmax = 10
        prof_an = T_anomalies
    elif var2plot == 'MR':
        label = 'Anomaly Mixing Ratio [g kg$^{-1}$]'
        xmin = -5
        xmax = 5
        prof_an = MR_anomalies
    elif var2plot == 'VW':
        label = 'Anomaly VW [m s$^{-1}$]'
        xmin = -2
        xmax = 2
        prof_an = VW_anomalies
    
    return label, xmin, xmax, prof_an
    
    
    
    
    
if __name__ == "__main__":
    main()
    