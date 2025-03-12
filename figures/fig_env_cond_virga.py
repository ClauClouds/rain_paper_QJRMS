"""
plot environmental conditions as a function of virga depths

"""


from readers.cloudtypes import read_cloud_class, read_rain_flags
from readers.lidars import read_anomalies_and_rename

import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from figures.mpl_style import CMAP, CMAP_an
import os
import matplotlib.ticker as ticker
import pdb

def main():
        
    # read virga depths
    flags = read_rain_flags()
    
    # read cloud classification 
    cloud_properties = read_cloud_class()
    
    # add variable virga depth to the cloud properties flags with shallow/congestus
    cloud_properties['flag_rain_ground'] = (('time'), flags.flag_rain_ground.values)
    cloud_properties['flag_rain'] = (('time'), flags.flag_rain.values)
    cloud_properties['virga_depth'] = (('time'), flags.virga_depth.values)
    cloud_properties['cloud_base'] = (('time'), flags.cloud_base.values)    

    # read anomalies for T, MR, VW
    T_anomaly  = read_anomalies_and_rename('T')
    MR_anomaly = read_anomalies_and_rename('MR')
    VW_anomaly = read_anomalies_and_rename('VW')
    
    # aligning anomalies and classification of clouds 
    T, MR, VW = xr.align(T_anomaly, MR_anomaly, VW_anomaly)      
    cloud_prop = cloud_properties.interp(time=T.time.values, method='nearest')
    
    # merging datasets obtained (classification and anomalies)
    merian_dataset = xr.merge([T, MR, VW, cloud_prop], compat='override')

    # print all variable names in merian dataset to check if everything is correct
    print(merian_dataset.variables)
    
    # constructing boolean arrays for shallow and congestus clouds
    is_shallow = cloud_prop.shape.values == 0
    is_no_surf_rain = cloud_prop.flag_rain_ground.values == 0
    is_rain_air = cloud_prop.flag_rain.values == 1
    is_congestus = cloud_prop.shape.values == 1
    
    # finding indeces of virga shallow
    virga_shallow = is_no_surf_rain & is_shallow & is_rain_air
    virga_congestus = is_no_surf_rain & is_congestus & is_rain_air
    
    # selecting shallow and congestus datasets
    anomalies_shallow = merian_dataset.isel(time=virga_shallow)
    anomalies_congestus = merian_dataset.isel(time=virga_congestus)
    print(len(anomalies_shallow.time.values), 'number of shallow virga cases')
    print(len(anomalies_congestus.time.values), 'number of congestus virga cases')
    pdb.set_trace()

    # calculating virga depht bins and labels
    virga_depth_binned_s, virga_depth_labels_s = f_calc_virga_bins(0., 250., 20.)
    virga_depth_binned_d, virga_depth_labels_d = f_calc_virga_bins(0., 750., 80.)
    
    # if file not existing then run the code
    if not os.path.exists('/work/plots_rain_paper/ncdf_ancillary_files_QJRMS/shallow_dataset_virga_env.nc'):
        print('calculating shallow dataset')
        shallow_dataset = f_calc_mean_std_arthus_profiles(anomalies_shallow, virga_depth_binned_s)
        shallow_dataset.to_netcdf('/net/ostro/plots_rain_paper/ncdf_ancillary_files_QJRMS/shallow_dataset_virga_env.nc')
    else:
        print('reading shallow dataset')
        shallow_dataset = xr.open_dataset('/work/plots_rain_paper/ncdf_ancillary_files_QJRMS/shallow_dataset_virga_env.nc')
    if not os.path.exists('/work/plots_rain_paper/ncdf_ancillary_files_QJRMS/congestus_dataset_virga_env.nc'):
        print('calculating congestus dataset')
        congestus_dataset = f_calc_mean_std_arthus_profiles(anomalies_congestus,virga_depth_binned_d)
        congestus_dataset.to_netcdf('/work/plots_rain_paper/ncdf_ancillary_files_QJRMS/congestus_dataset_virga_env.nc')
    else:
        print('reading congestus dataset')
        congestus_dataset = xr.open_dataset('/work/plots_rain_paper/ncdf_ancillary_files_QJRMS/congestus_dataset_virga_env.nc')
   
   
    # visualize plot
    visualize_env_conditions_virga(shallow_dataset, congestus_dataset)
        
    # plot counts
    visualize_counts(shallow_dataset, congestus_dataset)
        
        
        
        
        
        
        
        
def f_calc_virga_bins(virga_depth_min, virga_depth_max, virga_depth_bin):
    """
    function to calculate virga depth bnned and virga depth labels

    Args:
        virga_depth_min (_type_): _description_
        virga_depth_max (_type_): _description_
        virga_depth_bin (_type_): _description_
    """
    virga_depth_binned = np.arange(virga_depth_min, virga_depth_max, virga_depth_bin)
    virga_depth_labels = [(virga_depth_binned[i]+virga_depth_binned[i+1])/2. for i in range(len(virga_depth_binned)-1)]
    
    return(virga_depth_binned, virga_depth_labels)   
    
    
    
def f_calc_mean_std_arthus_profiles(data_virga, virga_depth_binned):
    """
    function to calculate mean profiles for each bin of virga depths
     
    Args:
        data_virga (xarray dataset): dataset with anomalies profiles
        virga_depth_binned (ndarray): array of virga depth bins
        virga_depth_bin (ndarray): bin sizes for splitting virga depths
    """

    height = data_virga.height.values
    T_profiles = np.zeros((len(virga_depth_binned)-1, len(height)))
    T_profiles.fill(np.nan)
    MR_profiles = np.zeros((len(virga_depth_binned)-1, len(height)))
    MR_profiles.fill(np.nan)
    VW_profiles = np.zeros((len(virga_depth_binned)-1, len(height)))
    VW_profiles.fill(np.nan)
    CB_mean = np.zeros((len(virga_depth_binned)-1))
    CB_std = np.zeros((len(virga_depth_binned)-1))

    T_count = np.zeros((len(virga_depth_binned)-1, len(height)))
    MR_count = np.zeros((len(virga_depth_binned)-1, len(height)))
    VW_count = np.zeros((len(virga_depth_binned)-1, len(height)))
    T_count.fill(np.nan)
    MR_count.fill(np.nan)
    VW_count.fill(np.nan)
    
    # calculating mean T and MR profiles for each virga_depth bin
    for ind_depth in range(len(virga_depth_binned)-1):
        
        virga_min = virga_depth_binned[ind_depth]
        virga_max = virga_depth_binned[ind_depth+1]

        #selecting time indeces corresponding to the bin of virga depths\
        ind_virga_cols = np.where((data_virga.virga_depth.values > virga_min) * \
                                    (data_virga.virga_depth.values <= virga_max))[0]
        if len(ind_virga_cols) == 0:
            continue
        data_bin = data_virga.isel(time=ind_virga_cols)

        # calculating mean and std T and MR profiles
        T_count[ind_depth,:] = np.count_nonzero(~np.isnan(data_bin.T_anomaly.values), axis=0)
        MR_count[ind_depth,:] = np.count_nonzero(~np.isnan(data_bin.MR_anomaly.values), axis=0)
        VW_count[ind_depth,:] = np.count_nonzero(~np.isnan(data_bin.VW_anomaly.values), axis=0)
        CB_mean[ind_depth] = np.nanmean(data_bin.cloud_base.values)
        CB_std[ind_depth] = np.nanstd(data_bin.cloud_base.values)
        T_profiles[ind_depth, :] = np.nanmean(data_bin.T_anomaly.values, axis=0)
        MR_profiles[ind_depth, :] = np.nanmean(data_bin.MR_anomaly.values, axis=0)
        VW_profiles[ind_depth, :] = np.nanmean(data_bin.VW_anomaly.values, axis=0)
        
    # create dataset for output
    output_dataset = xr.Dataset(
        data_vars = {
            "T_count"    :(('bins', 'height'), T_count),
            "MR_count"   :(('bins', 'height'), MR_count),
            "VW_count"   :(('bins', 'height'), VW_count),
            "CB_mean"    :(('bins'), CB_mean),
            "CB_std"     :(('bins'), CB_std),
            "T_profiles" :(('bins', 'height'), T_profiles),
            "MR_profiles":(('bins', 'height'), MR_profiles),
            "VW_profiles":(('bins', 'height'), VW_profiles),
        }, 
        coords = {
            "virga_bins": (('bins',), virga_depth_binned[:-1]),
            "range":(('height',), data_virga.height.values)
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
                        'DATA_DESCRIPTION' : 'Anomalies of T, MR, VW and mean cloud base for bins of virga depths',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing lidar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                        'DATA_SOURCE'      : 'arthus raman lidar data postprocessed',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/rain_paper_QJRMS',
                        'INSTRUMENT_MODEL' : 'Arthus Raman Lidar',
                        'COMMENT'          : 'Arthus Raman lidar mentors: Diego Lange, Arthus Raman lidar owned by Uni Hohenheim' }
    )
    
    return output_dataset

    
def visualize_env_conditions_virga(shallow_dataset, congestus_dataset):
    """
    plot of T, MR, VW anomalies as a function of virga depths for shallow and congestus clouds

    Args:
        shallow_dataset (xarray dataset): dataset containing binned profiles of t, mr, vw for shallwo clouds
        congestus_dataset (xarray dataset): dataset containing binned profiles of t, mr, vw for congestus clouds
    """

    fig, axs = plt.subplots(3,2, figsize=(25,20))
    thr_arthus = 100.
    thr_DL = 100.
    aspect_val = 10.
    T_count_filter_s = np.ma.masked_where(shallow_dataset.T_count.values > 25., shallow_dataset.T_count.values)
    MR_count_filter_s = np.ma.masked_where(shallow_dataset.MR_count.values > 25., shallow_dataset.MR_count.values)
    VW_count_filter_s = np.ma.masked_where(shallow_dataset.VW_count.values > 20., shallow_dataset.VW_count.values)
    T_count_filter_c = np.ma.masked_where(congestus_dataset.T_count.values > 200, congestus_dataset.T_count.values)
    MR_count_filter_c = np.ma.masked_where(congestus_dataset.MR_count.values > 200, congestus_dataset.MR_count.values)
    VW_count_filter_c = np.ma.masked_where(congestus_dataset.VW_count.values > 200, congestus_dataset.VW_count.values)
    
    
    vmin_T = -2
    vmax_T = 2
    mesh_T_s = axs[0,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.T_profiles.values.T, 
                                   cmap=CMAP_an)
    axs[0,0].pcolor(shallow_dataset.virga_bins.values, 
                    shallow_dataset.range.values,
                    T_count_filter_s.T, 
                    hatch='//', 
                    alpha=0.)

    mesh_T_d = axs[0,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.T_profiles.values.T, 
                                   vmin=vmin_T, 
                                   vmax=vmax_T, 
                                   cmap=CMAP_an)
    
    axs[0,1].pcolor(congestus_dataset.virga_bins.values, 
                    congestus_dataset.range.values,
                    T_count_filter_c.T, 
                    hatch='//', 
                    alpha=0.) 
    
    cbar = fig.colorbar(mesh_T_d, ax=axs[0], orientation='vertical', aspect=aspect_val)
    cbar.set_label('Anomaly \n Air Temperature [K]', fontsize=25)
    cbar.ax.tick_params(labelsize=25) 
    
    vmin_MR = -2
    vmax_MR = 2
    mesh_MR_s = axs[1,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.MR_profiles.values.T, 
                                   vmin=vmin_MR, 
                                   vmax=vmax_MR, 
                                   cmap=CMAP_an)
    
    axs[1,0].pcolor(shallow_dataset.virga_bins.values, 
                    shallow_dataset.range.values,
                    MR_count_filter_s.T, 
                    hatch='//', 
                    alpha=0.) 
    
    mesh_MR_d = axs[1,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.MR_profiles.values.T, 
                                   vmin=vmin_MR, 
                                   vmax=vmax_MR, 
                                   cmap=CMAP_an)
    
    axs[1,1].pcolor(congestus_dataset.virga_bins.values, 
                    congestus_dataset.range.values,
                    MR_count_filter_c.T, 
                    hatch='//', 
                    alpha=0.) 

    cbar = fig.colorbar(mesh_MR_d, ax=axs[1], orientation='vertical', aspect=aspect_val)
    cbar.set_label('Anomaly \n Mixing Ratio [g kg$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25) 

    vmin_VW = -1
    vmax_VW = 1
    mesh_VW_s = axs[2,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.VW_profiles.values.T, 
                                   vmin=vmin_VW, 
                                   vmax=vmax_VW, 
                                   cmap=CMAP_an)

    axs[2,0].pcolor(shallow_dataset.virga_bins.values, 
                    shallow_dataset.range.values,
                    VW_count_filter_s.T, 
                    hatch='//', 
                    alpha=0.)     
    
    mesh_VW_d = axs[2,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.VW_profiles.values.T, 
                                   vmin=vmin_VW, 
                                   vmax=vmax_VW, 
                                   cmap=CMAP_an)
    
    axs[2,1].pcolor(congestus_dataset.virga_bins.values, 
                    congestus_dataset.range.values,
                    VW_count_filter_c.T, 
                    hatch='//', 
                    alpha=0.) 
        
    cbar = fig.colorbar(mesh_VW_d, ax=axs[2], orientation='vertical', aspect=aspect_val)
    cbar.set_label('Anomaly \n Vertical wind [m s$^{-1}$]', fontsize=25)
    cbar.ax.tick_params(labelsize=25) 
    
    #cbar_VW = fig.colorbar(mesh_VW_d, 
    #                      ax=axs[2], 
    #                      orientation='vertical', 
    #                      label='Anomaly \n Vertical wind [m s$^{-1}$]')
    #cbar_VW.
    
    for ax, l in zip(axs[:,:].flatten(), ['a) T anomaly shallow',  'b) T anomaly congestus',\
                                          'c) MR anomaly shallow', 'd) MR anomaly congestus',\
                                              'e) VW anomaly shallow', 'f) VW anomaly congestus']):
        ax.text(-0.05, 1.04, l,  fontweight='black', fontsize=25, transform=ax.transAxes)
    
    
    for i in range(0,3):
        for j in range(0,2):
            axs[i,j].spines["top"].set_visible(False)
            axs[i,j].spines["right"].set_visible(False)
            axs[i,j].spines["bottom"].set_linewidth(3)
            axs[i,j].spines["left"].set_linewidth(3)
            axs[i,j].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
            axs[i,j].tick_params(which='minor', length=5, width=2, labelsize = 5)
            axs[i,j].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
            axs[i,j].tick_params(axis='both', labelsize=20)
            axs[i,j].get_xaxis().tick_bottom()
            axs[i,j].get_yaxis().tick_left()
            axs[i,0].set_ylim(250.,1100.)
            axs[i,1].set_ylim(250.,2000.)
            axs[i,j].grid(False)
            axs[i,0].set_xlim(30., 220.)
            axs[i,1].set_xlim(80., 600.)
            axs[2,j].set_xlabel('Virga depth [m]', fontsize=20)
            axs[i,j].set_ylabel('Height [m]', fontsize=20)
            axs[i,0].plot(shallow_dataset.virga_bins.values, 
                          shallow_dataset.CB_mean.values, 
                          color='black', 
                          linestyle='--')

            axs[i,0].fill_between(shallow_dataset.virga_bins.values, 
                     shallow_dataset.CB_mean.values + shallow_dataset.CB_std.values, 
                     shallow_dataset.CB_mean.values - shallow_dataset.CB_std.values, 
                     alpha=0.2, 
                     color='grey')  
             
            axs[i,1].plot(congestus_dataset.virga_bins.values, 
                          congestus_dataset.CB_mean.values, 
                          color='black', 
                          linestyle='--')

            axs[i,1].fill_between(congestus_dataset.virga_bins.values, 
                     congestus_dataset.CB_mean.values + congestus_dataset.CB_std.values, 
                     congestus_dataset.CB_mean.values - congestus_dataset.CB_std.values, 
                     alpha=0.2, 
                     color='grey')  
    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure07_env_conditions_virga.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()
    
    
    
def visualize_counts(shallow_dataset, congestus_dataset):
    
    fig, axs = plt.subplots(3,2, figsize=(25,20), sharey=True)
    
    mesh_T_s = axs[0,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.T_count.values.T, 
                                   cmap=CMAP_an)
    cbar_T = fig.colorbar(mesh_T_s, 
                          ax=axs[0,0], 
                          orientation='vertical', 
                          label='Anomaly \n Air Temperature [K]')
    
    mesh_T_d = axs[0,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.T_count.values.T, 
                                   cmap=CMAP_an)
    
    cbar_T = fig.colorbar(mesh_T_d, 
                          ax=axs[0,1], 
                          orientation='vertical', 
                          label='Anomaly \n Air Temperature [K]')
    
    mesh_MR_s = axs[1,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.MR_count.values.T, 
                                   cmap=CMAP_an)
    
    cbar_T = fig.colorbar(mesh_MR_s, 
                          ax=axs[1,0], 
                          orientation='vertical', 
                          label='Anomaly \n Mixing Ratio [g kg$^{-1}$]')
        
    mesh_MR_d = axs[1,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.MR_count.values.T, 
                                   cmap=CMAP_an)
    
    cbar_T = fig.colorbar(mesh_MR_d, 
                          ax=axs[1,1], 
                          orientation='vertical', 
                          label='Anomaly \n Mixing Ratio [g kg$^{-1}$]')


    mesh_VW_s = axs[2,0].pcolormesh(shallow_dataset.virga_bins.values, 
                                   shallow_dataset.range.values, 
                                   shallow_dataset.VW_count.values.T, 
                                   cmap=CMAP_an)
    
    cbar_VW = fig.colorbar(mesh_VW_s, 
                          ax=axs[2,0], 
                          orientation='vertical', 
                          label='Anomaly \n Vertical wind [m s$^{-1}$]')   
    
    mesh_VW_d = axs[2,1].pcolormesh(congestus_dataset.virga_bins.values, 
                                   congestus_dataset.range.values, 
                                   congestus_dataset.VW_count.values.T, 
                                   cmap=CMAP_an)
    
    cbar_VW = fig.colorbar(mesh_VW_d, 
                          ax=axs[2,1], 
                          orientation='vertical', 
                          label='Anomaly \n Vertical wind [m s$^{-1}$]')


    for ax, l in zip(axs[:,:].flatten(), ['a) Shallow clouds',  'b) Congestus clouds ']):
        ax.text(-0.05, 1.07, l,  fontweight='black', fontsize=25, transform=ax.transAxes)
        
    for i in range(0,3):
        for j in range(0,2):
            axs[i,j].spines["top"].set_visible(False)
            axs[i,j].spines["right"].set_visible(False)
            axs[i,j].spines["bottom"].set_linewidth(3)
            axs[i,j].spines["left"].set_linewidth(3)
            axs[i,j].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
            axs[i,j].tick_params(which='minor', length=5, width=2, labelsize = 5)
            axs[i,j].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
            axs[i,j].tick_params(axis='both', labelsize=20)
            axs[i,j].get_xaxis().tick_bottom()
            axs[i,j].get_yaxis().tick_left()
            axs[i,j].set_ylim(250.,1500.)
            axs[i,0].set_xlim(0., 250.)
            axs[i,1].set_xlim(0., 750.)
            axs[2,j].set_xlabel('Virga depth [m]', fontsize=20)
            axs[i,0].set_ylabel('Height [m]', fontsize=20)
            axs[i,0].plot(shallow_dataset.virga_bins.values, 
                          shallow_dataset.CB_mean.values, 
                          color='black', 
                          linestyle='--')

            axs[i,0].fill_between(shallow_dataset.virga_bins.values, 
                     shallow_dataset.CB_mean.values + shallow_dataset.CB_std.values, 
                     shallow_dataset.CB_mean.values - shallow_dataset.CB_std.values, 
                     alpha=0.2, 
                     color='grey')  
             
            axs[i,1].plot(congestus_dataset.virga_bins.values, 
                          congestus_dataset.CB_mean.values, 
                          color='black', 
                          linestyle='--')

            axs[i,1].fill_between(congestus_dataset.virga_bins.values, 
                     congestus_dataset.CB_mean.values + congestus_dataset.CB_std.values, 
                     congestus_dataset.CB_mean.values - congestus_dataset.CB_std.values, 
                     alpha=0.2, 
                     color='grey')  
    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure07_counts.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

            
if __name__ == "__main__":
    main()
    