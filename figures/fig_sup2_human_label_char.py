"""
Code obtained from merging Claudia and Sabrina's codes for plotting cloud properties.
it will contribute to the creation of figure 1 of the publication, by producing
- LWP time series of the 4 mesoscale patterns 
- median radar reflectivity profiles for the 4 mesoscale patterns

"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import glob
import netCDF4 as nc
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from matplotlib import rcParams
import matplotlib
from readers.cloudtypes import read_merian_classification, read_rain_flags
from readers.radar import read_radar_multiple, read_lwp

def main():

    # defining domain 
    minlon = 6.; maxlon = 15.; minlat = -61; maxlat = -50;
    extent_param = [minlon, maxlon, minlat, maxlat]
    
    
    # read mesoscale pattern classification
    orga = read_merian_classification()
    
    # read radar data
    merianwband = read_radar_multiple()
    
    # read MWR data
    merian = read_lwp()
    
    # read rain flags
    rainflags = read_rain_flags()
    
    #derive flags:
    combiprecipflag = (rainflags.flag_rain_ground.values == 1) | (rainflags.flag_rain.values == 1) #either ground_rain or rain
    cloudyflag = ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain
    clearflag = (combiprecipflag == 0) & (cloudyflag == 0) #everything thats not cloudy, not rainy
    
    #apply Clau's rain filters: conservative, ie flag both rain_ground and rain with nan in lwp dataset
    merian.lwp.values[combiprecipflag] = np.nan
    
    #interpolate organization index to merian measurements:
    orgaindex = orga.morph_ident[:,:].interp(time=merian.time.values)
    
    #due to interpolation, values between 0 and 1 (floats) can occur. flag those values at end and start of each label AS PART of the respective label:
    print('N = %i (ie %.2f ) of interpolated edge indices'%(np.sum(orgaindex.values != 0), np.sum(orgaindex.values != 0)/float(len(orgaindex.values))))
    orgaindex.values[orgaindex.values !=0] = 1.
    orgaindex = np.array(orgaindex)
    ntime = float(len(orgaindex))
    humanlabels = ['Sugar','Gravel','Flower','Fish']
    
    #calculate relative occurrences of labels in total merian measurements:
    for i in range(4):
        print('temporal fraction of Merian measurements in %s: %.2f percent (N=%i)'%(humanlabels[i], np.sum(orgaindex[:,i])/ntime*100, np.sum(orgaindex[:,i])))
    
    #calculate how often precipitation is observed per human label:
    for i in range(4):
        indexinds = orgaindex[:,i] == 1
        print('precipitation fraction in Merian measurements in %s: %.2f percent'%(humanlabels[i], np.sum(combiprecipflag[indexinds])/np.sum(indexinds)*100   ))
    
    
    #calculate how often clear-sky or cloudy is observed per human label: (clearsky is defined as not cloudy, not precipitating.
    for i in range(4):
        indexinds = orgaindex[:,i] == 1
        print('%s fraction of clearsky = %.2f and of cloudy = %.2f'%(humanlabels[i], np.sum(clearflag[indexinds])/np.sum(indexinds)*100 , np.sum(cloudyflag[indexinds])/np.sum(indexinds)*100 ))    
    
    # plot figure with LWP patterns
    plot_LWP_time_map_patterns(merian, orga, humanlabels, extent_param, orgaindex)
    
    
    # plot cloud fraction profiles for the 4 mesoscale patterns
    plot_Ze_profile_patterns(merianwband, orgaindex, humanlabels)
    
    print('sono qui')

    

def get_distribution(data, bins):
    '''
    INPUT:
    - data: array with values
    - bins: bins to determine distribution for
    OUTPUT:
    - hist: absolute counts per bin
    - histnorm: relative occurrence per bin normed to total number of valid data points in data, normed to binsize!
    
    edits:
    - 221103: included normalization to binsize.
    - 240129: function imported from merian_organization.ipynb to merian_organization_SSCA.py for harmonising codes for the publication
    
    author: Sabrina Schitt

    '''
    binsize = bins[1] - bins[0]
    hist, bins = np.histogram(data, bins=bins)
    histnorm = hist/float(np.sum(hist))/float(binsize)

    return hist, histnorm 


def f_extract_patterns_datasets(context_data, merian_dataset):
    '''function to extract the data corresponding to the cloud mesoscale patterns
     author: Sabrina Schitt
     
    '''
    # interpolate context_data on merian_dataset time resolution
    context_data_interp = context_data.interp(time=merian_dataset.time.values)

    # selecting indeces for the different cloud types
    # definition of cloud types
    sugar = context_data_interp.morph_ident[:,0].values
    gravel = context_data_interp.morph_ident[:,1].values
    flower = context_data_interp.morph_ident[:,2].values
    fish = context_data_interp.morph_ident[:,3].values

    # selecting indeces on the time array where sugar, gravel, flower, fish are occurring
    ind_sugar = (np.where(sugar == 1.))[0]
    ind_gravel = (np.where(gravel == 1.))[0]
    ind_flower = (np.where(flower == 1.))[0]
    ind_fish = (np.where(fish == 1.))[0]

    # derive mean mrr properties for each class
    merian_sugar = merian_dataset.isel(time=ind_sugar)
    merian_gravel = merian_dataset.isel(time=ind_gravel)
    merian_flower = merian_dataset.isel(time=ind_flower)
    merian_fish = merian_dataset.isel(time=ind_fish)

    return(merian_sugar, merian_gravel, merian_flower, merian_fish)   
    




def visualize_trajectory_in_subplots(orga, extent_param):
    """
    function to plot the trajectory from the data with the colored dots 
    
    """

    # plot settings
    # colors
    col_ocean = '#CAE9FB'
    col_land = 'grey'
    col_coastlines = 'darkgrey'
    cmap_track = 'plasma'
    
    # fontsizes
    fs_grid = 10
    fs_cbar_labels = 10
    fs_track_labels = 10
    
    # zorders
    zorder_land = 0
    zorder_coastlines = 1
    zorder_gridlines = 2
    zorder_day_marker = 4
    zorder_track = 3
    zorder_day_annotation = 5
    
    # size of the patterns
    sval = 20
    
    # plot minutely ship position
    plt.subplots(
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    # set map extent
    plt.set_extent(extents=(-61, -50, 6, 15), crs=ccrs.PlateCarree())
    # add land feature
    land_10m = cfeature.NaturalEarthFeature(category='physical', name='land', 
                                            scale='10m',
                                        edgecolor=col_coastlines,
                                        linewidth=0.5,
                                        facecolor=col_land)
    ax.add_feature(land_10m, zorder=zorder_land)
    
    # add lat lon grid
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='silver', 
                      draw_labels=True, zorder=zorder_gridlines,
                      x_inline=False, y_inline=False)
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-70, -40, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 16, 1))
    gl.xlabel_style = {'size': fs_grid, 'color': 'k'}
    gl.ylabel_style = {'size': fs_grid, 'color': 'k'}
    
        
    rect1 = matplotlib.patches.Rectangle((-61, 12.5), 
                                        11, 2.5, 
                                        color ='royalblue', 
                                        alpha=0.2) 
    #    minlon = 6.; maxlon = 15.; minlat = -61; maxlat = -50;

    rect2 = matplotlib.patches.Rectangle((-61, 10.5), 
                                        11, 2., 
                                        color ='lightsteelblue', 
                                        alpha=0.3) 
    
    rect3 = matplotlib.patches.Rectangle((-61, 6), 
                                        11, 4.5, 
                                        color ='lavender', 
                                        alpha=0.3) 
    
    ax.add_patch(rect1) 
    ax.add_patch(rect2) 
    ax.add_patch(rect3) 
    
    
    # plot ship track
    ax.plot(class_cloud_data.longitude.values, 
            class_cloud_data.latitude.values, 
            color='black', 
            transform=ccrs.PlateCarree(),
            zorder=zorder_track, 
            alpha=0.3,
            label="RV MSM trajectory"
            )
               
    ax.hlines(y=10.5, xmin=-61.5, xmax=-50, color='black', linestyles='--')
    ax.hlines(y=12.5, xmin=-61.5, xmax=-50, color='black', linestyles='--')
    
    
    # plot sugar matches
    sugar = np.where(class_cloud_data.morph_ident.values[:,0] == 1)
    ax.scatter(class_cloud_data.longitude.values[sugar], 
            class_cloud_data.latitude.values[sugar], 
            color='#E63946', 
            label='sugar', 
            marker='o', 
            s=sval)
    
    gravel = np.where(class_cloud_data.morph_ident.values[:,1] == 1)
    ax.scatter(class_cloud_data.longitude.values[gravel], 
            class_cloud_data.latitude.values[gravel], 
            color='#EC8A91', 
            label='gravel', 
            marker='o', 
            s=sval) 
    
    flower = np.where(class_cloud_data.morph_ident.values[:,2] == 1)
    ax.scatter(class_cloud_data.longitude.values[flower], 
            class_cloud_data.latitude.values[flower], 
            color='#457B9D', 
            label='flower', 
            marker='o', 
            s=sval) 
    
    fish = np.where(class_cloud_data.morph_ident.values[:,3] == 1)
    ax.scatter(class_cloud_data.longitude.values[fish], 
            class_cloud_data.latitude.values[fish], 
            color='#A8DADC', 
            label='fish', 
            marker='o', 
            s=sval) 
    
    ax.legend(frameon=True)   
    
        
def plot_Ze_profile_patterns(merianwband, orgaindex, humanlabels):
    
    colors = ['#E63946','#EC8A91', '#A8DADC','#457B9D','black']
    color_names = ['sugar', 'gravel', 'flower', 'fish','']
    
    fig, axs = plt.subplots(figsize=(8,8))
    plt.gcf().subplots_adjust(bottom=0.1)
    for i in range(4):
        axs.plot(np.nanmedian(merianwband.radar_reflectivity[(orgaindex[:,i] == True ), :],axis=0), 
                 merianwband.height,label=humanlabels[i],color=colors[i], linewidth=4)
        print(i,np.sum(orgaindex[:,i]))
        #ax[0].plot(np.nanmean(merianwband.radar_reflectivity[(shallow==True) &(orgaindex[:,i] == True ), :],axis=0), merianwband.height,label=humanlabels[i],color=colors[i])
        
        #ax[1].plot(np.nanmean(merianwband.radar_reflectivity[(deep==True) &(orgaindex[:,i] == True ), :],axis=0), merianwband.height,color=colors[i])
    axs.set_ylim(0,12500)
    axs.set_title('b) Median radar reflectivity profiles for sugar, \n gravel, fish and flower', fontsize=20, fontweight='black')
    axs.legend(frameon=False,loc=1, fontsize=18)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    #for i in range(2):
    #    ax[i].spines['top'].set_visible(False)
    axs.set_ylabel('Height / m', fontsize=20)
    axs.set_xlabel('Ze / dBZe', fontsize=20)
    axs.spines["bottom"].set_linewidth(3)
    axs.spines["left"].set_linewidth(3)
    axs.tick_params(which='major', length=7, width=3, labelsize=20)
   
    # saving figure as png
    plt.savefig(
        "/work/plots_rain_paper/merian_organization_Ze_mean.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    
    plt.close()
    
def plot_LWP_time_map_patterns(merian, orga, humanlabels, extent_param, orgaindex):
    
    colors = ['#E63946','#EC8A91', '#A8DADC','#457B9D','black']
    color_names = ['sugar', 'gravel', 'flower', 'fish','']
    
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(9,8))
    plt.gcf().subplots_adjust(bottom=0.1)
   

    for i in range(4):
        axs[i].plot(merian.time[orgaindex[:,i] == 1], merian.lwp.values[orgaindex[:,i] == 1],'.k',markersize=8, color=colors[i])
        axs[i].set_ylabel('LWP / \n gm$^{-2}$', fontsize=20)
        axs[i].text(0.8,0.8,humanlabels[i],transform=axs[i].transAxes, fontsize=18)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_linewidth(3)
        axs[i].spines["left"].set_linewidth(3)
        axs[i].tick_params(which='major', length=7, width=3, labelsize=18)
        axs[i].get_xaxis().tick_bottom()
        axs[i].get_yaxis().tick_left()
        
    
    import matplotlib.dates as mdates
    axs[0].set_ylim(-50,200)
    axs[0].set_title('a) LWP time series for sugar, \n gravel, fish and flower', fontsize=20, fontweight='black')
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axs[3].set_xlabel('Time', fontsize=20)
    # saving figure as png
    plt.savefig(
        "/work/plots_rain_paper/LWP_time_series_patterns.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()
    
if __name__ == "__main__":
    main()
