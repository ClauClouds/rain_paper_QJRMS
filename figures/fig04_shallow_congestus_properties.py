import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from figures.mpl_style import CMAP
import os

from readers.cloudtypes import read_in_clouds_radar_moments, read_cloud_class, read_rain_flags
from readers.radar import read_lwp



def main():
    """
    plot skewness mean Vd reflectivity scatters for shallow/congestus and LWP cumulative distributions from sabrina
    
    """
    # read radar moments above cloud base
    data = read_in_clouds_radar_moments()
    
    # read MWR data
    merian = read_lwp() 

    # read rain flags
    rainflags = read_rain_flags()
    
    #derive flags:
    combiprecipflag = (rainflags.flag_rain_ground.values == 1) | (rainflags.flag_rain.values == 1) #either ground_rain or rain
    cloudyflag = ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain

    # read cloud classification
    cloudclassdata = read_cloud_class()
    
    # interpolate classification on LWP time stamps and derive flags for shallow/congestus data
    cloudclassdataip = cloudclassdata.interp(time=merian.time)
    shallow = (cloudclassdataip.shape.values == 0) & (cloudyflag == 1)
    congestus = (cloudclassdataip.shape.values == 1) &(cloudyflag == 1)


    # plot figure 
    visualize_incloud_lwp(data, merian, cloudyflag, shallow, congestus)   
    

def visualize_incloud_lwp(data, merian, cloudyflag, shallow, congestus):   
    """
    function to visualize sk vs ze, vd vs ze and lwp cumulative distributions for shallow and deep clouds

    Args:
        data (xarray dataset): dataset containing radar moments
        merian (xarray dataset): dataset containing lwp data
        cloudyflag (ndarray): flag indicating everything thats cloudy in ground_rain and rain
    """

    # selecting radar moments for the plot for shallow and congestus clouds
    ze_s = data.ze_shallow.values.flatten()
    ze_c = data.ze_congestus.values.flatten()
    vd_s = data.vd_shallow.values.flatten()
    vd_c = data.vd_congestus.values.flatten()
    sk_s = data.sk_shallow.values.flatten()
    sk_c = data.sk_congestus.values.flatten()
    
    # selecting values different from nans in all variables for shallow and congestus clouds
    i_good_s = ~np.isnan(ze_s) * ~np.isnan(vd_s) * ~np.isnan(sk_s)
    i_good_c = ~np.isnan(ze_c) * ~np.isnan(vd_c) * ~np.isnan(sk_c)
    ze_s = ze_s[i_good_s]
    vd_s = vd_s[i_good_s]
    sk_s = sk_s[i_good_s]
    ze_c = ze_c[i_good_c]
    vd_c = vd_c[i_good_c]
    sk_c = sk_c[i_good_c]
    
    # deriving cumulative histogram of cloudy, non-precipitating LWP as function of shallow/congestus
    binss = np.arange(-50,1000,1)
    bins=np.array([np.mean([binss[i],binss[i+1]]) for i in range(len(binss)-1)])

    # selecting indeces of shallow/congestus 
    indexinds = (cloudyflag == 1) & (~np.isnan(merian.lwp))
    
    # calculating 2d histograms to plot
    hist_ZESK_cloud_s, x_ze_cloud_s, y_sk_cloud_s = np.histogram2d(ze_s, sk_s, bins=[40, 40], range=[[-50., 25.],[-1., 1.]], density=True)
    hist_ZEVD_cloud_s, x_ze2_cloud_s, y_vd_cloud_s = np.histogram2d(ze_s, vd_s, bins=[40, 40], range=[[-50., 25.],[-4., 2.]], density=True)

    # plotting data
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12,25))

    # first plot: cumulative distributions LWP
    hist=axs[0].hist(merian.lwp[shallow & indexinds],
                     bins=binss, 
                     histtype='step', 
                     color='#ff9500',
                     label='shallow',
                     density=True,
                     cumulative=True, 
                     linewidth=5)
    hist=axs[0].hist(merian.lwp[congestus & indexinds],
                     bins=binss, 
                     histtype='step', 
                     color='#008080',
                     label='deep',
                     density=True,
                     cumulative=True, 
                     linewidth=5)
    axs[0].legend(frameon=False,loc=4, fontsize=20)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    axs[0].set_xlabel('LWP / gm$^{-2}$', fontsize=25)
    axs[0].set_xlim(0,550)
    axs[0].set_ylim(0.3,1)
    axs[0].set_ylabel('Cumulated density / g$^{-1}$m$^{2}$', fontsize=25)
    
    
    # second plot 
    hist_d = axs[1].hist2d(ze_c, -sk_c, bins=(40, 40), range=([-50., 25.],[-1., 1.]), cmap=CMAP, density=True, cmin=0.0001)
    cbar = fig.colorbar(hist_d[3], ax=axs[1])
    cs = axs[1].contour(x_ze_cloud_s[:-1], 
                        -y_sk_cloud_s[:-1], 
                        hist_ZESK_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZESK_cloud_s), np.nanmax(hist_ZESK_cloud_s),
                                (np.nanmax(hist_ZESK_cloud_s)- np.nanmin(hist_ZESK_cloud_s))/10),
                        cmap=plt.cm.Greys)
    
    axs[1].clabel(cs, inline=True, fontsize=14)
    cbar.set_label('norm. occ. congestus clouds', fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    axs[1].set_ylabel("Skewness ", fontsize=25)
    axs[1].set_xlabel("Reflectivity [dBz] ", fontsize=25)
    axs[1].set_ylim(-1., 1.)
    axs[1].set_xlim(-50.,25.)
    
    # third plot 
    hist_d = axs[2].hist2d(ze_c, vd_c, bins=(40, 40), range=([-50., 25.],[-4., 2.]), cmap=CMAP, density=True, cmin=0.0001)
    cbar = fig.colorbar(hist_d[3], ax=axs[2])
    cs = axs[2].contour(x_ze2_cloud_s[:-1], 
                        y_vd_cloud_s[:-1], 
                        hist_ZEVD_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZEVD_cloud_s),
                                np.nanmax(hist_ZEVD_cloud_s), 
                                (np.nanmax(hist_ZEVD_cloud_s)- np.nanmin(hist_ZEVD_cloud_s))/10), cmap=plt.cm.Greys, linewidth=8)
    axs[2].clabel(cs, inline=True, fontsize=14)
    cbar.set_label('norm. occ. congestus clouds', fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    axs[2].set_ylabel("Mean Doppler velocity [ms$^{-1}$] ", fontsize=25)
    axs[2].set_xlabel("Reflectivity [dBz] ", fontsize=25)
    axs[2].set_ylim(-4., 2.)
    axs[2].set_xlim(-50.,25.)

    for ax in axs.flatten():    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(3)
        ax.spines["left"].set_linewidth(3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.tick_params(which='minor', length=5, width=3, labelsize=25)
        ax.tick_params(which='major', length=7, width=3, labelsize=25)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    

    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "fig_lwp_radar_moments_in_cloud.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


    
if __name__ == "__main__":
    main()
    