"""
Joint histogram of cloud base and cloud top height.
"""

import os
import sys
from matplotlib.lines import Line2D

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
#from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW  

# add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

from fig03_diurnal_all import prepare_data

from readers.cloudtypes import read_cloud_class, read_rain_flags
from readers.radar import read_lwp
#load_dotenv()#


def main():
    """
    Create diurnal cycle of hydrometeor fraction
    """

    ds = prepare_data()
    
    t_test_cb_ct(ds)

    da_hist = statistics(ds)

    

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
    #plot_histogram(da_hist=da_hist)
    plot_figure2(da_hist, merian, shallow, congestus, cloudyflag)
    
    # calculate mean LWP shallow and mean LWP congestus
    mean_lwp_shallow = np.nanmean(merian.lwp[shallow & cloudyflag])
    mean_lwp_congestus = np.nanmean(merian.lwp[congestus & cloudyflag])
    print(f"Mean LWP shallow: {mean_lwp_shallow:.2f} g/m^2")
    print(f"Mean LWP congestus: {mean_lwp_congestus:.2f} g/m^2")
    print(f"Number of shallow clouds: {np.sum(shallow & cloudyflag)}")
    print(f"Number of congestus clouds: {np.sum(congestus & cloudyflag)}")
    
   
def plot_figure2(da_hist, merian, shallow, congestus, cloudyflag):
    
    # deriving cumulative histogram of cloudy, non-precipitating LWP as function of shallow/congestus
    binss = np.arange(-50,1000,1)
    bins=np.array([np.mean([binss[i],binss[i+1]]) for i in range(len(binss)-1)])

    # selecting indeces of shallow/congestus 
    indexinds = (cloudyflag == 1) & (~np.isnan(merian.lwp))
    
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(merian.lwp[shallow & indexinds], merian.lwp[congestus & indexinds], equal_var=False)
    print(p_value)
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(f"The difference in rain rates is statistically significant (p-value: {p_value:.20f})")
    else:
        print(f"The difference in rain rates is not statistically significant (p-value: {p_value:.8f})")
        
    
    # constructing figure 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_title("a) Joint histogram of cloud base\n and cloud top height", loc='left', fontweight='black')
    
    im = ax[0].pcolormesh(
        da_hist.radar_cloud_base * 1e-3,
        da_hist.radar_cloud_top * 1e-3,
        da_hist,
        cmap=CMAP,
        norm=mcolors.LogNorm(vmin=1, vmax=30000),
        shading="nearest",
    )
    
    fig.colorbar(im, ax=ax[0], label="Count")

    # set equal aspect ratio
    ax[0].set_aspect("equal")

    ax[0].set_xlim(-1, 4)
    ax[0].set_ylim(-1, 4)

    ax[0].set_yticks(np.arange(-1, 5, 1))
    ax[0].set_yticklabels([-1, "LCL", 1, 2, 3, 4])

    ax[0].set_xticks(np.arange(-1, 5, 1))
    ax[0].set_xticklabels([-1, "LCL", 1, 2, 3, 4])

    # indicate dotted line for LCL height
    ax[0].axhline(0, color="k", linestyle="--", linewidth=1)
    ax[0].axvline(0, color="k", linestyle="--", linewidth=1)
    ax[0].axhline(0.6, color="k", linestyle=":", linewidth=1)

    ax[0].annotate(
        "LCL + 600 m",
        xy=(4, 0.6),
        xycoords="data",
        ha="right",
        va="bottom",
    )

    ax[0].plot([-1, 5], [-1, 5], color="k", linewidth=1)

    ax[0].set_xlabel("Cloud base height above LCL [km]")
    ax[0].set_ylabel("Cloud top height above LCL [km]")
    
    # second plot: cumulative distributions LWP
    ax[1].set_title("b) Cumulative distributions of LWP", loc='left', fontweight='black')

    hist=ax[1].hist(merian.lwp[shallow & indexinds],
                     bins=binss, 
                     histtype='step', 
                     color='#ff9500',
                     label='shallow',
                     density=True,
                     cumulative=True, 
                     linewidth=2)
    hist=ax[1].hist(merian.lwp[congestus & indexinds],
                     bins=binss, 
                     histtype='step', 
                     color='#008080',
                     label='deep',
                     density=True,
                     cumulative=True, 
                     linewidth=2)
    # construct labels for the legend
    handles = [Line2D([0], [0], 
               color=COLOR_CONGESTUS, 
               linewidth=4, 
               label='Congestus clouds'),  
                Line2D([0], [0], 
               color=COLOR_SHALLOW, 
               linewidth=4, 
               label='Shallow clouds')]
    ax[1].legend(handles=handles, frameon=False,loc=4)         
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax[1].set_xlabel('LWP [gm$^{-2}$]')
    ax[1].set_xlim(0,550)
    ax[1].set_ylim(0.3,1)
    #ax[1].set_ylabel('Cumulated density / g$^{-1}$m$^{2}$')
    ax[1].set_ylabel('Cumulated density')

    plt.savefig(
        os.path.join("/net/ostro/plots_rain_paper/", "figure02.png")
    )

def statistics(ds):
    """
    Calculate statistics
    """

    # filter only shallow and congestus clouds
    ds = ds.sel(time=((ds.shape == 1) | (ds.shape == 0)))

    # cloud base and cloud top height
    ds["cloud_height"] = xr.where(ds.cloud_mask == 0, np.nan, ds.height)
    ds["cloud_base"] = ds["cloud_height"].min("height", skipna=True)
    ds["cloud_top"] = ds["cloud_height"].max("height", skipna=True)

    # load cloud base and cloud top height
    ds["cloud_base"] = ds["cloud_base"].load()
    ds["cloud_top"] = ds["cloud_top"].load()

    # resample to 200 m bins
    dz = 200
    bins_wrt_lcl = np.arange(-1000, 5200 + dz, dz)
    bins_wrt_lcl_cen = (bins_wrt_lcl[1:] + bins_wrt_lcl[:-1]) / 2

    hist, _, _ = np.histogram2d(
        x=ds["cloud_top"].values,
        y=ds["cloud_base"].values,
        bins=[bins_wrt_lcl, bins_wrt_lcl],
    )

    # create data array
    da_hist = xr.DataArray(
        hist,
        dims=["radar_cloud_top", "radar_cloud_base"],
        coords={
            "radar_cloud_top": bins_wrt_lcl_cen,
            "radar_cloud_base": bins_wrt_lcl_cen,
        },
    )

    return da_hist


def plot_histogram(da_hist):
    """
    Plot histogram
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    im = ax.pcolormesh(
        da_hist.radar_cloud_base * 1e-3,
        da_hist.radar_cloud_top * 1e-3,
        da_hist,
        cmap=CMAP,
        norm=mcolors.LogNorm(vmin=1, vmax=30000),
        shading="nearest",
    )
    fig.colorbar(im, ax=ax, label="Count")

    # set equal aspect ratio
    ax.set_aspect("equal")

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    ax.set_yticks(np.arange(-1, 5, 1))
    ax.set_yticklabels([-1, "LCL", 1, 2, 3, 4])

    ax.set_xticks(np.arange(-1, 5, 1))
    ax.set_xticklabels([-1, "LCL", 1, 2, 3, 4])

    # indicate dotted line for LCL height
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.6, color="k", linestyle=":", linewidth=1)

    ax.annotate(
        "LCL + 600 m",
        xy=(4, 0.6),
        xycoords="data",
        ha="right",
        va="bottom",
    )

    ax.plot([-1, 5], [-1, 5], color="k", linewidth=1)

    ax.set_xlabel("Cloud base height above LCL [km]")
    ax.set_ylabel("Cloud top height above LCL [km]")

    plt.savefig(
        os.path.join(os.getenv("PATH_PLOT"), "basetop.png"),
    )


if __name__ == "__main__":
    main()
