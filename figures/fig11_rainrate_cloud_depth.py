'''Code to generate Figure 11: Rain rate as a function of cloud depth.
we take rain rates at the surface and cloud geometrical thicknesses 
from the cloud radar data analysis. We also differentiate 
between shallow and congestus clouds. 
'''
from readers.radar import get_radar_files
from readers.cloudtypes import read_cloud_thickness_top_base, read_cloud_class
from readers.lidars import f_read_merian_data, read_anomalies
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pdb
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW


def main():
    
    # read cloud thickness and cloud classification
    ds_cp = read_cloud_thickness_top_base()
    ds_cc = read_cloud_class()
    
    # read wband radar data
    ds_rr = read_radar_multiple()
    
    # align radar data with cloud data
    ds_cp, ds_cc, ds_rr = xr.align(ds_cp, ds_cc, ds_rr, join='inner')

    ds = xr.merge([ds_cp, ds_cc, ds_rr])
    print(ds)
    
    # selecting based on the cloud classification
    
    # selecting cloud types and rain/norain conditions
    is_shallow = ds.shape == 0
    is_congestus = ds.shape == 1

    # selecting prec and non prec
    is_prec_ground = ds.flag_rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 

    # segregating with respect to shallow and deep clouds
    ds_sl_prec = ds.isel(time=is_sl_prec)
    ds_sl_nonprec = ds.isel(time=is_sl_non_prec)
    ds_cg_prec = ds.isel(time=is_cg_prec)
    ds_cg_nonprec = ds.isel(time=is_cg_non_prec)
    
    # plot scatter plot of cloud thickness vs rain rate
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_scatter(ds_sl_prec, ax, "shallow precipitation", COLOR_SHALLOW)
    plot_scatter(ds_cg_prec, ax, "congestus precipitation", COLOR_CONGESTUS)

    ax.set_xlim(1, 20)
    ax.set_ylim(0, 3000)
    ax.set_ylabel('Cloud Geometrical Thickness (m)')
    ax.set_xlabel('Rain Rate (mm/hr)')
    ax.legend(frameon=False)
    fig.savefig('/net/ostro/plots_rain_paper/fig11_rainrate_cloud_depth.png')
    
def plot_scatter(ds, ax, label, color):
    
    # select only realistic values of cloud optical thickness
    ds = ds.where((ds.cloud_geometrical_thickness > 0) * 
             (ds.cloud_geometrical_thickness < 3000) , drop=True)
    
    # select only rain rates > 0
    ds = ds.where(ds.rain_rate > 0.1, drop=True)
    
    x = ds.rain_rate
    y = ds.cloud_geometrical_thickness
    # calculate linear fit and plot
    #m, b = np.polyfit(x, y, 1)
    # calculate quadratic fit and plot
    m, b, c = np.polyfit(x, y, 2)
    xx = np.linspace(0, 100, 100)
    ax.plot(xx, m*xx**2 + b*xx + c, color=color)
    
    
    #ax.plot(x, m*x + b, color=color, linestyle='--')
    
    ax.scatter(x, y, color=color, label=label)
    return(ax)


def read_radar_multiple():
    """
    Read all radar files at once. This uses the preprocessor to interpolate the
    radar data onto a common height grid.

    Returns
    -------
    ds: xarray dataset with radar reflectivity
    """

    files = get_radar_files()

    ds = xr.open_mfdataset(files, preprocess=preprocess)

    ds = ds.reset_coords(drop=True)

    return ds


def preprocess(ds):
    """
    Preprocessor to open all radar files at once with dask
    """

    ds = ds["rain_rate"]

    return ds


if __name__ == "__main__":
    main()
    