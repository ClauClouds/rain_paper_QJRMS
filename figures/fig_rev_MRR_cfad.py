"""
figure to plot the cfad plot of the radar reflectivities for the MRR
"""


import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from readers.radar import read_radar_single_file
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
import os
from readers.mrr import get_mrr_files, read_mrr_dsd
from xhistogram.xarray import histogram
import pdb
from dask.diagnostics import ProgressBar
ProgressBar().register()

# read mrr data
def main():
    
    # set directory where to store plots and gif
    directory_oout = '/work/plots_rain_paper/'   
    
    # read radar reflectivity
    mrr_filelist = get_mrr_files()
    drop_var_list = ['Doppler_spectra_att_corr', 
                     'drop_size_distribution', 
                     'Doppler_spectra', 
                     'rain_rate', 
                        'liquid_water_content',
                        'Ze',
                        'attenuation',
                        'instrument',
                        'lat',
                        'lon',
                        'Doppler_velocity', 
                        'drop_size_distribution_att_corr',
                        'diameters']
    ds_mrr = xr.open_mfdataset(mrr_filelist, drop_variables=drop_var_list)
    
    # defining bins of ze and height
    bins_ze = np.arange(-25, 70, 1)
    binz_height = np.arange(20, 1290, 10)

    cfad, ze_bin, height_bin = plot_cfads2(
        ds_mrr,
        'Zea',
        -25., 
        70.,)   
    
    # plot cfad
    fig, ax = plt.subplots(figsize=(12, 10))

    # colormap
    cmap = 'Blues'
    bounds = np.arange(20, 1290, 10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.grid(linestyle="--", linewidth=0.5, color='lightgrey')   
    
    # plot colormesh of congestus clouds 
    im = ax.pcolormesh(
        ze_bin,
        height_bin,
        cfad,
        vmin=0,
        vmax=np.nanmax(cfad),
        cmap=cmap,
        shading="nearest",
    )
    skip = 50
    # add colorbar
    cbar = fig.colorbar(im, 
                        ax=ax)
    cbar.set_label("Density of Attenuated Radar Reflectivity bins", fontsize=20)
    # set font size of the colorbar label
    
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel("Attenuated Radar Reflectivity (dBZ)", fontsize=20)
    ax.set_ylabel("Height (m)", fontsize=20)
    ax.set_title("CFAD of Attenuated Radar Reflectivity - MRR", fontsize=20)
    
    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure_rev_MRR_zea_cfad.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


def plot_cfads2(data_all, var_name, thr_min, thr_max):
    '''
    data = xarray dataset 
    var_name = string containing the variable name to be processed
    thr_min = min threshold for the variable values
    thr_max = max threshold for the variable values
    
    '''
    
    # creating a xvar and yvar
    yvar_matrix = (np.ones((len(data_all.time.values),1))*np.array([data_all.height.values]))
    xvar_matrix = data_all[var_name].values
    
    # flattening the series 
    yvar = yvar_matrix.flatten()
    xvar = xvar_matrix.flatten()
    N_tot = len(xvar)
    
    # defining bins for the 2d histogram
    y_edges = []
    for ind in range(len(data_all.height.values)-1):  
        h_val = data_all.height.values[ind]
        y_edges.append(h_val + float((data_all.height.values)[ind+1]-h_val)/2)
    y_final = y_edges[0:-1:3]
    x_edges = np.arange(start=thr_min, stop=thr_max , step=0.5)
    bins = [x_edges, y_final]
    
    # plot 2d histogram figure 
    i_good = ((~np.isnan(xvar)) * (~np.isnan(yvar)))
    hst, xedge, yedge = np.histogram2d(xvar[i_good], yvar[i_good], bins=bins)
    hst = hst.T
    
    # normalizing histogram and getting bins centers
    #hst = 100.*hst/np.nansum(hst,axis=1)[:,np.newaxis]
    hst = hst
    xcenter = (xedge[:-1] + xedge[1:])*0.5
    ycenter = (yedge[:-1] + yedge[1:])*0.5
    
    return(hst, xcenter, ycenter)


if __name__ == "__main__":
    main()
