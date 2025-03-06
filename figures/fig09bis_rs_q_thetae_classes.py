'''
code to read radiosondes, extract launch date at time, assign a cloud type classification
then:
- count how many we have for each class
- derive q and thet_e for each sounding
- plot q vs thet_e for each class

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
from readers.cloudtypes import read_cloud_class, read_rain_ground, read_cloud_top_base
import pdb
import os
from fig10_sounding_q_sigmae import calc_paluch_quantities
import matplotlib.colors as mcolors
from metpy.calc import equivalent_potential_temperature
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW, cmap_rs_shallow, cmap_rs_congestus

def main():
    
    # if ncdf file exists then skip othewise save ds in ncdf
    if os.path.exists('/net/ostro/ancillary_data_rain_paper/soundings_q_thetae.nc'):
        
        ds = xr.open_dataset('/net/ostro/ancillary_data_rain_paper/soundings_q_thetae.nc')
                
        # Load the data into memory
        ds.load()
        
    else:    
            
        # read radiosounding data   
        ds = read_merian_soundings()
    
        # calculate q and theta_e for each sounding
        ds = calc_q_theta_e(ds)
        
        # assign cloud type
        ds = assign_cloud_type(ds)
        
        # store data in ncdf
        ds.to_netcdf('/net/ostro/plots_rain_paper/soundings_q_thetae.nc')

        # Load the data into memory
        ds.load()
        
    # group rs for each category
    ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, ds_clear = group_soundings_per_class(ds)
    
    print(len(ds_sl_prec.sounding.values))
    print(len(ds_sl_nonprec.sounding.values))
    print(len(ds_cg_prec.sounding.values))
    print(len(ds_cg_nonprec.sounding.values))
    print(len(ds_clear.sounding.values))
        
    #plot_scatter_q_theta(ds_sl_nonprec, ds_cg_nonprec, ds_sl_prec, ds_cg_prec)
    
    # plot mean profiles of q vs theta_e for each class
    ds_mean_sl_nonprec = ds_sl_nonprec.mean(dim='sounding')
    ds_mean_sl_prec = ds_sl_prec.mean(dim='sounding')
    ds_mean_cg_nonprec = ds_cg_nonprec.mean(dim='sounding')
    ds_mean_cg_prec = ds_cg_prec.mean(dim='sounding')
    ds_clear = ds_clear.mean(dim='sounding')
    
    # calculate distributions of cloud tops
    ct_sl_nonprec = ds_sl_nonprec.cloud_top.values
    ct_sl_prec = ds_sl_prec.cloud_top.values
    ct_cg_nonprec = ds_cg_nonprec.cloud_top.values
    ct_cg_prec = ds_cg_prec.cloud_top.values
    
    # merge distributions for shallow and congestus
    ct_sl = np.concatenate((ct_sl_nonprec, ct_sl_prec))
    ct_cg = np.concatenate((ct_cg_nonprec, ct_cg_prec))
    
    # selects alt below 4 km
    ds_mean_sl_nonprec = ds_mean_sl_nonprec.where(ds_mean_sl_nonprec.alt < 4000, drop=True)
    ds_mean_sl_prec = ds_mean_sl_prec.where(ds_mean_sl_prec.alt < 4000, drop=True)
    ds_mean_cg_nonprec = ds_mean_cg_nonprec.where(ds_mean_cg_nonprec.alt < 4000, drop=True)
    ds_mean_cg_prec = ds_mean_cg_prec.where(ds_mean_cg_prec.alt < 4000, drop=True)
    ds_clear = ds_clear.where(ds_clear.alt < 4000, drop=True)

    fig_paper_v2(ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec, ds_clear, ct_sl, ct_cg)
    #fig_paper2(ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec, ds_clear)
    
def fig_paper2(ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec, ds_clear):
    
    # calculate q anomalies for each class
    q_an_cong_prec = ds_mean_cg_prec.q.values - ds_clear.q.values
    q_an_cong_nonprec = ds_mean_cg_nonprec.q.values - ds_clear.q.values
    q_an_sl_nonprec = ds_mean_sl_nonprec.q.values - ds_clear.q.values
    
    cmap_min = -4.
    cmap_max = 4.
    cmap = mpl.cm.get_cmap('seismic')
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    
    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    
    fig = plt.figure(figsize=(10,10))
    
    symbol_size = 150
    
    ax = fig.add_subplot(111)
    
    # plot profile of clear sky theta_e as a reference
    ax.plot(ds_clear.theta_e,
            ds_clear.alt,
            color='black', 
            linestyle='--',
            linewidth=2)
    
    ax.scatter(ds_mean_cg_prec.theta_e, 
               ds_mean_cg_prec.alt, 
               c=q_an_cong_prec, 
               cmap=cmap, 
               marker = 'o',
               s=symbol_size,
               #edgecolors='black',
               label='congestus prec',
               norm=norm)
    ax.scatter(ds_mean_cg_nonprec.theta_e, 
              ds_mean_cg_nonprec.alt, 
              c=q_an_cong_nonprec, 
              cmap=cmap, 
              marker = '^',
              s=symbol_size,
              #edgecolors='black',
              label='congestus non prec',
              norm=norm)
    ax.scatter(ds_mean_sl_nonprec.theta_e, 
              ds_mean_sl_nonprec.alt, 
              c=q_an_sl_nonprec, 
              cmap=cmap, 
              s=symbol_size,
              marker = '^',
              #edgecolors='black',
              label='congestus non prec',
              norm=norm)    
            
    ax.set_xlabel('$\Theta_e$ [K]')
    ax.set_xlim(320., 350.)
    ax.set_ylabel('Altitude (m)')
    
    # add colorbar 
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='seismic'),
                        ax=ax,
                        orientation='vertical',
                        label='q anomaly to clear sky (g/kg)')
    
    ax.legend(frameon=True)
    fig.savefig('/net/ostro/plots_rain_paper/figure9bis_rs_scatter.png') 
    return(fig)

def fig_paper_v2(ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec, ds_clear, ct_sl, ct_cg):
    '''
    plot figure for the paper to justify cold dry air coming dow'''
    from matplotlib import gridspec
    
    # calculate binned histograms of cloud top heights
    bins = np.arange(0., 4000., 200.)
    bin_width = bins[1] - bins[0]

    hist_shallow, bins_shallow = np.histogram(ct_sl, bins=bins, density=True)
    hist_congestus, bins_congestus = np.histogram(ct_cg, bins=bins, density=True)
    
    # Adjust bin coordinates
    bins_shallow_centered = bins_shallow[:-1] + bin_width / 2
    bins_congestus_centered = bins_congestus[:-1] + bin_width / 2
    

    # calculate q anomalies for each class to be plotted as colors
    q_an_cong_prec = ds_mean_cg_prec.q.values - ds_clear.q.values
    q_an_cong_nonprec = ds_mean_cg_nonprec.q.values - ds_clear.q.values
    q_an_sl_nonprec = ds_mean_sl_nonprec.q.values - ds_clear.q.values
    
    # setting color maps edges
    cmap_min = -4.
    cmap_max = 4.
    symbol_size = 150

    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    
    # create a figure with two subplots

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    
    # Customize the subplots as needed
    ax1.set_title('Shallow clouds')
    ax2.set_title('Congestus clouds')

    
    # plot distributions of cloud bases along y axis on first subplot
    ax1_twin = ax1.twiny()
    ax1_twin.plot(hist_shallow, bins_shallow_centered, color='lightgray', linewidth=2, alpha=0.5, label='cloud tops shallow')
    ax1_twin.plot(hist_congestus, bins_congestus_centered, color='dimgrey', linewidth=2, alpha=0.5, label='cloud tops congestus')
    ax1_twin.fill_betweenx(bins_shallow_centered, 0., hist_shallow, color='lightgray', alpha=0.1)
    ax1_twin.fill_betweenx(bins_congestus_centered, 0., hist_congestus, color='dimgrey', alpha=0.1)
    ax1_twin.axis('off')
    
    # reverse x axis
    #ax_twin.invert_xaxis()
    # set x axis limits

    # plot profile of clear sky theta_e as a reference
    ax1.plot(ds_clear.theta_e,
            ds_clear.alt,
            color='black', 
            linestyle='--',
            linewidth=2)
    
    
    # do the same as before but for the second subplot
    ax2_twin = ax2.twiny()
    ax2_twin.plot(hist_shallow, bins_shallow_centered, color='lightgray', linewidth=2, alpha=0.5, label='cloud tops shallow')
    ax2_twin.plot(hist_congestus, bins_congestus_centered, color='dimgrey', linewidth=2, alpha=0.5, label='cloud tops congestus')
    ax2_twin.fill_betweenx(bins_shallow_centered, 0., hist_shallow, color='lightgray', alpha=0.1)
    ax2_twin.fill_betweenx(bins_congestus_centered, 0., hist_congestus, color='dimgrey', alpha=0.1)
    ax2_twin.axis('off')
    
    # reverse x axis
    #ax_twin.invert_xaxis()
    # set x axis limits
    ax1_twin.set_xlim(0., 0.01)
    ax2_twin.set_xlim(0., 0.01)
    ax1_twin.legend(loc='upper right', frameon=True, facecolor='white')

    # plot profile of clear sky theta_e as a reference
    ax2.plot(ds_clear.theta_e,
            ds_clear.alt,
            color='black', 
            linestyle='--',
            linewidth=2, 
            alpha=0.5)
    
    # plot colored line for congestus non prec
    line_cong = plot_colored_line(ax2, 
                      ds_mean_cg_nonprec.theta_e, 
                      ds_mean_cg_nonprec.alt, 
                      q_an_cong_nonprec, 
                      cmap_rs_congestus, 
                      'congestus non prec',
                      cmap_min,
                      cmap_max)
    
     # plot colored line for congestus non prec
    line = plot_colored_line(ax2, 
                      ds_mean_cg_prec.theta_e, 
                      ds_mean_cg_prec.alt, 
                      q_an_cong_prec, 
                      cmap_rs_congestus, 
                      'congestus prec',
                      cmap_min,
                      cmap_max)
    
    # plot colored line for shallow non prec
    line_sh = plot_colored_line(ax1, 
                      ds_mean_sl_nonprec.theta_e, 
                      ds_mean_sl_nonprec.alt, 
                      q_an_sl_nonprec, 
                      cmap_rs_congestus, 
                      'shallow non prec',
                      cmap_min,
                      cmap_max)

    
    ax1.set_xlabel('$\Theta_e$ [K]')
    ax1.set_xlim(322., 350.)
    ax1.set_ylim(0., 4000.)
    ax1.set_ylabel('Altitude [m]')
    
    ax2.set_xlabel('$\Theta_e$ [K]')
    ax2.set_xlim(322., 350.)
    ax2.set_ylim(0., 4000.)
    fig.colorbar(line_cong, 
                 ax=ax2, 
                 orientation='horizontal', 
                 pad=0.15, 
                 label='q anomaly congestus [gkg$^{-1}$]')
    
    # add a second colorbar
    fig.colorbar(line_sh, 
                 ax=ax1, 
                 orientation='horizontal', 
                 pad=0.15, 
                 label='q anomaly shallow [gkg$^{-1}$]')

    
    # plot ct distribution along y axis
    #ax2.set_xlim(0., 0.1)

    
    #ax.legend(handles=[line_sh, line_cong, line], frameon=True)
    fig.savefig('/net/ostro/plots_rain_paper/figure9bis_rs_v2.png')
    
def fig_paper(ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec, ds_clear, ct_sl, ct_cg):
    '''
    plot figure for the paper to justify cold dry air coming dow'''
    from matplotlib import gridspec
    # calculate binned histograms of cloud top heights
    bins = np.arange(0., 4000., 200.)
    bin_width = bins[1] - bins[0]

    hist_shallow, bins_shallow = np.histogram(ct_sl, bins=bins, density=True)
    hist_congestus, bins_congestus = np.histogram(ct_cg, bins=bins, density=True)
    
    # Adjust bin coordinates
    bins_shallow_centered = bins_shallow[:-1] + bin_width / 2
    bins_congestus_centered = bins_congestus[:-1] + bin_width / 2
    

    # calculate q anomalies for each class to be plotted as colors
    q_an_cong_prec = ds_mean_cg_prec.q.values - ds_clear.q.values
    q_an_cong_nonprec = ds_mean_cg_nonprec.q.values - ds_clear.q.values
    q_an_sl_nonprec = ds_mean_sl_nonprec.q.values - ds_clear.q.values
    
    # setting color maps edges
    cmap_min = -4.
    cmap_max = 4.
    symbol_size = 150

    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    
    fig = plt.figure(figsize=(15, 10)) 

    ax = fig.add_subplot(111)
    
    
    ax_twin = ax.twiny()
    ax_twin.plot(hist_shallow, bins_shallow_centered, color='lightgray', linewidth=2, alpha=0.5, label='cloud tops shallow')
    ax_twin.plot(hist_congestus, bins_congestus_centered, color='dimgrey', linewidth=2, alpha=0.5, label='cloud tops congestus')
    ax_twin.fill_betweenx(bins_shallow_centered, 0., hist_shallow, color='lightgray', alpha=0.1)
    ax_twin.fill_betweenx(bins_congestus_centered, 0., hist_congestus, color='dimgrey', alpha=0.1)
    ax_twin.axis('off')
    
    # reverse x axis
    #ax_twin.invert_xaxis()
    # set x axis limits
    ax_twin.set_xlim(0., 0.01)
    ax_twin.legend(frameon=False, loc='upper right')
    
    
    # plot profile of clear sky theta_e as a reference
    ax.plot(ds_clear.theta_e,
            ds_clear.alt,
            color='black', 
            linestyle='--',
            linewidth=2)
    
    # plot colored line for congestus non prec
    line_cong = plot_colored_line(ax, 
                      ds_mean_cg_nonprec.theta_e, 
                      ds_mean_cg_nonprec.alt, 
                      q_an_cong_nonprec, 
                      cmap_rs_congestus, 
                      'congestus non prec',
                      cmap_min,
                      cmap_max)
    
     # plot colored line for congestus non prec
    line = plot_colored_line(ax, 
                      ds_mean_cg_prec.theta_e, 
                      ds_mean_cg_prec.alt, 
                      q_an_cong_prec, 
                      cmap_rs_congestus, 
                      'congestus prec',
                      cmap_min,
                      cmap_max)
    # plot colored line for congestus non prec
    line_sh = plot_colored_line(ax, 
                      ds_mean_sl_nonprec.theta_e, 
                      ds_mean_sl_nonprec.alt, 
                      q_an_sl_nonprec, 
                      cmap_rs_shallow, 
                      'shallow non prec',
                      cmap_min,
                      cmap_max)

    
    ax.set_xlabel('$\Theta_e$ [K]')
    ax.set_xlim(322., 350.)
    ax.set_ylim(0., 4000.)
    
    ax.set_ylabel('Altitude (m)')
    fig.colorbar(line_cong, ax=ax, label='q anomaly congestus (g/kg)')
    
    # add a second colorbar
    fig.colorbar(line_sh, ax=ax, label='q anomaly shallow (g/kg)')

    
    # plot ct distribution along y axis
    #ax2.set_xlim(0., 0.1)

    
    #ax.legend(handles=[line_sh, line_cong, line], frameon=True)
    fig.savefig('/net/ostro/plots_rain_paper/figure9bis_rs.png')
    

def plot_colored_line(ax, x, y, z, cmap_name, label, cmap_min, cmap_max):
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
        
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(cmap_min, cmap_max)


    # set line style based on class
    if label == 'congestus non prec':
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('--')
        lc.set_linewidth(4)

    elif label == 'congestus prec':
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('solid')
        lc.set_linewidth(6)
    else:
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        line = ax.add_collection(lc)
        line.set_linestyle('--')
        lc.set_linewidth(6)
    

    # Add the legen

    return(line)

    
def plot_mean_q_theta_e_color_altitude(ds_clear, ds_mean_sl_nonprec, ds_mean_cg_nonprec, ds_mean_sl_prec, ds_mean_cg_prec):
    ''' '''
        
    font_val = 24
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val-6
    mpl.rcParams['ytick.labelsize'] = font_val-6
    mpl.rcParams['legend.fontsize'] = font_val-6
    mpl.rcParams['figure.titlesize'] = font_val
    
    fig = plt.figure(figsize=(10,10))
    
    cmap = mpl.cm.get_cmap('Greys')
    norm = mpl.colors.Normalize(vmin=0., vmax=4000.)
    cmap.set_under('white')
    symbol_size = 150
    
    ax = fig.add_subplot(111)
    ax.scatter(ds_clear.theta_e, 
            ds_clear.q, 
            c=ds_clear.alt,
            label='clear sky', 
            marker='o',
            s=symbol_size,
            cmap=cmap,
            norm=norm,
            edgecolors='black')
    ax.scatter(ds_mean_sl_nonprec.theta_e, 
            ds_mean_sl_nonprec.q, 
            c=ds_mean_sl_nonprec.alt,
            label='shallow non prec', 
            marker='o',
            s=symbol_size,
            cmap=cmap,
            norm=norm,
            edgecolors=COLOR_SHALLOW)
    ax.scatter(ds_mean_cg_nonprec.theta_e,
            ds_mean_cg_nonprec.q,
            c=ds_mean_cg_nonprec.alt,
            label='congestus non prec',
            marker='o',
            s=symbol_size,
            cmap=cmap,
            norm=norm,
            edgecolors=COLOR_CONGESTUS)
    ax.scatter(ds_mean_sl_prec.theta_e,
            ds_mean_sl_prec.q,
            c=ds_mean_sl_prec.alt,
            label='shallow prec',
            marker='o',
            s=symbol_size,
            cmap=cmap,
            norm=norm,
            edgecolors=COLOR_SHALLOW)
    ax.scatter(ds_mean_cg_prec.theta_e,
            ds_mean_cg_prec.q,
            c=ds_mean_cg_prec.alt,
            label='congestus prec',
            marker='^',
            s=symbol_size,
            cmap=cmap,
            norm=norm,
            edgecolors=COLOR_CONGESTUS)
            
    ax.set_xlabel('$\Theta_e$ [K]')
    ax.set_xlim(320., 350.)
    ax.set_ylabel('q [g/kg]')
    ax.invert_yaxis()
    ax.set_ylim(17.5, 0)
    ax.legend(frameon=True)
        
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greys'), 
                        ax=ax,  # Extend colorbar for the entire height of the figure
                        orientation='vertical', 
                        label='Altitude (m)')

    
    
    fig.savefig('/net/ostro/plots_rain_paper/q_thetae_classes_mean_rs.png')
    return(fig)
    
def plot_scatter_q_theta(ds1, ds2, ds3, ds4):
    '''
    function to plot q vs theta_e for each class
    Arguments:
        ds1: xarray dataset of shallow non prec
        ds2: xarray dataset of congestus non prec
        ds3: xarray dataset of shallow prec
        ds4: xarray dataset of congestus prec
        
    returns:
    
    fig: figure with q vs theta_e for each class
    '''
    font_val = 20
    # Set the default font size to 20 for all fonts
    mpl.rcParams['font.size'] = font_val
    mpl.rcParams['axes.titlesize'] = font_val
    mpl.rcParams['axes.labelsize'] = font_val
    mpl.rcParams['xtick.labelsize'] = font_val
    mpl.rcParams['ytick.labelsize'] = font_val
    mpl.rcParams['legend.fontsize'] = font_val
    mpl.rcParams['figure.titlesize'] = font_val
    
    # plot scatter plot of q vs theta_e for each class
    fig, ax = plt.subplots(2,2, figsize=(25,20))
    
     # define colorbars for sounding 1 and sounding 2
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0., vmax=4.)
    cmap.set_under('white')
    
    # 4 subplots
    plot_scatter(ax[0,0], 
                 ds1, 
                 mpl.cm.get_cmap('viridis'), 
                 norm, 
                 'q [g/kg]',
                 '$\Theta_e$ [K]',
                 'shallow non prec',
                 )
    
    plot_scatter(ax[0,1], 
                 ds2, 
                 mpl.cm.get_cmap('viridis'), 
                 norm, 
                 'q [g/kg]',
                 '$\Theta_e$ [K]',
                 'congestus non prec',
                 )
    plot_scatter(ax[1,0], 
                 ds3, 
                 mpl.cm.get_cmap('viridis'), 
                 norm, 
                 'q [g/kg]',
                 '$\Theta_e$ [K]',
                 'shallow prec',
                 )
    
    plot_scatter(ax[1,1], 
                 ds4, 
                 mpl.cm.get_cmap('viridis'), 
                 norm, 
                 'q [g/kg]',
                 '$\Theta_e$ [K]',
                 'congestus prec',
                 )   
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                                        cmap=cmap), 
                  ax=ax[0], 
                  orientation='vertical', 
                  label='Altitude (km)')
    
    fig.savefig('/net/ostro/plots_rain_paper/q_thetae_classes_rs.png')
    
def plot_scatter(ax, ds, cmap, norm, x_label, y_label, title):
    '''function to plot scatter of points over a subplot and assigns color based on altitude
    '''

    
    # select values non nan in x and y in dataarrays
    x_scatter = ds.theta_e.values.flatten()
    y_scatter = ds.q.values.flatten()
    z_scatter = ds.height_matrix.values.flatten()
    
    # select values non nan in x and y
    i_good = ~np.isnan(x_scatter) * ~np.isnan(x_scatter)
    x_scatter = x_scatter[i_good]
    y_scatter = y_scatter[i_good]
    z_scatter = z_scatter[i_good]*1e-3 # convert altitude to km
    
    
    ax.scatter(x_scatter, 
               y_scatter, 
               c=z_scatter, 
               cmap=cmap, 
               marker = 'o',
               edgecolors='black',
               norm=norm)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(295, 350)
    ax.invert_yaxis()  # Invert the y-axis
    ax.set_ylim(20, 0)

    return(None)


    # plot all profiles of all classes for q
    plot_rs_profiles(ds_sl_prec, 'q', 'sl_prec', 0. ,20, 'q (g/kg)')
    plot_rs_profiles(ds_sl_nonprec, 'q','sl_nonprec', 0. ,20, 'q (g/kg)')
    plot_rs_profiles(ds_cg_prec, 'q','cg_prec', 0. ,20, 'q (g/kg)')
    plot_rs_profiles(ds_cg_nonprec, 'q','cg_nonprec', 0. ,20, 'q (g/kg)')
    
    # plot all profiles for all classes for theta_e
    plot_rs_profiles(ds_sl_prec, 'theta_e', 'sl_prec', 290. ,320, 'theta_e (K)')
    plot_rs_profiles(ds_sl_nonprec, 'theta_e','sl_nonprec', 290. ,320, 'theta_e (K)')
    plot_rs_profiles(ds_cg_prec, 'theta_e','cg_prec', 290. ,320, 'theta_e (K)')
    plot_rs_profiles(ds_cg_nonprec, 'theta_e','cg_nonprec', 290. ,320, 'theta_e (K)')
    
    
def plot_rs_profiles(ds, var_name, class_name, vmin, vmax, var_string):
    '''
    
    '''
    # plot figure q profiles
    fig, ax = plt.subplots()
    
    # read launch times for the class selected
    launch_times = pd.to_datetime(ds.launch_time.values)
    print('launch times', launch_times)
    
    # set colorbar based on launch_time
    # Normalize the timestamps
    norm = mcolors.Normalize(vmin=launch_times.min().value, vmax=launch_times.max().value)

    # Create a ScalarMappable
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])

    # Add the colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Timestamp')
    
    # Format the colorbar ticks as month-day-hour-minute
    #cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    for i_s, sounding_name in enumerate(ds.sounding.values):
        print('sounding name', sounding_name)
        ds_sel = ds.isel(sounding=i_s)
        print(ds_sel[var_name].values)
        pdb.set_trace()
        color = plt.cm.Blues(norm(launch_times[i_s].value))
        ax.plot(ds_sel[var_name].values, ds_sel.alt, color=color)
    
    ax.set_ylim(0, 4000)
    ax.set_xlim(vmin, vmax)
    ax.set_xlabel(var_string)
    ax.set_ylabel('Altitude (m)')
    
    fig.savefig('/net/ostro/plots_rain_paper/profiles_'+var_name+'_rs_'+class_name+'.png')
    
    
def calc_q_theta_e(ds):
    '''
    function to calculate specific humidity in g/kg and equivalent potential temperature in K for each sounding
    Arguments:
        ds: xarray dataset with radiosonde data
    Dependencies:
        metply.calc equivalent_potential_temperature: function to calculate equivalent potential temperature
        metpy.calc specific_humidity_from_dewpoint: function to calculate specific humidity from dewpoint
    Returns:
        ds: xarray dataset with specific humidity and equivalent potential temperature calculated for each sounding
    '''
    theta_e_matrix = np.zeros((len(ds.sounding.values), len(ds.alt.values))) # in C
    q_matrix = np.zeros((len(ds.sounding.values), len(ds.alt.values))) # in g/kg
    height_matrix = np.zeros((len(ds.sounding.values), len(ds.alt.values))) # in m
    
    
    for i_s, sounding_name in enumerate(ds.sounding.values):
        
        # reading the date
        date = sounding_name.split('__')[-1]
        print('processing date', date)
        
        # selecting the rs of the date
        ds_sel = ds.isel(sounding=i_s)
        
        # reading pressure, tempetrature and dewpoint
        p = ds_sel['p'].values # in Pa
        T = ds_sel['ta'].values # in K
        Td = ds_sel['dp'].values # in K
        
        # convert in the right units
        # convert p from Pa to hpa
        p = p/100
        # convert T from K to C
        T = T - 273.15
        # convert Td from K to C
        Td = Td - 273.15
        
        for i_h, z in enumerate(ds_sel.alt.values): 
            #print(p[i_h], T[i_h], Td[i_h])
            
            #print(equivalent_potential_temperature(p[i_h] * units.hPa, T[i_h] * units.degC, Td[i_h] * units.degC))
            theta_e_matrix[i_s,i_h] = equivalent_potential_temperature(p[i_h] * units.hPa, T[i_h] * units.degC, Td[i_h] * units.degC).magnitude
            q_matrix[i_s,i_h] = specific_humidity_from_dewpoint(p[i_h] * units.hPa, Td[i_h] * units.degC).to('g/kg').magnitude
            height_matrix[i_s,i_h] = z
            
    print('theta values max and min in K', np.nanmax(theta_e_matrix), np.nanmin(theta_e_matrix))
    
    # add new variables to the ds datase
    ds['theta_e'] = (('sounding', 'alt'), theta_e_matrix)
    ds['q'] = (('sounding', 'alt'), q_matrix)
    ds['height_matrix'] = (('sounding', 'alt'), height_matrix)
    # add units to theta_e and q
    ds['theta_e'].attrs['units'] = 'K'
    ds['q'].attrs['units'] = 'g/kg'
    
    return(ds)
    
    
def group_soundings_per_class(ds):
    ''' 
    function to group radiosondes by cloud type classification
    Arguments:
        ds: xarray dataset with radiosonde data
    Returns:
        ds_sl_prec: xarray dataset with shallow clouds and precipitation
        ds_sl_nonprec: xarray dataset with shallow clouds and no precipitation
        ds_cg_prec: xarray dataset with congestus clouds and precipitation
        ds_cg_nonprec: xarray dataset with congestus clouds and no precipitation        
    '''
    # group radiosondes by cloud type classification
    is_shallow = ds.shape == 0
    is_congestus = ds.shape == 1
    is_clear = ds.shape == -1

    # selecting prec and non prec
    is_prec_ground = ds.rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 
    
    # segregating with respect to shallow and deep clouds
    ds_sl_prec = ds.isel(sounding=is_sl_prec)
    ds_sl_nonprec = ds.isel(sounding=is_sl_non_prec)
    ds_cg_prec = ds.isel(sounding=is_cg_prec)
    ds_cg_nonprec = ds.isel(sounding=is_cg_non_prec)
    ds_clear = ds.isel(sounding=is_clear)
    
    return ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, ds_clear
    
def assign_cloud_type(ds):
    ''' 
    function to assign cloud type classification to each radiosonde
    Arguments:
        ds: xarray dataset with radiosonde data
    Dependencies:
        read_cloud_class: function to read cloud type classification
        read_rain_ground: function to read rain at the ground flag
        read_cloud_base: function to read cloud base heights
        
    Returns:
        ds: xarray dataset with cloud type classification assigned to each radiosonde
        
    '''
    # read cloud types (shallow and congestus)
    ds_ct = read_cloud_class()

    # read rain at the ground flag 
    ds_r = read_rain_ground()
    
    # read cloud base heights 
    ds_cb_ct = read_cloud_top_base()
    
    
    
    # align cloud type flag and rain ground flag
    ds_ct, ds_r, ds_cp = xr.align(ds_ct, ds_r, ds_cb_ct, join="inner")
    ds_cp = xr.merge([ds_ct, ds_r, ds_cb_ct])
        
    # for each radiosonde, find closest ds_cp time stamp with a tolerance of 30 seconds
    # assign the cloud type classification to the sounding
    shape = []
    rain_ground = []
    cloud_base = []
    cloud_top = []
    
    for i_t, time_rs in enumerate(ds.launch_time.values):
        
        time_rs = pd.to_datetime(time_rs)
                
        # find closest time stamp in ds_cp to the sounding datetime string
        
        time_diff = np.abs(pd.to_datetime(ds_cp.time.values) - time_rs)
        
        ind_min = np.where(time_diff == np.min(time_diff))[0][0]
        
        #print(' selected time', ds_cp.time.values[ind_min])
        #print('time difference in seconds', time_diff[ind_min].seconds)
        if time_diff[ind_min].seconds > 30:
            #print('time difference is too large, skipping')
            shape.append(np.nan)
            rain_ground.append(np.nan)
            cloud_base.append(np.nan)
            cloud_top.append(np.nan)
        else:
            #assign cloud type classification to sounding
            shape.append(ds_cp.shape[ind_min])
            rain_ground.append(ds_cp.flag_rain_ground[ind_min])
            cloud_base.append(ds_cp.cloud_base[ind_min])
            cloud_top.append(ds_cp.cloud_top_height[ind_min])
            
    # add cloud type classification to the dataset
    ds['shape'] = (('sounding'), shape)
    ds['rain_ground'] = (('sounding'), rain_ground)
    ds['cloud_base'] = (('sounding'), cloud_base)
    ds['cloud_top'] = (('sounding'), cloud_top)
     
    return(ds)

        
    
if __name__ == '__main__':
    main() 