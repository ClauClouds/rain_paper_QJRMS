"""
code to produce the fig04 of the final version of the paper.
the figure focuses on the cloud formation stage and development
showing what happens in the subcloud layer in terms of anomalies
and what happens in the cloud microphysics 

subplots: 
1) first row: diurnal cycle of LCL
2) second row: 3 subpanel with anomalies of vertical velocity, specific humidity and virtual pot temp
3) third row: 2 subpanels with vd vz Ze and sk vs Ze
"""
from readers.ship import read_and_save_P_file
from readers.lidars import read_anomalies
from readers.radar import read_lwp

from readers.cloudtypes import read_cloud_class, read_rain_ground, read_cloud_base, read_in_clouds_radar_moments
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_paper_plots
from readers.lcl import read_lcl, read_diurnal_cycle_lcl
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import xarray as xr
#from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import metpy.calc as mpcalc
from metpy.units import units
import pdb
from dask.diagnostics import ProgressBar
from figures.fig03_diurnal_all import prepare_data, statistics

ProgressBar().register()
from figures.fig03_diurnal_all import prepare_data, statistics
def main():
    
    # first row of the plot: relative occurrence of cloud types
    ds = prepare_data()
    dct_stats = statistics(ds)

    # first row of plot:  lcl diurnal cycle for the plot
    lcl_dc = read_diurnal_cycle_lcl(path_diurnal_cycle_arthus)
    
    # read cloud types (shallow and congestus)
    ds_ct = read_cloud_class()

    # read rain at the ground flag 
    ds_r = read_rain_ground()
    
    # read cloud base heights 
    ds_cb = read_cloud_base()
    
    # align cloud type flag and rain ground flag
    ds_ct, ds_r, ds_cp = xr.align(ds_ct, ds_r, ds_cb, join="inner")
    ds_cp = xr.merge([ds_ct, ds_r, ds_cb])
    
    # second row of plots: derive specific humidity and virtual potential temperature from lidar data
    ds_therm = calc_thermo_prop()
    
    # split datasets in q and theta_v
    ds_q, ds_theta_v, ds_td, ds_theta_e = split_dataset(ds_therm)
    
    ds_q_sl_prec, ds_q_sl_nonprec, ds_q_cg_prec, ds_q_cg_nonprec, ds_q_clear = prepare_therm_profiles(ds_q, ds_cp, 'q')
    ds_theta_v_sl_prec, ds_theta_v_sl_nonprec, ds_theta_v_cg_prec, ds_theta_v_cg_nonprec, ds_theta_v_clear = prepare_therm_profiles(ds_theta_v, ds_cp, 'theta_v')
    #ds_theta_e_sl_prec, ds_theta_e_sl_nonprec, ds_theta_e_cg_prec, ds_theta_e_cg_nonprec, ds_theta_e_clear = prepare_therm_profiles(ds_theta_e, ds_cp, 'theta_e')
    
    # scatter plot of q and theta_e for each class    
    #plot_scatter_q_theta_e(ds_q_sl_prec,
    #                       ds_q_sl_nonprec,
    #                       ds_q_cg_prec,
    #                       ds_q_cg_nonprec,
    #                       ds_theta_e_sl_prec,
    #                       ds_theta_e_sl_nonprec,
    #                       ds_theta_e_cg_prec,
    #                       ds_theta_e_cg_nonprec)
    
    # prepare data for horizontal wind speed ()
    ds_sl_hw, ds_sl_prec_hw, ds_sl_nonprec_hw, ds_cg_hw, ds_cg_prec_hw, ds_cg_nonprec_hw = prepare_anomaly_profiles(ds_cp, "H_wind_speed", lcl_dc)
    

    # prepare data (calculate mean and std of the anomaly profiles for shallow/congestus in prec and non prec)
    ds_sl, ds_cg, ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec = prepare_anomaly_profiles(ds_cp, "VW", lcl_dc)
    
    # calculate distributions of cloud base heights with respect to the LCL for each flag
    ds_cb_sllcl, ds_cb_cglcl = calc_cblcl(ds_cp)
    #plot_distr(ds_cb_sllcl, ds_cb_cglcl)   
    
    # read radar moments above cloud base
    data = read_in_clouds_radar_moments()

    # plot figure of anomalies of vertical velocity, specific humidity and virtual potential temperature
    # Example usage
    plot_multipanel_figure_V2(lcl_dc,
                           dct_stats, 
                           ds_sl_prec, 
                           ds_sl_nonprec, 
                           ds_cg_prec,
                           ds_cg_nonprec, 
                           ds_q_sl_prec, 
                           ds_q_sl_nonprec, 
                           ds_q_cg_prec, 
                           ds_q_cg_nonprec,
                           ds_q_clear,  
                           ds_theta_v_sl_prec, 
                           ds_theta_v_sl_nonprec, 
                           ds_theta_v_cg_prec, 
                           ds_theta_v_cg_nonprec, 
                           ds_theta_v_clear, 
                           ds_cb_sllcl, 
                           ds_cb_cglcl,
                           ds_sl_prec_hw,
                           ds_sl_nonprec_hw,
                           ds_cg_prec_hw,
                           ds_cg_nonprec_hw,
                           data)
    


def plot_scatter_q_theta_e(ds_q_sl_prec,
                           ds_q_sl_nonprec,
                           ds_q_cg_prec,
                           ds_q_cg_nonprec,
                           ds_theta_e_sl_prec,
                           ds_theta_e_sl_nonprec,
                           ds_theta_e_cg_prec,
                           ds_theta_e_cg_nonprec):
    
    # plot scatter plot of q vs theta_e for each class
    fig = plt.figure(figsize=(10, 10))
    
    # set colorbar with height
    # define colorbars for sounding 1 and sounding 2
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=3.0)
    cmap.set_under('white')
    
    # 4 subplots
    ax1 = fig.add_subplot(221)
    plot_scatter(ax1, 
                 ds_q_sl_prec, 
                 ds_theta_e_sl_prec, 
                 'q [g/kg]', 
                 '$\Theta_e$ [K]', 
                 'shallow prec', 
                 cmap, 
                 norm)
    
    # add second subplot
    ax2 = fig.add_subplot(222)
    plot_scatter(ax2,
                    ds_q_sl_nonprec,
                    ds_theta_e_sl_nonprec,
                    'q [g/kg]',
                    '$\Theta_e$ [K]',
                    'shallow non prec',
                    cmap, 
                    norm)
    
    # add third subplot
    ax3 = fig.add_subplot(223)
    plot_scatter(ax3,
                ds_q_cg_prec,
                ds_theta_e_cg_prec,
                'q [g/kg]',
                '$\Theta_e$ [K]',
                'congestus prec',
                    cmap, 
                    norm)
    
    
    # add fourth subplot
    ax4 = fig.add_subplot(224)
    plot_scatter(ax4,
                ds_q_cg_nonprec,
                ds_theta_e_cg_nonprec,
                'q [g/kg]',
                '$\Theta_e$ [K]',
                'congestus non prec',
                    cmap, 
                    norm)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), 
                        ax=[ax1, ax2, ax3, ax4],  # Extend colorbar for the entire height of the figure
                        orientation='vertical', 
                        label='Altitude (m)')

    fig.savefig('/net/ostro/plots_rain_paper/fig_04_scatter_q_theta_e.png')     
    return(fig)


def create_time(hours, month, year, today):
    """
    function to create a time array that works for plotting relative occurrences 
    over the diurnal cycle of lcl

    Args:
        hours (array): _description_
        month (scalar): current month
        year (scalar): current year
        today(string): insert today's date
    Returns:
        _type_: _description_
    """
    import numpy as np
    import pandas as pd

    hh = np.array(hours)
    base_date =  np.datetime64(f'{year}-{month:02d}-'+today)

    # Create the datetime64 array with the specified hours
    datetime_array = base_date + pd.to_timedelta(hh, unit='h')
    
    return datetime_array

def plot_scatter(ax, y, x, y_label, x_label, title, cmap, norm):
    
    # select values non nan in x and y in dataarrays
    x_scatter = x.values.flatten()
    y_scatter = y.values.flatten()
    
    print('time dimension', x.shape[0])
    print('height dimension', x.shape[1])
    
    
    # create a matrix of heights by repeating height values for each time step
    height_matrix = np.repeat(x.height.values[:, np.newaxis], 
                              x.shape[0], axis=0)
    
    print(np.shape(height_matrix), np.shape(x))
    
    z_scatter = height_matrix.flatten()
    i_good = ~np.isnan(x_scatter) * ~np.isnan(x_scatter)
    
    x_scatter = x_scatter[i_good]
    y_scatter = y_scatter[i_good]
    z_scatter = z_scatter[i_good]*1e-3
    print(len(x_scatter), len(y_scatter), len(z_scatter))
    print(z_scatter)
    pdb.set_trace()

    # calculating mean values and std values of the distribution
    mean_x = np.nanmean(x_scatter)
    mean_y = np.nanmean(y_scatter)
    std_x = np.nanstd(x_scatter)
    std_y = np.nanstd(y_scatter)
    
    ax.scatter(x_scatter, 
               y_scatter, 
               c=z_scatter, 
               cmap=cmap, 
               s=0.1,
               norm=norm,
               alpha=0.5)
    
    # plot position of mean plus/minus the standard deviation
    #ax.errorbar(mean_x,
    #            mean_y,
    #            xerr=std_x,
    #            yerr=std_y,
    #            fmt='o',
    #            color='black',
    #            label='mean $\pm$ std')
    
    ax.scatter(mean_x,
               mean_y,
               cmap='black',
               marker='x',
               label='mean')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 30)
    ax.set_xlim(293, 300)
    
    return(None)

# Example function to plot and save figures
def plot_multipanel_figure(lcl_dc, 
                           ds_stats,
                           ds_sl_prec, 
                           ds_sl_nonprec, 
                           ds_cg_prec,
                           ds_cg_nonprec, 
                           ds_q_sl_prec, 
                           ds_q_sl_nonprec, 
                           ds_q_cg_prec, 
                           ds_q_cg_nonprec,
                           ds_q_clear, 
                            ds_theta_v_sl_prec, 
                            ds_theta_v_sl_nonprec, 
                            ds_theta_v_cg_prec, 
                            ds_theta_v_cg_nonprec,
                            ds_theta_v_clear, 
                            ds_cb_sllcl,
                            ds_cb_cglcl,
                            ds_sl_prec_hw,
                            ds_sl_nonprec_hw,
                            ds_cg_prec_hw,
                            ds_cg_nonprec_hw,
                            data):
    
    # Set the default font size for all elements
    plt.rcParams.update({'font.size': 12})
    
    # read relative occurrences of cloud types
    rel_occ_co_r_diurnal=ds_stats["rel_occ_co_r_diurnal"]
    rel_occ_co_nr_diurnal=ds_stats["rel_occ_co_nr_diurnal"]
    rel_occ_sh_diurnal=ds_stats["rel_occ_sh_diurnal"]
    rel_occ_co_diurnal=ds_stats["rel_occ_co_diurnal"]
    
    # convert rel_occ_co_diurnal.hour into a datetime object
    time_rel_occ = create_time(rel_occ_co_diurnal.hour.values, 3, 2025, '11')
    
    bins = np.arange(-0.5, 1.75,0.25)
    bin_width = bins[1] - bins[0]

    hist_shallow, bins_shallow = np.histogram(ds_cb_sllcl, bins=bins, density=True)
    hist_congestus, bins_congestus = np.histogram(ds_cb_cglcl, bins=bins, density=True)
    
    # Adjust bin coordinates
    bins_shallow_centered = bins_shallow[:-1] + bin_width / 2
    bins_congestus_centered = bins_congestus[:-1] + bin_width / 2
    
    
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
    
    # calculating 2d histograms to plot
    hist_ZESK_cloud_s, x_ze_cloud_s, y_sk_cloud_s = np.histogram2d(ze_s, 
                                                                   sk_s, 
                                                                   bins=[40, 40], 
                                                                   range=[[-50., 25.],
                                                                          [-1., 1.]], 
                                                                   density=True)
    hist_ZEVD_cloud_s, x_ze2_cloud_s, y_vd_cloud_s = np.histogram2d(ze_s, vd_s, 
                                                                    bins=[40, 40], 
                                                                    range=[[-50., 25.],
                                                                           [-4., 2.]], 
                                                                    density=True)

    
    
    # Create a figure
    fig = plt.figure(figsize=(15, 15))
    
    # Create a GridSpec with 3 rows and 2 columns
    subplots = plt.subplots()
    gs = subplots[0].add_gridspec(3, 6, height_ratios=[1, 3, 4], width_ratios=[2, 2, 2, 2, 2, 2])
    
    # Add subplots with different sizes
    
    # lcl diurnal cycle
    ax0 = fig.add_subplot(gs[0, :])  # First row, spans all columns
    ax0.plot(lcl_dc.time.values, lcl_dc.lcl_dc.values, linewidth=3, color='black', label='LCL')
    ax0.set_title(' a) LCL diurnal cycle', loc='left', fontweight='black')
    ax0.set_ylabel('Height [m]')
    ax0.set_xlabel('Time [hh:mm] (Local time)')
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0.grid(False)
    ax0.set_xlim([lcl_dc.time.values[0], lcl_dc.time.values[-1]])

    # add new y axis for cloud relative occurrence
    ax0b = ax0.twinx()
    ax0b.grid(False)
    ax0b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax0b.set_ylabel("Rel. Occ.")
    ax0b.set_ylim(0, 0.07)
    ax0b.plot(
        time_rel_occ,
        rel_occ_sh_diurnal,
        color=COLOR_SHALLOW,
        label="Shallow",
        linewidth=3, 
        alpha=0.5
    )
    ax0b.plot(
        time_rel_occ,
        rel_occ_co_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus",
        linewidth=3,
        alpha=0.5

    )
    
    ax0b.plot(
        time_rel_occ,
        rel_occ_co_r_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus prec",
        linestyle='--',
        linewidth=3,
        alpha=0.5

    )
    
    ax0b.plot(
        time_rel_occ,
        rel_occ_co_nr_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus non prec",
        linestyle=':', 
        linewidth=3,
        alpha=0.5

    )


    # Retrieve the legend handles and labels from both axes
    handles0, labels0 = ax0.get_legend_handles_labels()
    handles0b, labels0b = ax0b.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles0 + handles0b
    labels = labels0 + labels0b

    # Create a single legend with the combined handles and labels
    ax0b.legend(handles, labels, loc="upper left", frameon=False)
        
    # cloud base height distributions
    ax1 = fig.add_subplot(gs[1, 0])  # Second row, first column, profiles of w
    ax1.plot(hist_shallow, bins_shallow_centered, color=COLOR_SHALLOW, linewidth=2)
    ax1.plot(hist_congestus, bins_congestus_centered, color=COLOR_CONGESTUS, linewidth=2)
    ax1.fill_betweenx(bins_shallow_centered, 0., hist_shallow, color=COLOR_SHALLOW, alpha=0.5)
    ax1.fill_betweenx(bins_congestus_centered, 0., hist_congestus, color=COLOR_CONGESTUS, alpha=0.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks(np.arange(-0.5, 1.5, 0.25), minor=True)
    ax1.set_yticklabels([-0.5, -0.25, "LCL", 0.25, 0.5, 1, 1.25, 1.5])
    ax1.set_ylabel("Height above\nLCL [km]")
    ax1.set_xlabel("Norm. Occ.")
    ax1.set_title(" b) CB distribution", loc='left', fontweight='black')
    
    # profiles of w
    ax2 = fig.add_subplot(gs[1, 1])  # Second row, first column, profiles of w
    plot_profiles(ax2, 
                  ds_sl_prec, 
                  ds_sl_nonprec, 
                  ds_cg_prec,
                  ds_cg_nonprec,
                  -0.4, 0.4, 'w anomaly [m/s]')
    ax2.set_title('c) w anomaly', loc='left', fontweight='black')

    ax2bis = fig.add_subplot(gs[1, 2])  # Second row, first column, profiles of w
    plot_profiles(ax2bis, 
                  ds_sl_prec_hw, 
                  ds_sl_nonprec_hw, 
                  ds_cg_prec_hw,
                  ds_cg_nonprec_hw,
                  -1, 1, 'H wind anomaly [m/s]')
    ax2bis.set_title('d) Hspeed anomaly', loc='left', fontweight='black')

    # profiles of q
    ax3 = fig.add_subplot(gs[1, 3], sharey=ax2)  # Second row, second column profiles of q
    plot_profiles(ax3, 
                  ds_q_sl_prec, 
                  ds_q_sl_nonprec, 
                  ds_q_cg_prec, 
                  ds_q_cg_nonprec, 
                  10, 16.5, 'q [g/kg]')
    ax3.plot(ds_q_clear.mean("time"),
             ds_q_clear.height* 1e-3,
             label='clear sky',
             color='black', 
             linewidth=3, 
             linestyle = 'dashdot')

    ax3.set_title('e) spec. humidity', loc='left', fontweight='black')

 
    # profiles of theta_v
    ax4 = fig.add_subplot(gs[1, 4], sharey=ax2)  # Second row, third column profiles of thetav
    plot_profiles(ax4, 
                  ds_theta_v_sl_prec, 
                  ds_theta_v_sl_nonprec, 
                  ds_theta_v_cg_prec, 
                  ds_theta_v_cg_nonprec, 
                  300, 310, '$\Theta_v$[K]')
    ax4.plot(ds_theta_v_clear.mean("time"),
             ds_theta_v_clear.height* 1e-3,
             label='clear sky',
             color='black', 
             linewidth=3, 
             linestyle = 'dashdot')
    ax4.set_title('f) Virt. pot. temp.', loc='left', fontweight='black')

    # last narrow subplot with the legend    
    ax_legend = fig.add_subplot(gs[1, 5])  # Second row, right plot
    ax_legend.axis('off')  # Turn off the axis

    # Generate the legend from ax4
    handles, labels = ax4.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='center', frameon=False)

            
    # scatter Ze vs Vd
    ax5 = fig.add_subplot(gs[2, 0:3])  # Third row, left plot
    ax5.set_title('g) Normalized Occurrences of Ze vs Vd', loc='left', fontweight='black')

    hist_d = ax5.hist2d(ze_c, 
                        vd_c, 
                        bins=(40, 40), 
                        range=([-50., 25.],[-4., 2.]), 
                        cmap=CMAP, 
                        density=True, 
                        cmin=0.0001)
    
    cbar = fig.colorbar(hist_d[3], ax=ax5, orientation='horizontal')
    cs = ax5.contour(x_ze2_cloud_s[:-1], 
                        y_vd_cloud_s[:-1], 
                        hist_ZEVD_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZEVD_cloud_s),
                                np.nanmax(hist_ZEVD_cloud_s), 
                                (np.nanmax(hist_ZEVD_cloud_s)- np.nanmin(hist_ZEVD_cloud_s))/10), 
                        cmap=plt.cm.Greys)
    ax5.clabel(cs, inline=True)
    cbar.set_label('norm. occ. congestus clouds')
    #cbar.ax.tick_params(labelsize=25)
    ax5.set_ylabel("Mean Doppler velocity [ms$^{-1}$] ")
    ax5.set_xlabel("Reflectivity [dBz] ")
    ax5.set_ylim(-4., 2.)
    ax5.set_xlim(-50.,25.)
    
    # scatter Ze vs Sk

    ax6 = fig.add_subplot(gs[2, 3:])  # Third row, right plot
    ax6.set_title('h) Normalized Occurrences of Ze vs Sk', loc='left', fontweight='black')

    hist_d = ax6.hist2d(ze_c, 
                        -sk_c, 
                        bins=(40, 40), 
                        range=([-50., 25.],[-1., 1.]), 
                        cmap=CMAP, 
                        density=True, 
                        cmin=0.0001)
    cbar = fig.colorbar(hist_d[3], ax=ax6, orientation='horizontal')
    cs = ax6.contour(x_ze_cloud_s[:-1], 
                        -y_sk_cloud_s[:-1], 
                        hist_ZESK_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZESK_cloud_s), np.nanmax(hist_ZESK_cloud_s),
                                (np.nanmax(hist_ZESK_cloud_s)- np.nanmin(hist_ZESK_cloud_s))/10),
                        cmap=plt.cm.Greys)
    
    ax6.clabel(cs, inline=True)
    cbar.set_label('norm. occ. congestus clouds')
    #cbar.ax.tick_params(labelsize=25)
    ax6.set_ylabel("Skewness ")
    ax6.set_xlabel("Reflectivity [dBz] ")
    ax6.set_ylim(-1., 1.)
    ax6.set_xlim(-50.,25.)
    
    for ax in [ax0, ax1, ax2, ax2bis, ax3, ax4, ax5, ax6]:  # Loop over all axes    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.tick_params(which='minor', length=3, width=2)
        ax.tick_params(which='major', length=5, width=2)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    
    # Adjust layout
    gs.tight_layout(fig, h_pad=0.1)

    plt.tight_layout()
    
    # Save the figure to a file
    fig.savefig('/net/ostro/plots_rain_paper/fig_04_multipanel_figure_new.png')

def plot_distr(shallow, congestus):
    
    bins = np.arange(-0.5, 1.75,0.25)
    bin_width = bins[1] - bins[0]

    hist_shallow, bins_shallow = np.histogram(shallow, bins=bins, density=True)
    hist_congestus, bins_congestus = np.histogram(congestus, bins=bins, density=True)
    
    # Adjust bin coordinates
    bins_shallow_centered = bins_shallow[:-1] + bin_width / 2
    bins_congestus_centered = bins_congestus[:-1] + bin_width / 2
    
    print(len(hist_shallow), len(bins_shallow))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(bins_shallow_centered, hist_shallow, color=COLOR_SHALLOW, alpha=0.5)
    ax.plot(bins_congestus_centered, hist_congestus, color=COLOR_CONGESTUS, alpha=0.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel('Height above LCL [m]')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of cloud base heights with respect to LCL')
    plt.savefig('/net/ostro/plots_rain_paper/distr_cb_lcl.png')
    
    return
    
def plot_profiles(ax, ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, vmin, vmax, var_name):
    
    ax.plot(ds_sl_nonprec.mean("time", skipna=True),
                 ds_sl_nonprec.height* 1e-3,
                 label='shallow non prec',
                 color=COLOR_SHALLOW, 
                 linewidth=3)
    ax.plot(ds_cg_nonprec.mean("time", skipna=True),
                 ds_cg_nonprec.height* 1e-3,
                 label='congestus non prec',
                color=COLOR_CONGESTUS, 
                linewidth=3)

    ax.plot(ds_sl_prec.mean("time", skipna=True),
                 ds_sl_prec.height* 1e-3,
                 label='shallow prec',
                 color=COLOR_SHALLOW, 
                 linestyle=':', 
                linewidth=3)

    ax.plot(ds_cg_prec.mean("time", skipna=True),
                 ds_cg_prec.height* 1e-3,
                 label='congestus prec',
                 color=COLOR_CONGESTUS, 
                 linestyle=':', 
                linewidth=3)

    ax.set_xlabel(var_name)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim([vmin, vmax])
            
    #ax.set_yticks(np.arange(-1, 1, 0.5), minor=True)
    ax.set_yticks(np.arange(-0.5, 1.5, 0.25), minor=True)
    ax.set_yticklabels([])
    #ax.set_xticks(np.arange(vmin, vmax), minor=True)
    #ax.set_yticklabels([-0.5, -0.25, "LCL", 0.25, 0.5, 1, 1.25, 1.5], fontsize=14)
    #ax.set_ylim(
    #    [
    #        ds_cg_prec.height.isel(
    #            height=ds_cg_prec.any("time").values.argmax()
    #        )
    #        * 1e-3,
    #        0.3,
    #    ]
    #)

    
    return

def calc_lcl_grid_no_dc(ds, lcl_ds, height_var, time_var, var_name):
    """
    function to convert and reinterpolate data on the height referred to the lcl height

    Args:
        ds (xarray dataset): dataset containing the data to be regridded
        lcl_ds (_type_): _description_
        height_var (_type_): _description_
        time_var (_type_): _description_
        var_name (_type_): _description_
    """

    if (height_var != 'height') and (time_var != 'time'):
        # rename time and height in standard way for processing
        ds = ds.rename({height_var:'height', time_var:'time'})

    # calculating vertical resolution
    #dz = 7.45
    dz = ds.height.diff(height_var).mean().values
    print('dz is', dz)
    
    if var_name == 'anomaly':
        dz = 10# 7.45

    print('dz', dz)


    # adding lcl to the dataset variables (also diurnal cycle)
    ds['lcl'] = lcl_ds.lcl.values
    
    # reading dataarray of the input variable and of lcl
    da_var = ds[var_name]
    da_lcl = ds.lcl

    # interpolate data on regular height grid of 7.45 m that covers the
    # height difference to the lcl
    z_rel = np.arange(
        da_var.height.min() - da_lcl.max(),
        da_var.height.max() - da_lcl.min() + dz,
        dz,
    )
    z_rel = z_rel - z_rel[z_rel > 0].min()  # center around zero

    da_var = da_var.interp(
        height=z_rel, method="nearest", kwargs={"fill_value": np.nan}
    )
    
    # calculate shift of all height values at each time step
    # positive shift means each height bin is shifted downward
    rows, columns = np.ogrid[: da_var.shape[0], : da_var.shape[1]]  # 2d indices
    shift = ((da_lcl + dz / 2) // dz).values.astype("int16") # numnber of bins to shift
    
    # reindexing columns with the shift
    columns = columns + shift[:, np.newaxis]

    # set limit to upper bound
    columns[columns >= columns.shape[1]] = columns.shape[1] - 1  # upper bound

    # reading values corresponding to the new indeces
    da_var[:] = da_var.values[rows, columns]
    
    return(da_var)

def calc_cblcl(ds_cp):
    """
    function to calculate the distribution of cloud base heights with respect to the LCL
    and split them for each classification of clouds and rain flags
    Args:
        ds_cb (xarray dataset): dataset containing cloud base heights
    """
    
    # read lcl and interpolate on ds thermodynamic profiles dataset
    ds_lcl = read_lcl()
    ds_lcl_interp = ds_lcl.interp(time=ds_cp.time.values, method='nearest')
    
    # align lcl dataset to the y dataset of the anomalies with flags
    ds_lcl_interp, ds_cp = xr.align(ds_lcl_interp, ds_cp, join="inner")
    
    # call classification of clouds and rain flags
    # selecting cloud types and rain/norain conditions
    is_shallow = ds_cp.shape == 0
    is_congestus = ds_cp.shape == 1

    
    # defining classes 
    is_cg = is_congestus 
    is_sl = is_shallow 

    # segregating with respect to shallow and deep clouds
    ds_sl= ds_cp.isel(time=is_sl)
    lcl_sl = ds_lcl_interp.isel(time=is_sl)
    ds_cg = ds_cp.isel(time=is_cg)    
    lcl_cg = ds_lcl_interp.isel(time=is_cg)
    
    # calculate difference between cloud base and corresponding lcl values in km
    ds_cb_sllcl = 1e-3 * (ds_sl.cloud_base.values - lcl_sl.lcl.values)
    ds_cb_cglcl = 1e-3 * (ds_cg.cloud_base.values - lcl_cg.lcl.values)
    
    return ds_cb_sllcl, ds_cb_cglcl

def prepare_therm_profiles(ds, ds_cp, var_string):
    """
    function to assign to the dataset in ds a cloud classification flag from ds_cp 
    and to interpolate data on the height referred to the lcl height

    Args:
        ds (datase): dataset containing the data to be regridded
        ds_cp (dataset): dataset containing cloud properties data
        var_string (string): name of the variable to be regridded

    Returns:
        _type_: _description_
    """
    # interpolate classification of clouds over time stamps of thermodynamic profiles
    class_interp = ds_cp.interp(time=ds.time.values, method='nearest')                                                           
                                                                    
    # read lcl and interpolate on ds thermodynamic profiles dataset
    ds_lcl = read_lcl()
    ds_lcl_interp = ds_lcl.interp(time=ds.time.values, method='nearest')
    
    # align lcl dataset to the y dataset of the anomalies with flags
    ds_lcl_interp, ds, class_interp = xr.align(ds_lcl_interp, ds, class_interp, join="inner")
    
    ds_therm_lcl = calc_lcl_grid_no_dc(ds, ds_lcl_interp, 'height', 'time', var_string)
    
    # call classification of clouds and rain flags
    # selecting cloud types and rain/norain conditions
    is_shallow = class_interp.shape == 0
    is_congestus = class_interp.shape == 1
    is_clear = class_interp.shape == -1

    # selecting prec and non prec
    is_prec_ground = class_interp.flag_rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 

    # segregating with respect to shallow and deep clouds
    ds_sl_prec = ds_therm_lcl.isel(time=is_sl_prec)
    ds_sl_nonprec = ds_therm_lcl.isel(time=is_sl_non_prec)
    ds_cg_prec = ds_therm_lcl.isel(time=is_cg_prec)
    ds_cg_nonprec = ds_therm_lcl.isel(time=is_cg_non_prec)
    ds_clear = ds_therm_lcl.isel(time=is_clear)
    
    return ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, ds_clear
    


def calc_thermo_prop():
    """
    function to calculate thermodynamic properties from lidar data. Lidar data are postprocessed
    using the standard postprocessing for arthus data
    We derive
    - specific humidity
    - virtual potential temperature
    - pressure
    - virtual temperature
    - equivalent potential temperature
    - dew point temperature
    returns a dataset with all these variables
    
    """
    from readers.lidars import f_call_postprocessing_arthus, f_read_variable_dictionary

    # read lidar data for temperature and humidity
    
    # loop on variables 
    var_string_arr = ['T', 'MR']
    arthus_data = []
    
    for ind, var_string in enumerate(var_string_arr):

        print(var_string)
        
        # read dictionary associated to the variable
        dict_var = f_read_variable_dictionary(var_string)

        # call postprocessing routine for arthus data
        data = f_call_postprocessing_arthus(var_string)

        # rename variables to create a dataset with all variables together
        ds = data.rename({'Height':'height', "Time":'time', 'Product':var_string})

        # Add units as attributes to the variables
        if var_string == 'T':
            ds[var_string].attrs['units'] = 'K'  # Kelvin
        elif var_string == 'MR':
            ds[var_string].attrs['units'] = 'g/kg'  # grams per kilogram

        arthus_data.append(ds)
    
    # merging data into a single dataset
    ds_arthus = arthus_data[0].merge(arthus_data[1])
    
    # calculation of specific humidity variable
    mr = ds_arthus.MR.values* 10**(-3) # g/kg to g/g

    q = mr/(1+mr) # g/g
    q = q*10**(3)   # g/kg
    print('shape of q', np.shape(q))
    
    
    # calculating virtual temperature using lidar data
    Tv = ds_arthus.T.values * (1 + 0.61 *ds_arthus.MR.values* 10**(-3)) # K
    dims   = ['time', 'height']
    coords = {"time":ds_arthus.time.values, "height":ds_arthus.height.values}
    Tv_data    = xr.DataArray(dims=dims, coords=coords, data=Tv,
                 attrs={'long_name':'virtual temperature',
                        'units':'$^{\circ}$K'})
    ds_arthus['Tv'] = Tv_data
    
    print('shape of TV', np.shape(Tv_data))

    # reading surface pressure from ship data re-sampled
    if os.path.exists('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_P.nc'):
        ship_data = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/ship_data/ship_dataset_P.nc')
    else:
        ship_data = read_and_save_P_file()
            
    # interpolate on time resolution of the thermodynamic profiles
    ship_arthus = ship_data.interp(time=ds_arthus.time.values, method='nearest')
    P_surf = ship_arthus.P.values # in Pa

    print('shape of P_surf', np.shape(P_surf))
    
    # calculate pressure using hydrostatic equation and surface pressure values from ship data
    dim_t = len(ds_arthus.time.values)
    dim_h = len(ds_arthus.height.values)
    height = ds_arthus.height.values
    P = np.zeros((dim_t, dim_h))
    P.fill(np.nan)
    
    # calculate mean TV for each profile
    ds_mean = ds_arthus.mean(dim='height', skipna=True)
    Tv_mean = ds_mean.Tv.values
    
    g = 9.8 # ms-2
    Rd = 287  # J (Kg K)-1
    for ind_height in range(dim_h):
        P[:,ind_height] = P_surf * np.exp( - (g *(height[ind_height]-20.))/(Rd*Tv_mean))

    print('shape of P', np.shape(P))
    
    # calculating profiles of virtual potential temperature
    Theta_v = np.zeros((dim_t, dim_h))
    Cp = 1004. # [J Kg-1 K-1]
    Rd = 287.058  # gas constant for dry air [Kg-1 K-1 J]
    mr = ds_arthus.MR.values* 10**(-3) # g/kg to g/g
    T = ds_arthus.T.values # in K
    for indHeight in range(dim_h):
        k = Rd*(1-0.23*mr[:, indHeight])/Cp
        Theta_v[:,indHeight] = ( (1 + 0.61 * mr[:, indHeight]) * T[:, indHeight] * (P_surf/P[:,indHeight])**k)
    
    # calculating dew point temperature profiles
    Td = np.zeros((dim_t, dim_h))
    theta_e = np.zeros((dim_t, dim_h))
    
    # converting T from K to C
    T = T - 273.15
    
    from metpy.calc import dewpoint_from_specific_humidity, equivalent_potential_temperature
    for i_h in range(dim_h):
        Td[:, i_h] = dewpoint_from_specific_humidity(P[:, i_h]*100 * units.hPa, 
                                                     T[:, i_h] * units('degree_Celsius'), 
                                                     q[:, i_h] * units('g/kg')) # in C
    
    for i_h in range(dim_h):
        theta_e[:, i_h] = equivalent_potential_temperature(P[:, i_h]*100 * units.hPa,
                                                           T[:, i_h] * units('degree_Celsius'),
                                                          Td[:, i_h] * units('degree_Celsius'))

    # convert theta_e in K from celsius
    theta_e = theta_e + 273.15
    
    # store q, theta and theta_v in dataset
    ds_out = xr.Dataset(
        {
            "q": (["time", "height"], q),
            "theta_v": (["time", "height"], Theta_v),
            'dew_point': (["time", "height"], Td),
            'theta_e': (["time", "height"], theta_e),
        },
        coords={"time": ds_arthus.time, "height": ds_arthus.height},
    )
    
    # add units attributtes to q, theta and theta_v
    ds_out.q.attrs["units"] = "g/kg"
    ds_out.theta_v.attrs["units"] = "K"
    ds_out.dew_point.attrs["units"] = "C"
    ds_out.theta_e.attrs["units"] = "K"
    
    return(ds_out)


def check_plot_mean_profs(ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, var_name, path_paper_plots, fig_name):
    
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    
    axes[0].annotate(
        "a) specific humidity",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )
    
    axes[1].annotate(
        "b) virtual potential temperature",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    ) 
    
    axes[0].plot(ds_sl_nonprec.q.mean("time"),
                 ds_sl_nonprec.height,
                 label='shallow non prec',
                 color=COLOR_SHALLOW)
    axes[0].plot(ds_cg_nonprec.q.mean("time"),
                 ds_cg_nonprec.height,
                 label='congestus non prec',
                color=COLOR_CONGESTUS)
    axes[0].plot(ds_sl_prec.q.mean("time"), 
                 ds_sl_prec.height,
                 label='shallow prec',
                 color=COLOR_SHALLOW, linestyle=':')
    axes[0].plot(ds_cg_prec.q.mean("time"),
                 ds_cg_prec.height,
                 label='congestus prec',
                 color=COLOR_CONGESTUS, linestyle=':')
    
    axes[0].legend(frameon=False, fontsize=11, loc='upper right')

    axes[1].plot(ds_sl_nonprec.theta_v.mean("time"),
                 ds_sl_nonprec.height,
                 label='shallow non prec',
                 color=COLOR_SHALLOW)
    axes[1].plot(ds_cg_nonprec.theta_v.mean("time"),
                 ds_cg_nonprec.height,
                 label='congestus non prec',
                color=COLOR_CONGESTUS)
    axes[1].plot(ds_sl_prec.theta_v.mean("time"), 
                 ds_sl_prec.height,
                 label='shallow prec',
                 color=COLOR_SHALLOW, linestyle=':')
    axes[1].plot(ds_cg_prec.theta_v.mean("time"),
                 ds_cg_prec.height,
                 label='congestus prec',
                 color=COLOR_CONGESTUS, linestyle=':')
    
    axes[1].legend(frameon=False, fontsize=11, loc='upper right')
    axes[0].set_ylabel("Height [m]", fontsize=20)
    axes[0].set_xlabel("q anomaly [g/kg]", fontsize=20)
    axes[0].set_yticks(np.arange(0, 2000, 500), minor=True)
    axes[0].set_ylim([0, 10000])
    axes[0].set_xlim([0, 16])
    axes[1].set_xlim([295, 330])
    axes[1].set_ylabel("Height [m]", fontsize=20)
    axes[1].set_xlabel("theta_v anomaly [K]", fontsize=20)
    axes[1].set_yticks(np.arange(0, 2000, 500), minor=True)
    axes[1].set_ylim([0, 10000])
    
    plt.savefig(
        os.path.join(
            path_paper_plots, "q_thetav_normal_height.png",
        ),
    )
    
def plot_q_theta_check(ds, path_paper_plots):
    
    print(ds)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    
    axs[0].annotate(
        "a) specific humidity",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )
    
    axs[1].annotate(
        "b) virtual potential temperature",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    ) 
    
    mesh_q = axs[0].pcolormesh(ds.time.values, ds.height, ds.q.T, cmap=CMAP, vmin=10, vmax=18)
    # add colorbar
    
    cbar = plt.colorbar(mesh_q, ax=axs[0], orientation='vertical')
    cbar.ax.tick_params(labelsize=25) 
    cbar.set_label('g/kg', fontsize=20)
    axs[0].set_ylabel("Height [m]", fontsize=20)
    axs[0].set_xlabel("Time", fontsize=20)
    
    mesh_theta = axs[1].pcolormesh(ds.time.values, ds.height, ds.theta_v.T, cmap=CMAP, vmin=295, vmax=301)
    # add colorbar
    cbar = plt.colorbar(mesh_theta, ax=axs[1], orientation='vertical')
    cbar.set_label('K', fontsize=20)
    axs[1].set_ylabel("Height [m]", fontsize=20)
    axs[1].set_xlabel("Time", fontsize=20)
    
    plt.savefig(
        os.path.join(
            path_paper_plots, "q_theta_v_check.png"
        ),
    )
    return(ds)

def split_dataset(ds):
    # Assuming ds is an xarray Dataset with exactly 4 variables, it returns variables splitted
    var1, var2, var3, var4 = ds.data_vars.keys()
    print(var1, var2, var3, var4)   
    
    ds1 = ds[[var1]]
    ds2 = ds[[var2]]
    ds3 = ds[[var3]]
    ds4 = ds[[var4]]
    
    return ds1, ds2, ds3, ds4


     
def prepare_anomaly_profiles(ds_cp, var_string, ds_lcl):
    """
    function to calculate mean and std of shallow and congestus profiles for prec and non prec conditions

    Args:
        ds_cp: xarray dataset containing cloud properties and rain flags
        var_string: string containing the variable to be analyzed
        ds_lcl: xarray dataset containing the diurnal cycle of the lifting condensation level
        
    Dependencies: 
    - read_anomalies function
    - calc_quantiles
    - calc_lcl_grid
    - read_lcl
    
    """
    # reading arthus data anomalies
    ds_an = read_anomalies(var_string)
    
    # check if time is a variable with capital letter in ds_an
    if 'Time' in ds_an:
        # rename variables to create a dataset with all variables together
        ds_an = ds_an.rename({'Height':'height', "Time":'time'})
    
    # interpolate classification of clouds over anomalies time stamps
    class_interp = ds_cp.interp(time=ds_an.time.values, method='nearest')                                                           
                                                                    
    # read lcl and calculate its diurnal cycle at 15 mins and at 30 mins (for fluxes)
    ds_lcl = read_lcl()
    
    ds_lcl_interp = ds_lcl.interp(time=ds_an.time.values, method='nearest')
    
    # align lcl dataset to the y dataset of the anomalies with flags
    ds_lcl, ds_an, ds_class = xr.align(ds_lcl_interp, ds_an, class_interp, join="inner")

    # regridding data to height grid referred to lcl
    ds_an_h_lcl = calc_lcl_grid_no_dc(ds_an, ds_lcl, 'height', 'time', 'anomaly')
    
    # selecting cloud types and rain/norain conditions
    is_shallow = ds_class.shape == 0
    is_congestus = ds_class.shape == 1

    # selecting prec and non prec
    is_prec_ground = ds_class.flag_rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 

    # segregating with respect to shallow and deep clouds
    ds_sl_prec = ds_an_h_lcl.isel(time=is_sl_prec)
    ds_sl_nonprec = ds_an_h_lcl.isel(time=is_sl_non_prec)
    ds_cg_prec = ds_an_h_lcl.isel(time=is_cg_prec)
    ds_cg_nonprec = ds_an_h_lcl.isel(time=is_cg_non_prec)
    
    ds_sl = ds_an_h_lcl.isel(time=is_shallow)
    ds_cg = ds_an_h_lcl.isel(time=is_congestus)
    
    return ds_sl, ds_cg, ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec
    
def calc_percentiles(ds, percentiles=[25, 50, 75]):
    """_summary_

    Args:
        ds (_type_): _description_
        percentiles (list, optional): _description_. Defaults to [0.25, 5, 0.75].
    """
    height = ds.height.values
    q = np.zeros((len(percentiles), len(height)))
    for i_h in range(len(height)):
        ds_h = ds.isel(height=i_h)
        an_s = ds_h.anomaly.values.flatten()
        q[:,i_h] = np.nanpercentile(an_s, percentiles)
        
    return(q)
    
def f_plot_datarray(da, x, y, var_name):
    
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(y, x, da[var_name].values.T, cmap=CMAP)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical')
    fig.savefig('/net/ostro/plots_rain_paper/test_'+var_name+'_all_profiles_lcl.png')    
    return(fig)






# Example function to plot and save figures
def plot_multipanel_figure_V2(lcl_dc, 
                           ds_stats,
                           ds_sl_prec, 
                           ds_sl_nonprec, 
                           ds_cg_prec,
                           ds_cg_nonprec, 
                           ds_q_sl_prec, 
                           ds_q_sl_nonprec, 
                           ds_q_cg_prec, 
                           ds_q_cg_nonprec,
                           ds_q_clear, 
                           ds_theta_v_sl_prec, 
                           ds_theta_v_sl_nonprec, 
                           ds_theta_v_cg_prec, 
                           ds_theta_v_cg_nonprec,
                           ds_theta_v_clear, 
                           ds_cb_sllcl,
                           ds_cb_cglcl,
                           ds_sl_prec_hw,
                           ds_sl_nonprec_hw,
                           ds_cg_prec_hw,
                           ds_cg_nonprec_hw,
                           data):
    
    
    # read relative occurrences of cloud types
    rel_occ_co_r_diurnal=ds_stats["rel_occ_co_r_diurnal"]
    rel_occ_co_nr_diurnal=ds_stats["rel_occ_co_nr_diurnal"]
    rel_occ_sh_diurnal=ds_stats["rel_occ_sh_diurnal"]
    rel_occ_co_diurnal=ds_stats["rel_occ_co_diurnal"]
    
    # convert rel_occ_co_diurnal.hour into a datetime object
    time_rel_occ = create_time(rel_occ_co_diurnal.hour.values, 3, 2025, '11')
    
    bins = np.arange(-0.5, 1.75,0.25)
    bin_width = bins[1] - bins[0]

    hist_shallow, bins_shallow = np.histogram(ds_cb_sllcl, bins=bins, density=True)
    hist_congestus, bins_congestus = np.histogram(ds_cb_cglcl, bins=bins, density=True)
    
    # Adjust bin coordinates
    bins_shallow_centered = bins_shallow[:-1] + bin_width / 2
    bins_congestus_centered = bins_congestus[:-1] + bin_width / 2
    
    
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
    
    # calculating 2d histograms to plot
    hist_ZESK_cloud_s, x_ze_cloud_s, y_sk_cloud_s = np.histogram2d(ze_s, 
                                                                   sk_s, 
                                                                   bins=[40, 40], 
                                                                   range=[[-50., 25.],
                                                                          [-1., 1.]], 
                                                                   density=True)
    hist_ZEVD_cloud_s, x_ze2_cloud_s, y_vd_cloud_s = np.histogram2d(ze_s, vd_s, 
                                                                    bins=[40, 40], 
                                                                    range=[[-50., 25.],
                                                                           [-4., 2.]], 
                                                                    density=True)

    
    
    # Create a figure
    fig = plt.figure(figsize=(22, 18))
    
    # set all fonts in the figure to 20
    plt.rcParams.update({'font.size': 18})
    
    # Create a GridSpec with 3 rows and 2 columns
    subplots = plt.subplots()
    gs = subplots[0].add_gridspec(2, 6, height_ratios=[3, 4]) #, width_ratios=[2, 2, 2, 2, 2, 2]
    #gs.subplots_adjust(hspace=0.3)
    # Add subplots with different sizes
    
    # cloud base height distributions
    ax1 = fig.add_subplot(gs[0, 0:2])  # Second row, first column, profiles of w
    ax1.plot(hist_shallow, bins_shallow_centered, color=COLOR_SHALLOW, linewidth=2)
    ax1.plot(hist_congestus, bins_congestus_centered, color=COLOR_CONGESTUS, linewidth=2)
    ax1.fill_betweenx(bins_shallow_centered, 0., hist_shallow, color=COLOR_SHALLOW, alpha=0.5)
    ax1.fill_betweenx(bins_congestus_centered, 0., hist_congestus, color=COLOR_CONGESTUS, alpha=0.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks(np.arange(-0.5, 1.5, 0.25), minor=True)
    ax1.set_yticklabels([-0.5, -0.25, "LCL", 0.25, 0.5, 1, 1.25, 1.5])
    ax1.set_ylabel("Height above\nLCL [km]")
    ax1.set_xlabel("Norm. Occ.")
    ax1.set_title(" a) CB distribution", loc='left', fontweight='black')
    
    # profiles of w
    ax2 = fig.add_subplot(gs[0, 2])  # Second row, first column, profiles of w
    plot_profiles(ax2, 
                  ds_sl_prec, 
                  ds_sl_nonprec, 
                  ds_cg_prec,
                  ds_cg_nonprec,
                  -0.4, 0.4, 'w anomaly [m/s]')
    ax2.set_title('b) W anomaly', loc='left', fontweight='black')

    ax2bis = fig.add_subplot(gs[0, 3])  # Second row, first column, profiles of w
    plot_profiles(ax2bis, 
                  ds_sl_prec_hw, 
                  ds_sl_nonprec_hw, 
                  ds_cg_prec_hw,
                  ds_cg_nonprec_hw,
                  -1, 1, 'H wind anomaly [m/s]')
    ax2bis.set_title('c) Hspeed \n anomaly', loc='left', fontweight='black')

    # profiles of q
    ax3 = fig.add_subplot(gs[0, 4], sharey=ax2)  # Second row, second column profiles of q
    plot_profiles(ax3, 
                  ds_q_sl_prec, 
                  ds_q_sl_nonprec, 
                  ds_q_cg_prec, 
                  ds_q_cg_nonprec, 
                  10, 16.5, 'q [g/kg]')
    ax3.plot(ds_q_clear.mean("time"),
             ds_q_clear.height* 1e-3,
             label='clear sky',
             color='black', 
             linewidth=3, 
             linestyle = 'dashdot')

    ax3.set_title('d) Spec. \n humidity', loc='left', fontweight='black')

 
    # profiles of theta_v
    ax4 = fig.add_subplot(gs[0, 5], sharey=ax2)  # Second row, third column profiles of thetav
    plot_profiles(ax4, 
                  ds_theta_v_sl_prec, 
                  ds_theta_v_sl_nonprec, 
                  ds_theta_v_cg_prec, 
                  ds_theta_v_cg_nonprec, 
                  300, 310, '$\Theta_v$[K]')
    ax4.plot(ds_theta_v_clear.mean("time"),
             ds_theta_v_clear.height* 1e-3,
             label='clear sky',
             color='black', 
             linewidth=3, 
             linestyle = 'dashdot')
    ax4.set_title('e) Virt. pot. \n temp.', loc='left', fontweight='black')

    # Generate the legend from ax4
    handles, labels = ax4.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='center', frameon=False, ncol=5, bbox_to_anchor=(0.5, 1.15))

    # scatter Ze vs Vd
    ax5 = fig.add_subplot(gs[1, 0:3])  # Third row, left plot
    ax5.set_title('f) Normalized Occurrences of Ze vs Vd', loc='left', fontweight='black')

    hist_d = ax5.hist2d(ze_c, 
                        vd_c, 
                        bins=(40, 40), 
                        range=([-50., 25.],[-4., 2.]), 
                        cmap=CMAP, 
                        density=True, 
                        cmin=0.0001)
    
    cbar = fig.colorbar(hist_d[3], ax=ax5, orientation='horizontal')
    cs = ax5.contour(x_ze2_cloud_s[:-1], 
                        y_vd_cloud_s[:-1], 
                        hist_ZEVD_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZEVD_cloud_s),
                                np.nanmax(hist_ZEVD_cloud_s), 
                                (np.nanmax(hist_ZEVD_cloud_s)- np.nanmin(hist_ZEVD_cloud_s))/10), 
                        cmap=plt.cm.Greys)
    ax5.clabel(cs, inline=True)
    cbar.set_label('norm. occ. congestus clouds')
    #cbar.ax.tick_params(labelsize=25)
    ax5.set_ylabel("Mean Doppler velocity [ms$^{-1}$] ")
    ax5.set_xlabel("Reflectivity [dBz] ")
    ax5.set_ylim(-4., 2.)
    ax5.set_xlim(-50.,25.)
    
    # scatter Ze vs Sk

    ax6 = fig.add_subplot(gs[1, 3:])  # Third row, right plot
    ax6.set_title('g) Normalized Occurrences of Ze vs Sk', loc='left', fontweight='black')

    hist_d = ax6.hist2d(ze_c, 
                        -sk_c, 
                        bins=(40, 40), 
                        range=([-50., 25.],[-1., 1.]), 
                        cmap=CMAP, 
                        density=True, 
                        cmin=0.0001)
    cbar = fig.colorbar(hist_d[3], ax=ax6, orientation='horizontal')
    cs = ax6.contour(x_ze_cloud_s[:-1], 
                        -y_sk_cloud_s[:-1], 
                        hist_ZESK_cloud_s.T, 
                        np.arange(np.nanmin(hist_ZESK_cloud_s), np.nanmax(hist_ZESK_cloud_s),
                                (np.nanmax(hist_ZESK_cloud_s)- np.nanmin(hist_ZESK_cloud_s))/10),
                        cmap=plt.cm.Greys)
    
    ax6.clabel(cs, inline=True)
    cbar.set_label('norm. occ. congestus clouds')
    #cbar.ax.tick_params(labelsize=25)
    ax6.set_ylabel("Skewness ")
    ax6.set_xlabel("Reflectivity [dBz] ")
    ax6.set_ylim(-1., 1.)
    ax6.set_xlim(-50.,25.)
    
    for ax in [ax1, ax2, ax2bis, ax3, ax4, ax5, ax6]:  # Loop over all axes    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        ax.tick_params(which='minor', length=3, width=2)
        ax.tick_params(which='major', length=5, width=2)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    # Adjust layout
    #gs.tight_layout(fig, h_pad=1.0, w_pad=1.0)

    fig.subplots_adjust(hspace=0.2, wspace=0.5)

    
    # Save the figure to a file
    fig.savefig('/net/ostro/plots_rain_paper/fig_04_multipanel_figure_new_V2.png')





if __name__ == "__main__":
    main()










