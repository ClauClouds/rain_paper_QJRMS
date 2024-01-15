"""
Created on Jan 11 2024
plot Ze sk Vd for shallow and congestus clouds based on previous code with new classification file
- 
@author: cacquist
"""
#%%
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

#%%
# path for output plot
path_out = '/work/plots_rain_paper/'

data = xr.open_dataset(path_out+'ze_sk_vd_above_cloud_base.nc')

# plot distribution of radar moments for shallow and congestus clouds
ze_s = data.ze_shallow.values.flatten()
ze_c = data.ze_congestus.values.flatten()
vd_s = data.vd_shallow.values.flatten()
vd_c = data.vd_congestus.values.flatten()
sk_s = data.sk_shallow.values.flatten()
sk_c = data.sk_congestus.values.flatten()

i_good_s = ~np.isnan(ze_s) * ~np.isnan(vd_s) * ~np.isnan(sk_s)

ze_s = ze_s[i_good_s]
vd_s = vd_s[i_good_s]
sk_s = sk_s[i_good_s]

print(len(ze_s), len(vd_s), len(sk_s))
i_good_c = ~np.isnan(ze_c) * ~np.isnan(vd_c) * ~np.isnan(sk_c)
ze_c = ze_c[i_good_c]
vd_c = vd_c[i_good_c]
sk_c = sk_c[i_good_c]

print(len(ze_c), len(vd_c), len(sk_c))

#%%



# plot histograms of single variables to check the distributions are ok
def plot_hist_mom(mom_s, mom_c, mom_name, mom_string):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
    
    dict_plot_settings = {
        'labelsizeaxes':16,
        'fontSizeTitle':16,
        'fontSizeX'    :16,
        'fontSizeY'    :16,
        'cbarAspect'   :16,
        'fontSizeCbar' :16,
        'rcparams_font':['Tahoma'],
        'savefig_dpi'  :100,
        'font_size'    :16, 
        'grid'         :True}

    # plots settings defined by user at the top
    labelsizeaxes   = dict_plot_settings['labelsizeaxes']
    fontSizeTitle   = dict_plot_settings['fontSizeTitle']
    fontSizeX       = dict_plot_settings['fontSizeX']
    fontSizeY       = dict_plot_settings['fontSizeY']
    cbarAspect      = dict_plot_settings['cbarAspect']
    fontSizeCbar    = dict_plot_settings['fontSizeCbar']
    rcParams['font.sans-serif'] = dict_plot_settings['rcparams_font']
    matplotlib.rcParams['savefig.dpi'] = dict_plot_settings['savefig_dpi']
    plt.rcParams.update({'font.size':dict_plot_settings['font_size']})
    grid = dict_plot_settings['grid']
    alpha_value = 0.2

    plt.gcf().subplots_adjust(bottom=0.1)

    ax = plt.subplot(2,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5)) 
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3) 
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.hist(mom_s, bins=50, label='shallow clouds', density=True, color='#ff9500')
    ax.set_xlabel(mom_string, fontsize=16)
    ax.set_ylabel("Occurrences [] ", fontsize=16)
    ax.set_title('Shallow clouds', fontsize=16)

    ax = plt.subplot(2,1,2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5)) 
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3) 
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.hist(mom_c, bins=50, label='congestus clouds', density=True, color='#008080')
    ax.set_xlabel("Ze [dBz] ", fontsize=16)
    ax.set_ylabel("Occurrences [] ", fontsize=16)
    #), bins=(40, 40), range=([0., 1000.],[0., 900.]), cmap=plt.cm.viridis, density=True, cmin=0.0001)
    ax.set_xlabel(mom_string, fontsize=16)
    ax.set_ylabel("Occurrences [] ", fontsize=16)

    ax.set_title('Congestus clouds', fontsize=16)
    #plt.ylim(-1., 1.)
    #plt.colorbar()
    #plt.legend(frameon=False, loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig('/work/plots_rain_paper/'+mom_name+'_shallow_congestus.png', format='png')
    return()

plot_ze = plot_hist_mom(ze_s, ze_c, 'ze', "Ze [dBz]")
plot_vd = plot_hist_mom(vd_s, vd_c, 'vd', "Vd [ms-1]")
plot_sk = plot_hist_mom(sk_s, sk_c, 'sk', "Sk []")

#%%

# calculating 2d histograms to plot
hist_ZESK_cloud_s, x_ze_cloud_s, y_sk_cloud_s = np.histogram2d(ze_s, sk_s, bins=[40, 40], range=[[-50., 25.],[-1., 1.]], density=True)
hist_ZEVD_cloud_s, x_ze2_cloud_s, y_vd_cloud_s = np.histogram2d(ze_s, vd_s, bins=[40, 40], range=[[-50., 25.],[-4., 2.]], density=True)


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,25))
plt.gcf().subplots_adjust(bottom=0.1)
dict_plot_settings = {
        'labelsizeaxes':32,
        'fontSizeTitle':32,
        'fontSizeX'    :32,
        'fontSizeY'    :32,
        'cbarAspect'   :32,
        'fontSizeCbar' :32,
        'rcparams_font':['Tahoma'],
        'savefig_dpi'  :100,
        'font_size'    :32, 
        'grid'         :True}
labelsizeaxes   = dict_plot_settings['labelsizeaxes']
fontSizeTitle   = dict_plot_settings['fontSizeTitle']
fontSizeX       = dict_plot_settings['fontSizeX']
fontSizeY       = dict_plot_settings['fontSizeY']
cbarAspect      = dict_plot_settings['cbarAspect']
fontSizeCbar    = dict_plot_settings['fontSizeCbar']
rcParams['font.sans-serif'] = dict_plot_settings['rcparams_font']
matplotlib.rcParams['savefig.dpi'] = dict_plot_settings['savefig_dpi']
plt.rcParams.update({'font.size':dict_plot_settings['font_size']})
grid = dict_plot_settings['grid']
matplotlib.rc('xtick', labelsize=32)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=32) # sets dimension of ticks in the plots

axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
axs[0].tick_params(which='minor', length=5, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
hist_d = axs[0].hist2d(ze_c, -sk_c, bins=(40, 40), range=([-50., 25.],[-1., 1.]), cmap=plt.cm.Greys, density=True, cmin=0.0001)
cbar = fig.colorbar(hist_d[3], ax=axs[0])
cs = axs[0].contour(x_ze_cloud_s[:-1], -y_sk_cloud_s[:-1], hist_ZESK_cloud_s.T, np.arange(np.nanmin(hist_ZESK_cloud_s), np.nanmax(hist_ZESK_cloud_s), (np.nanmax(hist_ZESK_cloud_s)- np.nanmin(hist_ZESK_cloud_s))/10))
axs[0].clabel(cs, inline=True, fontsize=14)
cbar.set_label('norm. occ. congestus clouds')
cbar.ax.tick_params(labelsize=14)
#cbar2 = plt.colorbar(cs,ax=axs[0])#,location='bottom',shrink=0.75)
#cbar2.set_label('normalized occurrences shallow clouds', fontsize=14)
#cbar2.ax.tick_params(labelsize=14)
axs[0].set_ylabel("Skewness ")
#axs[0].set_xlabel("Reflectivity [dBz] ")
#axs[0].set_title('shallow clouds', fontsize=16)
axs[0].set_ylim(-1., 1.)
axs[0].set_xlim(-50.,25.)
#plt.colorbar(hist_s)
#plt.colorbar(hist_d)
#axs[0].colorbar(hist_s, orientation='vertical')

#ax = plt.subplot(2,2,2)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
axs[1].tick_params(which='minor', length=5, width=2)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
hist_d = axs[1].hist2d(ze_c, vd_c, bins=(40, 40), range=([-50., 25.],[-4., 2.]), cmap=plt.cm.Greys, density=True, cmin=0.0001)
cbar = fig.colorbar(hist_d[3], ax=axs[1])
cs = axs[1].contour(x_ze2_cloud_s[:-1], y_vd_cloud_s[:-1], hist_ZEVD_cloud_s.T, np.arange(np.nanmin(hist_ZEVD_cloud_s), np.nanmax(hist_ZEVD_cloud_s), (np.nanmax(hist_ZEVD_cloud_s)- np.nanmin(hist_ZEVD_cloud_s))/10))
axs[1].clabel(cs, inline=True, fontsize=14)
cbar.set_label('norm. occ. congestus clouds', fontsize=fontSizeX)
cbar.ax.tick_params(labelsize=14)
#hist_s = axs[1].hist2d(ZE_s.flatten(), VD_s.flatten(), bins=(40, 40), range=([-50., 20.],[-5., 5.]), cmap=plt.cm.Blues, density=True, cmin=0.0001)
#cs = axs[1].contour(x_ze2_s[:-1], y_vd_s[:-1], hist_ZEVD, np.arange(np.nanmin(hist_ZEVD),np.nanmax(hist_ZEVD), (np.nanmax(hist_ZEVD)- np.nanmin(hist_ZEVD))/10))
#axs[1].clabel(cs, inline=True, fontsize=10)

axs[1].set_ylabel("Mean Doppler velocity [ms$^{-1}$] ")
axs[1].set_xlabel("Reflectivity [dBz] ")
axs[1].set_ylim(-4., 2.)
axs[1].set_xlim(-50.,25.)
#plt.colorbar(hist_s)
#plt.colorbar(hist_d)

figure_name = path_out+'fig_radar_moments_in_cloud.png'
plt.savefig(figure_name,bbox_inches='tight')




#%%