"""
Code to generate boxplot profiles for the revised paper figures.
This script will create boxplots for the specified variables and save them as images.

"""
import matplotlib.pyplot as plt
from readers.lidars import read_anomalies_and_rename
import xarray as xr
import numpy as np
import os

from figures.mpl_style import COLOR_CONGESTUS, COLOR_SHALLOW



def main():
    
    # read anomalies of the variable
    ds_anomalies = read_anomalies_and_rename('T')
    var_name = 'Temperature'
    units = 'K'

    # call plot function
    plot_profiles(ds_anomalies, var_name, units)
    # plot percentiles for reviewer figure
    #plot_percentiles_reviewer_fig(ds_anomalies, var_name, units)
    
def plot_profiles(ds, var_name, units):
    '''
    function to plot percentiles of variables as a function of height
    '''
    if var_name == 'Temperature'
        variable = 'T_anomaly'
    elif var_name == 'Relative humidity':
        variable = 'RH_anomaly'
    elif var_name == 'Specific humidity':
                  
        
    data = ds[variable].values
    height = ds.height.values
    
    print('data shape', data.shape)
    
    median = np.nanmedian(data, axis=0)
    std = np.nanstd(data, axis=1)
    perc_25 = np.nanpercentile(data, 25, axis=0)
    perc_75 = np.nanpercentile(data, 75, axis=0) 
    perc_10 = np.nanpercentile(data, 10, axis=0)
    perc_90 = np.nanpercentile(data, 90, axis=0) 
    print('perc 25', perc_25)
    print('perc 75', perc_75 )   
        
    plt.figure(figsize=(6, 8))
    plt.plot(median, height, label='Median', color=COLOR_SHALLOW, linewidth=4)
    plt.plot(perc_25, height, color=COLOR_SHALLOW, linestyle='--', label='25th perc', linewidth=2)
    plt.plot(perc_75, height, color=COLOR_SHALLOW, linestyle='--', label='75th perc', linewidth=2)
    plt.plot(perc_10, height, color=COLOR_SHALLOW, linestyle=':', label='10th perc', linewidth=2)
    plt.plot(perc_90, height, color=COLOR_SHALLOW, linestyle=':', label='90th perc', linewidth=2)
    plt.fill_betweenx(height, perc_25, perc_75, color=COLOR_SHALLOW, alpha=0.3, label='±25th percentile')
    plt.fill_betweenx(height, perc_10, perc_90, color=COLOR_SHALLOW, alpha=0.1, label='±40th percentile')
    #plot dashed lines for median + and - 25th percentile

    plt.xlabel(var_name+' ['+units+']')
    
    plt.ylabel('Height [m]')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/net/ostro/plots_rain_paper/fig_04_percentiles_profiles_'+var_name+'_reviewer.png', dpi=300)

    return()

def plot_percentiles_reviewer_fig(ds, var_name, units):
    """
    function to calculate percentiles of the input variable at selected heights of 500, 650, and 1000 m
    and plot the percentiles in a figure, along the vertical profile, as wisker plots.
    

    Args:
        ds_sl_prec (dataset): shallow prec dataset
        ds_sl_nonprec (dataset): shallow non prec dataset
        ds_cg_prec (dataset): congestus prec dataset
        ds_cg_nonprec (dataset): congestus non prec dataset
        var_name (str): name of the variable to plot percentiles for
        
    """

    import numpy as np
    
    # resample the dataset to 200 m height intervals for boxplotting
    # Create height bins every 200m
    heights = ds.height.values
    height_bins = np.arange(heights.min(), heights.max() + 200, 200)
    
    # calculate array of height of center of the bins
    height_bins_center = (height_bins[:-1] + height_bins[1:]) / 2
    print('height center bins lenght', len(height_bins_center))
    
    # resampling array on height bins
    #ds_resampled = ds.groupby_bins('height', height_bins).mean()
    ds_resampled = ds.groupby_bins('height', height_bins).reduce(np.nanmean, dim='height')
    print('ds_resampled', ds_resampled)
    print(len(height_bins))
    data = ds_resampled['T_anomaly'].values
    print('data shape', data.shape)
    
    print('stats of t anomaly')
    print(np.nanmax(data), np.nanmin(data), np.nanmean(data))
    
    # plot boxplot with positions height
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.boxplot(data.T, vert=False, positions=height_bins_center, 
               widths=150, patch_artist=True, notch=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               whiskerprops=dict(color='blue'),
               capprops=dict(color='blue'),
               medianprops=dict(color='red'))
    ax.set_xlabel('Variable (e.g., concentration)') 
    ax.set_ylabel('Height (m)')
    ax.set_title('Vertical Profile: Boxplots by Height')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('/net/ostro/plots_rain_paper/fig_04_percentiles_'+var_name+'_reviewer.png', dpi=300)
    print('saving figure to /net/ostro/plots_rain_paper/fig_04_percentiles_'+var_name+'_reviewer.png')
    return

if __name__ == "__main__":
    main()  
    