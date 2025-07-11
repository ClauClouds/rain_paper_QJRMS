"""
histograms of virga depths and rain rate for shallow and congestus clouds

"""

from readers.radar import read_radar_multiple, get_radar_files
from readers.cloudtypes import read_rain_flags, read_cloud_class
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import pdb
from fig_rev_mean_values_table import man_withney_u_test


def main():
    
    # reading wband radar data to get rain rate from surface weather station in the radar
    files = get_radar_files()

    wband_data = xr.open_mfdataset(files)
    #wband_data = read_radar_multiple()  # uncomment this line if you want to use the function directly 

        
    #wband_data = read_radar_multiple()
    
    # read cloud classification
    all_flags = read_rain_flags()
    
    # read shape from classification original file
    cloud_properties = read_cloud_class()  
    cloud_properties['flag_rain_ground'] = (('time'), all_flags.flag_rain_ground.values)
    cloud_properties['flag_rain'] = (('time'), all_flags.flag_rain.values)
    cloud_properties['virga_depth'] = (('time'), all_flags.virga_depth.values)

    # merging data together to create a single dataset
    merian_dataset = xr.merge([wband_data, cloud_properties])

    # selecting virga data (rain not reaching the ground) and rain data (reaching the ground)
    ind_rain = np.where(merian_dataset.flag_rain_ground.values == 1)[0]
    rain = merian_dataset.isel(time=ind_rain)
    
    ind_virga = np.where((merian_dataset.flag_rain_ground.values == 0) * (merian_dataset.flag_rain.values == 1))[0]
    virga = merian_dataset.isel(time=ind_virga)
    
    # selecting shallow and congestus datasets
    ind_shallow_v = np.where((virga.shape.values == 0))[0]
    ind_deep_v = np.where((virga.shape.values == 1))[0]
    virga_shallow = virga.isel(time=ind_shallow_v)
    virga_deep = virga.isel(time=ind_deep_v)
    
    ind_shallow_r = np.where((rain.shape.values == 0))[0]
    ind_deep_r = np.where((rain.shape.values == 1))[0]
    rain_shallow = rain.isel(time=ind_shallow_r)
    rain_deep = rain.isel(time=ind_deep_r)    
    
    rain_shallow = rain_shallow.where(rain_shallow.rain_rate > 0)
    rain_deep = rain_deep.where(rain_deep.rain_rate > 0)
    
    # call the function to do the Mann-Whitney U test
    alpha=0.05
    man_withney_u_test(rain_shallow, rain_deep, "rain rate", alpha)

    # calculating t-test for the distributions of rain rate for shallow and congestus cases
    
    
    # print mean rain rate in shallow and in deep cases
    # remove rain rates values equal to 0
    rain_shallow = rain_shallow.where(rain_shallow.rain_rate > 0)
    rain_deep = rain_deep.where(rain_deep.rain_rate > 0)
    
    mean_rr_shallow = np.nanmean(rain_shallow.rain_rate.values) 
    mean_rr_deep = np.nanmean(rain_deep.rain_rate.values)
    print(f"Mean rain rate shallow: {mean_rr_shallow:.2f} mm/h")
    print(f"Mean rain rate deep: {mean_rr_deep:.2f} mm/h")
    
    # print std and percentiles of rain rates
    std_rr_shallow = np.nanstd(rain_shallow.rain_rate.values)
    std_rr_deep = np.nanstd(rain_deep.rain_rate.values)
    percentiles_rr_shallow = np.nanpercentile(rain_shallow.rain_rate.values, [0, 10, 50, 90, 100])
    percentiles_rr_deep = np.nanpercentile(rain_deep.rain_rate.values, [0, 10, 50, 90, 100])
    print(f"Std rain rate shallow: {std_rr_shallow:.2f} mm/h")
    print(f"Std rain rate deep: {std_rr_deep:.2f} mm/h")
    print(f"Percentiles rain rate shallow: {percentiles_rr_shallow}")
    print(f"Percentiles rain rate deep: {percentiles_rr_deep}")
    

    #set all zero values to nan so it does not appear in the histogram
    #virga depth
    shallow_vd_data = zero_to_nan(virga_shallow.virga_depth.values)
    deep_vd_data = zero_to_nan(virga_deep.virga_depth.values)

    #rain_rate
    shallow_rr_data = zero_to_nan(rain_shallow.rain_rate.values)
    deep_rr_data = zero_to_nan(rain_deep.rain_rate.values)

    # calculating histograms
    #virga depth
    hist_vd_s, bins_vd_s = np.histogram(shallow_vd_data, bins=9, range=(0., 1000.), density=True)
    hist_vd_d, bins_vd_d = np.histogram(deep_vd_data, bins=9, range=(0., 1000.), density=True)
    hist_vd_s = hist_vd_s / hist_vd_s.sum()
    hist_vd_d = hist_vd_d / hist_vd_d.sum()
    
    #rain rate
    hist_rr_s, bins_rr_s = np.histogram(shallow_rr_data, bins=9, range=(0., 35.), density=True)
    hist_rr_d, bins_rr_d = np.histogram(deep_rr_data, bins=9, range=(0., 35.), density=True)
    hist_rr_s = hist_rr_s / hist_rr_s.sum()
    hist_rr_d = hist_rr_d / hist_rr_d.sum()
    
    visualize_histograms(hist_vd_s, hist_vd_d, hist_rr_s, hist_rr_d, bins_vd_s, bins_vd_d, bins_rr_s, bins_rr_d)
    

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def visualize_histograms(hist_vd_s, hist_vd_d, hist_rr_s, hist_rr_d, bins_vd_s, bins_vd_d, bins_rr_s, bins_rr_d):
    """_summary_

    Args:
        hist_vd_s (_type_): _description_
        hist_vd_d (_type_): _description_
        hist_rr_s (_type_): _description_
        hist_rr_d (_type_): _description_
    """
    
    shallow_color = '#ff9500'
    congestus_color = '#008080'
        
    fig, axs = plt.subplots(1,2, figsize=(20,8))

    width_rr = 2.7
    axs[0].bar(bins_rr_s[:-1], 
               hist_rr_s, 
               width=width_rr, 
               color=shallow_color, 
               label='shallow')
    axs[0].bar([i-0.025*width_rr for i in bins_rr_d[:-1]], 
               hist_rr_d, 
               align='center', 
               width=0.5*width_rr, 
               color=congestus_color,
               alpha=1., 
               label='congestus')
    axs[0].set_xlabel('Rain rate [mm h$^{-1}$ ]', fontsize=20)

    width_vd = 80
    axs[1].bar(bins_vd_s[:-1],
               hist_vd_s,
               width=width_vd, 
               color=shallow_color, 
               label='shallow')
    axs[1].bar([i-0.025*width_vd for i in bins_vd_d[:-1]], 
               hist_vd_d, align='center',
               width=0.5*width_vd, 
               color=congestus_color, 
               alpha=1., 
               label='congestus')
    axs[1].set_xlabel('Virga depth [m]', fontsize=20)

    for ax, l in zip(axs[:].flatten(), ['a) Rain rate ',  'b) Virga depth ']):
        ax.text(-0.05, 1.07, l,  fontweight='black', fontsize=25, transform=ax.transAxes)
    for i in range(0,2):
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['bottom'].set_linewidth(3)
        axs[i].spines['left'].set_linewidth(3)
        #axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        axs[i].tick_params(which='minor', length=5, width=3, labelsize=20)
        axs[i].tick_params(which='major', length=7, width=3, labelsize=20)
        axs[i].get_xaxis().tick_bottom()
        axs[i].get_yaxis().tick_left()
        axs[i].set_ylabel('Normalized occurrences', fontsize=20)
        axs[i].legend(frameon = False, fontsize=20)

    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure06_histograms_rain_virga.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

if __name__ == "__main__":
    main()
    