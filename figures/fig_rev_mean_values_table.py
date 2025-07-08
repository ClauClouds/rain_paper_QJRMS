
""" 
calculate mean variables for shallow and congestus clouds and do T test """
import xarray as xr
from cloudtypes.path_folders import path_paper_plots
from readers.cloudtypes import read_cloud_class, read_rain_flags
from readers.cloudtypes import read_cloud_class, read_rain_flags
from readers.radar import read_lwp
import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from readers.radar import read_radar_multiple, get_radar_files


def main():
    
        
    # read radar data
    ds = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/cloud_class_prop_flags.nc')

    # read rain flags
    rainflags = read_rain_flags()
        
    # read cloud classification
    cloudclassdata = read_cloud_class()

    # merge datasets
    cloudclassdata = xr.merge([cloudclassdata, rainflags], compat="override")
    
    # flag for cloudy data
    cloudyflag = ((cloudclassdata.flag_rain_ground.values == 0) & (cloudclassdata.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain

    shallow = (cloudclassdata.shape.values == 0) 
    congestus = (cloudclassdata.shape.values == 1) 

    # find number of shallow precipitation time stamps
    shallow_cloudy = shallow & cloudyflag
    congestus_cloudy = congestus & cloudyflag
    
    ct_shallow = cloudclassdata.cloud_top_height.values[shallow_cloudy]
    ct_congestus = cloudclassdata.cloud_top_height.values[congestus_cloudy]
    cb_shallow = cloudclassdata.cloud_base.values[shallow_cloudy]
    cb_congestus = cloudclassdata.cloud_base.values[congestus_cloudy]

    # read LWP data
    merian = read_lwp() 

    # read rain flags
    rainflags = read_rain_flags()
    cloudyflag = ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain

  # interpolate classification on LWP time stamps and derive flags for shallow/congestus data
    cloudclassdataip = cloudclassdata.interp(time=merian.time)
    shallow = (cloudclassdataip.shape.values == 0) & (cloudyflag == 1)
    mean_lwp_shallow = np.nanmean(merian.lwp[shallow & cloudyflag])

    # print mean LWP for shallow clouds
    print(f"Mean LWP shallow: {mean_lwp_shallow:.2f} g/m^2")
    shallow_cloud_thickness = np.nanmean(ct_shallow) - np.nanmean(cb_shallow) # in m
    print('Mean shallow cloud thickness:', shallow_cloud_thickness, 'm')
    gamma_ad = 2e-3 # adiabatic lapse rate in gm-4
    LWP_ad = 0.5 * gamma_ad * shallow_cloud_thickness**2 # in gm-2
    print(f"Mean LWP adiabatic: {LWP_ad:.2f} g/m^2")
    # print adiabaticiity
    print(f"Adiabaticity shallow: {mean_lwp_shallow/LWP_ad:.2f}")    
    ct_shallow = ct_shallow[ct_shallow > 0]
    ct_congestus = ct_congestus[ct_congestus > 0]
    cb_shallow = cb_shallow[cb_shallow > 0]
    cb_congestus = cb_congestus[cb_congestus > 0]
    
    # print mean CT shallow
    print(f"Mean cloud top height shallow: {np.nanmean(ct_shallow):.2f} m")
    # print mean CB shallow
    print(f"Mean cloud base height shallow: {np.nanmean(cb_shallow):.2f} m")
    
    
    
    # calculate percentiles of distributions
    ct_shallow_percentiles = np.nanpercentile(ct_shallow, [0, 10, 50, 90, 100])
    ct_congestus_percentiles = np.nanpercentile(ct_congestus, [0, 10, 50, 90, 100])
    
    # print percentiles
    print("Cloud Top Height Percentiles (Shallow):", ct_shallow_percentiles)
    print("Cloud Top Height Percentiles (Congestus):", ct_congestus_percentiles)
    print("Cloud Base Height Percentiles (Shallow):", np.nanpercentile(cb_shallow, [0, 10, 50, 90, 100]))
    print("Cloud Base Height Percentiles (Congestus):", np.nanpercentile(cb_congestus, [0, 10, 50, 90, 100]))

    # print standard deviation
    print("Cloud Top Height Std (Shallow):", np.nanstd(ct_shallow))
    print("Cloud Top Height Std (Congestus):", np.nanstd(ct_congestus))
    print("Cloud Base Height Std (Shallow):", np.nanstd(cb_shallow))
    print("Cloud Base Height Std (Congestus):", np.nanstd(cb_congestus))
    
    # plot cb and ct histograms
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.hist(ct_congestus, bins=50, alpha=0.5, label='congestus ct', color='blue', density=True)
    plt.hist(cb_congestus, bins=50, alpha=0.5, label='Congestus cb', color='red', density=True)
    plt.title('Cloud properties - congestus clouds')
    plt.xlabel('Height (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(0, 5000)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.hist(ct_shallow, bins=100, alpha=0.5, label='Shallow ct', color='blue', density=True)
    plt.hist(cb_shallow, bins=50, alpha=0.5, label='shallow cb', color='red', density=True)
    plt.title('Cloud properties - shallow clouds')
    plt.xlabel('Height (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(0, 5000)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path_paper_plots, 'cloud_base_top_height_histogram.png'), dpi=300)    
    
    print(f"Mean cloud base height shallow: {np.nanmean(cb_shallow):.2f} m")  
    print(f"Mean cloud base height congestus: {np.nanmean(cb_congestus):.2f} m")
    print(f"Mean cloud top height shallow: {np.nanmean(ct_shallow):.2f} m")
    print(f"Mean cloud top height congestus: {np.nanmean(ct_congestus):.2f} m")
    
    # run mann-whitney U test for cloud base and cloud top
    alpha=0.05
    man_withney_u_test(ct_congestus, ct_shallow, "cloud top height", alpha)
    man_withney_u_test(cb_congestus, cb_shallow, "cloud base height", alpha)

    
def man_withney_u_test(sample1, sample2, var_name, alpha=0.05, ):
    """
    Perform Mann-Whitney U test on two samples and print the results.
    Parameters:
    sample1 (array-like): First sample data.
    sample2 (array-like): Second sample data.
    alpha (float): Significance level for the test.
    var_name (str): Name of the variable being tested.
    """
    
    # try mann withney u test
    from scipy.stats import mannwhitneyu
    u_stat, p_value = mannwhitneyu(
        sample1,
        sample2,
        alternative='two-sided',
    )    
    
    
    if p_value < alpha:
        # print result
        print(f"The difference in {var_name} is statistically significant (p-value: {p_value:.20f})")        
    else:
        # print result
        print(f"The difference in {var_name} is not statistically significant (p-value: {p_value:.8f})")
        
    
    
if __name__ == "__main__":
    main()
    