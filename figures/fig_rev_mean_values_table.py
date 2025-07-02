
""" 
calculate mean variables for shallow and congestus clouds and do T test """
import xarray as xr
from cloudtypes.path_folders import path_paper_plots

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr



def main():
    
        
    # read radar data
    from readers.radar import read_radar_multiple, get_radar_files

    
    ds = xr.open_dataset('/data/obs/campaigns/eurec4a/msm/cloud_class_prop_flags.nc')
        
    t_test_cb_ct(ds)

def t_test_cb_ct(ds):
    
    # select shallow clouds
    shallow = ds.shape.where(ds.shape == 0, drop=True)
    # select congestus clouds
    congestus = ds.shape.where(ds.shape == 1, drop=True)
    
    
    cb_shallow = ds.cloud_base.where(ds.shape == 0, drop=True)
    cb_congestus = ds.cloud_base.where(ds.shape == 1, drop=True)
    ct_shallow = ds.cloud_top_height.where(ds.shape == 0, drop=True)
    ct_congestus = ds.cloud_top_height.where(ds.shape == 1, drop=True)
    
    # plot cb and ct histograms
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.hist(ct_congestus.values.flatten(), bins=50, alpha=0.5, label='congestus ct', color='blue')
    plt.hist(cb_congestus.values.flatten(), bins=50, alpha=0.5, label='Congestus cb', color='red')
    plt.title('Cloud Base Height Histogram')
    plt.xlabel('Cloud Base Height (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.hist(ct_shallow.values.flatten(), bins=50, alpha=0.5, label='Shallow ct', color='blue')
    plt.hist(cb_shallow.values.flatten(), bins=50, alpha=0.5, label='shallow cb', color='red')

    plt.title('Cloud Top Height Histogram')
    plt.xlabel('Cloud Top Height (m)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path_paper_plots, 'cloud_base_top_height_histogram.png'), dpi=300)
    
    # calculate t-student for cloud base and cloud top
    from scipy.stats import ttest_ind
    
    # cloud base
    t_stat, p_value = ttest_ind(
        cb_shallow.values.flatten(),
        cb_congestus.values.flatten(),
        equal_var=False,
    )
    print(f"p-value for cloud base height: {p_value:.20f}")
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(f"The difference in cloud top height is statistically significant (p-value: {p_value:.20f})")
    else:
        print(f"The difference in cloud top height is not statistically significant (p-value: {p_value:.8f})")
    
    # cloud top
    t_stat, p_value = ttest_ind(
        ct_shallow.values.flatten(),
        ct_congestus.values.flatten(),
        equal_var=False,
    )
    print(f"p-value for cloud top height: {p_value:.20f}")
    
    if p_value < alpha:
        print(f"The difference in cloud base height is statistically significant (p-value: {p_value:.20f})")
    else:
        print(f"The difference in cloud base height is not statistically significant (p-value: {p_value:.8f})")
    print(f"Mean cloud base height shallow: {np.nanmean(cb_shallow):.2f} m")  
    print(f"Mean cloud base height congestus: {np.nanmean(cb_congestus):.2f} m")
    print(f"Mean cloud top height shallow: {np.nanmean(ct_shallow):.2f} m")
    print(f"Mean cloud top height congestus: {np.nanmean(ct_congestus):.2f} m")
    
if __name__ == "__main__":
    main()
    