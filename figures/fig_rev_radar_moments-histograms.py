

import xarray as xr
from cloudtypes.path_folders import path_paper_plots


def main():
    
    
    # read radar data
    from readers.radar import read_radar_multiple, get_radar_files
        
    # read radar reflectivity
    files = get_radar_files()

    ds = xr.open_mfdataset(files)
    
    
    # plot histogram of mean Doppler velocity
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    mdv_values = ds.mean_doppler_velocity.values.flatten()
    mdv_values = mdv_values[~np.isnan(mdv_values)]
    
    sk_values = ds.skewness.values.flatten()
    sk_values = sk_values[~np.isnan(sk_values)]
        
    print('minimum value', np.nanmin(sk_values))
    print('maximum value', np.nanmax(sk_values))
    
    print('minimum value', np.nanmin(mdv_values))
    print('maximum value', np.nanmax(mdv_values))
    
    plt.figure(figsize=(10, 6))
    # make 2 subplots
    plt.subplot(2, 1, 1)
    plt.hist(sk_values, bins=50, color='red', alpha=0.7, density=True)
    plt.title('Histogram of Skewness')
    plt.xlabel('Skewness')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.subplot(2, 1, 2)
    # plot histogram of mean Doppler velocity
    plt.hist(mdv_values, bins=50, color='blue', alpha=0.7, density=True)
    plt.title('Histogram of Mean Doppler Velocity')
    plt.xlabel('Mean Doppler Velocity (m/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xlim(-20, 20)
    plt.savefig(path_paper_plots+'/mean_doppler_velocity_skeewness_histogram.png', dpi=300)
    
if __name__ == "__main__":
    main()