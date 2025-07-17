

import xarray as xr

def read_surf_anomaly(variable, start_date, end_date):
    """
    Read surface anomaly data for a specific variable.
    Args:
        variable (str): The variable for which to read the surface anomaly data.
    Returns:
        xarray.dataset: The surface anomaly dataset for the specified variable.
    """
    file_path = f'/work/4sibylle/surface_diurnal_cycles/{variable}_surface_anomaly.nc'
    ds = xr.open_dataset(file_path)
    
    # select dates between start_date and end_date
    ds_sel = ds.sel(time=slice(start_date, end_date))
    
    # rename variable time to Time
    ds_sel = ds_sel.rename({'time': 'Time'})

    return ds_sel

