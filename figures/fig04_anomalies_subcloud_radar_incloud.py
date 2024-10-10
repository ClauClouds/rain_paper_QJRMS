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
from readers.ship import read_ship_pressure
from readers.lidars import read_anomalies
from readers.cloudtypes import read_cloud_class, read_rain_ground
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_paper_plots
from readers.lcl import read_lcl, read_diurnal_cycle_lcl
from fig03_diurnal_all import calc_lcl_grid, calc_diurnal_lcl
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
#from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
from dask.diagnostics import ProgressBar
ProgressBar().register()

def main():
    
    # first row of plot:  lcl diurnal cycle for the plot
    lcl_dc = read_diurnal_cycle_lcl(path_diurnal_cycle_arthus)
    
    # read cloud types (shallow and congestus)
    ds_ct = read_cloud_class()

    # read rain at the ground flag 
    ds_r = read_rain_ground()
    
    # align cloud type flag and rain ground flag
    ds_ct, ds_r = xr.align(ds_ct, ds_r, join="inner")
    ds_cp = xr.merge([ds_ct, ds_r])
    
    # second row of plots: derive specific humidity and virtual potential temperature from lidar data
    ds_therm = calc_thermo_prop()
    ds_therm = xr.align(ds_therm, ds_ct, ds_cp, join="inner")
    
    # regrid heights with respect to lcl
    ds_therm_lcl = calc_lcl_grid_no_dc(ds_therm, lcl_ds, ds_class, height_var, time_var, var_name)
    #plot_q_theta_check(ds_therm, path_paper_plots)

    # align cloud type flags and rain ground flag
    ds_therm_sl, ds_therm_cg, ds_therm_sl_prec, ds_therm_sl_nonprec, ds_therm_cg_prec, ds_therm_cg_nonprec = assign_flags_to_profiles(ds_therm, ds_cp, ds_r)

    
    # second row of plots: read lidar data
    
    # prepare data (calculate mean and std of the profiles for shallow/congestus in prec and non prec)
    ds_sl, ds_cg, ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec = prepare_anomaly_profiles(ds_cp, "VW", lcl_dc)
    
    plot_w_subfigure(ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, path_paper_plots, 'w_subcloud_lcl')
    
    plot_w_subfigure(ds_therm_sl_prec, ds_therm_sl_nonprec, ds_therm_cg_prec, ds_therm_cg_nonprec, path_paper_plots, 'therm_subcloud_lcl')

    
    
    #plot_figure(lcl_dc, dct_vw_q)
def assign_flags_to_profiles(ds_therm, ds_cp, ds_r):
    
    # align cloud type flags and rain ground flag
    ds_therm, ds_cp, ds_r = xr.align(ds_therm, ds_cp, ds_r, join="inner")
    
    # selecting cloud types and rain/norain conditions
    is_shallow = ds_cp.shape == 0
    is_congestus = ds_cp.shape == 1

    # selecting prec and non prec
    is_prec_ground = ds_cp.flag_rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 

    # segregating with respect to shallow and deep clouds
    ds_sl_prec = ds_therm.isel(time=is_sl_prec)
    ds_sl_nonprec = ds_therm.isel(time=is_sl_non_prec)
    ds_cg_prec = ds_therm.isel(time=is_cg_prec)
    ds_cg_nonprec = ds_therm.isel(time=is_cg_non_prec)
    
    ds_sl = ds_therm.isel(time=is_shallow)
    ds_cg = ds_therm.isel(time=is_congestus)
    
    return ds_sl, ds_cg, ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec


def calc_thermo_prop():
    """
    function to calculate thermodynamic properties from lidar data
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
    ship_data = read_ship_pressure()
    ship_arthus = ship_data.interp(time=ds_arthus.time.values, method='nearest')
    P_surf = ship_arthus.P.values

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
    T = ds_arthus.T.values
    for indHeight in range(dim_h):
        k = Rd*(1-0.23*mr[:, indHeight])/Cp
        Theta_v[:,indHeight] = ( (1 + 0.61 * mr[:, indHeight]) * T[:, indHeight] * (P_surf/P[:,indHeight])**k)
    

    # store q, theta and theta_v in dataset
    ds_out = xr.Dataset(
        {
            "q": (["time", "height"], q),
            "theta_v": (["time", "height"], Theta_v),
        },
        coords={"time": ds_arthus.time, "height": ds_arthus.height},
    )
    
    # add units attributtes to q, theta and theta_v
    ds_out.q.attrs["units"] = "g/kg"
    ds_out.theta_v.attrs["units"] = "K"
    
    print(ds_out)
    return(ds_out)



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




        
def plot_w_subfigure(ds_sl_prec, ds_sl_nonprec, ds_cg_prec, ds_cg_nonprec, path_paper_plots, fig_name):
     
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    axes[0].annotate(
        "a) vertical wind anomaly",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )
    axes[0].plot(ds_sl_nonprec.mean("time"),
                 ds_sl_nonprec.height* 1e-3,
                 label='shallow non prec',
                 color=COLOR_SHALLOW)
    axes[0].plot(ds_cg_nonprec.mean("time"),
                 ds_cg_nonprec.height* 1e-3,
                 label='congestus non prec',
                color=COLOR_CONGESTUS)
    axes[0].plot(ds_sl_prec.mean("time"), 
                 ds_sl_prec.height* 1e-3,
                 label='shallow prec',
                 color=COLOR_SHALLOW, linestyle=':')
    axes[0].plot(ds_cg_prec.mean("time"),
                 ds_cg_prec.height* 1e-3,
                 label='congestus prec',
                 color=COLOR_CONGESTUS, linestyle=':')
    
    axes[0].legend(frameon=False, fontsize=11, loc='lower left')
    axes[0].set_ylabel("Height above\nLCL [km]", fontsize=20)
    axes[0].set_xlabel("w anomaly [m/s]", fontsize=20)
    axes[0].set_yticks(np.arange(-1, 1, 0.5), minor=True)
    axes[0].set_yticklabels([-1, -0.5, "LCL", 0.5, 1, 1.5], fontsize=20)
    axes[0].set_ylim(
        [
            ds_cg_prec.height.isel(
                height=ds_cg_prec.any("time").values.argmax()
            )
            * 1e-3,
            0.3,
        ]
    )
    
    plt.savefig(
        os.path.join(
            path_paper_plots, fig_name+".png"
        ),
    )
     
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

    # interpolate classification of clouds over anomalies time stamps
    class_interp = ds_cp.interp(time=ds_an.Time.values, method='nearest')                                                           
                                                                    
    # read lcl and calculate its diurnal cycle at 15 mins and at 30 mins (for fluxes)
    ds_lcl = read_lcl()
    
    ds_lcl_interp = ds_lcl.interp(time=ds_an.Time.values, method='nearest')
    
    # align lcl dataset to the y dataset of the anomalies with flags
    ds_lcl, ds_an, ds_class = xr.align(ds_lcl_interp, ds_an, class_interp, join="inner")

    # regridding data to height grid referred to lcl
    ds_an_h_lcl = calc_lcl_grid_no_dc(ds_an, ds_lcl, ds_class, 'Height', 'Time', 'anomaly')
    
    
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
    


def plot_figure(lcl_dc, dct_vw_q):
    
    
    
    fig, axs = plt.subplots(3, 3, figsize=(25,25), sharey=True, constrained_layout=True)# 
    


def calc_lcl_grid_no_dc(ds, lcl_ds, ds_class, height_var, time_var, var_name):
    """
    function to convert and reinterpolate data on the height referred to the lcl height

    Args:
        ds (xarray dataset): dataset containing the data to be regridded
        lcl_ds (_type_): _description_
        height_var (_type_): _description_
        time_var (_type_): _description_
        var_name (_type_): _description_
    """
    dz = 7.45
    if (height_var != 'height') and (time_var != 'time'):
        # rename time and height in standard way for processing
        ds = ds.rename({height_var:'height', time_var:'time'})

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
        height=z_rel, method="nearest", kwargs={"fill_value": 0}
    )
    print('shape of da_var', np.shape(da_var))
    
    # calculate shift of all height values at each time step
    # positive shift means each height bin is shifted downward
    rows, columns = np.ogrid[: da_var.shape[0], : da_var.shape[1]]  # 2d indices
    shift = ((da_lcl + dz / 2) // dz).values.astype("int16") # numnber of bins to shift
    print('Shape of rows:', rows.shape)
    print('Shape of columns:', columns.shape)
    print('Shape of shift:', shift.shape)
    print(shift[0:30])
    print(columns[0,0:10])
    
    # reindexing columns with the shift
    columns = columns + shift[:, np.newaxis]
    print(columns[0,0:10])
    print(columns[6,0:10])

    # set limit to upper bound
    columns[columns >= columns.shape[1]] = columns.shape[1] - 1  # upper bound
    print('shape of columns', np.shape(columns))
    print('shape of da_var.values', np.shape(da_var.values))


    # reading values corresponding to the new indeces
    da_var[:] = da_var.values[rows, columns]
    
    return(da_var)



if __name__ == "__main__":
    main()


