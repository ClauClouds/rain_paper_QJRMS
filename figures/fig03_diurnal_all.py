"""
code to create a figure containing diurnal cycles of:
- diurnal cycle of shallow and congestus (prec/non prec?)
- temperature
- mixing ratio
- horizontal wind speed
- fluxes LHF/SHF

"""
    
from cloudtypes.cloudtypes import classify_region, cloud_mask_lcl_grid
from readers.cloudtypes import read_cloud_class, read_rain_ground
from readers.lidars import read_all_lidar_diurnal_cycle_files, read_h_wind, read_fluxes
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_paper_plots
from readers.lcl import read_lcl
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

    
def main():
    """
    Create diurnal cycle of hydrometeor fraction
    """
    import pandas as pd
    ds = prepare_data()

    dct_stats = statistics(ds)

    
    # read and regrid lidar data with respect to lcl    
    da_T_lcl, da_MR_lcl, da_HW_lcl, da_LF_lcl, da_SF_lcl = read_and_regrid_lidar_wrt_lcl(path_diurnal_cycle_arthus)
  
    
    # call plotting script to produce plot of the publication
    plot_diurnal(dct_stats, da_T_lcl, da_MR_lcl, da_HW_lcl, da_LF_lcl, da_SF_lcl)





def calc_diurnal_lcl(ds, path_out, avg_time='15'):
    '''
    function to calculate diurnal cycle for lcl data

    '''
    # calculating the mean of the variable over the time interval requested
    ds = ds.resample(time=avg_time+'T').mean()
    # re-writing time array as hh:mm for then being able to group
    ds['time'] = pd.to_datetime(ds.time.values).strftime("%H:%M")
    # grouping and calculating mean of the profiles
    grouped_mean = ds.groupby('time').mean()
    
    
    # storing diurnal cycle in ncdf file
    dims             = ['time']
    coords           = {"time":pd.to_datetime(grouped_mean['time'].values)}
    lcl_diurnal  = xr.DataArray(dims=dims, coords=coords, data=grouped_mean['lcl'].values,\
                         attrs={'long_name':'diurnal cycle over '+avg_time+'min for '+'lcl',\
                                'units':'m'})
    
    global_attributes = {'CREATED_BY'       : 'Claudia Acquistapace',
                        'CREATED_ON'       :  str(datetime.now()),
                        'FILL_VALUE'       :  'NaN',
                        'AUTHOR_NAME'          : 'Claudia Acquistapace',
                        'AUTHOR_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'AUTHOR_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'AUTHOR_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'diurnal cycle of the surface weather station variables calculated on '+avg_time+'minutes',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - weather station on wband radar',
                        'DATA_GROUP'       : 'Experimental;Moving',
                        'DATA_SOURCE'      : 'wband radar data',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/rain_paper_QJRSM/',
                        'INSTRUMENT_MODEL' : '',
                         'COMMENT'         : 'original data postprocessed by Claudia Acquistapace' }
    ds_lcl_dc    = xr.Dataset(data_vars = {'lcl_dc':lcl_diurnal},
                                       coords = coords,
                                       attrs = global_attributes)
    # storing data to ncdf
    ds_lcl_dc.to_netcdf(path_out+'_lcl_diurnal_cycle.nc',
                   encoding={'lcl_dc':{"zlib":True, "complevel":9},\
                    "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
    
    return ds_lcl_dc

def read_and_regrid_lidar_wrt_lcl(path_to_dc_files):
    """_summary_

    Args:
        path_to_dc_files (_type_): _description_
        
    Dependencies: 
    - read_all_lidar_diurnal_cycle_files
    - read_lcl
    - calc_diurnal_lcl
    """
    
    # read datasets of diurnal cycle
    lidar_ds = read_all_lidar_diurnal_cycle_files(path_to_dc_files)
    fluxes_ds = read_fluxes(path_to_dc_files)
    hw_ds = read_h_wind(path_to_dc_files)
        
    # read lcl and calculate its diurnal cycle at 15 mins and at 30 mins (for fluxes)
    ds_lcl = read_lcl()
    ds_lcl_diurnal_15 = calc_diurnal_lcl(ds_lcl, path_diurnal_cycle_arthus, '15')
    ds_lcl_diurnal_30 = calc_diurnal_lcl(ds_lcl, path_diurnal_cycle_arthus, '30')

    # convert to height grid referred to lcl all variables
    T_lcl = calc_lcl_grid(lidar_ds, ds_lcl_diurnal_15, 'Height', 'Time', 'T')
    MR_lcl = calc_lcl_grid(lidar_ds, ds_lcl_diurnal_15, 'Height', 'Time', 'MR')
    HW_lcl = calc_lcl_grid(hw_ds, ds_lcl_diurnal_15, 'height_h', 'time_h', 'HW')
    LF_lcl = calc_lcl_grid(fluxes_ds, ds_lcl_diurnal_30, 'height_coarse', 'time_coarse', 'LHF')
    SF_lcl = calc_lcl_grid(fluxes_ds, ds_lcl_diurnal_30, 'height_coarse', 'time_coarse', 'SHF')

    return(T_lcl, MR_lcl, HW_lcl, LF_lcl, SF_lcl)

def calc_lcl_grid(ds, lcl_ds, height_var, time_var, var_name):
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
    print(height_var, time_var)
    if (height_var != 'height') and (time_var != 'time'):
        # rename time and height in standard way for processing
        ds = ds.rename({height_var:'height', time_var:'time'})

    # adding lcl to the dataset variables (also diurnal cycle)
    ds['lcl_dc'] = lcl_ds.lcl_dc.values
    
    # reading dataarray of the input variable and of lcl
    da_var = ds[var_name]
    da_lcl = ds.lcl_dc

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

    
    # calculate shift of all height values at each time step
    # positive shift means each height bin is shifted downward
    rows, columns = np.ogrid[: da_var.shape[0], : da_var.shape[1]]  # 2d indices
    print('shape of columns', np.shape(columns))
    print('shape of rows', np.shape(rows))
    shift = ((da_lcl + dz / 2) // dz).values.astype("int16")
    columns = columns + shift[:, np.newaxis]
    columns[columns >= columns.shape[1]] = columns.shape[1] - 1  # upper bound
    da_var[:] = da_var.values[rows, columns]

    return(da_var)


def statistics(ds):
    """
    Hydrometeor fraction statistics

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the cloud mask with height bins relative to LCL,
        cloud types, and region classification
    """

    dct_stats = {}

    is_shallow = ds.shape == 0
    is_congestus = ds.shape == 1

    # diurnal cycle of shallow and congestus hydrometeor fraction
    dct_stats["hf_sh_diurnal"] = (
        ds.cloud_mask.sel(time=is_shallow).groupby("time.hour").mean()
    )
    dct_stats["hf_co_diurnal"] = (
        ds.cloud_mask.sel(time=is_congestus).groupby("time.hour").mean()
    )

    # diurnal cycle of shallow and congestus occurrence
    dct_stats["occ_sh_diurnal"] = (
        ds.shape.sel(time=is_shallow).groupby("time.hour").count()
    )
    dct_stats["occ_co_diurnal"] = (
        ds.shape.sel(time=is_congestus).groupby("time.hour").count()
    )

    # counts of shallow/stratiform for entire campaign
    dct_stats["occ_sh"] = dct_stats["occ_sh_diurnal"].sum("hour")
    dct_stats["occ_co"] = dct_stats["occ_co_diurnal"].sum("hour")

    # relative occurrence
    dct_stats["rel_occ_sh_diurnal"] = (
        dct_stats["occ_sh_diurnal"] / dct_stats["occ_sh"]
    )
    dct_stats["rel_occ_co_diurnal"] = (
        dct_stats["occ_co_diurnal"] / dct_stats["occ_co"]
    )
    
    # adding stats for congestus prec and non prec
    is_prec_ground = ds.flag_rain_ground == 1
    is_cong_prec = is_congestus & is_prec_ground
    is_cong_non_prec = is_congestus & ~is_prec_ground

    # diurnal cycle of congestus prec/non prec hydrometeor fraction
    dct_stats["hf_co_prec_diurnal"] = (
        ds.cloud_mask.sel(time=is_cong_prec).groupby("time.hour").mean()
    )
    dct_stats["hf_co_non_prec_diurnal"] = (
        ds.cloud_mask.sel(time=is_cong_non_prec).groupby("time.hour").mean()
    )


    # diurnal cycle of shallow and congestus occurrence
    dct_stats["occ_con_r_diurnal"] = (
        ds.shape.sel(time=is_cong_prec).groupby("time.hour").count()
    )
    dct_stats["occ_con_nr_diurnal"] = (
        ds.shape.sel(time=is_cong_non_prec).groupby("time.hour").count()
    )

    # counts of congeestus rain and no rain surf for entire campaign
    dct_stats["occ_co_r"] = dct_stats["occ_con_r_diurnal"].sum("hour")
    dct_stats["occ_co_nr"] = dct_stats["occ_con_nr_diurnal"].sum("hour")

    # relative occurrence
    dct_stats["rel_occ_co_r_diurnal"] = (
        dct_stats["occ_con_r_diurnal"] / dct_stats["occ_co"]
    )
    dct_stats["rel_occ_co_nr_diurnal"] = (
        dct_stats["occ_con_nr_diurnal"] / dct_stats["occ_co"]
    )
    return dct_stats

    
def prepare_data():
    
    """
    Prepare data needed for the figure
    """
    from cloudtypes.path_folders import path_diurnal_cycle_arthus

    # calculate cloud mask with respect to lcl
    da_cm_lcl = cloud_mask_lcl_grid()
    
    # read cloud types
    ds_ct = read_cloud_class()

    # read rain at the ground flag
    ds_r = read_rain_ground()
    
    
    # classify by region
    da_pos_class = classify_region()

    # align the gps-based and radar-based data
    da_cm_lcl, ds_ct, da_pos_class, ds_r = xr.align(
        da_cm_lcl, ds_ct, da_pos_class, ds_r, join="inner"
    )

    ds = xr.merge([ds_ct, da_pos_class, da_cm_lcl, ds_r])

    return ds




def plot_diurnal(dct_stats, da_T, da_MR, da_HW, da_LF, da_SF):
    """
    Plot diurnal cycle
    """
    from datetime import datetime
    import pandas as pd
    import matplotlib.dates as mdates

    hf_sh_diurnal=dct_stats["hf_sh_diurnal"]
    hf_co_diurnal=dct_stats["hf_co_diurnal"]
    rel_occ_sh_diurnal=dct_stats["rel_occ_sh_diurnal"]
    rel_occ_co_diurnal=dct_stats["rel_occ_co_diurnal"]
    occ_sh=dct_stats["occ_sh"]
    occ_co=dct_stats["occ_co"]
    hf_co_r_diurnal=dct_stats["hf_co_prec_diurnal"]
    hf_co_nr_diurnal=dct_stats["hf_co_non_prec_diurnal"]
    rel_occ_co_r_diurnal=dct_stats["rel_occ_co_r_diurnal"]
    rel_occ_co_nr_diurnal=dct_stats["rel_occ_co_nr_diurnal"]
    occ_co_r=dct_stats["occ_co_r"]
    occ_co_nr=dct_stats["occ_co_nr"]
    #time_MR = pd.to_datetime(da_MR.time.values).hour
    #time_T = time_MR
    #time_HW =pd.to_datetime(da_HW.time.values).hour
    #time_LF =pd.to_datetime(da_LF.time.values).hour
    #time_SF = time_LF    
    #h_MR
    # defining masks for each data from lidar
    mask_T = np.ma.masked_where(da_T.values > 0. ,da_T.values)
    mask_MR = np.ma.masked_where(da_MR.values > 0. ,da_MR.values)
    mask_HW = np.ma.masked_where(da_HW.values > 0. ,da_HW.values)
    mask_LF = np.ma.masked_where(da_LF.values > 0. ,da_LF.values)
    mask_SF = np.ma.masked_where(da_SF.values > 0. ,da_SF.values)

    norm = mcolors.BoundaryNorm(np.arange(0, 0.81, 0.05), CMAP.N)

    fig, axes = plt.subplots(5, 2, figsize=(25, 20), constrained_layout=True)

    axes[0,0].annotate(
        "a) Shallow hydrometeor fraction (#{:,})".format(occ_sh.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    axes[1,0].annotate(
        "b) Congestus hydrometeor fraction (#{:,})".format(occ_co.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    
    axes[2,0].annotate(
        "c) Congestus rain (#{:,})".format(occ_co_r.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    
    axes[3,0].annotate(
        "d) Congestus non rain (#{:,})".format(occ_co_nr.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )

    axes[4,0].annotate(
        "e) Relative occurrence",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )

    kwargs = dict(shading="nearest", cmap=CMAP, norm=norm)
    im = axes[0,0].pcolormesh(
        hf_sh_diurnal.hour,
        hf_sh_diurnal.height * 1e-3,
        hf_sh_diurnal.T,
        **kwargs
    )
    im = axes[1,0].pcolormesh(
        hf_co_diurnal.hour,
        hf_co_diurnal.height * 1e-3,
        hf_co_diurnal.T,
        **kwargs
    )
    
    im = axes[2,0].pcolormesh(
        hf_co_r_diurnal.hour,
        hf_co_r_diurnal.height * 1e-3,
        hf_co_r_diurnal.T,
        **kwargs
    )
    
    im = axes[3,0].pcolormesh(
        hf_co_nr_diurnal.hour,
        hf_co_nr_diurnal.height * 1e-3,
        hf_co_nr_diurnal.T,
        **kwargs
    )
    
    axes[4,0].plot(
        rel_occ_sh_diurnal.hour,
        rel_occ_sh_diurnal,
        color=COLOR_SHALLOW,
        label="Shallow",
        linewidth=3
    )
    axes[4,0].plot(
        rel_occ_co_diurnal.hour,
        rel_occ_co_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus",
        linewidth=3
    )
    
    axes[4,0].plot(
        rel_occ_co_r_diurnal.hour,
        rel_occ_co_r_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus rain",
        linestyle='--',
        linewidth=3
    )
    
    axes[4,0].plot(
        rel_occ_co_nr_diurnal.hour,
        rel_occ_co_nr_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus non rain",
        linestyle=':', 
        linewidth=3,
    )
    
    axes[4,0].legend(loc="upper right", frameon=False, fontsize=15)
    
    #leg.set_in_layout(False)

    for ax in axes[1:4,0]:
        ax.set_ylabel("Height above\nLCL [km]", fontsize=25)
        ax.set_yticks(np.arange(-0.5, 4.5, 0.25), minor=True)
        ax.set_yticklabels([-1, -0.5, "LCL", 0.5, 1., 1.5], fontsize=20)

    for ax in axes[:,0].flatten():
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticks(np.arange(0, 24, 1), minor=True)
        ax.set_xticklabels(np.arange(0, 24, 2), fontsize=20)
        ax.set_xlim([-0.5, 23.5])
    for ax in axes[:,1].flatten():
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticks(np.arange(0, 24, 1), minor=True)
        ax.set_xticklabels(np.arange(0, 24, 2), fontsize=20)
        ax.set_xlim([0, 24])

    # shallow axis
    axes[0,0].set_yticks(np.arange(-1, 1, 0.5))
    axes[0,0].set_ylabel("Height above\nLCL [km]", fontsize=25)

    axes[0,0].set_yticklabels([-1, -0.5, "LCL", 0.5], fontsize=20)
    axes[0,0].set_ylim(
        [
            hf_sh_diurnal.height.isel(
                height=hf_sh_diurnal.any("hour").values.argmax()
            )
            * 1e-3,
            0.6,
        ]
    )

    # congestus axis
    for ax in axes[1:-1,0]:
        ax.set_yticks(np.arange(-1, 5, 1))
        ax.set_yticklabels([-1, "LCL", 1, 2, 3, 4], fontsize=20)
        ax.set_ylim(
            [
                hf_sh_diurnal.height.isel(
                    height=hf_co_diurnal.any("hour").values.argmax()
                )
                * 1e-3,
                4,
            ]
        )

    axes[4,0].set_ylim([0, 0.07])
    axes[4,0].set_ylabel("Density [h$^{-1}$]", fontsize=25)
    axes[4,0].set_xlabel("Hour [LT, UTC-4]", fontsize=25)
    axes[4,1].set_xlabel("Hour [LT, UTC-4]", fontsize=25)


    axes[0,1].annotate(
        "f) Temperature ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    axes[1,1].annotate(
        "g) Water vapor mixing ratio ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    
    axes[2,1].annotate(
        "h) Horizontal wind speed ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    
    axes[3,1].annotate(
        "i) Sensible heat flux",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )

    axes[4,1].annotate(
        "j) Latent heat flux",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
        fontsize=25,
        fontweight='black'
    )
    
    cbar_hf = fig.colorbar(
        im,
        ax=axes[:4,0],
        ticks=np.arange(0, 0.8, 0.1),
    )
    cbar_hf.set_label("Hydrometeor fraction",fontsize=20)
    cbar_hf.ax.tick_params(labelsize=20)  

    kwargs_lidar = dict(shading="nearest", cmap=CMAP)
    im_T = axes[0,1].pcolormesh(
        pd.to_datetime(da_T.time.values).hour,
        da_T.height.values* 1e-3,
        da_T.T,
        vmin=296,
        vmax=300,
        **kwargs_lidar
    )
    # mesh over nan areas
    c_T = axes[0,1].contourf(pd.to_datetime(da_T.time.values).hour,
        da_T.height.values* 1e-3,
        mask_T.T, 
        hatches='//',
        cmap='gray', extend='both', alpha=1)
    
    
    im_MR = axes[1,1].pcolormesh(
        pd.to_datetime(da_MR.time.values).hour,
        da_MR.height.values* 1e-3,
        da_MR.T,
        vmin=12,
        vmax=18,
        **kwargs_lidar
    )
    # mesh over nan areas
    c_MR = axes[1,1].contourf(pd.to_datetime(da_MR.time.values).hour,
        da_MR.height.values* 1e-3,
        mask_MR.T, 
        hatches='//',
        cmap='gray', extend='both', alpha=1)
       
    im_HW = axes[2,1].pcolormesh(
        pd.to_datetime(da_HW.time.values).hour,
        da_HW.height.values* 1e-3,
        da_HW.T,
        vmin=7,
        vmax=10,
        **kwargs_lidar
    )
    # mesh over nan areas
    c_HW = axes[2,1].contourf(pd.to_datetime(da_HW.time.values).hour,
        da_HW.height.values* 1e-3,
        mask_HW.T, 
        hatches='//',
        cmap='gray', extend='both', alpha=1)
    
    im_SF = axes[3,1].pcolormesh(
        pd.to_datetime(da_SF.time.values).hour,
        da_SF.height.values* 1e-3,
        da_SF.T,
        vmin=-10,
        vmax=20,
        **kwargs_lidar
    )
    # mesh over nan areas
    c_SF = axes[3,1].contourf(pd.to_datetime(da_SF.time.values).hour,
        da_SF.height.values* 1e-3,
        mask_SF.T, 
        hatches='//',
        cmap='Greys', extend='both', alpha=0.6)    
    
    
    im_LF = axes[4,1].pcolormesh(
        pd.to_datetime(da_LF.time.values).hour,
        da_LF.height.values* 1e-3,
        da_LF.T,
        vmin=-10,
        vmax=100,
        **kwargs_lidar,
    )
    # mesh over nan areas
    c_LF = axes[4,1].contourf(pd.to_datetime(da_LF.time.values).hour,
        da_LF.height.values* 1e-3,
        mask_LF.T, 
        hatches='//',
        cmap='Greys', extend='both', alpha=0.6)    
    
        
    # T/MR axis
    axes[0,1].set_yticks(np.arange(-1, 1, 0.5))
    axes[0,1].set_ylabel("Height above\nLCL [km]", fontsize=25)
    axes[0,1].set_yticklabels([-1, -0.5, "LCL", 0.5], fontsize=20)
    axes[0,1].set_ylim(-0.5, 0.2)
    #    [
    ##        da_T.height.isel(
    #            height=da_T.any("time").values.argmax()
    ##        )
    #        * 1e-3,
    #        0.6,
    #    ]
    #)

    axes[1,1].set_ylabel("Height above\nLCL [km]", fontsize=25)
    axes[1,1].set_yticks(np.arange(-1, 1, 0.5))
    axes[1,1].set_yticklabels([-1, -0.5, "LCL", 0.5], fontsize=20)
    axes[1,1].set_ylim(-0.5, 0.2)
    #    [
    #        da_MR.height.isel(
    #            height=da_MR.any("time").values.argmax()
    #        )
    #        * 1e-3,
    #        0.6,
    #    ]
    #)
         
    # HW axis
    axes[2,1].set_ylabel("Height above\nLCL [km]", fontsize=25)
    axes[2,1].set_yticks(np.arange(-1, 1, 0.5), minor=True)
    axes[2,1].set_yticklabels([-1, -0.5, "LCL", 0.5, 1, 1.5], fontsize=20)
    axes[2,1].set_ylim(
        [
            da_HW.height.isel(
                height=da_HW.any("time").values.argmax()
            )
            * 1e-3,
            0.3,
        ]
    )
       
    # LHF/SHF axis
    axes[3,1].set_ylabel("Height above\nLCL [km]", fontsize=25)
    axes[3,1].set_yticks(np.arange(-1, 1, 0.5))
    axes[3,1].set_ylim(-0.5, 0.2)
    axes[3,1].set_yticklabels([-1, -0.5, "LCL", 0.5], fontsize=20)

    axes[4,1].set_ylabel("Height above\nLCL [km]", fontsize=25)
    axes[4,1].set_yticks(np.arange(-1, 1, 0.5))
    axes[4,1].set_yticklabels([-1, -0.5, "LCL", 0.5], fontsize=20)
    axes[4,1].set_ylim(-0.5, 0.2)
    
    # set now all colorbars for lidar plots
    
    cbar = fig.colorbar(
        im_T,
        ax=axes[0,1],
        ticks=np.arange(296, 300, 1),
    )
    cbar.set_label('[K]', fontsize=20)
    cbar.ax.tick_params(labelsize=20)  
    
    cbar = fig.colorbar(
        im_MR,
        ax=axes[1,1],
        label="[gkg$^{-1}$]",
        ticks=np.arange(12, 18, 1),
    )
    cbar.set_label("[Kgm$^{-2}$]", fontsize=20)
    cbar.ax.tick_params(labelsize=20)  

    cbar = fig.colorbar(
        im_HW,
        ax=axes[2,1],
        ticks=np.arange(7, 10, 0.5),
    )
    cbar.set_label("[m$^{-1}$]",fontsize=20)
    cbar.ax.tick_params(labelsize=20)  

    cbar = fig.colorbar(
        im_SF,
        ax=axes[3,1],
        ticks=np.arange(-10, 20, 5),
    )
    cbar.set_label("[Wm$^{-2}$]", fontsize=20)
    cbar.ax.tick_params(labelsize=20)  
     
    cbar = fig.colorbar(
        im_LF,
        ax=axes[4,1],
        ticks=np.arange(-10, 100, 10),
    )   
    cbar.set_label("[Wm$^{-2}$]", fontsize=20)
    cbar.ax.tick_params(labelsize=20)  
    
    for ax in (axes[:,:].flatten()):
        ax.grid(False)

    plt.savefig(
        os.path.join(
            path_paper_plots, "diurnal.png"
        ),
    )
    
        
        
if __name__ == "__main__":
    main()
