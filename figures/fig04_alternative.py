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

import os
import sys

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

    # add lidar data 
    lidar_ds = read_all_lidar_diurnal_cycle_files(path_diurnal_cycle_arthus)
    fluxes_ds = read_fluxes(path_diurnal_cycle_arthus)
    hw_ds = read_h_wind(path_diurnal_cycle_arthus)
    
    # calculate heights with respect to lcl
     = calc_lcl_grid()

    
    
    plot_diurnal(dct_stats, ds, lidar_ds, fluxes_ds, hw_ds)

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


def calc_lcl_grid(data):
    """
    funcction to regrid heights of the input dataset as a function of
    distance from lcl

    Args:
        data (xarray dataset): input data
    """
    from readers.lcl import read_lcl

    # lifting condensation level
    ds_lcl = read_lcl()
    
    # align both time series
    data, ds_lcl = xr.align(data, ds_lcl)
    assert len(ds_lcl.time) == len(data.time)

    
    # interpolate data on regular height grid of 7.45 m that covers the
    # height difference to the lcl
    dz = 7.45
    z_rel = np.arange(
        data.height.min() - data.max(),
        data.height.max() - data.min() + dz,
        dz,
    )
    z_rel = z_rel - z_rel[z_rel > 0].min()  # center around zero
    data = data.interp(
        height=z_rel, method="nearest", kwargs={"fill_value": 0}
    )
    
    
    # calculate shift of all height values at each time step
    # positive shift means each height bin is shifted downward
    rows, columns = np.ogrid[: data.shape[0], : data.shape[1]]  # 2d indices
    shift = ((ds_lcl + dz / 2) // dz).values.astype("int16")
    columns = columns + shift[:, np.newaxis]
    columns[columns >= columns.shape[1]] = columns.shape[1] - 1  # upper bound
    data[:] = data.values[rows, columns]
    return(data)
    
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




def plot_diurnal(dct_stats, ds, lidar_ds, fluxes_ds, hw_ds):
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
    temp = lidar_ds['T']
    MR = lidar_ds['MR']
    HW = hw_ds['HW']
    SHF = fluxes_ds['SHF']
    LHF = fluxes_ds['LHF']
    time_lidar = pd.to_datetime(lidar_ds.Time.values).hour
    time_coarse_lidar = pd.to_datetime(fluxes_ds.time_coarse.values).hour
    time_h =pd.to_datetime(hw_ds.time_h.values).hour
    
    norm = mcolors.BoundaryNorm(np.arange(0, 0.81, 0.05), CMAP.N)

    fig, axes = plt.subplots(5, 2, figsize=(25, 20), constrained_layout=True)

    axes[0,0].annotate(
        "a) Shallow (#{:,})".format(occ_sh.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    axes[1,0].annotate(
        "b) Congestus (#{:,})".format(occ_co.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    
    axes[2,0].annotate(
        "c) Congestus rain (#{:,})".format(occ_co_r.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    
    axes[3,0].annotate(
        "d) Congestus non rain (#{:,})".format(occ_co_nr.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )

    axes[4,0].annotate(
        "e) Relative occurrence",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
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
    leg = axes[2,0].legend(loc="lower left", ncol=4)
    leg.set_in_layout(False)

    for ax in axes[1:4,0]:
        ax.set_ylabel("Height above\nLCL [km]")
        ax.set_yticks(np.arange(-0.5, 4.5, 0.25), minor=True)

    for ax in axes[:,0]:
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticks(np.arange(0, 24, 1), minor=True)
        ax.set_xticklabels(np.arange(0, 24, 2))
        ax.set_xlim([-0.5, 23.5])

    # shallow axis
    axes[0,0].set_yticks(np.arange(-1, 1, 0.5))
    axes[0,0].set_yticklabels([-1, -0.5, "LCL", 0.5])
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
        ax.set_yticklabels([-1, "LCL", 1, 2, 3, 4])
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
    axes[4,0].set_ylabel("Density [h$^{-1}$]")
    axes[4,0].set_xlabel("Hour [LT, UTC-4]")


    axes[0,1].annotate(
        "f) Temperature ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    axes[1,1].annotate(
        "g) Water vapor mixing ratio ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    
    axes[2,1].annotate(
        "h) Horizontal wind speed ",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    
    axes[3,1].annotate(
        "i) Sensible heat flux",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )

    axes[4,1].annotate(
        "f) Latent heat flux",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    
    fig.colorbar(
        im,
        ax=axes[:4,0],
        label="Hydrometeor fraction",
        ticks=np.arange(0, 0.7, 0.1),
    )

    kwargs_lidar = dict(shading="nearest", cmap=CMAP)
    im = axes[0,1].pcolormesh(
        time_lidar, 
        lidar_ds.Height.values,
        temp.T,
        vmin=290,
        vmax=300,
        **kwargs_lidar
    )
    im = axes[1,1].pcolormesh(
        time_lidar, 
        lidar_ds.Height.values,
        MR.T,
        vmin=10,
        vmax=18,
        **kwargs_lidar
    )
    
    im = axes[2,1].pcolormesh(
        time_h, 
        hw_ds.height_h.values,
        HW.T,
        vmin=6,
        vmax=10,
        **kwargs_lidar
    )
    
    im = axes[3,1].pcolormesh(
        time_coarse_lidar,
        fluxes_ds.height_coarse.values,        
        SHF.T,
        vmin=-20,
        vmax=20,
        **kwargs_lidar
    )
    
    axes[4,1].pcolormesh(
        time_coarse_lidar,
        fluxes_ds.height_coarse.values,  
        LHF.T, 
        vmin=30,
        vmax=100,
        **kwargs_lidar,
    )


    for ax in axes[:,1]:
        ax.set_ylabel("Height above\nLCL [km]")
        #ax.set_yticks(np.arange(-0.5, 4.5, 0.25), minor=True)
        ax.set_ylim(0., 1200)
        #ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        
    for ax in (axes[:,:].flatten()):
        ax.grid(False)

    plt.savefig(
        os.path.join(
            path_paper_plots, "diurnal.png"
        ),
    )
if __name__ == "__main__":
    main()
