"""
Diurnal cycle of hydrometeor fraction.
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW

# add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

from cloudtypes.cloudtypes import classify_region, cloud_mask_lcl_grid
from readers.cloudtypes import read_cloudtypes

load_dotenv()


def main():
    """
    Create diurnal cycle of hydrometeor fraction
    """

    ds = prepare_data()

    # convert from UTC to local time by substracting 4 hours
    ds["time"] = ds.time - np.timedelta64(4, "h")

    dct_stats = statistics(ds)

    plot_diurnal(
        hf_sh_diurnal=dct_stats["hf_sh_diurnal"],
        hf_co_diurnal=dct_stats["hf_co_diurnal"],
        rel_occ_sh_diurnal=dct_stats["rel_occ_sh_diurnal"],
        rel_occ_co_diurnal=dct_stats["rel_occ_co_diurnal"],
        occ_sh=dct_stats["occ_sh"],
        occ_co=dct_stats["occ_co"],
    )


def prepare_data():
    """
    Prepare data needed for the figure
    """

    # calculate cloud mask with respect to lcl
    da_cm_lcl = cloud_mask_lcl_grid()

    # read cloud types
    ds_ct = read_cloudtypes()

    # classify by region
    da_pos_class = classify_region()

    # align the gps-based and radar-based data
    da_cm_lcl, ds_ct, da_pos_class = xr.align(
        da_cm_lcl, ds_ct, da_pos_class, join="inner"
    )

    ds = xr.merge([ds_ct, da_pos_class, da_cm_lcl])

    return ds


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

    return dct_stats


def plot_diurnal(
    hf_sh_diurnal,
    hf_co_diurnal,
    rel_occ_sh_diurnal,
    rel_occ_co_diurnal,
    occ_sh,
    occ_co,
):
    """
    Plot diurnal cycle
    """

    hf_co_diurnal = hf_co_diurnal.load()
    hf_sh_diurnal = hf_sh_diurnal.load()

    norm = mcolors.BoundaryNorm(np.arange(0, 0.61, 0.05), CMAP.N)

    fig, axes = plt.subplots(3, 1, figsize=(6, 5), constrained_layout=True)

    axes[0].annotate(
        "a) Shallow (#{:,})".format(occ_sh.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
    axes[1].annotate(
        "b) Congestus (#{:,})".format(occ_co.values),
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )

    axes[2].annotate(
        "c) Relative occurrence",
        xy=(0, 1),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )

    kwargs = dict(shading="nearest", cmap=CMAP, norm=norm)
    im = axes[0].pcolormesh(
        hf_sh_diurnal.hour,
        hf_sh_diurnal.height * 1e-3,
        hf_sh_diurnal.T,
        **kwargs
    )
    im = axes[1].pcolormesh(
        hf_co_diurnal.hour,
        hf_co_diurnal.height * 1e-3,
        hf_co_diurnal.T,
        **kwargs
    )

    axes[2].plot(
        rel_occ_sh_diurnal.hour,
        rel_occ_sh_diurnal,
        color=COLOR_SHALLOW,
        label="Shallow",
    )
    axes[2].plot(
        rel_occ_co_diurnal.hour,
        rel_occ_co_diurnal,
        color=COLOR_CONGESTUS,
        label="Congestus",
    )
    leg = axes[2].legend(loc="lower left", ncol=2)
    leg.set_in_layout(False)

    for ax in axes[:2]:
        ax.set_ylabel("Height above\nLCL [km]")
        ax.set_yticks(np.arange(-0.5, 4.5, 0.25), minor=True)

    for ax in axes:
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticks(np.arange(0, 24, 1), minor=True)
        ax.set_xticklabels(np.arange(0, 24, 2))
        ax.set_xlim([-0.5, 23.5])

    # shallow axis
    axes[0].set_yticks(np.arange(-1, 1, 0.5))
    axes[0].set_yticklabels([-1, -0.5, "LCL", 0.5])
    axes[0].set_ylim(
        [
            hf_sh_diurnal.height.isel(
                height=hf_sh_diurnal.any("hour").values.argmax()
            )
            * 1e-3,
            0.6,
        ]
    )

    # congestus axis
    axes[1].set_yticks(np.arange(-1, 5, 1))
    axes[1].set_yticklabels([-1, "LCL", 1, 2, 3, 4])
    axes[1].set_ylim(
        [
            hf_sh_diurnal.height.isel(
                height=hf_co_diurnal.any("hour").values.argmax()
            )
            * 1e-3,
            4,
        ]
    )

    axes[2].set_ylim([0, 0.07])
    axes[2].set_ylabel("Density [h$^{-1}$]")
    axes[2].set_xlabel("Hour [LT, UTC-4]")

    fig.colorbar(
        im,
        ax=axes[:2],
        label="Hydrometeor fraction",
        ticks=np.arange(0, 0.7, 0.1),
    )

    for ax in axes:
        ax.grid(False)

    plt.savefig(
        os.path.join(
            os.environ["PATH_PLOT"], "diurnal.png"
        ),
    )


if __name__ == "__main__":
    main()
