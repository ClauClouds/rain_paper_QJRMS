"""
Joint histogram of cloud base and cloud top height.
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dotenv import load_dotenv
from mpl_style import CMAP

# add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

from fig04_diurnal import prepare_data

load_dotenv()


def main():
    """
    Create diurnal cycle of hydrometeor fraction
    """

    ds = prepare_data()

    da_hist = statistics(ds)

    plot_histogram(da_hist=da_hist)


def statistics(ds):
    """
    Calculate statistics
    """

    # filter only shallow and congestus clouds
    ds = ds.sel(time=((ds.shape == 1) | (ds.shape == 0)))

    # cloud base and cloud top height
    ds["cloud_height"] = xr.where(ds.cloud_mask == 0, np.nan, ds.height)
    ds["cloud_base"] = ds["cloud_height"].min("height", skipna=True)
    ds["cloud_top"] = ds["cloud_height"].max("height", skipna=True)

    # load cloud base and cloud top height
    ds["cloud_base"] = ds["cloud_base"].load()
    ds["cloud_top"] = ds["cloud_top"].load()

    # resample to 200 m bins
    dz = 200
    bins_wrt_lcl = np.arange(-1000, 5200 + dz, dz)
    bins_wrt_lcl_cen = (bins_wrt_lcl[1:] + bins_wrt_lcl[:-1]) / 2

    hist, _, _ = np.histogram2d(
        x=ds["cloud_top"].values,
        y=ds["cloud_base"].values,
        bins=[bins_wrt_lcl, bins_wrt_lcl],
    )

    # create data array
    da_hist = xr.DataArray(
        hist,
        dims=["radar_cloud_top", "radar_cloud_base"],
        coords={
            "radar_cloud_top": bins_wrt_lcl_cen,
            "radar_cloud_base": bins_wrt_lcl_cen,
        },
    )

    return da_hist


def plot_histogram(da_hist):
    """
    Plot histogram
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    im = ax.pcolormesh(
        da_hist.radar_cloud_base * 1e-3,
        da_hist.radar_cloud_top * 1e-3,
        da_hist,
        cmap=CMAP,
        norm=mcolors.LogNorm(vmin=1, vmax=30000),
        shading="nearest",
    )
    fig.colorbar(im, ax=ax, label="Count")

    # set equal aspect ratio
    ax.set_aspect("equal")

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)

    ax.set_yticks(np.arange(-1, 5, 1))
    ax.set_yticklabels([-1, "LCL", 1, 2, 3, 4])

    ax.set_xticks(np.arange(-1, 5, 1))
    ax.set_xticklabels([-1, "LCL", 1, 2, 3, 4])

    # indicate dotted line for LCL height
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.6, color="k", linestyle=":", linewidth=1)

    ax.annotate(
        "LCL + 600 m",
        xy=(4, 0.6),
        xycoords="data",
        ha="right",
        va="bottom",
    )

    ax.plot([-1, 5], [-1, 5], color="k", linewidth=1)

    ax.set_xlabel("Cloud base height above LCL [km]")
    ax.set_ylabel("Cloud top height above LCL [km]")

    plt.savefig(
        os.path.join(os.getenv("PATH_PLOT"), "basetop.png"),
    )


if __name__ == "__main__":
    main()
