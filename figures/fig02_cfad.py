"""
CFADs for different LWP levels.
"""

import os
import sys
from string import ascii_lowercase

# add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv
from mpl_style import CMAP
from xhistogram.xarray import histogram

from readers.cloudtypes import read_cloud_class
from readers.lwp import read_lwp
from readers.radar import read_radar_multiple

load_dotenv()
ProgressBar().register()


def main():
    """
    Calculate radar cfads for specific LWP levels

    The bins of the height correspond to the actual radar height bins.
    """

    ds_ze, ds_lwp = prepare_data()

    # compute histogram
    bins_lwp = np.array([-100, 10, 50, 100, 300, 1000])
    bins_ze = np.arange(-100, 50.2, 0.2)
    da_cfad = compute_cfad(
        ze=ds_ze.radar_reflectivity,
        lwp=ds_lwp.wlwp,
        bins_ze=bins_ze,
        bins_lwp=bins_lwp,
    )

    # change lwp bin labels to the upper boundary
    da_cfad["wlwp_bin"] = bins_lwp[1:]

    # compute normalized cfad
    da_cfad_norm = normalize_cfad(da_cfad)

    # visualize cfads
    visualize(da_cfad, da_cfad_norm)


def prepare_data():
    """
    Prepare data for CFAD computations. The data for the entire campaign is
    prepared at once. Radar reflectivity data is reduced to times, where
    stratiform or shallow cumulus clouds occur

    The following datasets are required:
    - liquid water path
    - cloud classification
    - radar reflectivity

    All datasets are aligned temporally. No interpolation is performed.

    Returns
    -------
    ds_ze: radar reflectivity dataset
    ds_lwp: liquid water path dataset
    """

    ds_lwp = read_lwp()
    ds_ct = read_cloud_class()
    ds_ze = read_radar_multiple()

    # on common timestamp
    ds_ze, ds_ct = xr.align(ds_ze, ds_ct)

    # reduce to times with stratiform or shallow cumulus clouds
    ds_ze = ds_ze.sel(time=((ds_ct.shape == 1) | (ds_ct.shape == 0)))

    ds_ze, ds_lwp = xr.align(ds_ze, ds_lwp)

    return ds_ze, ds_lwp


def compute_cfad(ze, lwp, bins_ze, bins_lwp):
    """
    Compute radar reflectivity CFAD for specific LWP bins.

    Parameters
    ----------
    ze: radar reflectivity data array
    lwp: liquid water path data array
    bins_ze: radar reflectivity bin edges
    bins_lwp: liquid water path bin edges

    Returns
    -------
    cfad: multi-dimensional histogram of radar reflectivity in absolute counts
    """

    da_cfad = histogram(ze, lwp, bins=[bins_ze, bins_lwp], dim=["time"])
    da_cfad = da_cfad.compute()

    return da_cfad


def normalize_cfad(da_cfad):
    """
    Normalize CFAD given in absolute counts by the maximum value for each lwp
    bin.

    Parameters
    ----------
    da_cfad: CFAD in absolute counts

    Returns
    -------
    da_cfad_norm: normalized cfad
    """

    da_cfad_norm = da_cfad / da_cfad.max(["height", "radar_reflectivity_bin"])

    return da_cfad_norm


def visualize(da_cfad, da_cfad_norm):
    """
    Visualize CFADs for different LWP levels.

    Parameters
    ----------
    da_cfad: same as da_cfad_norm, but in counts and not normalized
    da_cfad_norm: normalized histogram of radar reflectivity as a function of
      height and liquid water path. The liquid water path values represent the
      right boundary of the interval.
    """

    # this makes sure that values with 0 count are not drawn
    da_cfad_norm = da_cfad_norm.where(da_cfad_norm > 0)

    # liquid water path upper boundary
    headers = [
        "0$-$10 g m$^{-2}$",
        "10$-$50 g m$^{-2}$",
        "50$-$100 g m$^{-2}$",
        "100$-$300 g m$^{-2}$",
        "300$-$1000 g m$^{-2}$",
        "All observations",
    ]

    # colormap
    bounds = np.arange(0, 1.1, 0.1)
    norm = mcolors.BoundaryNorm(bounds, CMAP.N)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7, 4),
        constrained_layout=True,
        sharex="all",
        sharey="all",
    )

    for ax in axes.flatten():
        ax.grid(False)

    for i, ax in enumerate(axes.flatten()[:-1]):
        ax.pcolormesh(
            da_cfad_norm.radar_reflectivity_bin,
            da_cfad_norm.height * 1e-3,
            da_cfad_norm.isel(wlwp_bin=i),
            norm=norm,
            cmap=CMAP,
            shading="nearest",
        )

    # all observations
    da_cfad_all = da_cfad.sum("wlwp_bin")
    da_cfad_all_norm = da_cfad_all / da_cfad_all.max(
        ["radar_reflectivity_bin", "height"]
    )
    da_cfad_all_norm = da_cfad_all_norm.where(da_cfad_all_norm > 0)
    im = axes[-1, -1].pcolormesh(
        da_cfad_all_norm.radar_reflectivity_bin,
        da_cfad_all_norm.height * 1e-3,
        da_cfad_all_norm,
        norm=norm,
        cmap=CMAP,
        shading="nearest",
    )

    fig.colorbar(
        im, ax=axes, label="Relative density", ticks=np.arange(0, 1.1, 0.2)
    )

    for ax in axes.flatten():
        ax.set_xticks(np.arange(-60, 50, 5), minor=True)
        ax.set_xticks(np.arange(-60, 50, 20))
        ax.set_yticks(np.arange(0, 4.01, 0.25), minor=True)
        ax.set_yticks(np.arange(0, 5, 1))

    axes[0, 0].set_xlim([-60, 30])
    axes[0, 0].set_ylim([0, 4])
    axes[0, 0].set_yticks(np.arange(0, 5, 1))

    axes[-1, 0].set_xlabel("Radar reflectivity [dBZ]")
    axes[-1, 0].set_ylabel("Height [km]")

    # annotate letter labels
    for i, ax in enumerate(axes.flatten()):
        ax.annotate(
            f"{ascii_lowercase[i]}) {headers[i]}",
            xy=(0, 1),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
        )

    plt.savefig(
        os.path.join('/net/ostro/plots_rain_paper/', "cfad.png"),
    )
    plt.close()


if __name__ == "__main__":
    main()
