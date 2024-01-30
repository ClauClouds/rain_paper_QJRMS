"""
CFADs for different LWP levels.
"""

import os
from string import ascii_lowercase as abc

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xhistogram.xarray import histogram

from readers.cloudtypes import read_cloud_class
from readers.wband import read_lwp
from readers.wband import read_radar_multiple
from figures.mpl_style import CMAP
from scipy.ndimage.filters import gaussian_filter


def main():
    """
    Calculate radar cfads for specific LWP levels

    The bins of the height correspond to the actual radar height bins.
    
    
    edits:
    - edited by Claudia on 30.01.2024 to process shallow and congestus separately
    """

    ds_ze_s, ds_lwp_s, ds_ze_c, ds_lwp_c, ds_ze, ds_lwp = prepare_data()

    # compute histogram
    bins_lwp = np.array([-100, 10, 50, 100, 300, 1000])
    bins_ze = np.arange(-100, 50.2, 0.2)
    
    
    da_cfad_s = compute_cfad(
        ze=ds_ze_s.radar_reflectivity,
        lwp=ds_lwp_s.lwp,
        bins_ze=bins_ze,
        bins_lwp=bins_lwp,
    )
    
    da_cfad_c = compute_cfad(
        ze=ds_ze_c.radar_reflectivity,
        lwp=ds_lwp_c.lwp,
        bins_ze=bins_ze,
        bins_lwp=bins_lwp,
    )
     
    da_cfad = compute_cfad(
        ze=ds_ze.radar_reflectivity,
        lwp=ds_lwp.lwp,
        bins_ze=bins_ze,
        bins_lwp=bins_lwp,
    )       
 
    # change lwp bin labels to the upper boundary
    da_cfad_s["lwp_bin"] = bins_lwp[1:]
    da_cfad_c["lwp_bin"] = bins_lwp[1:]
    da_cfad["lwp_bin"] = bins_lwp[1:]

    # compute normalized cfad
    da_cfad_norm_s = normalize_cfad(da_cfad_s)
    da_cfad_norm_c = normalize_cfad(da_cfad_c)
    da_cfad_norm = normalize_cfad(da_cfad)

    # visualize cfads
    visualize(da_cfad_s, da_cfad_norm_s, da_cfad_c, da_cfad_norm_c, da_cfad, da_cfad_norm)


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

    edits:
    - modified by Claudia on 30.01.2024 to derive ze/lwp for shallow and congestus clouds separately
    
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

    # reduce to times with stratiform or shallow cumulus clouds and both 
    ds_ze_s = ds_ze.sel(time=(ds_ct.shape == 0))
    ds_ze_c = ds_ze.sel(time=(ds_ct.shape == 1))
    ds_ze = ds_ze.sel(time=((ds_ct.shape == 1) | (ds_ct.shape == 0)))

    ds_ze_s, ds_lwp_s = xr.align(ds_ze_s, ds_lwp)
    ds_ze_c, ds_lwp_c = xr.align(ds_ze_c, ds_lwp)
    ds_ze, ds_lwp = xr.align(ds_ze, ds_lwp)

    return ds_ze_s, ds_lwp_s, ds_ze_c, ds_lwp_c, ds_ze, ds_lwp


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


def visualize(da_cfad_s, da_cfad_norm_s, da_cfad_c, da_cfad_norm_c, da_cfad, da_cfad_norm):
    """
    Visualize CFADs for different LWP levels.
    edits:
    - modified by Claudia on 30.01.2024 to plot congestus with colormesh and shallow with contours on top
   
    Parameters
    ----------
    da_cfad: same as da_cfad_norm, but in counts and not normalized
    da_cfad_norm: normalized histogram of radar reflectivity as a function of
      height and liquid water path. The liquid water path values represent the
      right boundary of the interval.
    """

    # this makes sure that values with 0 count are not drawn
    da_cfad_norm_s = da_cfad_norm_s.where(da_cfad_norm_s > 0)
    da_cfad_norm_c = da_cfad_norm_c.where(da_cfad_norm_c > 0)
    da_cfad_norm = da_cfad_norm.where(da_cfad_norm > 0)

    
    
    # liquid water path upper boundary
    headers = [
        "< 10 g m$^{-2}$",
        "< 50 g m$^{-2}$",
        "< 100 g m$^{-2}$",
        "< 300 g m$^{-2}$",
        "< 1000 g m$^{-2}$",
        "all",
    ]

    # colormap
    cmap = CMAP
    bounds = np.arange(0, 1.1, 0.1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6, 4),
        constrained_layout=True,
        sharex="all",
        sharey="all",
    )

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    for i, ax in enumerate(axes.flatten()[:-1]):
        
        # plot colormesh of congestus clouds 
        ax.pcolormesh(
            da_cfad_norm_c.radar_reflectivity_bin,
            da_cfad_norm_c.height * 1e-3,
            da_cfad_norm_c.isel(lwp_bin=i),
            norm=norm,
            cmap=cmap,
            shading="nearest",
        )


        # plot contours of shallow clouds
        cs = ax.contour(da_cfad_norm_s.radar_reflectivity_bin, 
                        da_cfad_norm_s.height * 1e-3, 
                        gaussian_filter(da_cfad_norm_s.isel(lwp_bin=i),3),
                        levels = [0.3, 0.5, 0.7],
                        cmap=plt.cm.Greys, 
                        interpolation='none'
        )
    
        
        
    # all observations
    da_cfad_all = da_cfad.sum("lwp_bin")
    da_cfad_all_norm = da_cfad_all / da_cfad_all.max(
        ["radar_reflectivity_bin", "height"]
    )
    da_cfad_all_norm = da_cfad_all_norm.where(da_cfad_all_norm > 0)
    im = axes[-1, -1].pcolormesh(
        da_cfad_all_norm.radar_reflectivity_bin,
        da_cfad_all_norm.height * 1e-3,
        da_cfad_all_norm,
        norm=norm,
        cmap=cmap,
        shading="nearest",
    )

    fig.colorbar(im, ax=axes, label="Density", shrink=0.5, ticks=bounds)

    for ax in axes.flatten():
        ax.set_xticks(np.arange(-60, 50, 5), minor=True)
        ax.set_xticks(np.arange(-60, 50, 20), minor=False)

    axes[0, 0].set_xlim([-60, 30])
    axes[0, 0].set_ylim([0, 4])
    axes[0, 0].set_yticks(np.arange(0, 5, 1))

    axes[-1, 0].set_xlabel("Radar reflectivity [dBZ]")
    axes[-1, 0].set_ylabel("Height [km]")

    # annotate letter labels
    for i, ax in enumerate(axes.flatten()):
        ax.annotate(
            f"{abc[i]})   {headers[i]}",
            xy=(0, 1),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
        )

    plt.savefig(
        os.path.join('/work/plots_rain_paper/', "figure3.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


if __name__ == "__main__":
    main()
