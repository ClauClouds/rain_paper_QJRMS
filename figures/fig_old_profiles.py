"""
Hydrometeor fraction profiles.
"""

import os
import sys

# add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

import matplotlib.pyplot as plt
import numpy as np
#from dotenv import load_dotenv
from fig04_diurnal import prepare_data
from mpl_style import COLOR_N, COLOR_S, COLOR_T

#load_dotenv()


def main():
    """
    Creates hydrometeor fraction profiles
    """

    ds = prepare_data()

    dct_stats = statistics(ds)

    plot_profiles(
        hf_all=dct_stats["hf_all"],
        hf_n=dct_stats["hf_n"],
        hf_t=dct_stats["hf_t"],
        hf_s=dct_stats["hf_s"],
        hf_shco=dct_stats["hf_shco"],
        hf_n_shco=dct_stats["hf_n_shco"],
        hf_t_shco=dct_stats["hf_t_shco"],
        hf_s_shco=dct_stats["hf_s_shco"],
    )


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

    # hydrometeor fraction from all observations
    dct_stats["hf_all"] = ds.cloud_mask.mean("time")

    # hydrometeor fraction from all observations by region
    dct_stats["hf_s"] = ds.cloud_mask.isel(time=ds.region == 0).mean("time")
    dct_stats["hf_t"] = ds.cloud_mask.isel(time=ds.region == 1).mean("time")
    dct_stats["hf_n"] = ds.cloud_mask.isel(time=ds.region == 2).mean("time")

    # same but for observations that are either shallow or congestus clouds
    is_shco = (ds.shape == 0) | (ds.shape == 1)
    dct_stats["hf_shco"] = ds.cloud_mask.isel(time=is_shco).mean("time")
    dct_stats["hf_s_shco"] = ds.cloud_mask.isel(
        time=(ds.region == 0) & is_shco
    ).mean("time")
    dct_stats["hf_t_shco"] = ds.cloud_mask.isel(
        time=(ds.region == 1) & is_shco
    ).mean("time")
    dct_stats["hf_n_shco"] = ds.cloud_mask.isel(
        time=(ds.region == 2) & is_shco
    ).mean("time")

    # load data
    dct_stats = {k: v.load() for k, v in dct_stats.items()}

    return dct_stats


def plot_profiles(
    hf_all, hf_n, hf_t, hf_s, hf_shco, hf_n_shco, hf_t_shco, hf_s_shco
):
    """
    Plot hydrometeor fraction profiles for all clouds and for shallow and
    congestus clouds only. The profiles are plotted for the entire campaign
    and for the three regions separately.

    First panel: all observations on entire height grid
    Second panel: all observations for the lowest 4 km only
    Third panel: shallow and congestus clouds for the lowest 4 km only
    """

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), constrained_layout=True)

    axes[0].annotate(
        "a) All observations",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )

    axes[1].annotate(
        "b) All observations, zoom",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )

    axes[2].annotate(
        "c) Shallow and congestus",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )

    # all observations
    for ax in axes[:2]:
        # northern region
        ax.plot(
            hf_n, hf_n.height * 1e-3, 
            color=COLOR_N, label="Northern", zorder=1
        )

        # transition region
        ax.plot(
            hf_t,
            hf_t.height * 1e-3,
            color=COLOR_T,
            label="Transition",
            zorder=1,
        )

        # southern region
        ax.plot(
            hf_s, hf_s.height * 1e-3, color=COLOR_S, label="Southern", zorder=1
        )

        # all clouds
        ax.plot(
            hf_all,
            hf_all.height * 1e-3,
            color="k",
            label="All regions",
            zorder=1,
        )

        # fill area between the region and all
        kwargs = dict(
            x1=hf_all, y=hf_all.height * 1e-3, linewidth=0, alpha=0.5, zorder=0
        )
        ax.fill_betweenx(x2=hf_s, color=COLOR_S, **kwargs)
        ax.fill_betweenx(x2=hf_t, color=COLOR_T, **kwargs)
        ax.fill_betweenx(x2=hf_n, color=COLOR_N, **kwargs)

    # shallow or congestus clouds
    # northern region
    axes[2].plot(
        hf_n_shco,
        hf_n_shco.height * 1e-3,
        color=COLOR_N,
        label="Northern",
        zorder=1,
    )

    # transition region
    axes[2].plot(
        hf_t_shco,
        hf_t_shco.height * 1e-3,
        color=COLOR_T,
        label="Transition",
        zorder=1,
    )

    # southern region
    axes[2].plot(
        hf_s_shco,
        hf_s_shco.height * 1e-3,
        color=COLOR_S,
        label="Southern",
        zorder=1,
    )

    # all regions
    axes[2].plot(
        hf_shco,
        hf_shco.height * 1e-3,
        color="k",
        label="All regions",
        zorder=1,
    )

    # fill area between the region and all
    kwargs = dict(
        x1=hf_shco, y=hf_shco.height * 1e-3, linewidth=0, alpha=0.5, zorder=0
    )
    axes[2].fill_betweenx(x2=hf_s_shco, color=COLOR_S, **kwargs)
    axes[2].fill_betweenx(x2=hf_t_shco, color=COLOR_T, **kwargs)
    axes[2].fill_betweenx(x2=hf_n_shco, color=COLOR_N, **kwargs)

    axes[2].legend(loc="upper right")

    axes[0].set_xlim([0, 0.2])
    axes[0].set_yticks(np.arange(-1, 11, 1))
    axes[0].set_yticklabels([-1, "LCL", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axes[0].set_ylim([-1, 10])

    axes[1].set_xlim([0, 0.2])
    axes[1].set_yticks(np.arange(-1, 4.5, 1))
    axes[1].set_yticklabels([-1, "LCL", 1, 2, 3, 4])
    axes[1].set_ylim([-1, 4])

    axes[2].set_xlim([0, 0.4])
    axes[2].set_yticks(np.arange(-1, 4.5, 1))
    axes[2].set_yticklabels([-1, "LCL", 1, 2, 3, 4])
    axes[2].set_ylim([-1, 4])

    for ax in axes:
        ax.set_xlabel("Hydrometeor fraction")

    axes[0].set_ylabel("Height above LCL [km]")

    plt.savefig(
        os.path.join(os.environ["PATH_PLOT"], "profiles.png"),
    )


if __name__ == "__main__":
    main()
