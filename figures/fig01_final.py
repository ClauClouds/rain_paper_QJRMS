""" figure 1 after receiving comments from Isabel and Sabrina: the idea is to have the trajeectory with 
the human label flagged, and then the cloud fraction profiles for the three distinct regions. Then, we produce 
also an additional material figure with the other LWP time serie panel for human labelled patterns and the 
cloud reflectivity profiles for human labelled patterns."""


from readers.cloudtypes import read_merian_classification
import matplotlib.pyplot as plt
import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cfeature           # import features
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
import pdb
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib
from figures.mpl_style import CMAP_HF_ALL, COLOR_SHALLOW, COLOR_CONGESTUS, COLOR_N, COLOR_S, COLOR_T
from cloudtypes.cloudtypes import classify_region, cloud_mask_lcl_grid
from readers.cloudtypes import read_cloudtypes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def main():
    

    """
    calculate the trajectory path with the mesoscale patterns in it
    
    """
    # read the trajectory patterns
    class_cloud_data = read_merian_classification()

    """
    Creates hydrometeor fraction profile
    """
    ds = prepare_data()
    dct_stats = statistics(ds)

    # defining domain 
    minlon = 6.; maxlon = 15.; minlat = -61; maxlat = -50.
    extent_param = [minlon, maxlon, minlat, maxlat]
    
    # plot minutely ship position
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig = plt.figure(figsize=(16, 10))
   

    # Set the aspect ratio of both subplots to be equal
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("b) RV MSM trajectory during EUREC$^{4}$A", loc='left', fontweight='black')
    ax2.set_xlim([0, 0.55])
    ax2.set_yticks(np.arange(-1, 4.5, 1))
    ax2.set_yticklabels([-1, "LCL", 1, 2, 3, 4])
    ax2.set_ylim([-1, 4])
    
    font_size = 16
    ax2.set_xlabel("Hydrometeor fraction",fontsize=font_size)
    ax2.set_ylabel("Height above LCL [km]",fontsize=font_size)
    plot_profiles_regions(ax2, 
                          hf_sh=dct_stats["hf_sh"],
                          hf_n_sh=dct_stats["hf_n_sh"], 
                          hf_t_sh=dct_stats["hf_t_sh"], 
                          hf_s_sh=dct_stats["hf_s_sh"],
                          hf_co=dct_stats["hf_co"],
                          hf_n_co=dct_stats["hf_n_co"], 
                          hf_t_co=dct_stats["hf_t_co"], 
                          hf_s_co=dct_stats["hf_s_co"]
                          )

    #ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.set_title("a) Hydrometeor fraction for different regions", loc='left', fontweight='black')

    # visualize trajectory
    visualize_trajectory(fig, ax1, class_cloud_data, extent_param, hf=dct_stats["hf_all"])
    
    ax1.grid(False) # remove grid
    fig.tight_layout()
    
    
    # saving figure as png
    plt.savefig(
        "/work/plots_rain_paper/fig_1_new.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()




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



def visualize_trajectory(fig, ax, class_cloud_data, extent_param, hf):
    """
    function to plot the trajectory from the data with the colored dots 
    input:
    fig
    gs: axis object
    class_cloud_data: xarray dataset
    extent_param: list with the extent of the plot
    hf: hydrometeor fraction to plot on the trajeectory
    """
     # Set the projection for the first subplot
    #ax = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())

    # plot settings
    # colors
    col_ocean = '#CAE9FB'
    col_land = 'grey'
    col_coastlines = 'darkgrey'
    cmap_track = 'plasma'
    cmap_hf = CMAP_HF_ALL
    
    # fontsizes
    fs_grid = 16
    fs_cbar_labels = 16
    fs_track_labels = 16
    
    # zorders
    zorder_land = 0
    zorder_coastlines = 1
    zorder_gridlines = 2
    zorder_day_marker = 4
    zorder_track = 3
    zorder_day_annotation = 5
    
    # size of the patterns
    sval = 40
    

    # set map extent
    ax.set_extent(extents=(-61, -50, 6, 15), crs=ccrs.PlateCarree())
    # add land feature
    land_10m = cfeature.NaturalEarthFeature(category='physical', name='land', 
                                            scale='10m',
                                        edgecolor=col_coastlines,
                                        linewidth=0.5,
                                        facecolor=col_land)
    ax.add_feature(land_10m, zorder=zorder_land)
    
    # add lat lon grid
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='silver', 
                      draw_labels=True, zorder=zorder_gridlines,
                      x_inline=False, y_inline=False)
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-70, -40, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 16, 1))
    gl.xlabel_style = {'size': fs_grid, 'color': 'k'}
    gl.ylabel_style = {'size': fs_grid, 'color': 'k'}
    
        
    rect1 = matplotlib.patches.Rectangle((-61, 12.5), 
                                        11, 2.5, 
                                        color ='royalblue', 
                                        alpha=0.2) 
    #    minlon = 6.; maxlon = 15.; minlat = -61; maxlat = -50;

    rect2 = matplotlib.patches.Rectangle((-61, 10.5), 
                                        11, 2., 
                                        color ='lightsteelblue', 
                                        alpha=0.3) 
    
    rect3 = matplotlib.patches.Rectangle((-61, 6), 
                                        11, 4.5, 
                                        color ='lavender', 
                                        alpha=0.3) 
    
    ax.add_patch(rect1) 
    ax.add_patch(rect2) 
    ax.add_patch(rect3) 
    
    
    # plot ship track
 
    ax.plot(class_cloud_data.longitude.values, 
            class_cloud_data.latitude.values, 
            color='black', 
            transform=ccrs.PlateCarree(),
            zorder=zorder_track, 
            alpha=0.3,
            label="RV MSM trajectory"
           )
    # set colorbar for 
    ax.hlines(y=10.5, xmin=-61.5, xmax=-50, color='black', linestyles='--')
    ax.hlines(y=12.5, xmin=-61.5, xmax=-50, color='black', linestyles='--')

    # plot sugar matches
    sugar = np.where(class_cloud_data.morph_ident.values[:,0] == 1)
    ax.scatter(class_cloud_data.longitude.values[sugar], 
            class_cloud_data.latitude.values[sugar], 
            color='#E63946', 
            label='sugar', 
            marker='o', 
            s=sval)
    
    gravel = np.where(class_cloud_data.morph_ident.values[:,1] == 1)
    ax.scatter(class_cloud_data.longitude.values[gravel], 
            class_cloud_data.latitude.values[gravel], 
            color='#EC8A91', 
            label='gravel', 
            marker='o', 
            s=sval) 
    
    flower = np.where(class_cloud_data.morph_ident.values[:,2] == 1)
    ax.scatter(class_cloud_data.longitude.values[flower], 
            class_cloud_data.latitude.values[flower], 
            color='#457B9D', 
            label='flower', 
            marker='o', 
            s=sval) 
    
    fish = np.where(class_cloud_data.morph_ident.values[:,3] == 1)
    ax.scatter(class_cloud_data.longitude.values[fish], 
            class_cloud_data.latitude.values[fish], 
            color='#A8DADC', 
            label='fish', 
            marker='o', 
            s=sval) 
    
    ax.legend(frameon=True, fontsize=16)   
    
        
    return(ax)

def plot_profiles_regions(ax, 
                          hf_sh, 
                          hf_n_sh, 
                          hf_t_sh, 
                          hf_s_sh, 
                          hf_co,
                          hf_n_co, 
                          hf_t_co, 
                          hf_s_co):
    
    ax.grid(False)
    # Color the area below the LCL height in grey
    lcl_height = 0  # Assuming LCL height is at 0 km
    ax.fill_between([0, 0.55], 
                    -1, 
                    lcl_height, 
                    color='grey', 
                    alpha=0.5)
    

    # shallow clouds for regions
    # northern region
    ax.plot(
        hf_n_sh,
        hf_n_sh.height * 1e-3,
        color=COLOR_N,
        label="Northern",
        zorder=1,
    )


    # transition region
    ax.plot(
        hf_t_sh,
        hf_t_sh.height * 1e-3,
        color=COLOR_T,
        label="Transition",
        zorder=1,
    )


    # southern region
    ax.plot(
        hf_s_sh,
        hf_s_sh.height * 1e-3,
        color=COLOR_S,
        label="Southern",
        zorder=1,
    )

    # fill area between the region and all
    kwargs = dict(
        x1=hf_sh, y=hf_sh.height * 1e-3, linewidth=0, alpha=0.5, zorder=0
    )
    ax.fill_betweenx(x2=hf_s_sh, color=COLOR_S, **kwargs)
    ax.fill_betweenx(x2=hf_t_sh, color=COLOR_T, **kwargs)
    ax.fill_betweenx(x2=hf_n_sh, color=COLOR_N, **kwargs)
    
    # set fontsize in ax axis ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # congestus clouds for regions
    # northern region
    ax.plot(
        hf_n_co,
        hf_n_co.height * 1e-3,
        color=COLOR_N,
        zorder=1,
    )


    # transition region
    ax.plot(
        hf_t_co,
        hf_t_co.height * 1e-3,
        color=COLOR_T,
        zorder=1,
    )


    # southern region
    ax.plot(
        hf_s_co,
        hf_s_co.height * 1e-3,
        color=COLOR_S,
        zorder=1,
    )

    # fill area between the region and all
    kwargs = dict(
        x1=hf_co, y=hf_sh.height * 1e-3, linewidth=0, alpha=0.5, zorder=0
    )
    ax.fill_betweenx(x2=hf_s_co, color=COLOR_S, **kwargs)
    ax.fill_betweenx(x2=hf_t_co, color=COLOR_T, **kwargs)
    ax.fill_betweenx(x2=hf_n_co, color=COLOR_N, **kwargs)

     # shallow  clouds
    # plot mean of all regions
    ax.plot(
        hf_sh,
        hf_sh.height * 1e-3,
        color=COLOR_SHALLOW,
        label="Shallow",
        linewidth=4,
        zorder=1,
    )
    
    # congestus clouds
    # plot mean of all regions
    ax.plot(
        hf_co,
        hf_co.height * 1e-3,
        color=COLOR_CONGESTUS,
        label="Congestus",
        linewidth=4,
        zorder=1,
    )


    # set up the legend
    # print legend
        
        
    # Create custom legend handles
    legend_handles = [
        Patch(facecolor=COLOR_S, 
              edgecolor='none', 
              alpha=0.5, 
              label='Southern'),
        Patch(facecolor=COLOR_T, 
              edgecolor='none', 
              alpha=0.5, 
              label='Transition'),
        Patch(facecolor=COLOR_N, 
              edgecolor='none', 
              alpha=0.5, 
              label='Northern'), 
        Line2D([0], [0], 
               color=COLOR_SHALLOW, 
               linewidth=4, 
               label='Shallow clouds'),
        Line2D([0], [0], 
               color=COLOR_CONGESTUS, 
               linewidth=4, 
               label='Congestus clouds')        
    ]

    # Add the custom legend handles to the legend
    ax.legend(handles=legend_handles, loc="upper right", fontsize=16)

    return(ax)
    




def statistics(ds):


    """
    Hydrometeor fraction statistics

    Parameters
    -------
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
    dct_stats["hf_s_shco"] = ds.cloud_mask.isel(time=(ds.region == 0) & is_shco).mean("time")

    dct_stats["hf_t_shco"] = ds.cloud_mask.isel(time=(ds.region == 1) & is_shco).mean("time")
    dct_stats["hf_n_shco"] = ds.cloud_mask.isel(time=(ds.region == 2) & is_shco).mean("time")

    # same but with shallow only
    is_sh = (ds.shape == 0)
    dct_stats["hf_sh"] = ds.cloud_mask.isel(time=is_sh).mean("time")
    dct_stats["hf_s_sh"] = ds.cloud_mask.isel(time=(ds.region == 0) & is_sh).mean("time")
    dct_stats["hf_t_sh"] = ds.cloud_mask.isel(time=(ds.region == 1) & is_sh).mean("time")
    dct_stats["hf_n_sh"] = ds.cloud_mask.isel(time=(ds.region == 2) & is_sh).mean("time")
    
    # same but with congestus only
    is_co = (ds.shape == 1)
    dct_stats["hf_co"] = ds.cloud_mask.isel(time=is_co).mean("time")
    dct_stats["hf_s_co"] = ds.cloud_mask.isel(time=(ds.region == 0) & is_co).mean("time")
    dct_stats["hf_t_co"] = ds.cloud_mask.isel(time=(ds.region == 1) & is_co).mean("time")
    dct_stats["hf_n_co"] = ds.cloud_mask.isel(time=(ds.region == 2) & is_co).mean("time")
    
    # load data
    dct_stats = {k: v.load() for k, v in dct_stats.items()}

    return dct_stats


    
    
    
    
if __name__ == "__main__":
    main()
