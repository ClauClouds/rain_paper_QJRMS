
from readers.cloudtypes import read_merian_classification
import matplotlib.pyplot as plt
import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cfeature           # import features
import matplotlib.ticker as mticker
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib

def main():
    """
    calculate the trajectory path with the mesoscale patterns in it
    
    """
    
    # read the trajectory patterns
    class_cloud_data = read_merian_classification()


    # defining domain 
    minlon = 6.; maxlon = 15.; minlat = -61; maxlat = -50;
    extent_param = [minlon, maxlon, minlat, maxlat]
    
    
    # visualize trajectory
    visualize_trajectory(class_cloud_data, extent_param)
    

def visualize_trajectory(class_cloud_data, extent_param):
    """
    function to plot the trajectory from the data with the colored dots 
    
    """

    # plot settings
    # colors
    col_ocean = '#CAE9FB'
    col_land = 'grey'
    col_coastlines = 'darkgrey'
    cmap_track = 'plasma'
    
    # fontsizes
    fs_grid = 25
    fs_cbar_labels = 25
    fs_track_labels = 25
    
    # zorders
    zorder_land = 0
    zorder_coastlines = 1
    zorder_gridlines = 2
    zorder_day_marker = 4
    zorder_track = 3
    zorder_day_annotation = 5
    
    # size of the patterns
    sval = 20
    
    # plot minutely ship position
    fig, ax = plt.subplots(figsize=(12,16), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    
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
    
    ax.legend(frameon=True, fontsize=fs_track_labels)   
    
    ax.set_title('a) RV MSM trajectory and mesoscale patterns',     
            loc='left', 
            fontsize=fs_track_labels,
            fontweight='black')
    
    # saving figure as png
    plt.savefig(
        "/work/plots_rain_paper/fig_1_a.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
