''''
Code to display mesoscale conditions over the study area 
for the case study of the 20200212. We want to understand what
were the wind conditions over the area spanned by the ship on the 
selected day, mainly to undestand wind at surface, lcl, and 500 hpa.
as well as vertical velocity at surface, lcl and 500 hpa.

'''
from readers.ship import read_ship
from readers.lcl import read_lcl
import os
import subprocess
import pandas as pd
import xarray as xr
import pdb
from glob import glob
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units
from figures.mpl_style import CMAP, cmap_era5, cmap_wind_era5
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.mpl.ticker as cticker
import iris
import iris.plot as iplt
import iris.quickplot as qplt



def plot_png_w_wind_dir(ds_selected, 
                        ds_ship_hour, 
                        p_string, 
                        date_string):
    
    """
    function to plot specific humidity and horizontal wind direction at the selected pressure level"""
    
    
    # set max and min range for horizontal wind speed
    v_min = -0.5
    v_max = 0.5  # adjust as needed for wind speed
    cbar_label = 'Vertical velocity (m/s)'
    w = ds_selected['w'].values
    
    # read wind sped, u and v and calculate normalization in m/s over the smaller domain
    u_selected = ds_selected['u'].values
    v_selected = ds_selected['v'].values
    windspeed = ds_selected['horizontal_wind_speed'].values
    
    # Normalise the data for uniform arrow size.
    u_norm = u_selected / windspeed
    v_norm = v_selected / windspeed

    # read lat lon for plotting
    lat = ds_selected['latitude'].values
    lon = ds_selected['longitude'].values

    # Create a meshgrid for the lat/lon coordinates
    # Thin out the data for easier plotting
    skip = 2  # more dense than 5
    u_plot = u_norm[::skip, ::skip]
    v_plot = v_norm[::skip, ::skip]
    lat_plot = lat[::skip]
    lon_plot = lon[::skip]
    lon2d, lat2d = np.meshgrid(lon_plot, lat_plot)

    # Set up map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())

    # Add features
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS)

    # plot contour with wind speed
    sc = ax.pcolormesh(lon2d,
                          lat2d,
                          w[::skip, ::skip],
                          cmap=cmap_era5,
                          vmin=v_min,
                          vmax=v_max,
                          shading='auto',
                          transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Plot wind vectors 
     # scale values: Smaller = longer arrows; try 100, 50, 20, etc.
    ax.quiver(lon2d, 
              lat2d, 
              u_plot, 
              v_plot, 
              transform=ccrs.PlateCarree(), 
              color='black', 
              scale=20) 

    # Add ship trajectory
    ax.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.savefig(f'/work/plots_rain_paper/era5_20200212/w_wind_dir_{timestamp}_{p_string}.png', dpi=300)




def plot_png_q_wind_dir(ds_selected, 
                        ds_ship_hour, 
                        p_string, 
                        date_string):
    
    """
    function to plot specific humidity and horizontal wind direction at the selected pressure level"""
    
    
    # set max and min range for horizontal wind speed
    v_min = 2
    v_max = 10  # adjust as needed for wind speed
    cbar_label = 'Specific humidity (g/kg)'
    q = ds_selected['q'].values
    
    # read wind sped, u and v and calculate normalization in m/s over the smaller domain
    u_selected = ds_selected['u'].values
    v_selected = ds_selected['v'].values
    windspeed = ds_selected['horizontal_wind_speed'].values
    
    # Normalise the data for uniform arrow size.
    u_norm = u_selected / windspeed
    v_norm = v_selected / windspeed

    # read lat lon for plotting
    lat = ds_selected['latitude'].values
    lon = ds_selected['longitude'].values

    # Create a meshgrid for the lat/lon coordinates
    # Thin out the data for easier plotting
    skip = 2  # more dense than 5
    u_plot = u_norm[::skip, ::skip]
    v_plot = v_norm[::skip, ::skip]
    lat_plot = lat[::skip]
    lon_plot = lon[::skip]
    lon2d, lat2d = np.meshgrid(lon_plot, lat_plot)

    # Set up map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())

    # Add features
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS)

    # plot contour with wind speed
    sc = ax.pcolormesh(lon2d,
                          lat2d,
                          q[::skip, ::skip],
                          cmap=cmap_wind_era5,
                          vmin=v_min,
                          vmax=v_max,
                          shading='auto',
                          transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Plot wind vectors 
     # scale values: Smaller = longer arrows; try 100, 50, 20, etc.
    ax.quiver(lon2d, 
              lat2d, 
              u_plot, 
              v_plot, 
              transform=ccrs.PlateCarree(), 
              color='black', 
              scale=20) 

    # Add ship trajectory
    ax.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.savefig(f'/work/plots_rain_paper/era5_20200212/q_wind_dir_{timestamp}_{p_string}.png', dpi=300)

def plot_h_wind_speed_dir(ds_era5, ds_crop, ds_ship_hour, p_string, timestamp, date_string, var_name='Horizontal Wind Speed'):
    """
    Function to plot horizontal wind speed and direction at the selected pressure level
    and save the plot as a PNG file.
    
    Parameters
    ----------
    var_array: nparray containing ERA5 data (lat, lon)
    ds_era5: xarray dataset containing ERA5 data for the selected time with lat lon coordinates for plotting
    ds_ship_hour: xarray dataset containing ship data for the selected hour
    var_name: str, variable name to plot ('Horizontal Wind Speed' or 'Horizontal Wind Direction')
    closest_height: float, height in meters corresponding to the pressure level
    p_string: str, string indicating the pressure level ('surface', 'lcl', 'cloud_top', '500hPa')
    date_string: string for the date in the format 'YYYY-MM-DD'
    dependencies;
    - sample_data
    
    'w': (('latitude', 'longitude'), w_selected),
    'u': (('latitude', 'longitude'), u_selected),
    'v': (('latitude', 'longitude'), v_selected),
    'horizontal_wind_speed': (('latitude', 'longitude'), horizontal_wind_speed),
    'wind_direction': (('latitude', 'longitude'), wind_direction)
  
    """
    
    # set max and min range for horizontal wind speed
    v_min = 8
    v_max = 15  # adjust as needed for wind speed
    cbar_label = 'Horizontal Wind Speed (m/s)'

    # read wind sped, u and v and calculate normalization in m/s over the smaller domain
    u_selected = ds_crop['u'].values
    v_selected = ds_crop['v'].values
    windspeed = ds_crop['horizontal_wind_speed'].values
    
    # Normalise the data for uniform arrow size.
    u_norm = u_selected / windspeed
    v_norm = v_selected / windspeed

    # read lat lon for plotting
    lat = ds_crop['latitude'].values
    lon = ds_crop['longitude'].values

    # Create a meshgrid for the lat/lon coordinates
    # Thin out the data for easier plotting
    skip = 2  # more dense than 5
    u_plot = u_norm[::skip, ::skip]
    v_plot = v_norm[::skip, ::skip]
    lat_plot = lat[::skip]
    lon_plot = lon[::skip]
    lon2d, lat2d = np.meshgrid(lon_plot, lat_plot)

    # Set up map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())

    # Add features
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS)

    # plot contour with wind speed
    sc = ax.pcolormesh(lon2d,
                          lat2d,
                          windspeed[::skip, ::skip],
                          cmap=cmap_wind_era5,
                          vmin=v_min,
                          vmax=v_max,
                          shading='auto',
                          transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Plot wind vectors 
     # scale values: Smaller = longer arrows; try 100, 50, 20, etc.
    ax.quiver(lon2d, 
              lat2d, 
              u_plot, 
              v_plot, 
              transform=ccrs.PlateCarree(), 
              color='black', 
              scale=20) 

    # Add ship trajectory
    ax.scatter(ds_ship_hour.lon,
                ds_ship_hour.lat,
                color='white',
                marker='o',
                s=50,
                label='Ship Trajectory',
                transform=ccrs.PlateCarree())
    
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.savefig(f'/work/plots_rain_paper/era5_20200212/{var_name}_{timestamp}_{p_string}.png', dpi=300)


def same_image_seq_as_mp4(out_root, images, day, channel, domain_name, fps=5):
    """
    script to save a sequence of images as an mp4 video file using ffmpeg. it stores
    the mp4 in the same folder as the input images

    Args:
        out_root (string): path where the images are stored
        images (list): list of image filenames to be included in the video.
        day (string): day of the images, used for naming the output file. 
        channel (string): channel name, used for naming the output file.
        domain_name (_typstringe_): name of the domain, used for naming the output file.
        fps (int, optional): . Defaults to 10.
    
    """
    
    # Create symlinked frames for ffmpeg
    temp_dir = os.path.join(out_root, "ffmpeg/")
    os.makedirs(temp_dir, exist_ok=True)
    
    print('temp dir', temp_dir)

    # loop on the images to create symlinks in the temp directory
    for idx, img in enumerate(images):
        src = os.path.join(out_root, img)
        dst = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        if not os.path.exists(dst):
            os.symlink(src, dst)

    mp4_filename = f"{day}_{channel}_quicklook_raw_{domain_name}.mp4"
    mp4_path = os.path.join(out_root, mp4_filename)
    print(f"MP4 output path: {mp4_path}")
    
    if os.path.exists(mp4_path):
        print(f"🟡 MP4 already exists, skipping: {mp4_filename}")
    else:
        print(f"🎬 Creating MP4: {mp4_path}")
        try:
            subprocess.run([
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",         # compression trade-off: slower = smaller file
            "-crf", "32",              # quality vs. file size (23 is default; try 28–30)
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart",
            mp4_path], check=True)
            print(f"✅ MP4 created at: {mp4_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")

    return


def plot_png_series_and_mp4(var_array, ds_era5, ds_ship_hour, var_name, closest_height, p_string, date_string):
    """
    Function to plot vertical velocity or horizontal wind speed at the selected pressure level
    and save the plot as a PNG file and MP4 file. It also plots the ship trajectory on the map.
    Parameters
    ----------
    var_array: nparray containing ERA5 data (lat, lon)
    ds_era5: xarray dataset containing ERA5 data for the selected time with lat lon coordinates for plotting
    ds_ship_hour: xarray dataset containing ship data for the selected hour
    var_name: str, variable name to plot ('w' for vertical velocity or 'Horizontal Wind Speed' for wind speed)
    closest_height: float, height in meters corresponding to the pressure level
    p_string: str, string indicating the pressure level ('surface', 'lcl', 'cloud_top', '500hPa')
    date_string: string for the date in the format 'YYYY-MM-DD'
    """
    
    if var_name == 'w':
        v_min = -0.5
        v_max = 0.5
        cbar_label = 'Vertical Velocity (m/s)'
        cmap_to_plot = cmap_era5
        
    elif var_name == 'Horizontal Wind Speed':
        v_min = 0
        v_max = 20  # adjust as needed for wind speed
        cbar_label = 'Horizontal Wind Speed (m/s)'
        cmap_to_plot = CMAP
        
    elif var_name == 'Horizontal Wind direction':
        v_min = 0
        v_max = 360
        cbar_label = 'Horizontal Wind Direction (°)'
        cmap_to_plot = CMAP
    elif var_name == 'q':
        v_min = 0.00* 1e3
        v_max = 0.010 * 1e3
        cbar_label = 'Specific Humidity (g/kg)'
        cmap_to_plot = CMAP
        
    # plot map of the variable with trajectory of the ship
    # Create a map with Cartopy
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    ax.set_title(f"{timestamp} UTC", fontsize=16)
    
    # Plot the ship trajectory with dots with border but no fill
    

    
    # Plot vertical velocity at the selected pressure level
    sc = ax.pcolormesh(ds_era5.longitude, 
                    ds_era5.latitude, 
                    var_array, 
                    cmap=cmap_to_plot, 
                    shading='auto',
                    vmin=v_min,
                    vmax=v_max,
                    transform=ccrs.PlateCarree())
    

    ax.scatter(ds_ship_hour.lon, 
            ds_ship_hour.lat, 
            color = 'black',
            marker='o', 
            s=50, 
            label='Ship Trajectory')
    
    
    cbar = plt.colorbar(sc, 
                        ax=ax, 
                        orientation='horizontal', 
                        pad=0.1, 
                        aspect=10)
    cbar.set_label(cbar_label, fontsize=20)
    # set fonts of ticks of the axis 
    ax.tick_params(labelsize=18)    
    cbar.ax.tick_params(labelsize=10)
    #ax.legend(loc='upper right', fontsize=18)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f'/work/plots_rain_paper/era5_20200212/{var_name}_{timestamp}_{p_string}.png', dpi=300)
    print(f'Saved plot for {var_name} at {timestamp} and {p_string} pressure as PNG file.')
    # close all figures
    plt.close(fig)
    
    return()

# read era5 
ds_era5 = xr.open_dataset('/work/plots_rain_paper/era5_20200212/589d85aa1852a07ab4af850847f42b71.nc')

# read ship trajectory
ds_ship = read_ship()

# read lcl 
ds_lcl = read_lcl()


# Convert to height (in meters)
pressure_levels = ds_era5.pressure_level.values # in hPa
heights = pressure_to_height_std(pressure_levels * units("hPa"))
print(heights)

# remove units from heights for further processing
heights = 1e3 * heights.magnitude  # convert to numpy array without units

# extract date from ds_era5 valid_time
ds_era5['valid_time'] = pd.to_datetime(ds_era5['valid_time'].values)

# read yy, mm, dd
year = ds_era5.valid_time.dt.year.values[0]
month = ds_era5.valid_time.dt.month.values[0]
day = ds_era5.valid_time.dt.day.values[0]
# print date and define a date string
date_string = f"{year:04d}-{month:02d}-{day:02d}"


# loop on ds_era5 time steps to extract data at ship location
for i_time, timestamp in enumerate(ds_era5.valid_time.values):
    
    #print era5 time stamp
    print(f"Processing time: {timestamp}")
    
    # find all time stamps from the ship data that are within 1 hour of the current era5 time stamp
    era5_time = pd.to_datetime(timestamp) 
    timestamp_end = era5_time + pd.Timedelta(hours=1)

    # selecting ship data in the hour to plot ship trajectory for the hour
    ds_ship_hour = ds_ship.sel(time=slice(era5_time, timestamp_end))
    
    # select heights at which we want to look at the profiles
    
    # 1) surface
    surface_height = 0  # meters, sea level
    surface_pressure = pressure_levels[0]  # assuming first level is surface pressure

    # 2) LCL
    # select and calculate mean lcl in the selected hour of timestamp
    ds_lcl_hour = ds_lcl.sel(time=slice(timestamp, timestamp_end))
    lcl_mean = np.nanmean(ds_lcl_hour.lcl.values) 

    # find height level closest to lcl_mean values
    lcl_height_index = np.abs(heights - lcl_mean).argmin()
    closest_height = heights[lcl_height_index]
    
    # find corresponding pressure level
    lcl_pressure = pressure_levels[lcl_height_index]
    
    # 3) 2000 m (corresponding to mean cloud top height in the area)
    cloud_top_height = 2000  # meters
    # find closest height to cloud top height
    cloud_top_height_index = np.abs(heights - cloud_top_height).argmin()
    closest_cloud_top_height = heights[cloud_top_height_index]
    closest_cloud_top_pressure = pressure_levels[cloud_top_height_index]
    
    # 4) 500 hPa
    pressure_500hPa = 500  # hPa
    # find closest height to 500 hPa
    pressure_500hPa_index = np.abs(pressure_levels - pressure_500hPa).argmin()
    closest_500hPa_height = heights[pressure_500hPa_index]
    closest_500hPa_pressure = pressure_levels[pressure_500hPa_index]
    

    # print pressure of the selected levels
    #print(f"Surface Pressure: {surface_pressure} hPa")
    #print(f"LCL Pressure: {lcl_pressure} hPa")
    #print(f"Cloud Top Pressure: {closest_cloud_top_pressure} hPa")
    #print(f"500 hPa Pressure: {closest_500hPa_pressure} hPa")
    
    # read vertical velocity from Era5 at the selected pressure levels
    p_selected_arr = np.array([surface_pressure, 
                           lcl_pressure, 
                           closest_cloud_top_pressure, 
                           closest_500hPa_pressure])
    
    
    # loop on selected pressure levels to extract and plot variables
    for i_p, pressure in enumerate(p_selected_arr):
        
        if pressure == surface_pressure:
            closest_height = surface_height
            p_string = 'surface'
        elif pressure == lcl_pressure:
            closest_height = closest_height
            p_string = 'lcl'
            
        elif pressure == closest_cloud_top_pressure:
            closest_height = closest_cloud_top_height
            p_string = 'cloud_top'
            
        elif pressure == closest_500hPa_pressure:
            closest_height = closest_500hPa_height
            p_string = '500hPa'
            
        print('printing pressure being processed', p_string, 'at', closest_height, 'm')
        
        # select vertical velocity at the selected pressure levels 
        ds_w_selected = ds_era5.sel(pressure_level=pressure, 
                                 valid_time=timestamp)['w'].squeeze()
    
        # convert vertical velocity to numpy array
        w_selected = ds_w_selected.values
        
        # select u and v wind components at the selected pressure levels
        ds_u_selected = ds_era5.sel(pressure_level=pressure, 
                                    valid_time=timestamp)['u'].squeeze()
        ds_v_selected = ds_era5.sel(pressure_level=pressure,
                                    valid_time=timestamp)['v'].squeeze()
        

        # convert u and v wind components to numpy arrays
        u_selected = ds_u_selected.values
        v_selected = ds_v_selected.values
        
        # calculate horizontal wind speed 
        horizontal_wind_speed = np.sqrt(u_selected**2 + v_selected**2)
        
        # calculate wind direction from u and v
        wind_direction = np.arctan2(v_selected, u_selected) * (180 / np.pi)

        # convert to meteorological wind direction, i.e. direction the wind is coming from, measured clockwise from north
        wind_direction = (wind_direction + 360) % 360  # ensure direction is in [0, 360) degrees
        
        # adding variables to ds_era5 for plotting
        # create a dataset with the selected variables
        ds_selected = xr.Dataset({
            'w': (('latitude', 'longitude'), w_selected),
            'u': (('latitude', 'longitude'), u_selected),
            'v': (('latitude', 'longitude'), v_selected),
            'horizontal_wind_speed': (('latitude', 'longitude'), horizontal_wind_speed),
            'wind_direction': (('latitude', 'longitude'), wind_direction)
        }, coords={
            'latitude': ds_era5.latitude,
            'longitude': ds_era5.longitude
        })
        
        # crop the dataset to the area of interest 
        # ax.set_extent([-60., -55.5, 12, 15], crs=ccrs.PlateCarree())
        ds_crop = ds_selected.where((ds_selected.latitude >= 12) & (ds_selected.latitude <= 15) &
                                        (ds_selected.longitude >= -60) & (ds_selected.longitude <= -55.5), drop=True)
        #plot_h_wind_speed_dir(ds_era5,
        #                      ds_crop,
        #                     ds_ship_hour,
        #                      p_string,
        ##                      timestamp, 
        ##                     date_string,
        #                      "Horizontal Wind Speed")
        

        # call function to plot vertical velocity variables at the selected pressure level
        # input par: var_array, ds_era5, ds_ship_hour, var_name, closest_height, p_string
        #plot_png_series_and_mp4(w_selected,
        #                        ds_era5, 
        #                        ds_ship_hour, 
        #                        'w', 
        #                        closest_height, 
        #                        p_string, 
        #                        date_string)
        
        # call function to plot horizontal wind speed at the selected pressure level
        #plot_png_series_and_mp4(horizontal_wind_speed, 
        #                        ds_era5, 
        #                        ds_ship_hour, 
        #                        'Horizontal Wind Speed', 
        ##                        closest_height, 
         #                       p_string, 
         #                       date_string)
    
    
    # define pressures for humidity levels
     # 1) pressure heights closest to 2500 m 
    height_level_1 = 2500  # meters
    # find closest height to 2500 m
    height_level_1_index = np.abs(heights - height_level_1).argmin()
    closest_height_1 = heights[height_level_1_index]
    closest_pressure_1 = pressure_levels[height_level_1_index]
    
    # 2) pressure heights closest to 3000 m
    height_level_2 = 3000  # meters
    # find closest height to 3000 m
    height_level_2_index = np.abs(heights - height_level_2).argmin()
    closest_height_2 = heights[height_level_2_index]
    closest_pressure_2 = pressure_levels[height_level_2_index]
    
    # 3) pressure heights closest to 4000 m
    height_level_3 = 4000  # meters
    # find closest height to 4000 m
    height_level_3_index = np.abs(heights - height_level_3).argmin()
    closest_height_3 = heights[height_level_3_index]
    closest_pressure_3 = pressure_levels[height_level_3_index]
    
    # loop on pressure levels for specific humidity
    pressure_hum = np.array([closest_pressure_1, closest_pressure_2, closest_pressure_3])
    # loop on pressure levels to extract and plot specific humidity
    for i_p, pressure in enumerate(pressure_hum):
        
        if pressure == closest_pressure_1:
            closest_height = closest_height_1
            p_string = '2500m'
        elif pressure == closest_pressure_2:
            closest_height = closest_height_2
            p_string = '3000m'
        elif pressure == closest_pressure_3:
            closest_height = closest_height_3
            p_string = '4000m'
        
        print('printing specific humidity at', p_string, 'at', closest_height, 'm')

        # select specific humidity values at the selected pressure levels
        ds_q_selected = ds_era5.sel(pressure_level=pressure,
                                    valid_time=timestamp)['q'].squeeze()
        
        q_selected = ds_q_selected.values * 1e3  # convert to g/kg
         
        # select u and v wind components at the selected pressure levels
        ds_u_selected = ds_era5.sel(pressure_level=pressure, 
                                    valid_time=timestamp)['u'].squeeze()
        ds_v_selected = ds_era5.sel(pressure_level=pressure,
                                    valid_time=timestamp)['v'].squeeze()
        

        # convert u and v wind components to numpy arrays
        u_selected = ds_u_selected.values
        v_selected = ds_v_selected.values
        
        # calculate horizontal wind speed 
        horizontal_wind_speed = np.sqrt(u_selected**2 + v_selected**2)
        
        # calculate wind direction from u and v
        wind_direction = np.arctan2(v_selected, u_selected) * (180 / np.pi)

        # convert to meteorological wind direction, i.e. direction the wind is coming from, measured clockwise from north
        wind_direction = (wind_direction + 360) % 360  # ensure direction is in [0, 360) degrees
        
        # select vertical velocity at the selected pressure levels 
        ds_w_selected = ds_era5.sel(pressure_level=pressure, 
                                 valid_time=timestamp)['w'].squeeze()
    
        # convert vertical velocity to numpy array
        w_selected = ds_w_selected.values
        
        
        # adding variables to ds_era5 for plotting
        # create a dataset with the selected variables
        ds_selected = xr.Dataset({
            'q': (('latitude', 'longitude'), q_selected),
            'u': (('latitude', 'longitude'), u_selected),
            'v': (('latitude', 'longitude'), v_selected),
            'w': (('latitude', 'longitude'), w_selected),
            'horizontal_wind_speed': (('latitude', 'longitude'), horizontal_wind_speed)}, 
        coords={
            'latitude': ds_era5.latitude,
            'longitude': ds_era5.longitude
        })
        
        
        plot_png_q_wind_dir(ds_selected, 
                                ds_ship_hour, 
                                p_string, 
                                date_string)
        
        plot_png_w_wind_dir(ds_selected, 
                                ds_ship_hour, 
                                p_string, 
                                date_string)
                
    # create mp4 files for the png series
    #out_root = '/work/plots_rain_paper/era5_20200212/'
    # list images for each variable and pressure level
    #vars = ['w', 'Horizontal Wind Speed', 'Horizontal Wind direction']
    #vars = ['Horizontal Wind Speed']
    # loop on variables to create mp4 files
    #for i, var_name in enumerate(vars):
    #    for i_p, p_string in enumerate(['surface', 'lcl', 'cloud_top', '500hPa']):
        
            # find all images for the current variable and time stamp
    #        images = np.sort(glob(os.path.join(out_root, f'{var_name}*{p_string}.png')))
            
            # create mp4 file from the images
    #        same_image_seq_as_mp4(out_root, 
    #                            images, 
    #                            date_string, 
    #                            var_name+'_'+p_string, 
    #                            'era5_20200212')
      