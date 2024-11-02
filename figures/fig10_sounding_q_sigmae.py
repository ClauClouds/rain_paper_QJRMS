'''
code to produce a plot of specific humidity on y axis and equivalent potential temperature on 
x axis. 

'''
from readers.soundings import read_merian_soundings

import numpy as np
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
from glob import glob
import metpy.calc as mpcalc 
from metpy.calc import equivalent_potential_temperature
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

def main():
    
    # read radiosounding data
    ds = read_merian_soundings()
    
    # set date for searching for radiosondes
    date = '20200212'
    sound1 = 'MS-Merian__ascent__13.83_-57.20__202002121415'
    sound2 = 'MS-Merian__ascent__13.52_-57.40__202002121855'
    
    ds_s1 = ds.sel(sounding=sound1)
    ds_s2 = ds.sel(sounding=sound2)
    theta_e_1, q_1 = calc_paluch_quantities(ds_s1, '202002121415')
    theta_e_2, q_2 = calc_paluch_quantities(ds_s2, '202002121855')
    
    print(np.nanmax(ds_s1.alt.values), np.nanmin(ds_s1.alt.values))
    
    
    
    # reading heights of the soundings
    z_1 = ds_s1['alt'].values
    z_2 = ds_s2['alt'].values

    # define colorbars for sounding 1 and sounding 2
    cmap_s1 = mpl.cm.get_cmap('viridis')
    norm_s1 = mpl.colors.Normalize(vmin=0, vmax=4000.)
    cmap_s1.set_under('white')
    
    # plot figure Paluch plot
    fig, ax = plt.subplots()
    ax.invert_yaxis()  # Invert y-axis to decrease from top to bottom
    ax.scatter(theta_e_1, 
               q_1,
               c=z_1, 
               marker='o',
               cmap=cmap_s1,
               norm=norm_s1, 
               edgecolors='black',
               label='14:15 LT')
    ax.scatter(theta_e_2, 
               q_2, 
               c=z_2, 
               marker='^',
               edgecolors='grey',
               cmap=cmap_s1, 
               norm=norm_s1, 
               label='18:55 LT')
    
    # plot colorbars
    cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_s1,
                                               cmap=cmap_s1), 
                         ax=ax, 
                         orientation='vertical', 
                         label='Altitude (m)')

    
    ax.set_xlabel('Equivalent potential temperature (K)')
    ax.set_ylabel('Specific humidity (g/kg)')
    ax.legend()
    ax.set_xlim(310, 350)
    ax.set_ylim(16,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    fig.savefig('/net/ostro/plots_rain_paper/fig10_paluch_'+date+'.png')


def calc_paluch_quantities(ds, date=202002121415):
    """
    function to calculate the quantities needed for the Paluch plot
    

    Args:
        ds (xarray dataset): dataset from one radiosonde

    Returns:
        theta_e: array
        q: array
        
    """
    
    p = ds['p'].values # in Pa
    T = ds['ta'].values # in K
    Td = ds['dp'].values # in K
    
    # convert in the right units
    # convert p from Pa to hpa
    p = p/100
    # convert T from K to C
    T = T - 273.15
    # convert Td from K to C
    Td = Td - 273.15

    # calculate equivalent potential temperature
    theta_e = equivalent_potential_temperature(p * units.hPa, T * units.degC, Td * units.degC)
    
    # calculate specific humidity
    q = specific_humidity_from_dewpoint(p * units.hPa, Td * units.degC).to('g/kg')
    
    
    # plot the input of the function
    fig, ax = plt.subplots()
    ax.plot(T, p, label='Temperature')
    ax.plot(Td, p, label='Dewpoint')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Pressure (hPa)')
    ax.legend()
    fig.savefig('/net/ostro/plots_rain_paper/profiles_rs_'+date+'.png')
    
    
    return theta_e, q
    




if __name__ == "__main__":
    main()
