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
from readers.cloudtypes import read_cloud_class, read_rain_ground, read_cloud_base, read_in_clouds_radar_moments
import pdb
import os
from fig10_sounding_q_sigmae import calc_paluch_quantities
import matplotlib.colors as mcolors
from metpy.calc import equivalent_potential_temperature
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW, cmap_rs_congestus, cmap_rs_shallow


def main():
    
    # read cloud types (shallow and congestus)
    ds_ct = read_cloud_class()

    # read rain at the ground flag 
    ds_r = read_rain_ground()
    
    # read cloud base heights 
    ds_cb = read_cloud_base()
    
    
    
if __name__ == "__main__":
    main()
    