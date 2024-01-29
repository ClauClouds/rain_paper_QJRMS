"""
reader for satellite data

"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
import matplotlib as mpl
import os.path
import itertools
import os.path


# dictionary containing image filename and classification for the selected case study of the paper
dic = {
    '20200431100041_2348_ship.tif': 5,
    '20200431130041_2350_ship.tif': 5,
    '20200431130041_2350_ship.tif': 5,
    '20200431140041_2351_ship.tif': 5,
    '20200431200041_2352_ship.tif': 5,
    '20200431210041_2353_ship.tif': 5,
    '20200431230041_2354_ship.tif': 5,
    '20200431240040_2355_ship.tif': 5,
    '20200431300040_2356_ship.tif': 5,
    '20200431310040_2357_ship.tif': 5,
    '20200431340040_2359_ship.tif': 5,
    '20200431400040_2360_ship.tif': 5,
    '20200431410040_2361_ship.tif': 5,
    '20200431430040_2362_ship.tif': 5,
    '20200431440039_2363_ship.tif': 5,
    '20200431500039_2364_ship.tif': 2,
    '20200431510039_2365_ship.tif': 2,
    '20200431530039_2366_ship.tif': 2,
    '20200431540039_2367_ship.tif': 2,
    '20200431600039_2368_ship.tif': 2,
    '20200431610039_2369_ship.tif': 2,
    '20200431630039_2370_ship.tif': 2,
    '20200431640038_2371_ship.tif': 2,
    '20200431700038_2372_ship.tif': 2,
    '20200431710036_2373_ship.tif': 2,
    '20200431730036_2374_ship.tif': 2,
    '20200431740036_2375_ship.tif': 2,
    '20200431800035_2376_ship.tif': 2,
    '20200431810035_2377_ship.tif': 2,
    '20200431830035_2378_ship.tif': 2,
    '20200431840035_2379_ship.tif': 2,
    '20200431900035_2380_ship.tif': 2,
    '20200431910035_2381_ship.tif': 2,
    '20200431930035_2382_ship.tif': 2,
    '20200431940035_2383_ship.tif': 5,
    '20200432000035_2384_ship.tif': 5,
    '20200432010035_2385_ship.tif': 5,
    '20200432030035_2386_ship.tif': 5,
    '20200432040035_2387_ship.tif': 5,
}


def nc_fday_of_year(image_files):
    '''
       takes the rwa nc goes file as input and 
       returns details of the date from file addresss
    '''
    date = str(image_files)  # may have to be changed
    date = datetime.strptime(date, '%Y%j%H%M%S%f')
    day_of_year = date.timetuple().tm_yday
    year = date.timetuple().tm_year
    month = date.timetuple().tm_mon
    dayofmonth = date.timetuple().tm_mday
    hour = date.timetuple().tm_hour
    minute = date.timetuple().tm_min
    
    time_stamp = datetime(year, month, dayofmonth, hour, minute)
    
    #str(year)+'_'+str(month)+'_'+str(dayofmonth)+'_'+str(hour)+'_'+str(minute)
    return time_stamp


def read_satellite_classification():
    """
    function to read satellite optical thickness images and their classification in a xarray dataset
    """
    
    file_names = list(dic.keys())
    images_class = list(dic.values())
    time_array = []
    file_name_list = []
    
    # loop on all files of the list for assigning path and classification 
    for i_file, filename in enumerate(file_names):
        time_array.append(nc_fday_of_year(filename[:-len('_2387_ship.tif')]))
        file_name_list.append(filename[:-len('_ship.tif')])
        
    # creating xarray dataset of the variables
    data_vars = {'file_names':(['time'], np.asarray(file_name_list), 
                            {'units': '', 
                            'long_name':'filename of satellite images'}),
                'classification':(['time'], np.asarray(images_class), 
                            {'units': '2:fish, 5:between flower and fish', 
                            'long_name':'satellite image classification based on DC algorithm'}), 
                }

    # define coordinates
    coords = {'time': (['time'], time_array)}

    # define global attributes
    attrs = {'creation_date':str(datetime.now()), 
            'author':'Claudia Acquistapace', 
            'email':'cacquist@uni-koeln.de'}

    # create dataset
    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)
    
    return ds