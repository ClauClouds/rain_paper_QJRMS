"""
code to produce the fig04 of the final version of the paper.
the figure focuses on the cloud formation stage and development
showing what happens in the subcloud layer in terms of anomalies
and what happens in the cloud microphysics 

subplots: 
1) first row: diurnal cycle of LCL
2) second row: 3 subpanel with anomalies of vertical velocity, specific humidity and virtual pot temp
3) third row: 2 subpanels with vd vz Ze and sk vs Ze
"""

from readers.lidars import read_anomalies
from readers.cloudtypes import read_cloud_class, read_rain_ground
from cloudtypes.path_folders import path_diurnal_cycle_arthus, path_paper_plots
from readers.lcl import read_lcl, read_diurnal_cycle_lcl
from fig03_diurnal_all import calc_lcl_grid, calc_diurnal_lcl
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
#from dotenv import load_dotenv
from mpl_style import CMAP, COLOR_CONGESTUS, COLOR_SHALLOW
from dask.diagnostics import ProgressBar
ProgressBar().register()

def main():
    
    # first row of plot:  lcl diurnal cycle for the plot
    lcl_dc = read_diurnal_cycle_lcl(path_diurnal_cycle_arthus)
    
    # read cloud types
    ds_ct = read_cloud_class()

    # read rain at the ground flag
    ds_r = read_rain_ground()
    
    # align cloud type flag and rain ground flag
    ds_ct, ds_r = xr.align(ds_ct, ds_r, join="inner")
    ds_cp = xr.merge([ds_ct, ds_r])

    # second row of plots: read lidar data
    # prepare data (calculate mean and std of the profiles for shallow/congestus in prec and non prec)
    dct_vw_q = prepare_anomaly_profiles(ds_cp, "VW", lcl_dc)

    
    # second row of plots: derive specific humidity and virtual potential temperature from lidar data
    #ds_therm = calc_thermo_prop()
    
    
    #plot_figure(lcl_dc, dct_vw_q)
    
    
            
if __name__ == "__main__":
    main()



    
def prepare_anomaly_profiles(ds_cp, var_string, ds_lcl):
    """
    function to calculate mean and std of shallow and congestus profiles for prec and non prec conditiosn

    Args:
        cloud_properties (xarray dataset): contains cloud classification
        rain_flags (xarray dataset): contains rainy flags and 
        
    Dependencies: 
    - read_anomalies function
    - calc_quantiles
    - calc_lcl_grid
    - read_lcl
    
    """
    
    # reading arthus data anomalies
    ds_an = read_anomalies(var_string)

    # interpolate classification of clouds over anomalies time stamps
    class_interp = ds_cp.interp(time=ds_an.time.values, method='nearest')                                                           
                                                                    
    # merging datasets obtained (classification and anomalies)
    merian_dataset = xr.merge([class_interp, ds_an])  

    
    # selecting cloud types and rain/norain conditions
    is_shallow = ds_cp.shape == 0
    is_congestus = ds_cp.shape == 1

    # selecting prec and non prec
    is_prec_ground = ds_cp.flag_rain_ground == 1
    
    # defining classes 
    is_cg_prec = is_congestus & is_prec_ground
    is_cg_non_prec = is_congestus & ~is_prec_ground 
    is_sl_prec = is_shallow & is_prec_ground
    is_sl_non_prec = is_shallow & ~is_prec_ground 

    # segregating with respect to shallow and deep clouds
    ds_sl_prec = merian_dataset.isel(time=is_sl_prec)
    ds_sl_nonprec = merian_dataset.isel(time=is_sl_non_prec)
    ds_cg_prec = merian_dataset.isel(time=is_cg_prec)
    ds_cg_nonprec = merian_dataset.isel(time=is_cg_non_prec)
    
    # regridding data to height grid referred to lcl
    # read lcl and calculate its diurnal cycle at 15 mins and at 30 mins (for fluxes)
    ds_lcl = read_lcl()
    da_sl_prec = calc_lcl_grid(ds_sl_prec, ds_lcl, 'height', 'time', 'anomaly')
    print(da_sl_prec) 
    strasuka
    # calculating percentiles of 25, 50, and 75 for vertical wind
    q_stats = {}
    q_stats["q_sl_prec"] = (calc_percentiles(ds_sl_prec))
    q_stats["q_sl_nonprec"] = (calc_percentiles(ds_sl_nonprec))
    q_stats["q_cg_prec"] = (calc_percentiles(ds_cg_prec))
    q_stats["ds_cg_nonprec"] = (calc_percentiles(ds_cg_nonprec))
    
    return q_stats
    
def calc_percentiles(ds, percentiles=[25, 50, 75]):
    """_summary_

    Args:
        ds (_type_): _description_
        percentiles (list, optional): _description_. Defaults to [0.25, 5, 0.75].
    """
    height = ds.height.values
    q = np.zeros((len(percentiles), len(height)))
    for i_h in range(len(height)):
        ds_h = ds.isel(height=i_h)
        an_s = ds_h.VW_anomaly.values.flatten()
        q[:,i_h] = np.nanpercentile(an_s, percentiles)
        
    return(q)
    
    
def f_calcThermodynamics(P,Q,T, LTS, time, height, Hsurf, date):
    """ 
    author: claudia Acquistapace
    date; 25 July 2019 (heat wave in Cologne)
    contact: cacquist@meteo.uni-koeln.de
    goal: derive thermodinamic quantities of interest for the analysis: 
     output: dictionary containing the following variables: 
                    'mixingRatio':r, 
                    'relativeHumidity':rh, 
                    'virtualTemperature':tv,
                    'cclHeight':result_ccl['z_ccl'],
                    'cclTemperature':result_ccl['T_ccl'],
                    'lclHeight':lclArray,
                    'surfaceTemperature':TSurf, 
                    'virtualPotentialTemperature':Theta_v,
                    'time':time, 
                    'height':height,
                    'LTS':LTS,
     input:         matrices of Pressure (pa), 
                    temperature (K), 
                    absolute humidity (Kg/Kg), 
                    time, 
                    height, 
                    height of the surface
    """
    r  = np.zeros((len(time), len(height)))
    rh = np.zeros((len(time), len(height)))
    tv = np.zeros((len(time), len(height)))
        
    # calculation of mixing ratio and relative humidity
    T0 = 273.15
    for iTime in range(len(time)):
        for iHeight in range(len(height)):
            r[iTime,iHeight]  = (Q[iTime, iHeight])/(1-Q[iTime, iHeight])
            rh[iTime,iHeight] = 0.263*P[iTime, iHeight] * \
            Q[iTime, iHeight] * (np.exp( 17.67 * (T[iTime, iHeight]-T0) / (T[iTime, iHeight] - 29.65)))**(-1)

    #print('RH', rh[0,0], rh[0,-1])
    #print('pressure ' , P[0,0]*0.001, P[0,-1]*0.001)
    #print('temp', T[0,0], T[0,-1])


    # calculation of virtual temperature
    for indH in range(len(height)-1, 0, -1):
        tv[:,indH] = T[:,indH]*(1+0.608 * Q[:,indH])
            
        
    indSurf = len(height)-1
    PSurf    = P[:,indSurf]
    TSurf    = T[:,indSurf]         # T in K
    rhSurf   = rh[:,indSurf-1]
    lclArray = []

    for iTime in range(len(time)):
        lclArray.append(lcl(PSurf[iTime],TSurf[iTime],rhSurf[iTime]/100.))
    
    # calculation of potential and virtual potential temperature (P in pascal)
    Rd = 287.058  # gas constant for dry air [Kg-1 K-1 J]
    Cp = 1004.
    Theta = np.zeros((len(time), len(height)))
    Theta_v = np.zeros((len(time), len(height)))
    for indTime in range(len(time)):
        for indHeight in range(len(height)):
            k_val = Rd*(1-0.23*r[indTime, indHeight])/Cp
            Theta_v[indTime, indHeight] = ( (1 + 0.61 * r[indTime, indHeight]) * \
                   T[indTime, indHeight] * (100000./P[indTime, indHeight])**k_val)
            Theta[indTime, indHeight]   = T[indTime, indHeight] * (100000./P[indTime, indHeight])**k_val
    
    
    ThermodynPar={'mixingRatio':r, 
                  'relativeHumidity':rh, 
                  'virtualTemperature':tv,
                  'cclHeight':result_ccl['z_ccl'],
                  'cclTemperature':result_ccl['T_ground_ccl'],
                  'lclHeight':lclArray,
                  'surfaceTemperature':TSurf, 
                  'virtualPotentialTemperature':Theta_v,
                  'potentialTemperature':Theta,
                  'time':time, 
                  'height':height,
                  'LTS':LTS,
                  }
    
    return(ThermodynPar)



def plot_figure(lcl_dc, dct_vw_q):
    
    
    
    fig, axs = plt.subplots(3, 3, figsize=(25,25), sharey=True, constrained_layout=True)# 