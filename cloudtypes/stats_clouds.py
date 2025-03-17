"""code to derive statistics of cloud types from the cloud mask
Mainly the following ones: 
- how much prec congestus clouds occur in southern, transition and northern regions
- mean LWP value for  congestus/  shallow 
- % of prec congestus over cong, % of shallow congestus over shallow"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from readers.radar import read_lwp
from readers.cloudtypes import read_cloud_class, read_rain_flags

def main():
    
    
    # read lwp
    lwp_ds = read_lwp()
        
    # read rain flags
    rainflags = read_rain_flags()
        
    #derive flags:
    combiprecipflag = (rainflags.flag_rain_ground.values == 1) | (rainflags.flag_rain.values == 1) #either ground_rain or rain
    cloudyflag = ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 0)) #everything thats cloudy in ground_rain and rain
    clearflag = (combiprecipflag == 0) & (cloudyflag == 0) #everything thats not cloudy, not rainy
    virgaflag =  ((rainflags.flag_rain_ground.values == 0) & (rainflags.flag_rain.values == 1)) 
    #apply Clau's rain filters: conservative, ie flag both rain_ground and rain with nan in lwp dataset
    lwp_ds.lwp.values[combiprecipflag] = np.nan
    
    # read cloud classification
    cloudclassdata = read_cloud_class()
    
    print(cloudclassdata)
    
    # interpolate classification on LWP time stamps and derive flags for shallow/congestus data
    cloudclassdataip = cloudclassdata.interp(time=lwp_ds.time)
    shallow = (cloudclassdataip.shape.values == 0) 
    congestus = (cloudclassdataip.shape.values == 1) 
    
    # calculate number of congestus time stamps
    congestus_count = np.sum(congestus)
    
    # calculate number of shallow time stamps
    shallow_count = np.sum(shallow)
    
    # find number of congestus precipitation time stamps
    congestus_precip = congestus & combiprecipflag
    congestus_precip_count = np.sum(congestus_precip)
    
    # find number of shallow precipitation time stamps
    shallow_precip = shallow & combiprecipflag
    shallow_precip_count = np.sum(shallow_precip)
    
    print('congestus count', congestus_count)
    print('shallow count', shallow_count)
    print('congestus precip count', congestus_precip_count)
    print('shallow precip count', shallow_precip_count)
    
    # calculate percentage of congestus precipitation
    congestus_precip_percentage = congestus_precip_count / congestus_count * 100
    # calculate percentage of shallow precipitation
    shallow_precip_percentage = shallow_precip_count / shallow_count * 100
    
    print('congestus precip percentage', congestus_precip_percentage)
    print('shallow precip percentage', shallow_precip_percentage)
    
    # calculating number of virga in congestus
    congestus_virga = congestus & virgaflag
    congestus_virga_count = np.sum(congestus_virga)
    
    # percentage of congestus virga
    congestus_virga_percentage = congestus_virga_count / congestus_count * 100
    
    # calculating number of virga in shallow
    shallow_virga = shallow & virgaflag
    shallow_virga_count = np.sum(shallow_virga)
    
    # percentage of shallow virga
    shallow_virga_percentage = shallow_virga_count / shallow_count * 100
    
    print('congestus virga percentage', congestus_virga_percentage)
    print('shallow virga percentage', shallow_virga_percentage)
    
    # calculate mean lwp for shallow
    lwp_shallow = lwp_ds.lwp.values[shallow]
    lwp_shallow_mean = np.nanmean(lwp_shallow)
    
    # calculate mean lwp for congestus
    lwp_congestus = lwp_ds.lwp.values[congestus]
    lwp_congestus_mean = np.nanmean(lwp_congestus)
    
    print('shallow mean lwp', lwp_shallow_mean)
    print('congestus mean lwp', lwp_congestus_mean)
    
    # find percentage of congestus over all precipitating time stamps
    prec_count = np.sum(combiprecipflag)
    congestus_prec_percentage = congestus_precip_count / prec_count * 100
    print('congestus precip percentage', congestus_prec_percentage)
    
    # find percentage of shallow over all precipitating time stamps
    shallow_prec_percentage = shallow_precip_count / prec_count * 100
    print('shallow precip percentage', shallow_prec_percentage)
    
    # find where the precipitating congestus clouds occur
    # find northern, southern, transition regions
    #southern region
    ind_southern = np.where(lwp_ds.lat < 10.5)[0]
    # transition region
    ind_transition = np.where((lwp_ds.lat <= 12.5) & (lwp_ds.lat >= 10.5))[0]
    # northern region
    ind_northern = np.where(lwp_ds.lat > 12.5)[0]
    
    # count number of congestus in the region
    congestus_southern = congestus[ind_southern]
    congestus_transition = congestus[ind_transition]
    congestus_northern = congestus[ind_northern]
    
    congestus_southern_count = np.sum(congestus_southern)
    congestus_transition_count = np.sum(congestus_transition)
    congestus_northern_count = np.sum(congestus_northern)
    
    print('congestus southern count', congestus_southern_count)
    print('congestus transition count', congestus_transition_count)
    print('congestus northern count', congestus_northern_count)
    
    # print percentage of congestus in regions
    print('congestus southern percentage', congestus_southern_count / congestus_count * 100)
    print('congestus transition percentage', congestus_transition_count / congestus_count * 100)
    print('congestus northern percentage', congestus_northern_count / congestus_count * 100)
    
    # count number of shallow in the regions
    shallow_southern = shallow[ind_southern]
    shallow_transition = shallow[ind_transition]
    shallow_northern = shallow[ind_northern]
    
    shallow_southern_count = np.sum(shallow_southern)
    shallow_transition_count = np.sum(shallow_transition)
    shallow_northern_count = np.sum(shallow_northern)
    
    print('shallow southern count', shallow_southern_count)
    print('shallow transition count', shallow_transition_count)
    print('shallow northern count', shallow_northern_count)
    
    # print percentage of shallow in regions
    print('shallow southern percentage', shallow_southern_count / shallow_count * 100)
    print('shallow transition percentage', shallow_transition_count / shallow_count * 100)
    print('shallow northern percentage', shallow_northern_count / shallow_count * 100)
    
    # print percentage of congestus precipitating in the regions
    congestus_precip_southern = congestus_precip[ind_southern]
    congestus_precip_transition = congestus_precip[ind_transition]
    congestus_precip_northern = congestus_precip[ind_northern]
    
    congestus_precip_southern_count = np.sum(congestus_precip_southern)
    congestus_precip_transition_count = np.sum(congestus_precip_transition)
    congestus_precip_northern_count = np.sum(congestus_precip_northern)
    
    print('congestus precip southern count', congestus_precip_southern_count)
    print('congestus precip transition count', congestus_precip_transition_count)
    print('congestus precip northern count', congestus_precip_northern_count)
    
    print('congestus precip southern percentage', congestus_precip_southern_count / congestus_precip_count * 100)
    print('congestus precip transition percentage', congestus_precip_transition_count / congestus_precip_count * 100)
    print('congestus precip northern percentage', congestus_precip_northern_count / congestus_precip_count * 100)
    
    
if __name__ == "__main__":
    main()
