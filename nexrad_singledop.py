# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:55:42 2020

@author: sambit
"""

from warnings import warn
import numpy as np
import os
import glob
import wradlib as wrl
import pyart
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn import mixture
import pandas as pd
from matplotlib import colors
import matplotlib as mpl
import random
from matplotlib.patches import Ellipse
import datetime
import singledop
import numpy as np
import netCDF4
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

############################ Functions ###############################
# """
# Simulated WSW Winds
# Let's get started with some simulated data. We presume a strong WSW flow field (U = 20 and V = 1 m/s). 
# The decorrelation length scale, L, is set to 60 km. Gaussian noise in the simulated radar system sampling 
# these winds is enabled. Additionally, we assume that the radar is only performing 180-deg PPI sectors, 
# and data are masked within 10 km. We accept the SingleDop default that the analysis will only extend 
# out to +/- 60 km from the radar and the resolution of the analysis will be 1 km.
# """

# sd_test = singledop.SingleDoppler2D(range_limits=[10, 100], azimuth_limits=[180, 360], 
                                    # L=60.0, U=20.0, V=1.0, noise=True)
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# display = singledop.AnalysisDisplay(sd_test)
# display.plot_velocity_vectors(title='Mostly Westerly Flow', thin=6, scale=600, legend=20)

# fig = plt.figure(figsize=(11, 4))
# ax1 = fig.add_subplot(121)
# display.plot_velocity_contours('vr', cmap='PiYG', 
                               # title='(a) Radial Velocity', mesh_flag=True)
# ax2 = fig.add_subplot(122)
# display.plot_velocity_contours('vt', title='(b) Tangential Velocity')

# display.plot_radial_tangential_contours(mesh_flag=True, cmap='rainbow')

# """

# In this analysis, note how the western sector is different than the eastern sector, 
# despite the uniform wind field. This is the result of our masking of the simulated radar data using the 
# azimuth_limits and range_limits keywords in the original SingleDoppler2D call.
# """

# # More Complex Wind Fields
# #OK, uniform wind fields are kind of boring. What if we had a frontal feature?
# gs = 121
# xgrid = np.arange(gs) - 60.0 
# ygrid = np.arange(gs) - 60.0
# x, y = np.meshgrid(xgrid, ygrid)
# cond = y > 0.75 * x + 30
# Uin = 0.0 + np.zeros((gs, gs), dtype='float')
# Vin = 20.0 + np.zeros((gs, gs), dtype='float')
# Uin[cond] = 15.0
# Vin[cond] = -15.0

# fig = plt.figure(figsize=(5, 5))
# cond = np.logical_and(x % 4 == 0, y % 4 == 0)
# ax = fig.add_subplot(111)
# ax.quiver(x[cond], y[cond], Uin[cond], Vin[cond], scale=600)
# """
# SingleDoppler2D also can take arrays for U and V, so let's do that but now assume the radar 
# is not masked at all (i.e., full 360-deg surveillance). We alter L and a few other parameters too.
# """
# sd_test2 = singledop.SingleDoppler2D(range_limits=[0, 90], azimuth_limits=[0, 360], 
                                     # L=30.0, xgrid=np.arange(gs)-60.0, noise=True,
                                     # ygrid=np.arange(gs)-60.0, U=Uin, V=Vin)


# display = singledop.AnalysisDisplay(sd_test2)
# display.plot_radial_tangential_contours(levels=-50.0+4.0*np.arange(25), mesh_flag=True)

# fig = plt.figure(figsize=(5, 5))
# display.plot_velocity_vectors(legend=20)

"""
The analysis is not perfect, but the result is consistent with Xu et al. (2006) Figure 4, 
from which this example from adapted. The analysis successfully represents the presence of the 2D front, 
despite the use of a single simulated radar. So the module is working as intended.
"""

# Real Radar Data
#sounding = np.loadtxt('../sounding/30112017_sounding.txt',skiprows=4)
 
def radar_coords_to_cart(rng, az, ele, debug=False):
    """
    TJL - taken from old Py-ART version
    Calculate Cartesian coordinate from radar coordinates
    Parameters
    ----------
    rng : array
        Distances to the center of the radar gates (bins) in kilometers.
    az : array
        Azimuth angle of the radar in degrees.
    ele : array
        Elevation angle of the radar in degrees.
    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:\\theta_a is the azimuth angle, :math:\\theta_e is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = ele * np.pi / 180.0  # elevation angle in radians.
    theta_a = az * np.pi / 180.0  # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    r = rng * 1000.0  # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z

def adjust_fhc_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def adjust_meth_colorbar_for_pyart(cb, tropical=False):
    if not tropical:
        cb.set_ticks(np.arange(1.25, 5, 0.833))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z)', 'R(Zrain)'])
    else:
        cb.set_ticks(np.arange(1.3, 6, 0.85))
        cb.ax.set_yticklabels(['R(Kdp, Zdr)', 'R(Kdp)', 'R(Z, Zdr)', 'R(Z_all)', 'R(Z_c)', 'R(Z_s)'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def convert_to_pyart(filename1,filename2):
    # We will read the nc data with netCDF4.Dataset
#    start_time1 =  datetime.datetime.now()
    print("Running Hydroclassification code for TERLS DWR!!! \n")              
    print("File: %s" %(sorted_zfiles[i]))
    data1 = wrl.io.read_generic_netcdf('%s\%s' %(zloc,filename1))
#    data2 = wrl.io.read_generic_netcdf('%s\%s' %(zdrloc,filename2))
#    data3 = wrl.io.read_generic_netcdf('%s\%s' %(kdploc,filename3))
#    data4 = wrl.io.read_generic_netcdf('%s\%s' %(rholoc,filename4))
    data2 = wrl.io.read_generic_netcdf('%s\%s' %(velloc,filename2))
    z = data1['variables']['BaseReflectivityDR']['data']
#    zdr = data2['variables']['zdr']['data'][0]
#    kdp = data3['variables']['kdp']['data'][0]
#    rho = data4['variables']['cc']['data'][0]
    vel = data2['variables']['BaseVelocityDV']['data']
#    data = netCDF4.Dataset(filename)
    # Lets get an idea of the shapes for rays and gates and the keys in this dataset.
#    data.variables.keys()
    tstring = filename1[9:-3]
    tstart = tstring[0:4]+'-'+tstring[5:7]+'-'+tstring[8:10]+'T'+tstring[11:13]+':'+tstring[14:16]+':'+tstring[17:19]+'Z'
    print('azimuth: ', data1['dimensions']['azimuth']['size'])
    print('range: ', data1['dimensions']['gate']['size'])
    print('elevation: ', data1['variables']['elevation']['data'][0])
    # Make a empty radar with the dimensions of the dataset.
    radar = pyart.testing.make_empty_ppi_radar(data1['dimensions']['gate']['size'],  data1['dimensions']['azimuth']['size'], 1)
    radar.metadata = {'instrument_name': 'NEXRAD Houston (HGX)'}
    # Start filling the radar attributes with variables in the dataset.
    radar.time['data'] = data1['variables']['rays_time']['data']
    radar.time['units'] = 'milliseconds since 1970-01-01 00:00 UTC' 
    # Fill the radar coordinates
    radar.latitude['data'] = data1['RadarLatitude']
    radar.longitude['data'] = data1['RadarLongitude']
    radar.altitude['data'] = data1['RadarAltitude']
    radar.range['data'] = data1['variables']['gate']['data']
    radar.fixed_angle['data'] = data1['variables']['elevation']['data'][0]
    radar.sweep_number['data'] = data1['ElevationNumber']
#    radar.sweep_start_ray_index['data'] = np.array(0)
#    radar.sweep_end_ray_index['data'] = np.array(0)
    radar.azimuth['data'] = data1['variables']['azimuth']['data']
#    radar.sweep_mode['data'] = np.array(data['sweep_mode'])
    radar.elevation['data'] = data1['variables']['elevation']['data']
#    radar.fixed_angle['data'] = np.array(data['fixed_angle'])
    radar.sweep_start_ray_index['data'] = np.arange(0, data1['dimensions']['azimuth']['size'],360, dtype='int64')
    radar.sweep_end_ray_index['data'] = radar.sweep_start_ray_index['data'] + int(360-1)
    radar.init_gate_longitude_latitude()
    radar.init_gate_altitude()
    radar.init_gate_x_y_z()
    # Let's work on the field data, we will just do reflectivity for now, but any of the
    # other fields can be done the same way and added as a key pair in the fields dict.
    dbz = z.copy()
    fill_value = np.nan
    long_name='Equivalent_Reflectivity_Factor_Horizontal'
    standard_name='equivalent_reflectivity_factor'
    ref_dict = {'data': dbz,
                'units': 'dBZ',
                'long_name': long_name,
                'standard_name': standard_name,
                '_FillValue': fill_value}
    radar.add_field('DBZ', ref_dict, replace_existing=True)
    
    fill_value = np.nan
    long_name='Radial_Velocity_of_Scatterers_away_from_radar'
    standard_name='radial_velocity_of_scatterers_away_from_instrument'
    vel_dict = {'data': vel,
                'units': 'm/s',
                'long_name': long_name,
                'standard_name': standard_name,
                '_FillValue': fill_value}
    radar.add_field('VEL', vel_dict, replace_existing=True)
    
#    fill_value = data['WIDTH'][0,0]
#    long_name='Spectrum_Width'
#    standard_name='doppler_spectrum_width'
#    sigma_dict = {'data': np.array(data['WIDTH']),
#                'units': 'm/s',
#                'long_name': long_name,
#                'standard_name': standard_name,
#                '_FillValue': fill_value}
#    radar.add_field('WIDTH', sigma_dict, replace_existing=True)
#    
#    fill_value = data['ZDR'][0,0]
#    long_name='Log_Differential_Reflectivity_HV'
#    standard_name='log_differential_reflectivity_hv'
#    zdr_dict = {'data': np.array(data['ZDR']),
#                'units': 'dB',
#                'long_name': long_name,
#                'standard_name': standard_name,
#                '_FillValue': fill_value}
#    radar.add_field('ZDR', zdr_dict, replace_existing=True)
#    
#    fill_value = data['PHIDP'][0,0]
#    long_name='Differential_Phase_HV'
#    standard_name='differential_phase_hv'
#    pdp_dict = {'data': np.array(data['PHIDP']),
#                'units': 'degrees',
#                'long_name': long_name,
#                'standard_name': standard_name,
#                '_FillValue': fill_value}
#    radar.add_field('PHIDP', pdp_dict, replace_existing=True)
#    
#    fill_value = data['RHOHV'][0,0]
#    long_name='Cross_Correlation_Ratio_HV'
#    standard_name='cross_correlation_ratio_hv'
#    rho_dict = {'data': np.array(data['RHOHV']),
#                'units': 'unitless',
#                'long_name': long_name,
#                'standard_name': standard_name,
#                '_FillValue': fill_value}
#    radar.add_field('RHOHV', rho_dict, replace_existing=True)
    
    return radar

def two_panel_plot(radar, sweep=0, var1='reflectivity', vmin1=0, vmax1=65,
                   cmap1='RdYlBu_r', units1='dBZ', var2='differential_reflectivity',
                   vmin2=-5, vmax2=5, cmap2='RdYlBu_r', units2='dB', return_flag=False,
                   xlim=[-150,150], ylim=[-150,150]):
    display = pyart.graph.RadarDisplay(radar)
    fig = plt.figure(figsize=(13,5))
    ax1 = fig.add_subplot(121)
    display.plot_ppi(var1, sweep=sweep, vmin=vmin1, vmax=vmax1, cmap=cmap1, 
                     colorbar_label=units1, mask_outside=True)
    display.set_limits(xlim=xlim, ylim=ylim)
    ax2 = fig.add_subplot(122)
    display.plot_ppi(var2, sweep=sweep, vmin=vmin2, vmax=vmax2, cmap=cmap2, 
                     colorbar_label=units2, mask_outside=True)
    display.set_limits(xlim=xlim, ylim=ylim)
    if return_flag:
        return fig, ax1, ax2, display

def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID',
                              dz_field='DBZ'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

#######################################################################


location = os.getcwd()
dirpath = os.getcwd()+"/singledop"
                     
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
zloc = location + '/reflectivity'
#zdrloc = os.getcwd()+'\ZDR'
velloc = location + '/velocity'
#rholoc = 'E:/NEXRAD_HOUSTON_EXPORTED_FILES/temp'
#kdploc = os.getcwd()+'\KDP'

c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
zfiles = []
zdrfiles = []
kdpfiles = []
rhofiles = []    
velfiles = []
                  
for file in os.listdir(zloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            zfiles.append(str(file))
            c1 = c1+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_zfiles = sorted(zfiles)

#for file in os.listdir(zdrloc):
#    try:
#        if file.endswith(".nc"):
#            #print "txt file found:\t", file
#            zdrfiles.append(str(file))
#            c2 = c2+1
#    except Exception as e:
#        raise e
#        print("No files found here!")
#
#sorted_zdrfiles = sorted(zdrfiles)
#
#for file in os.listdir(kdploc):
#    try:
#        if file.endswith(".nc"):
#            #print "txt file found:\t", file
#            kdpfiles.append(str(file))
#            c3 = c3+1
#    except Exception as e:
#        raise e
#        print("No files found here!")
#
#sorted_kdpfiles = sorted(kdpfiles)
#
#for file in os.listdir(rholoc):
#    try:
#        if file.endswith(".nc"):
#            #print "txt file found:\t", file
#            rhofiles.append(str(file))
#            c4 = c4+1
#    except Exception as e:
#        raise e
#        print("No files found here!")
#
#sorted_rhofiles = sorted(rhofiles)

for file in os.listdir(velloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            velfiles.append(str(file))
            c5 = c5+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_velfiles = sorted(velfiles)

for i in range(c1):
    print("File: %s" %(sorted_zfiles[i]))
    convert_to_pyart(sorted_zfiles[i],sorted_velfiles[i])
