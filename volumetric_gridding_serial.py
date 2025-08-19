#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:09:22 2019

@author: sambit
"""

import wradlib as wrl 
import numpy as np
from osgeo import osr
import datetime as dt
import sys
import os
import itertools
import glob
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import *
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()
print("Running new VPR code!!!")
location = os.getcwd()
dirpath1 = os.getcwd()+"/new_volumetric_gridded_dbz"
dirpath2 = os.getcwd()+"/new_volumetric_gridded_zdr"
if not os.path.exists(dirpath1):
    os.makedirs(dirpath1)
    os.makedirs(dirpath2)
### for writing the volume file ###
def write_volfile(fname,z,dpr,shp,fmode,grid):
    np.savetxt("trgxyz.txt",grid)
    if fmode == "dprf":
          #c = c+1
       name = fname[27:] + "_dprf"
    #for f in range(shp):
       np.savetxt("./new_volumetric_gridded_dbz/%s_vol_dbz.txt" %(name),z.reshape((481*81,481)))
#       np.savetxt("./ipshita/new_volfiles/%s_vol_sw.txt" %(name),sw.reshape((481*81,481)))
#       np.savetxt("./ipshita/new_volfiles/%s_vol_phidp.txt" %(name),phi.reshape((481*81,481)))
       np.savetxt("./new_volumetric_gridded_zdr/%s_vol_zdr.txt" %(name),dpr.reshape((481*81,481)))
#       np.savetxt("./ipshita/new_volfiles/%s_vol_rhohv.txt" %(name),rho.reshape((481*81,481)))
#       np.savetxt("./ipshita/new_volfiles/%s_vol_kdp.txt" %(name),kdp.reshape((481*81,481)))
### for plotting the volumetrically gridded data ####        
def plot_volfile(trg,shape,name,z_vol,zdr_vol):
    
    # diagnostic plot 
    trgx=trg[:,0].reshape(shape)[0,0,:] 
    trgy=trg[:,1].reshape(shape)[0,:,0] 
    trgz=trg[:,2].reshape(shape)[:,0,0] 
    wrl.vis.plot_max_plan_and_vert(trgx,trgy,trgz,np.ma.masked_invalid(z_vol),unit="dBZ", levels=range(0,60),cmap = mpl.cm.jet,saveto="./new_volumetric_gridded_dbz/%s_voldbz.png" %(name[27:]))
    wrl.vis.plot_max_plan_and_vert(trgx,trgy,trgz,np.ma.masked_invalid(zdr_vol),unit="ZDR", levels=range(-2,8),cmap = mpl.cm.jet,saveto="./new_volumetric_gridded_zdr/%s_volzdr.png" %(name[27:]))

### for processing ncfile ###
def process_ncfile(filename):
    # this is the radar position tuple (longitude, latitude, altitude) 
    sitecoords=(76.8657, 8.5374, 27.0)

    # define your cartesian reference system 
    # proj = wradlib.georef.create_osr(32632) 
    proj=osr.SpatialReference() 
    proj.ImportFromEPSG(32643) ##for Indain region (NS-Central-covering Kerala)
    
    # read the nc file
    raw=wrl.io.read_generic_netcdf(filename) 

    # containers to hold Cartesian bin coordinates and data 
    xyz,data=np.array([]).reshape((-1,3)),np.array([])
    zdr_data =  data.copy()
    elevs=raw['variables']['fixed_angle']['data'] 
    num_elev = raw['dimensions']['sweep']['size']
    num_bins = raw['dimensions']['range']['size']
    num_azim = len(raw['variables']['azimuth']['data'])/num_elev
   #az = raw['variables']['azimuth']['data']
    elevation = raw['variables']['elevation']['data']
    # variables
    h_range = raw['variables']['range']['data']
    max_range = np.max(h_range)
    range_resol = h_range[1] - h_range[0]
    
    if range_resol == 150.0:
       mode = "dprf"
       
    if range_resol == 300.0:
       mode = "sprf"
       
    ###select only dprf files###
    if mode == "dprf" and num_bins == 1600 and num_elev == 11:
       #c = c+1
       
       dbZ = raw['variables']['DBZ']['data']
       vel = raw['variables']['VEL']['data']         
       sigma = raw['variables']['WIDTH']['data']
       zdr = raw['variables']['ZDR']['data']
       phidata = raw['variables']['PHIDP']['data']
       rhohv = raw['variables']['RHOHV']['data']
    
       # define arrays of polar coordinate arrays (azimuth and range) 
       az=np.arange(0.,360.,1) 
#       r=np.arange(0.,max_range,range_resol)
       r = h_range.copy()
       # generate 3-D Cartesian target grid coordinates 
       maxrange=range_resol*num_bins
       minelev=elevs.min() 
       maxelev=elevs.max()
       maxalt=20000 
       horiz_res=1000. 
       vert_res=250. 
       trgxyz,trgshape=wrl.vpr.make_3d_grid(sitecoords,proj,maxrange,maxalt,horiz_res,vert_res)
    
    
       # iterate over all elevation angles 
       for e in range(num_elev): 
           # derive 3-D Cartesian coordinate tuples 
           xyz_=wrl.vpr.volcoords_from_polar(sitecoords,elevs[e], az,r,proj)
           a_min = 360*e
           a_max = 360*(e+1)
           #refl = dbZ[a_min:a_max,0:num_bins]
           data_= dbZ[a_min:a_max,0:num_bins]
           data_[data_== 25.0] = np.nan
           data_ = wrl.dp.linear_despeckle(data_,3)
           velocity = vel[a_min:a_max,0:num_bins]
           velocity[velocity==vel[0,0]] = np.nan
           velocity = wrl.dp.linear_despeckle(velocity,3)
           dp_ratio = zdr[a_min:a_max,0:num_bins]
           dp_ratio[dp_ratio == dp_ratio[0,0]] = np.nan
           dp_ratio = wrl.dp.linear_despeckle(dp_ratio,3)        
           phi = phidata[a_min:a_max,0:num_bins]
           phi[phi == phi[0,0]] = np.nan
           phi = wrl.dp.linear_despeckle(phi,3)   
           rho = rhohv[a_min:a_max,0:num_bins]
           rho[rho == rho[0,0]] = np.nan
           rho = wrl.dp.linear_despeckle(rho,3)
           sw = sigma[a_min:a_max,0:num_bins]
           sw[sw==sigma[0,0]] = np.nan
           sw = wrl.dp.linear_despeckle(sw,3)  
           #### calculate kdp from phidp####
           kdp = wrl.dp.kdp_from_phidp(phi)
           kdp = wrl.dp.linear_despeckle(kdp,3)
           #################################
           clutter = wrl.clutter.filter_gabella(data_,wsize=3,thrsnorain=0.0,tr1=10.0,n_p=8,tr2=1.3)
           mdata = data_.copy()
           mdata = np.ma.array(mdata,mask=clutter)
           mdata[mdata.mask] = np.nan
           clutter = wrl.clutter.filter_gabella(mdata,wsize=5,thrsnorain=0.0,tr1=10.0,n_p=8,tr2=1.3)
           mdata = np.ma.array(mdata,mask=clutter)
           mdata[mdata.mask] = np.nan
           dat = {}
           dat["rho"] = rho
           dat["phi"] = phi
           dat["ref"] = mdata
           dat["dop"] = velocity
           dat["zdr"] = dp_ratio
           dat["map"] = clutter
           weights = {"zdr":0.5,"rho":0.1,"rho2":0.1,"phi":0.2,"dop":0.3,"map":0.5}
           cmap,nanmask = wrl.clutter.classify_echo_fuzzy(dat,weights=weights,thresh=0.5)

           rdata = np.ma.array(mdata.copy(),mask=cmap)
#               rdata[rdata<25.0] = np.nan
           rdata[rdata.mask] = np.nan
           rdata = wrl.dp.linear_despeckle(rdata,5)

           ### remove clutter pixels from other variables ###
#           velocity[np.where(np.isnan(rdata))] = np.nan
#           phi[np.where(np.isnan(rdata))] = np.nan
           dp_ratio[np.where(np.isnan(rdata))] = np.nan
#           rho[np.where(np.isnan(rdata))] = np.nan 
#           sw[np.where(np.isnan(rdata))] = np.nan
#           kdp[np.where(np.isnan(rdata))] = np.nan   
           
           # transfer to containers 
           xyz=np.vstack((xyz,xyz_))
           data=np.append(data,rdata.ravel())
           zdr_data=np.append(zdr_data,dp_ratio.ravel())  
#           phidp_data=np.append(phidp_data,phi.ravel())
#           rhohv_data=np.append(rhohv_data,rho.ravel())
#           kdp_data=np.append(kdp_data,kdp.ravel()
#           sw_data=np.append(sw_data,sw.ravel())
       # interpolate to Cartesian 3-D volume grid 
#       tstart=dt.datetime.now() 
       gridder=wrl.vpr.CAPPI(xyz,trgxyz,trgshape,maxrange,minelev,maxelev,ipclass=wrl.ipol.Idw) 
       volz=np.ma.masked_invalid(gridder(data).reshape(trgshape))
       volz[volz.mask]=np.nan     
#       vol_phi=np.ma.masked_invalid(gridder(phidp_data).reshape(trgshape))
#       vol_phi[vol_phi.mask]=np.nan
       vol_zdr=np.ma.masked_invalid(gridder(zdr_data).reshape(trgshape))
       vol_zdr[vol_zdr.mask]=np.nan
#       vol_rho=np.ma.masked_invalid(gridder(rhohv_data).reshape(trgshape))
#       vol_rho[vol_rho.mask]=np.nan
#       vol_kdp=np.ma.masked_invalid(gridder(kdp_data).reshape(trgshape))
#       vol_kdp[vol_kdp.mask]=np.nan 
#       vol_sw=np.ma.masked_invalid(gridder(sw_data).reshape(trgshape))
#       vol_sw[vol_sw.mask]=np.nan        
       print("3-D interpolation took:",dt.datetime.now()-tstart)
#       vol_lat=np.ma.masked_invalid(gridder(d_lat).reshape(trgshape))
#       vol_lon=np.ma.masked_invalid(gridder(d_lon).reshape(trgshape))
       #vol_height=np.ma.masked_invalid(gridder(d_height).reshape(trgshape))
       write_volfile(filename, volz, vol_zdr,trgshape[0], mode,trgxyz)
       plot_volfile(trgxyz,trgshape,filename,volz,vol_zdr)
       
       return

        
#### main program starts ####
#os.chdir("/sachome1/usr/sambit/TERLS/01122017/")   

counter = 0
ncfiles = []
num = 0
num1= 0
c = 0

for file in os.listdir(location):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            ncfiles.append(str(file))
            counter = counter+1
    except Exception as e:
        raise e
        print("No files found here!")

sortedfiles = sorted(ncfiles)
       
#for i in range(counter):
#pool = multiprocessing.Pool(processes=16)


for i in range(counter):
    print("File: %s" %(sortedfiles[i]))
    tstart=dt.datetime.now() 
    process_ncfile(sortedfiles[i])
             
print("3-D interpolation program took total:",dt.datetime.now()-tstart) 
#def mp_handler():
#	p=multiprocessing.Pool(6)
#	p.map(process_ncfile, sortedfiles)
#
#
#if __name__=='__main__' :
#	mp_handler()



