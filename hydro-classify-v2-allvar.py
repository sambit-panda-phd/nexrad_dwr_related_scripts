# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:44:15 2019

@author: sambit
"""

from warnings import warn
import numpy as np
import os
import glob
import wradlib as wrl
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


location = os.getcwd()
dirpath = os.getcwd()+"/hydroclass_v2_allvar"
seltpath = dirpath+"/selected_temp"
selzpath = dirpath+"/selected_z"
selzdrpath = dirpath+"/selected_zdr"
selkdppath = dirpath+"/selected_kdp"
selrhopath = dirpath+"/selected_rho"
center1path = dirpath+"/centers1"
center2path = dirpath+"/centers2"
prob1path = dirpath+"/prob1"
prob2path = dirpath+"/prob2"
plotspath = dirpath+"/plots"
hydrolabel1path = dirpath+"/hydrolabel1"
hydrolabel2path = dirpath+"/hydrolabel2"
                     
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    os.makedirs(seltpath)
    os.makedirs(selzpath)
    os.makedirs(selzdrpath)
    os.makedirs(center1path)
    os.makedirs(center2path)
    os.makedirs(prob1path)
    os.makedirs(prob2path)
    os.makedirs(hydrolabel1path)
    os.makedirs(hydrolabel2path)
    
zloc = os.getcwd()+'\Z'
zdrloc = os.getcwd()+'\ZDR'
hidloc = os.getcwd()+'\HID'
rholoc = 'E:/NEXRAD_HOUSTON_EXPORTED_FILES/temp'
kdploc = os.getcwd()+'\KDP'

c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
zfiles = []
zdrfiles = []
kdpfiles = []
rhofiles = []    
hidfiles = []
                  
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

for file in os.listdir(zdrloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            zdrfiles.append(str(file))
            c2 = c2+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_zdrfiles = sorted(zdrfiles)

for file in os.listdir(kdploc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            kdpfiles.append(str(file))
            c3 = c3+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_kdpfiles = sorted(kdpfiles)

for file in os.listdir(rholoc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            rhofiles.append(str(file))
            c4 = c4+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_rhofiles = sorted(rhofiles)

for file in os.listdir(hidloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            hidfiles.append(str(file))
            c5 = c5+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_hidfiles = sorted(hidfiles)

nclasses = 9
nvariables = 4
mass_centers = np.zeros((nclasses, nvariables))
# C-band centroids derived for MeteoSwiss Albis radar
#                       Zh        ZDR     kdp   RhoHV    
mass_centers[0, :] = [13.5829,  0.4063, 0.0497, 0.9868]  # DS
mass_centers[1, :] = [02.8453,  0.2457, 0.0000, 0.9798]  # CR
mass_centers[2, :] = [07.6597,  0.2180, 0.0019, 0.9799]  # LR
mass_centers[3, :] = [31.6815,  0.3926, 0.0828, 0.9978]  # GR
mass_centers[4, :] = [39.4703,  1.0734, 0.4919, 0.9876]  # RN
mass_centers[5, :] = [04.8267, -0.5690, 0.0000, 0.9691]  # VI
mass_centers[6, :] = [30.8613,  0.9819, 0.1998, 0.9845]  # WS
mass_centers[7, :] = [52.3969,  2.1094, 2.4675, 0.9730]  # MH
mass_centers[8, :] = [50.6186, -0.0649, 0.0946, 0.9904]  # IH/HDG


#mass_centers[0, :] = [13.5829,  0.4063]  # DS
#mass_centers[1, :] = [02.8453,  0.2457]  # CR
#mass_centers[2, :] = [07.6597,  0.2180]  # LR
#mass_centers[3, :] = [31.6815,  0.3926]  # GR
#mass_centers[4, :] = [39.4703,  1.0734]  # RN
#mass_centers[5, :] = [04.8267, -0.5690]  # VI
#mass_centers[6, :] = [30.8613,  0.9819]  # WS
#mass_centers[7, :] = [52.3969,  2.1094]  # MH
#mass_centers[8, :] = [50.6186, -0.0649]  # IH/HDG

for i in range(c1):
    start_time1 =  datetime.datetime.now()
    print("Running Hydroclassification code for TERLS DWR!!! \n")              
    print("File: %s" %(sorted_zfiles[i]))
    data1 = wrl.io.read_generic_netcdf('%s\%s' %(zloc,sorted_zfiles[i]))
    data2 = wrl.io.read_generic_netcdf('%s\%s' %(zdrloc,sorted_zdrfiles[i]))
    data3 = wrl.io.read_generic_netcdf('%s\%s' %(kdploc,sorted_kdpfiles[i]))
    data4 = wrl.io.read_generic_netcdf('%s\%s' %(rholoc,sorted_rhofiles[i]))
    data5 = wrl.io.read_generic_netcdf('%s\%s' %(hidloc,sorted_hidfiles[i]))
    z = data1['variables']['bref']['data'][0]
    zdr = data2['variables']['zdr']['data'][0]
    kdp = data3['variables']['kdp']['data'][0]
    rho = data4['variables']['cc']['data'][0]
    hid = data5['variables']['class']['data'][0]
    
    z_c = z.copy()
    zdr_c = zdr.copy()
    kdp_c = kdp.copy()
    rho_c = rho.copy()
    
    z_nan=np.where(np.isnan(z)) or np.where(np.isnan(kdp)) or np.where(np.isnan(rho)) or np.where(np.isnan(zdr))
    
    zdr_c[z_nan[0],z_nan[1]] = np.nan
    z_c[z_nan[0],z_nan[1]] = np.nan
    kdp_c[z_nan[0],z_nan[1]] = np.nan
    rho_c[z_nan[0],z_nan[1]] = np.nan
    index_na_z = np.where(np.isnan(z_c))
    index_na_zdr = np.where(np.isnan(zdr_c))
    index_na_kdp = np.where(np.isnan(kdp_c))
    index_na_rho = np.where(np.isnan(rho_c))
    
    z_c[index_na_zdr]=np.nan
    z_c[index_na_kdp]=np.nan
    z_c[index_na_rho]=np.nan
    zdr_c[np.where(np.isnan(z_c))]=np.nan
    kdp_c[np.where(np.isnan(z_c))]=np.nan
    rho_c[np.where(np.isnan(z_c))]=np.nan
    index_finite = np.where(np.isfinite(z_c))
    index_na = np.where(np.isnan(z_c))
#    t3d_c = t3d.copy()
#    t3d_c = np.reshape(t3d_c,(481*81,481))
#    t3d_c[np.where(np.isnan(z_c))] = np.nan
    
#    z_new=z_c[np.where(np.isfinite(z_c)) and np.where(np.isfinite(kdp_c)) and np.where(np.isfinite(rho_c)) and np.where(np.isfinite(zdr_c))]
#    zdr_new=zdr_c[np.where(np.isfinite(z_c)) and np.where(np.isfinite(kdp_c)) and np.where(np.isfinite(rho_c)) and np.where(np.isfinite(zdr_c))]
#    kdp_new=kdp_c[np.where(np.isfinite(z_c)) and np.where(np.isfinite(kdp_c)) and np.where(np.isfinite(rho_c)) and np.where(np.isfinite(zdr_c))]
#    rho_new=rho_c[np.where(np.isfinite(z_c)) and np.where(np.isfinite(kdp_c)) and np.where(np.isfinite(rho_c)) and np.where(np.isfinite(zdr_c))]
#    
#     
    z_new = z_c[np.where(np.isfinite(z_c))]
    zdr_new = zdr_c[np.where(np.isfinite(zdr_c))]
    kdp_new = kdp_c[np.where(np.isfinite(kdp_c))]  
    rho_new = rho_c[np.where(np.isfinite(rho_c))]  
#    t_new = t3d_c[np.isfinite(t3d_c)]
    #index_na_zdr = np.where(np.isnan(zdr))
#    index_finite = np.where(np.isfinite(z_c))
    
    if index_finite[0].size >= 9:
        df = pd.DataFrame({'DBZ':z_new,'ZDR':zdr_new,'KDP':kdp_new,'RHO':rho_new})
#        df = pd.DataFrame({'DBZ':z_new,'ZDR':zdr_new,'TEMP':t_new})
#        df.dropna(inplace=True)
#        a = df.as_matrix()
    
        print("Processing File: %s" %(sorted_zfiles[i]))
        
        
        np.savetxt('%s/allvar_v2_selected_z_%s.txt' %(selzpath,sorted_zfiles[i][0:19]),z_c)
        np.savetxt('%s/allvar_v2_selected_zdr_%s.txt' %(selzdrpath,sorted_zdrfiles[i][0:19]),zdr_c)
        np.savetxt('%s/allvar_v2_selected_kdp_%s.txt' %(selkdppath,sorted_kdpfiles[i][0:19]),kdp_c)
        np.savetxt('%s/allvar_v2_selected_rho_%s.txt' %(selrhopath,sorted_rhofiles[i][0:19]),rho_c)
#        np.savetxt('%s/selected_temp_%s.txt' %(seltpath,sorted_zfiles[i][0:19]),t_new)
        
        z1 = np.array(df['DBZ'])
        zdr1 = np.array(df['ZDR'])
        kdp1 = np.array(df['KDP'])
        rho1 = np.array(df['RHO'])
#        temp1 = np.array(df['TEMP']) 
        
        x=np.array((z1,zdr1,kdp1,rho1))
#        x=np.array((z1,zdr1,temp1))
        x= x.T
        nc = 0
        
        #zdr_select = zdr_arg[zdr_arg>-5.0]
    #    seeds1 = np.array([[39.4703,1.0734,300.0],[52.3969,2.1094,273.0]])
        gmm1 = mixture.GaussianMixture(n_components=9,means_init=mass_centers,covariance_type='full').fit(x)
        labels1 = gmm1.predict(x)
        means1 = gmm1.means_
        probs1 = gmm1.predict_proba(x)
        df1 = pd.DataFrame({'z':z1,'zdr':zdr1,'kdp':kdp1,'rho':rho1,'label':labels1}) 
        
        end_time1 =  datetime.datetime.now() 
        diff_time1 = end_time1 - start_time1
        
        print("The first classification method took: %s secs" %(diff_time1.seconds))
        
        hydro_label1 = np.empty((643,800))
        hydro_label1[index_na[0],index_na[1]]=np.nan
        hydro_label1[index_finite[0],index_finite[1]]=labels1
        
        np.savetxt('%s/allvar_v2_%s_hydrolabels1.txt' %(hydrolabel1path,sorted_zfiles[i]),hydro_label1)
        np.savetxt('%s/allvar_v2_%s_centers1.txt' %(center1path,sorted_zfiles[i]),means1)                
        np.savetxt('%s/allvar_v2_%s_probability1.txt' %(prob1path,sorted_zfiles[i]),probs1)
        
#        plt.figure(figsize=(10,8),dpi=600)
#        cmap1 = mpl.cm.get_cmap('jet',14)
#        plt.imshow(hid,origin='lower',cmap=cmap1)
#        plt.colorbar()
#        plt.title('nexrad_product')
#        plt.savefig('%s_nexrad_hid.png' %(sorted_hidfiles[i]),dpi=600)
#        plt.close()

        plt.figure(figsize=(10,8),dpi=600)  
        cmap2 = mpl.cm.get_cmap('jet',9)
        plt.imshow(hydro_label1,origin='lower',cmap=cmap2,vmin=0.0,vmax=8.0)
        plt.colorbar()
        plt.title('sac_algorithm_%s' %(sorted_zfiles[i][9:24]))
        plt.savefig('./hydroclass_v2_allvar/plots/%s_sac_hid.png' %(sorted_zfiles[i]),dpi=600)
        plt.close()