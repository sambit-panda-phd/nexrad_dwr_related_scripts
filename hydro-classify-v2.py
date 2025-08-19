#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:00:18 2018

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
dirpath = os.getcwd()+"/hydroclass_v2"
seltpath = dirpath+"/selected_temp"
selzpath = dirpath+"/selected_z"
selzdrpath = dirpath+"/selected_zdr"
center1path = dirpath+"/centers1"
center2path = dirpath+"/centers2"
prob1path = dirpath+"/prob1"
prob2path = dirpath+"/prob2"
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
    
    
#def draw_ellipse(position, covariance, ax=None, **kwargs):
#    """Draw an ellipse with a given position and covariance"""
#    ax = ax or plt.gca()
#    
#    # Convert covariance to principal axes
#    if covariance.shape == (2, 2):
#        U, s, Vt = np.linalg.svd(covariance)
#        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#        width, height = 2 * np.sqrt(s)
#    else:
#        angle = 0
#        width, height = 2 * np.sqrt(covariance)
#    
#    # Draw the Ellipse
#    for nsig in range(1, 4):
#        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
#                             angle, **kwargs))
#        
#def plot_gmm(gmm, X, label=True, ax=None):
#    ax = ax or plt.gca()
#    labels = gmm.fit(X).predict(X)
#    if label:
#        ax.scatter(X[:, 0], X[:, 1], c=labels, label=labels, s=40, cmap='viridis', zorder=2)
#        ax.legend()
#    else:
#        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
#    ax.axis('equal')
#    
#    w_factor = 0.2 / gmm.weights_.max()
#    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#        draw_ellipse(pos, covar, alpha=w * w_factor)
#
#height = np.arange(0,81*250,250)/1000.0
#temp = np.zeros((81))                  
#temp[0] = 300.0
#    
#for h in range(1,81):
#    temp[h] = temp[h-1]-(6.5*0.25) 
#
#cc = np.ones((481,481))
#t3d = np.empty((81,481,481))
#for h in range(81):
#    t3d[h] = cc*temp[h]

zloc = os.getcwd()+'\Z'
zdrloc = os.getcwd()+'\ZDR'
hidloc = os.getcwd()+'\HID'

c1 = 0
c2 = 0
c3 = 0
volzfiles = []
volzdrfiles = []    
hidfiles = []
                  
for file in os.listdir(zloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            volzfiles.append(str(file))
            c1 = c1+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_zfiles = sorted(volzfiles)

for file in os.listdir(zdrloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            volzdrfiles.append(str(file))
            c2 = c2+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_zdrfiles = sorted(volzdrfiles)

for file in os.listdir(hidloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            hidfiles.append(str(file))
            c3 = c3+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_hidfiles = sorted(hidfiles)

nclasses = 9
nvariables = 2
mass_centers = np.zeros((nclasses, nvariables))
mass_centers[0, :] = [13.5829,  0.4063]  # DS
mass_centers[1, :] = [02.8453,  0.2457]  # CR
mass_centers[2, :] = [07.6597,  0.2180]  # LR
mass_centers[3, :] = [31.6815,  0.3926]  # GR
mass_centers[4, :] = [39.4703,  1.0734]  # RN
mass_centers[5, :] = [04.8267, -0.5690]  # VI
mass_centers[6, :] = [30.8613,  0.9819]  # WS
mass_centers[7, :] = [52.3969,  2.1094]  # MH
mass_centers[8, :] = [50.6186, -0.0649]  # IH/HDG

#centers = np.loadtxt('/sachome1/usr/sambit/TERLS/01122017/ipshita/TERLS_DWR_JINYA/20171130/centers1/new_20171130050347_centers1_9.txt')
#mass_centers = centers[:,0:2]                   
for i in range(c1):
    start_time1 =  datetime.datetime.now()
    print("Running Hydroclassification code for TERLS DWR!!! \n")              
    print("File: %s" %(sorted_zfiles[i]))
    data1 = wrl.io.read_generic_netcdf('%s\%s' %(zloc,sorted_zfiles[i]))
    data2 = wrl.io.read_generic_netcdf('%s\%s' %(zdrloc,sorted_zdrfiles[i]))
    data3 = wrl.io.read_generic_netcdf('%s\%s' %(hidloc,sorted_hidfiles[i]))
    z = data1['variables']['bref']['data'][0]
    zdr = data2['variables']['zdr']['data'][0]
    hid = data3['variables']['class']['data'][0]
    z_c = z.copy()
    zdr_c = zdr.copy()
#    zdr_c[np.where(zdr<-2.0)]=np.nan 
    zdr_c[np.where(np.isnan(z_c))] = np.nan
    z_c[np.where(np.isnan(zdr_c))] = np.nan
    zdr_c[np.where(np.isnan(z_c))] = np.nan
    index_na = np.where(np.isnan(z_c))
#    t3d_c = t3d.copy()
#    t3d_c = np.reshape(t3d_c,(481*81,481))
#    t3d_c[np.where(np.isnan(z_c))] = np.nan
    
    z_new = z_c[np.isfinite(z_c)]
    zdr_new = zdr_c[np.isfinite(zdr_c)]  
#    t_new = t3d_c[np.isfinite(t3d_c)]
    #index_na_zdr = np.where(np.isnan(zdr))
    index_finite = np.where(np.isfinite(z_c))
    
    if np.size(z_new) >= 9:
        df = pd.DataFrame({'DBZ':z_new,'ZDR':zdr_new})
#        df = pd.DataFrame({'DBZ':z_new,'ZDR':zdr_new,'TEMP':t_new})
        #    df.dropna(inplace=True)
        #    a = df.as_matrix()
    
        print("Processing File: %s" %(sorted_zfiles[i]))
        
        
        np.savetxt('%s/v2_selected_z_%s.txt' %(selzpath,sorted_zfiles[i][0:19]),z_new)
        np.savetxt('%s/v2_selected_zdr_%s.txt' %(selzdrpath,sorted_zfiles[i][0:19]),zdr_new)
#        np.savetxt('%s/selected_temp_%s.txt' %(seltpath,sorted_zfiles[i][0:19]),t_new)
        
        z1 = np.array(df['DBZ'])
        zdr1 = np.array(df['ZDR'])
#        temp1 = np.array(df['TEMP']) 
        
        x=np.array((z1,zdr1))
#        x=np.array((z1,zdr1,temp1))
        x= x.T
        nc = 0
        
        #zdr_select = zdr_arg[zdr_arg>-5.0]
    #    seeds1 = np.array([[39.4703,1.0734,300.0],[52.3969,2.1094,273.0]])
        gmm1 = mixture.GaussianMixture(n_components=9,means_init=mass_centers,covariance_type='full').fit(x)
        labels1 = gmm1.predict(x)
        means1 = gmm1.means_
        probs1 = gmm1.predict_proba(x)
        df1 = pd.DataFrame({'z':z1,'zdr':zdr1,'label':labels1}) 
        
        end_time1 =  datetime.datetime.now() 
        diff_time1 = end_time1 - start_time1
        
        print("The first classification method took: %s secs" %(diff_time1.seconds))
    #    #create a new figure
    #    fig1,ax1 = plt.subplots(figsize=(10,8))
    #    ax1.set_title('Hydrometeor Clasification1')
    #    #loop through labels and plot each cluster
    #    for j in range(9):
    #    
    #        #add data points 
    #        ax1.scatter(x=df1.loc[df1['label']==j, 'z'], 
    #                    y=df1.loc[df1['label']==j,'zdr'], 
    #                    alpha=0.20,label='('+str(round(means1[j,0],2))+','+str(round(means1[j,1],2))+')')
    #        
    #        #add label
    #        ax1.annotate((round(means1[j,0],2),round(means1[j,1],2)),df1.loc[df1['label']==j,['z','zdr']].mean(),
    #                     horizontalalignment='center',
    #                     verticalalignment='center',
    #                     size=10, weight='bold') 
    #    ax1.legend()
    #    plt.savefig('%s_hydroclass.png' %(sorted_zfiles[i][0:19]))
    #    plt.close()
    #    plt.scatter(x[:,0],x[:,1],c=labels1,s=40,cmap='jet')
        
        hydro_label1 = np.empty((643,800))
        hydro_label1[index_na[0],index_na[1]]=np.nan
        hydro_label1[index_finite[0],index_finite[1]]=labels1
        
        np.savetxt('%s/v2_%s_hydrolabels1.txt' %(hydrolabel1path,sorted_zfiles[i]),hydro_label1)
        np.savetxt('%s/v2_%s_centers1.txt' %(center1path,sorted_zfiles[i]),means1)                
        np.savetxt('%s/v2_%s_probability1.txt' %(prob1path,sorted_zfiles[i]),probs1)
        
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
        plt.savefig('./hydroclass_v2/plots/%s_sac_hid.png' %(sorted_zfiles[i]),dpi=600)
        plt.close()
    ############# 2nd Method #################    
#        start_time2 =  datetime.datetime.now()  
#        
#        gmm2 = mixture.GaussianMixture(n_components=9,covariance_type='full').fit(x)
#        probs2 = gmm2.predict_proba(x)
#        labels2 = gmm2.predict(x)
#        means2 = gmm2.means_
#        df2 = pd.DataFrame({'z':z1,'zdr':zdr1,'label':labels2})
#        
#        end_time2 =  datetime.datetime.now() 
#        diff_time2 = end_time2 - start_time2    
#        
#        print("The second classification method took: %s secs" %(diff_time2.seconds))
#        
#        hydro_label2 = np.empty((643,800))
#        hydro_label2[index_na[0],index_na[1]]=np.nan
#        hydro_label2[index_finite[0],index_finite[1]]=labels2
#        
#        np.savetxt('%s/v2_%s_hydrolabels2.txt' %(hydrolabel2path,sorted_zfiles[i][0:19]),hydro_label2)
#        np.savetxt('%s/v2_%s_centers2.txt' %(center2path,sorted_zfiles[i][0:19]),means2)
#        np.savetxt('%s/v2_%s_probability2.txt' %(prob2path,sorted_zfiles[i][0:19]),probs2)
        
    #    #create a new figure
    #    fig2,ax2 = plt.subplots(figsize=(10,8))
    #    ax2.set_title('Hydrometeor Clasification2')
    #    #loop through labels and plot each cluster
    #    for j in range(9):
    #    
    #        #add data points 
    #        ax2.scatter(x=df2.loc[df2['label']==j, 'z'], 
    #                    y=df2.loc[df2['label']==j,'zdr'], 
    #                    alpha=0.20,label='('+str(round(means2[j,0],2))+','+str(round(means2[j,1],2))+')')
    #        
    #        #add label
    #        ax2.annotate((round(means2[j,0],2),round(means2[j,1],2)),df2.loc[df2['label']==j,['z','zdr']].mean(),
    #                     horizontalalignment='center',
    #                     verticalalignment='center',
    #                     size=10, weight='bold') 
    #    ax2.legend()
    #    plt.savefig('%s_hydroclass2.png' %(sorted_zfiles[i][0:19]))
    #    plt.close()
    #        
        
    
 










   
#    n_components = np.arange(2,21)
#    models = [mixture.GaussianMixture(n,covariance_type='full',random_state=0).fit(x) for n in n_components]
    

    #plt.plot(n_components,[m.bic(x) for m in models],label='BIC')
    #plt.plot(n_components,[m.aic(x) for m in models],label='AIC')
    #plt.legend(loc='best')
    #plt.xlabel('n_components')
        
    #    fig3, ax3 = plt.subplots()
    #    ax3.set_title('Hydrometeor Clasification')
    #    for j in range(i):
    #        ax3.scatter(x[labels == j, 0],
    #                 x[labels == j, 1], 'o',
    #                 label='z:zdr ' + str(centers[j,0]) + ':' + str(centers[j,1]))
    #    ax3.legend()
    #    plt.savefig('test_hydro_classify_%s.png' %(i))
    #    plt.close()
    
    
    
    #    fig3, ax3 = plt.subplots()
    #    ax3.set_title('Hydrometeor Clasification')
    #    for j in range(i):
    #        ax3.plot(x1[labels == j, 0],
    #                 x1[labels == j, 1], 'o',
    #                 label='z:zdr ' + str(centers[j,0]) + ':' + str(centers[j,1]))
    #    ax3.legend()
    #    plt.savefig('test3_hydro_classify_%s.png' %(i))
    #    plt.close()
    #    hydro_class = hydro_label.reshape((81,481,481))
    #    cmap = colors.ListedColormap(['red','green','blue'])
    #    bounds = np.arange(0,3)
    #    norm = colors.BoundaryNorm(bounds,cmap.N)
    #    img = plt.imshow(hydro_class[20],interpolation='nearest',origin='lower',cmap=cmap,norm=norm)
    #    plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds,ticks=[0,1,2])
    #    plt.savefig('test3_hydro_class_%s.png' %(i))
    #    plt.close()
    #    
    #plt.imshow(hydro_class[20],origin='lower',cmap=plt.cm.jet)
    #plt.colorbar()
