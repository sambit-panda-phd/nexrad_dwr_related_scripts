# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:28:38 2019

@author: sambit
"""
import numpy as np
import os
import wradlib as wrl

hidloc = os.getcwd()+"\HID"
sacloc = os.getcwd()+"\hydroclass_v2/hydrolabel1"
dirpath = os.getcwd()+"\common_labels"

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

c1 = 0
c2 = 0
hidfiles = []
sacfiles = []

for file in os.listdir(hidloc):
    try:
        if file.endswith(".nc"):
            #print "txt file found:\t", file
            hidfiles.append(str(file))
            c1 = c1+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_hidfiles = sorted(hidfiles)

for file in os.listdir(sacloc):
    try:
        if file.endswith(".txt"):
            #print "txt file found:\t", file
            sacfiles.append(str(file))
            c2 = c2+1
    except Exception as e:
        raise e
        print("No files found here!")

sorted_sacfiles = sorted(sacfiles)

cid_array = np.empty((c1,7))
sac_array = np.empty((c1,7))
hid_array = np.empty((c1,7))
time_array = np.array([])

for c in range(c1):
    print("File: %s" %(sorted_hidfiles[c]))
    time_array = np.append(time_array,sorted_hidfiles[c][9:24])
    data3 = wrl.io.read_generic_netcdf('%s\%s' %(hidloc,sorted_hidfiles[c]))
    hid = data3['variables']['class']['data'][0]
    hydrolabel1 = np.loadtxt('%s\%s' %(sacloc,sorted_sacfiles[c]))
        
    ## For total count of each species
    n1 = len(hid[hid==9.0])
    n2 = len(hid[hid==3.0])
    n3 = len(hid[hid==10.0]) + len(hid[hid==11.0]) + len(hid[hid==12.0])
    n4 = len(hid[hid==4.0])
    n5 = len(hid[hid==5.0])
    n6 = len(hid[hid==6.0])
    n7 = len(hid[hid==7.0]) + len(hid[hid==8.0])
    hid_array[c] = [n1,n2,n3,n4,n5,n6,n7]
    
    s1 = len(hydrolabel1[hydrolabel1==3.0])
    s2 = len(hydrolabel1[hydrolabel1==1.0]) + len(hydrolabel1[hydrolabel1==5.0])
    s3 = len(hydrolabel1[hydrolabel1==7.0]) + len(hydrolabel1[hydrolabel1==8.0]) 
    s4 = len(hydrolabel1[hydrolabel1==0.0])
    s5 = len(hydrolabel1[hydrolabel1==6.0])
    s6 = len(hydrolabel1[hydrolabel1==2.0])
    s7 = len(hydrolabel1[hydrolabel1==4.0]) 
    sac_array[c] = [s1,s2,s3,s4,s5,s6,s7]
    #####################################
    
    gr_list = []
    ic_list = []
    hl_list = []
    ds_list = []
    ws_list = []
    lr_list = []
    rn_list = []
    
    ## For graupels ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 9.0):
                if(hydrolabel1[m,n] == 3.0):
                   gr_list.append((m,n))
    
    ## For Ice Crystals ## 
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 3.0):
                if(hydrolabel1[m,n] == 1.0 or hydrolabel1[m,n] == 5.0):
                   ic_list.append((m,n))    
    
    ## For Hail ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 10.0 or hid[m,n] == 11.0 or hid[m,n] == 12.0):
                if(hydrolabel1[m,n] == 7.0 or hydrolabel1[m,n] == 8.0):
                   hl_list.append((m,n))  
                                 
    ## For Dry Snow ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 4.0):
                if(hydrolabel1[m,n] == 0.0):
                   ds_list.append((m,n))
                   
    ## For Wet Snow ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 5.0):
                if(hydrolabel1[m,n] == 6.0):
                   ws_list.append((m,n))
                   
    ## For Light/Moderate Rain ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 6.0):
                if(hydrolabel1[m,n] == 2.0):
                   lr_list.append((m,n))
                   
    ## For Heavy Rain ##
    for m in range(643):
        for n in range(800):
            if(hid[m,n] == 7.0 or hid[m,n] == 8.0):
                if(hydrolabel1[m,n] == 4.0):
                   rn_list.append((m,n))
    
    cid_array[c] = [len(gr_list),len(ic_list),len(hl_list),len(ds_list),len(ws_list),len(lr_list),len(rn_list)]

np.savetxt('common_labels_20170826.txt', cid_array)    
np.savetxt('nexrad_hid_counts_20170826.txt', hid_array)
np.savetxt('sac_hid_counts_20170826.txt', sac_array)
np.savetxt('time_list.txt', time_array,fmt='%s')
