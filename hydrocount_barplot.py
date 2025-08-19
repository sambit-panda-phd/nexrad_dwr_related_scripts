# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:01:58 2019

@author: sambit
"""

import numpy as np
import matplotlib.pyplot as plt

clb = np.loadtxt('./common_labels/common_labels_20170826.txt')
nlb = np.loadtxt('./counts/nexrad_hid_counts_20170826.txt')
slb = np.loadtxt('./counts/sac_hid_counts_20170826.txt')
time = np.loadtxt('./time_list.txt')

# set width of bar
barwidth = 0.2

for i in range(0,nlb.shape[0],5):
    print('Time of Observation: %s' %(str(int(time[i]))))
    # set height of bar
    bars1 = [nlb[i,0],nlb[i,1],nlb[i,2],nlb[i,3],nlb[i,4],nlb[i,5],nlb[i,6]]
    bars2 = [slb[i,0],slb[i,1],slb[i,2],slb[i,3],slb[i,4],slb[i,5],slb[i,6]]
    bars3 = [clb[i,0],clb[i,1],clb[i,2],clb[i,3],clb[i,4],clb[i,5],clb[i,6]]
    
    # set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barwidth for x in r1]
    r3 = [x + barwidth for x in r2]
    
    # Make the plot
    plt.bar(r1,bars1,width=barwidth,edgecolor='white',label='Nexrad',align='center')
    plt.bar(r2,bars2,width=barwidth,edgecolor='white',label='SAC',align='center')
    plt.bar(r3,bars3,width=barwidth,edgecolor='white',label='Common',align='center')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Hydro_species',fontweight='bold')
    plt.ylabel('Counts', fontweight='bold')
    plt.ylim(0.0,43000.0)
    plt.xticks([r + barwidth for r in range(len(bars1))], ['GR', 'IC', 'HL', 'DS', 'WS', 'LR', 'HR'])
    
    # Create legend & Save graphic
    plt.legend()
    plt.title('Counts comparison: %s' %(str(int(time[i]))))
    plt.savefig('./count_plots/%s_counts.png' %(str(int(time[i]))), dpi=600)
    plt.clf()
    plt.close()
    