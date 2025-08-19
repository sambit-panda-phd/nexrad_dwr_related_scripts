# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:34:16 2019

@author: sambit
"""
cmap1 = mpl.cm.get_cmap('jet',14)
plt.imshow(hid1,origin='lower',cmap=cmap1)
plt.colorbar()

cmap2 = mpl.cm.get_cmap('jet',9)
plt.imshow(hydro_label1,origin='lower',cmap=cmap2)
plt.colorbar()

plt.imshow(z_c[300:600,500:800],origin='lower',cmap=mpl.cm.jet)
plt.colorbar()

plt.imshow(zdr_c,origin='lower',cmap=mpl.cm.jet)
plt.colorbar()