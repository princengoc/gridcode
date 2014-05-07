# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:52:19 2014

@author: princengoc

Understand the discrete grid code
for N = 3

"""

import pylab
import numpy as np
import scipy
import importlib
importlib.import_module('mpl_toolkits').__path__
from mpl_toolkits.mplot3d import Axes3D
from gridcells import Gridcode
#-----------

lam = [15, 22, 29]
rho = 0.45
gc = Gridcode(lam, rho)
gc.computeWordDict()
xyz = np.array(gc.goodWord.values())
#reshape to an r x 3 array
xyz = np.reshape(xyz,(len(xyz),len(lam)))
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

sumxyz = x+y+z
fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z,c='b')

#plot points with given sum
lower = 18
upper = 18
idx = [i for i in xrange(gc.r) if(lower <= sumxyz[i] <= upper)]
ax.scatter(x[idx],y[idx],z[idx],s=60,c='r')

lower = 20
upper = 20
idx = [i for i in xrange(gc.r) if(lower <= sumxyz[i] <= upper)]
ax.scatter(x[idx],y[idx],z[idx],s=60,c='black')


#plot a few surfaces x + y + z = c for different c

#project on the surface x + y + z = 0


u = scipy.linspace(0,14,15)
v = scipy.linspace(0,21,22)
[u,v] = scipy.meshgrid(u,v)

w = 20-(u + v)
ax.plot_surface(u,v,w, color = "green", shade = False)
#pylab.show()


#
#ax.scatter(x.flatten(), y.flatten(), z.flatten())

