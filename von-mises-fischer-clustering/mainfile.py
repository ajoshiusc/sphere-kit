# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:18:26 2017

@author: Bhavana
"""


import vonmisesGenerate as vmg
import sphericalclustering as snn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import time

start_time = time.clock()

mu1 = np.zeros([1,1499])
mu2 = np.append(mu1,1)
mu3 = np.ndarray.tolist(mu2)

#generating random samples using the VMF distribution
data = vmg.randVMF(10000,mu3,1)
H=len(data)
W=len(data[0])

#mention the number of clusters
no_clusters=200

#Clustering using the sphericalclustering function
clusters = snn.sphericalknn(data,no_clusters)

#displaying the clusters
plt.plot(clusters, marker = '.', linestyle = '')

#Plot first 3 colums of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = itertools.cycle(["r", "b", "g", "k", "m", "y"])
marker = itertools.cycle(["o", "*", "+", "x", "D", "s", "^"])

for i in range (0,no_clusters):
    a = np.zeros([1,W])
    j=0
    for p in range (0,H):
        if clusters[p] == i:
            a = np.vstack((a,data[p,:]))
            j=j+1
    a = np.delete(a,(0),axis=0)
    ax.scatter(a[:,0],a[:,1],a[:,2], c=next(colors), marker=next(marker))
plt.show()

print((time.clock() - start_time)/60,"minutes")