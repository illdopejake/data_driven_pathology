#MIT License

#Copyright (c) 2017 Jake Vogel

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#1) Anyone using this script must cite the original manuscript explaining the
#methods upon which this work is based. Effectively, this script is a
#simplified and slightly modified version of the Bootstrap Analysis of Stable
#Clusters (BASC). As such, please cite the following manuscript:

#Pierre Bellec, Pedro Rosa-Neto, Oliver C. Lyttelton, Habib Benali, Alan C.
#Evans, Multi-level bootstrap analysis of stable clusters in resting-state fMRI,
#NeuroImage, Volume 51, Issue 3, 2010, Pages 1126-1139, ISSN 1053-8119,
#http://dx.doi.org/10.1016/j.neuroimage.2010.02.082.

#2) As this material is not ready for full public use, and use of this script
#and other contained in this repo must involve direct collaboration and express
#permission from Mr. Vogel. That is, direct communication should be made with 
# Mr. Vogel if this code is used, and details of the collaboration must have
# Mr. Vogel's explicit documented approval.

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#~            




import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from scipy import stats
import os
from glob import glob
import nibabel as ni
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
import itertools

def poormans_basc(in_mtx,n_clust,n_iter,checker,
                  bootstrap = False,
                  inner_cluster_object = None,
                  connect = False, neighbors = 20,
                  plotit = True):
    
    clust_mtx = pandas.DataFrame(index=in_mtx.index, copy=True)
    print('running cluster analyses')
    if type(inner_cluster_object) == type(None):
        inner_cluster_object = cluster.KMeans()
    for i in range(n_iter):
        tmp_mtx = pandas.DataFrame(in_mtx,copy=True)
        if i%checker == 0:
            print('working on iteration',i)
        if bootstrap:
            new_ind = np.random.choice(in_mtx.columns,len(in_mtx.columns))
            tmp_mtx = tmp_mtx[new_ind]
        if connect:
            connectivity = kneighbors_graph(tmp_mtx, n_neighbors=neighbors, 
                                            include_self=False)
            nlabs = cluster.AgglomerativeClustering(n_clust,connectivity=connectivity
                                              ).fit(tmp_mtx).labels_
            clust_mtx.loc[:,'i%s'%i] = nlabs
        else:
            clust_mtx.loc[:,'i%s'%i] = inner_cluster_object.fit(tmp_mtx).labels_
    print('creating stability matrix')
    id_mtx = np.zeros((len(clust_mtx),len(clust_mtx)))
    for i in range(n_iter):
        if i%checker == 0:
            print('working on iteration',i)
        icol = pandas.Series(clust_mtx.values[:,i])
        for u in np.unique(icol):
            id_mtx[[x[0] for x in itertools.combinations(icol[icol==u].index.tolist(),2)],[
                    y[1] for y in itertools.combinations(icol[icol==u].index.tolist(),2)]] += 1
    stab_mtx = id_mtx/n_iter
    
    stab_mtx[np.tril_indices_from(stab_mtx)] = stab_mtx.transpose()[
                                            np.tril_indices_from(stab_mtx)]
    
    if plotit:
        plt.close()
        sns.clustermap(stab_mtx, cmap = 'RdBu_r')
        plt.show()
    
    connectivity = kneighbors_graph(stab_mtx, n_neighbors=neighbors, include_self=False)
    aggclust = cluster.AgglomerativeClustering(n_clust,connectivity=connectivity
                                              ).fit(stab_mtx)
    
    newdf = pandas.DataFrame(in_mtx, copy=True)
    newdf.loc[:,'order'] = aggclust.labels_
    output = {'cluster_object': aggclust, 'stability': stab_mtx,
              'dataframe': newdf, 'labels': aggclust.labels_}
    
    return output
    
