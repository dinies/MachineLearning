#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:31:17 2016

@author: edoardoghini
"""


from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    

#plotting boundaries
def plot_boundaries_clustering_with_centers(X,y,clusterer,cmap,ccenters, title):
    h=0.1
    x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
    y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_vim)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap)
    plt.scatter(ccenters[:,0],ccenters[:,1], marker="*", s=500, c='w')
    plt.title(title)
    
    #plotting boundaries
def plot_boundaries_clustering_with_GMM(X,y,clusterer,cmap, title):
    h=0.1
    x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
    y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_vim)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap)
    plt.title(title)

    
#purity
def purity(Confusion_Matrix,k):

    purity = 0.

    NumberElements = 0.

    COne = Confusion_Matrix.ravel()

    for j in range(1,COne.shape[0]):

        NumberElements = NumberElements + COne[j-1]

    for j in range(1,k):

         purity = purity + np.max(Confusion_Matrix[j-1,:])

    return purity / NumberElements
    
    
def clusterize(X,y,clusterer, n_clusters,title,centers=False):
    clusterer.fit(X)
    if centers:
        ccenters = clusterer.cluster_centers_
    #print(ccenters)
    #computing metric scores of homogenity and mutual information
    prediction=clusterer.predict(X)
    hom=metrics.homogeneity_score(y,prediction)
    mut=metrics.normalized_mutual_info_score(y,prediction)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y,prediction)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    #plot_confusion_matrix(cnf_matrix,classes= np.arange(0,5), normalize=True,
                           #                     title='Normalized confusion matrix')

    #compute purity
    pur= purity(cnf_matrix,n_clusters)
    if centers:
        plot_boundaries_clustering_with_centers(X,y,clusterer= clusterer,cmap=cmap_vim,ccenters=ccenters,title=title)
    else: 
        plot_boundaries_clustering_with_GMM(X,y,clusterer= clusterer,cmap=cmap_vim,title=title)
        return [hom, mut ,pur]
    
    
#colormap
cmap_vim = colors.ListedColormap(['#C71585', '#FFFFE0', '#483D8B','#228B22','#CD5C5C'])

#data loading, standardization, PCanalyzation
digits = datasets.load_digits()

X = digits.data
y = digits.target

X = X[y<5]
y = y[y<5]

X_scaler = preprocessing.StandardScaler()
X_std= X_scaler.fit_transform(X)

X=PCA(2).fit_transform(X_std)




#Kmeans itaration
kmeans_hom=[]
kmeans_mut=[]
kmeans_pur=[]
kmeans_range= np.arange(3,11)
for i in kmeans_range:
    number_of_clusters=i
    hom, mut, pur=  clusterize(X,y,KMeans(number_of_clusters), number_of_clusters, title= "Kmeans clustering on "+str(number_of_clusters)+" centers")
    kmeans_hom.append(hom)
    kmeans_mut.append(mut)
    kmeans_pur.append(pur)
#plot kmeans score statistics    
plt.figure()
plt.plot(kmeans_range,kmeans_hom,c='r')
plt.plot(kmeans_range,kmeans_mut,c='g')
plt.plot(kmeans_range,kmeans_pur,c='b')
plt.xlabel('# of centers')
#legend of the plot


red_patch = mpatches.Patch(color='red', label='homogenity')
green_patch = mpatches.Patch(color='green', label='normalized mutual info score')
blue_patch = mpatches.Patch(color='blue', label='purity')
plt.legend(handles=[red_patch,green_patch, blue_patch],loc=4)
plt.show()
plt.title("kmeans")
    


#GMM itaration
gmm_hom=[]
gmm_mut=[]
gmm_pur=[]
gmm_range= np.arange(2,11)
for i in gmm_range:
    number_of_clusters=i
    hom, mut, pur=  clusterize(X,y,GaussianMixture(number_of_clusters), number_of_clusters,title= "Gaussian mixture model clustering on "+str(number_of_clusters)+" centers")
    gmm_hom.append(hom)
    gmm_mut.append(mut)
    gmm_pur.append(pur)
#plot kmeans score statistics    
plt.figure()
plt.plot(gmm_range,gmm_hom,c='r')
plt.plot(gmm_range,gmm_mut,c='g')
plt.plot(gmm_range,gmm_pur,c='b')
plt.xlabel('# of centers')

red_patch = mpatches.Patch(color='red', label='homogenity')
green_patch = mpatches.Patch(color='green', label='normalized mutual info score')
blue_patch = mpatches.Patch(color='blue', label='purity')
plt.legend(handles=[red_patch,green_patch, blue_patch],loc=4)
plt.show()

plt.title("gmm")
    