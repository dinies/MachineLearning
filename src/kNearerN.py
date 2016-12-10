#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:15:39 2016

@author: edoardoghini
"""
from sklearn import neighbors, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import colors, pyplot




def plot_boundaries(test_matrix,test_labels,neighbour_classifier,cmap):
    X= test_matrix
    h=0.1
    x_min, x_max= X[:,0].min() -1, X[:,0].max() +1
    y_min, y_max= X[:,1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= neighbour_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    pyplot.figure()
    pyplot.pcolormesh(xx, yy, Z, cmap=cmap_light)
    pyplot.scatter(X[:,0],X[:,1],c=test_labels,cmap=cmap)

    
#parameters definition
split_rate=0.4
pca_param=2
num_neighbors=40
cmap_light = colors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

#loading data sets
iris = datasets.load_iris()
X=iris.data
y= iris.target
#splitting into train and test
X_train_not_std, X_test_not_std, y_train, y_test= train_test_split(X, y,test_size=split_rate,random_state=90)
#standardizing dataset 
X_scaler = preprocessing.StandardScaler()
X_train = X_scaler.fit_transform(X_train_not_std)
X_test = X_scaler.transform(X_test_not_std)
#pca reduction
X_train_pca=PCA(pca_param).fit_transform(X_train)
X_test_pca=PCA(pca_param).fit_transform(X_test)

#plot of training data
pyplot.figure()
pyplot.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train,cmap=cmap_light)
    
#iteration of different values for neighbors
scores={}
for i in range(1,11):
    
    #training the model
    clf= neighbors.KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_pca, y_train)
    #store statistics & plotting boundaries
    scores[i]= clf.score(X_test_pca, y_test)
    plot_boundaries(X_test_pca,y_test,clf,cmap_light)
#score statistics
pyplot.figure()
pyplot.plot(np.arange(1,11),list(scores.values()))

#weight functions showdown uniform vs distance
#training
clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')
clf.fit(X_train_pca, y_train)
#boundaries
plot_boundaries(X_test_pca,y_test,clf,cmap_light)
#training (explicit parameter for weights)
clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights='uniform')
clf.fit(X_train_pca, y_train)
#boundaries
plot_boundaries(X_test_pca,y_test,clf,cmap_light)

for alfa in [0.1,10,100,1000]:
    

    #square_distance_gaussian weights
    def square_gaussian_distance(distance):
        np.exp(  - 0.1 * alfa * distance** 2)
    #training
    clf= neighbors.KNeighborsClassifier(n_neighbors=3,weights=square_gaussian_distance)
    clf.fit(X_train_pca, y_train)
    #boundaries
    print ("alfa",alfa)
    print('score',clf.score(X_test_pca, y_test))
    plot_boundaries(X_test_pca,y_test,clf,cmap_light)
